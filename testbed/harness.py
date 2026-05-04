"""AgentHop evaluation harness — async multi-turn tool-use loop.

Drives LLM agents through benchmark samples:
  1. Present question + tool descriptions + budget
  2. Agent calls tools → harness executes → returns results
  3. Repeat until submit_answer, budget exhausted, or max turns
  4. Log full trajectory

Supports OpenAI-compatible (incl. sglang/vllm) and Anthropic backends.
Runs multiple samples concurrently via asyncio worker pool.
"""

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from tools import AgentHopTools, TOOL_SCHEMAS, TOOL_COSTS, ToolResult


# ── Answer-letter parser ─────────────────────────────────────────────────────

_ANSWER_LETTER_RE = re.compile(r"^\s*\(?([ABCD])\)?\s*[\.:\)]?\s*$", re.IGNORECASE)


def _parse_answer_letter(answer: str) -> int:
    """Parse an answer string to an option index (0-3), or -1 if unparseable.

    Accepts bare letters ("A"), parenthesised letters ("(A)"), and trailing
    punctuation ("A)", "A.", "A:"). Earlier model runs that submitted "(A)"
    were silently rejected by a stricter parser; this lenient form preserves
    backward compatibility while not over-matching prose.
    """
    if not isinstance(answer, str):
        return -1
    m = _ANSWER_LETTER_RE.match(answer)
    if not m:
        return -1
    return ord(m.group(1).upper()) - ord("A")


# ── Provider auto-detection ──────────────────────────────────────────────────

# Model prefix → (env var for API key, base URL, backend)
_PROVIDER_MAP = [
    # Codex models MUST be checked before generic gpt- prefix (they use Responses API)
    (("codex-", "gpt-5.3-codex", "gpt-5.1-codex", "gpt-5-codex"),
                                          "OPENAI_API_KEY",    None,                           "responses"),
    # GPT-5.x reasoning models: function tools + reasoning_effort are only
    # supported on /v1/responses, not /v1/chat/completions.
    (("gpt-5.4", "gpt-5.3", "gpt-5.2", "gpt-5.1", "gpt-5-"),
                                          "OPENAI_API_KEY",    None,                           "responses"),
    (("gpt-", "o1-", "o3-", "o4-"),     "OPENAI_API_KEY",    None,                           "openai"),
    (("deepseek-",),                      "DEEPSEEK_API_KEY",  "https://api.deepseek.com/v1",  "openai"),
    (("claude-",),                        "ANTHROPIC_API_KEY",  None,                          "anthropic"),
    # Gemini 3.x requires thought_signature round-tripping which the OpenAI
    # compat layer drops — route to the native google-genai SDK ("google" backend).
    (("gemini-3", "gemini-2.5"),           "GOOGLE_API_KEY",   None,                            "google"),
    (("gemini-",),                        "GOOGLE_API_KEY",    "https://generativelanguage.googleapis.com/v1beta/openai/", "openai"),
    (("zai-org/", "moonshotai/", "MiniMaxAI/", "minimaxai/"),
                                          "TOGETHER_API_KEY",  "https://api.together.xyz/v1",  "openai"),
    (("grok-",),                          "XAI_API_KEY",       "https://api.x.ai/v1",          "openai"),
]


def _resolve_provider(model: str, api_key: str | None, base_url: str | None, backend: str) -> tuple[str | None, str | None, str]:
    """Auto-detect API key, base URL, and backend from model name."""
    if api_key and base_url:
        return api_key, base_url, backend  # explicit overrides, skip detection

    m_lower = model.lower()
    for prefixes, env_var, default_url, default_backend in _PROVIDER_MAP:
        if any(m_lower.startswith(p.lower()) for p in prefixes):
            resolved_key = api_key or os.environ.get(env_var)
            resolved_url = base_url or default_url
            # Use auto-detected backend unless user explicitly set a non-default backend
            resolved_backend = default_backend if backend == "openai" else backend
            return resolved_key, resolved_url, resolved_backend

    # Unknown provider — fall back to what was given
    return api_key, base_url, backend


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class HarnessConfig:
    """Evaluation harness configuration."""
    budget: int = 30                # total tool-call budget
    max_turns: int = 20             # hard cap on conversation turns
    model: str = "gpt-4.1"         # model identifier
    backend: str = "openai"        # "openai" or "anthropic"
    temperature: float | None = None  # None = don't send (API uses its own default; required for GPT-5 reasoning)
    top_p: float | None = None      # nucleus sampling (None = API default)
    top_k: int | None = None        # top-k sampling (OpenAI extra_body only)
    max_tokens: int = 16384         # per-turn generation limit (16K; avoids truncation on reasoning outputs)
    base_url: str | None = None     # optional API base URL override
    api_key: str | None = None      # optional API key override
    workers: int = 1                # concurrent sample workers
    max_total_tokens: int = 200_000 # internal safety cap on cumulative tokens per sample
    max_run_tokens: int = 0         # run-wide cumulative-token cap (0 = off). Aborts queue when exceeded.
    strategy: str = "direct"        # prompting strategy: direct, react, plan, noretrieval
    text_tools: bool = False        # text-based tool calling (no API tools param — universal compatibility)
    reasoning_effort: str | None = None  # reasoning effort: "low", "medium", "high" (OpenAI); or token budget like "8192" (Anthropic)
    disabled_tools: tuple[str, ...] = ()  # tool names to exclude from the toolset (e.g. ("search_papers",))

    def __post_init__(self):
        self.api_key, self.base_url, self.backend = _resolve_provider(
            self.model, self.api_key, self.base_url, self.backend
        )


# ── Trajectory logging ───────────────────────────────────────────────────────

@dataclass
class TurnRecord:
    """One turn of the conversation."""
    turn: int
    role: str                       # "assistant" or "tool"
    content: str | None = None      # assistant text
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    budget_before: int = 0
    budget_after: int = 0
    timestamp: float = 0.0
    # Silent tracking (not shown to agent)
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class Trajectory:
    """Full evaluation trajectory for one sample."""
    sample_id: str = ""
    model: str = ""
    question_type: str = ""
    depth: int = 0
    reasoning_type: str = ""
    correct_index: int = -1
    predicted_answer: str = ""
    predicted_index: int = -1
    is_correct: bool = False
    reasoning: str = ""
    budget_total: int = 0
    budget_used: int = 0
    total_turns: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    cache_read_tokens: int = 0       # Anthropic prompt cache hits
    cache_creation_tokens: int = 0   # Anthropic prompt cache writes
    wall_time_s: float = 0.0
    turns: list[TurnRecord] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)  # full conversation history
    tool_errors: int = 0             # malformed args, unknown tools, etc.
    api_error_count: int = 0         # non-transient API errors (advance turn_idx)
    transient_retry_count: int = 0   # rate-limit / 5xx retries that did NOT consume turn budget
    retry_count: int = 0             # sample-level retries done (0 = first-try; max 2 = 3 attempts total)
    terminated_by: str = ""         # "answer", "max_turns", "budget", "token_limit", "error"


# ── System prompt builder ────────────────────────────────────────────────────

# ── Prompt strategy registry ────────────────────────────────────────────────

# Shared tool description block (reused across all strategies)
_TOOLS_BLOCK = """\
## Tools

You have seven tools. Each tool call costs budget points. Plan your exploration carefully.

### Navigation Tools (1 point each)
These tools help you discover and identify papers. They are cheap — use them freely to orient yourself.

- **get_paper_info**(paper_id) — Returns a paper's title, year, abstract, and whether full text is available. Use this to quickly assess whether a paper is relevant before committing to reading it.
- **get_references**(paper_id) — Returns the list of papers cited by a given paper, with their IDs, titles, years, and whether full text is available. This is your primary navigation tool for traversing the citation graph.
- **search_papers**(query, top_k) — Keyword search across all papers in the pool. Returns matching papers ranked by relevance. Useful when you know what topic you need but not which paper contains it.
- **list_sections**(paper_id) — Returns the section names and sizes of a paper. Use this before read_section to identify which section likely contains the information you need.

### Reading Tool (5 points)
- **read_section**(paper_id, section) — Returns the full text of one section. This is expensive — only use it when you have identified a specific section that likely contains evidence relevant to the question.

### Free Tools (0 points)
- **think**(reasoning) — Organize your thoughts, plan next steps, or review evidence gathered so far. Use this anytime to reason about what you've learned before deciding your next action.
- **submit_answer**(answer, reasoning) — Submit your final answer (A, B, C, or D) with brief reasoning. You may only submit once."""

_BUDGET_BLOCK = """\
## Constraints
You operate under three independent limits. The run terminates when any one is hit:
- **Budget**: {budget} points total. Navigation tools cost 1 pt; read_section costs 5 pts.
- **Turns**: at most {max_turns} conversational turns.
- **Tokens**: at most {max_total_tokens:,} cumulative prompt + completion tokens per sample.

After every tool result you will see a status line reporting counters only:
`Turn: T  |  Tokens used: U  |  Budget used: B`.
You are responsible for tracking these against the limits above.

Manage your resources strategically:
- Use navigation tools (1 pt) to identify the right papers and sections first.
- Use read_section (5 pts) only on sections you have reason to believe contain relevant evidence.
- Submit your answer when you have gathered enough evidence. Do not exhaust your budget exploring if you already have a strong basis for an answer."""

_TASK_BLOCK = """\
## Seed Paper
Your starting point in the citation graph:
- **ID**: {seed_id}
- **Title**: {seed_title}

## Question
{question}

## Answer Options
{options_text}

Select exactly one option (A, B, C, or D) and submit using submit_answer."""


# ── Strategy: direct (default) ──────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """\
You are a research assistant tasked with answering a multi-hop scientific question. \
You will navigate a citation graph of academic papers using a set of tools. \
Your goal is to gather sufficient evidence from the papers to select the correct answer.

## Task Overview
You are given a seed paper as your starting point. The answer requires information from \
one or more papers reachable through the citation graph — papers cited by the seed, or \
papers cited by those papers (up to 2 hops). You must explore the graph strategically, \
read relevant sections, and synthesize evidence to choose the best answer.

{tools_block}

{budget_block}

{task_block}
"""

# ── Strategy: react ─────────────────────────────────────────────────────────

REACT_SYSTEM_PROMPT_TEMPLATE = """\
You are a research assistant tasked with answering a multi-hop scientific question. \
You will navigate a citation graph of academic papers using a set of tools.

## Approach: Think → Act → Observe

You MUST follow this structured loop for every step:

1. **Think**: Before each tool call, use the `think` tool to reason about:
   - What information you have gathered so far
   - What information you still need
   - Which tool call will most efficiently get you closer to the answer
   - Why you chose this specific tool and arguments

2. **Act**: Make exactly one tool call based on your reasoning.

3. **Observe**: After receiving the result, use `think` again to process what you learned before your next action.

Never call a navigation or reading tool without first calling `think` to justify your choice. \
This disciplined approach prevents wasted budget on irrelevant exploration.

{tools_block}

{budget_block}

{task_block}
"""

# ── Strategy: plan ──────────────────────────────────────────────────────────

PLAN_SYSTEM_PROMPT_TEMPLATE = """\
You are a research assistant tasked with answering a multi-hop scientific question. \
You will navigate a citation graph of academic papers using a set of tools.

## Approach: Plan First, Then Execute

Before making any tool calls, you MUST first create a plan:

1. **Analyze the question**: Use `think` to break down what the question is asking. \
Identify the key entities, relationships, and what type of evidence you need.

2. **Decompose into sub-questions**: Identify the information chain needed. For example:
   - Sub-Q1: What papers does the seed cite that relate to [topic X]?
   - Sub-Q2: In that paper, what does the [relevant section] say about [specific claim]?
   - Sub-Q3: How does that evidence help distinguish between the answer options?

3. **Create a navigation plan**: Map out which tools to call in what order, with estimated budget.

4. **Execute the plan**: Follow your plan step by step. After each tool result, briefly \
reassess whether the plan needs adjustment.

5. **Synthesize and answer**: Once you have gathered sufficient evidence, review all \
findings and submit your answer.

Start by calling `think` with your full analysis and plan before making any other tool call.

{tools_block}

{budget_block}

{task_block}
"""

# ── Strategy: noretrieval (closed-book) ─────────────────────────────────────

NORETRIEVAL_SYSTEM_PROMPT_TEMPLATE = """\
You are a research assistant tasked with answering a multi-hop scientific question. \
You must answer based solely on your own knowledge — no tools are available.

Read the question and answer options carefully. Use your knowledge of the scientific \
literature to reason about which answer is most likely correct. Consider:
- The claims made in each option
- Whether the described findings are consistent with known results in the field
- Technical plausibility and specificity of each option

{task_block}

You MUST call submit_answer with your best guess. Reason carefully, then choose.
"""

# ── Strategy registry ───────────────────────────────────────────────────────

STRATEGY_TEMPLATES = {
    "direct": SYSTEM_PROMPT_TEMPLATE,
    "react": REACT_SYSTEM_PROMPT_TEMPLATE,
    "plan": PLAN_SYSTEM_PROMPT_TEMPLATE,
    "noretrieval": NORETRIEVAL_SYSTEM_PROMPT_TEMPLATE,
}

STRATEGY_FIRST_MESSAGES = {
    "direct": "Please solve the question above using the available tools. Start by examining the seed paper.",
    "react": "Please solve the question using the Think → Act → Observe approach described above. Start by calling think to analyze the question, then examine the seed paper.",
    "plan": "Please solve the question using the Plan First approach described above. Start by calling think to create your full analysis and navigation plan.",
    "noretrieval": "Please answer the question based on your knowledge. Reason carefully about each option, then call submit_answer.",
}


def build_system_prompt(
    sample: dict,
    budget_total: int,
    budget_remaining: int,
    strategy: str = "direct",
    text_tools: bool = False,
    max_turns: int = 20,
    max_total_tokens: int = 200_000,
    disabled_tools: tuple[str, ...] = (),
) -> str:
    """Build the system prompt for a benchmark sample."""
    options = sample["options"]
    options_text = "\n".join(
        f"({chr(65+i)}) {opt}" for i, opt in enumerate(options)
    )

    task_block = _TASK_BLOCK.format(
        question=sample["question"],
        options_text=options_text,
        seed_id=sample["seed_paper_id"],
        seed_title=sample["seed_title"],
    )
    budget_block = _BUDGET_BLOCK.format(
        budget=budget_total,
        budget_remaining=budget_remaining,
        max_turns=max_turns,
        max_total_tokens=max_total_tokens,
    )

    # Text-tools mode: include tool-call format instructions in prompt
    tools_block = _build_text_tools_prompt(strategy) if text_tools else _TOOLS_BLOCK
    # Strip bullet lines for any disabled tools (e.g. --disable-tools search_papers)
    if disabled_tools:
        for name in disabled_tools:
            tools_block = re.sub(
                rf"^- \*\*{re.escape(name)}\*\*\([^\n]*\n",
                "",
                tools_block,
                flags=re.MULTILINE,
            )

    template = STRATEGY_TEMPLATES.get(strategy, SYSTEM_PROMPT_TEMPLATE)
    if strategy == "noretrieval":
        return template.format(task_block=task_block)
    return template.format(
        tools_block=tools_block,
        budget_block=budget_block,
        task_block=task_block,
    )


def build_budget_notice(
    budget_remaining: int,
    budget_total: int,
    turn_idx: int = -1,
    max_turns: int = 0,
    cumulative_tokens: int = 0,
    max_total_tokens: int = 0,
) -> str:
    """Minimal status line appended after every tool result.

    Reports counters only — no limits — so the agent must track its own
    position against the constraints stated in the system prompt.
    Format: "Turn: T  |  Tokens used: U  |  Budget used: B".
    """
    budget_used = budget_total - budget_remaining
    parts = []
    if turn_idx >= 0:
        parts.append(f"Turn: {turn_idx + 1}")
    if cumulative_tokens >= 0:
        parts.append(f"Tokens used: {cumulative_tokens:,}")
    parts.append(f"Budget used: {budget_used}")
    return "\n" + "  |  ".join(parts)


# ── Tool format converters ───────────────────────────────────────────────────

_NORETRIEVAL_TOOLS = {"think", "submit_answer"}


def _filtered_schemas(strategy: str, disabled: tuple[str, ...] = ()) -> list[dict]:
    """Apply strategy + disabled-tools filtering to TOOL_SCHEMAS."""
    schemas = TOOL_SCHEMAS if strategy != "noretrieval" else [
        s for s in TOOL_SCHEMAS if s["name"] in _NORETRIEVAL_TOOLS
    ]
    if disabled:
        dset = set(disabled)
        schemas = [s for s in schemas if s["name"] not in dset]
    return schemas


def _make_openai_tools(strategy: str = "direct", disabled: tuple[str, ...] = ()) -> list[dict]:
    """Convert TOOL_SCHEMAS to OpenAI function-calling format."""
    schemas = _filtered_schemas(strategy, disabled)
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": s["parameters"],
            },
        }
        for s in schemas
    ]


def _make_responses_tools(strategy: str = "direct", disabled: tuple[str, ...] = ()) -> list[dict]:
    """Convert TOOL_SCHEMAS to OpenAI Responses API format (flat, not nested)."""
    schemas = _filtered_schemas(strategy, disabled)
    return [
        {
            "type": "function",
            "name": s["name"],
            "description": s["description"],
            "parameters": s["parameters"],
        }
        for s in schemas
    ]


def _make_anthropic_tools(strategy: str = "direct", disabled: tuple[str, ...] = ()) -> list[dict]:
    """Convert TOOL_SCHEMAS to Anthropic tool format."""
    schemas = _filtered_schemas(strategy, disabled)
    return [
        {
            "name": s["name"],
            "description": s["description"],
            "input_schema": s["parameters"],
        }
        for s in schemas
    ]


def _make_genai_tools(strategy: str = "direct", disabled: tuple[str, ...] = ()):
    """Convert TOOL_SCHEMAS to google-genai FunctionDeclaration list."""
    from google.genai import types as genai_types
    schemas = _filtered_schemas(strategy, disabled)
    # genai accepts dict-form function_declarations with capitalized type names
    def convert_schema(sch: dict) -> dict:
        if not isinstance(sch, dict):
            return sch
        out = {}
        for k, v in sch.items():
            if k == "type" and isinstance(v, str):
                out[k] = v.upper()
            elif k == "properties" and isinstance(v, dict):
                out[k] = {pk: convert_schema(pv) for pk, pv in v.items()}
            elif k == "items" and isinstance(v, dict):
                out[k] = convert_schema(v)
            elif k == "default":
                # genai Schema rejects "default"; skip
                continue
            else:
                out[k] = v
        return out

    function_declarations = [
        {
            "name": s["name"],
            "description": s["description"],
            "parameters": convert_schema(s["parameters"]),
        }
        for s in schemas
    ]
    return [genai_types.Tool(function_declarations=function_declarations)]


# ── Text-based tool calling (universal compatibility) ──────────────────────

import re

_TEXT_TOOLS_BLOCK = """\

## How to Call Tools

To use a tool, output a JSON object wrapped in <tool_call> tags:

<tool_call>
{"name": "get_paper_info", "arguments": {"paper_id": "abc123def456"}}
</tool_call>

You may call ONE tool per turn. After your tool call, you will receive the result, \
then you can call another tool or submit your answer.

To submit your final answer:
<tool_call>
{"name": "submit_answer", "arguments": {"answer": "A", "reasoning": "brief explanation"}}
</tool_call>

IMPORTANT: Always use the exact <tool_call>...</tool_call> format. Do not use any other format."""


def _build_text_tools_prompt(strategy: str = "direct") -> str:
    """Build the tool description block for text-based tool calling."""
    if strategy == "noretrieval":
        return _TEXT_TOOLS_BLOCK
    return _TOOLS_BLOCK + "\n" + _TEXT_TOOLS_BLOCK


# Known tool names for text-based parsing (used to validate parsed calls)
_KNOWN_TOOL_NAMES = {s["name"] for s in TOOL_SCHEMAS}


def _parse_python_call(call_str: str) -> dict | None:
    """Parse a Python-style function call like  func_name(key='value', key2='value2').

    Returns {"name": ..., "arguments": {...}} or None on failure.
    Handles both single-quoted and double-quoted string arguments, as well as
    integer values.
    """
    call_str = call_str.strip()
    m = re.match(r'(\w+)\s*\((.*)\)\s*$', call_str, re.DOTALL)
    if not m:
        return None
    name = m.group(1)
    if name not in _KNOWN_TOOL_NAMES:
        return None
    args_str = m.group(2).strip()
    if not args_str:
        return {"name": name, "arguments": {}}
    # Parse key=value pairs
    args = {}
    for pair in re.finditer(
        r"(\w+)\s*=\s*(?:'([^']*)'|\"([^\"]*)\"|([\d]+))", args_str
    ):
        key = pair.group(1)
        val = pair.group(2) if pair.group(2) is not None else (
              pair.group(3) if pair.group(3) is not None else int(pair.group(4)))
        args[key] = val
    return {"name": name, "arguments": args}


_CHANNEL_BLOCK_RE = re.compile(
    r'<\|channel\|>(\w+)(?:\s+[^<]*?)?<\|(?:constrain\|>[^<]*<\|)?message\|>(.*?)<\|end\|>',
    re.DOTALL,
)


def sanitize_channel_content(text: str) -> str:
    """Strip OSS <|channel|> markup from assistant text.

    GPT-OSS and similar models emit content with special-token channel markers:
      <|channel|>analysis<|message|>...thinking...<|end|>
      <|channel|>commentary to=functions.X<|message|>{args}<|end|>
      <|channel|>final<|message|>...answer...<|end|>

    sglang's serving layer rejects inbound conversation history that still
    contains these tags in the content field. We strip all channel blocks
    except the 'final' channel, whose message body is kept as plain text.
    Tool-call channels ('commentary to=functions.*') are already parsed into
    structured tool_calls by parse_text_tool_calls, so dropping them here is
    safe.
    """
    if not text or '<|channel|>' not in text:
        return text

    final_parts = []
    for m in _CHANNEL_BLOCK_RE.finditer(text):
        channel = m.group(1)
        body = m.group(2)
        if channel == 'final':
            final_parts.append(body.strip())
    if final_parts:
        return '\n'.join(final_parts)

    # No final channel — strip all channel blocks and return any residue
    cleaned = _CHANNEL_BLOCK_RE.sub('', text).strip()
    # Also strip orphan markers (unterminated blocks)
    cleaned = re.sub(r'<\|(?:channel|message|end|constrain|start)\|>[^<]*', '', cleaned).strip()
    return cleaned


def parse_text_tool_calls(text: str) -> list[dict]:
    """Parse tool calls from model text output.

    Supports multiple formats for robustness:
    1. <tool_call>{"name": ..., "arguments": ...}</tool_call>  (primary / Qwen/Hermes)
    2. ```tool_call\\n{"name": ..., "arguments": ...}\\n```     (markdown JSON)
    3. ```tool_code\\nfunc(args)\\n```                         (Gemma-style Python call)
    4. <|channel|>...<|message|>{JSON}                          (OSS special-token format)
    5. Bare JSON {"name": ..., "arguments": ...} on its own line (fallback)
    6. Standalone Python function calls func(arg=val)           (last resort)
    7. Pythonic list format [func(arg=val), ...]                (Gemma/Llama pythonic)
    """
    tool_calls = []
    call_id_counter = 0

    def _make_call(parsed: dict) -> dict | None:
        nonlocal call_id_counter
        name = parsed.get("name")
        args = parsed.get("arguments", {})
        if not name or name not in _KNOWN_TOOL_NAMES:
            return None
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        call_id_counter += 1
        return {"id": f"text_{call_id_counter}", "name": name, "args": args}

    # Pattern 1: <tool_call>...</tool_call>
    for m in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL):
        try:
            parsed = json.loads(m.group(1))
            tc = _make_call(parsed)
            if tc:
                tool_calls.append(tc)
        except json.JSONDecodeError:
            pass

    if tool_calls:
        return tool_calls

    # Pattern 2: ```tool_call ... ``` or ```json ... ```  (JSON inside markdown)
    for m in re.finditer(r"```(?:tool_call|json)\s*\n(.*?)\n\s*```", text, re.DOTALL):
        try:
            parsed = json.loads(m.group(1))
            if "name" in parsed:
                tc = _make_call(parsed)
                if tc:
                    tool_calls.append(tc)
        except json.JSONDecodeError:
            pass

    if tool_calls:
        return tool_calls

    # Pattern 3: ```tool_code ... ```  (Gemma-style Python function calls)
    for m in re.finditer(r"```tool_code\s*\n(.*?)\n\s*```", text, re.DOTALL):
        body = m.group(1).strip()
        # Each line may be a separate function call
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            parsed = _parse_python_call(line)
            if parsed:
                tc = _make_call(parsed)
                if tc:
                    tool_calls.append(tc)

    if tool_calls:
        return tool_calls

    # Pattern 4: OSS special-token format  <|channel|>...<|message|>{JSON}
    # e.g. <|channel|>commentary to=functions.get_references <|constrain|>json<|message|>{"paper_id":"..."}
    for m in re.finditer(
        r'<\|channel\|>[^<]*?to=functions\.(\w+)[^<]*<\|(?:constrain\|>[^<]*<\|)?message\|>\s*(\{.*?\})',
        text, re.DOTALL,
    ):
        fn_name = m.group(1)
        if fn_name not in _KNOWN_TOOL_NAMES:
            continue
        try:
            args = json.loads(m.group(2))
        except json.JSONDecodeError:
            args = {}
        call_id_counter += 1
        tool_calls.append({"id": f"text_{call_id_counter}", "name": fn_name, "args": args})

    if tool_calls:
        return tool_calls

    # Pattern 5: bare JSON with "name" and "arguments" keys
    for m in re.finditer(r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]*\}[^}]*\}', text):
        try:
            parsed = json.loads(m.group(0))
            tc = _make_call(parsed)
            if tc:
                tool_calls.append(tc)
        except json.JSONDecodeError:
            pass

    # Pattern 6: standalone Python function calls (no code fence) — last resort
    if not tool_calls:
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parsed = _parse_python_call(line)
            if parsed:
                tc = _make_call(parsed)
                if tc:
                    tool_calls.append(tc)

    if tool_calls:
        return tool_calls

    # Pattern 7: Pythonic list format  [func(arg=val), func2(arg=val)]
    # Used by Gemma 3, Llama 3.2 small, Llama 4 with pythonic parser
    for m in re.finditer(r'\[\s*((?:\w+\s*\([^)]*\)\s*,?\s*)+)\]', text, re.DOTALL):
        body = m.group(1).strip()
        for call_m in re.finditer(r'(\w+)\s*\(([^)]*)\)', body):
            fn_name = call_m.group(1)
            if fn_name not in _KNOWN_TOOL_NAMES:
                continue
            args_str = call_m.group(2).strip()
            # Parse keyword arguments: key="value" or key='value' or key=123
            args = {}
            for pair in re.finditer(
                r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([\d]+))', args_str
            ):
                key = pair.group(1)
                val = pair.group(2) if pair.group(2) is not None else (
                      pair.group(3) if pair.group(3) is not None else int(pair.group(4)))
                args[key] = val
            call_id_counter += 1
            tool_calls.append({"id": f"text_{call_id_counter}", "name": fn_name, "args": args})

    return tool_calls


def _format_text_tool_results(
    tool_calls: list[dict],
    results: list,
    budget_remaining: int,
    budget_total: int,
    turn_idx: int = -1,
    max_turns: int = 0,
    cumulative_tokens: int = 0,
    max_total_tokens: int = 0,
) -> str:
    """Format tool results as a user message for text-based tool calling."""
    parts = []
    for tc, result in zip(tool_calls, results):
        content = result.output + build_budget_notice(
            budget_remaining, budget_total,
            turn_idx, max_turns, cumulative_tokens, max_total_tokens,
        )
        parts.append(f"<tool_result name=\"{tc['name']}\">\n{content}\n</tool_result>")
    return "\n\n".join(parts)


# ── Async LLM backends ──────────────────────────────────────────────────────

async def call_openai_async(messages: list[dict], config: HarnessConfig):
    """Call OpenAI-compatible API (works with sglang/vllm too)."""
    from openai import AsyncOpenAI

    kwargs = {}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.base_url:
        kwargs["base_url"] = config.base_url
    client = AsyncOpenAI(**kwargs)
    # Newer OpenAI models (gpt-5+) require max_completion_tokens
    is_gpt5 = config.model.startswith("gpt-5")
    token_param = "max_completion_tokens" if is_gpt5 else "max_tokens"
    create_kwargs = {
        "model": config.model,
        "messages": messages,
        token_param: config.max_tokens,
    }
    # Text-tools mode: no API tools param — tools are in the system prompt
    if not config.text_tools:
        create_kwargs["tools"] = _make_openai_tools(config.strategy, config.disabled_tools)
    # GPT-5+ reasoning models don't support temperature/top_p (unless reasoning_effort=none)
    if not is_gpt5:
        if config.temperature is not None:
            create_kwargs["temperature"] = config.temperature
        if config.top_p is not None:
            create_kwargs["top_p"] = config.top_p
    # Reasoning effort for o-series / GPT-5+ models
    if config.reasoning_effort and is_gpt5:
        create_kwargs["reasoning_effort"] = config.reasoning_effort
    # top_k via extra_body (sglang/vllm support, not native OpenAI)
    if config.top_k is not None:
        create_kwargs["extra_body"] = {"top_k": config.top_k}
    resp = await client.chat.completions.create(**create_kwargs)
    return resp


async def call_responses_async(
    input_data: list[dict] | str,
    config: HarnessConfig,
    instructions: str | None = None,
    previous_response_id: str | None = None,
):
    """Call OpenAI Responses API (for Codex models).

    For the first turn, pass input_data as the initial messages.
    For subsequent turns, pass tool results as input_data and the
    previous_response_id for conversation continuity.
    """
    from openai import AsyncOpenAI

    kwargs = {}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.base_url:
        kwargs["base_url"] = config.base_url
    client = AsyncOpenAI(**kwargs)

    tools = _make_responses_tools(config.strategy, config.disabled_tools)

    create_kwargs = {
        "model": config.model,
        "input": input_data,
        "tools": tools,
        "max_output_tokens": config.max_tokens,
    }
    if instructions:
        create_kwargs["instructions"] = instructions
    if previous_response_id:
        create_kwargs["previous_response_id"] = previous_response_id
    if config.temperature is not None:
        create_kwargs["temperature"] = config.temperature
    if config.top_p is not None:
        create_kwargs["top_p"] = config.top_p
    if config.reasoning_effort:
        create_kwargs["reasoning"] = {"effort": config.reasoning_effort}

    resp = await client.responses.create(**create_kwargs)
    return resp


def parse_responses_response(resp) -> tuple[str | None, list[dict], int, int, int]:
    """Parse OpenAI Responses API response → (text, tool_calls, prompt_tokens, completion_tokens, cached_tokens)."""
    text_parts = []
    tool_calls = []

    for item in resp.output:
        if item.type == "message":
            for content in item.content:
                if hasattr(content, "text"):
                    text_parts.append(content.text)
        elif item.type == "function_call":
            try:
                args = json.loads(item.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {"_raw": item.arguments}
            tool_calls.append({
                "id": item.call_id,
                "name": item.name,
                "args": args,
            })

    text = "\n".join(text_parts) if text_parts else None
    usage = resp.usage
    p_tok = getattr(usage, "input_tokens", 0) or 0
    c_tok = getattr(usage, "output_tokens", 0) or 0
    cached = 0
    details = getattr(usage, "input_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", 0) or 0
    return text, tool_calls, p_tok, c_tok, cached


def build_responses_tool_results(
    tool_calls: list[dict],
    results: list,
    budget_remaining: int,
    budget_total: int,
    turn_idx: int = -1,
    max_turns: int = 0,
    cumulative_tokens: int = 0,
    max_total_tokens: int = 0,
) -> list[dict]:
    """Build function_call_output items to send as input for next Responses API turn."""
    items = []
    for tc, result in zip(tool_calls, results):
        content = result.output + build_budget_notice(
            budget_remaining, budget_total,
            turn_idx, max_turns, cumulative_tokens, max_total_tokens,
        )
        items.append({
            "type": "function_call_output",
            "call_id": tc["id"],
            "output": content,
        })
    return items


async def call_anthropic_async(messages: list[dict], system: str, config: HarnessConfig):
    """Call Anthropic API."""
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=config.api_key)

    # Prompt caching: wrap system prompt and tools with cache_control
    # so the growing conversation prefix is cached across turns.
    system_blocks = [
        {
            "type": "text",
            "text": system,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    tools = _make_anthropic_tools(config.strategy, config.disabled_tools)
    if tools:
        tools[-1]["cache_control"] = {"type": "ephemeral"}

    create_kwargs = {
        "model": config.model,
        "system": system_blocks,
        "messages": messages,
        "tools": tools,
        "max_tokens": config.max_tokens,
    }
    if config.temperature is not None:
        create_kwargs["temperature"] = config.temperature
    # Extended thinking for Claude (reasoning_effort as budget_tokens)
    if config.reasoning_effort:
        _EFFORT_MAP = {"low": 4096, "medium": 8192, "high": 16384}
        budget = _EFFORT_MAP.get(config.reasoning_effort)
        if budget is None and config.reasoning_effort.isdigit():
            budget = int(config.reasoning_effort)
        if budget:
            create_kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
            # Extended thinking requires temperature=1 and no top_p
            create_kwargs["temperature"] = 1.0
    resp = await client.messages.create(**create_kwargs)
    return resp


# ── Google (native google-genai) ────────────────────────────────────────────
# Uses google.genai SDK directly so that thought_signature round-tripping
# works correctly for Gemini 3.x multi-turn tool use. The OpenAI-compatibility
# endpoint drops thought_signatures, which causes 400 errors after a few
# tool calls.

async def call_google_async(contents: list, system: str, config: HarnessConfig):
    """Call Google Gemini via native google-genai SDK."""
    from google import genai as genai_mod
    from google.genai import types as genai_types

    client = genai_mod.Client(api_key=config.api_key)
    tools = _make_genai_tools(config.strategy, config.disabled_tools)

    cfg_kwargs = {
        "system_instruction": system,
        "tools": tools,
        "max_output_tokens": config.max_tokens,
    }
    if config.temperature is not None:
        cfg_kwargs["temperature"] = config.temperature
    if config.top_p is not None:
        cfg_kwargs["top_p"] = config.top_p
    if config.top_k is not None:
        cfg_kwargs["top_k"] = config.top_k
    # Reasoning / thinking: Gemini 3 accepts thinking_level strings
    if config.reasoning_effort in ("minimal", "low", "medium", "high"):
        cfg_kwargs["thinking_config"] = genai_types.ThinkingConfig(
            thinking_level=config.reasoning_effort,
        )
    # Disable automatic function calling so we handle tool execution ourselves
    cfg_kwargs["automatic_function_calling"] = genai_types.AutomaticFunctionCallingConfig(
        disable=True,
    )

    config_obj = genai_types.GenerateContentConfig(**cfg_kwargs)

    resp = await client.aio.models.generate_content(
        model=config.model,
        contents=contents,
        config=config_obj,
    )
    return resp


# ── Response parsing ─────────────────────────────────────────────────────────

def parse_openai_response(resp) -> tuple[str | None, list[dict], int, int, int]:
    """Parse OpenAI response → (text, tool_calls, prompt_tokens, completion_tokens, cached_tokens).

    cached_tokens is the portion of prompt_tokens served from automatic prefix cache.
    Handles: OpenAI (prompt_tokens_details.cached_tokens), DeepSeek (prompt_cache_hit_tokens),
    Gemini via openai-compat shim (prompt_tokens_details.cached_tokens), Together.ai (varies).
    """
    msg = resp.choices[0].message
    text = msg.content

    # Debug: dump full raw response for diagnosing local model issues
    raw_dump = {}
    if hasattr(resp, "model_dump"):
        try:
            raw_dump = resp.model_dump()
            choice = raw_dump.get("choices", [{}])[0]
            raw_msg = choice.get("message", {})
            _KNOWN_KEYS = {"content", "role", "tool_calls", "function_call",
                          "reasoning_content", "reasoning", "refusal", "annotations", "audio",
                          "extra_content"}
            extra = {k: v for k, v in raw_msg.items()
                     if k not in _KNOWN_KEYS and v}
            if extra or (not text and not msg.tool_calls):
                import sys
                print(f"  [DEBUG] Raw message keys: {list(raw_msg.keys())}", file=sys.stderr)
                for k, v in extra.items():
                    preview = str(v)[:200] if v else "(empty)"
                    print(f"  [DEBUG]   {k}: {preview}", file=sys.stderr)
                if not text and not msg.tool_calls:
                    print(f"  [DEBUG] content={repr(raw_msg.get('content',''))[:100]}", file=sys.stderr)
                    print(f"  [DEBUG] tool_calls={raw_msg.get('tool_calls')}", file=sys.stderr)
        except Exception:
            pass

    # Handle reasoning models that put thinking in a separate field.
    # Qwen3.5 uses `reasoning_content`; Kimi-K2.5 (Moonshot) uses `reasoning`.
    raw_msg = raw_dump.get("choices", [{}])[0].get("message", {}) if raw_dump else {}
    reasoning = (
        getattr(msg, "reasoning_content", None)
        or raw_msg.get("reasoning_content")
        or getattr(msg, "reasoning", None)
        or raw_msg.get("reasoning")
    )
    if not text and reasoning:
        text = reasoning  # use reasoning as text if content is empty
    elif reasoning and text:
        text = f"[thinking] {reasoning}\n[answer] {text}"

    tool_calls = []
    if msg.tool_calls:
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            tool_calls.append({
                "id": tc.id,
                "name": tc.function.name,
                "args": args,
            })
    usage = resp.usage
    cached = 0
    # OpenAI / Gemini (openai-compat) / Together.ai
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", 0) or 0
    # DeepSeek exposes a different field
    if cached == 0:
        cached = getattr(usage, "prompt_cache_hit_tokens", 0) or 0
    return text, tool_calls, usage.prompt_tokens, usage.completion_tokens, cached


def parse_anthropic_response(resp) -> tuple[str | None, list[dict], int, int, int, int]:
    """Parse Anthropic response → (text, tool_calls, input_tokens, output_tokens, cache_read, cache_creation)."""
    text_parts = []
    tool_calls = []
    thinking_parts = []
    for block in resp.content:
        if block.type == "thinking":
            thinking_parts.append(block.thinking)
        elif block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append({
                "id": block.id,
                "name": block.name,
                "args": block.input,
            })
    # Merge thinking + text into a single output string
    parts = []
    if thinking_parts:
        parts.append("[thinking] " + " ".join(thinking_parts))
    if text_parts:
        parts.append("\n".join(text_parts))
    text = "\n".join(parts) if parts else None
    usage = resp.usage
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
    return text, tool_calls, usage.input_tokens, usage.output_tokens, cache_read, cache_creation


def parse_google_response(resp):
    """Parse google-genai response → (text, tool_calls, prompt_tokens, completion_tokens, cache_read, model_content).

    model_content is the full Content object that MUST be appended to the
    conversation history verbatim so thought_signature round-trips correctly.
    """
    text_parts = []
    tool_calls = []
    model_content = None

    candidates = getattr(resp, "candidates", None) or []
    if candidates:
        model_content = candidates[0].content
        parts = getattr(model_content, "parts", None) or []
        call_id_counter = 0
        for part in parts:
            if getattr(part, "text", None):
                text_parts.append(part.text)
            fc = getattr(part, "function_call", None)
            if fc is not None:
                args = dict(fc.args) if fc.args else {}
                call_id_counter += 1
                # Gemini function_calls have no id; synthesize one but
                # preserve function_call.id if SDK provides it later.
                call_id = getattr(fc, "id", None) or f"genai_{call_id_counter}"
                tool_calls.append({
                    "id": call_id,
                    "name": fc.name,
                    "args": args,
                })

    text = "\n".join(text_parts) if text_parts else None

    usage = getattr(resp, "usage_metadata", None)
    prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0 if usage else 0
    completion_tokens = (getattr(usage, "candidates_token_count", 0) or 0) if usage else 0
    # Thoughts (reasoning) tokens are billed like completion — include them.
    thoughts = (getattr(usage, "thoughts_token_count", 0) or 0) if usage else 0
    completion_tokens += thoughts
    cache_read = (getattr(usage, "cached_content_token_count", 0) or 0) if usage else 0
    # Subtract cached from regular prompt so total_prompt_tokens is "regular
    # (non-cached) input only" — consistent with the other provider paths.
    prompt_tokens = max(0, prompt_tokens - cache_read)
    return text, tool_calls, prompt_tokens, completion_tokens, cache_read, model_content


# ── Message builders ─────────────────────────────────────────────────────────

def append_tool_results_openai(
    messages: list[dict],
    raw_resp,
    tool_calls: list[dict],
    results: list[ToolResult],
    budget_remaining: int,
    budget_total: int,
    turn_idx: int = -1,
    max_turns: int = 0,
    cumulative_tokens: int = 0,
    max_total_tokens: int = 0,
) -> None:
    """Append assistant message + tool results to OpenAI message list.

    We reconstruct the assistant message dict manually instead of using
    ``raw_resp.choices[0].message.to_dict()`` because some serving frameworks
    (e.g. sglang) cannot deserialize the dict produced by the OpenAI SDK back
    into their internal format — they expect attribute access on nested objects
    but receive plain dicts, causing ``'dict object' has no attribute 'function'``.
    """
    msg = raw_resp.choices[0].message
    assistant_msg: dict = {"role": "assistant"}
    if msg.content:
        assistant_msg["content"] = msg.content
    else:
        assistant_msg["content"] = None
    # Preserve reasoning field for multi-turn continuity (Qwen3 thinking, etc.)
    # vLLM/OpenAI-compat servers return reasoning as a separate field; it must
    # be passed back in the assistant message so the model keeps its CoT context.
    reasoning = (
        getattr(msg, "reasoning_content", None)
        or getattr(msg, "reasoning", None)
    )
    if reasoning:
        assistant_msg["reasoning_content"] = reasoning
    if msg.tool_calls:
        assistant_msg["tool_calls"] = [
            {
                "id": tc_obj.id,
                "type": "function",
                "function": {
                    "name": tc_obj.function.name,
                    "arguments": tc_obj.function.arguments,
                },
            }
            for tc_obj in msg.tool_calls
        ]
    messages.append(assistant_msg)
    for tc, result in zip(tool_calls, results):
        content = result.output + build_budget_notice(
            budget_remaining, budget_total,
            turn_idx, max_turns, cumulative_tokens, max_total_tokens,
        )
        messages.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": content,
        })


def append_tool_results_anthropic(
    messages: list[dict],
    raw_resp,
    tool_calls: list[dict],
    results: list[ToolResult],
    budget_remaining: int,
    budget_total: int,
    turn_idx: int = -1,
    max_turns: int = 0,
    cumulative_tokens: int = 0,
    max_total_tokens: int = 0,
) -> None:
    """Append assistant message + tool results to Anthropic message list.

    Cache-breakpoint policy:
      Anthropic caches everything up to and including a cache_control block.
      A breakpoint on turn T's tool_result means turn T+1's API call reads
      that whole prefix (including the big content) from cache at 0.1× rate.

      - If this turn called read_section (heavy content arrived): strip
        all prior dynamic cache_controls, place a new one on this turn's
        last tool_result. Cost of the write (~1.25× input) pays back on
        the very next turn.
      - If this turn was light (get_paper_info, list_sections, think, etc.):
        leave the prior cache_control intact. No write premium paid; future
        turns still hit cache through the last heavy turn's boundary.

      Static cache_control markers on system_prompt + last tool definition
      plus at most ONE dynamic marker keeps us well under Anthropic's
      4-breakpoint-per-request limit.
    """
    has_heavy = any(tc.get("name") == "read_section" for tc in tool_calls)

    if has_heavy:
        # New heavy content — strip prior dynamic markers, mark this turn.
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                for blk in content:
                    if isinstance(blk, dict) and blk.get("type") == "tool_result":
                        blk.pop("cache_control", None)

    messages.append({
        "role": "assistant",
        "content": [block.to_dict() for block in raw_resp.content],
    })
    tool_result_blocks = []
    for i, (tc, result) in enumerate(zip(tool_calls, results)):
        content = result.output + build_budget_notice(
            budget_remaining, budget_total,
            turn_idx, max_turns, cumulative_tokens, max_total_tokens,
        )
        block = {
            "type": "tool_result",
            "tool_use_id": tc["id"],
            "content": content,
        }
        if has_heavy and i == len(tool_calls) - 1:
            block["cache_control"] = {"type": "ephemeral"}
        tool_result_blocks.append(block)
    messages.append({"role": "user", "content": tool_result_blocks})


def append_tool_results_google(
    contents: list,
    model_content,
    tool_calls: list[dict],
    results: list[ToolResult],
    budget_remaining: int,
    budget_total: int,
    turn_idx: int = -1,
    max_turns: int = 0,
    cumulative_tokens: int = 0,
    max_total_tokens: int = 0,
) -> None:
    """Append model response and tool results to google-genai contents list.

    IMPORTANT: append the full `model_content` object verbatim so
    thought_signature metadata on function_call parts is preserved for
    subsequent turns (otherwise Gemini 3.x returns 400 errors).
    """
    from google.genai import types as genai_types

    # 1. Append the model's response content exactly as received.
    if model_content is not None:
        contents.append(model_content)

    # 2. Append a user-role Content containing function_response parts.
    function_response_parts = []
    for tc, result in zip(tool_calls, results):
        content_text = result.output + build_budget_notice(
            budget_remaining, budget_total,
            turn_idx, max_turns, cumulative_tokens, max_total_tokens,
        )
        function_response_parts.append(
            genai_types.Part.from_function_response(
                name=tc["name"],
                response={"output": content_text},
            )
        )
    contents.append(
        genai_types.Content(role="user", parts=function_response_parts)
    )


# ── Main evaluation loop ─────────────────────────────────────────────────────

async def run_sample(sample: dict, config: HarnessConfig) -> Trajectory:
    """Run a single benchmark sample through the evaluation harness."""
    tools = AgentHopTools(sample)
    budget_remaining = config.budget
    trajectory = Trajectory(
        sample_id=sample["id"],
        model=config.model,
        question_type=sample["question_type"],
        depth=sample["depth"],
        reasoning_type=sample.get("reasoning_type", ""),
        correct_index=sample["correct_index"],
        budget_total=config.budget,
    )

    t0 = time.time()
    system_prompt = build_system_prompt(
        sample, config.budget, budget_remaining, config.strategy,
        text_tools=config.text_tools,
        max_turns=config.max_turns,
        max_total_tokens=config.max_total_tokens,
        disabled_tools=config.disabled_tools,
    )
    first_message = STRATEGY_FIRST_MESSAGES.get(config.strategy, STRATEGY_FIRST_MESSAGES["direct"])

    if config.backend == "anthropic":
        messages = [
            {"role": "user", "content": first_message},
        ]
        genai_contents = None
    elif config.backend == "google":
        # Native google-genai SDK uses types.Content, not OpenAI messages.
        from google.genai import types as genai_types
        messages = None
        genai_contents = [
            genai_types.Content(
                role="user",
                parts=[genai_types.Part.from_text(text=first_message)],
            )
        ]
    else:
        # Both "openai" and "responses" use system + user messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": first_message},
        ]
        genai_contents = None

    answer_submitted = False
    budget_rejections = 0  # consecutive budget-rejected tool calls
    consecutive_errors = 0  # consecutive API/parsing errors
    tool_errors = 0  # total tool call errors (malformed args, unknown tools)

    # Responses API state (Codex models use previous_response_id for multi-turn)
    prev_response_id = None
    responses_input = None  # input for next Responses API call

    for turn_idx in range(config.max_turns):
        # Transient API errors (rate limit, server 5xx) retry without consuming
        # the turn budget; non-transient errors fall through and count as a turn.
        transient_retries = 0
        got_response = False
        while True:
            try:
                if config.backend == "responses":
                    if prev_response_id is None:
                        raw_resp = await call_responses_async(
                            [{"role": "user", "content": first_message}],
                            config,
                            instructions=system_prompt,
                        )
                    else:
                        raw_resp = await call_responses_async(
                            responses_input,
                            config,
                            instructions=system_prompt,
                            previous_response_id=prev_response_id,
                        )
                    prev_response_id = raw_resp.id
                    text, tool_calls, p_tok, c_tok, cached_tok = parse_responses_response(raw_resp)
                    p_tok = max(0, p_tok - cached_tok)
                    trajectory.cache_read_tokens += cached_tok
                elif config.backend == "openai":
                    raw_resp = await call_openai_async(messages, config)
                    text, tool_calls, p_tok, c_tok, cached_tok = parse_openai_response(raw_resp)
                    p_tok = max(0, p_tok - cached_tok)
                    trajectory.cache_read_tokens += cached_tok
                    # Fallback: open models that emit tool calls as text.
                    if not tool_calls and text:
                        tool_calls = parse_text_tool_calls(text)
                        if tool_calls and not config.text_tools:
                            import sys
                            print(f"  [INFO] Parsed {len(tool_calls)} tool call(s) from text output (text fallback)", file=sys.stderr)
                elif config.backend == "google":
                    raw_resp = await call_google_async(genai_contents, system_prompt, config)
                    text, tool_calls, p_tok, c_tok, cached_tok, model_content = parse_google_response(raw_resp)
                    trajectory.cache_read_tokens += cached_tok
                else:
                    raw_resp = await call_anthropic_async(messages, system_prompt, config)
                    text, tool_calls, p_tok, c_tok, cr_tok, cw_tok = parse_anthropic_response(raw_resp)
                    trajectory.cache_read_tokens += cr_tok
                    trajectory.cache_creation_tokens += cw_tok
                got_response = True
                break  # inner while — success
            except Exception as e:
                err_str = str(e)
                is_rate_limit = "429" in err_str or "rate" in err_str.lower() or "quota" in err_str.lower() or "RESOURCE_EXHAUSTED" in err_str
                is_server_error = "500" in err_str or "502" in err_str or "503" in err_str or "overloaded" in err_str.lower()
                if is_rate_limit or is_server_error:
                    transient_retries += 1
                    trajectory.transient_retry_count += 1
                    if transient_retries > 10:
                        trajectory.terminated_by = "error"
                        trajectory.turns.append(TurnRecord(
                            turn=turn_idx, role="error",
                            content=f"Abandoned after {transient_retries} transient retries: {err_str}",
                            timestamp=time.time() - t0,
                        ))
                        break  # inner while
                    # Exponential backoff on dedicated counter
                    wait = min(30 * (2 ** min(transient_retries - 1, 3)), 120)
                    await asyncio.sleep(wait)
                    continue  # inner while — retry without advancing turn_idx
                # Non-transient error: record and advance turn_idx
                consecutive_errors += 1
                trajectory.api_error_count += 1
                trajectory.turns.append(TurnRecord(
                    turn=turn_idx, role="error", content=err_str,
                    timestamp=time.time() - t0,
                ))
                if consecutive_errors >= 5:
                    trajectory.terminated_by = "error"
                else:
                    await asyncio.sleep(min(2 ** consecutive_errors, 15))
                break  # inner while

        if trajectory.terminated_by == "error":
            break  # outer for
        if not got_response:
            continue  # non-transient error: advance to next turn_idx

        consecutive_errors = 0  # reset on successful API call

        turn = TurnRecord(
            turn=turn_idx,
            role="assistant",
            content=text,
            budget_before=budget_remaining,
            timestamp=time.time() - t0,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
        )
        trajectory.total_prompt_tokens += p_tok
        trajectory.total_completion_tokens += c_tok

        # Hard token-limit cutoff (no soft warn — models use full allowance).
        # Count "new tokens processed this sample": regular prompt + cache
        # creation (newly written to cache on this call) + completion. Cache
        # READS are excluded — they are repeats of content already counted in
        # the turn that wrote them, and without this exclusion the cap would
        # fire on long multi-turn samples where the growing prefix is served
        # from cache many times.
        cumulative_tokens = (
            trajectory.total_prompt_tokens
            + trajectory.total_completion_tokens
            + trajectory.cache_creation_tokens
        )
        if cumulative_tokens > config.max_total_tokens:
            trajectory.terminated_by = "token_limit"
            trajectory.turns.append(TurnRecord(
                turn=turn_idx, role="assistant",
                content=f"[HARNESS] Token limit reached ({cumulative_tokens:,} > {config.max_total_tokens:,}). Forcing termination.",
                budget_before=budget_remaining, budget_after=budget_remaining,
                timestamp=time.time() - t0,
                prompt_tokens=p_tok, completion_tokens=c_tok,
            ))
            break

        # No tool calls — agent is just talking
        if not tool_calls:
            turn.budget_after = budget_remaining
            trajectory.turns.append(turn)
            if not text:
                # Model returned empty content AND no tool calls. Preserve whatever
                # raw fields the provider sent (reasoning_content, refusal,
                # channel-tagged partial output, etc.) so we can diagnose later.
                trajectory.terminated_by = "error"
                try:
                    if hasattr(raw_resp, "model_dump"):
                        raw_dump = raw_resp.model_dump()
                        turn.content = (
                            f"[HARNESS] Empty response — terminated as error. "
                            f"Raw dump keys: {list(raw_dump.keys())}"
                        )
                        # Also fold into messages so the conversation trail is complete
                        choice = raw_dump.get("choices", [{}])[0]
                        raw_msg = choice.get("message", {}) if choice else {}
                        if config.backend != "responses":
                            messages.append({
                                "role": "assistant",
                                "content": raw_msg.get("content"),
                                "_error_reason": "empty_response",
                                "_reasoning_content": raw_msg.get("reasoning_content"),
                                "_reasoning": raw_msg.get("reasoning"),
                                "_refusal": raw_msg.get("refusal"),
                            })
                except Exception:
                    pass
                break
            nudge = f"Please use the available tools to investigate, or call submit_answer if you're ready.{build_budget_notice(budget_remaining, config.budget, turn_idx, config.max_turns, cumulative_tokens, config.max_total_tokens)}"
            if config.backend == "responses":
                responses_input = [{"role": "user", "content": nudge}]
            elif config.backend == "google":
                from google.genai import types as genai_types
                if model_content is not None:
                    genai_contents.append(model_content)
                genai_contents.append(
                    genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=nudge)])
                )
            else:
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": nudge})
            continue

        # Execute tool calls (local, instant)
        turn.tool_calls = tool_calls
        results = []
        for tc in tool_calls:
            name = tc["name"]
            args = tc["args"]
            cost = TOOL_COSTS.get(name, 0)

            if name in config.disabled_tools:
                result = ToolResult(
                    name=name, args=args,
                    output=(
                        f"REJECTED: Tool '{name}' is disabled in this run. "
                        "Use only the tools listed in the system prompt."
                    ),
                    cost=0,
                )
                results.append(result)
                turn.tool_results.append({
                    "name": result.name, "args": result.args,
                    "output": result.output, "cost": 0,
                })
                continue

            if name != "submit_answer" and budget_remaining < cost:
                result = ToolResult(
                    name=name, args=args,
                    output=(
                        f"REJECTED: Insufficient budget ({budget_remaining} remaining, {name} costs {cost}). "
                        "You cannot use this tool. Call submit_answer immediately with your best guess based on evidence gathered so far."
                    ),
                    cost=0,
                )
                budget_rejections += 1
            else:
                result = tools.call(name, args)
                budget_remaining -= result.cost
                budget_rejections = 0  # reset on successful call
                # Track tool-level errors (unknown tool, missing args, etc.)
                if result.output.startswith(("Unknown tool", "Error calling", "Tool '")):
                    tool_errors += 1

            results.append(result)
            turn.tool_results.append({
                "name": result.name,
                "args": result.args,
                "output": result.output[:500] + "..." if len(result.output) > 500 else result.output,
                "cost": result.cost,
            })

            if name == "submit_answer":
                answer = args.get("answer", "")
                reasoning = args.get("reasoning", "")
                trajectory.predicted_answer = answer
                trajectory.predicted_index = _parse_answer_letter(answer)
                trajectory.is_correct = trajectory.predicted_index == sample["correct_index"]
                trajectory.reasoning = reasoning
                answer_submitted = True

        turn.budget_after = budget_remaining
        trajectory.turns.append(turn)
        trajectory.budget_used = config.budget - budget_remaining

        if answer_submitted:
            trajectory.terminated_by = "answer"
            break

        # Append results to conversation
        status_kwargs = dict(
            turn_idx=turn_idx,
            max_turns=config.max_turns,
            cumulative_tokens=cumulative_tokens,
            max_total_tokens=config.max_total_tokens,
        )
        if config.backend == "responses":
            responses_input = build_responses_tool_results(
                tool_calls, results, budget_remaining, config.budget, **status_kwargs,
            )
        elif config.backend == "openai" and config.text_tools:
            # Text-tools mode: plain text messages (no tool_call API objects).
            # Strip OSS-style channel markup from the content before re-sending,
            # since sglang's GPT-OSS template rejects inbound history that still
            # contains <|channel|> tags in the content field.
            cleaned_text = sanitize_channel_content(text or "")
            messages.append({"role": "assistant", "content": cleaned_text})
            result_text = _format_text_tool_results(
                tool_calls, results, budget_remaining, config.budget, **status_kwargs,
            )
            messages.append({"role": "user", "content": result_text})
        elif config.backend == "openai":
            append_tool_results_openai(
                messages, raw_resp, tool_calls, results,
                budget_remaining, config.budget, **status_kwargs,
            )
        elif config.backend == "google":
            append_tool_results_google(
                genai_contents, model_content, tool_calls, results,
                budget_remaining, config.budget, **status_kwargs,
            )
        else:
            append_tool_results_anthropic(
                messages, raw_resp, tool_calls, results,
                budget_remaining, config.budget, **status_kwargs,
            )

        # Hard terminate on two consecutive budget-rejected tool calls: the
        # model is ignoring the rejection signal and would just churn until
        # max_turns otherwise. This classifies cleanly as "budget" rather
        # than polluting the "max_turns" bucket.
        if budget_rejections >= 2:
            trajectory.terminated_by = "budget"
            break

        # Soft advisory on first budget exhaustion — give the model one
        # chance to submit before the next rejection triggers the hard stop.
        force_msg = None
        if budget_remaining <= 0:
            force_msg = "Budget exhausted. You MUST call submit_answer now with your best guess."

        if force_msg:
            if config.backend == "responses":
                # Append nudge to the tool results already in responses_input
                responses_input.append({"role": "user", "content": force_msg})
            elif config.backend == "google":
                from google.genai import types as genai_types
                genai_contents.append(
                    genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=force_msg)])
                )
            else:
                messages.append({"role": "user", "content": force_msg})
            continue

    else:
        if not answer_submitted:
            trajectory.terminated_by = "max_turns"

    trajectory.total_turns = len(trajectory.turns)
    trajectory.tool_errors = tool_errors
    trajectory.wall_time_s = time.time() - t0

    # Save serializable copy of full conversation history
    import base64
    def _bytes_safe(obj):
        """Recursively convert bytes → base64-prefixed strings so json.dump works.
        Gemini's thought_signature is raw bytes; preserve via base64 encoding."""
        if isinstance(obj, bytes):
            return "b64:" + base64.b64encode(obj).decode("ascii")
        if isinstance(obj, dict):
            return {k: _bytes_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_bytes_safe(x) for x in obj]
        return obj

    def _serialize_msg(msg):
        """Convert message to JSON-serializable dict."""
        if isinstance(msg, dict):
            return _bytes_safe(msg)
        if hasattr(msg, 'to_dict'):
            return _bytes_safe(msg.to_dict())
        if hasattr(msg, 'model_dump'):
            return _bytes_safe(msg.model_dump())
        return str(msg)

    hist = genai_contents if config.backend == "google" else messages
    trajectory.messages = [_serialize_msg(m) for m in (hist or [])]

    return trajectory


# ── Async batch runner ───────────────────────────────────────────────────────

async def _worker(
    name: str,
    queue: asyncio.Queue,
    config: HarnessConfig,
    out_path: Path,
    results: list,
    lock: asyncio.Lock,
    counter: dict,
):
    """Worker coroutine that pulls samples from the queue."""
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break

        idx, sample = item
        sid = sample["id"]

        # Sample-level retry: up to 3 attempts (1 original + 2 retries).
        # Retries fire ONLY when terminated_by=="error" (API/infra failure);
        # model-behavior failures (max_turns/budget/token_limit) are final.
        # Token costs from failed attempts are preserved for accurate cost
        # reconstruction against provider invoices.
        max_retries = 3
        retry_prompt = retry_completion = retry_cache_r = retry_cache_w = 0
        retry_api_err = retry_transient = 0
        attempt = 0  # 0-indexed; equals number of retries that preceded final attempt
        for attempt in range(max_retries):
            traj = await run_sample(sample, config)
            if traj.terminated_by != "error":
                break
            # Save failed attempt to sidecar for forensic analysis before discarding.
            # The canonical per-sample JSON tracks only the final attempt; forensic
            # traces live under _failed_attempts/ so main-pipeline schema is stable.
            failed_dir = out_path / "_failed_attempts"
            failed_dir.mkdir(exist_ok=True)
            with open(failed_dir / f"{sid}.attempt_{attempt}.json", "w") as f:
                json.dump(asdict(traj), f, indent=2, ensure_ascii=False)
            retry_prompt += traj.total_prompt_tokens
            retry_completion += traj.total_completion_tokens
            retry_cache_r += traj.cache_read_tokens
            retry_cache_w += traj.cache_creation_tokens
            retry_api_err += traj.api_error_count
            retry_transient += traj.transient_retry_count
            if attempt < max_retries - 1:
                wait = 30 * (attempt + 1)
                await asyncio.sleep(wait)
        # Fold pre-retry costs into the final trajectory
        traj.total_prompt_tokens += retry_prompt
        traj.total_completion_tokens += retry_completion
        traj.cache_read_tokens += retry_cache_r
        traj.cache_creation_tokens += retry_cache_w
        traj.api_error_count += retry_api_err
        traj.transient_retry_count += retry_transient
        # retry_count = number of retries done before final attempt. Whether
        # the final attempt succeeded or all failed, `attempt` holds the
        # 0-indexed final attempt number, which equals retries preceding it.
        traj.retry_count = attempt

        # Save result
        result_file = out_path / f"{sid}.json"
        with open(result_file, "w") as f:
            json.dump(asdict(traj), f, indent=2, ensure_ascii=False)

        async with lock:
            results.append(traj)
            counter["done"] += 1
            counter["correct"] += int(traj.is_correct)
            # Run-wide counter uses same "new tokens" semantics as the
            # per-sample cap (regular + completion + cache creation; cache
            # reads are repeats and excluded).
            counter["tokens"] = counter.get("tokens", 0) + (
                traj.total_prompt_tokens + traj.total_completion_tokens
                + traj.cache_creation_tokens
            )
            n = counter["done"]
            total = counter["total"]
            acc = counter["correct"] / n if n > 0 else 0
            run_tok = counter["tokens"]
            kill = config.max_run_tokens and run_tok > config.max_run_tokens

        status = "✓" if traj.is_correct else "✗"
        print(
            f"[{n}/{total}] {sid} "
            f"{status} [{traj.terminated_by}] "
            f"ans={traj.predicted_answer} "
            f"budget={traj.budget_used}/{config.budget} "
            f"turns={traj.total_turns} "
            f"t={traj.wall_time_s:.1f}s "
            f"(acc={acc:.1%})",
            flush=True,
        )
        queue.task_done()

        if kill:
            print(
                f"\n⚠️  KILL-SWITCH: run tokens {run_tok:,} exceed "
                f"max_run_tokens={config.max_run_tokens:,}. Draining queue.",
                flush=True,
            )
            # Drain queue without running more samples
            while True:
                try:
                    item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                queue.task_done()
            return


async def run_benchmark(
    samples: list[dict],
    config: HarnessConfig,
    output_dir: str = "results",
    resume: bool = True,
    agenthop_dir: str | Path | None = None,
) -> list[Trajectory]:
    """Run evaluation on multiple samples with async parallelism.

    Args:
        samples: List of benchmark sample dicts.
        config: Harness configuration.
        output_dir: Directory to save per-sample trajectories.
        resume: Skip samples that already have result files.

    Returns:
        List of Trajectory objects.
    """
    model_dir = config.model.replace("/", "_")
    if config.strategy != "direct":
        model_dir = f"{model_dir}_{config.strategy}"

    # Guard against double-nesting: if output_dir already ends with the model name, use it as-is
    out_dir = Path(output_dir)
    if out_dir.name == model_dir:
        out_path = out_dir
    else:
        out_path = out_dir / model_dir
    out_path.mkdir(parents=True, exist_ok=True)

    # Separate resumed vs pending
    trajectories = []
    pending = []
    for i, sample in enumerate(samples):
        sid = sample["id"]
        result_file = out_path / f"{sid}.json"
        if resume and result_file.exists():
            with open(result_file) as f:
                saved = json.load(f)
            traj = Trajectory(**{k: v for k, v in saved.items() if k != "turns"})
            traj.turns = [TurnRecord(**t) for t in saved.get("turns", [])]
            trajectories.append(traj)
        else:
            pending.append((i, sample))

    resumed = len(trajectories)
    if resumed:
        correct_resumed = sum(1 for t in trajectories if t.is_correct)
        print(f"Resumed {resumed} completed samples ({correct_resumed} correct)")

    if not pending:
        print("All samples already completed.")
        _print_summary(trajectories, config, out_path, samples, agenthop_dir)
        return trajectories

    print(f"Running {len(pending)} samples with {config.workers} workers...")

    # Async worker pool
    queue: asyncio.Queue = asyncio.Queue()
    results: list[Trajectory] = []
    lock = asyncio.Lock()
    resumed_tokens = sum(
        t.total_prompt_tokens + t.total_completion_tokens + t.cache_creation_tokens
        for t in trajectories
    )
    counter = {
        "done": 0,
        "correct": 0,
        "total": len(pending),
        "tokens": resumed_tokens,  # run-wide total; resumed samples count toward the cap
    }

    for item in pending:
        await queue.put(item)
    # Poison pills
    for _ in range(config.workers):
        await queue.put(None)

    workers = [
        asyncio.create_task(
            _worker(f"w{i}", queue, config, out_path, results, lock, counter)
        )
        for i in range(config.workers)
    ]

    await asyncio.gather(*workers)

    all_trajectories = trajectories + results
    _print_summary(all_trajectories, config, out_path, samples, agenthop_dir)
    return all_trajectories


def _print_summary(
    trajectories: list[Trajectory],
    config: HarnessConfig,
    out_path: Path,
    samples: list[dict] | None = None,
    agenthop_dir: str | Path | None = None,
):
    """Print console summary and write full evaluation to summary.json.

    The summary always reflects EVERYTHING in the output dir (scanned fresh
    from disk), not just the samples loaded in the current run. This keeps
    filtered runs (e.g. --question-type multi-target --depth 2) from
    clobbering a broader run's aggregate. Use `--output-dir` to isolate
    per-slice summaries.
    """
    # Load every per-sample trajectory in out_path so summary isn't skewed by
    # filtered-run subsets. Falls back to passed trajectories if the dir has
    # nothing readable (e.g., mid-migration).
    disk_trajs = []
    for f in sorted(out_path.glob("*.json")):
        if f.name == "summary.json":
            continue
        try:
            saved = json.load(open(f))
            t = Trajectory(**{k: v for k, v in saved.items() if k != "turns"})
            t.turns = [TurnRecord(**tr) for tr in saved.get("turns", [])]
            disk_trajs.append(t)
        except (json.JSONDecodeError, TypeError, OSError):
            continue
    if disk_trajs:
        trajectories = disk_trajs

    total = len(trajectories)
    if total == 0:
        return

    correct = sum(1 for t in trajectories if t.is_correct)
    acc = correct / total

    print(f"\n{'═' * 60}")
    print(f"  Model: {config.model} | Strategy: {config.strategy}")
    print(f"  Accuracy: {correct}/{total} ({acc:.1%})")
    print(f"  Avg budget used: {sum(t.budget_used for t in trajectories) / total:.1f}/{config.budget}")
    print(f"  Avg turns: {sum(t.total_turns for t in trajectories) / total:.1f}")
    print(f"  Avg wall time: {sum(t.wall_time_s for t in trajectories) / total:.1f}s")
    print(f"  Avg tokens: {sum(t.total_prompt_tokens for t in trajectories) / total:.0f} prompt + {sum(t.total_completion_tokens for t in trajectories) / total:.0f} completion")
    total_cache_read = sum(t.cache_read_tokens for t in trajectories)
    total_cache_write = sum(t.cache_creation_tokens for t in trajectories)
    total_api_errors = sum(t.api_error_count for t in trajectories)
    total_transient = sum(t.transient_retry_count for t in trajectories)
    total_tool_errors = sum(t.tool_errors for t in trajectories)
    if total_cache_read or total_cache_write:
        print(f"  Cache: {total_cache_read:,} read + {total_cache_write:,} write tokens")
    if total_api_errors or total_transient or total_tool_errors:
        print(f"  Errors: {total_api_errors} API, {total_transient} transient retries, {total_tool_errors} tool-call")
    print(f"{'═' * 60}")

    # Full evaluation (accuracy/retrieval/tools/cost/errors) — requires samples
    from evaluate import (
        build_evaluation,
        load_recall_labels,
        benchmark_sha256,
        trajectory_to_dict,
    )
    from datetime import datetime, timezone

    # Always build samples_by_id from the unfiltered benchmark so evaluation
    # metadata (distractor_types, gold_arxiv_ids) is available for every
    # trajectory on disk, even if the current CLI run applied filters.
    samples_by_id = {s["id"]: s for s in (samples or [])}
    if agenthop_dir:
        try:
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from data_loader import load_agenthop
            unfiltered = load_agenthop(Path(agenthop_dir))
            for s in unfiltered:
                samples_by_id.setdefault(s["id"], s)
        except Exception:
            pass  # best-effort; fall back to passed `samples`
    recall_by_id = load_recall_labels(Path(agenthop_dir)) if agenthop_dir else {}
    benchmark_sha = benchmark_sha256(Path(agenthop_dir)) if agenthop_dir else ""
    traj_dicts = [trajectory_to_dict(t) for t in trajectories]

    config_dict = asdict(config)
    for _secret in ("api_key", "base_url"):
        config_dict.pop(_secret, None)
    eval_report = build_evaluation(
        trajectories=traj_dicts,
        samples_by_id=samples_by_id,
        recall_by_id=recall_by_id,
        config_dict=config_dict,
        benchmark_sha=benchmark_sha,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )

    with open(out_path / "summary.json", "w") as f:
        json.dump(eval_report, f, indent=2, ensure_ascii=False)
    print(f"  Summary saved to {out_path / 'summary.json'}")
