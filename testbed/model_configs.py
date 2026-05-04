"""AgentHop model configuration registry.

Provides two complementary registries:
  (1) Tool-calling parser configs for local serving (sglang/vllm)
  (2) Recommended inference hyperparameters per model (temperature,
      top_p, top_k, reasoning_effort, max_tokens)

Usage:
    from model_configs import get_model_config, suggest_server_command, get_hyperparams

    cfg = get_model_config("google/gemma-3-27b-it")  # tool-calling config
    hp = get_hyperparams("claude-sonnet-4-6")        # sampling defaults
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ── Tool call support levels ────────────────────────────────────────────────

TOOL_SUPPORT_SERVER = "server"      # Use server-side --tool-call-parser (preferred)
TOOL_SUPPORT_TEXT = "text"          # Use --text-tools (server has no parser for this model)
TOOL_SUPPORT_NATIVE = "native"     # API provider handles it (OpenAI, Anthropic, etc.)


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a model family's tool-calling support."""
    family: str                          # e.g. "gemma3", "llama3", "qwen3"
    display_name: str                    # e.g. "Gemma 3"

    # Server-side tool-call parser names (None = not supported)
    sglang_parser: str | None = None
    vllm_parser: str | None = None

    # Custom chat template required? (path relative to engine's examples/)
    vllm_chat_template: str | None = None
    sglang_chat_template: str | None = None

    # Preferred tool-call method
    tool_call_support: str = TOOL_SUPPORT_SERVER

    # Extra server flags
    sglang_extra_flags: list[str] = field(default_factory=list)
    vllm_extra_flags: list[str] = field(default_factory=list)

    # Known issues / notes
    notes: str = ""

    # Text-tools parsing hints (which parse_text_tool_calls patterns work)
    text_tool_patterns: list[str] = field(default_factory=list)


# ── Model family registry ───────────────────────────────────────────────────

_REGISTRY: dict[str, ModelConfig] = {}


def _register(prefixes: list[str], config: ModelConfig) -> None:
    """Register a config for multiple model name prefixes."""
    for prefix in prefixes:
        _REGISTRY[prefix.lower()] = config


# ── Gemma 3 ─────────────────────────────────────────────────────────────────

_register(
    ["google/gemma-3-", "gemma-3-"],
    ModelConfig(
        family="gemma3",
        display_name="Gemma 3",
        sglang_parser=None,  # sglang has no dedicated Gemma parser yet
        vllm_parser="pythonic",
        vllm_chat_template="examples/tool_chat_template_gemma3_pythonic.jinja",
        tool_call_support=TOOL_SUPPORT_SERVER,
        notes=(
            "Gemma 3 has no dedicated tool-call tokens. Requires a custom Jinja "
            "template that instructs the model to use pythonic format [func(arg=val)]. "
            "sglang does not have a dedicated parser — use vllm, or use --text-tools "
            "with sglang as fallback."
        ),
        text_tool_patterns=["pythonic", "tool_code", "bare_python"],
    ),
)

# ── Gemma 4 ─────────────────────────────────────────────────────────────────

_register(
    ["google/gemma-4-", "gemma-4-"],
    ModelConfig(
        family="gemma4",
        display_name="Gemma 4",
        sglang_parser="gemma4",
        vllm_parser="gemma4",
        tool_call_support=TOOL_SUPPORT_SERVER,
        sglang_extra_flags=["--reasoning-parser gemma4"],
        vllm_extra_flags=["--reasoning-parser gemma4"],
        notes=(
            "Gemma 4 uses native special-token tool calling format "
            "(<|tool_call>...<tool_call|>). Both sglang and vllm have dedicated "
            "gemma4 parsers (added day-0). Also supports a reasoning parser for "
            "structured <|channel> tokens."
        ),
        text_tool_patterns=["json"],
    ),
)

# ── Llama 3.x ──────────────────────────────────────────────────────────────

_register(
    ["meta-llama/llama-3.1-", "meta-llama/llama-3.3-", "llama-3.1-", "llama-3.3-"],
    ModelConfig(
        family="llama3",
        display_name="Llama 3.1/3.3",
        sglang_parser="llama3",
        vllm_parser="llama3_json",
        vllm_chat_template="examples/tool_chat_template_llama3.1_json.jinja",
        tool_call_support=TOOL_SUPPORT_SERVER,
        notes="Well-supported. Uses <|python_tag|> + JSON format.",
        text_tool_patterns=["tool_call", "json"],
    ),
)

_register(
    ["meta-llama/llama-3.2-", "llama-3.2-"],
    ModelConfig(
        family="llama3_small",
        display_name="Llama 3.2 (1B/3B)",
        sglang_parser="pythonic",
        vllm_parser="pythonic",
        tool_call_support=TOOL_SUPPORT_SERVER,
        notes="Small Llama 3.2 models use pythonic format [func(arg=val)].",
        text_tool_patterns=["pythonic", "bare_python"],
    ),
)

_register(
    ["meta-llama/llama-4-", "llama-4-"],
    ModelConfig(
        family="llama4",
        display_name="Llama 4",
        sglang_parser="llama4",
        vllm_parser="llama4_pythonic",
        tool_call_support=TOOL_SUPPORT_SERVER,
        notes="Llama 4 supports both JSON and pythonic; pythonic is recommended.",
        text_tool_patterns=["pythonic", "bare_python"],
    ),
)

# ── Qwen ────────────────────────────────────────────────────────────────────

_register(
    ["qwen/qwen2.5-", "qwen2.5-"],
    ModelConfig(
        family="qwen25",
        display_name="Qwen 2.5",
        sglang_parser="qwen25",
        vllm_parser="hermes",
        tool_call_support=TOOL_SUPPORT_SERVER,
        notes="Uses Hermes-style <tool_call> XML tags. Well-supported.",
        text_tool_patterns=["tool_call", "json"],
    ),
)

_register(
    ["qwen/qwen3-", "qwen3-"],
    ModelConfig(
        family="qwen3",
        display_name="Qwen 3",
        sglang_parser="qwen",
        vllm_parser="hermes",
        tool_call_support=TOOL_SUPPORT_SERVER,
        notes=(
            "Uses Hermes-style <tool_call> XML tags. For thinking/reasoning models, "
            "avoid ReAct-style stop-word templates — the model may emit stop words "
            "inside <think> blocks."
        ),
        text_tool_patterns=["tool_call", "json"],
    ),
)

_register(
    ["qwen/qwen3.5-", "qwen3.5-"],
    ModelConfig(
        family="qwen35",
        display_name="Qwen 3.5",
        sglang_parser="qwen",
        vllm_parser="hermes",
        tool_call_support=TOOL_SUPPORT_SERVER,
        notes="Same as Qwen 3. May put reasoning in reasoning_content field.",
        text_tool_patterns=["tool_call", "json"],
    ),
)

# ── Mistral ─────────────────────────────────────────────────────────────────

_register(
    ["mistralai/mistral-", "mistralai/mixtral-", "mistral-", "mixtral-"],
    ModelConfig(
        family="mistral",
        display_name="Mistral/Mixtral",
        sglang_parser="mistral",
        vllm_parser="mistral",
        tool_call_support=TOOL_SUPPORT_SERVER,
        notes="Uses [TOOL_CALLS] special token. Supported from v0.3 onwards.",
        text_tool_patterns=["json"],
    ),
)

# ── DeepSeek ────────────────────────────────────────────────────────────────

_register(
    ["deepseek-ai/deepseek-v3", "deepseek-v3"],
    ModelConfig(
        family="deepseek_v3",
        display_name="DeepSeek V3",
        sglang_parser="deepseekv3",
        vllm_parser="deepseek_v3",
        vllm_chat_template="examples/tool_chat_template_deepseekv3.jinja",
        tool_call_support=TOOL_SUPPORT_SERVER,
        notes="Requires matching chat template for local deployment.",
        text_tool_patterns=["json", "tool_call"],
    ),
)

_register(
    ["deepseek-ai/deepseek-r1", "deepseek-r1"],
    ModelConfig(
        family="deepseek_r1",
        display_name="DeepSeek R1",
        sglang_parser="deepseekv3",
        vllm_parser="deepseek_v3",
        vllm_chat_template="examples/tool_chat_template_deepseekv3.jinja",
        tool_call_support=TOOL_SUPPORT_SERVER,
        notes="Same parser as DeepSeek V3. Reasoning model — may have long think tokens.",
        text_tool_patterns=["json", "tool_call"],
    ),
)

# ── GPT-OSS (open-source GPT variants) ─────────────────────────────────────

_register(
    ["gpt-oss-", "openai/gpt-oss-"],
    ModelConfig(
        family="gpt_oss",
        display_name="GPT-OSS",
        sglang_parser="gpt-oss",
        vllm_parser="openai",
        tool_call_support=TOOL_SUPPORT_SERVER,
        notes="OpenAI open-source models. sglang filters analysis channel events.",
        text_tool_patterns=["channel_message", "json"],
    ),
)

# ── Command-R ───────────────────────────────────────────────────────────────

_register(
    ["cohereforai/c4ai-command-r", "command-r"],
    ModelConfig(
        family="command_r",
        display_name="Command-R",
        sglang_parser=None,
        vllm_parser=None,
        tool_call_support=TOOL_SUPPORT_TEXT,
        notes=(
            "Not in sglang/vllm built-in parser list. Requires custom chat template "
            "and parser plugin, or use --text-tools fallback."
        ),
        text_tool_patterns=["json"],
    ),
)


# ── Lookup functions ────────────────────────────────────────────────────────

def get_model_config(model_name: str) -> ModelConfig | None:
    """Look up config for a model by name.

    Matches by longest prefix. Returns None if no match.
    """
    name = model_name.lower()
    best_match = None
    best_len = 0
    for prefix, config in _REGISTRY.items():
        if name.startswith(prefix) and len(prefix) > best_len:
            best_match = config
            best_len = len(prefix)
    return best_match


def list_supported_families() -> list[ModelConfig]:
    """Return unique model configs (deduplicated by family)."""
    seen = set()
    result = []
    for config in _REGISTRY.values():
        if config.family not in seen:
            seen.add(config.family)
            result.append(config)
    return result


def suggest_server_command(
    model_name: str,
    engine: str = "sglang",
    port: int = 8000,
    tp: int = 1,
    extra_flags: str = "",
) -> str | None:
    """Generate a suggested server launch command for a model.

    Returns None if the model isn't in the registry.
    """
    cfg = get_model_config(model_name)
    if cfg is None:
        return None

    if engine == "sglang":
        parser = cfg.sglang_parser
        if parser is None:
            return (
                f"# No sglang parser for {cfg.display_name}. "
                f"Use vllm instead, or run with --text-tools.\n"
                f"# See: {cfg.notes}"
            )
        parts = [
            "python3 -m sglang.launch_server",
            f"  --model-path {model_name}",
            f"  --tool-call-parser {parser}",
            f"  --port {port}",
        ]
        if tp > 1:
            parts.append(f"  --tp {tp}")
        if cfg.sglang_chat_template:
            parts.append(f"  --chat-template {cfg.sglang_chat_template}")
        for flag in cfg.sglang_extra_flags:
            parts.append(f"  {flag}")

    elif engine == "vllm":
        parser = cfg.vllm_parser
        if parser is None:
            return (
                f"# No vllm parser for {cfg.display_name}. "
                f"Use --text-tools fallback.\n"
                f"# See: {cfg.notes}"
            )
        parts = [
            f"vllm serve {model_name}",
            "  --enable-auto-tool-choice",
            f"  --tool-call-parser {parser}",
            f"  --port {port}",
        ]
        if tp > 1:
            parts.append(f"  --tensor-parallel-size {tp}")
        if cfg.vllm_chat_template:
            parts.append(f"  --chat-template {cfg.vllm_chat_template}")
        for flag in cfg.vllm_extra_flags:
            parts.append(f"  {flag}")
    else:
        return None

    if extra_flags:
        parts.append(f"  {extra_flags}")

    return " \\\n".join(parts)


# ── Recommended inference hyperparameters ───────────────────────────────────
#
# Sourced from official docs as of 2026-04. Only entries that have a clear
# recommendation are set; the rest fall back to harness defaults (T=0.7,
# top_p/top_k=None, reasoning_effort=None, max_tokens=4096).
#
# Gotchas encoded per model:
#   - deepseek-reasoner: temperature/top_p silently ignored. Function
#     calling is supported (as of early 2026), but tool_choice='required'
#     returns 400; the harness uses 'auto' so this is fine.
#   - gemini-3*: max_output_tokens default 8K is too low; bump to 65K.
#   - kimi-k2.5: temp depends on mode (thinking=1.0, instant=0.6).
#   - claude 4.6: use adaptive thinking via `reasoning_effort="medium"` which
#     the harness translates to Anthropic's thinking config.

@dataclass(frozen=True)
class HyperparameterPreset:
    """Recommended inference knobs for a model family."""
    # None means "don't set (use API/harness default)"
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    # reasoning_effort: "low"/"medium"/"high" (OpenAI/Anthropic) or
    # token-budget string e.g. "8192" (Anthropic legacy); or None (non-reasoning)
    reasoning_effort: str | None = None
    max_tokens: int | None = None           # per-turn generation cap
    # Flag: if True, this model cannot run this benchmark (e.g., no tool-use)
    unsupported_reason: str | None = None
    # Free-text note for humans
    notes: str = ""


_HYPERPARAMS: dict[str, HyperparameterPreset] = {}


def _hp(prefixes: list[str], preset: HyperparameterPreset) -> None:
    for p in prefixes:
        _HYPERPARAMS[p.lower()] = preset


# Conventions:
#   - Only NON-default values are set. Defaults: temperature=1.0,
#     top_p=0.95, max_tokens=16384 (all handled by HarnessConfig).
#   - top_k is always explicit when set (no universal default).
#   - reasoning_effort set only for reasoning-capable models.

# Anthropic Claude 4.6 — adaptive thinking
_hp(["claude-opus-4-6"], HyperparameterPreset(
    reasoning_effort="medium",
    notes="Adaptive thinking; effort='high' for harder reasoning.",
))
_hp(["claude-sonnet-4-6"], HyperparameterPreset(
    reasoning_effort="medium",
))
# Earlier Claude 4.x — legacy budget_tokens thinking
_hp(["claude-opus-4-1", "claude-opus-4", "claude-opus-4-5"], HyperparameterPreset(
    reasoning_effort="8192",
))
_hp(["claude-sonnet-4-5", "claude-sonnet-4"], HyperparameterPreset(
    reasoning_effort="8192",
))
_hp(["claude-haiku-4-5", "claude-haiku-4"], HyperparameterPreset(
    notes="No thinking mode.",
))

# OpenAI GPT-5 family — reasoning models (temperature/top_p silently ignored)
_hp(["gpt-5.4"], HyperparameterPreset(
    reasoning_effort="medium",
    notes="Reasoning model routed to /v1/responses backend.",
))
_hp(["gpt-5.3-codex"], HyperparameterPreset(
    reasoning_effort="medium",
    notes="Responses-API only.",
))
_hp(["gpt-5"], HyperparameterPreset(
    reasoning_effort="medium",
))

# DeepSeek
_hp(["deepseek-chat"], HyperparameterPreset(
    max_tokens=8192,
    notes="API caps max_tokens at 8192. Instruction-tuned; defaults T=1.0 top_p=0.95.",
))
_hp(["deepseek-reasoner"], HyperparameterPreset(
    max_tokens=8192,
    notes=(
        "Reasoning model; function calling supported (early 2026). "
        "tool_choice='required' returns 400 — harness uses 'auto'. "
        "Temperature/top_p silently ignored."
    ),
))

# Google Gemini 3 — set top_k and thinking; leave T/top_p/max_tokens at defaults
_hp(["gemini-3.1-pro"], HyperparameterPreset(
    top_k=64,
    reasoning_effort="medium",
))
_hp(["gemini-3.1-flash", "gemini-3-flash-preview", "gemini-3-flash"], HyperparameterPreset(
    top_k=64,
    reasoning_effort="medium",
))
_hp(["gemini-2.5-pro"], HyperparameterPreset(
    top_k=64,
))
_hp(["gemini-2.5-flash"], HyperparameterPreset(
    top_k=64,
))

# Together.ai
_hp(["zai-org/glm-5.1", "glm-5.1"], HyperparameterPreset(
    notes="No official sampling params from Zhipu; use API defaults.",
))
_hp(["minimaxai/minimax-m2.7", "minimax-m2.7"], HyperparameterPreset(
    top_k=40,
    notes="Interleaved thinking; set reasoning_split=True on Together endpoint.",
))
_hp(["moonshotai/kimi-k2.5", "kimi-k2.5"], HyperparameterPreset(
    temperature=0.6,
    notes=(
        "Instant mode (T=0.6). Thinking mode (T=1.0) causes over-exploration "
        "and commit-failure in agentic workflows — observed 13% no_answer rate "
        "in thinking-mode pilot (2026-04) vs ~2% for other API models."
    ),
))

# Self-hosted open-weight (via sglang/vllm)
_hp(["qwen/qwen3.5-", "qwen3.5-"], HyperparameterPreset(
    top_k=20,
    notes="Thinking mode defaults. Non-thinking would use T=0.7, top_p=0.8.",
))
_hp(["qwen/qwen3-", "qwen3-"], HyperparameterPreset(
    temperature=0.7,
    top_p=0.8,
    top_k=20,
))
_hp(["google/gemma-4-", "gemma-4-"], HyperparameterPreset(
    top_k=64,
    notes="Gemma 4 configurable thinking + function calling.",
))
_hp(["google/gemma-3-", "gemma-3-"], HyperparameterPreset(
    top_k=64,
))
_hp(["meta-llama/llama-3.3-", "llama-3.3-"], HyperparameterPreset(
    temperature=0.6,
    top_p=0.9,
    notes="Instruct model; lower T reduces multi-turn drift.",
))


def get_hyperparams(model_name: str) -> HyperparameterPreset | None:
    """Look up recommended hyperparameters by longest prefix match."""
    name = model_name.lower()
    best = None
    best_len = 0
    for prefix, preset in _HYPERPARAMS.items():
        if (name.startswith(prefix) or name == prefix) and len(prefix) > best_len:
            best = preset
            best_len = len(prefix)
    return best


def check_tool_support(model_name: str, engine: str = "sglang") -> dict:
    """Check tool-calling support for a model + engine combo.

    Returns a dict with:
      - supported: bool
      - method: "server" | "text" | None
      - parser: str | None
      - suggestion: str (human-readable recommendation)
    """
    cfg = get_model_config(model_name)
    if cfg is None:
        return {
            "supported": True,
            "method": "text",
            "parser": None,
            "suggestion": (
                f"Model '{model_name}' not in registry. Using --text-tools fallback. "
                "If this model supports tool calling, consider adding it to model_configs.py."
            ),
        }

    parser = cfg.sglang_parser if engine == "sglang" else cfg.vllm_parser

    if parser:
        cmd = suggest_server_command(model_name, engine=engine)
        return {
            "supported": True,
            "method": "server",
            "parser": parser,
            "suggestion": (
                f"Start server with --tool-call-parser {parser}. "
                f"Example:\n{cmd}"
            ),
        }
    else:
        alt_engine = "vllm" if engine == "sglang" else "sglang"
        alt_parser = cfg.vllm_parser if engine == "sglang" else cfg.sglang_parser
        if alt_parser:
            return {
                "supported": True,
                "method": "server",
                "parser": alt_parser,
                "suggestion": (
                    f"No {engine} parser for {cfg.display_name}. "
                    f"Use {alt_engine} with --tool-call-parser {alt_parser} instead, "
                    f"or use --text-tools as fallback."
                ),
            }
        return {
            "supported": True,
            "method": "text",
            "parser": None,
            "suggestion": (
                f"No server-side tool parser for {cfg.display_name}. "
                f"Use --text-tools flag for text-based tool calling."
            ),
        }
