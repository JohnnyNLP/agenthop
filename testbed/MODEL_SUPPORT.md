# AgentHop Local Model Support

This document describes how to run local LLMs against the AgentHop benchmark using **sglang** or **vllm** as inference backends.

## Quick Start

```bash
# Via the helper script (handles server launch + eval automatically)
bash evaluation/run_local.sh Qwen/Qwen3-8B --gpus 0

# Or manually: start server, then run eval
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B \
  --tool-call-parser qwen --port 8000

python testbed/run.py \
  --data pipeline/data/benchmark/lite \
  --model Qwen/Qwen3-8B \
  --base-url http://localhost:8000/v1 \
  --engine sglang --workers 8
```

## How Tool Calling Works

AgentHop requires models to call 7 tools (e.g., `get_paper_info`, `get_references`, `read_section`). There are two modes:

### 1. Server-Side Tool Calling (Recommended)

The inference server translates OpenAI-style `tools` parameters into the model's native format. The harness sends standard API requests — no client-side template handling needed.

**Requirements:** Start the server with `--tool-call-parser <parser>`.

### 2. Text-Based Tool Calling (`--text-tools`)

Tool descriptions are embedded in the system prompt. The model outputs tool calls as text (e.g., `<tool_call>` tags or Python-style calls), and the harness parses them.

**Use when:** The server has no parser for the model, or you want maximum compatibility.

Supported text formats (parsed in priority order):
1. `<tool_call>{"name": ..., "arguments": ...}</tool_call>` — Qwen/Hermes style
2. `` ```tool_call ... ``` `` — Markdown JSON blocks
3. `` ```tool_code ... ``` `` — Gemma-style Python calls
4. `<|channel|>...<|message|>{...}` — OSS special-token format
5. Bare JSON `{"name": ..., "arguments": ...}`
6. Standalone `func(arg="val")` calls
7. Pythonic list `[func(arg="val"), ...]` — Gemma/Llama pythonic

## Model Compatibility Table

| Model Family | sglang Parser | vllm Parser | Chat Template | Notes |
|---|---|---|---|---|
| **Qwen 2.5** | `qwen25` | `hermes` | built-in | Well-supported. Hermes-style `<tool_call>` XML. |
| **Qwen 3** | `qwen` | `hermes` | built-in | Same format. Avoid ReAct stop-words with thinking models. |
| **Qwen 3.5** | `qwen` | `hermes` | built-in | Same as Qwen 3. May use `reasoning_content` field. |
| **Llama 3.1 / 3.3** | `llama3` | `llama3_json` | `tool_chat_template_llama3.1_json.jinja` | Uses `<\|python_tag\|>` + JSON. |
| **Llama 3.2 (1B/3B)** | `pythonic` | `pythonic` | built-in | Small models use pythonic format. |
| **Llama 4** | `llama4` | `llama4_pythonic` | built-in | Pythonic recommended. |
| **Gemma 3** | **N/A** | `pythonic` | `tool_chat_template_gemma3_pythonic.jinja` | No sglang parser. Use vllm, or `--text-tools` with sglang. |
| **Mistral / Mixtral** | `mistral` | `mistral` | built-in | `[TOOL_CALLS]` token. v0.3+ only. |
| **DeepSeek V3** | `deepseekv3` | `deepseek_v3` | `tool_chat_template_deepseekv3.jinja` | Requires chat template. |
| **DeepSeek R1** | `deepseekv3` | `deepseek_v3` | `tool_chat_template_deepseekv3.jinja` | Same parser as V3. |
| **GPT-OSS** | `gpt-oss` | `openai` | built-in | Filters analysis channel events. |
| **Command-R** | N/A | N/A | custom required | No built-in parser. Use `--text-tools`. |

## Server Launch Examples

### sglang

```bash
# Qwen3 (single GPU)
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B \
  --tool-call-parser qwen \
  --port 8000

# Qwen3-32B (multi-GPU)
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-32B \
  --tool-call-parser qwen \
  --tp 4 --port 8000

# Llama 3.3-70B
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-3.3-70B-Instruct \
  --tool-call-parser llama3 \
  --tp 4 --port 8000

# Gemma 3 (no parser — use text-tools)
python3 -m sglang.launch_server \
  --model-path google/gemma-3-27b-it \
  --mem-fraction-static 0.78 \
  --tp 2 --port 8000
# Then run eval with: --text-tools --engine sglang
```

### vllm

```bash
# Qwen3
vllm serve Qwen/Qwen3-8B \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --port 8000

# Gemma 3 (with custom template)
vllm serve google/gemma-3-27b-it \
  --enable-auto-tool-choice \
  --tool-call-parser pythonic \
  --chat-template examples/tool_chat_template_gemma3_pythonic.jinja \
  --port 8000

# Llama 3.3-70B
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --enable-auto-tool-choice \
  --tool-call-parser llama3_json \
  --chat-template examples/tool_chat_template_llama3.1_json.jinja \
  --tensor-parallel-size 4 --port 8000
```

## Known Issues

### Gemma 3 OOM on sglang

Gemma-3-27b-it has a large KV cache (128K context, large head dimensions). The default `--mem-fraction-static 0.88` can cause OOM even with TP=4 on A6000 GPUs. Use `0.78` or lower:

```bash
--mem-fraction-static 0.78
```

The `run_local.sh` script auto-detects this and lowers the fraction.

### Qwen 3 Thinking Models

Qwen 3 thinking/reasoning models may emit stop words (like `Observation:`) inside `<think>` blocks, which can break ReAct-style parsing. Use Hermes-style `<tool_call>` tags (the default) rather than stop-word-based templates.

### tool_choice Behavior

- `tool_choice="auto"` does **not** guarantee schema-conformant arguments in vllm. Arguments may be malformed.
- Use `tool_choice="required"` for guaranteed valid JSON arguments (schema-constrained via structured outputs).
- sglang requires `--grammar-backend xgrammar` (default) for full `tool_choice` support.

## Evaluation Commands

```bash
# Eval with server-side tool calling
python testbed/run.py \
  --data pipeline/data/benchmark/lite \
  --model Qwen/Qwen3-8B \
  --base-url http://localhost:8000/v1 \
  --engine sglang --workers 8

# Eval with text-tools fallback (for Gemma, Command-R, etc.)
python testbed/run.py \
  --data pipeline/data/benchmark/lite \
  --model google/gemma-3-27b-it \
  --base-url http://localhost:8000/v1 \
  --engine sglang --text-tools --workers 8

# Single sample debug
python testbed/run.py \
  --sample pipeline/data/benchmark/lite/bfs_0001.json \
  --model Qwen/Qwen3-8B \
  --base-url http://localhost:8000/v1 \
  --engine sglang
```

## Adding New Models

1. Add an entry in `testbed/model_configs.py` with the model's parser names and any required chat templates.
2. Add a case to `detect_tool_parser()` in `evaluation/run_local.sh`.
3. Test with a single sample first: `--sample <file> --limit 1`.
4. If server-side tool calling doesn't work, fall back to `--text-tools` and check which text format the model uses.
