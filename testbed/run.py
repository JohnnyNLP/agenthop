"""AgentHop evaluation CLI.

Usage:
    # Run on the HuggingFace release directory
    #   huggingface-cli download nlpai-lab/agenthop --repo-type dataset \
    #       --local-dir ../AgentHop_release
    python run.py --data ../AgentHop_release --model gpt-4.1 --workers 8
    python run.py --data ../AgentHop_release --model claude-opus-4-6 --workers 4

    # Single sample debug
    python run.py --sample ../AgentHop_release/qa/full.jsonl --model gpt-4.1

    # Local model via sglang/vllm
    python run.py --data ../AgentHop_release --model Qwen/Qwen3-32B \
        --base-url http://localhost:8000/v1 --workers 16

    # Prompting strategy variants
    python run.py --data ../AgentHop_release --model gpt-4.1 --strategy react
    python run.py --data ../AgentHop_release --model gpt-4.1 --strategy noretrieval

    # Results saved to ../experiments/{model_name}/

API keys are read from environment variables (OPENAI_API_KEY,
ANTHROPIC_API_KEY, GOOGLE_API_KEY, DEEPSEEK_API_KEY, TOGETHER_API_KEY,
XAI_API_KEY). If a `.env` file is present in the repo root and
`python-dotenv` is installed, it will be auto-loaded.
"""

import argparse
import asyncio
import json
import glob
import os
import sys
from pathlib import Path

from harness import HarnessConfig, STRATEGY_TEMPLATES, run_benchmark, run_sample
from tools import TOOL_COSTS
from model_configs import (
    get_model_config,
    check_tool_support,
    suggest_server_command,
    get_hyperparams,
)

# Optional .env loading (no-op if dotenv or .env file is absent)
_ENV = Path(__file__).resolve().parent.parent / ".env"
if _ENV.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_ENV)
    except ImportError:
        pass

# data_loader.py lives at the repo root, one level above testbed/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_samples(data_path: str, limit: int | None = None) -> list[dict]:
    """Load benchmark samples.

    Accepts either:
      1. The HuggingFace release directory (qa/, graphs/, paper_pool/, audit/)
      2. A single sample JSON file (for ad-hoc debugging)
    """
    p = Path(data_path)

    if (p / "qa").exists():
        from data_loader import load_agenthop
        print(f"Loading AgentHop from {p} ...")
        return load_agenthop(p, limit=limit)

    if p.is_file():
        with open(p) as f:
            data = json.load(f)
        samples = [data] if isinstance(data, dict) else data
        return samples[:limit] if limit else samples

    raise FileNotFoundError(
        f"--data {data_path} is neither an AgentHop release directory "
        f"(missing qa/full.jsonl) nor a single sample JSON file."
    )


async def main_async():
    parser = argparse.ArgumentParser(description="AgentHop Evaluation Runner")
    parser.add_argument("--data", type=str, help="Path to the AgentHop release directory (downloaded from HuggingFace)")
    parser.add_argument("--sample", type=str, help="Single sample JSON (overrides --data)")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--backend", type=str, default="openai", choices=["openai", "anthropic", "responses", "google"])
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature. Default: model-specific preset (see model_configs.py); fallback 0.7.")
    parser.add_argument("--top-p", type=float, default=None,
                        help="Nucleus sampling. Default: model-specific preset; fallback None (API default).")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Top-k sampling. Default: model-specific preset; fallback None.")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Per-turn generation cap. Default: model-specific preset; fallback 4096.")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent.parent / "experiments"))
    parser.add_argument("--workers", type=int, default=1, help="Concurrent workers (default: 1)")
    parser.add_argument("--limit", type=int, default=None, help="Max samples (for piloting)")
    parser.add_argument("--question-type", type=str, default=None,
                        choices=["single-target", "multi-target"],
                        help="Filter by question_type. Combine with --depth for slice-runs (e.g. mt_depth2).")
    parser.add_argument("--depth", type=int, default=None, choices=[1, 2],
                        help="Filter by depth.")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--max-total-tokens", type=int, default=200_000,
                        help="Internal safety cap on cumulative tokens per sample (default: 200k)")
    parser.add_argument("--max-run-tokens", type=int, default=0,
                        help="Run-wide token cap (0 = off). Aborts queue when exceeded — last defense for budget.")
    parser.add_argument("--strategy", type=str, default="direct",
                        choices=list(STRATEGY_TEMPLATES.keys()),
                        help="Prompting strategy (default: direct)")
    parser.add_argument("--text-tools", action="store_true",
                        help="Text-based tool calling (no API tools param — universal LLM compatibility)")
    parser.add_argument("--disable-tools", type=str, default="",
                        help="Comma-separated tool names to disable (e.g. 'search_papers'). "
                             "Use 'all' to disable everything except submit_answer "
                             "(zero-tool closed-book baseline). Used for ablation studies.")
    parser.add_argument("--reasoning-effort", type=str, default=None,
                        help="Reasoning effort: 'low'/'medium'/'high' (OpenAI) or token budget e.g. '8192' (Anthropic)")
    parser.add_argument("--engine", type=str, default=None, choices=["sglang", "vllm"],
                        help="Local inference engine (sglang or vllm). Enables model-specific guidance.")
    args = parser.parse_args()

    data_path = args.sample or args.data
    if not data_path:
        parser.error("Must specify --data or --sample")

    # ── Model config guidance for local models ──────────────────────────────
    if args.base_url and not args.text_tools:
        engine = args.engine or "sglang"  # default assumption
        model_cfg = get_model_config(args.model)
        if model_cfg:
            support = check_tool_support(args.model, engine)
            print(f"[Model Config] {model_cfg.display_name} ({model_cfg.family})")
            if support["method"] == "server" and support["parser"]:
                print(f"[Model Config] Expecting server-side tool calling via --tool-call-parser {support['parser']}")
                print(f"[Model Config] Make sure your {engine} server was started with this flag.")
                if model_cfg.notes:
                    print(f"[Model Config] Note: {model_cfg.notes}")
            elif support["method"] == "text":
                print(f"[Model Config] No server-side parser available. Auto-enabling --text-tools.")
                args.text_tools = True
            print()
        else:
            print(f"[Model Config] Model '{args.model}' not in registry. Using default tool-calling mode.")
            print(f"[Model Config] If tool calls fail, try --text-tools for universal compatibility.")
            print()

    samples = load_samples(data_path, limit=None)
    # Apply filters in a specific order: type/depth before limit so --limit N
    # means "N samples of the filtered slice", not "N from the full set".
    if args.question_type is not None:
        samples = [s for s in samples if s.get("question_type") == args.question_type]
    if args.depth is not None:
        samples = [s for s in samples if s.get("depth") == args.depth]
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        print("No valid samples found after filtering.", file=sys.stderr)
        sys.exit(1)
    filter_desc = []
    if args.question_type: filter_desc.append(f"type={args.question_type}")
    if args.depth is not None: filter_desc.append(f"depth={args.depth}")
    if filter_desc:
        print(f"[Filter] {len(samples)} samples after filters: {', '.join(filter_desc)}")

    # ── Apply model-specific hyperparameter preset ─────────────────────────
    # CLI-supplied values (non-None) take precedence; otherwise use preset.
    # If no preset matches, fall back to hard-coded defaults.
    preset = get_hyperparams(args.model)
    if preset is not None:
        if preset.unsupported_reason:
            print(f"[Hyperparams] ERROR: {args.model} is unsupported for this benchmark.")
            print(f"              Reason: {preset.unsupported_reason}")
            sys.exit(2)
        print(f"[Hyperparams] Preset found for {args.model}: "
              f"T={preset.temperature} top_p={preset.top_p} top_k={preset.top_k} "
              f"reasoning_effort={preset.reasoning_effort} max_tokens={preset.max_tokens}")
        if preset.notes:
            print(f"              Note: {preset.notes}")
        # Preset's None means "don't set this param" (API will use its own default).
        # CLI value (if any) overrides preset. Fallbacks kick in only when BOTH are None.
        temperature   = args.temperature       if args.temperature       is not None else preset.temperature
        top_p         = args.top_p             if args.top_p             is not None else preset.top_p
        top_k         = args.top_k             if args.top_k             is not None else preset.top_k
        max_tokens    = args.max_tokens        if args.max_tokens        is not None else preset.max_tokens
        reasoning_effort = args.reasoning_effort if args.reasoning_effort is not None else preset.reasoning_effort
    else:
        print(f"[Hyperparams] No preset for {args.model}; using harness defaults.")
        temperature = args.temperature
        top_p = args.top_p
        top_k = args.top_k
        max_tokens = args.max_tokens
        reasoning_effort = args.reasoning_effort
    # Final safety fallback for max_tokens (harness needs a value).
    # Temperature is left as None when preset says so; the backends skip
    # it for reasoning models that don't accept it.
    if max_tokens is None:
        max_tokens = 16384

    if (args.disable_tools or "").strip().lower() == "all":
        from tools import TOOL_COSTS
        disabled_tools = tuple(t for t in TOOL_COSTS if t != "submit_answer")
    else:
        disabled_tools = tuple(
            t.strip() for t in (args.disable_tools or "").split(",") if t.strip()
        )
    config = HarnessConfig(
        budget=args.budget,
        max_turns=args.max_turns,
        model=args.model,
        backend=args.backend,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        base_url=args.base_url,
        api_key=args.api_key,
        workers=args.workers,
        max_total_tokens=args.max_total_tokens,
        max_run_tokens=args.max_run_tokens,
        strategy=args.strategy,
        text_tools=args.text_tools,
        reasoning_effort=reasoning_effort,
        disabled_tools=disabled_tools,
    )
    if disabled_tools:
        print(f"[Ablation] Disabled tools: {list(disabled_tools)}")

    print(f"Loaded {len(samples)} samples")
    print(f"Model: {args.model} ({args.backend}) | Strategy: {args.strategy}")
    print(f"Budget: {args.budget} | Max turns: {args.max_turns} | Workers: {args.workers} | Token cap: {args.max_total_tokens:,}")
    print()

    if args.sample and len(samples) == 1:
        # Single sample — detailed output
        traj = await run_sample(samples[0], config)
        print(f"\nSample: {traj.sample_id}")
        print(f"Answer: {traj.predicted_answer} ({'correct' if traj.is_correct else 'wrong'})")
        print(f"Expected: {chr(65 + traj.correct_index)}")
        print(f"Reasoning: {traj.reasoning}")
        print(f"Budget: {traj.budget_used}/{traj.budget_total}")
        print(f"Turns: {traj.total_turns} | Terminated: {traj.terminated_by}")
        print(f"Time: {traj.wall_time_s:.1f}s")
        print(f"Tokens: {traj.total_prompt_tokens} prompt + {traj.total_completion_tokens} completion")
        print(f"\n--- Tool call trace ---")
        for turn in traj.turns:
            for tc in turn.tool_calls:
                cost = TOOL_COSTS.get(tc["name"], 0)
                args_str = json.dumps(tc["args"], ensure_ascii=False)
                if len(args_str) > 80:
                    args_str = args_str[:80] + "..."
                print(f"  T{turn.turn}: {tc['name']}({args_str}) [cost={cost}]")
    else:
        # Batch mode
        agenthop_dir = None
        if args.data:
            p = Path(args.data)
            if (p / "qa").exists():
                agenthop_dir = str(p)
        await run_benchmark(
            samples, config,
            output_dir=args.output_dir,
            resume=not args.no_resume,
            agenthop_dir=agenthop_dir,
        )


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
