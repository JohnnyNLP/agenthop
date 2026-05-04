#!/usr/bin/env python3
"""End-to-end AgentHop pipeline runner.

Executes all pipeline stages sequentially with a single command.
Supports dry-run mode, stage selection, and cost tracking.

Usage:
    # Full pipeline (stages 1-5, seeds already combined in data/seeds.json)
    python run_pipeline.py

    # Dry run — show what would execute without running
    python run_pipeline.py --dry-run

    # Run specific stages only
    python run_pipeline.py --stages 3a 3b 4 5

    # Resume from a specific stage
    python run_pipeline.py --from 3a

    # Use batch APIs for 50% cost reduction (recommended)
    python run_pipeline.py --batch

    # Quick test with limited items
    python run_pipeline.py --limit 5
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PIPELINE_DIR = Path(__file__).parent

# ── Stage definitions ────────────────────────────────────────────────────────

STAGES = {
    "1": {
        "name": "Seed generation",
        "script": "stage1_seed_generation.py",
        "description": "Stratified seed selection from Semantic Scholar",
    },
    "2a": {
        "name": "Chain collection (expansion)",
        "script": "stage2a_chain_expansion.py",
        "description": "Two-hop citation neighbourhood + LLM screening",
    },
    "2b": {
        "name": "Chain collection (corpus)",
        "script": "stage2b_corpus_collection.py",
        "description": "Section-structured paper bodies via ar5iv/arxiv",
    },
    "3a": {
        "name": "Q/A generation (single-target)",
        "script": "stage3a_qa_generation_st.py",
        "description": "Single-target QA pairs along citation chains",
    },
    "3b": {
        "name": "Q/A generation (multi-target)",
        "script": "stage3b_qa_generation_mt.py",
        "description": "Multi-target QA pairs requiring two-paper synthesis",
    },
    "4": {
        "name": "Distractor generation",
        "script": "stage4_distractor_generation.py",
        "description": "Three engineered distractors per QA pair",
    },
    "5": {
        "name": "Model-ensemble filtering",
        "script": "stage5_ensemble_filtering.py",
        "description": "Three-model consensus shortcut filter",
    },
    "6": {
        "name": "Data sanity check",
        "script": "stage6_data_sanity_check.py",
        "description": "Structural-integrity pass over gold-path papers",
    },
}

# Stages 7 (seven-auditor human review) and 8 (post-audit triage) live in
# the audit/ directory and are not invoked by this runner.
STAGE_ORDER = ["1", "2a", "2b", "3a", "3b", "4", "5", "6"]


def build_stage_args(stage_id: str, args: argparse.Namespace) -> list[str]:
    """Build command-line arguments for a specific stage."""
    cmd = [sys.executable, str(PIPELINE_DIR / STAGES[stage_id]["script"])]

    # Stage 1 (seed generation) is run manually before the pipeline; the
    # runner picks up at stage 2a with seeds.json already in place.
    if stage_id == "2a":
        cmd.extend(["--batch", str(PIPELINE_DIR / "data" / "seeds.json")])
        cmd.extend(["--provider", args.screen_provider])
        cmd.append("--skip-existing")
        if args.batch:
            cmd.append("--batch-api")
        if args.dry_run:
            cmd.append("--dry-run")
        if args.limit:
            cmd.extend(["--limit", str(args.limit)])

    elif stage_id == "2b":
        cmd.append("--skip-cached")
        if args.dry_run:
            cmd.append("--parse-only")

    elif stage_id in ("3a", "3b"):
        cmd.extend(["--provider", args.gen_provider])
        if args.gen_model:
            cmd.extend(["--model", args.gen_model])
        if args.batch:
            cmd.append("--batch-api")
        if args.dry_run:
            cmd.append("--dry-run")
        if args.limit:
            cmd.extend(["--limit", str(args.limit)])

    elif stage_id == "4":
        # Find QA files emitted by stages 3a and 3b.
        q_dir = PIPELINE_DIR / "data" / "questions"
        input_files = []
        st_file = q_dir / "generated_questions.json"
        mt_file = q_dir / "dfs_questions.json"
        if st_file.exists():
            input_files.append(str(st_file))
        if mt_file.exists():
            input_files.append(str(mt_file))
        if not input_files:
            print(f"  [!] No QA files found in {q_dir}")
            return None
        cmd.extend(input_files)
        cmd.extend(["--provider", args.gen_provider])
        if args.gen_model:
            cmd.extend(["--model", args.gen_model])
        if args.batch:
            cmd.append("--batch-api")
        if args.dry_run:
            cmd.append("--dry-run")
        if args.limit:
            cmd.extend(["--limit", str(args.limit)])

    elif stage_id == "5":
        # Find MCQ file emitted by stage 4.
        q_dir = PIPELINE_DIR / "data" / "questions"
        mcq_file = q_dir / "mcq_combined.json"
        if not mcq_file.exists():
            mcq_files = sorted(q_dir.glob("*_mcq*.json"))
            if mcq_files:
                mcq_file = mcq_files[-1]
            else:
                print(f"  [!] No MCQ files found in {q_dir}")
                return None
        cmd.append(str(mcq_file))
        if args.batch:
            cmd.append("--batch")
        if args.dry_run:
            cmd.append("--dry-run")
        if args.limit:
            cmd.extend(["--limit", str(args.limit)])

    elif stage_id == "6":
        cmd.extend(["--target", "full"])
        if args.dry_run:
            # Sanity check is read-only by default; --remove is the destructive flag.
            pass

    return cmd


def run_stage(stage_id: str, args: argparse.Namespace) -> bool:
    """Run a single pipeline stage. Returns True on success."""
    stage = STAGES[stage_id]
    header = f"Stage {stage_id}: {stage['name']}"

    print(f"\n{'=' * 65}")
    print(f"  {header}")
    print(f"  {stage['description']}")
    print(f"{'=' * 65}")

    cmd = build_stage_args(stage_id, args)
    if cmd is None:
        print(f"  [SKIP] Missing input files for stage {stage_id}")
        return False

    cmd_str = " ".join(cmd)
    if args.dry_run:
        print(f"\n  [DRY RUN] Would execute:")
        print(f"  $ {cmd_str}\n")
        # Still run with --dry-run flag (scripts handle it internally)
        if "--dry-run" not in cmd_str and "--parse-only" not in cmd_str:
            return True

    print(f"  $ {cmd_str}\n")
    start = time.time()

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PIPELINE_DIR),
            check=True,
        )
        elapsed = time.time() - start
        print(f"\n  [OK] Stage {stage_id} completed in {elapsed:.0f}s")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        print(f"\n  [FAIL] Stage {stage_id} failed after {elapsed:.0f}s "
              f"(exit code {e.returncode})")
        return False
    except KeyboardInterrupt:
        print(f"\n  [INTERRUPTED] Stage {stage_id}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="AgentHop end-to-end pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --dry-run              # Preview all commands
  python run_pipeline.py --from 3a              # Resume from BFS generation
  python run_pipeline.py --stages 4 5           # Run MCQ + filter only
  python run_pipeline.py --batch                # Production run (50% off)
        """,
    )

    # Stage selection
    parser.add_argument("--stages", nargs="+", choices=STAGE_ORDER,
                        help="Run only specific stages")
    parser.add_argument("--from", dest="from_stage", choices=STAGE_ORDER,
                        help="Resume from this stage (inclusive)")
    parser.add_argument("--to", dest="to_stage", choices=STAGE_ORDER,
                        help="Stop after this stage (inclusive)")

    # Execution mode
    parser.add_argument("--dry-run", action="store_true",
                        help="Show commands without executing (scripts also "
                             "run in their own dry-run mode where supported)")

    # Scale
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit items per stage (for testing)")

    # Model configuration
    parser.add_argument("--gen-provider", default="openai",
                        choices=["openai", "deepseek", "deepseek-reasoner",
                                 "anthropic", "gemini"],
                        help="Provider for question/MCQ generation "
                             "(default: openai → gpt-5.4)")
    parser.add_argument("--gen-model", default="gpt-5.4",
                        help="Model for generation stages (default: gpt-5.4)")
    parser.add_argument("--screen-provider", default="openai",
                        choices=["openai", "deepseek", "anthropic"],
                        help="Provider for chain screening (default: openai → "
                             "gpt-4.1-mini)")

    # Pipeline options (one-per-seed is now always on by default in 3a/3b)
    parser.add_argument("--batch", action="store_true",
                        help="Use batch APIs for filtering (50%% cheaper)")

    # Control
    parser.add_argument("--stop-on-fail", action="store_true", default=True,
                        help="Stop pipeline on first failure (default: True)")
    parser.add_argument("--no-stop-on-fail", action="store_false",
                        dest="stop_on_fail",
                        help="Continue pipeline even if a stage fails")

    args = parser.parse_args()

    # Determine which stages to run
    if args.stages:
        stages_to_run = args.stages
    else:
        stages_to_run = list(STAGE_ORDER)
        if args.from_stage:
            idx = STAGE_ORDER.index(args.from_stage)
            stages_to_run = stages_to_run[idx:]
        if args.to_stage:
            idx = STAGE_ORDER.index(args.to_stage)
            stages_to_run = [s for s in stages_to_run
                             if STAGE_ORDER.index(s) <= idx]

    # Print plan
    print(f"\n{'━' * 65}")
    print(f"  AgentHop Pipeline Runner")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'━' * 65}")
    print(f"  Stages: {' → '.join(stages_to_run)}")
    print(f"  Mode:   {'DRY RUN' if args.dry_run else 'LIVE'}")
    if args.gen_model:
        print(f"  Gen:    {args.gen_provider}/{args.gen_model}")
    else:
        print(f"  Gen:    {args.gen_provider} (default model)")
    print(f"  Screen: {args.screen_provider}")
    print(f"  One-per-seed: enabled (default)")
    if args.batch:
        print(f"  Batch API: enabled (50% off filtering)")
    if args.limit:
        print(f"  Limit: {args.limit} items per stage")
    print(f"{'━' * 65}")

    # Run stages
    start_time = time.time()
    results = {}

    for stage_id in stages_to_run:
        success = run_stage(stage_id, args)
        results[stage_id] = success
        if not success and args.stop_on_fail:
            print(f"\n  Pipeline stopped at stage {stage_id}.")
            break

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'━' * 65}")
    print(f"  Pipeline Summary ({elapsed:.0f}s total)")
    print(f"{'━' * 65}")
    for stage_id, success in results.items():
        status = "OK" if success else "FAIL"
        name = STAGES[stage_id]["name"]
        print(f"  [{status:4s}] Stage {stage_id}: {name}")
    print(f"{'━' * 65}")

    # Show cost report if any API calls were made
    if not args.dry_run:
        try:
            from cost_tracker import tracker
            tracker.report()
        except Exception:
            pass

    failed = [s for s, ok in results.items() if not ok]
    if failed:
        print(f"\n  {len(failed)} stage(s) failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"\n  All {len(results)} stages completed successfully.")


if __name__ == "__main__":
    main()
