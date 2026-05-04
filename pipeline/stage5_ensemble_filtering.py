"""Stage 5: Model-ensemble filtering.

Three-model ensemble (GPT-4.1, Claude Sonnet 4.6, deepseek-chat) probes
each MCQ for shortcut-answerable items and gates them by consensus. Default
filters validate question quality:
  Filter A — Seed-only: discard if answerable from the seed paper alone
  Filter C — All papers: discard if NOT answerable with full context

Filter B (terminal-only) is disabled by default. AgentHop tests navigation
(finding the right paper via citation chains), not cross-paper synthesis.
The answer living in the terminal paper is by design — the challenge is
finding that paper. Use --no-skip-b to re-enable if needed.

Usage:
    python 5.triple_filter.py data/questions/mcq_combined.json
    python 5.triple_filter.py data/questions/mcq_combined.json --no-skip-b
    python 5.triple_filter.py data/questions/mcq_combined.json --full-analysis
    python 5.triple_filter.py data/questions/mcq_combined.json --batch
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from config import CHAINS_DIR, QUESTIONS_DIR
from llm_client import llm_batch


# ── Constants ────────────────────────────────────────────────────────────────

# No content truncation — full paper content is passed to filters.
# GPT-5.4/4.1 and Claude handle full paper lengths fine.

FILTER_AB_SYSTEM = """\
You are answering a multiple-choice question about academic research. \
You have access to ONE paper. Use ONLY the provided paper to answer. \
If no option fits well, pick the closest one.

You MUST respond with ONLY this JSON and nothing else: {"answer": "X"} \
where X is A, B, C, or D."""

FILTER_C_SYSTEM = """\
You are answering a multiple-choice question about academic research. \
You have access to all papers needed to answer. Read carefully and \
select the correct answer.

You MUST respond with ONLY this JSON and nothing else: {"answer": "X"} \
where X is A, B, C, or D."""


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_paper_excerpt(contents: dict, arxiv_id: str) -> str:
    """Get full paper content (no truncation)."""
    paper = contents.get(arxiv_id, {})
    sections = paper.get("sections", [])
    if not sections:
        return "(paper content not available)"

    parts = []
    for s in sections:
        header = s.get("header", "")
        text = s.get("text", "")
        parts.append(f"[{header}]\n{text}\n")

    return "\n".join(parts)


def _build_seed_arxiv_lookup(chains_dir: str) -> dict:
    """Build {seed_paperId: seed_arxivId} from chain files."""
    chains_path = Path(chains_dir)
    lookup = {}
    for fpath in sorted(chains_path.glob("chains_d*.json")):
        try:
            with open(fpath) as f:
                chains = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if chains and isinstance(chains, list):
            seed = chains[0].get("seed", {})
            pid = seed.get("paperId", "")
            aid = seed.get("arxivId", "")
            if pid and aid:
                lookup[pid] = aid
    return lookup


def _get_seed_arxiv(question: dict, seed_lookup: dict) -> str | None:
    """Get seed paper arxivId for a question."""
    q_type = question.get("question_type", "bfs")
    if q_type == "bfs":
        return question.get("chain", {}).get("seed", {}).get("arxivId")
    else:
        pid = question.get("group", {}).get("seed", {}).get("paperId", "")
        return seed_lookup.get(pid)


def _get_terminal_arxivs(question: dict) -> list[str]:
    """Get terminal/target paper arxivIds."""
    q_type = question.get("question_type", "bfs")
    if q_type == "bfs":
        aid = question.get("chain", {}).get("terminal", {}).get("arxivId", "")
        return [aid] if aid else []
    else:
        return [
            t.get("arxivId", "")
            for t in question.get("group", {}).get("targets", [])
            if t.get("arxivId")
        ]


def _format_question_block(question: dict) -> str:
    """Format question + options as text."""
    parts = [f"Question: {question.get('question', '')}"]
    for i, opt in enumerate(question.get("options", [])):
        parts.append(f"  {chr(65 + i)}) {opt}")
    return "\n".join(parts)


def extract_answer(result: dict) -> str | None:
    """Extract A/B/C/D from LLM result. Returns None if unparseable."""
    # Try JSON "answer" field first
    ans = result.get("answer", "")
    if isinstance(ans, str) and ans.strip().upper() in ("A", "B", "C", "D"):
        return ans.strip().upper()

    # Fallback: scan all string values for single-letter answers
    for v in result.values():
        if isinstance(v, str):
            v_stripped = v.strip().upper()
            if v_stripped in ("A", "B", "C", "D"):
                return v_stripped

    # Fallback: regex scan for answer patterns in any text field
    # Handles prose responses like "The answer is B" or "I would choose A)"
    for v in result.values():
        if isinstance(v, str) and len(v) > 1:
            # Try common patterns: "answer is X", "select X", "option X", "X)"
            m = re.search(
                r'(?:answer\s*(?:is|:)\s*|select\s+|choose\s+|option\s+)'
                r'["\']?\b([A-D])\b["\']?',
                v, re.IGNORECASE,
            )
            if m:
                return m.group(1).upper()
            # Try standalone "X)" or "(X)" at end of text
            m = re.search(r'\b([A-D])\)?\s*$', v.strip(), re.IGNORECASE)
            if m:
                return m.group(1).upper()

    return None


# ── Prompt Builders ──────────────────────────────────────────────────────────

def build_single_paper_prompt(question: dict, paper_content: str) -> str:
    """Filter A/B: question + one paper."""
    return (
        f"=== PAPER ===\n{paper_content}\n\n"
        f"=== QUESTION ===\n{_format_question_block(question)}"
    )


def build_multi_paper_prompt(question: dict,
                             paper_blocks: list[tuple[str, str]]) -> str:
    """Filter C: question + all papers.
    paper_blocks: list of (label, content) tuples."""
    parts = []
    for label, content in paper_blocks:
        parts.append(f"=== {label} ===\n{content}")
    parts.append(f"=== QUESTION ===\n{_format_question_block(question)}")
    return "\n\n".join(parts)


# ── Filter Execution ─────────────────────────────────────────────────────────

def _correct_letter(question: dict) -> str:
    """Get the correct answer letter for a question."""
    return chr(65 + question.get("correct_index", 0))


FILTER_MODELS = {
    "openai": "gpt-4.1",
    "anthropic": "claude-sonnet-4-6",
    "deepseek": "deepseek-chat",
}


_filter_cache_dir: Path | None = None  # set in main()


def _cache_path(label: str) -> Path | None:
    """Get cache file path for a filter+provider combo, e.g. 'A/openai'."""
    if _filter_cache_dir is None:
        return None
    safe = label.replace("/", "_")
    return _filter_cache_dir / f"{safe}.json"


def run_filter(prompts: list[str], system: str, provider: str,
               label: str, use_batch_api: bool = False) -> list[dict]:
    """Run a batch of filter prompts and return raw results.
    Caches results per filter+provider to avoid re-running on crash/retry."""
    if not prompts:
        return []

    # Check cache
    cp = _cache_path(label)
    if cp and cp.exists():
        with open(cp) as f:
            cached = json.load(f)
        if len(cached) == len(prompts):
            print(f"      {label}: loaded {len(cached)} cached results")
            return cached
        print(f"      {label}: cache size mismatch ({len(cached)} vs {len(prompts)}), re-running")

    model = FILTER_MODELS.get(provider)
    mode = "batch-api" if use_batch_api else "realtime"
    print(f"      {label}: {len(prompts)} prompts via {provider}/{model} ({mode})...")
    results = llm_batch(
        prompts, system, provider=provider, model=model,
        max_tokens=1024, temperature=0.0,
        use_batch_api=use_batch_api,
    )

    # Save cache
    if cp:
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(results, ensure_ascii=False))
        print(f"      {label}: cached {len(results)} results → {cp.name}")

    return results


def run_filter_a(questions: list[dict], contents: dict,
                 seed_lookup: dict, providers: list[str],
                 threshold: int,
                 use_batch_api: bool = False) -> tuple[list[dict], list[dict]]:
    """Filter A — Seed-only. Discard if ≥threshold validators answer correctly."""
    print(f"\n  Filter A — Seed-only ({len(questions)} questions)")

    # Build prompts for all questions
    prompts = []
    valid_indices = []  # track questions with available seed content
    for i, q in enumerate(questions):
        seed_aid = _get_seed_arxiv(q, seed_lookup)
        if not seed_aid or not contents.get(seed_aid):
            valid_indices.append(None)
            continue
        content = _get_paper_excerpt(contents, seed_aid)
        prompts.append(build_single_paper_prompt(q, content))
        valid_indices.append(len(prompts) - 1)

    # Run per provider
    provider_results = {}
    for prov in providers:
        results = run_filter(prompts, FILTER_AB_SYSTEM, prov,
                             f"A/{prov}", use_batch_api=use_batch_api)
        provider_results[prov] = results

    # Evaluate
    passed = []
    discarded = []
    for i, q in enumerate(questions):
        correct = _correct_letter(q)
        prompt_idx = valid_indices[i]

        votes = {}
        correct_count = 0
        for prov in providers:
            if prompt_idx is None:
                votes[prov] = None
            else:
                ans = extract_answer(provider_results[prov][prompt_idx])
                votes[prov] = ans
                if ans == correct:
                    correct_count += 1

        filter_passed = correct_count < threshold
        q.setdefault("triple_filter", {})["filter_a"] = {
            "votes": votes,
            "correct_count": correct_count,
            "passed": filter_passed,
        }

        if filter_passed:
            passed.append(q)
        else:
            discarded.append(q)

    print(f"    A result: {len(passed)} passed, {len(discarded)} discarded")
    return passed, discarded


def run_filter_b(questions: list[dict], contents: dict,
                 seed_lookup: dict, providers: list[str],
                 threshold: int,
                 use_batch_api: bool = False) -> tuple[list[dict], list[dict]]:
    """Filter B — Terminal-only. Discard if ≥threshold answer correctly.
    BFS: test terminal paper. DFS: test each target individually."""
    print(f"\n  Filter B — Terminal-only ({len(questions)} questions)")

    # Build per-question test specs: list of (question_idx, paper_arxiv, prompt)
    test_specs = []  # (q_idx, paper_label, prompt)
    q_test_ranges = {}  # q_idx -> (start, count) in test_specs

    for i, q in enumerate(questions):
        q_type = q.get("question_type", "bfs")
        start = len(test_specs)

        if q_type == "bfs":
            terminal_aids = _get_terminal_arxivs(q)
            if terminal_aids and contents.get(terminal_aids[0]):
                content = _get_paper_excerpt(contents, terminal_aids[0])
                prompt = build_single_paper_prompt(q, content)
                test_specs.append((i, "terminal", prompt))
        else:  # DFS — test each target individually
            for t in q.get("group", {}).get("targets", []):
                t_aid = t.get("arxivId", "")
                if t_aid and contents.get(t_aid):
                    content = _get_paper_excerpt(contents, t_aid)
                    prompt = build_single_paper_prompt(q, content)
                    test_specs.append((i, t_aid, prompt))

        q_test_ranges[i] = (start, len(test_specs) - start)

    all_prompts = [spec[2] for spec in test_specs]

    # Run per provider
    provider_results = {}
    for prov in providers:
        results = run_filter(all_prompts, FILTER_AB_SYSTEM, prov,
                             f"B/{prov}", use_batch_api=use_batch_api)
        provider_results[prov] = results

    # Evaluate
    passed = []
    discarded = []
    for i, q in enumerate(questions):
        correct = _correct_letter(q)
        start, count = q_test_ranges[i]

        tests = []
        any_paper_sufficient = False

        for j in range(count):
            spec_idx = start + j
            paper_label = test_specs[spec_idx][1]
            votes = {}
            correct_count = 0
            for prov in providers:
                ans = extract_answer(provider_results[prov][spec_idx])
                votes[prov] = ans
                if ans == correct:
                    correct_count += 1

            tests.append({
                "paper": paper_label,
                "votes": votes,
                "correct_count": correct_count,
            })
            if correct_count >= threshold:
                any_paper_sufficient = True

        # No tests = no terminal content available → pass by default
        filter_passed = not any_paper_sufficient
        q.setdefault("triple_filter", {})["filter_b"] = {
            "tests": tests,
            "passed": filter_passed,
        }

        if filter_passed:
            passed.append(q)
        else:
            discarded.append(q)

    print(f"    B result: {len(passed)} passed, {len(discarded)} discarded")
    return passed, discarded


def run_filter_c(questions: list[dict], contents: dict,
                 seed_lookup: dict, providers: list[str],
                 threshold: int,
                 use_batch_api: bool = False) -> tuple[list[dict], list[dict]]:
    """Filter C — All papers. Keep if ≥threshold answer correctly."""
    print(f"\n  Filter C — All papers ({len(questions)} questions)")

    prompts = []
    valid_indices = []

    for i, q in enumerate(questions):
        q_type = q.get("question_type", "bfs")
        paper_blocks = []

        # Seed paper
        seed_aid = _get_seed_arxiv(q, seed_lookup)
        if seed_aid and contents.get(seed_aid):
            paper_blocks.append(
                ("Paper 1", _get_paper_excerpt(contents, seed_aid)))

        # Terminal/target papers
        terminal_aids = _get_terminal_arxivs(q)
        for j, t_aid in enumerate(terminal_aids):
            if t_aid and contents.get(t_aid):
                label = f"Paper {len(paper_blocks) + 1}"
                paper_blocks.append(
                    (label, _get_paper_excerpt(contents, t_aid)))

        if len(paper_blocks) < 2:
            # Not enough papers → can't validate, keep conservatively
            valid_indices.append(None)
            continue

        prompts.append(build_multi_paper_prompt(q, paper_blocks))
        valid_indices.append(len(prompts) - 1)

    # Run per provider
    provider_results = {}
    for prov in providers:
        results = run_filter(prompts, FILTER_C_SYSTEM, prov,
                             f"C/{prov}", use_batch_api=use_batch_api)
        provider_results[prov] = results

    # Evaluate
    passed = []
    discarded = []
    for i, q in enumerate(questions):
        correct = _correct_letter(q)
        prompt_idx = valid_indices[i]

        votes = {}
        correct_count = 0
        for prov in providers:
            if prompt_idx is None:
                votes[prov] = None
                # Conservative: count missing as correct for keep logic
                correct_count += 1
            else:
                ans = extract_answer(provider_results[prov][prompt_idx])
                votes[prov] = ans
                if ans == correct:
                    correct_count += 1

        filter_passed = correct_count >= threshold
        q.setdefault("triple_filter", {})["filter_c"] = {
            "votes": votes,
            "correct_count": correct_count,
            "passed": filter_passed,
        }

        if filter_passed:
            passed.append(q)
        else:
            discarded.append(q)

    print(f"    C result: {len(passed)} passed, {len(discarded)} discarded")
    return passed, discarded


# ── Full Analysis Mode ───────────────────────────────────────────────────────

def run_full_analysis(questions: list[dict], contents: dict,
                      seed_lookup: dict, providers: list[str],
                      threshold: int, use_batch_api: bool = False):
    """Run all three filters on ALL questions for diagnostic stats."""
    print("\n  === FULL ANALYSIS MODE ===")
    print("  Running all filters on all questions (no sequential reduction)\n")

    all_q = list(questions)
    n = len(all_q)

    # Run each filter independently on the full set
    a_passed, a_disc = run_filter_a(all_q, contents, seed_lookup,
                                     providers, threshold,
                                     use_batch_api=use_batch_api)
    b_passed, b_disc = run_filter_b(all_q, contents, seed_lookup,
                                     providers, threshold,
                                     use_batch_api=use_batch_api)
    c_passed, c_disc = run_filter_c(all_q, contents, seed_lookup,
                                     providers, threshold,
                                     use_batch_api=use_batch_api)

    # Collect per-question results
    a_pass_ids = {q["id"] for q in a_passed}
    b_pass_ids = {q["id"] for q in b_passed}
    c_pass_ids = {q["id"] for q in c_passed}
    all_pass = a_pass_ids & b_pass_ids & c_pass_ids

    # Type breakdown
    bfs_ids = {q["id"] for q in all_q if q.get("question_type") == "bfs"}
    dfs_ids = {q["id"] for q in all_q if q.get("question_type") == "dfs"}

    print(f"\n{'='*70}")
    print(f"  FULL ANALYSIS RESULTS")
    print(f"{'='*70}")
    print(f"  Total questions: {n}")
    print(f"    BFS: {len(bfs_ids)}, DFS: {len(dfs_ids)}")
    print(f"\n  Filter A (seed-only discard):")
    print(f"    Passed: {len(a_pass_ids)} ({len(a_pass_ids)*100/n:.1f}%)")
    print(f"    Discarded: {n - len(a_pass_ids)}")
    print(f"      BFS: {len(a_pass_ids & bfs_ids)}/{len(bfs_ids)} passed")
    print(f"      DFS: {len(a_pass_ids & dfs_ids)}/{len(dfs_ids)} passed")
    print(f"\n  Filter B (terminal-only discard):")
    print(f"    Passed: {len(b_pass_ids)} ({len(b_pass_ids)*100/n:.1f}%)")
    print(f"    Discarded: {n - len(b_pass_ids)}")
    print(f"      BFS: {len(b_pass_ids & bfs_ids)}/{len(bfs_ids)} passed")
    print(f"      DFS: {len(b_pass_ids & dfs_ids)}/{len(dfs_ids)} passed")
    print(f"\n  Filter C (all-papers keep):")
    print(f"    Passed: {len(c_pass_ids)} ({len(c_pass_ids)*100/n:.1f}%)")
    print(f"    Discarded: {n - len(c_pass_ids)}")
    print(f"      BFS: {len(c_pass_ids & bfs_ids)}/{len(bfs_ids)} passed")
    print(f"      DFS: {len(c_pass_ids & dfs_ids)}/{len(dfs_ids)} passed")
    print(f"\n  All three filters passed:")
    print(f"    {len(all_pass)} ({len(all_pass)*100/n:.1f}%)")
    print(f"      BFS: {len(all_pass & bfs_ids)}/{len(bfs_ids)}")
    print(f"      DFS: {len(all_pass & dfs_ids)}/{len(dfs_ids)}")

    # Vote distribution analysis
    for fname, fkey in [("A", "filter_a"), ("B", "filter_b"),
                         ("C", "filter_c")]:
        if fkey == "filter_b":
            continue  # B has different structure (tests list)
        counts = Counter()
        for q in all_q:
            tf = q.get("triple_filter", {}).get(fkey, {})
            counts[tf.get("correct_count", 0)] += 1
        print(f"\n  Filter {fname} vote distribution:")
        for k in sorted(counts):
            print(f"    {k} correct: {counts[k]}")

    return all_q


# ── Consensus Tier Labeling ──────────────────────────────────────────────────

def label_consensus_tiers(questions: list[dict]) -> None:
    """Label each question with a consensus tier based on filter votes.

    Tiers (for validated/passed questions):
      gold   — Filter A 0/3 correct AND Filter C 3/3 correct
      silver — Filter A 1/3 correct AND Filter C 3/3 correct
      bronze — Filter C 2/3 correct (regardless of A)

    For discarded questions, tier is set to 'rejected'.
    Mutates questions in place (adds 'consensus_tier' field).
    """
    for q in questions:
        tf = q.get("triple_filter", {})
        fa = tf.get("filter_a", {})
        fc = tf.get("filter_c", {})

        # Discarded = any filter failed
        if not fa.get("passed", True) or not fc.get("passed", True):
            q["consensus_tier"] = "rejected"
            continue

        a_correct = fa.get("correct_count", 0)
        c_correct = fc.get("correct_count", 0)

        if c_correct == 3 and a_correct == 0:
            q["consensus_tier"] = "gold"
        elif c_correct == 3:
            q["consensus_tier"] = "silver"
        else:
            q["consensus_tier"] = "bronze"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Quality assurance via triple filtering for MCQs"
    )
    parser.add_argument("input", help="Input question JSON file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run filters but don't write output file")
    parser.add_argument("--full-analysis", action="store_true",
                        help="Run all filters on all questions for diagnostics")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N questions")
    parser.add_argument("--skip-a", action="store_true",
                        help="Skip Filter A (seed-only)")
    parser.add_argument("--skip-b", action="store_true", default=True,
                        help="Skip Filter B (terminal-only) — default: True. "
                             "B tests single-paper answerability, but AgentHop "
                             "tests navigation (finding the paper), not synthesis.")
    parser.add_argument("--no-skip-b", action="store_false", dest="skip_b",
                        help="Re-enable Filter B (terminal-only)")
    parser.add_argument("--skip-c", action="store_true",
                        help="Skip Filter C (all-papers)")
    parser.add_argument("--providers", nargs="+",
                        default=["openai", "anthropic", "deepseek"],
                        help="LLM providers for validation (default: "
                             "gpt-4.1, claude-sonnet-4-6, deepseek-chat)")
    parser.add_argument("--threshold", type=int, default=2,
                        help="Min correct votes for discard (A/B) / keep (C)")
    parser.add_argument("--contents", type=str, default=None,
                        help="Path to paper_contents.json")
    parser.add_argument("--chains-dir", type=str, default=None,
                        help="Path to chains directory")
    parser.add_argument("--suffix", type=str, default="_validated",
                        help="Suffix for output filename")
    parser.add_argument("--batch", action="store_true",
                        help="Use batch APIs for 50%% cost reduction "
                             "(OpenAI/Anthropic async, may take minutes-hours)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable per-provider result caching")
    args = parser.parse_args()

    # ── Set up filter cache ──────────────────────────────────────────────
    global _filter_cache_dir
    if args.no_cache:
        _filter_cache_dir = None
    else:
        _filter_cache_dir = Path(QUESTIONS_DIR) / ".filter_cache"
        _filter_cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Filter cache: {_filter_cache_dir}")

    # ── Load input ───────────────────────────────────────────────────────
    fpath = Path(args.input)
    if not fpath.exists():
        fpath = Path(QUESTIONS_DIR) / args.input
    if not fpath.exists():
        print(f"File not found: {args.input}")
        return

    with open(fpath) as f:
        questions = json.load(f)

    if args.limit:
        questions = questions[:args.limit]

    bfs_count = sum(1 for q in questions if q.get("question_type") == "bfs")
    dfs_count = sum(1 for q in questions if q.get("question_type") == "dfs")
    print(f"Loaded {len(questions)} questions ({bfs_count} BFS, {dfs_count} DFS)")

    # ── Load paper contents ──────────────────────────────────────────────
    contents_path = Path(args.contents) if args.contents else (
        Path(CHAINS_DIR) / "content" / "paper_contents.json"
    )
    contents = {}
    if contents_path.exists():
        print(f"Loading paper contents from {contents_path.name}...")
        with open(contents_path) as f:
            contents = json.load(f)
        print(f"  {len(contents)} papers loaded")
    else:
        print(f"Warning: paper contents not found at {contents_path}")
        return

    # ── Build seed arxiv lookup for DFS ──────────────────────────────────
    chains_dir = args.chains_dir or CHAINS_DIR
    seed_lookup = _build_seed_arxiv_lookup(chains_dir)
    print(f"  Seed arxiv lookup: {len(seed_lookup)} entries")

    # Check coverage
    dfs_qs = [q for q in questions if q.get("question_type") == "dfs"]
    if dfs_qs:
        covered = sum(1 for q in dfs_qs
                      if _get_seed_arxiv(q, seed_lookup) is not None)
        print(f"  DFS seed coverage: {covered}/{len(dfs_qs)}")

    # ── Run filters ──────────────────────────────────────────────────────
    print(f"\nProviders: {', '.join(args.providers)}")
    print(f"Threshold: {args.threshold}")
    if args.batch:
        print(f"Batch API: enabled (50% cost reduction)")

    if args.full_analysis:
        questions = run_full_analysis(questions, contents, seed_lookup,
                                      args.providers, args.threshold,
                                      use_batch_api=args.batch)
        if not args.dry_run:
            out_path = fpath.with_stem(fpath.stem + "_analysis")
            out_path.write_text(
                json.dumps(questions, indent=2, ensure_ascii=False))
            print(f"\n  Saved analysis: {out_path} ({len(questions)} questions)")
        return

    # Sequential filtering: A → B → C
    current = list(questions)
    all_discarded = []

    if not args.skip_a:
        current, disc = run_filter_a(current, contents, seed_lookup,
                                      args.providers, args.threshold,
                                      use_batch_api=args.batch)
        all_discarded.extend(disc)

    if not args.skip_b:
        current, disc = run_filter_b(current, contents, seed_lookup,
                                      args.providers, args.threshold,
                                      use_batch_api=args.batch)
        all_discarded.extend(disc)

    if not args.skip_c:
        current, disc = run_filter_c(current, contents, seed_lookup,
                                      args.providers, args.threshold,
                                      use_batch_api=args.batch)
        all_discarded.extend(disc)

    # ── Summary ──────────────────────────────────────────────────────────
    bfs_surv = sum(1 for q in current if q.get("question_type") == "bfs")
    dfs_surv = sum(1 for q in current if q.get("question_type") == "dfs")

    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Input:      {len(questions)} ({bfs_count} BFS, {dfs_count} DFS)")
    print(f"  Survivors:  {len(current)} ({bfs_surv} BFS, {dfs_surv} DFS)")
    print(f"  Discarded:  {len(all_discarded)}")

    filters_run = []
    if not args.skip_a:
        filters_run.append("A")
    if not args.skip_b:
        filters_run.append("B")
    if not args.skip_c:
        filters_run.append("C")
    print(f"  Filters:    {', '.join(filters_run)}")
    print(f"{'='*70}")

    if args.dry_run:
        print("  (dry run — no output file written)")
        return

    # Label consensus tiers
    label_consensus_tiers(current)
    label_consensus_tiers(all_discarded)

    tier_counts = Counter(q.get("consensus_tier") for q in current)
    print(f"  Tiers:      gold={tier_counts.get('gold',0)}, "
          f"silver={tier_counts.get('silver',0)}, "
          f"bronze={tier_counts.get('bronze',0)}")

    # Save survivors
    out_path = fpath.with_stem(fpath.stem + args.suffix)
    out_path.write_text(json.dumps(current, indent=2, ensure_ascii=False))
    print(f"  Saved: {out_path} ({len(current)} questions)")

    # Save discarded questions with filter reasons
    if all_discarded:
        disc_path = fpath.with_stem(fpath.stem + "_discarded")
        disc_path.write_text(json.dumps(all_discarded, indent=2, ensure_ascii=False))
        print(f"  Saved discarded: {disc_path} ({len(all_discarded)} questions)")

    # Clean up cache after successful completion
    if _filter_cache_dir and _filter_cache_dir.exists():
        import shutil
        shutil.rmtree(_filter_cache_dir)
        print(f"  Cleared filter cache")


if __name__ == "__main__":
    main()
