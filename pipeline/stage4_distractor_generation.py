"""Stage 4: Distractor generation.

Turns each QA pair (from stages 4a/4b) into a four-option MCQ by attaching
three engineered distractors. Distractors are not paraphrases of the wrong
answer — each is a genuine answer the model would produce from a different
(wrong or missing) source-material configuration:

  1. seed_only    — model answers from the seed paper (agent didn't retrieve)
  2. wrong_paper  — model answers from a sibling paper (followed wrong citation)
  3. no_context   — model answers from parametric knowledge (no retrieval at all)

The model never sees the words "distractor", "MCQ", or "wrong answer" — it
genuinely tries to answer each time. Answers are wrong because the *source*
is wrong/absent, not because we told the model to be wrong.

Pipeline order: 3a/3b → 4.generate_mcq → 5.sanitize → validation

Usage:
    python 4.generate_mcq.py data/questions/bfs_questions.json data/questions/dfs_questions.json
    python 4.generate_mcq.py data/questions/*.json --dry-run --limit 5
    python 4.generate_mcq.py data/questions/*.json --provider anthropic -o mcq_v2.json
"""

import argparse
import importlib.util as _ilu
import json
import random
from pathlib import Path

from config import CHAINS_DIR, QUESTIONS_DIR
from llm_client import llm_batch

# ── Import formatting helpers from 3a ────────────────────────────────────────

_spec = _ilu.spec_from_file_location(
    "gen_bfs_questions", Path(__file__).parent / "3a.generate_bfs_questions.py"
)
_bfs = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_bfs)
format_paper_content = _bfs.format_paper_content
format_contexts = _bfs.format_contexts


# ── Constants ────────────────────────────────────────────────────────────────

OVERLAP_THRESHOLD = 0.6  # Jaccard word overlap — drop distractor if exceeded


# ── System prompt ────────────────────────────────────────────────────────────
# Mirrors 3a/3b answer rules verbatim so correct and distractor answers share
# the same tone, length, and specificity. The only difference from step 3:
# instead of "output null", we say "use the most relevant information" — this
# prevents the model from declining when the paper doesn't directly answer.

SYSTEM_PROMPT = """\
You answer research questions using the provided paper content.

=== ANSWER RULES ===

The answer must be 1-2 sentences describing a specific finding, design choice, \
or result from the paper. It must be drawn directly from what the paper \
states — do not speculate, interpret, or add reasoning beyond what the paper \
says. Do not give a bare number; describe the finding.

If the provided content does not directly address the question, use the most \
relevant information available to construct a specific, affirmative answer. \
NEVER say "the paper does not", "does not discuss", "does not specify", \
"does not mention", "not directly addressed", or any similar qualification. \
Always answer with what IS in the content, never with what is missing.

=== ANSWER_CONTEXT RULES ===

The answer_context field must be EXACT TEXT copied from the paper content \
above. Do not add any framing, attribution, or modification:
- WRONG: "From Table 3: the model achieves 94.1%..."
- WRONG: "The authors report that..."
- RIGHT: "We use T = 1000 for all experiments. We set the forward process \
variances to constants increasing linearly from β1 = 10^-4 to βT = 0.02."

If the answer comes from a table, copy the relevant row(s) exactly as they \
appear in the table content. Do not paraphrase or summarize table data.

Output as JSON:
{"answer": "your 1-2 sentence answer", "answer_context": "exact text from source"}"""

NO_CONTEXT_SYSTEM_PROMPT = """\
You answer research questions based on your knowledge of the literature.

=== ANSWER RULES ===

The answer must be 1-2 sentences describing a specific finding, design choice, \
or result. Be concrete and specific — include numbers, method names, and \
dataset names where possible. Do not give a bare number; describe the finding.

NEVER say "I don't know", "I'm not sure", or any similar qualification. \
Always give a specific, confident answer.

Output as JSON:
{"answer": "your 1-2 sentence answer"}"""


# ── Sibling index ────────────────────────────────────────────────────────────

def build_seed_sibling_index(chains_dir: str) -> dict:
    """One-time scan of chain files → {seed_paperId: [{arxivId, title, paperId}]}.

    Maps each seed paper to all terminal papers across its chains,
    providing the pool of sibling papers for wrong_paper distractors.
    """
    chains_path = Path(chains_dir)
    index = {}

    for fpath in sorted(chains_path.glob("chains_d*.json")):
        try:
            with open(fpath) as f:
                chains = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        if not chains or not isinstance(chains, list):
            continue

        seed_id = chains[0].get("seed", {}).get("paperId", "")
        if not seed_id:
            continue

        if seed_id not in index:
            index[seed_id] = []

        seen = {t["paperId"] for t in index[seed_id]}
        for chain in chains:
            terminal = chain.get("terminal", {})
            pid = terminal.get("paperId", "")
            if pid and pid not in seen:
                seen.add(pid)
                index[seed_id].append({
                    "paperId": pid,
                    "arxivId": terminal.get("arxivId", ""),
                    "title": terminal.get("title", ""),
                })

    return index


# ── Content formatting helpers ───────────────────────────────────────────────

def _get_paper_content(contents: dict, arxiv_id: str) -> str:
    """Format a paper's full content using the 3a priority ordering."""
    if not arxiv_id:
        return ""
    paper = contents.get(arxiv_id, {})
    sections = paper.get("sections", [])
    tables = paper.get("tables", [])
    if not sections and not tables:
        return ""

    return format_paper_content(sections, tables)



# ── Context gathering ────────────────────────────────────────────────────────

def gather_distractor_contexts(question: dict, contents: dict,
                               sibling_index: dict,
                               seed_arxiv_lookup: dict | None = None) -> dict:
    """Return {seed_context, wrong_paper_context, available_types,
    sibling_arxivs}."""
    q_type = question.get("question_type", "bfs")

    if q_type == "bfs":
        return _gather_bfs_context(question, contents, sibling_index)
    else:
        return _gather_dfs_context(question, contents, sibling_index,
                                   seed_arxiv_lookup or {})


def _paper_meta(paper_dict: dict) -> dict:
    """Extract title/year/abstract/venue from a chain paper dict."""
    return {
        "title": paper_dict.get("title", ""),
        "year": paper_dict.get("year", ""),
        "abstract": paper_dict.get("abstract", ""),
        "venue": paper_dict.get("venue", ""),
    }


def _gather_bfs_context(question: dict, contents: dict,
                        sibling_index: dict) -> dict:
    """Context gathering for BFS (depth-navigation) questions."""
    chain = question.get("chain", {})
    seed = chain.get("seed", {})
    terminal = chain.get("terminal", {})
    seed_arxiv = seed.get("arxivId", "")
    seed_id = seed.get("paperId", "")
    terminal_arxiv = terminal.get("arxivId", "")
    answer_context = question.get("answer_context", "")

    # seed_only: full seed paper content
    seed_ctx = _get_paper_content(contents, seed_arxiv)
    seed_meta = _paper_meta(seed)

    # wrong_paper: one sibling paper
    siblings = sibling_index.get(seed_id, [])
    sibling_papers = [
        s for s in siblings
        if s.get("arxivId") and s["arxivId"] != terminal_arxiv
    ]
    wrong_paper_ctx = ""
    wrong_paper_meta = {}
    sibling_arxivs = []
    for sib in sibling_papers[:3]:
        arxiv_id = sib.get("arxivId", "")
        ctx = _get_paper_content(contents, arxiv_id)
        if ctx:
            wrong_paper_ctx = ctx
            wrong_paper_meta = {"title": sib.get("title", ""), "arxivId": arxiv_id}
            sibling_arxivs = [arxiv_id]
            break

    available = []
    if seed_ctx:
        available.append("seed_only")
    if wrong_paper_ctx:
        available.append("wrong_paper")
    # no_context is always available — no paper needed
    available.append("no_context")

    return {
        "seed_context": seed_ctx,
        "seed_meta": seed_meta,
        "wrong_paper_context": wrong_paper_ctx,
        "wrong_paper_meta": wrong_paper_meta,
        "available_types": available,
        "sibling_arxivs": sibling_arxivs,
    }


def _gather_dfs_context(question: dict, contents: dict,
                        sibling_index: dict,
                        seed_arxiv_lookup: dict = None) -> dict:
    """Context gathering for DFS (breadth-synthesis) questions."""
    group = question.get("group", {})
    seed = group.get("seed", {})
    seed_id = seed.get("paperId", "")
    seed_arxiv = (seed_arxiv_lookup or {}).get(seed_id, "")

    targets = group.get("targets", [])
    target_arxivs = {t.get("arxivId", "") for t in targets}
    target_arxivs.discard("")

    answer_sources = question.get("answer_sources", {})
    exclude_facts = []
    for src in answer_sources.values():
        fact = src.get("fact", "")
        if fact:
            exclude_facts.append(fact)

    # seed_only: full seed paper content
    seed_ctx = _get_paper_content(contents, seed_arxiv)
    seed_meta = _paper_meta(seed)

    # wrong_paper: one correct target + one wrong sibling
    siblings = sibling_index.get(seed_id, [])
    sibling_papers = [
        s for s in siblings
        if s.get("arxivId") and s["arxivId"] not in target_arxivs
    ]
    wrong_paper_ctx = ""
    wrong_paper_meta = {}
    sibling_arxivs = []
    for sib in sibling_papers[:3]:
        arxiv_id = sib.get("arxivId", "")
        ctx = _get_paper_content(contents, arxiv_id)
        if ctx:
            # For DFS: prepend one kept fact from correct paper
            if answer_sources:
                first_src = list(answer_sources.values())[0]
                kept_fact = first_src.get("fact", "")
                if kept_fact:
                    wrong_paper_ctx = (
                        f"=== FACT FROM PAPER 1 ===\n{kept_fact}\n\n"
                        f"=== PAPER 2 CONTENT ===\n{ctx}"
                    )
                else:
                    wrong_paper_ctx = ctx
            else:
                wrong_paper_ctx = ctx
            wrong_paper_meta = {"title": sib.get("title", ""), "arxivId": arxiv_id}
            sibling_arxivs = [arxiv_id]
            break

    available = []
    if seed_ctx:
        available.append("seed_only")
    if wrong_paper_ctx:
        available.append("wrong_paper")
    # no_context is always available — no paper needed
    available.append("no_context")

    return {
        "seed_context": seed_ctx,
        "seed_meta": seed_meta,
        "wrong_paper_context": wrong_paper_ctx,
        "wrong_paper_meta": wrong_paper_meta,
        "available_types": available,
        "sibling_arxivs": sibling_arxivs,
    }


# ── Prompt builders ──────────────────────────────────────────────────────────
# All prompts mirror step 3's format: paper metadata → paper content → question.
# The model sees the same kind of prompt it saw when generating the correct
# answer — it just happens to be looking at the wrong paper/section.

def _format_paper_block(meta: dict, content: str) -> str:
    """Format a paper block with metadata, matching step 3's presentation."""
    title = meta.get("title", "")
    year = meta.get("year", "")
    abstract = meta.get("abstract", "")
    venue = meta.get("venue", "")

    header_parts = []
    if title:
        header_parts.append(f"Title: {title}")
    if year:
        header_parts.append(f"Year: {year}")
    if venue:
        header_parts.append(f"Venue: {venue}")
    if abstract:
        header_parts.append(f"Abstract: {abstract}")

    header = "\n".join(header_parts)
    if header:
        return f"=== PAPER ===\n{header}\n\n=== PAPER CONTENT ===\n{content}"
    return f"=== PAPER CONTENT ===\n{content}"


def build_seed_only_prompt(question: dict, ctx: dict) -> str:
    """Prompt using seed paper content (agent didn't retrieve at all)."""
    q_text = question.get("question", "")
    paper_block = _format_paper_block(ctx.get("seed_meta", {}),
                                      ctx["seed_context"])
    return (
        f"{paper_block}\n\n"
        f"=== QUESTION ===\n{q_text}\n\n"
        "Answer this question using ONLY the paper content above."
    )


def build_wrong_paper_prompt(question: dict, ctx: dict) -> str:
    """Prompt using sibling paper content (agent followed wrong citation)."""
    q_text = question.get("question", "")
    paper_block = _format_paper_block(ctx.get("wrong_paper_meta", {}),
                                      ctx["wrong_paper_context"])
    return (
        f"{paper_block}\n\n"
        f"=== QUESTION ===\n{q_text}\n\n"
        "Answer this question using ONLY the paper content above."
    )


def build_no_context_prompt(question: dict, ctx: dict) -> str:
    """Prompt with no paper — model answers from parametric knowledge only."""
    q_text = question.get("question", "")
    return (
        f"=== QUESTION ===\n{q_text}\n\n"
        "Answer this question based on your knowledge of the research literature."
    )


PROMPT_BUILDERS = {
    "seed_only": build_seed_only_prompt,
    "wrong_paper": build_wrong_paper_prompt,
    "no_context": build_no_context_prompt,
}


# ── Overlap filtering ───────────────────────────────────────────────────────

def word_overlap(a: str, b: str) -> float:
    """Jaccard word overlap between two strings."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    union = wa | wb
    return len(wa & wb) / len(union) if union else 0


def numeric_overlap(a: str, b: str) -> float:
    """Fraction of non-trivial numbers in `a` that also appear in `b`."""
    import re
    na = set(re.findall(r'\d+\.?\d*', a)) - {"0", "1", "2", "3", "4", "5"}
    nb = set(re.findall(r'\d+\.?\d*', b)) - {"0", "1", "2", "3", "4", "5"}
    if len(na) < 2 or len(nb) < 2:
        return 0.0
    return len(na & nb) / len(na)


# ── Main ─────────────────────────────────────────────────────────────────────

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


def main():
    parser = argparse.ArgumentParser(
        description="Generate 4-option MCQs from QA pairs"
    )
    parser.add_argument("inputs", nargs="+",
                        help="Input question JSON files (from 3a/3b)")
    parser.add_argument("--provider",
                        choices=["deepseek", "deepseek-reasoner", "anthropic", "openai", "gemini"],
                        default="openai",
                        help="LLM provider (default: openai)")
    parser.add_argument("--model", type=str, default="gpt-5.4",
                        help="Model (default: gpt-5.4)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N questions per file")
    parser.add_argument("--max-tokens", type=int, default=2000,
                        help="Max tokens per LLM response (default: 2000)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (default: 1.0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show context gathering only, no LLM calls")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling (default: 42)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file (default: input_mcq.json)")
    parser.add_argument("--contents", type=str, default=None,
                        help="Path to paper_contents.json")
    parser.add_argument("--chains-dir", type=str, default=None,
                        help="Path to chains directory")
    parser.add_argument("--batch-api", action="store_true",
                        help="Use async batch API for 50%% cost reduction (OpenAI/Anthropic)")
    args = parser.parse_args()

    # Load paper contents
    contents_path = (
        args.contents
        or str(Path(CHAINS_DIR) / "content" / "paper_contents.json")
    )
    print(f"Loading paper contents from {contents_path}...")
    with open(contents_path) as f:
        contents = json.load(f)
    print(f"  Loaded {len(contents)} papers")

    # Build sibling index and seed arxivId lookup
    chains_dir = args.chains_dir or CHAINS_DIR
    print(f"Building sibling index from {chains_dir}...")
    sibling_index = build_seed_sibling_index(chains_dir)
    seed_arxiv_lookup = _build_seed_arxiv_lookup(chains_dir)
    print(f"  Indexed {len(sibling_index)} seeds, "
          f"{sum(len(v) for v in sibling_index.values())} terminal papers")

    # Determine output file early (for skip-existing check)
    if args.output:
        out_path = Path(args.output)
    elif len(args.inputs) == 1:
        out_path = Path(args.inputs[0]).with_stem(
            Path(args.inputs[0]).stem + "_mcq")
    else:
        out_path = Path(QUESTIONS_DIR) / "mcq_combined.json"

    # Load existing MCQs for incremental mode
    existing_mcqs = []
    done_ids = set()
    if out_path.exists():
        with open(out_path) as f:
            existing_mcqs = json.load(f)
        done_ids = {q.get("id") for q in existing_mcqs if q.get("id")}
        if done_ids:
            print(f"Found {len(existing_mcqs)} existing MCQs "
                  f"({len(done_ids)} question IDs already done)")

    # Collect all questions from all input files
    all_questions = []

    for input_path in args.inputs:
        fpath = Path(input_path)
        if not fpath.exists():
            fpath = Path(QUESTIONS_DIR) / input_path
        if not fpath.exists():
            print(f"File not found: {input_path}")
            continue

        with open(fpath) as f:
            questions = json.load(f)
        if args.limit:
            questions = questions[:args.limit]

        all_questions.extend(questions)
        print(f"  {fpath.name}: {len(questions)} questions")

    if not all_questions:
        print("No questions to process.")
        return

    # Skip already-processed questions (incremental mode)
    if done_ids:
        before = len(all_questions)
        all_questions = [q for q in all_questions if q.get("id") not in done_ids]
        print(f"Skipping {before - len(all_questions)} already-processed questions, "
              f"{len(all_questions)} new to process")

    if not all_questions:
        print("All questions already processed. Nothing to do.")
        return

    n_bfs = sum(1 for q in all_questions if not (q.get("id", "").startswith("dfs_") or "group" in q))
    n_dfs = len(all_questions) - n_bfs
    print(f"  Total: {n_bfs} BFS, {n_dfs} DFS ({len(all_questions)} total)")

    # Gather distractor context and drop questions missing any type
    print(f"\nGathering distractor context for {len(all_questions)} questions...")
    full_questions = []
    full_contexts = []
    dropped = 0

    for q in all_questions:
        ctx = gather_distractor_contexts(q, contents, sibling_index,
                                         seed_arxiv_lookup)
        types = set(ctx["available_types"])
        if types >= {"seed_only", "wrong_paper", "no_context"}:
            full_questions.append(q)
            full_contexts.append(ctx)
        else:
            dropped += 1

    all_questions = full_questions
    contexts = full_contexts
    print(f"  Kept: {len(all_questions)} (all 3 distractor types available)")
    print(f"  Dropped: {dropped} (missing context for >=1 distractor type)")

    if args.dry_run:
        print("\n[DRY RUN] Showing prompts for first question:\n")
        q, ctx = all_questions[0], contexts[0]
        for dtype in ctx["available_types"]:
            builder = PROMPT_BUILDERS[dtype]
            prompt = builder(q, ctx)
            print(f"{'='*70}")
            print(f"[{q.get('id', '?')}] type={dtype}")
            print(f"{'='*70}")
            print(prompt[:2000])
            print(f"... ({len(prompt)} chars total)\n")
        print(f"System prompt:\n{SYSTEM_PROMPT}")
        return

    # Build prompts for all 3 distractor types.
    # seed_only and wrong_paper share SYSTEM_PROMPT (paper-grounded);
    # no_context uses NO_CONTEXT_SYSTEM_PROMPT (parametric knowledge).
    distractor_types = ["seed_only", "wrong_paper", "no_context"]
    type_results = {dt: [None] * len(all_questions) for dt in distractor_types}

    paper_prompts, paper_keys = [], []
    noctx_prompts, noctx_keys = [], []

    for i, (q, ctx) in enumerate(zip(all_questions, contexts)):
        for dtype in distractor_types:
            if dtype not in ctx["available_types"]:
                continue
            builder = PROMPT_BUILDERS[dtype]
            prompt = builder(q, ctx)
            if dtype == "no_context":
                noctx_prompts.append(prompt)
                noctx_keys.append(("no_context", i))
            else:
                paper_prompts.append(prompt)
                paper_keys.append((dtype, i))

    print(f"  seed_only + wrong_paper: {len(paper_prompts)} prompts")
    print(f"  no_context: {len(noctx_prompts)} prompts")

    # Batch 1: paper-grounded distractors
    if paper_prompts:
        print(f"\n  Calling {args.provider} for {len(paper_prompts)} "
              f"paper-grounded prompts...")
        paper_results = llm_batch(
            paper_prompts, SYSTEM_PROMPT,
            provider=args.provider, model=args.model,
            max_tokens=args.max_tokens, temperature=args.temperature,
            use_batch_api=args.batch_api,
        )
        for (dtype, qi), result in zip(paper_keys, paper_results):
            type_results[dtype][qi] = result

    # Batch 2: no-context (parametric knowledge)
    if noctx_prompts:
        print(f"  Calling {args.provider} for {len(noctx_prompts)} "
              f"no-context prompts...")
        noctx_results = llm_batch(
            noctx_prompts, NO_CONTEXT_SYSTEM_PROMPT,
            provider=args.provider, model=args.model,
            max_tokens=args.max_tokens, temperature=args.temperature,
            use_batch_api=args.batch_api,
        )
        for (dtype, qi), result in zip(noctx_keys, noctx_results):
            type_results[dtype][qi] = result

    # Collect all 3 independent results per question, with overlap filtering
    per_q_valid = []
    llm_failed = 0
    overlap_dropped = 0

    for i, (q, ctx) in enumerate(zip(all_questions, contexts)):
        correct_answer = q.get("answer", "").strip()
        valid = []
        seen_lower = {correct_answer.lower()}

        for dtype in distractor_types:
            r = type_results[dtype][i]
            if not r:
                continue
            text = r.get("answer", "").strip()
            if not text:
                continue
            # Exact-match dedup
            if text.lower() in seen_lower:
                continue
            # Overlap filtering: drop if too similar to correct answer
            if (word_overlap(text, correct_answer) > OVERLAP_THRESHOLD
                    or numeric_overlap(correct_answer, text) > 0.5):
                overlap_dropped += 1
                continue
            seen_lower.add(text.lower())
            valid.append({
                "type": dtype,
                "text": text,
                "answer_context": r.get("answer_context", ""),
            })

        per_q_valid.append(valid)
        if len(valid) < 3:
            llm_failed += 1

    if llm_failed:
        print(f"\n  {llm_failed} questions got <3 valid distractors "
              "(parse error, exact-match collision, or overlap filter)")
    if overlap_dropped:
        print(f"  {overlap_dropped} distractors dropped by overlap filter "
              f"(Jaccard > {OVERLAP_THRESHOLD})")

    # Build final MCQ output
    mcq_questions = []
    skipped = 0
    for i, (q, ctx) in enumerate(zip(all_questions, contexts)):
        valid = per_q_valid[i]
        if len(valid) < 3:
            skipped += 1
            continue

        # Build options and shuffle
        answer = q.get("answer", "")
        options = [answer] + [d["text"] for d in valid[:3]]

        q_id = q.get("id", "unknown")
        shuffle_rng = random.Random(f"{args.seed}_{q_id}")
        indices = list(range(4))
        shuffle_rng.shuffle(indices)

        shuffled_options = [options[j] for j in indices]
        correct_index = indices.index(0)

        distractor_sources = []
        for d in valid[:3]:
            source = {"type": d["type"]}
            if d.get("answer_context"):
                source["answer_context"] = d["answer_context"]
            distractor_sources.append(source)

        # Preserve raw seed_only response for validation
        seed_raw = type_results["seed_only"][i]
        if seed_raw and seed_raw.get("answer"):
            seed_only_text = seed_raw["answer"].strip()
        else:
            seed_only_text = None

        enriched = dict(q)
        enriched["options"] = shuffled_options
        enriched["correct_index"] = correct_index
        enriched["distractor_sources"] = distractor_sources
        if seed_only_text:
            enriched["seed_only_response"] = seed_only_text

        # Attach sibling arxivIds
        sibling_arxivs = ctx.get("sibling_arxivs", [])
        for ds in distractor_sources:
            if ds["type"] == "wrong_paper" and sibling_arxivs:
                ds["arxivId"] = sibling_arxivs[0]

        mcq_questions.append(enriched)

    print(f"\n  MCQs generated: {len(mcq_questions)}/{len(all_questions)}")
    if skipped:
        print(f"  Skipped: {skipped} (LLM failure, collision, or overlap)")

    # Save output (append to existing)
    combined = existing_mcqs + mcq_questions
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False))
    print(f"  New: {len(mcq_questions)}, Existing: {len(existing_mcqs)}")
    print(f"  Total: {len(combined)} MCQs → {out_path}")

    # Show sample
    for q in mcq_questions[:2]:
        if "options" in q:
            print(f"\n  [{q.get('id', '?')}]")
            for j, opt in enumerate(q["options"]):
                marker = "*" if j == q["correct_index"] else " "
                print(f"    {marker} {chr(65+j)}) {opt[:100]}")


if __name__ == "__main__":
    main()
