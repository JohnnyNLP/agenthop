"""Stage 3 (Q/A generation — single-target half): single-gold-paper QA pairs.

Generates single-target QA pairs along a citation chain — one gold paper,
reachable from the seed by one or two citation hops. Companion file
``stage4b_qa_generation_mt.py`` handles the multi-target case.

Usage:
    python 3.generate_questions.py --limit 10               # test on 10 chains
    python 3.generate_questions.py --limit 10 --dry-run      # preview prompts
    python 3.generate_questions.py                           # all chains
    python 3.generate_questions.py --provider anthropic      # use Claude
    python 3.generate_questions.py --min-depth 2             # skip depth-1 chains
"""

import argparse
import json
import glob
from collections import Counter, defaultdict
from pathlib import Path

from config import CHAINS_DIR, QUESTIONS_DIR
from llm_client import llm_batch
from question_modes import assign_modes, get_mode_instructions


# ── Prompt templates ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are creating research questions that model how a researcher THINKS while reading a paper.

A researcher reads the seed paper, notices a specific claim, comparison, or referenced technique, and follows citations to the terminal paper where they find the answer.

Your job: given a citation chain (seed → ... → terminal) with full paper content, create a question that captures this natural research curiosity. Each prompt includes a QUESTION MODE specifying the citation relationship and cognitive skill to target.

=== STEP 1: UNDERSTAND THE SEED PAPER ===

Before generating any question, first read the seed paper carefully and identify:
1. What is the paper's core contribution or thesis?
2. What specific claims, comparisons, or design choices does it make?
3. How does it reference or rely on the cited works in this chain?

Articulate this understanding in the "seed_understanding" field. This grounds everything that follows — a good question cannot come from a superficial reading.

=== ANSWER RULES ===

The answer must be 1-2 sentences stating a specific finding, design choice, or result as a direct factual claim. Write as if asserting the fact itself — do NOT attribute it to a paper ("the paper reports...", "the cited work shows...", "the authors found..."). Just state the finding.

- WRONG: "The cited paper reports that the model achieves 94.1% accuracy on COCO."
- WRONG: "According to the referenced work, tiling reduces memory by 3x."
- RIGHT: "The model achieves 94.1% accuracy on COCO with a ResNet-50 backbone."
- RIGHT: "Tiling the attention computation reduces peak memory usage by 3x."

Do not give a bare number; describe the finding. Do not speculate or add reasoning beyond what the terminal paper states.

If no good answer exists in the terminal paper content, output "answer": null.

=== ANSWER_CONTEXT RULES ===

The answer_context field must be EXACT TEXT copied from the terminal paper content above. Do not add any framing, attribution, or modification:
- WRONG: "From Table 3: the model achieves 94.1%..."
- WRONG: "From the terminal paper: 'We use T=1000...'"
- WRONG: "The authors report that..."
- RIGHT: "We use T = 1000 for all experiments. We set the forward process variances to constants increasing linearly from β1 = 10^-4 to βT = 0.02."

If the answer comes from a table, copy the relevant row(s) exactly as they appear in the table content. Do not paraphrase or summarize table data.

=== QUESTION RULES ===

The question must be GROUNDED in a specific detail from the seed paper — a claim, comparison, design choice, limitation, or cited result.

CRITICAL — SEED DEPENDENCY: The question must require information from the seed paper to answer, not just as a navigation clue but as CONTEXT that determines what to look for in the terminal. The question text should reference a concrete detail from the seed (a number, a claim, a comparison, a stated limitation) that CONSTRAINS which part of the terminal paper is relevant.

Self-check: if someone receives ONLY the terminal paper and this question (without the seed), would they immediately know which of the terminal's many findings to focus on? If yes, the seed is just a pointer — rewrite so the seed's specific claim narrows the answer.

Good pattern: "Seed claims X about cited work → what does cited work report about X?"
Bad pattern: "Seed mentions cited work → what does cited work do?"

Naming:
- You MAY freely name the seed paper, its methods/models, and intermediate papers
- MUST NOT name the terminal paper by title — the agent must discover it through navigation

Phrasing — do NOT use any of these patterns:
- "In [paper], the authors..."
- "While reading [paper], a researcher..."
- "A researcher reading [paper]..."
- "To contextualize..."

Instead, write direct questions grounded in specific paper content.

Output as JSON:
{
  "seed_understanding": "2-3 sentences: the seed paper's core contribution and how the citation chain relates to it",
  "seed_detail": "The specific claim/comparison/detail from the seed paper that motivates the question",
  "seed_dependency": "Explain how the seed's detail constrains what to look for in the terminal — why can't someone answer this from the terminal alone?",
  "question": "The research question (2-4 sentences, direct phrasing)",
  "answer": "1-2 sentence answer drawn from the terminal paper",
  "answer_context": "The sentence/paragraph containing the answer (copy verbatim from terminal content)",
  "hops_required": "what each hop contributes to answering"
}"""


# ── Content formatting ───────────────────────────────────────────────────────

def format_contexts(contexts: list[str], max_per: int = 3) -> str:
    """Format citation contexts for the prompt."""
    if not contexts:
        return "(no citation context available)"
    parts = []
    for i, ctx in enumerate(contexts[:max_per]):
        parts.append(f'  [{i+1}] "{ctx}"')
    return "\n".join(parts)


def format_paper_content(sections: list[dict], tables: list[dict]) -> str:
    """Format paper sections and tables — full content, no truncation.

    Sections are ordered by relevance: results/experiments first, then
    methods, then everything else. All content is included.
    """
    if not sections and not tables:
        return "(paper content not available — use abstract only)"

    priority_kw = ["result", "experiment", "evaluation", "analysis",
                   "performance", "ablation", "comparison"]
    secondary_kw = ["method", "approach", "model", "setup", "implementation",
                    "architecture"]

    priority, secondary, other = [], [], []
    for sec in sections:
        h = sec["header"].lower()
        if any(kw in h for kw in priority_kw):
            priority.append(sec)
        elif any(kw in h for kw in secondary_kw):
            secondary.append(sec)
        else:
            other.append(sec)

    parts = []
    for group in [priority, secondary, other]:
        for sec in group:
            parts.append(f"### {sec['header']}\n{sec['text']}")

    # Tables
    for table in tables:
        caption = table.get("caption", "")
        rows = table.get("rows", [])
        table_str = f"\n### Table: {caption}\n"
        for row in rows:
            table_str += " | ".join(str(cell) for cell in row) + "\n"
        parts.append(table_str)

    return "\n\n".join(parts) if parts else "(paper content not available)"


# ── Prompt building ──────────────────────────────────────────────────────────

def build_chain_prompt(chain: dict, contents: dict,
                       mode: tuple[str, str] | None = None) -> str:
    """Build the user prompt for a chain of any depth.

    All papers get full content so the LLM can build a rich narrative:
      - Seed: full content (researcher reads this carefully)
      - Intermediate hops: full content + citation contexts (narrative bridge)
      - Terminal: full content (answer lives here)

    Chain format: path[-1].paper == terminal. So:
      - path[:-1] = intermediate hops
      - path[-1] = terminal hop (with edge_from_parent = citation edge to terminal)

    If mode is provided, injects mode-specific instructions (relationship x skill).
    """
    seed = chain["seed"]
    terminal = chain["terminal"]
    path = chain.get("path", [])
    depth = chain.get("depth", len(path))

    # Split path: intermediates vs terminal hop
    intermediate_hops = path[:-1]  # empty for depth-1
    terminal_hop = path[-1] if path else {}

    parts = []

    # Seed paper — full content
    seed_aid = seed.get("arxivId", "")
    seed_content = contents.get(seed_aid, {})

    parts.append(f"""=== SEED PAPER ===
Title: {seed.get('title', '?')}
Year: {seed.get('year', '?')}
Abstract: {seed.get('abstract') or '(no abstract)'}

=== SEED PAPER CONTENT ===
{format_paper_content(
    seed_content.get('sections', []),
    seed_content.get('tables', []),
)}""")

    # Intermediate hops (depth 2+): full content + citation contexts
    for i, hop in enumerate(intermediate_hops):
        paper = hop.get("paper", {})
        edge = hop.get("edge_from_parent", {})
        contexts = edge.get("contexts", [])
        parent_label = "seed" if i == 0 else f"hop {i}"

        hop_aid = paper.get("arxivId", "")
        hop_content = contents.get(hop_aid, {})

        parts.append(f"""
=== HOP {i+1} PAPER (intermediate) ===
Title: {paper.get('title', '?')}
Year: {paper.get('year', '?')}
Venue: {paper.get('venue') or 'N/A'}
Abstract: {paper.get('abstract') or '(no abstract)'}

Citation context (how {parent_label} cites this):
{format_contexts(contexts)}

=== HOP {i+1} PAPER CONTENT ===
{format_paper_content(
    hop_content.get('sections', []),
    hop_content.get('tables', []),
)}""")

    # Terminal paper — full content
    terminal_edge = terminal_hop.get("edge_from_parent", {})
    terminal_contexts = terminal_edge.get("contexts", [])
    parent_label = "seed" if depth == 1 else f"hop {depth - 1}"

    terminal_aid = terminal.get("arxivId", "")
    terminal_content = contents.get(terminal_aid, {})

    parts.append(f"""
=== TERMINAL PAPER (hop {depth}) ===
Title: {terminal.get('title', '?')}
Year: {terminal.get('year', '?')}
Venue: {terminal.get('venue') or 'N/A'}
Abstract: {terminal.get('abstract') or '(no abstract)'}

Citation context (how {parent_label} cites this):
{format_contexts(terminal_contexts)}

=== TERMINAL PAPER CONTENT ===
{format_paper_content(
    terminal_content.get('sections', []),
    terminal_content.get('tables', []),
)}""")

    # Instructions
    hop_label = f"{depth}-hop" if depth > 1 else "1-hop"

    mode_block = ""
    if mode:
        rel, skill = mode
        mode_block = "\n" + get_mode_instructions(rel, skill, "bfs") + "\n"

    parts.append(f"""
=== INSTRUCTIONS ===
This is a {hop_label} chain. Generate a question that requires following this chain.
{mode_block}
1. Identify a specific claim, comparison, or characterization in the SEED paper about the terminal paper's work.
2. Find a corresponding finding, result, or detail in the TERMINAL paper that addresses or verifies the seed's claim.
3. Write the question so it EMBEDS the seed's specific claim and asks for the terminal's corresponding detail:
   - You MAY name the seed paper, its methods/models, and intermediate papers
   - DO NOT name the terminal paper — the agent must discover it
   - For each hop, describe the curiosity gap that motivates following that citation
   - The seed's claim must CONSTRAIN which part of the terminal paper is relevant
4. Verify: the question does NOT mention the terminal paper by title or method name.
5. Verify: someone with ONLY the terminal paper cannot determine which finding the question targets.
6. Verify: the question matches the assigned relationship and skill constraints.

Output as JSON only.""")

    return "\n".join(parts)


# ── Chain loading ────────────────────────────────────────────────────────────

def load_all_chains(min_depth: int = 1, max_depth: int = 3) -> list[dict]:
    """Load chains from all chain files, filtered by depth."""
    chains = []
    for fpath in sorted(glob.glob(str(Path(CHAINS_DIR) / "chains_d*.json"))):
        with open(fpath) as f:
            file_chains = json.load(f)
        for c in file_chains:
            d = c.get("depth", 1)
            if min_depth <= d <= max_depth:
                chains.append(c)
    return chains


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-hop research questions from citation chains"
    )
    parser.add_argument("--chains", type=str, default=None,
                        help="Path to specific chains JSON file (default: all)")
    parser.add_argument("--contents", type=str, default=None,
                        help="Path to paper contents JSON")
    parser.add_argument("--provider", choices=["deepseek", "deepseek-reasoner", "anthropic", "openai", "gemini"],
                        default="openai")
    parser.add_argument("--model", type=str, default="gpt-5.4")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only N chains (for testing)")
    parser.add_argument("--min-depth", type=int, default=1,
                        help="Minimum chain depth (default: 1)")
    parser.add_argument("--max-depth", type=int, default=2,
                        help="Maximum chain depth (default: 2)")
    parser.add_argument("--max-tokens", type=int, default=2000,
                        help="Max output tokens per question (default: 2000)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompts without calling API")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for reproducible mode assignment (default: 42)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file path")
    parser.add_argument("--one-per-seed", action="store_true", default=True,
                        help="Generate at most one question per seed (best chain only, default: True)")
    parser.add_argument("--no-one-per-seed", action="store_false", dest="one_per_seed",
                        help="Allow multiple questions per seed")
    parser.add_argument("--batch-api", action="store_true",
                        help="Use async batch API for 50%% cost reduction (OpenAI/Anthropic)")
    args = parser.parse_args()

    # Determine output file early (for skip-existing check)
    out_dir = Path(QUESTIONS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = Path(args.output) if args.output else out_dir / "generated_questions.json"

    # Load existing results for incremental mode
    existing_output = []
    done_seeds = set()
    if out_file.exists():
        with open(out_file) as f:
            existing_output = json.load(f)
        done_seeds = {
            q["chain"]["seed"]["paperId"]
            for q in existing_output
            if q.get("chain", {}).get("seed", {}).get("paperId")
        }
        if done_seeds:
            print(f"Found {len(existing_output)} existing questions "
                  f"({len(done_seeds)} seeds already done)")

    # Load chains
    if args.chains:
        with open(args.chains) as f:
            chains = json.load(f)
        chains = [c for c in chains
                  if args.min_depth <= c.get("depth", 1) <= args.max_depth]
    else:
        chains = load_all_chains(args.min_depth, args.max_depth)

    print(f"Loaded {len(chains)} chains (depth {args.min_depth}-{args.max_depth})")

    # Seed-diverse sampling: pick top chains per seed (round-robin by seed)
    # to avoid concentration on high-yield seeds
    chains.sort(key=lambda c: -(c.get("screen", {}).get("score", 0)))

    if args.one_per_seed:
        # Pick the single best chain per seed (already sorted by score desc)
        seen_seeds = {}
        for c in chains:
            sid = c["seed"].get("paperId", "?")
            if sid not in seen_seeds:
                seen_seeds[sid] = c
        chains = list(seen_seeds.values())
        if args.limit:
            chains = chains[:args.limit]
        print(f"One-per-seed: {len(chains)} chains (1 per seed)")
    elif args.limit:
        by_seed = defaultdict(list)
        for c in chains:
            sid = c["seed"].get("paperId", "?")
            by_seed[sid].append(c)

        # Round-robin: take 1 from each seed, then 2nd from each, etc.
        selected = []
        round_num = 0
        while len(selected) < args.limit:
            added_this_round = False
            for sid in list(by_seed.keys()):
                if round_num < len(by_seed[sid]):
                    selected.append(by_seed[sid][round_num])
                    added_this_round = True
                    if len(selected) >= args.limit:
                        break
            if not added_this_round:
                break
            round_num += 1

        chains = selected
        n_seeds = len(set(c["seed"].get("paperId") for c in chains))
        print(f"Selected {len(chains)} chains from {n_seeds} unique seeds")

    # Skip already-processed seeds (incremental mode)
    if done_seeds:
        before = len(chains)
        chains = [c for c in chains
                  if c["seed"].get("paperId") not in done_seeds]
        print(f"Skipping {before - len(chains)} already-processed seeds, "
              f"{len(chains)} new chains to process")

    if not chains:
        print("All chains already processed. Nothing to do.")
        return

    # Load paper contents
    contents_file = args.contents or str(Path(CHAINS_DIR) / "content" / "paper_contents.json")
    contents = {}
    if Path(contents_file).exists():
        with open(contents_file) as f:
            contents = json.load(f)
        print(f"Paper contents loaded: {len(contents)} papers")
    else:
        print(f"Warning: {contents_file} not found — using abstracts only")

    # Depth distribution
    depth_dist = Counter(c.get("depth", 1) for c in chains)
    print(f"Depth distribution: {dict(sorted(depth_dist.items()))}")

    # Assign question modes
    modes = assign_modes(chains, seed=args.seed)
    mode_dist = Counter(modes)
    rel_dist = Counter(r for r, s in modes)
    skill_dist = Counter(s for r, s in modes)
    print(f"Mode assignments: {len(modes)} total")
    print(f"  Relationships: {dict(sorted(rel_dist.items()))}")
    print(f"  Skills: {dict(sorted(skill_dist.items()))}")

    # Build prompts
    prompts = []
    chain_refs = []  # parallel list to track which chain each prompt came from
    for chain, mode in zip(chains, modes):
        prompt = build_chain_prompt(chain, contents, mode=mode)
        prompts.append(prompt)
        chain_refs.append(chain)

    if args.dry_run:
        for i, (prompt, chain, mode) in enumerate(zip(prompts, chain_refs, modes)):
            rel, skill = mode
            seed_title = chain["seed"]["title"][:50]
            term_title = chain["terminal"]["title"][:50]
            print(f"\n{'='*70}")
            print(f"Chain {i+1} (depth {chain['depth']}): {seed_title} → ... → {term_title}")
            print(f"Mode: {rel} × {skill}")
            print(f"Prompt length: {len(prompt)} chars")
            print(f"{'='*70}")
            print(prompt[:2000])
            if len(prompt) > 2000:
                print(f"... ({len(prompt) - 2000} more chars)")
        return

    # Generate questions via LLM
    print(f"\nGenerating {len(prompts)} questions via {args.provider}...")
    results = llm_batch(
        prompts,
        SYSTEM_PROMPT,
        provider=args.provider,
        model=args.model,
        max_tokens=args.max_tokens,
        use_batch_api=args.batch_api,
    )

    # Merge results with chain metadata
    output = []
    errors = 0
    for i, (qa, chain, mode) in enumerate(zip(results, chain_refs, modes)):
        rel, skill = mode
        seed_title = chain["seed"]["title"][:50]
        term_title = chain["terminal"]["title"][:50]

        if qa.get("score") == 0 and "error" in qa.get("reasoning", ""):
            print(f"  [{i+1}] ERROR: {seed_title} → {term_title}")
            errors += 1
            continue

        # Skip null answers
        if not qa.get("question") or not qa.get("answer"):
            print(f"  [{i+1}] NULL: {seed_title} → {term_title}")
            errors += 1
            continue

        entry = {
            "id": f"q_{len(existing_output) + len(output):04d}",
            "question_type": "bfs",
            "depth": chain["depth"],
            "question_mode": {"relationship": rel, "skill": skill},
            "seed_understanding": qa.get("seed_understanding", ""),
            "seed_detail": qa.get("seed_detail", ""),
            "question": qa.get("question", ""),
            "answer": qa.get("answer", ""),
            "answer_context": qa.get("answer_context", ""),
            "reasoning_type": rel,
            "hops_required": qa.get("hops_required", ""),
            "chain": {
                "seed": {
                    "paperId": chain["seed"].get("paperId"),
                    "arxivId": chain["seed"].get("arxivId"),
                    "title": chain["seed"].get("title"),
                },
                "terminal": {
                    "paperId": chain["terminal"].get("paperId"),
                    "arxivId": chain["terminal"].get("arxivId"),
                    "title": chain["terminal"].get("title"),
                },
                "path_titles": [
                    hop["paper"].get("title")
                    for hop in chain.get("path", [])
                ],
                "depth": chain["depth"],
                "screen_score": chain.get("screen", {}).get("score"),
                "chain_score": chain.get("metadata", {}).get("chain_score"),
            },
        }
        output.append(entry)

        # Print preview
        q_preview = (qa.get("question") or "")[:120]
        a_preview = (qa.get("answer") or "(null)")[:80]
        print(f"  [{i+1}] d={chain['depth']} {rel}×{skill} | {seed_title}")
        print(f"       Q: {q_preview}")
        print(f"       A: {a_preview}")

    # Save (append to existing)
    combined = existing_output + output
    out_file.write_text(json.dumps(combined, indent=2, ensure_ascii=False))

    print(f"\n{'='*70}")
    print(f"New: {len(output)} questions ({errors} errors)")
    if existing_output:
        print(f"Existing: {len(existing_output)} (kept)")
    print(f"Total: {len(combined)} questions")
    print(f"Depth breakdown: {dict(Counter(q['depth'] for q in combined))}")
    print(f"Relationships: {dict(Counter(q['question_mode']['relationship'] for q in combined))}")
    print(f"Skills: {dict(Counter(q['question_mode']['skill'] for q in combined))}")
    print(f"Saved to: {out_file}")


if __name__ == "__main__":
    main()
