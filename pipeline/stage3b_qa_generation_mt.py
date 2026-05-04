"""Stage 3 (Q/A generation — multi-target half): two-paper-synthesis QA pairs.

Generates multi-target QA pairs whose answer requires synthesising content
from two cited papers simultaneously, neither sufficient on its own.
Companion file ``stage4a_qa_generation_st.py`` handles the single-target
case.

Supports two depths:
  Depth 1 — Both targets are direct citations of the seed
  Depth 2 — Both targets are citations of the same hop-1 (intermediate) paper

The input unit is fundamentally different: sibling groups of 2-3 papers
sharing a common seed, rather than a single citation chain.

Sibling grouping algorithm (3 tiers):
  Tier 1 — Co-citation: papers share an exact citation context string
  Tier 2 — Same comparison type: both have 'comparison' context_type
  Tier 3 — Methodology fallback: both have 'methodology' intent, score ≥ 4

Usage:
    python 3b.generate_dfs_questions.py --limit 10               # test
    python 3b.generate_dfs_questions.py --limit 10 --dry-run     # preview prompts
    python 3b.generate_dfs_questions.py --save-groups             # save grouping
    python 3b.generate_dfs_questions.py                           # full run
"""

import argparse
import json
import glob
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from config import CHAINS_DIR, QUESTIONS_DIR
from llm_client import llm_batch
from question_modes import assign_dfs_modes, get_mode_instructions

# Reuse formatting helpers from BFS script
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "gen_questions", Path(__file__).parent / "3a.generate_bfs_questions.py"
)
_bfs = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_bfs)
format_paper_content = _bfs.format_paper_content
format_contexts = _bfs.format_contexts


# ── Prompt templates ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are creating research questions that require SYNTHESIZING information across multiple cited papers.

A researcher reads the seed paper, notices a passage discussing or comparing two or three cited works, and realizes they need to read BOTH cited works and combine their findings to fully understand the comparison.

Your job: given a seed paper and 2-3 target papers it cites (with full content), create a question that:
1. Guides the reader from the seed paper TOWARD the target papers through narrative clues
2. REQUIRES reading multiple target papers to answer

Each prompt includes a QUESTION MODE specifying the citation relationship and cognitive skill to target.

=== STEP 1: UNDERSTAND THE SEED PAPER ===

Before generating any question, first read the seed paper carefully and identify:
1. What is the paper's core contribution or thesis?
2. What specific passage discusses or compares the target papers together?
3. What role does each target paper play in the seed's narrative (baseline, inspiration, competing approach, etc.)?

Articulate this understanding in the "seed_understanding" field. This grounds everything that follows — a good synthesis question cannot come from a superficial reading.

=== ANSWER RULES ===

The answer must be 1-2 sentences combining specific observations from both papers as direct factual claims. Write as if asserting the facts — do NOT attribute them to papers ("the paper reports...", "the cited work shows...", "the authors found..."). Just state the findings.

- WRONG: "The first paper reports 94.1% AP while the second paper achieves 91.3% AP."
- RIGHT: "The sparse detector achieves 94.1% AP while the conditional query approach reaches 91.3% AP on COCO val2017."

MUST NOT name target papers — use role descriptions ("the first approach", "the sparse detector"). Do not speculate or add reasoning beyond what the papers state.

If no good synthesis question exists, output null for all fields.

=== ANSWER_SOURCES RULES ===

Each "fact" in answer_sources must be EXACT TEXT copied from that paper's content above. Do not add any framing, attribution, or modification:
- WRONG: "Table 3 shows that the model achieves 94.1%..."
- WRONG: "The authors report that..."
- RIGHT: "achieves 94.1% accuracy on COCO val2017 with a ResNet-50 backbone"

If the fact comes from a table, copy the relevant row(s) exactly as they appear in the table content. Do not paraphrase or summarize table data.

=== QUESTION RULES ===

The question must be grounded in a specific passage from the seed paper where multiple cited works are discussed together. Describe each target paper by the ROLE or CHARACTERIZATION the seed gives it — not by name.

CRITICAL — SEED DEPENDENCY: The question must require information from the seed paper to answer, not just as a way to identify which papers to read. The seed should provide context that CONSTRAINS what to look for across the target papers — a specific comparison frame, a claim to verify, or a characterization that narrows the relevant findings.

Self-check: if someone receives ONLY the target papers and this question (without the seed), would they immediately know which findings to compare and how? If yes, the seed is unnecessary — rewrite so the seed's specific framing determines the answer.

Naming:
- You MAY freely name the SEED paper, its methods, models, and datasets
- MUST NOT name target papers by title, method name, model name, or acronym
- Instead, DESCRIBE each target paper by the role the seed gives it:
    - "a sparse detector that learns proposals directly" (not "Sparse R-CNN")
    - "an approach that accelerates convergence via conditional spatial queries" (not "Conditional DETR")
- The description must be specific enough that a reader of the seed paper can identify which citation to follow

Phrasing — do NOT use any of these patterns:
- "In [paper], the authors..."
- "While reading [paper], a researcher..."
- "A researcher reading [paper]..."
- "To contextualize..."

Instead, write direct questions grounded in specific paper content.

Output as JSON:
{
  "seed_understanding": "2-3 sentences: the seed paper's core contribution and how the target papers relate to it",
  "seed_detail": "The specific passage from the seed paper motivating this question",
  "seed_dependency": "Explain how the seed's framing constrains what to look for across target papers — why can't someone answer this from the targets alone?",
  "question": "The synthesis question (2-4 sentences, target papers described by role not name)",
  "answer": "1-2 sentence answer combining observations from both papers (no target paper names)",
  "answer_sources": {
    "paper_1": {"arxivId": "...", "fact": "specific fact from paper 1"},
    "paper_2": {"arxivId": "...", "fact": "specific fact from paper 2"}
  },
  "why_multi_paper": "Brief explanation of why reading multiple papers is required"
}"""


# ── Sibling grouping ────────────────────────────────────────────────────────

def load_depth1_chains() -> list[dict]:
    """Load all depth-1 chains from chain files."""
    chains = []
    for fpath in sorted(glob.glob(str(Path(CHAINS_DIR) / "chains_d*.json"))):
        with open(fpath) as f:
            file_chains = json.load(f)
        for c in file_chains:
            if c.get("depth") == 1:
                chains.append(c)
    return chains


def load_depth2_chains() -> list[dict]:
    """Load all depth-2 chains from chain files."""
    chains = []
    for fpath in sorted(glob.glob(str(Path(CHAINS_DIR) / "chains_d*.json"))):
        with open(fpath) as f:
            file_chains = json.load(f)
        for c in file_chains:
            if c.get("depth") == 2 and len(c.get("path", [])) >= 2:
                chains.append(c)
    return chains


def group_by_seed(chains: list[dict]) -> dict[str, list[dict]]:
    """Group chains by seed paperId."""
    by_seed = defaultdict(list)
    for c in chains:
        sid = c["seed"].get("paperId", "")
        if sid:
            by_seed[sid].append(c)
    return dict(by_seed)


def _get_context_set(chain: dict) -> set[str]:
    """Get the set of citation context strings for a chain's edge."""
    edge = chain["path"][0].get("edge_from_parent", {})
    return set(edge.get("contexts", []))


def _get_context_types(chain: dict) -> list[str]:
    """Get context_types from chain metadata."""
    return chain.get("metadata", {}).get("context_types", [])


def _get_screen_score(chain: dict) -> int:
    """Get the screening score."""
    return chain.get("screen", {}).get("score", 0)


def _terminal_id(chain: dict) -> str:
    """Get the terminal paper's arxivId."""
    return chain["terminal"].get("arxivId", "")


def _count_citations_in_context(ctx: str) -> int:
    """Count how many papers are referenced in a citation context string.

    Used to distinguish focused co-citations (2-3 papers discussed together)
    from laundry-list co-citations (comparison table listing 5+ baselines).
    """
    parens = re.findall(r'\([^)]*\d{4}[^)]*\)', ctx)
    total = 0
    for p in parens:
        total += p.count(';') + 1
    return max(total, 1)


def _context_specificity(shared_contexts: list[str]) -> int:
    """Score how specific the shared context is (lower = more focused = better).

    Returns the minimum citation count across shared context strings.
    A context mentioning only 2 papers is highly specific (the seed discusses
    exactly this pair together). A context listing 6+ is a comparison table.
    """
    if not shared_contexts:
        return 99
    return min(_count_citations_in_context(c) for c in shared_contexts)


def find_sibling_groups(
    seed_chains: list[dict],
    max_groups: int = 8,
) -> list[dict]:
    """Find sibling groups among depth-1 chains sharing a seed.

    Returns groups sorted by quality, capped at max_groups.
    Each group is a dict with: tier, reason, specificity, chains.

    Sorting priority:
      1. Context specificity (fewer papers in shared context = better)
      2. Tier (co-citation > comparison > methodology)
    This ensures focused pairs ("seed discusses exactly these 2 papers
    together") rank above laundry-list pairs ("seed lists 6 baselines
    in one sentence and these 2 happen to be among them").
    """
    if len(seed_chains) < 2:
        return []

    groups = []
    used_pairs = set()  # track (terminal_id_a, terminal_id_b) to avoid duplicates

    # --- Tier 1: Co-citation (shared exact context string) ---
    for i, ca in enumerate(seed_chains):
        ctx_a = _get_context_set(ca)
        if not ctx_a:
            continue
        for j, cb in enumerate(seed_chains):
            if j <= i:
                continue
            ctx_b = _get_context_set(cb)
            shared = ctx_a & ctx_b
            if shared:
                tid_a, tid_b = _terminal_id(ca), _terminal_id(cb)
                pair_key = tuple(sorted([tid_a, tid_b]))
                if pair_key in used_pairs or not tid_a or not tid_b or tid_a == tid_b:
                    continue
                used_pairs.add(pair_key)
                shared_list = list(shared)[:2]
                groups.append({
                    "tier": 1,
                    "reason": "co-citation",
                    "shared_contexts": shared_list,
                    "specificity": _context_specificity(shared_list),
                    "chains": [ca, cb],
                })

    # --- Tier 2: Same comparison type (both have 'comparison' in context_types) ---
    comparison_chains = [
        c for c in seed_chains if "comparison" in _get_context_types(c)
    ]
    for ca, cb in combinations(comparison_chains, 2):
        tid_a, tid_b = _terminal_id(ca), _terminal_id(cb)
        pair_key = tuple(sorted([tid_a, tid_b]))
        if pair_key in used_pairs or not tid_a or not tid_b or tid_a == tid_b:
            continue
        used_pairs.add(pair_key)
        groups.append({
            "tier": 2,
            "reason": "same_comparison_type",
            "specificity": 50,  # no shared context — rank below focused T1
            "chains": [ca, cb],
        })

    # --- Tier 3: Methodology intent, both scored ≥ 4 ---
    method_chains = [
        c for c in seed_chains
        if "methodology" in str(_get_context_types(c))
        and _get_screen_score(c) >= 4
    ]
    for ca, cb in combinations(method_chains, 2):
        tid_a, tid_b = _terminal_id(ca), _terminal_id(cb)
        pair_key = tuple(sorted([tid_a, tid_b]))
        if pair_key in used_pairs or not tid_a or not tid_b or tid_a == tid_b:
            continue
        used_pairs.add(pair_key)
        groups.append({
            "tier": 3,
            "reason": "methodology_intent",
            "specificity": 50,
            "chains": [ca, cb],
        })

    # Sort: specificity first (focused co-citations beat laundry lists),
    # then tier as tiebreaker
    groups.sort(key=lambda g: (g["specificity"], g["tier"]))
    return groups[:max_groups]


def find_depth2_sibling_groups(
    depth2_chains: list[dict],
    max_groups: int = 8,
) -> list[dict]:
    """Find sibling groups among depth-2 chains sharing the same hop-1 paper.

    Two depth-2 chains are siblings if they share the same seed AND the same
    hop-1 (path[0]) paper but have different terminals. The sibling
    relationship is based on the hop1→terminal edge (path[1].edge_from_parent).
    """
    # Group by hop-1 paper
    by_hop1 = defaultdict(list)
    for c in depth2_chains:
        hop1_id = c["path"][0]["paper"].get("paperId", "")
        if hop1_id:
            by_hop1[hop1_id].append(c)

    groups = []
    for hop1_id, hop1_chains in by_hop1.items():
        if len(hop1_chains) < 2:
            continue
        used_pairs = set()

        # Helper: get context from hop1→terminal edge (path[1])
        def _ctx_set(chain):
            return set(chain["path"][1].get("edge_from_parent", {}).get("contexts", []))

        # Tier 1: Co-citation at hop1→terminal level
        for i, ca in enumerate(hop1_chains):
            ctx_a = _ctx_set(ca)
            if not ctx_a:
                continue
            for j, cb in enumerate(hop1_chains):
                if j <= i:
                    continue
                ctx_b = _ctx_set(cb)
                shared = ctx_a & ctx_b
                if shared:
                    tid_a, tid_b = _terminal_id(ca), _terminal_id(cb)
                    pair_key = tuple(sorted([tid_a, tid_b]))
                    if pair_key in used_pairs or not tid_a or not tid_b or tid_a == tid_b:
                        continue
                    used_pairs.add(pair_key)
                    shared_list = list(shared)[:2]
                    groups.append({
                        "tier": 1,
                        "reason": "co-citation",
                        "shared_contexts": shared_list,
                        "specificity": _context_specificity(shared_list),
                        "chains": [ca, cb],
                        "dfs_depth": 2,
                        "hop1": ca["path"][0]["paper"],
                    })

        # Tier 2: Same comparison type
        comp_chains = [
            c for c in hop1_chains
            if "comparison" in str(c.get("metadata", {}).get("context_types", []))
        ]
        for ca, cb in combinations(comp_chains, 2):
            tid_a, tid_b = _terminal_id(ca), _terminal_id(cb)
            pair_key = tuple(sorted([tid_a, tid_b]))
            if pair_key in used_pairs or not tid_a or not tid_b or tid_a == tid_b:
                continue
            used_pairs.add(pair_key)
            groups.append({
                "tier": 2, "reason": "same_comparison_type",
                "specificity": 50, "chains": [ca, cb],
                "dfs_depth": 2, "hop1": ca["path"][0]["paper"],
            })

        # Tier 3: Methodology intent, scored ≥ 4
        meth_chains = [
            c for c in hop1_chains
            if "methodology" in str(c.get("metadata", {}).get("context_types", []))
            and _get_screen_score(c) >= 4
        ]
        for ca, cb in combinations(meth_chains, 2):
            tid_a, tid_b = _terminal_id(ca), _terminal_id(cb)
            pair_key = tuple(sorted([tid_a, tid_b]))
            if pair_key in used_pairs or not tid_a or not tid_b or tid_a == tid_b:
                continue
            used_pairs.add(pair_key)
            groups.append({
                "tier": 3, "reason": "methodology_intent",
                "specificity": 50, "chains": [ca, cb],
                "dfs_depth": 2, "hop1": ca["path"][0]["paper"],
            })

    groups.sort(key=lambda g: (g["specificity"], g["tier"]))
    return groups[:max_groups]


# ── Prompt building ──────────────────────────────────────────────────────────

def build_dfs_prompt(group: dict, contents: dict,
                     mode: tuple[str, str] | None = None) -> str:
    """Build the user prompt for a sibling group.

    Includes: seed paper (full content) + shared citation context +
    2-3 target papers (full content each).

    If mode is provided, injects mode-specific instructions (relationship x skill).
    """
    chains = group["chains"]
    seed = chains[0]["seed"]
    seed_aid = seed.get("arxivId", "")
    seed_content = contents.get(seed_aid, {})

    parts = []

    # Seed paper
    parts.append(f"""=== SEED PAPER ===
Title: {seed.get('title', '?')}
Year: {seed.get('year', '?')}
Abstract: {seed.get('abstract') or '(no abstract)'}

=== SEED PAPER CONTENT ===
{format_paper_content(
    seed_content.get('sections', []),
    seed_content.get('tables', []),
)}""")

    # Intermediate paper for depth-2 DFS
    is_depth2 = group.get("dfs_depth") == 2
    if is_depth2 and group.get("hop1"):
        hop1 = group["hop1"]
        hop1_aid = hop1.get("arxivId", "")
        hop1_content = contents.get(hop1_aid, {})
        parts.append(f"""
=== INTERMEDIATE PAPER (hop-1, cited by seed) ===
Title: {hop1.get('title', '?')}
Year: {hop1.get('year', '?')}
Abstract: {hop1.get('abstract') or '(no abstract)'}

=== INTERMEDIATE PAPER CONTENT ===
{format_paper_content(
    hop1_content.get('sections', []),
    hop1_content.get('tables', []),
)}""")

    # Shared citation contexts (if co-citation)
    if group.get("shared_contexts"):
        ctx_text = "\n".join(
            f'  [{i+1}] "{ctx}"'
            for i, ctx in enumerate(group["shared_contexts"][:3])
        )
        ctx_source = "intermediate paper" if is_depth2 else "seed"
        parts.append(f"""
=== SHARED CITATION CONTEXT ({ctx_source} mentions these papers together) ===
{ctx_text}""")

    # Target papers
    for idx, chain in enumerate(chains):
        terminal = chain["terminal"]
        terminal_aid = terminal.get("arxivId", "")
        terminal_content = contents.get(terminal_aid, {})
        # Depth-2: use hop1→terminal edge (path[1]); depth-1: use seed→terminal (path[0])
        edge_idx = 1 if is_depth2 else 0
        edge = chain["path"][edge_idx].get("edge_from_parent", {})
        contexts = edge.get("contexts", [])

        parts.append(f"""
=== TARGET PAPER {idx + 1} ===
Title: {terminal.get('title', '?')}
ArxivId: {terminal_aid}
Year: {terminal.get('year', '?')}
Venue: {terminal.get('venue') or 'N/A'}
Abstract: {terminal.get('abstract') or '(no abstract)'}

Citation context (how {'intermediate paper' if is_depth2 else 'seed'} cites this paper):
{format_contexts(contexts)}

=== TARGET PAPER {idx + 1} CONTENT ===
{format_paper_content(
    terminal_content.get('sections', []),
    terminal_content.get('tables', []),
)}""")

    # Instructions
    n_papers = len(chains)
    tier_label = {1: "co-citation", 2: "comparison", 3: "methodology"}.get(
        group["tier"], "related"
    )

    mode_block = ""
    if mode:
        rel, skill = mode
        mode_block = "\n" + get_mode_instructions(rel, skill, "dfs") + "\n"

    parts.append(f"""
=== INSTRUCTIONS ===
This is a {tier_label} group of {n_papers} target papers cited by the seed.
{mode_block}
Generate a SYNTHESIS question that:
1. Describes each target paper by its ROLE in the seed (e.g., "a sparse detector that learns proposals directly"), NOT by name/acronym
2. Gives enough narrative clues that a reader of the seed paper can identify which citations to follow
3. REQUIRES reading {n_papers} target papers to answer (not just the seed)
4. Has a 1-2 sentence answer combining specific observations from both papers
5. Cannot be answered from any single paper alone
6. Matches the assigned question mode (relationship + skill) above

CRITICAL: Do NOT use target paper titles, method names, or acronyms in the question OR answer.
Describe them through the seed paper's characterization of each work.

Output as JSON only.""")

    return "\n".join(parts)


# ── Content filtering ────────────────────────────────────────────────────────

def filter_groups_by_content(
    groups: list[dict], contents: dict
) -> list[dict]:
    """Keep only groups where ALL required papers have content."""
    filtered = []
    for g in groups:
        all_have_content = True
        # Check target papers
        for chain in g["chains"]:
            tid = chain["terminal"].get("arxivId", "")
            if not tid or tid not in contents:
                all_have_content = False
                break
            paper_content = contents[tid]
            if not paper_content.get("sections"):
                all_have_content = False
                break
        # Check hop-1 paper for depth-2 groups
        if all_have_content and g.get("dfs_depth") == 2 and g.get("hop1"):
            hop1_aid = g["hop1"].get("arxivId", "")
            if not hop1_aid or hop1_aid not in contents:
                all_have_content = False
            elif not contents[hop1_aid].get("sections"):
                all_have_content = False
        if all_have_content:
            filtered.append(g)
    return filtered


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate DFS (breadth-synthesis) questions from sibling paper groups"
    )
    parser.add_argument("--provider", choices=["deepseek", "deepseek-reasoner", "anthropic", "openai", "gemini"],
                        default="openai")
    parser.add_argument("--model", type=str, default="gpt-5.4")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only N groups (for testing)")
    parser.add_argument("--max-tokens", type=int, default=2000,
                        help="Max output tokens per question (default: 2000)")
    parser.add_argument("--max-per-seed", type=int, default=8,
                        help="Max sibling groups per seed (default: 8)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompts without calling API")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for reproducible mode assignment (default: 42)")
    parser.add_argument("--save-groups", action="store_true",
                        help="Save intermediate sibling grouping to JSON")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file path")
    parser.add_argument("--contents", type=str, default=None,
                        help="Path to paper contents JSON")
    parser.add_argument("--chains", type=str, default=None,
                        help="Path to specific chains JSON file (default: all from CHAINS_DIR)")
    parser.add_argument("--one-per-seed", action="store_true", default=True,
                        help="Generate at most one question per seed (best group only, default: True)")
    parser.add_argument("--no-one-per-seed", action="store_false", dest="one_per_seed",
                        help="Allow multiple questions per seed")
    parser.add_argument("--batch-api", action="store_true",
                        help="Use async batch API for 50%% cost reduction (OpenAI/Anthropic)")
    args = parser.parse_args()

    # Determine output file early (for skip-existing check)
    out_dir = Path(QUESTIONS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = Path(args.output) if args.output else out_dir / "dfs_questions.json"

    # Load existing results for incremental mode
    existing_output = []
    done_seeds = set()
    if out_file.exists():
        with open(out_file) as f:
            existing_output = json.load(f)
        done_seeds = {
            q["group"]["seed"]["paperId"]
            for q in existing_output
            if q.get("group", {}).get("seed", {}).get("paperId")
        }
        if done_seeds:
            print(f"Found {len(existing_output)} existing DFS questions "
                  f"({len(done_seeds)} seeds already done)")

    # Load chains (depth-1 and depth-2)
    if args.chains:
        with open(args.chains) as f:
            all_chains = json.load(f)
        chains_d1 = [c for c in all_chains if c.get("depth") == 1]
        chains_d2 = [c for c in all_chains if c.get("depth") == 2
                     and len(c.get("path", [])) >= 2]
    else:
        chains_d1 = load_depth1_chains()
        chains_d2 = load_depth2_chains()
    print(f"Loaded {len(chains_d1)} depth-1 chains, {len(chains_d2)} depth-2 chains")

    by_seed_d1 = group_by_seed(chains_d1)
    by_seed_d2 = group_by_seed(chains_d2)
    all_seed_ids = set(by_seed_d1.keys()) | set(by_seed_d2.keys())
    print(f"From {len(all_seed_ids)} unique seeds")

    # Load paper contents
    contents_file = args.contents or str(
        Path(CHAINS_DIR) / "content" / "paper_contents.json"
    )
    contents = {}
    if Path(contents_file).exists():
        with open(contents_file) as f:
            contents = json.load(f)
        print(f"Paper contents loaded: {len(contents)} papers")
    else:
        print(f"Warning: {contents_file} not found — using abstracts only")

    # Find sibling groups per seed (depth-1)
    all_groups = []
    seeds_with_groups = 0
    tier_counts = defaultdict(int)
    depth_counts = defaultdict(int)

    for sid, seed_chains in sorted(by_seed_d1.items()):
        groups = find_sibling_groups(seed_chains, max_groups=args.max_per_seed)
        groups = filter_groups_by_content(groups, contents)
        if groups:
            seeds_with_groups += 1
            for g in groups:
                g["dfs_depth"] = 1
                g["seed_paperId"] = sid
                g["seed_title"] = seed_chains[0]["seed"].get("title", "?")
                tier_counts[g["tier"]] += 1
                depth_counts[1] += 1
            all_groups.extend(groups)

    # Find sibling groups per seed (depth-2: shared hop-1 paper)
    for sid, seed_chains in sorted(by_seed_d2.items()):
        groups = find_depth2_sibling_groups(seed_chains, max_groups=args.max_per_seed)
        groups = filter_groups_by_content(groups, contents)
        if groups:
            if sid not in by_seed_d1 or not any(
                g["seed_paperId"] == sid for g in all_groups
            ):
                seeds_with_groups += 1
            for g in groups:
                g["seed_paperId"] = sid
                g["seed_title"] = seed_chains[0]["seed"].get("title", "?")
                tier_counts[g["tier"]] += 1
                depth_counts[2] += 1
            all_groups.extend(groups)

    # Specificity breakdown
    focused = sum(1 for g in all_groups if g.get("specificity", 99) <= 3)
    broad = sum(1 for g in all_groups if g.get("specificity", 99) >= 4)

    print(f"\nSibling groups found: {len(all_groups)} across {seeds_with_groups} seeds")
    print(f"  Depth-1: {depth_counts[1]}, Depth-2: {depth_counts[2]}")
    print(f"  Tier 1 (co-citation):  {tier_counts[1]}")
    print(f"  Tier 2 (comparison):   {tier_counts[2]}")
    print(f"  Tier 3 (methodology):  {tier_counts[3]}")
    print(f"  Focused (≤3 papers in ctx): {focused}")
    print(f"  Broad (4+ papers in ctx):   {broad}")

    # Save groups if requested
    if args.save_groups:
        out_dir = Path(QUESTIONS_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        groups_file = out_dir / "dfs_sibling_groups.json"
        # Serialize groups (strip large content for readability)
        serializable = []
        for g in all_groups:
            sg = {
                "tier": g["tier"],
                "reason": g["reason"],
                "specificity": g.get("specificity", 99),
                "seed_paperId": g["seed_paperId"],
                "seed_title": g["seed_title"],
                "shared_contexts": g.get("shared_contexts", []),
                "targets": [
                    {
                        "arxivId": c["terminal"].get("arxivId"),
                        "title": c["terminal"].get("title"),
                        "context_types": c.get("metadata", {}).get("context_types", []),
                        "screen_score": c.get("screen", {}).get("score"),
                    }
                    for c in g["chains"]
                ],
            }
            serializable.append(sg)
        groups_file.write_text(json.dumps(serializable, indent=2, ensure_ascii=False))
        print(f"Saved sibling groups to: {groups_file}")

    if args.limit == 0:
        print("--limit 0: grouping only, exiting.")
        return

    # One-per-seed: keep only the best group per seed
    if args.one_per_seed:
        best_per_seed = {}
        for g in all_groups:
            sid = g["seed_paperId"]
            if sid not in best_per_seed:
                best_per_seed[sid] = g
        all_groups = list(best_per_seed.values())
        if args.limit:
            all_groups = all_groups[:args.limit]
        print(f"One-per-seed: {len(all_groups)} groups")

    # Apply limit with round-robin across seeds for diversity
    elif args.limit:
        by_seed_groups = defaultdict(list)
        for g in all_groups:
            by_seed_groups[g["seed_paperId"]].append(g)

        selected = []
        round_num = 0
        while len(selected) < args.limit:
            added = False
            for sid in list(by_seed_groups.keys()):
                if round_num < len(by_seed_groups[sid]):
                    selected.append(by_seed_groups[sid][round_num])
                    added = True
                    if len(selected) >= args.limit:
                        break
            if not added:
                break
            round_num += 1

        all_groups = selected
        n_seeds = len(set(g["seed_paperId"] for g in all_groups))
        print(f"Limited to {len(all_groups)} groups from {n_seeds} seeds (round-robin)")

    # Skip already-processed seeds (incremental mode)
    if done_seeds:
        before = len(all_groups)
        all_groups = [g for g in all_groups
                      if g["seed_paperId"] not in done_seeds]
        print(f"Skipping {before - len(all_groups)} already-processed seeds, "
              f"{len(all_groups)} new groups to process")

    if not all_groups:
        if existing_output:
            print("All groups already processed. Nothing to do.")
        else:
            print("No groups to process. Exiting.")
        return

    # Assign question modes
    modes = assign_dfs_modes(all_groups, seed=args.seed)
    rel_dist = Counter(r for r, s in modes)
    skill_dist = Counter(s for r, s in modes)
    print(f"\nMode assignments: {len(modes)} total")
    print(f"  Relationships: {dict(sorted(rel_dist.items()))}")
    print(f"  Skills: {dict(sorted(skill_dist.items()))}")

    # Build prompts
    prompts = []
    group_refs = []
    for group, mode in zip(all_groups, modes):
        prompt = build_dfs_prompt(group, contents, mode=mode)
        prompts.append(prompt)
        group_refs.append(group)

    if args.dry_run:
        for i, (prompt, group, mode) in enumerate(zip(prompts, group_refs, modes)):
            rel, skill = mode
            seed_title = group["seed_title"][:50]
            targets = " + ".join(
                c["terminal"]["title"][:30] for c in group["chains"]
            )
            print(f"\n{'='*70}")
            print(f"Group {i+1} (tier {group['tier']}): {seed_title}")
            print(f"  Mode: {rel} × {skill}")
            print(f"  Targets: {targets}")
            print(f"  Prompt length: {len(prompt)} chars (~{len(prompt)//4} tokens)")
            print(f"{'='*70}")
            print(prompt[:2000])
            if len(prompt) > 2000:
                print(f"... ({len(prompt) - 2000} more chars)")
        return

    # Generate questions via LLM
    print(f"\nGenerating {len(prompts)} DFS questions via {args.provider}...")
    results = llm_batch(
        prompts,
        SYSTEM_PROMPT,
        provider=args.provider,
        model=args.model,
        max_tokens=args.max_tokens,
        use_batch_api=args.batch_api,
    )

    # Merge results with group metadata
    output = []
    errors = 0

    for i, (qa, group, mode) in enumerate(zip(results, group_refs, modes)):
        rel, skill = mode
        seed_title = group["seed_title"][:50]
        targets = [c["terminal"]["title"][:40] for c in group["chains"]]

        # Check for errors
        if qa.get("score") == 0 and "error" in str(qa.get("reasoning", "")):
            print(f"  [{i+1}] ERROR: {seed_title}")
            errors += 1
            continue

        # Skip null answers
        if not qa.get("question") or not qa.get("answer"):
            print(f"  [{i+1}] NULL: {seed_title} (no valid synthesis question)")
            errors += 1
            continue

        entry = {
            "id": f"dfs_{len(existing_output) + len(output):04d}",
            "question_type": "dfs",
            "depth": group.get("dfs_depth", 1),
            "question_mode": {"relationship": rel, "skill": skill},
            "seed_understanding": qa.get("seed_understanding", ""),
            "seed_detail": qa.get("seed_detail", ""),
            "question": qa.get("question", ""),
            "answer": qa.get("answer", ""),
            "answer_sources": qa.get("answer_sources", {}),
            "synthesis_type": rel,
            "why_multi_paper": qa.get("why_multi_paper", ""),
            "group": {
                "tier": group["tier"],
                "reason": group["reason"],
                "dfs_depth": group.get("dfs_depth", 1),
                "seed": {
                    "paperId": group["seed_paperId"],
                    "title": group["seed_title"],
                },
                "hop1": (
                    {
                        "paperId": group["hop1"].get("paperId"),
                        "arxivId": group["hop1"].get("arxivId"),
                        "title": group["hop1"].get("title"),
                    }
                    if group.get("hop1") else None
                ),
                "targets": [
                    {
                        "paperId": c["terminal"].get("paperId"),
                        "arxivId": c["terminal"].get("arxivId"),
                        "title": c["terminal"].get("title"),
                    }
                    for c in group["chains"]
                ],
                "shared_contexts": group.get("shared_contexts", []),
            },
        }
        output.append(entry)

        # Print preview
        q_preview = (qa.get("question") or "")[:120]
        a_preview = (qa.get("answer") or "(null)")[:80]
        print(f"  [{i+1}] tier={group['tier']} {rel}×{skill} | {seed_title}")
        print(f"       Targets: {' + '.join(targets)}")
        print(f"       Q: {q_preview}")
        print(f"       A: {a_preview}")

    # Save (append to existing)
    combined = existing_output + output
    out_file.write_text(json.dumps(combined, indent=2, ensure_ascii=False))

    print(f"\n{'='*70}")
    print(f"New: {len(output)} DFS questions ({errors} errors/skipped)")
    if existing_output:
        print(f"Existing: {len(existing_output)} (kept)")
    print(f"Total: {len(combined)} DFS questions")
    print(f"Tier breakdown: {dict(Counter(q['group']['tier'] for q in combined))}")
    print(f"Relationships: {dict(Counter(q['question_mode']['relationship'] for q in combined))}")
    print(f"Skills: {dict(Counter(q['question_mode']['skill'] for q in combined))}")
    print(f"Saved to: {out_file}")


if __name__ == "__main__":
    main()
