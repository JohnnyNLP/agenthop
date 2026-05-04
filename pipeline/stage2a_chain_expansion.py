"""Stage 2 (chain collection — expansion half): build the depth-2 citation neighbourhood.

Expands every seed paper into a depth-2 citation neighbourhood that will
serve as the agent's per-query navigation pool. Semantic Scholar signals
(citation contexts, intents, isInfluential) handle candidate pre-selection
(top-k); an LLM evaluator then judges purely on abstracts + citation path
(no numeric metadata, to avoid anchoring bias).

Architecture:
  For each depth d = 1..max_depth:
    For each surviving parent from depth d-1:
      Fetch references → S2-signal top-k → backfill abstracts
      → LLM screen on abstracts + path → survivors become parents for d+1

  Candidates per hop: [10, 7] (configurable).
  Threshold: score >= 3.
  Deduplication: global seen set — each paper assigned to shallowest depth.

Usage:
    python 1.extract_and_filter.py <seed_id>
    python 1.extract_and_filter.py <seed_id> --max-depth 1
    python 1.extract_and_filter.py <seed_id> --candidates 10,7
    python 1.extract_and_filter.py <seed_id> --no-llm
    python 1.extract_and_filter.py <seed_id> --dry-run
"""

import json
import re
import sys
import argparse
from pathlib import Path
from config import (
    CHAINS_DIR,
    CHAIN_MIN_CONTEXT_LENGTH,
    CHAIN_CITED_YEAR_MIN,
    EXTRACT_CANDIDATES_PER_HOP,
    EXTRACT_MAX_DEPTH,
    FILTER_LLM_THRESHOLD,
)
from s2_client import S2Client

# Import reusable pieces from existing scripts (names start with digits,
# so we use spec_from_file_location instead of regular import)
import importlib.util as _ilu

def _load_module(name, path):
    spec = _ilu.spec_from_file_location(name, Path(__file__).parent / path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_extract = _load_module("extract_chains", "legacy/1.extract_chains.py")
score_reference = _extract.score_reference
get_arxiv_id = _extract.get_arxiv_id

_filter = _load_module("filter_chains", "legacy/2.filter_chains.py")
classify_contexts = _filter.classify_contexts
compute_reference_overlap = _filter.compute_reference_overlap


# ── LLM client ─────────────────────────────────────────────────────────────

from llm_client import llm_batch, parse_llm_json, error_result


# ── Unified LLM evaluator prompt ──────────────────────────────────────────

EVAL_SYSTEM = """You are screening candidate papers for a multi-hop research QA benchmark. Given a seed paper and a candidate paper reachable through citation links, evaluate whether the candidate is a good question target.

A good question target has:
1. A specific curiosity gap: a reader of the seed would have a natural reason to seek this paper
2. Specific factual content: the candidate's abstract mentions concrete results, numbers, metrics, or findings
3. Non-obviousness: the answer requires actually navigating to and reading the candidate paper

You will receive the seed paper's abstract, the candidate paper's abstract, and the citation path connecting them. Focus on the content of the abstracts and the research narrative implied by the path.

=== FEW-SHOT EXAMPLES ===

Example 1 — KEEP (score: 4)
Seed: "RAG with Knowledge Graphs for Customer Service QA" (2024)
Candidate: "UniKGQA: Unified Retrieval and Reasoning for Multi-hop KGQA" (2022)
Path: Seed → "Reasoning on Graphs" → Candidate
Evaluation: {"score": 4, "curiosity_gap": "Seed uses KG-augmented RAG; hop1 discusses KG reasoning for LLMs and explicitly states it improves on prior KGQA methods; candidate is the SOTA method being improved upon — a reader would want to know what the baseline approach was", "candidate_fact": "UniKGQA unifies retrieval and reasoning in both architecture and parameter learning for multi-hop KGQA", "reasoning": "The citation path shows a clear improvement relationship with specific technical claims. The candidate has concrete contributions that a seed reader would naturally want to understand."}

Example 2 — REJECT (score: 2)
Seed: "MIRAGE: Metric-Intensive Benchmark for RAG Evaluation" (2025)
Candidate: "How Much Knowledge Can You Pack into Parameters of a Language Model?" (2020)
Path: Seed → "RAG for Knowledge-Intensive NLP Tasks" → Candidate
Evaluation: {"score": 2, "curiosity_gap": "The connection is tangential — hop1 cites the candidate as a closed-book QA baseline for comparison, not as a follow-up research question", "candidate_fact": "Mentions that storing knowledge in parameters scales with model size, but no specific metrics in abstract", "reasoning": "The candidate is cited as a contrasting paradigm (retrieval vs parametric), not as part of a research trajectory a seed reader would naturally follow."}

Example 3 — BORDERLINE (score: 3)
Seed: "MIRAGE: Metric-Intensive Benchmark for RAG Evaluation" (2025)
Candidate: "Dense Passage Retrieval for Open-Domain Question Answering" (2020)
Path: Seed → "RAG for Knowledge-Intensive NLP Tasks" → Candidate
Evaluation: {"score": 3, "curiosity_gap": "RAG uses DPR as its retriever — a reader might want to understand the retrieval component, but this is more of a technical dependency than a research curiosity gap", "candidate_fact": "Dense retriever outperforms BM25 by 9-19% absolute in top-20 passage retrieval accuracy", "reasoning": "The candidate has specific metrics but the connection from the seed is about implementation detail (which retriever RAG uses) rather than a research question. Useful but generic."}

=== END EXAMPLES ===

Score 1-5:
1 = no viable question (group citation, no specific facts, or answer is obvious from seed)
2 = weak (vague connection or abstract lacks specific findings)
3 = possible but generic (connection exists but question would be shallow)
4 = good (specific curiosity gap + concrete answerable facts in candidate)
5 = excellent (clear research narrative + specific metrics/findings)

Respond as JSON only:
{
  "score": N,
  "curiosity_gap": "what specific question would a seed reader have about this paper?",
  "candidate_fact": "a specific fact/number from the candidate's abstract, or null",
  "reasoning": "one sentence explaining your score based on the abstracts and path"
}"""

EVAL_USER = """=== SEED PAPER ===
Title: {seed_title} ({seed_year})
Abstract: {seed_abstract}

=== CANDIDATE PAPER ===
Title: {candidate_title} ({candidate_year})
Abstract: {candidate_abstract}

=== CITATION PATH ({depth_label}) ===
{path_description}

Evaluate and respond as JSON only."""






# ── Abstract backfill ────────────────────────────────────────────────────────

def _backfill_abstracts(client: S2Client, papers: list[dict]) -> int:
    """Fetch abstracts for papers missing them. Mutates in place. Returns count."""
    to_fetch = [p for p in papers if not p.get("abstract") and p.get("paperId")]
    if not to_fetch:
        return 0

    fetched = 0
    for p in to_fetch:
        data = client.get_paper(p["paperId"])
        if data.get("abstract"):
            p["abstract"] = data["abstract"]
            fetched += 1

    return fetched


# ── Candidate building ───────────────────────────────────────────────────────

def _build_path_description(path: list[dict]) -> str:
    """Build a human-readable path description for the LLM prompt."""
    if not path:
        return "(direct reference from seed)"

    parts = ["Seed"]
    for step in path:
        title = step["paper"].get("title", "?")[:60]
        edge = step.get("edge_from_parent", {})
        ctxs = edge.get("contexts", [])[:1]
        ctx_preview = f' — cited as: "{ctxs[0][:120]}..."' if ctxs else ""
        parts.append(f'→ "{title}"{ctx_preview}')
    return "\n".join(parts)


def _build_eval_prompt(seed: dict, candidate: dict, path: list[dict],
                       depth: int) -> str:
    """Build the user prompt for the unified evaluator."""
    depth_label = f"{depth} hop" if depth == 1 else f"{depth} hops"
    return EVAL_USER.format(
        seed_title=seed.get("title", "Unknown"),
        seed_year=seed.get("year", "?"),
        seed_abstract=(seed.get("abstract") or "(no abstract)")[:500],
        candidate_title=candidate.get("title", "Unknown"),
        candidate_year=candidate.get("year", "?"),
        candidate_abstract=(candidate.get("abstract") or "(no abstract)")[:500],
        depth_label=depth_label,
        path_description=_build_path_description(path),
    )


# ── Main orchestrator ────────────────────────────────────────────────────────

def extract_and_filter(
    seed_id: str,
    max_depth: int = EXTRACT_MAX_DEPTH,
    candidates_per_hop: list[int] = None,
    min_s2_score: float = 3.0,
    no_llm: bool = False,
    dry_run: bool = False,
    provider: str = "openai",
    model: str = None,
) -> list[dict]:
    """Extract and filter N-hop chains from a seed paper.

    Args:
        seed_id: S2 paper ID or arxiv ID for the seed paper.
        max_depth: Maximum chain depth (default: 2).
        candidates_per_hop: Number of candidates to screen at each depth.
            Defaults to EXTRACT_CANDIDATES_PER_HOP.
        min_s2_score: Minimum S2 signal score for candidate selection.
        no_llm: Skip LLM screening (S2 signals only).
        dry_run: Show candidates at each depth without LLM screening.
        provider: LLM provider ("anthropic" or "openai").
        model: LLM model override.

    Returns:
        List of chain dicts with depth, path, screen, and metadata fields.
    """
    if candidates_per_hop is None:
        candidates_per_hop = list(EXTRACT_CANDIDATES_PER_HOP)

    # Pad if max_depth exceeds the configured list
    while len(candidates_per_hop) < max_depth:
        candidates_per_hop.append(candidates_per_hop[-1])

    client = S2Client()

    # ── Fetch seed paper ──
    print(f"[Seed] Fetching paper: {seed_id}")
    seed_paper = client.get_paper(seed_id)
    if not seed_paper.get("paperId"):
        print(f"Error: Could not find paper {seed_id}")
        return []

    seed = {
        "paperId": seed_paper["paperId"],
        "arxivId": get_arxiv_id(seed_paper),
        "title": seed_paper.get("title"),
        "year": seed_paper.get("year"),
        "abstract": seed_paper.get("abstract"),
    }
    print(f"  Title: {seed['title']}")
    print(f"  Year: {seed['year']}")

    # Backfill seed abstract if missing
    _backfill_abstracts(client, [seed])

    # ── Iterative depth expansion ──
    # Each "parent" is a dict with: paper, path (list of steps from seed), paperId
    initial_parent = {
        "paper": seed,
        "path": [],  # empty for seed
        "paperId": seed["paperId"],
    }
    current_parents = [initial_parent]
    seen_papers = {seed["paperId"]}  # global dedup set
    all_chains = []  # accumulated results across all depths
    screening_log = []  # all candidates with LLM results (including rejected)

    for depth in range(1, max_depth + 1):
        k = candidates_per_hop[depth - 1]
        # Threshold: score >= 3 to survive screening
        threshold = 3

        print(f"\n{'='*60}")
        print(f"[Depth {depth}] Exploring {len(current_parents)} parents, top {k} candidates each (threshold={threshold})")
        print(f"{'='*60}")

        # Collect all candidates at this depth
        candidates = []  # list of dicts with all info needed for screening
        depth_seen = set()  # dedup within this depth (first-parent-wins)

        for pi, parent in enumerate(current_parents):
            parent_paper = parent["paper"]
            parent_id = parent["paperId"]
            parent_title = parent_paper.get("title", "?")[:55]
            print(f"\n  [{pi+1}/{len(current_parents)}] {parent_title}")

            # Fetch parent's references
            refs = client.get_references(parent_id)

            # Score and rank
            scored_refs = []
            for ref in refs:
                s = score_reference(ref, "cited")
                cited = ref.get("citedPaper", {})
                year = cited.get("year")
                if year and year < CHAIN_CITED_YEAR_MIN:
                    continue
                cited_id = cited.get("paperId")
                if not cited_id or cited_id in seen_papers or cited_id in depth_seen:
                    continue  # dedup: global + within-depth
                # Skip non-English papers (CJK titles leak in via citation graph)
                title = cited.get("title") or ""
                if re.search(r'[\u3000-\u9fff\uac00-\ud7af]', title):
                    continue
                if s >= min_s2_score:
                    scored_refs.append((s, ref))

            # Fallback: if S2 metadata is absent (no contexts/intents for any ref),
            # rank by citation count instead of S2 signals
            if not scored_refs:
                fallback_refs = []
                for ref in refs:
                    cited = ref.get("citedPaper", {})
                    year = cited.get("year")
                    if year and year < CHAIN_CITED_YEAR_MIN:
                        continue
                    cited_id = cited.get("paperId")
                    if not cited_id or not cited.get("title"):
                        continue
                    if cited_id in seen_papers or cited_id in depth_seen:
                        continue
                    cc = cited.get("citationCount") or 0
                    fallback_refs.append((cc, ref))

                if fallback_refs:
                    fallback_refs.sort(key=lambda x: -x[0])
                    # Use citation count as the score (normalized to a comparable range)
                    scored_refs = [(min(r[0] / 100, 50.0), r[1]) for r in fallback_refs]
                    print(f"    ⚠ No S2 citation metadata — falling back to citation count ranking")

            scored_refs.sort(key=lambda x: -x[0])
            top_refs = scored_refs[:k]
            print(f"    Refs: {len(refs)} total, {len(scored_refs)} viable (after dedup+threshold), taking top {len(top_refs)}")

            # Show top candidates
            for s, ref in top_refs[:5]:
                cited = ref.get("citedPaper", {})
                inf = "★" if ref.get("isInfluential") else " "
                print(f"      {inf} S2={s:5.1f} | {cited.get('title', '?')[:60]}")
            if len(top_refs) > 5:
                print(f"      ... and {len(top_refs) - 5} more")

            # Backfill abstracts for candidates
            candidate_papers = [ref["citedPaper"] for _, ref in top_refs]
            n_backfilled = _backfill_abstracts(client, candidate_papers)
            if n_backfilled:
                print(f"    Backfilled {n_backfilled} abstracts")

            # Build candidate objects
            for s2_score, ref in top_refs:
                cited = ref["citedPaper"]
                cited_id = cited["paperId"]
                contexts = [c for c in ref.get("contexts", []) if c and len(c) >= CHAIN_MIN_CONTEXT_LENGTH]

                edge = {
                    "contexts": contexts,
                    "intents": ref.get("intents", []),
                    "isInfluential": ref.get("isInfluential", False),
                    "relevance_score": s2_score,
                }

                # L1: Jaccard (seed vs parent — how topically related is the parent to the seed)
                if depth == 1:
                    jaccard = 0.0  # seed IS the parent at depth 1, Jaccard with self is meaningless
                else:
                    jaccard = compute_reference_overlap(client, seed["paperId"], parent_id)

                # L2: Context type classification
                context_type = classify_contexts(contexts)

                # Build path: parent's path + this step
                step = {
                    "paper": {
                        "paperId": cited_id,
                        "arxivId": get_arxiv_id(cited),
                        "title": cited.get("title"),
                        "year": cited.get("year"),
                        "venue": cited.get("venue"),
                        "citationCount": cited.get("citationCount"),
                        "abstract": cited.get("abstract"),
                    },
                    "edge_from_parent": edge,
                }
                path = parent["path"] + [step]

                candidates.append({
                    "candidate_paper": step["paper"],
                    "candidate_id": cited_id,
                    "path": path,
                    "depth": depth,
                    "s2_score": s2_score,
                    "jaccard": jaccard,
                    "context_type": context_type,
                    "edge": edge,
                    "parent": parent,
                })
                depth_seen.add(cited_id)

        if not candidates:
            print(f"\n  No candidates at depth {depth}. Stopping.")
            break

        print(f"\n  Total candidates at depth {depth}: {len(candidates)}")

        # ── LLM screening ──
        if dry_run:
            print(f"  [Dry run] Skipping LLM. Showing {len(candidates)} candidates.")
            for c in candidates:
                title = c["candidate_paper"].get("title", "?")[:55]
                print(f"    S2={c['s2_score']:5.1f} J={c['jaccard']:.3f} ctx={c['context_type']:20s} | {title}")
            # In dry run, all candidates survive (no LLM gate)
            survivors = candidates
        elif no_llm:
            # Keep all based on S2 signals
            survivors = candidates
            print(f"  [No LLM] All {len(candidates)} candidates survive.")
        else:
            # Build prompts
            prompts = [
                _build_eval_prompt(
                    seed=seed,
                    candidate=c["candidate_paper"],
                    path=c["path"],
                    depth=c["depth"],
                )
                for c in candidates
            ]

            print(f"  Sending {len(prompts)} candidates to LLM...")
            llm_results = llm_batch(prompts, EVAL_SYSTEM, provider=provider, model=model)

            # Filter by threshold
            survivors = []
            for cand, llm_result in zip(candidates, llm_results):
                score = llm_result.get("score", 0)
                cand["screen"] = llm_result
                title = cand["candidate_paper"].get("title", "?")[:55]
                survived = score >= threshold

                if survived:
                    survivors.append(cand)
                    print(f"    ✓ score={score} S2={cand['s2_score']:5.1f} J={cand['jaccard']:.3f} ctx={cand['context_type']:15s} | {title}")
                else:
                    print(f"    ✗ score={score} S2={cand['s2_score']:5.1f} J={cand['jaccard']:.3f} ctx={cand['context_type']:15s} | {title}")

                # Log every candidate regardless of outcome
                screening_log.append({
                    "depth": cand["depth"],
                    "survived": survived,
                    "threshold": threshold,
                    "candidate": cand["candidate_paper"],
                    "s2_score": cand["s2_score"],
                    "jaccard": cand["jaccard"],
                    "context_type": cand["context_type"],
                    "screen": llm_result,
                })

            print(f"\n  Survivors: {len(survivors)} / {len(candidates)} (threshold={threshold})")

        # ── Record surviving chains ──
        for s in survivors:
            # Compute aggregate metadata
            s2_scores = [step["edge_from_parent"]["relevance_score"] for step in s["path"]]
            context_types = [classify_contexts(step["edge_from_parent"].get("contexts", []))
                            for step in s["path"]]

            chain = {
                "seed": seed,
                "terminal": s["candidate_paper"],
                "depth": s["depth"],
                "path": s["path"],
                "screen": s.get("screen"),
                "metadata": {
                    "jaccard_seed_parent": round(s["jaccard"], 4),
                    "context_types": context_types,
                    "s2_scores": s2_scores,
                    "chain_score": sum(s2_scores),
                },
            }
            all_chains.append(chain)

        # ── Mark all evaluated candidates as seen (prevents re-evaluation at deeper depths) ──
        for c in candidates:
            seen_papers.add(c["candidate_id"])

        # ── Survivors become parents for next depth ──
        next_parents = []
        for s in survivors:
            next_parents.append({
                "paper": s["candidate_paper"],
                "path": s["path"],
                "paperId": s["candidate_id"],
            })
        current_parents = next_parents

        if not current_parents:
            print(f"\n  No survivors at depth {depth}. Stopping.")
            break

    # Sort by chain_score descending
    all_chains.sort(key=lambda c: -c["metadata"]["chain_score"])

    print(f"\n{'='*60}")
    print(f"[Done] {len(all_chains)} chains across depths 1-{max_depth}")
    depth_counts = {}
    for c in all_chains:
        d = c["depth"]
        depth_counts[d] = depth_counts.get(d, 0) + 1
    for d in sorted(depth_counts):
        print(f"  Depth {d}: {depth_counts[d]} chains")

    return all_chains, screening_log


# ── Single-seed runner ────────────────────────────────────────────────────────

def run_single_seed(seed_id, max_depth=EXTRACT_MAX_DEPTH, candidates_per_hop=None,
                    no_llm=False, dry_run=False, provider="openai",
                    model=None, output=None):
    """Run extraction + filtering for one seed. Returns (chains, screening_log)."""
    chains, screening_log = extract_and_filter(
        seed_id=seed_id,
        max_depth=max_depth,
        candidates_per_hop=candidates_per_hop,
        no_llm=no_llm,
        dry_run=dry_run,
        provider=provider,
        model=model,
    )

    # ── Save output ──
    out_dir = Path(CHAINS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_id = seed_id[:16]
    depth_tag = f"d{max_depth}"

    # Save screening log (always, even with 0 chains)
    if screening_log:
        log_file = out_dir / f"screening_{safe_id}.json"
        log_file.write_text(json.dumps(screening_log, indent=2, ensure_ascii=False))
        print(f"\nSaved screening log ({len(screening_log)} candidates) to {log_file}")

    if not chains:
        print("No chains produced.")
        return chains, screening_log

    out_file = Path(output) if output else out_dir / f"chains_{depth_tag}_{safe_id}.json"
    out_file.write_text(json.dumps(chains, indent=2, ensure_ascii=False))
    print(f"\nSaved {len(chains)} chains to {out_file}")

    # ── Summary ──
    print(f"\nSummary:")
    depth_counts = {}
    for c in chains:
        d = c["depth"]
        depth_counts[d] = depth_counts.get(d, 0) + 1
    for d in sorted(depth_counts):
        print(f"  Depth {d}: {depth_counts[d]} chains")

    unique_terminals = {c["terminal"]["paperId"] for c in chains}
    print(f"  Unique terminal papers: {len(unique_terminals)}")

    # Score distribution
    llm_chains = [c for c in chains if c.get("screen")]
    if llm_chains:
        scores = [c["screen"].get("score", 0) for c in llm_chains]
        print(f"\n  LLM scores ({len(llm_chains)} screened chains):")
        print(f"    min={min(scores)} median={sorted(scores)[len(scores)//2]} max={max(scores)}")

    # Top chains
    print(f"\nTop chains:")
    for chain in chains[:8]:
        d = chain["depth"]
        cs = chain["metadata"]["chain_score"]
        terminal = chain["terminal"]["title"][:55]
        score = (chain.get("screen") or {}).get("score", "?")
        path_titles = " → ".join(
            step["paper"]["title"][:30] for step in chain["path"]
        )
        print(f"\n  [d={d} score={score} S2={cs:.1f}] {terminal}")
        print(f"    Path: Seed → {path_titles}")
        if (chain.get("screen") or {}).get("curiosity_gap"):
            print(f"    Gap: {chain['screen']['curiosity_gap'][:100]}")

    return chains, screening_log


# ── Batch mode: depth-level LLM batching ─────────────────────────────────────

def _run_batch(seeds_file, max_depth, candidates_per_hop, no_llm, dry_run,
               provider, model, limit, skip_existing, cph_display, llm_label,
               use_batch_api=False):
    """Batch processing with depth-level LLM batching.

    Instead of running LLM screening per seed, this collects all candidates
    at each depth across ALL seeds, then runs one large llm_batch call.
    This is much more efficient — especially with batch API.

    Flow:
      Phase 1: All seeds → S2 fetch depth-1 refs → collect prompts
               → ONE llm_batch → distribute survivors
      Phase 2: All depth-1 survivors → S2 fetch depth-2 refs → collect prompts
               → ONE llm_batch → distribute survivors
      Save: Write per-seed chain files
    """
    seeds = json.loads(Path(seeds_file).read_text())
    if limit:
        seeds = seeds[:limit]

    out_dir = Path(CHAINS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    depth_tag = f"d{max_depth}"

    if candidates_per_hop is None:
        candidates_per_hop = list(EXTRACT_CANDIDATES_PER_HOP)
    while len(candidates_per_hop) < max_depth:
        candidates_per_hop.append(candidates_per_hop[-1])

    print(f"{'='*60}")
    print(f"BATCH MODE (depth-level batching): {len(seeds)} seeds")
    print(f"Max depth:  {max_depth}")
    print(f"Candidates: {cph_display[:max_depth]}")
    print(f"LLM:        {llm_label}")
    print(f"{'='*60}\n")

    client = S2Client()
    threshold = 3

    # ── Filter seeds: skip existing, fetch seed papers ──
    active_seeds = []  # list of {seed_info, seed_paper}
    skipped = []
    for i, seed in enumerate(seeds):
        sid = seed["paperId"]
        title = (seed.get("title") or "?")[:55]
        safe_id = sid[:16]

        if skip_existing:
            existing = out_dir / f"chains_{depth_tag}_{safe_id}.json"
            if existing.exists():
                skipped.append(seed)
                continue

        active_seeds.append(seed)

    if skipped:
        print(f"  Skipping {len(skipped)} seeds with existing results")
    print(f"  Processing {len(active_seeds)} seeds\n")

    # ── Fetch seed papers ──
    print(f"[Phase 0] Fetching seed paper metadata...")
    seed_states = {}  # sid -> {seed, parents, seen, chains, screening_log}
    errors = []
    for i, seed in enumerate(active_seeds):
        sid = seed["paperId"]
        try:
            seed_paper = client.get_paper(sid)
            if not seed_paper.get("paperId"):
                errors.append({"seed": sid, "title": seed.get("title"),
                               "status": "error: paper not found"})
                continue
            seed_info = {
                "paperId": seed_paper["paperId"],
                "arxivId": get_arxiv_id(seed_paper),
                "title": seed_paper.get("title"),
                "year": seed_paper.get("year"),
                "abstract": seed_paper.get("abstract"),
            }
            _backfill_abstracts(client, [seed_info])
            seed_states[sid] = {
                "seed": seed_info,
                "parents": [{
                    "paper": seed_info,
                    "path": [],
                    "paperId": seed_info["paperId"],
                }],
                "seen": {seed_info["paperId"]},
                "chains": [],
                "screening_log": [],
            }
        except Exception as e:
            errors.append({"seed": sid, "title": seed.get("title"),
                           "status": f"error: {e}"})

        if (i + 1) % 50 == 0:
            print(f"  Fetched {i + 1}/{len(active_seeds)} seeds...")

    print(f"  Fetched {len(seed_states)} seed papers "
          f"({len(errors)} errors)\n")

    # ── Depth loop ──
    for depth in range(1, max_depth + 1):
        k = candidates_per_hop[depth - 1]

        print(f"{'='*60}")
        print(f"[Depth {depth}] Gathering candidates across {len(seed_states)} seeds (top {k} per parent)")
        print(f"{'='*60}")

        # Collect candidates across ALL seeds
        all_candidates = []  # list of (sid, candidate_dict)
        for sid, state in seed_states.items():
            seed_info = state["seed"]
            depth_seen = set()

            for parent in state["parents"]:
                parent_id = parent["paperId"]

                # Fetch references from S2
                try:
                    refs = client.get_references(parent_id)
                except Exception as e:
                    print(f"  [!] S2 error for {parent_id[:16]}: {e}")
                    continue

                # Score and rank
                scored_refs = []
                for ref in refs:
                    s = score_reference(ref, "cited")
                    cited = ref.get("citedPaper", {})
                    year = cited.get("year")
                    if year and year < CHAIN_CITED_YEAR_MIN:
                        continue
                    cited_id = cited.get("paperId")
                    if not cited_id or cited_id in state["seen"] or cited_id in depth_seen:
                        continue
                    title = cited.get("title") or ""
                    if re.search(r'[\u3000-\u9fff\uac00-\ud7af]', title):
                        continue
                    if s >= 3.0:
                        scored_refs.append((s, ref))

                # Fallback to citation count
                if not scored_refs:
                    fallback_refs = []
                    for ref in refs:
                        cited = ref.get("citedPaper", {})
                        year = cited.get("year")
                        if year and year < CHAIN_CITED_YEAR_MIN:
                            continue
                        cited_id = cited.get("paperId")
                        if not cited_id or not cited.get("title"):
                            continue
                        if cited_id in state["seen"] or cited_id in depth_seen:
                            continue
                        cc = cited.get("citationCount") or 0
                        fallback_refs.append((cc, ref))
                    if fallback_refs:
                        fallback_refs.sort(key=lambda x: -x[0])
                        scored_refs = [(min(r[0] / 100, 50.0), r[1])
                                       for r in fallback_refs]

                scored_refs.sort(key=lambda x: -x[0])
                top_refs = scored_refs[:k]

                # Backfill abstracts
                candidate_papers = [ref["citedPaper"] for _, ref in top_refs]
                _backfill_abstracts(client, candidate_papers)

                # Build candidate objects
                for s2_score, ref in top_refs:
                    cited = ref["citedPaper"]
                    cited_id = cited["paperId"]
                    contexts = [c for c in ref.get("contexts", [])
                                if c and len(c) >= CHAIN_MIN_CONTEXT_LENGTH]
                    edge = {
                        "contexts": contexts,
                        "intents": ref.get("intents", []),
                        "isInfluential": ref.get("isInfluential", False),
                        "relevance_score": s2_score,
                    }
                    if depth == 1:
                        jaccard = 0.0
                    else:
                        jaccard = compute_reference_overlap(
                            client, seed_info["paperId"], parent_id)
                    context_type = classify_contexts(contexts)

                    step = {
                        "paper": {
                            "paperId": cited_id,
                            "arxivId": get_arxiv_id(cited),
                            "title": cited.get("title"),
                            "year": cited.get("year"),
                            "venue": cited.get("venue"),
                            "citationCount": cited.get("citationCount"),
                            "abstract": cited.get("abstract"),
                        },
                        "edge_from_parent": edge,
                    }
                    path = parent["path"] + [step]

                    cand = {
                        "candidate_paper": step["paper"],
                        "candidate_id": cited_id,
                        "path": path,
                        "depth": depth,
                        "s2_score": s2_score,
                        "jaccard": jaccard,
                        "context_type": context_type,
                        "edge": edge,
                        "parent": parent,
                    }
                    all_candidates.append((sid, cand))
                    depth_seen.add(cited_id)

            # Mark depth-seen as globally seen for this seed
            state["seen"].update(depth_seen)

        print(f"  Collected {len(all_candidates)} candidates across all seeds")

        if not all_candidates:
            print(f"  No candidates at depth {depth}. Stopping.")
            break

        # ── LLM screening: ONE batch call for all candidates ──
        if dry_run or no_llm:
            mode = "dry-run" if dry_run else "no-llm"
            print(f"  [{mode}] All {len(all_candidates)} candidates survive.")
            for sid, cand in all_candidates:
                cand["screen"] = {"score": 3, "reasoning": mode}
        else:
            prompts = []
            for sid, cand in all_candidates:
                seed_info = seed_states[sid]["seed"]
                prompts.append(_build_eval_prompt(
                    seed=seed_info,
                    candidate=cand["candidate_paper"],
                    path=cand["path"],
                    depth=cand["depth"],
                ))

            mode_label = "batch-api" if use_batch_api else "realtime"
            print(f"  Sending {len(prompts)} prompts to LLM ({provider}, {mode_label})...")
            llm_results = llm_batch(prompts, EVAL_SYSTEM,
                                    provider=provider, model=model,
                                    use_batch_api=use_batch_api)

            for (sid, cand), result in zip(all_candidates, llm_results):
                cand["screen"] = result

        # ── Distribute results back to per-seed states ──
        depth_survivors = {sid: [] for sid in seed_states}
        score_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

        for sid, cand in all_candidates:
            score = cand["screen"].get("score", 0)
            score_counts[score] = score_counts.get(score, 0) + 1
            survived = score >= threshold

            # Log
            seed_states[sid]["screening_log"].append({
                "depth": cand["depth"],
                "survived": survived,
                "threshold": threshold,
                "candidate": cand["candidate_paper"],
                "s2_score": cand["s2_score"],
                "jaccard": cand["jaccard"],
                "context_type": cand["context_type"],
                "screen": cand["screen"],
            })

            if survived:
                # Record chain
                s2_scores = [step["edge_from_parent"]["relevance_score"]
                             for step in cand["path"]]
                context_types = [classify_contexts(
                    step["edge_from_parent"].get("contexts", []))
                    for step in cand["path"]]
                chain = {
                    "seed": seed_states[sid]["seed"],
                    "terminal": cand["candidate_paper"],
                    "depth": cand["depth"],
                    "path": cand["path"],
                    "screen": cand["screen"],
                    "metadata": {
                        "jaccard_seed_parent": round(cand["jaccard"], 4),
                        "context_types": context_types,
                        "s2_scores": s2_scores,
                        "chain_score": sum(s2_scores),
                    },
                }
                seed_states[sid]["chains"].append(chain)
                depth_survivors[sid].append({
                    "paper": cand["candidate_paper"],
                    "path": cand["path"],
                    "paperId": cand["candidate_id"],
                })

        total_survived = sum(len(v) for v in depth_survivors.values())
        print(f"  Score distribution: {dict(sorted(score_counts.items()))}")
        print(f"  Survivors: {total_survived}/{len(all_candidates)} "
              f"(threshold={threshold})")

        # Update parents for next depth
        for sid in seed_states:
            seed_states[sid]["parents"] = depth_survivors[sid]

    # ── Save per-seed results ──
    print(f"\n{'='*60}")
    print(f"Saving results...")
    print(f"{'='*60}")

    results = []
    for seed in active_seeds:
        sid = seed["paperId"]
        if sid not in seed_states:
            continue
        state = seed_states[sid]
        chains = state["chains"]
        chains.sort(key=lambda c: -c["metadata"]["chain_score"])
        safe_id = sid[:16]

        # Save screening log
        if state["screening_log"]:
            log_file = out_dir / f"screening_{safe_id}.json"
            log_file.write_text(json.dumps(
                state["screening_log"], indent=2, ensure_ascii=False))

        # Save chains
        if chains:
            out_file = out_dir / f"chains_{depth_tag}_{safe_id}.json"
            out_file.write_text(json.dumps(
                chains, indent=2, ensure_ascii=False))

        results.append({
            "seed": sid,
            "title": seed.get("title"),
            "status": "ok",
            "chains": len(chains),
        })

    # ── Batch summary ──
    total_chains = sum(r.get("chains", 0) for r in results)
    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE: {len(seeds)} seeds")
    print(f"{'='*60}")
    print(f"  Processed: {len(results)} ({total_chains} total chains)")
    if skipped:
        print(f"  Skipped:   {len(skipped)}")
    if errors:
        print(f"  Errors:    {len(errors)}")
        for r in errors:
            print(f"    {r['title'][:50]}: {r['status']}")

    # Save batch summary
    all_results = results + [
        {"seed": s["paperId"], "title": s.get("title"), "status": "skipped"}
        for s in skipped
    ] + errors
    summary_file = out_dir / "batch_summary.json"
    summary_file.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"\nBatch summary saved to {summary_file}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="N-hop chain extraction and filtering with unified LLM evaluator"
    )
    parser.add_argument(
        "seed_id", nargs="?", default=None,
        help="S2 paper ID or arxiv ID for the seed paper",
    )
    parser.add_argument(
        "--max-depth", type=int, default=EXTRACT_MAX_DEPTH,
        help=f"Maximum chain depth (default: {EXTRACT_MAX_DEPTH}, hard cap 2)",
    )
    parser.add_argument(
        "--candidates", type=str, default=None,
        help="Candidates per hop, comma-separated (default: 10,7)",
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Skip LLM screening (S2 signals only)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show candidates at each depth without LLM screening",
    )
    parser.add_argument(
        "--provider", choices=["anthropic", "openai", "deepseek"], default="openai",
        help="LLM provider (default: openai → gpt-4.1-mini)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="LLM model override",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output file path (default: auto-named in data/chains/)",
    )
    parser.add_argument(
        "--batch", type=str, default=None,
        help="Path to seeds JSON file for batch processing",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N seeds from the batch file",
    )
    parser.add_argument(
        "--skip-existing", action="store_true", default=True,
        help="Skip seeds that already have output files (default: True)",
    )
    parser.add_argument(
        "--no-skip-existing", action="store_false", dest="skip_existing",
        help="Re-process all seeds even if output files exist",
    )
    parser.add_argument(
        "--batch-api", action="store_true",
        help="Use batch API for 50%% cost reduction on LLM screening (OpenAI/Anthropic)",
    )
    args = parser.parse_args()

    # Enforce max depth cap
    if args.max_depth > EXTRACT_MAX_DEPTH:
        print(f"  [!] max_depth capped to {EXTRACT_MAX_DEPTH} (was {args.max_depth})")
        args.max_depth = EXTRACT_MAX_DEPTH

    # Parse candidates
    if args.candidates:
        candidates_per_hop = [int(x.strip()) for x in args.candidates.split(",")]
    else:
        candidates_per_hop = None

    cph_display = candidates_per_hop or EXTRACT_CANDIDATES_PER_HOP
    llm_label = "disabled" if args.no_llm else f"{args.provider} ({args.model or 'default'})"

    # ── Batch mode ──
    if args.batch:
        _run_batch(
            seeds_file=args.batch,
            max_depth=args.max_depth,
            candidates_per_hop=candidates_per_hop,
            no_llm=args.no_llm,
            dry_run=args.dry_run,
            provider=args.provider,
            model=args.model,
            limit=args.limit,
            skip_existing=args.skip_existing,
            cph_display=cph_display,
            llm_label=llm_label,
            use_batch_api=args.batch_api,
        )
        return

    # ── Single seed mode ──
    if not args.seed_id:
        parser.error("seed_id is required (or use --batch)")

    print(f"Seed:       {args.seed_id}")
    print(f"Max depth:  {args.max_depth}")
    print(f"Candidates: {cph_display[:args.max_depth]}")
    print(f"LLM:        {llm_label}")
    print(f"Dry run:    {args.dry_run}")
    print()

    run_single_seed(
        seed_id=args.seed_id,
        max_depth=args.max_depth,
        candidates_per_hop=candidates_per_hop,
        no_llm=args.no_llm,
        dry_run=args.dry_run,
        provider=args.provider,
        model=args.model,
        output=args.output,
    )


if __name__ == "__main__":
    main()
