"""Filter citation chains by intellectual depth.

Sits between extract_chains.py and generate_questions.py.
Reads chains_*.json, applies three filtering layers, writes filtered_chains_*.json.

Layer 1: Shared reference overlap (Jaccard on reference lists — hits S2 cache)
Layer 2: Citation context classification (keyword pattern matching)
Layer 3: LLM pre-screening (cheap model scores intellectual dependency 1-5)

Usage:
    python filter_chains.py --chains data/chains/chains_xxx.json
    python filter_chains.py --chains data/chains/chains_xxx.json --no-llm
    python filter_chains.py --chains data/chains/chains_xxx.json --dry-run
"""

import json
import re
import sys
import argparse
from pathlib import Path

from config import CHAINS_DIR, FILTER_MIN_JACCARD, FILTER_LLM_THRESHOLD
from s2_client import S2Client


# ── Layer 2: Context classification patterns ─────────────────────────────────

CONTEXT_PATTERNS = {
    "limitation_resolution": [
        r"improv(?:es?|ing)",
        r"address(?:es|ing)?",
        r"limitation",
        r"suffers?\s+from",
        r"overcome",
        r"mitigat(?:es?|ing)",
        r"shortcoming",
        r"drawback",
    ],
    "comparison": [
        r"in\s+contrast",
        r"unlike",
        r"whereas",
        r"different\s+from",
        r"alternatively",
        r"compar(?:ed?|ing|ison)",
        r"outperform",
        r"superior",
        r"inferior",
    ],
    "evolution": [
        r"extends?",
        r"builds?\s+(?:on|upon)",
        r"modif(?:ies|ying|ied)",
        r"adapts?",
        r"generaliz(?:es?|ing)",
        r"refin(?:es?|ing)",
        r"augment",
    ],
    "provenance": [
        r"originally\s+proposed",
        r"first\s+introduced",
        r"adapted\s+from",
        r"inspired\s+by",
        r"proposed\s+(?:by|in)",
        r"introduced\s+(?:by|in)",
        r"pioneer",
    ],
    "evidence_assessment": [
        r"contradicts?",
        r"inconsistent\s+with",
        r"challeng(?:es?|ing)",
        r"refut(?:es?|ing)",
        r"conflicts?\s+with",
        r"calls?\s+into\s+question",
        r"dampens?",
    ],
    "data_reuse": [
        r"evaluated?\s+on",
        r"dataset\s+from",
        r"benchmark(?:ed|ing)?",
        r"using\s+the\s+data",
        r"test(?:ed|ing)?\s+on",
        r"trained?\s+on",
        r"fine.?tun(?:ed|ing)\s+on",
    ],
    "background_cite": [
        r"as\s+shown\s+in",
        r"demonstrated\s+that",
        r"found\s+that",
        r"shown?\s+(?:by|in|that)",
        r"noted\s+(?:by|in|that)",
        r"reported\s+(?:by|in|that)",
        r"observed\s+(?:by|in|that)",
    ],
}

# Which types suggest high question potential
HIGH_POTENTIAL_TYPES = {
    "limitation_resolution",
    "comparison",
    "evolution",
    "provenance",
    "evidence_assessment",
}
LOW_POTENTIAL_TYPES = {"data_reuse", "background_cite"}

# Compile patterns once
_COMPILED_PATTERNS = {
    ctype: [re.compile(p, re.IGNORECASE) for p in patterns]
    for ctype, patterns in CONTEXT_PATTERNS.items()
}


# ── Layer 3: LLM pre-screening prompt ────────────────────────────────────────
# Split into system (cached) + user (variable) for Anthropic prompt caching.

LLM_PRESCREEN_SYSTEM = """You are screening citation chains for a research QA benchmark.
A good question requires: (1) a specific curiosity gap motivating each hop,
(2) a specific factual answer (number, metric, finding) in the target paper,
(3) an answer that is NOT guessable from the seed paper alone.

For EACH pair, answer these screening questions:

**Pair 1 (1-hop): Seed → Hop1**
- Curiosity gap: Does the citation context reveal a SPECIFIC reason a reader of the seed would seek out hop1? (not just "see also" or a group citation)
- Factual answer: Does hop1's abstract mention specific results, numbers, metrics, or concrete findings that could serve as an answer?
- Non-obvious: Would the answer require actually reading hop1, or is it already stated/implied in the seed?

**Pair 2 (2-hop): Seed → Hop1 → Hop2**
- Coherent narrative: Do the two hops form a logical research trajectory? (not two unrelated aspects of hop1)
- Curiosity gap at hop2: Does the hop1→hop2 citation context suggest a specific follow-up question, not just background?
- Factual answer: Does hop2's abstract mention specific results, numbers, or findings?
- Non-obvious: Would the answer require following BOTH hops from the seed?

Score each pair 1-5:
1 = no viable question (group citation, no specific facts, or answer is obvious)
2 = weak (vague connection, abstract lacks specific findings)
3 = possible but generic (connection exists but question would be shallow)
4 = good (specific curiosity gap + abstract suggests concrete answerable facts)
5 = excellent (clear research narrative + specific metrics/findings in abstract)

Respond as JSON only:
{
  "pair1_score": N,
  "pair1_curiosity_gap": "what specific question would a seed reader have?",
  "pair1_candidate_fact": "a specific fact/number from hop1 abstract, or null if none",
  "pair2_score": N,
  "pair2_narrative": "one sentence describing the 2-hop research trajectory",
  "pair2_candidate_fact": "a specific fact/number from hop2 abstract, or null if none",
  "best_pair": 1 or 2,
  "reasoning": "one sentence on which pair is stronger and why"
}"""

LLM_PRESCREEN_USER = """=== SEED PAPER ===
Title: {seed_title} ({seed_year})
Abstract: {seed_abstract}

=== HOP 1 (seed cites this) ===
Title: {hop1_title} ({hop1_year})
Abstract: {hop1_abstract}
How seed cites hop1: {seed_hop1_contexts}

=== HOP 2 (hop1 cites this) ===
Title: {hop2_title} ({hop2_year})
Abstract: {hop2_abstract}
How hop1 cites hop2: {hop1_hop2_contexts}

Evaluate both pairs and respond as JSON only."""


# ── Layer 1: Reference overlap ───────────────────────────────────────────────

def compute_reference_overlap(client: S2Client, paper_id_a: str, paper_id_b: str) -> float:
    """Compute Jaccard similarity between reference lists of two papers.

    Uses S2 get_references which will hit cache if extract_chains.py already fetched them.
    Returns float in [0, 1].
    """
    refs_a = client.get_references(paper_id_a)
    refs_b = client.get_references(paper_id_b)

    ids_a = {
        ref.get("citedPaper", {}).get("paperId")
        for ref in refs_a
        if ref.get("citedPaper", {}).get("paperId")
    }
    ids_b = {
        ref.get("citedPaper", {}).get("paperId")
        for ref in refs_b
        if ref.get("citedPaper", {}).get("paperId")
    }

    if not ids_a or not ids_b:
        return 0.0

    intersection = ids_a & ids_b
    union = ids_a | ids_b
    return len(intersection) / len(union)


# ── Layer 2: Context classification ──────────────────────────────────────────

def classify_contexts(contexts: list[str]) -> str:
    """Classify citation contexts into a relationship type using keyword patterns.

    Returns the type with the most pattern matches across all contexts.
    Falls back to "unclassified" if no patterns match.
    """
    if not contexts:
        return "unclassified"

    combined = " ".join(contexts)
    scores = {}

    for ctype, patterns in _COMPILED_PATTERNS.items():
        count = sum(1 for p in patterns if p.search(combined))
        if count > 0:
            scores[ctype] = count

    if not scores:
        return "unclassified"

    # Return highest-scoring type
    return max(scores, key=scores.get)


# ── Layer 3: LLM pre-screening ──────────────────────────────────────────────

def _build_user_prompt(chain: dict) -> str:
    """Build the variable part of the LLM prompt from chain data."""
    seed = chain["seed"]
    hop1 = chain["hop1"]
    hop2 = chain["hop2"]
    edge1 = chain["edge_seed_hop1"]
    edge2 = chain["edge_hop1_hop2"]

    def _fmt_contexts(edge):
        ctxs = edge.get("contexts", [])[:3]
        if not ctxs:
            return "(no citation context available)"
        return "\n".join(f'  [{i+1}] "{c}"' for i, c in enumerate(ctxs))

    return LLM_PRESCREEN_USER.format(
        seed_title=seed.get("title", "Unknown"),
        seed_year=seed.get("year", "?"),
        seed_abstract=(seed.get("abstract") or "(no abstract)")[:400],
        hop1_title=hop1.get("title", "Unknown"),
        hop1_year=hop1.get("year", "?"),
        hop1_abstract=(hop1.get("abstract") or "(no abstract)")[:400],
        seed_hop1_contexts=_fmt_contexts(edge1),
        hop2_title=hop2.get("title", "Unknown"),
        hop2_year=hop2.get("year", "?"),
        hop2_abstract=(hop2.get("abstract") or "(no abstract)")[:400],
        hop1_hop2_contexts=_fmt_contexts(edge2),
    )


def _parse_llm_response(content: str) -> dict:
    """Extract JSON from LLM response text."""
    json_match = re.search(r'\{[\s\S]*\}', content)
    if json_match:
        return json.loads(json_match.group())
    return {"pair1_score": 0, "pair2_score": 0, "best_pair": 0,
            "reasoning": f"parse_error: {content[:200]}"}


def llm_prescreen_batch(
    chains: list[dict],
    provider: str = "anthropic",
    model: str = None,
    max_workers: int = 4,
) -> list[dict]:
    """Score multiple chains concurrently.

    Returns list of LLM result dicts in the same order as input chains.
    Uses prompt caching (Anthropic) so the system prompt is only billed once.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    prompts = [_build_user_prompt(c) for c in chains]

    if provider == "anthropic":
        model = model or "claude-haiku-4-5-20251001"
        results = _batch_anthropic(prompts, model, max_workers)
    elif provider == "openai":
        model = model or "gpt-4o-mini"
        results = _batch_openai(prompts, model, max_workers)
    else:
        results = [{"pair1_score": 0, "pair2_score": 0, "reasoning": f"Unknown provider: {provider}"}] * len(chains)

    return results


def _batch_anthropic(prompts: list[str], model: str, max_workers: int) -> list[dict]:
    from anthropic import Anthropic
    client = Anthropic()

    # System prompt with cache_control for prompt caching
    system_msg = [{
        "type": "text",
        "text": LLM_PRESCREEN_SYSTEM,
        "cache_control": {"type": "ephemeral"},
    }]

    results = [None] * len(prompts)

    def _call(idx, user_prompt):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=600,
                system=system_msg,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.0,
            )
            return idx, _parse_llm_response(resp.content[0].text)
        except Exception as e:
            return idx, {"pair1_score": 0, "pair2_score": 0, "best_pair": 0,
                         "reasoning": f"error: {e}"}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_call, i, p) for i, p in enumerate(prompts)]
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result

    return results


def _batch_openai(prompts: list[str], model: str, max_workers: int) -> list[dict]:
    from openai import OpenAI
    client = OpenAI()

    results = [None] * len(prompts)

    def _call(idx, user_prompt):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": LLM_PRESCREEN_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return idx, json.loads(resp.choices[0].message.content)
        except Exception as e:
            return idx, {"pair1_score": 0, "pair2_score": 0, "best_pair": 0,
                         "reasoning": f"error: {e}"}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_call, i, p) for i, p in enumerate(prompts)]
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result

    return results


# ── Main filter logic ────────────────────────────────────────────────────────

def filter_chains(
    chains_file: str,
    no_llm: bool = False,
    dry_run: bool = False,
    provider: str = "anthropic",
    model: str = None,
) -> list[dict]:
    """Apply three filtering layers to chains. Returns filtered chains with metadata."""

    chains = json.loads(Path(chains_file).read_text())
    print(f"Loaded {len(chains)} chains from {chains_file}")

    client = S2Client()

    # Backfill missing abstracts — get_references often drops them due to S2 500s
    _abstract_cache = {}
    papers_to_fetch = set()
    for chain in chains:
        for node in ["seed", "hop1", "hop2"]:
            if not chain[node].get("abstract") and chain[node].get("paperId"):
                papers_to_fetch.add(chain[node]["paperId"])

    if papers_to_fetch:
        print(f"Fetching abstracts for {len(papers_to_fetch)} papers...")
        for pid in papers_to_fetch:
            paper = client.get_paper(pid)
            if paper.get("abstract"):
                _abstract_cache[pid] = paper["abstract"]
        print(f"  Got {len(_abstract_cache)}/{len(papers_to_fetch)} abstracts")

        # Write abstracts back into chains
        for chain in chains:
            for node in ["seed", "hop1", "hop2"]:
                if not chain[node].get("abstract"):
                    chain[node]["abstract"] = _abstract_cache.get(chain[node]["paperId"])

    results = []
    llm_candidates = []  # (index_in_chains, chain, filter_info) for L3 batch

    # ── Pass 1: L1 + L2 (no API calls beyond cached S2) ──
    print("\n--- Layer 1-2: Reference overlap + Context classification ---")
    for i, chain in enumerate(chains):
        seed_title = chain["seed"]["title"][:50]
        hop1_title = chain["hop1"]["title"][:50]
        hop2_title = chain["hop2"]["title"][:50]
        print(f"\n[{i+1}/{len(chains)}] {seed_title}")
        print(f"  → {hop1_title}")
        print(f"  → {hop2_title}")

        filter_info = {}
        signals_negative = 0

        # ── Layer 1: Reference overlap ──
        seed_id = chain["seed"]["paperId"]
        hop1_id = chain["hop1"]["paperId"]
        jaccard = compute_reference_overlap(client, seed_id, hop1_id)
        filter_info["ref_overlap_jaccard"] = round(jaccard, 4)

        if jaccard >= FILTER_MIN_JACCARD:
            print(f"  L1 ref overlap: {jaccard:.3f} ✓")
        else:
            print(f"  L1 ref overlap: {jaccard:.3f} (low)")
            signals_negative += 1

        # ── Layer 2: Context classification ──
        ctx_type_e1 = classify_contexts(chain["edge_seed_hop1"].get("contexts", []))
        ctx_type_e2 = classify_contexts(chain["edge_hop1_hop2"].get("contexts", []))
        filter_info["context_type_edge1"] = ctx_type_e1
        filter_info["context_type_edge2"] = ctx_type_e2

        e1_high = ctx_type_e1 in HIGH_POTENTIAL_TYPES
        e2_high = ctx_type_e2 in HIGH_POTENTIAL_TYPES
        both_low = (
            ctx_type_e1 in LOW_POTENTIAL_TYPES
            and ctx_type_e2 in LOW_POTENTIAL_TYPES
        )
        both_high = e1_high and e2_high

        if both_low:
            print(f"  L2 contexts: {ctx_type_e1} + {ctx_type_e2} → DROP (both low-potential)")
            filter_info["llm"] = None
            filter_info["llm_reasoning"] = "skipped — dropped by L2"
            filter_info["decision"] = "drop_l2"
            if dry_run:
                chain["filter"] = filter_info
                results.append(chain)
            continue
        elif both_high:
            print(f"  L2 contexts: {ctx_type_e1} + {ctx_type_e2} ✓ (both high-potential)")
        else:
            print(f"  L2 contexts: {ctx_type_e1} + {ctx_type_e2} (one high / mixed)")

        # ── Queue for L3 or resolve without LLM ──
        if no_llm:
            filter_info["llm"] = None
            filter_info["llm_reasoning"] = "skipped — --no-llm"
            if both_high and signals_negative == 0:
                filter_info["decision"] = "keep_no_llm"
                print(f"  → KEEP (both edges high + good overlap)")
            elif both_high:
                filter_info["decision"] = "keep_no_llm_marginal"
                print(f"  → KEEP marginal (both edges high but low overlap)")
            else:
                filter_info["decision"] = "drop_no_llm"
                print(f"  → DROP (needs LLM to confirm weak edge)")
                if not dry_run:
                    continue
            chain["filter"] = filter_info
            results.append(chain)
        elif dry_run:
            filter_info["llm"] = None
            filter_info["llm_reasoning"] = "skipped — --dry-run"
            filter_info["decision"] = "pending_llm"
            print(f"  → queued for LLM (dry-run)")
            chain["filter"] = filter_info
            results.append(chain)
        else:
            print(f"  → queued for LLM")
            llm_candidates.append((i, chain, filter_info))

    # ── Pass 2: Batch LLM scoring (concurrent) ──
    if llm_candidates:
        candidate_chains = [c for _, c, _ in llm_candidates]
        print(f"\n--- Layer 3: LLM screening ({len(llm_candidates)} chains, concurrent) ---")
        llm_results = llm_prescreen_batch(
            candidate_chains, provider=provider, model=model,
        )

        for (orig_idx, chain, filter_info), llm_result in zip(llm_candidates, llm_results):
            p1 = llm_result.get("pair1_score", 0)
            p2 = llm_result.get("pair2_score", 0)
            best = llm_result.get("best_pair", 0)
            best_score = p1 if best == 1 else p2
            filter_info["llm"] = llm_result
            filter_info["llm_reasoning"] = llm_result.get("reasoning", "")

            hop1_title = chain["hop1"]["title"][:45]
            hop2_title = chain["hop2"]["title"][:45]

            if best_score >= FILTER_LLM_THRESHOLD:
                filter_info["decision"] = f"keep_{best}hop"
                print(f"  [{orig_idx+1}] pair1={p1} pair2={p2} best={best}-hop ✓  {hop1_title}")
                print(f"    {llm_result.get('reasoning', '')[:100]}")
            else:
                filter_info["decision"] = "drop_l3"
                print(f"  [{orig_idx+1}] pair1={p1} pair2={p2} → DROP  {hop1_title}")
                print(f"    {llm_result.get('reasoning', '')[:100]}")
                if not dry_run:
                    continue

            chain["filter"] = filter_info
            results.append(chain)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Filter citation chains by intellectual depth"
    )
    parser.add_argument(
        "--chains", type=str,
        help="Path to chains JSON file (default: latest in data/chains/)",
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Skip Layer 3 LLM pre-screening (layers 1-2 only)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run all layers but don't drop chains (show what would happen)",
    )
    parser.add_argument(
        "--provider", choices=["anthropic", "openai"], default="anthropic",
        help="LLM provider for Layer 3 (default: anthropic)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model for Layer 3 (default: claude-haiku-4-5-20251001 / gpt-4o-mini)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output file path (default: filtered_chains_<id>.json in same dir)",
    )
    args = parser.parse_args()

    # Find chains file
    if args.chains:
        chains_file = args.chains
    else:
        chains_dir = Path(CHAINS_DIR)
        chain_files = sorted(chains_dir.glob("chains_*.json"))
        # Exclude files already filtered
        chain_files = [f for f in chain_files if not f.name.startswith("filtered_")]
        if chain_files:
            chains_file = str(chain_files[-1])
        else:
            print("No chains file found. Run extract_chains.py first.")
            sys.exit(1)

    print(f"Input:    {chains_file}")
    llm_label = "disabled" if args.no_llm else f"{args.provider} ({args.model or 'default'})"
    print(f"LLM:     {llm_label}")
    print(f"Dry run: {args.dry_run}")
    print()

    filtered = filter_chains(
        chains_file,
        no_llm=args.no_llm,
        dry_run=args.dry_run,
        provider=args.provider,
        model=args.model,
    )

    # Summary
    total = len(json.loads(Path(chains_file).read_text()))
    decisions = {}
    for c in filtered:
        d = c.get("filter", {}).get("decision", "unknown")
        decisions[d] = decisions.get(d, 0) + 1

    print(f"\n{'='*60}")
    print(f"Summary: {len(filtered)} / {total} chains {'kept' if not args.dry_run else 'in output'}")
    for d, count in sorted(decisions.items()):
        print(f"  {d}: {count}")

    # Jaccard distribution
    jaccards = [c["filter"]["ref_overlap_jaccard"] for c in filtered if "filter" in c]
    if jaccards:
        print(f"\nJaccard distribution: min={min(jaccards):.3f}  median={sorted(jaccards)[len(jaccards)//2]:.3f}  max={max(jaccards):.3f}")

    # Context type distribution
    types_e1 = [c["filter"]["context_type_edge1"] for c in filtered if "filter" in c]
    types_e2 = [c["filter"]["context_type_edge2"] for c in filtered if "filter" in c]
    all_types = types_e1 + types_e2
    type_counts = {}
    for t in all_types:
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"\nContext types (across both edges):")
    for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {count}")

    # LLM score distribution
    llm_chains = [c for c in filtered if c.get("filter", {}).get("llm") is not None]
    if llm_chains:
        p1_scores = [c["filter"]["llm"].get("pair1_score", 0) for c in llm_chains]
        p2_scores = [c["filter"]["llm"].get("pair2_score", 0) for c in llm_chains]
        print(f"\nLLM scores ({len(llm_chains)} chains evaluated):")
        print(f"  Pair1 (1-hop): min={min(p1_scores)} median={sorted(p1_scores)[len(p1_scores)//2]} max={max(p1_scores)}")
        print(f"  Pair2 (2-hop): min={min(p2_scores)} median={sorted(p2_scores)[len(p2_scores)//2]} max={max(p2_scores)}")
        best_counts = {"1-hop": 0, "2-hop": 0}
        for c in llm_chains:
            bp = c["filter"]["llm"].get("best_pair", 0)
            if bp == 1:
                best_counts["1-hop"] += 1
            elif bp == 2:
                best_counts["2-hop"] += 1
        print(f"  Best pair: {best_counts}")

    # Save output
    if args.output:
        out_file = Path(args.output)
    else:
        stem = Path(chains_file).stem  # e.g. "chains_1f3332443c19238a"
        chain_id = stem.replace("chains_", "")
        out_file = Path(chains_file).parent / f"filtered_chains_{chain_id}.json"

    out_file.write_text(json.dumps(filtered, indent=2, ensure_ascii=False))
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
