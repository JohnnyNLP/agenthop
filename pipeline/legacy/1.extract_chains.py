"""Extract citation chains from a seed paper using S2 API.

Given a seed paper, finds 2-hop citation chains:
  Seed → Hop1 (paper seed cites) → Hop2 (paper hop1 cites)

Uses relevance scoring to identify the most methodologically important
references, not just any cited paper. Out of 40-50 references, typically
only 3-5 are truly central to a paper's methodology.

Relevance signals (strongest to weakest):
  1. isInfluential (S2 classifier) — best single signal
  2. Number of citation contexts — more mentions = more central
  3. Intent classification — methodology > result > background
  4. Context length — longer contexts = more substantive discussion
"""

import json
import sys
from pathlib import Path

from config import (
    CHAINS_DIR,
    CHAIN_MIN_CONTEXT_LENGTH,
    CHAIN_CITED_YEAR_MIN,
)
from s2_client import S2Client


def score_reference(ref: dict, direction: str = "cited") -> float:
    """Score a reference edge by methodological relevance.

    Returns a relevance score (higher = more relevant).
    Typical distribution: 1-2 papers score >15, most score 1-5.
    """
    paper_key = "citedPaper" if direction == "cited" else "citingPaper"
    paper = ref.get(paper_key, {})

    if not paper.get("paperId") or not paper.get("title"):
        return -1

    score = 0.0

    # Signal 1: isInfluential (strongest — S2's own classifier)
    if ref.get("isInfluential"):
        score += 10

    # Signal 2: Number of citation contexts (more mentions = more central)
    contexts = [c for c in ref.get("contexts", []) if c]
    substantive = [c for c in contexts if len(c) >= CHAIN_MIN_CONTEXT_LENGTH]
    score += len(substantive) * 2  # 2 points per substantive context
    score += len(contexts) * 0.5   # 0.5 per any context

    # Signal 3: Intent classification
    intents = ref.get("intents", [])
    if "methodology" in intents:
        score += 5  # Core method reference
    if "result" in intents:
        score += 3  # Compared against in experiments
    # "background" alone gets 0 — these are filler citations

    # Signal 4: Context richness (longer = more substantive discussion)
    if substantive:
        avg_len = sum(len(c) for c in substantive) / len(substantive)
        if avg_len > 200:
            score += 2  # Long, detailed discussion
        elif avg_len > 120:
            score += 1

    return score


def get_arxiv_id(paper: dict) -> str:
    """Extract arxiv ID from S2 paper data."""
    ext = paper.get("externalIds") or {}
    return ext.get("ArXiv", "")


def extract_chains(
    seed_id: str,
    max_hop1: int = 10,
    max_hop2_per: int = 5,
    min_score_hop1: float = 3.0,
    min_score_hop2: float = 3.0,
):
    """Extract 2-hop chains from a seed paper.

    Only follows high-relevance citation edges. Out of ~40 references,
    typically 5-10 pass the relevance threshold at each hop.

    Args:
        seed_id: S2 paper ID or arxiv ID for the seed paper.
        max_hop1: Max number of hop-1 references to explore.
        max_hop2_per: Max hop-2 references per hop-1 paper.
        min_score_hop1: Minimum relevance score for hop-1 edges.
        min_score_hop2: Minimum relevance score for hop-2 edges.

    Returns:
        List of chain dicts, sorted by combined relevance score.
    """
    client = S2Client()

    # Get seed paper details
    print(f"[Seed] Fetching paper: {seed_id}")
    seed_paper = client.get_paper(seed_id)
    if not seed_paper.get("paperId"):
        print(f"Error: Could not find paper {seed_id}")
        return []

    print(f"  Title: {seed_paper.get('title')}")
    print(f"  Year: {seed_paper.get('year')}")
    print(f"  References: {seed_paper.get('referenceCount')}")

    # Get hop-1: papers the seed cites
    print(f"\n[Hop 1] Fetching references of seed paper...")
    refs = client.get_references(seed_id)
    print(f"  Total references: {len(refs)}")

    # Score and rank references
    scored_refs = []
    for ref in refs:
        s = score_reference(ref, "cited")
        cited = ref.get("citedPaper", {})
        year = cited.get("year")
        # Year filter (skip very old papers)
        if year and year < CHAIN_CITED_YEAR_MIN:
            continue
        if s >= min_score_hop1:
            scored_refs.append((s, ref))

    scored_refs.sort(key=lambda x: -x[0])
    print(f"  Above threshold (>={min_score_hop1}): {len(scored_refs)} references")

    # Show scoring breakdown
    print(f"\n  Relevance ranking:")
    for s, ref in scored_refs[:15]:
        cited = ref.get("citedPaper", {})
        inf = "★" if ref.get("isInfluential") else " "
        intents = ",".join(ref.get("intents", [])) or "?"
        n_ctx = len([c for c in ref.get("contexts", []) if c])
        print(f"    {inf} Score={s:5.1f} [{intents:12s}] ctx={n_ctx} | {cited.get('title', '?')[:70]}")

    # Take top references for hop-2 exploration
    top_refs = scored_refs[:max_hop1]

    chains = []
    for i, (score1, ref1) in enumerate(top_refs):
        hop1 = ref1["citedPaper"]
        hop1_id = hop1["paperId"]
        hop1_arxiv = get_arxiv_id(hop1)
        hop1_contexts = [c for c in ref1.get("contexts", []) if c and len(c) >= CHAIN_MIN_CONTEXT_LENGTH]

        print(f"\n[Hop 2] ({i+1}/{len(top_refs)}) Score={score1:.1f} | {hop1.get('title', '')[:70]}")
        print(f"  S2 ID: {hop1_id}, ArXiv: {hop1_arxiv or 'N/A'}")

        # Get hop-2: papers that hop-1 cites
        refs2 = client.get_references(hop1_id)

        scored_refs2 = []
        for ref2 in refs2:
            s2 = score_reference(ref2, "cited")
            cited2 = ref2.get("citedPaper", {})
            year2 = cited2.get("year")
            if year2 and year2 < CHAIN_CITED_YEAR_MIN:
                continue
            if s2 >= min_score_hop2:
                scored_refs2.append((s2, ref2))

        scored_refs2.sort(key=lambda x: -x[0])
        top_refs2 = scored_refs2[:max_hop2_per]

        print(f"  Hop1 refs: {len(refs2)} total, {len(scored_refs2)} above threshold, taking top {len(top_refs2)}")

        for score2, ref2 in top_refs2:
            hop2 = ref2["citedPaper"]
            hop2_id = hop2["paperId"]
            hop2_arxiv = get_arxiv_id(hop2)
            hop2_contexts = [c for c in ref2.get("contexts", []) if c and len(c) >= CHAIN_MIN_CONTEXT_LENGTH]

            chain = {
                "seed": {
                    "paperId": seed_paper["paperId"],
                    "arxivId": get_arxiv_id(seed_paper),
                    "title": seed_paper.get("title"),
                    "year": seed_paper.get("year"),
                },
                "hop1": {
                    "paperId": hop1_id,
                    "arxivId": hop1_arxiv,
                    "title": hop1.get("title"),
                    "year": hop1.get("year"),
                    "venue": hop1.get("venue"),
                    "citationCount": hop1.get("citationCount"),
                    "abstract": hop1.get("abstract"),
                },
                "hop2": {
                    "paperId": hop2_id,
                    "arxivId": hop2_arxiv,
                    "title": hop2.get("title"),
                    "year": hop2.get("year"),
                    "venue": hop2.get("venue"),
                    "citationCount": hop2.get("citationCount"),
                    "abstract": hop2.get("abstract"),
                },
                "edge_seed_hop1": {
                    "contexts": hop1_contexts,
                    "intents": ref1.get("intents", []),
                    "isInfluential": ref1.get("isInfluential", False),
                    "relevance_score": score1,
                },
                "edge_hop1_hop2": {
                    "contexts": hop2_contexts,
                    "intents": ref2.get("intents", []),
                    "isInfluential": ref2.get("isInfluential", False),
                    "relevance_score": score2,
                },
                "chain_score": score1 + score2,  # Combined relevance
            }
            chains.append(chain)

    # Sort chains by combined score
    chains.sort(key=lambda c: -c["chain_score"])

    print(f"\n[Done] Extracted {len(chains)} 2-hop chains")
    return chains


def main():
    seed_id = sys.argv[1] if len(sys.argv) > 1 else "1f3332443c19238a2799f9f9c7878c97b76b4ad2"  # MIRAGE S2 ID
    max_hop1 = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    max_hop2 = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    chains = extract_chains(seed_id, max_hop1=max_hop1, max_hop2_per=max_hop2)

    # Save chains
    out_dir = Path(CHAINS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_id = seed_id[:16]  # Truncate long S2 IDs
    out_file = out_dir / f"chains_{safe_id}.json"
    out_file.write_text(json.dumps(chains, indent=2, ensure_ascii=False))
    print(f"\nSaved {len(chains)} chains to {out_file}")

    # Print summary
    hop1_papers = {c["hop1"]["paperId"] for c in chains}
    hop2_papers = {c["hop2"]["paperId"] for c in chains}

    print(f"\nSummary:")
    print(f"  Unique hop-1 papers: {len(hop1_papers)}")
    print(f"  Unique hop-2 papers: {len(hop2_papers)}")

    # Show top chains by combined relevance
    print(f"\nTop chains (by combined relevance score):")
    for chain in chains[:5]:
        s1 = chain["edge_seed_hop1"]["relevance_score"]
        s2 = chain["edge_hop1_hop2"]["relevance_score"]
        print(f"\n  [{chain['chain_score']:.1f} = {s1:.1f}+{s2:.1f}]")
        print(f"  {chain['seed']['title'][:60]}")
        print(f"    → {chain['hop1']['title'][:60]}")
        ctx1 = chain["edge_seed_hop1"]["contexts"]
        if ctx1:
            print(f"      \"{ctx1[0][:100]}...\"")
        print(f"    → {chain['hop2']['title'][:60]}")
        ctx2 = chain["edge_hop1_hop2"]["contexts"]
        if ctx2:
            print(f"      \"{ctx2[0][:100]}...\"")


if __name__ == "__main__":
    main()
