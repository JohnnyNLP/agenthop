"""Load AgentHop benchmark from the JSONL release format.

Reconstructs the per-sample format expected by the evaluation harness
from the deduplicated release structure:
  - qa/full.jsonl              — QA metadata (question, options, types, etc.)
  - graphs/full.jsonl          — per-sample citation neighbourhoods
  - paper_pool/papers.jsonl    — deduplicated section text for the 7,205-paper pool
  - audit/recall_labels.jsonl  — auditor-labelled answer-bearing sections

Usage:
    from loader import load_agenthop, load_recall_labels

    samples = load_agenthop("path/to/AgentHop")
    labels  = load_recall_labels("path/to/AgentHop")

Each sample dict contains: id, question, options, correct_index, seed/gold/bridge IDs,
question_type, reasoning_type, depth, consensus_tier, distractor_types, plus the
per-sample citation graph and the section text of every paper reachable from the seed.
"""

import json
from pathlib import Path


def load_paper_pool(agenthop_dir: str | Path) -> dict:
    """Load deduplicated paper pool. Returns {arxiv_id: {title, sections, ...}}."""
    pool_path = Path(agenthop_dir) / "paper_pool" / "papers.jsonl"
    pool = {}
    with open(pool_path) as f:
        for line in f:
            paper = json.loads(line)
            pool[paper["arxiv_id"]] = paper
    return pool


def load_recall_labels(agenthop_dir: str | Path) -> dict:
    """Load auditor-labelled answer-bearing sections per sample.

    Returns {sample_id: {analysis: {section_recall_labels: [...], ...}}}.
    The harness uses ``section_recall_labels`` to compute the search-axis
    recall metric (whether the agent called read_section on a labelled section
    of every gold paper of the sample).
    """
    labels_path = Path(agenthop_dir) / "audit" / "recall_labels.jsonl"
    labels = {}
    with open(labels_path) as f:
        for line in f:
            row = json.loads(line)
            labels[row["sample_id"]] = row
    return labels


def load_agenthop(
    agenthop_dir: str | Path,
    limit: int | None = None,
) -> list[dict]:
    """Load AgentHop samples in the format expected by the evaluation harness.

    Merges QA metadata + graph + paper_pool into the per-sample dict:
        {
            "id": str,
            "question": str,
            "question_type": str,        # "single-target" or "multi-target"
            "reasoning_type": str,        # "GROUND" / "METHOD" / "MOTIVE" / "RESULT"
            "depth": int,                 # 1 or 2
            "options": list[str],         # 4 strings
            "correct_index": int,         # 0..3
            "seed_paper_id": str,         # Semantic Scholar hash
            "seed_arxiv_id": str,         # arXiv ID of the seed paper
            "gold_paper_ids": list[str],
            "gold_arxiv_ids": list[str],
            "bridge_paper_ids": list[str],
            "bridge_arxiv_ids": list[str],
            "consensus_tier": str,        # "gold" / "silver" / "bronze"
            "distractor_types": list[str],
            "venue": str,
            "graph": {"nodes": {...}, "edges": {...}},
            "paper_pool": {arxiv_id: {"title": str, "sections": [...]}},
        }
    """
    base = Path(agenthop_dir)

    # Load QA rows
    qa_path = base / "qa" / "full.jsonl"
    qa_rows = []
    with open(qa_path) as f:
        for line in f:
            qa_rows.append(json.loads(line))

    # Load graphs (index by sample_id)
    graph_path = base / "graphs" / "full.jsonl"
    graphs = {}
    with open(graph_path) as f:
        for line in f:
            g = json.loads(line)
            graphs[g["sample_id"]] = g

    # Load paper pool (shared, deduplicated)
    pool = load_paper_pool(base)

    # Merge into per-sample format
    samples = []
    for qa in qa_rows:
        sid = qa["id"]
        graph = graphs.get(sid, {})
        nodes = graph.get("nodes", {})

        # Build per-sample paper_pool from the shared pool
        sample_pool = {}
        for pid, node_info in nodes.items():
            arxiv_id = node_info.get("arxivId", "")
            if arxiv_id and arxiv_id in pool:
                paper = pool[arxiv_id]
                sample_pool[arxiv_id] = {
                    "title": paper.get("title", ""),
                    "sections": paper.get("sections", []),
                }

        sample = {
            "id": sid,
            "question_type": qa.get("question_type", ""),
            "depth": qa.get("depth", 0),
            "reasoning_type": qa.get("reasoning_type", ""),
            "cognitive_skill": qa.get("cognitive_skill", ""),
            "venue": qa.get("venue", ""),
            "consensus_tier": qa.get("consensus_tier", ""),
            "question": qa.get("question", ""),
            "options": qa.get("options", []),
            "correct_index": qa.get("correct_index", 0),
            "distractor_types": qa.get("distractor_types"),
            "seed_paper_id": qa.get("seed_paper_id", ""),
            "seed_arxiv_id": qa.get("seed_arxiv_id", ""),
            "seed_title": qa.get("seed_title", ""),
            "gold_paper_ids": qa.get("gold_paper_ids", []),
            "gold_arxiv_ids": qa.get("gold_arxiv_ids", []),
            "bridge_paper_ids": qa.get("bridge_paper_ids", []),
            "bridge_arxiv_ids": qa.get("bridge_arxiv_ids", []),
            "filter_agreement": qa.get("filter_agreement"),
            "graph": {
                "nodes": nodes,
                "edges": graph.get("edges", {}),
            },
            "paper_pool": sample_pool,
        }
        samples.append(sample)

        if limit and len(samples) >= limit:
            break

    return samples


if __name__ == "__main__":
    import sys
    base = sys.argv[1] if len(sys.argv) > 1 else "."
    samples = load_agenthop(base)
    print(f"Loaded {len(samples)} AgentHop samples.")
    print(f"  Single-target: {sum(1 for s in samples if s['question_type'] == 'single-target')}")
    print(f"  Multi-target:  {sum(1 for s in samples if s['question_type'] == 'multi-target')}")
    print(f"\nExample (sample 0): {samples[0]['id']}")
    print(f"  Question: {samples[0]['question'][:120]}...")
    print(f"  Reasoning type: {samples[0]['reasoning_type']}, Depth: {samples[0]['depth']}")
    print(f"  Gold arXiv IDs: {samples[0]['gold_arxiv_ids']}")
    print(f"  Pool size for this sample: {len(samples[0]['paper_pool'])} papers")

    labels = load_recall_labels(base)
    print(f"\nLoaded {len(labels)} recall-label records.")
