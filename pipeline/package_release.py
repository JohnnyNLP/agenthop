"""Post-pipeline packaging — release-format builder.

Runs after Stages 1–6 and after the human-review and post-audit-triage
stages (which live outside this code path — see ``audit/`` for the Gradio
review interface and triage scripts). For each surviving MCQ, builds the
depth-2 citation neighbourhood (the agent's per-query navigation pool) and
bundles section-structured paper content for every reachable paper.

Output: the four JSONL files that make up the HuggingFace release —
qa/full.jsonl, graphs/full.jsonl, paper_pool/papers.jsonl, and
audit/recall_labels.jsonl.

Usage:
    python 7.package_benchmark.py --lite               # package AgentHop-Lite (204)
    python 7.package_benchmark.py --full               # package full set (1149)
    python 7.package_benchmark.py --lite --skip-fetch   # use only cached data
    python 7.package_benchmark.py --lite --dry-run      # preview without writing
"""

import argparse
import json
import glob
import re
import sys
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config import CACHE_DIR, CHAINS_DIR
from s2_client import S2Client

# ── Paths ───────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"
QUESTIONS_DIR = DATA_DIR / "questions"
BENCHMARK_DIR = DATA_DIR / "benchmark"
SEEDS_FILE = DATA_DIR / "seeds.json"
CONTENT_FILE = Path(CHAINS_DIR) / "content" / "paper_contents.json"
HTML_CACHE = Path(CACHE_DIR) / "html"

MAX_REFS_PER_HOP = 30

# Venue name normalization
VENUE_MAP = {
    "Conference on Empirical Methods in Natural Language Processing": "EMNLP",
    "Annual Meeting of the Association for Computational Linguistics": "ACL",
    "North American Chapter of the Association for Computational Linguistics": "NAACL",
}


# ── Reference graph builder ────────────────────────────────────────────────

def build_citation_graph(
    seed_paper_id: str,
    s2: S2Client,
    ref_cache: dict,
    max_refs: int = MAX_REFS_PER_HOP,
) -> dict:
    """Build a 30×30 depth-2 citation graph from a seed paper.

    Returns:
        {
            "seed": seed_paper_id,
            "nodes": {paperId: {"title", "arxivId", "year", "abstract"}},
            "edges": {paperId: [cited_paperId, ...]},
        }
    """
    nodes = {}
    edges = defaultdict(list)

    def _get_refs(paper_id: str) -> list:
        """Get references, using cache or S2 API."""
        if paper_id in ref_cache:
            return ref_cache[paper_id]
        refs = s2.get_references(paper_id)
        ref_cache[paper_id] = refs
        return refs

    def _add_node(paper: dict) -> str | None:
        pid = paper.get("paperId")
        if not pid:
            return None
        if pid not in nodes:
            ext = paper.get("externalIds", {}) or {}
            nodes[pid] = {
                "title": paper.get("title", ""),
                "arxivId": ext.get("ArXiv") or paper.get("arxivId", ""),
                "year": paper.get("year"),
                "abstract": paper.get("abstract", ""),
            }
        return pid

    # Depth 0: seed
    seed_info = s2.get_paper(seed_paper_id)
    if not seed_info:
        return {"seed": seed_paper_id, "nodes": {}, "edges": {}}
    _add_node(seed_info)

    # Depth 1: seed's references (top 30)
    d1_refs = _get_refs(seed_paper_id) or []
    d1_paper_ids = []
    for ref in d1_refs[:max_refs]:
        cited = ref.get("citedPaper", {})
        pid = _add_node(cited)
        if pid:
            edges[seed_paper_id].append(pid)
            d1_paper_ids.append(pid)

    # Depth 2: each depth-1 paper's references (top 30 each)
    for d1_pid in d1_paper_ids:
        d2_refs = _get_refs(d1_pid) or []
        for ref in d2_refs[:max_refs]:
            cited = ref.get("citedPaper", {})
            pid = _add_node(cited)
            if pid:
                edges[d1_pid].append(pid)

    return {
        "seed": seed_paper_id,
        "nodes": nodes,
        "edges": dict(edges),
    }


# ── Content collection ──────────────────────────────────────────────────────

def collect_content_for_graph(
    graph: dict,
    contents: dict,
    fetch_missing: bool = True,
) -> dict:
    """Collect paper content (sections/tables) for all papers in the graph.

    Returns: {arxivId: {"sections": [...], "tables": [...]}}
    """
    paper_content = {}

    for pid, info in graph["nodes"].items():
        arxiv_id = info.get("arxivId", "")
        if not arxiv_id:
            continue

        if arxiv_id in contents:
            c = contents[arxiv_id]
            paper_content[arxiv_id] = {
                "sections": c.get("sections", []),
                "tables": c.get("tables", []),
            }
        elif fetch_missing:
            # Try to parse from HTML cache
            cache_file = HTML_CACHE / f"{arxiv_id.replace('/', '_')}.html"
            if cache_file.exists():
                from pipeline_parse import parse_paper_cached
                parsed = parse_paper_cached(cache_file)
                if parsed and parsed.get("sections"):
                    paper_content[arxiv_id] = {
                        "sections": parsed["sections"],
                        "tables": parsed.get("tables", []),
                    }
                    # Add to in-memory cache
                    contents[arxiv_id] = parsed

    return paper_content


def _load_fetch_html_module():
    """Load 2.fetch_html.py as a module (name starts with digit)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "fetch_html", Path(__file__).parent / "2.fetch_html.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_fetch_html_mod = None

def parse_paper_from_html(html_path: Path) -> dict | None:
    """Parse a cached HTML file into sections/tables."""
    global _fetch_html_mod
    try:
        if _fetch_html_mod is None:
            _fetch_html_mod = _load_fetch_html_module()
        return _fetch_html_mod.parse_paper(html_path.read_text())
    except Exception:
        return None


# ── Sample packaging ─────────────────────────────────────────────────────────

def get_seed_paper_id(q: dict) -> str | None:
    """Extract seed paperId from a question."""
    if q["question_type"] == "bfs":
        return q.get("chain", {}).get("seed", {}).get("paperId")
    else:
        return q.get("group", {}).get("seed", {}).get("paperId")


def package_sample(
    q: dict,
    graph: dict,
    paper_content: dict,
    seed_lookup: dict,
) -> dict:
    """Package a single MCQ into a benchmark sample."""
    sid = get_seed_paper_id(q)
    seed_info = seed_lookup.get(sid, {})
    venue = seed_info.get("venue", "?")
    venue = VENUE_MAP.get(venue, venue)

    mode = q.get("question_mode", {})

    sample = {
        "id": q["id"],
        "question_type": q["question_type"],
        "depth": q["depth"],
        "reasoning_type": q.get("reasoning_type") or mode.get("relationship", ""),
        "cognitive_skill": mode.get("skill", ""),
        "venue": venue,
        "consensus_tier": q.get("consensus_tier", ""),
        "question": q["question"],
        "options": q["options"],
        "correct_index": q["correct_index"],
        "seed_paper_id": graph["seed"],
        "seed_title": graph["nodes"].get(graph["seed"], {}).get("title", ""),
        "graph": {
            "seed": graph["seed"],
            "nodes": {
                pid: {
                    "title": info["title"],
                    "arxivId": info["arxivId"],
                    "year": info["year"],
                    "abstract": info.get("abstract", ""),
                }
                for pid, info in graph["nodes"].items()
            },
            "edges": graph["edges"],
        },
        "paper_pool": paper_content,
    }

    return sample


# ── Abstract enrichment ──────────────────────────────────────────────────────

def enrich_abstracts(
    graphs: dict[str, dict],
    ref_cache: dict,
    seed_lookup: dict,
    skip_fetch: bool = False,
    all_content: dict | None = None,
) -> int:
    """Fill in missing abstracts across all graphs.

    Sources (in order of preference):
      1. Chain data (seed/terminal/path papers)
      2. S2 reference cache (citedPaper entries)
      3. S2 batch API (by paperId, 500/request)
      4. arXiv API (by arxivId)
      5. Intro text from paper content (fallback)

    Modifies graphs in place. Returns number of abstracts filled.
    """
    # Collect all papers missing abstracts
    missing = {}  # pid -> {arxivId}
    for graph in graphs.values():
        for pid, info in graph["nodes"].items():
            if not info.get("abstract") and pid not in missing:
                missing[pid] = {"arxivId": info.get("arxivId", "")}

    if not missing:
        return 0

    print(f"\nAbstract enrichment: {len(missing)} papers missing abstracts")

    # Source 1: chain data
    lookup = {}
    for cf in sorted(glob.glob(str(Path(CHAINS_DIR) / "chains_*.json"))):
        with open(cf) as f:
            chains = json.load(f)
        for c in chains:
            for src in [c.get("seed", {}), c.get("terminal", {})] + \
                       [n.get("paper", {}) for n in c.get("path", [])]:
                if src.get("abstract") and src.get("paperId"):
                    lookup[src["paperId"]] = src["abstract"]

    filled_chains = _apply_lookup(graphs, missing, lookup)
    print(f"  From chain data: {filled_chains}")

    # Source 2: S2 reference cache
    for pid, refs in ref_cache.items():
        if not refs:
            continue
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            cited = ref.get("citedPaper", {})
            if cited and cited.get("abstract") and cited.get("paperId"):
                lookup[cited["paperId"]] = cited["abstract"]

    filled_cache = _apply_lookup(graphs, missing, lookup)
    print(f"  From reference cache: {filled_cache}")

    if skip_fetch:
        print(f"  Skipping API fetch (--skip-fetch)")
        print(f"  Remaining missing: {len(missing)}")
        return filled_chains + filled_cache

    # Source 3: S2 batch API
    pids_to_fetch = list(missing.keys())
    filled_s2 = 0
    for i in range(0, len(pids_to_fetch), 500):
        batch = pids_to_fetch[i:i+500]
        try:
            payload = json.dumps({"ids": batch}).encode("utf-8")
            url = "https://api.semanticscholar.org/graph/v1/paper/batch?fields=abstract,paperId"
            req = urllib.request.Request(url, data=payload, method="POST")
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=30) as resp:
                results = json.loads(resp.read().decode())
            for paper in results:
                if paper and paper.get("abstract") and paper.get("paperId"):
                    lookup[paper["paperId"]] = paper["abstract"]
            time.sleep(1)
        except Exception as e:
            print(f"    S2 batch error: {e}")
            time.sleep(3)

    filled_s2 = _apply_lookup(graphs, missing, lookup)
    print(f"  From S2 batch API: {filled_s2}")

    # Source 4: arXiv API (for papers with arxivId)
    arxiv_pids = {pid: info["arxivId"] for pid, info in missing.items()
                  if info.get("arxivId")}
    filled_arxiv = 0
    if arxiv_pids:
        # Batch arxiv queries (up to 50 IDs per request)
        aid_to_pid = {aid: pid for pid, aid in arxiv_pids.items()}
        aid_list = list(arxiv_pids.values())
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        for i in range(0, len(aid_list), 50):
            batch = aid_list[i:i+50]
            try:
                id_str = ",".join(batch)
                url = f"http://export.arxiv.org/api/query?id_list={id_str}&max_results={len(batch)}"
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    xml_data = resp.read().decode()
                root = ET.fromstring(xml_data)
                for entry in root.findall("atom:entry", ns):
                    id_url = entry.find("atom:id", ns).text
                    aid = id_url.split("/abs/")[-1].split("v")[0] if "/abs/" in id_url else ""
                    abstract_el = entry.find("atom:summary", ns)
                    if aid and abstract_el is not None and abstract_el.text:
                        abstract = abstract_el.text.strip()
                        if len(abstract) >= 50 and aid in aid_to_pid:
                            lookup[aid_to_pid[aid]] = abstract
                time.sleep(3)  # arxiv rate limit
            except Exception as e:
                print(f"    arXiv batch error: {e}")
                time.sleep(5)

        filled_arxiv = _apply_lookup(graphs, missing, lookup)
        print(f"  From arXiv API: {filled_arxiv}")

    total = filled_chains + filled_cache + filled_s2 + filled_arxiv
    print(f"  Total filled: {total}, still missing: {len(missing)}")

    # ── Fallback: extract intro from paper content as pseudo-abstract ──
    filled_intro = 0
    if not all_content:
        all_content = {}
    # Build arxivId→content lookup across all seeds
    content_lookup = {}
    for sid_content in all_content.values():
        if isinstance(sid_content, dict):
            content_lookup.update(sid_content)
    for pid in list(missing.keys()):
        # Find this paper's arxivId and content
        for graph in graphs.values():
            node = graph["nodes"].get(pid)
            if not node:
                continue
            aid = node.get("arxivId", "")
            if not aid or aid not in content_lookup:
                continue
            sections = content_lookup[aid].get("sections", [])
            if not sections:
                continue
            # Use the first section (usually intro) or first with "intro" in header
            intro_text = None
            for sec in sections:
                hdr = (sec.get("header") or "").lower()
                if "introduction" in hdr or "abstract" in hdr:
                    intro_text = sec.get("text", "").strip()
                    break
            if not intro_text and sections[0].get("text", "").strip():
                intro_text = sections[0]["text"].strip()
            if intro_text and len(intro_text) >= 100:
                # Truncate to ~first 500 chars (roughly abstract-length)
                if len(intro_text) > 500:
                    # Cut at sentence boundary
                    cut = intro_text[:500].rfind(". ")
                    if cut > 200:
                        intro_text = intro_text[:cut + 1]
                    else:
                        intro_text = intro_text[:500]
                # Apply to all graphs containing this paper
                for g in graphs.values():
                    if pid in g["nodes"] and not g["nodes"][pid].get("abstract"):
                        g["nodes"][pid]["abstract"] = intro_text
                del missing[pid]
                filled_intro += 1
                break  # found content, move to next pid

    if filled_intro:
        print(f"  From intro text: {filled_intro}")
    total += filled_intro
    print(f"  Final total filled: {total}, still missing: {len(missing)}")
    return total


def _apply_lookup(
    graphs: dict[str, dict],
    missing: dict,
    lookup: dict,
) -> int:
    """Apply abstract lookup to all graphs. Removes filled entries from missing."""
    filled = 0
    to_remove = []
    for pid in list(missing.keys()):
        if pid in lookup and len(lookup[pid].strip()) >= 50:
            # Apply to all graphs containing this paper
            for graph in graphs.values():
                if pid in graph["nodes"] and not graph["nodes"][pid].get("abstract"):
                    graph["nodes"][pid]["abstract"] = lookup[pid].strip()
            to_remove.append(pid)
            filled += 1

    for pid in to_remove:
        del missing[pid]

    return filled


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Package AgentHop benchmark samples")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--lite", action="store_true", help="Package AgentHop-Lite (204)")
    group.add_argument("--full", action="store_true", help="Package full set (1149)")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Don't fetch missing references from S2 API")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without writing files")
    parser.add_argument("--max-refs", type=int, default=MAX_REFS_PER_HOP,
                        help=f"Max references per hop (default: {MAX_REFS_PER_HOP})")
    args = parser.parse_args()

    max_refs = args.max_refs

    # Load input questions
    if args.lite:
        input_file = QUESTIONS_DIR / "agenthop_lite.json"
        out_dir = BENCHMARK_DIR / "lite"
    else:
        input_file = QUESTIONS_DIR / "mcq_combined_validated.json"
        out_dir = BENCHMARK_DIR / "full"

    with open(input_file) as f:
        questions = json.load(f)
    print(f"Input: {len(questions)} questions from {input_file.name}")

    # Load seeds
    with open(SEEDS_FILE) as f:
        seeds = json.load(f)
    seed_lookup = {s["paperId"]: s for s in seeds}

    # Load paper contents
    print("Loading paper contents...")
    with open(CONTENT_FILE) as f:
        contents = json.load(f)
    print(f"  {len(contents)} papers in content pool")

    # Pre-load S2 reference cache
    print("Loading S2 reference cache...")
    ref_cache = {}
    for fpath in glob.glob(str(Path(CACHE_DIR) / "paper_*_references*")):
        fname = Path(fpath).name
        pid = fname.split("_references")[0].replace("paper_", "")
        with open(fpath) as f:
            data = json.load(f)
        ref_cache[pid] = data.get("data", []) if isinstance(data, dict) else data
    print(f"  {len(ref_cache)} papers with cached references")

    # Initialize S2 client (for fetching missing refs)
    s2 = S2Client() if not args.skip_fetch else None

    # Group questions by seed to avoid rebuilding the same graph
    by_seed = defaultdict(list)
    for q in questions:
        sid = get_seed_paper_id(q)
        if sid:
            by_seed[sid].append(q)
        else:
            print(f"  Warning: no seed ID for {q['id']}")

    print(f"Unique seeds: {len(by_seed)}")

    # ── Phase 1: Build citation graphs ─────────────────────────────────────
    print("\n── Phase 1: Building citation graphs ──")
    out_dir.mkdir(parents=True, exist_ok=True)
    total_nodes = 0
    total_content = 0
    graphs_built = 0

    all_graphs = {}       # sid -> graph
    all_content = {}      # sid -> paper_content

    for i, (sid, seed_questions) in enumerate(sorted(by_seed.items())):
        seed_title = seed_lookup.get(sid, {}).get("title", "?")[:60]
        print(f"[{i+1}/{len(by_seed)}] {sid[:12]}... {seed_title}", end="", flush=True)

        # Build citation graph
        if s2:
            graph = build_citation_graph(sid, s2, ref_cache, max_refs=max_refs)
        else:
            graph = build_citation_graph_cached(sid, ref_cache, seed_lookup, max_refs=max_refs)

        n_nodes = len(graph["nodes"])
        total_nodes += n_nodes
        graphs_built += 1

        # Collect content
        paper_content = {}
        for pid, info in graph["nodes"].items():
            arxiv_id = info.get("arxivId", "")
            if not arxiv_id:
                continue
            if arxiv_id in contents:
                c = contents[arxiv_id]
                paper_content[arxiv_id] = {
                    "sections": c.get("sections", []),
                    "tables": c.get("tables", []),
                }
            else:
                cache_file = HTML_CACHE / f"{arxiv_id.replace('/', '_')}.html"
                if cache_file.exists():
                    parsed = parse_paper_from_html(cache_file)
                    if parsed and parsed.get("sections"):
                        paper_content[arxiv_id] = {
                            "sections": parsed["sections"],
                            "tables": parsed.get("tables", []),
                        }
                        contents[arxiv_id] = parsed

        total_content += len(paper_content)
        print(f"  {n_nodes} nodes, {len(paper_content)} with content")

        all_graphs[sid] = graph
        all_content[sid] = paper_content

    # ── Phase 2: Enrich abstracts ────────────────────────────────────────
    print("\n── Phase 2: Enriching abstracts ──")
    enrich_abstracts(all_graphs, ref_cache, seed_lookup, skip_fetch=args.skip_fetch, all_content=all_content)

    # ── Phase 3: Package samples ─────────────────────────────────────────
    print("\n── Phase 3: Packaging samples ──")
    total_samples = 0

    for sid, seed_questions in sorted(by_seed.items()):
        graph = all_graphs[sid]
        paper_content = all_content[sid]

        for q in seed_questions:
            sample = package_sample(q, graph, paper_content, seed_lookup)

            if not args.dry_run:
                out_file = out_dir / f"{q['id']}.json"
                with open(out_file, "w") as f:
                    json.dump(sample, f, indent=2, ensure_ascii=False)

            total_samples += 1

    # ── Summary ──────────────────────────────────────────────────────────
    # Count abstract coverage
    all_nodes = 0
    nodes_with_abstract = 0
    for graph in all_graphs.values():
        for info in graph["nodes"].values():
            all_nodes += 1
            if info.get("abstract"):
                nodes_with_abstract += 1
    abs_pct = nodes_with_abstract / all_nodes * 100 if all_nodes else 0

    print(f"\n{'═' * 60}")
    print(f"  Benchmark Packaging Complete")
    print(f"{'═' * 60}")
    print(f"  Samples: {total_samples}")
    print(f"  Graphs built: {graphs_built}")
    print(f"  Avg nodes/graph: {total_nodes / max(graphs_built, 1):.0f}")
    print(f"  Avg papers with content/graph: {total_content / max(graphs_built, 1):.0f}")
    print(f"  Abstract coverage: {nodes_with_abstract}/{all_nodes} ({abs_pct:.1f}%)")
    if not args.dry_run:
        print(f"  Output: {out_dir}/")

        # Also write a combined file
        all_samples = []
        for fpath in sorted(out_dir.glob("*.json")):
            if fpath.name.startswith("agenthop"):
                continue
            with open(fpath) as f:
                all_samples.append(json.load(f))

        combined_name = "agenthop_lite.json" if args.lite else "agenthop_full.json"
        combined_file = out_dir / combined_name
        with open(combined_file, "w") as f:
            json.dump(all_samples, f, indent=2, ensure_ascii=False)
        print(f"  Combined: {combined_file} ({len(all_samples)} samples)")
    else:
        print(f"  [dry-run] No files written")
    print(f"{'═' * 60}")


def build_citation_graph_cached(
    seed_paper_id: str,
    ref_cache: dict,
    seed_lookup: dict,
    max_refs: int = MAX_REFS_PER_HOP,
) -> dict:
    """Build graph from cached data only (no API calls)."""
    nodes = {}
    edges = defaultdict(list)

    # Seed node
    seed_info = seed_lookup.get(seed_paper_id, {})
    nodes[seed_paper_id] = {
        "title": seed_info.get("title", ""),
        "arxivId": seed_info.get("arxivId", ""),
        "year": seed_info.get("year"),
        "abstract": seed_info.get("abstract", ""),
    }

    # Depth 1
    d1_refs = ref_cache.get(seed_paper_id, [])
    d1_pids = []
    for ref in d1_refs[:max_refs]:
        cited = ref.get("citedPaper", {})
        pid = cited.get("paperId")
        if not pid:
            continue
        ext = cited.get("externalIds", {}) or {}
        nodes[pid] = {
            "title": cited.get("title", ""),
            "arxivId": ext.get("ArXiv") or cited.get("arxivId", ""),
            "year": cited.get("year"),
            "abstract": cited.get("abstract", ""),
        }
        edges[seed_paper_id].append(pid)
        d1_pids.append(pid)

    # Depth 2
    for d1_pid in d1_pids:
        d2_refs = ref_cache.get(d1_pid) or []
        for ref in d2_refs[:max_refs]:
            cited = ref.get("citedPaper", {})
            pid = cited.get("paperId")
            if not pid:
                continue
            ext = cited.get("externalIds", {}) or {}
            if pid not in nodes:
                nodes[pid] = {
                    "title": cited.get("title", ""),
                    "arxivId": ext.get("ArXiv") or cited.get("arxivId", ""),
                    "year": cited.get("year"),
                    "abstract": cited.get("abstract", ""),
                }
            edges[d1_pid].append(pid)

    return {
        "seed": seed_paper_id,
        "nodes": nodes,
        "edges": dict(edges),
    }


if __name__ == "__main__":
    main()
