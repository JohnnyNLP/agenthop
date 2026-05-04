"""Stage 7 (audit prep): pre-bake a lightweight Gradio-loadable copy of the data.

Reduces ~10GB full set to ~200MB by keeping only the gold chain papers.

Usage:
    python stage7_prebake_data.py --target lite --output audit_data/lite
    python stage7_prebake_data.py --target full --output audit_data/full
    python stage7_prebake_data.py --target both
"""

import argparse
import json
import glob
import sys
from pathlib import Path

MCQ_VALIDATED = Path(__file__).resolve().parent / "data/questions/mcq_combined_validated.json"
BENCHMARK_DIR = Path(__file__).resolve().parent / "data/benchmark"


def main():
    parser = argparse.ArgumentParser(description="Pre-bake lightweight audit data")
    parser.add_argument("--target", choices=["lite", "full", "both"], default="both")
    parser.add_argument("--output", type=str, default=None,
                        help="Output dir (default: audit/audit_data/<target>)")
    args = parser.parse_args()

    # Load gold chain metadata to know which papers to keep
    print("Loading metadata...")
    with open(MCQ_VALIDATED) as f:
        validated = json.load(f)
    meta_map = {}
    for v in validated:
        meta_map[v["id"]] = v

    targets = []
    if args.target in ("lite", "both"):
        targets.append(("lite", BENCHMARK_DIR / "lite"))
    if args.target in ("full", "both"):
        targets.append(("full", BENCHMARK_DIR / "full"))

    for label, src_dir in targets:
        out_dir = Path(args.output) if args.output else Path(__file__).resolve().parent / "audit_data" / label
        out_dir.mkdir(parents=True, exist_ok=True)

        files = sorted([f for f in glob.glob(str(src_dir / "*.json"))
                        if not Path(f).name.startswith("agenthop")])
        print(f"\n{label}: {len(files)} files")

        total_orig = 0
        total_slim = 0

        for fpath in files:
            with open(fpath) as f:
                s = json.load(f)

            orig_size = len(json.dumps(s))
            total_orig += orig_size

            # Find which arxivIds we need to keep
            sid = s["id"]
            meta = meta_map.get(sid)
            # Try old ID format
            if not meta:
                old_id = sid
                if sid.startswith("st_"):
                    old_id = "q_" + sid[3:]
                elif sid.startswith("mt_"):
                    old_id = "dfs_" + sid[3:]
                meta = meta_map.get(old_id)

            keep_pids = set()
            nodes = s["graph"]["nodes"]
            edges = s["graph"]["edges"]

            if meta:
                # BFS: keep terminal + bridge
                chain = meta.get("chain", {})
                if chain and "terminal" in chain:
                    t_pid = chain["terminal"]["paperId"]
                    keep_pids.add(t_pid)
                    # Bridge for d2
                    if s.get("depth", 0) == 2:
                        seed = s["seed_paper_id"]
                        for ref in edges.get(seed, []):
                            if ref in edges and t_pid in edges[ref]:
                                keep_pids.add(ref)
                                break

                # DFS: keep targets
                group = meta.get("group", {})
                for t in group.get("targets", []):
                    if "paperId" in t:
                        keep_pids.add(t["paperId"])

            # Always keep seed
            keep_pids.add(s["seed_paper_id"])

            # Map paperIds to arxivIds
            keep_arxivs = set()
            for pid in keep_pids:
                arxiv = nodes.get(pid, {}).get("arxivId", "")
                if arxiv:
                    keep_arxivs.add(arxiv)

            # Strip paper_pool to only needed papers
            slim_pool = {k: v for k, v in s.get("paper_pool", {}).items() if k in keep_arxivs}
            s["paper_pool"] = slim_pool

            # Strip graph nodes to only those with edges or in keep set
            # Keep: seed, all edge sources, all edge targets, keep_pids
            needed_nodes = set(keep_pids)
            needed_nodes.add(s["seed_paper_id"])
            for src, tgts in edges.items():
                needed_nodes.add(src)
                needed_nodes.update(tgts)
            s["graph"]["nodes"] = {k: v for k, v in nodes.items() if k in needed_nodes}

            slim_size = len(json.dumps(s))
            total_slim += slim_size

            out_path = out_dir / Path(fpath).name
            with open(out_path, "w") as f:
                json.dump(s, f, ensure_ascii=False)

        ratio = total_slim / total_orig * 100 if total_orig else 0
        print(f"  Original: {total_orig / 1024 / 1024:.0f} MB")
        print(f"  Slimmed:  {total_slim / 1024 / 1024:.0f} MB ({ratio:.1f}%)")
        print(f"  Saved to: {out_dir}")


if __name__ == "__main__":
    main()
