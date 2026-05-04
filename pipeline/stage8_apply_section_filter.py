"""Stage 8 (post-audit triage): structural section-length pass on gold-path papers.

Drops samples where any gold-path paper (seed/bridge/terminal/target) contains
a section exceeding p99.9 (70,015 characters). These outliers are parser-
collapsed appendix dumps.

Also strips non-gold >70K sections from the paper_pool (dead weight).

Usage:
    python stage8_apply_section_filter.py                # dry-run
    python stage8_apply_section_filter.py --apply
"""
import argparse
import json
import shutil
from pathlib import Path

import os
ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("AGENTHOP_DATA", ROOT / "AgentHop_release"))
QA = DATA / "qa/full.jsonl"
GRAPHS = DATA / "graphs/full.jsonl"
RECALL = DATA / "audit/recall_labels.jsonl"
HUMAN = DATA / "audit/human_verdicts.jsonl"
POOL = DATA / "paper_pool/papers.jsonl"

P99_9 = 70_015


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    # Load pool
    pool = {}
    with open(POOL) as f:
        for line in f:
            r = json.loads(line)
            pool[r["arxiv_id"]] = r

    # Scan QA for gold-paper oversized-section drops
    drop_sids = set()
    over_gold = []
    with open(QA) as f:
        for line in f:
            q = json.loads(line)
            sid = q["id"]
            gold_aids = []
            if q.get("seed_arxiv_id"):
                gold_aids.append(q["seed_arxiv_id"])
            gold_aids += q.get("bridge_arxiv_ids", []) or []
            gold_aids += q.get("gold_arxiv_ids", []) or []
            for aid in gold_aids:
                p = pool.get(aid)
                if not p:
                    continue
                for s in p.get("sections", []):
                    if len(s.get("text", "")) > P99_9:
                        drop_sids.add(sid)
                        over_gold.append((sid, aid, s.get("header", ""), len(s.get("text", ""))))
                        break
                if sid in drop_sids:
                    break

    # Non-gold sections >P99_9 to strip from pool
    gold_aids_all = set()
    with open(QA) as f:
        for line in f:
            q = json.loads(line)
            if q["id"] in drop_sids:
                continue
            if q.get("seed_arxiv_id"):
                gold_aids_all.add(q["seed_arxiv_id"])
            for a in q.get("bridge_arxiv_ids", []) or []: gold_aids_all.add(a)
            for a in q.get("gold_arxiv_ids", []) or []: gold_aids_all.add(a)

    strip_count = 0
    strip_chars = 0
    for aid, p in pool.items():
        if aid in gold_aids_all:
            continue
        new_secs = []
        for s in p.get("sections", []):
            if len(s.get("text", "")) > P99_9:
                strip_count += 1
                strip_chars += len(s.get("text", ""))
                continue
            new_secs.append(s)
        p["sections"] = new_secs

    print(f"Gold-paper oversized sections (samples to drop): {len(drop_sids)}")
    for sid, aid, hdr, n in over_gold[:10]:
        print(f"  {sid}: {aid} §{hdr[:60]} ({n:,} chars)")
    if len(over_gold) > 10:
        print(f"  ... +{len(over_gold)-10} more")
    print()
    print(f"Non-gold oversized sections stripped from pool: {strip_count}")
    print(f"Chars stripped from pool: {strip_chars:,} (~{strip_chars/1_048_576:.1f} MB)")

    # Current sizes
    n_qa = sum(1 for _ in open(QA))
    n_graphs = sum(1 for _ in open(GRAPHS))
    n_recall = sum(1 for _ in open(RECALL))
    n_human = sum(1 for _ in open(HUMAN))
    print(f"\nCurrent sizes:")
    print(f"  qa:     {n_qa}")
    print(f"  graphs: {n_graphs}")
    print(f"  recall: {n_recall}")
    print(f"  human:  {n_human}")
    print(f"After applying:")
    print(f"  qa:     {n_qa - len(drop_sids)}")
    print(f"  graphs: {n_graphs - len(drop_sids)}")
    print(f"  recall: {n_recall - len(drop_sids)}")
    print(f"  human:  {n_human - len(drop_sids)}")

    if not args.apply:
        print("\n(dry-run — rerun with --apply to write)")
        return

    # Backup + filter jsonl files
    for p in [QA, GRAPHS, RECALL, HUMAN, POOL]:
        shutil.copy(p, p.with_suffix(p.suffix + ".presize_bak"))

    def filter_file(path, id_key):
        records = []
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                sid = r.get(id_key) or r.get("id")
                if sid in drop_sids:
                    continue
                records.append(r)
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return len(records)

    for path, key in [(QA, "id"), (GRAPHS, "sample_id"), (RECALL, "sample_id"), (HUMAN, "sample_id")]:
        n = filter_file(path, key)
        print(f"  wrote {path.name}: {n} records")

    # Rewrite pool
    with open(POOL, "w") as f:
        for aid, p in pool.items():
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"  rewrote pool: {len(pool)} papers, {strip_count} oversized non-gold sections stripped")
    print("\n✔ done. Backups: *.presize_bak")


if __name__ == "__main__":
    main()
