"""Stage 8 (post-audit triage): apply audit verdicts back to the release files.

Reads verdict JSONs from audit_results/recall_triage/{auditor}/
and applies them to AgentHop/audit/recall_labels.jsonl.

Drops samples that received 'Drop sample' verdicts from qa/full.jsonl,
graphs/full.jsonl, audit/recall_labels.jsonl, and audit/human_verdicts.jsonl.

Writes patched files in place. A backup copy of each original file is made.

Usage:
    python stage8_apply_verdicts.py --auditor alice              # dry-run
    python stage8_apply_verdicts.py --auditor alice --apply      # write
"""
import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import os
ROOT = Path(__file__).resolve().parent.parent
TRIAGE = Path(__file__).resolve().parent / "audit_results/recall_triage"
DATA = Path(os.environ.get("AGENTHOP_DATA", ROOT / "AgentHop_release"))
QA = DATA / "qa/full.jsonl"
GRAPHS = DATA / "graphs/full.jsonl"
RECALL = DATA / "audit/recall_labels.jsonl"
HUMAN = DATA / "audit/human_verdicts.jsonl"

CASE_ID_RE = re.compile(r"^(vf|us)_(\w+)_[cl](\d+)$")


def load_verdicts(auditor: str) -> list[dict]:
    d = TRIAGE / auditor
    if not d.exists():
        raise SystemExit(f"No verdicts for auditor '{auditor}': {d} missing")
    out = []
    for f in sorted(d.glob("*.json")):
        out.append(json.load(open(f)))
    return out


def parse_case_id(case_id: str) -> tuple[str, str, int]:
    m = CASE_ID_RE.match(case_id)
    if not m:
        raise ValueError(f"Bad case_id: {case_id}")
    kind, sid, idx = m.group(1), m.group(2), int(m.group(3))
    return kind, sid, idx


def summarize(verdicts: list[dict]) -> dict:
    stats = defaultdict(int)
    per_sid = defaultdict(list)
    for v in verdicts:
        vtype = v["case_type"]
        verd = v.get("verdict") or "(blank)"
        stats[f"{vtype}:{verd}"] += 1
        per_sid[v["sample_id"]].append(v)
    return {"stats": dict(stats), "per_sid": dict(per_sid)}


def apply_to_recall_labels(verdicts: list[dict]) -> tuple[list[dict], set[str]]:
    """Return (new_recall_records, drop_sids)."""
    # Index verdicts by (sid, kind, idx)
    vmap = {}  # (sid, kind, idx) -> verdict
    drop_sids = set()
    for v in verdicts:
        kind, sid, idx = parse_case_id(v["case_id"])
        vmap[(sid, kind, idx)] = v
        if "Drop sample" in (v.get("verdict") or ""):
            drop_sids.add(sid)

    records = []
    repoint_claim = 0
    repoint_label = 0
    drop_claim = 0
    with open(RECALL) as f:
        for line in f:
            r = json.loads(line)
            sid = r["sample_id"]
            if sid in drop_sids:
                continue  # will be dropped
            # Patch verified_false claims
            claims = r.get("analysis", {}).get("answer_validity", {}).get("claims", [])
            new_claims = []
            for i, c in enumerate(claims):
                v = vmap.get((sid, "vf", i))
                if v is None:
                    new_claims.append(c)
                    continue
                verd = v.get("verdict") or ""
                if verd.startswith("Repoint"):
                    if v.get("repoint_to"):
                        c = dict(c)
                        c["source_section"] = v["repoint_to"]
                        c["_repointed_from"] = v.get("claimed_section")
                        repoint_claim += 1
                    new_claims.append(c)
                elif verd == "Drop claim only":
                    drop_claim += 1
                    continue  # skip this claim
                else:
                    # Keep or Drop sample (sample-drop handled elsewhere)
                    new_claims.append(c)
            if "answer_validity" in r.get("analysis", {}):
                r["analysis"]["answer_validity"]["claims"] = new_claims

            # Patch unresolved section_recall_labels
            labels = r.get("analysis", {}).get("section_recall_labels", [])
            for i, lbl in enumerate(labels):
                v = vmap.get((sid, "us", i))
                if v is None:
                    continue
                verd = v.get("verdict") or ""
                if verd.startswith("Repoint") and v.get("repoint_to"):
                    lbl["_repointed_from"] = lbl.get("section")
                    lbl["section"] = v["repoint_to"]
                    repoint_label += 1
                elif verd == "Demote to supporting":
                    lbl["relevance"] = "supporting"
                    lbl["_demoted_from"] = "direct"
                # Drop sample handled above
            records.append(r)
    return records, drop_sids, {
        "repoint_claim": repoint_claim,
        "repoint_label": repoint_label,
        "drop_claim": drop_claim,
    }


def filter_jsonl(path: Path, drop_sids: set[str], id_key: str = "sample_id") -> tuple[int, int]:
    """Filter a JSONL file in-place (after backup) to remove drop_sids.
    Returns (kept, dropped)."""
    records = []
    dropped = 0
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            sid = r.get(id_key) or r.get("id")
            if sid in drop_sids:
                dropped += 1
                continue
            records.append(r)
    return records, dropped


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auditor", default="alice")
    parser.add_argument("--apply", action="store_true", help="actually write files")
    args = parser.parse_args()

    verdicts = load_verdicts(args.auditor)
    summary = summarize(verdicts)
    print(f"Loaded {len(verdicts)} verdicts")
    for k, v in sorted(summary["stats"].items()):
        print(f"  {k}: {v}")

    # Apply to recall_labels
    new_records, drop_sids, patch_stats = apply_to_recall_labels(verdicts)
    print(f"\nPatches to apply:")
    print(f"  repoint (claims):  {patch_stats['repoint_claim']}")
    print(f"  repoint (labels):  {patch_stats['repoint_label']}")
    print(f"  drop (claims):     {patch_stats['drop_claim']}")
    print(f"  drop (samples):    {len(drop_sids)}  → {sorted(drop_sids)}")

    # Existing file sizes
    n_qa = sum(1 for _ in open(QA))
    n_graphs = sum(1 for _ in open(GRAPHS))
    n_recall = sum(1 for _ in open(RECALL))
    n_human = sum(1 for _ in open(HUMAN))
    print(f"\nCurrent sizes:")
    print(f"  qa/full.jsonl:                 {n_qa}")
    print(f"  graphs/full.jsonl:             {n_graphs}")
    print(f"  audit/recall_labels.jsonl:     {n_recall}")
    print(f"  audit/human_verdicts.jsonl:    {n_human}")

    qa_new, qa_drop = filter_jsonl(QA, drop_sids, id_key="id")
    graphs_new, graphs_drop = filter_jsonl(GRAPHS, drop_sids, id_key="sample_id")
    human_new, human_drop = filter_jsonl(HUMAN, drop_sids, id_key="sample_id")
    print(f"\nAfter applying drops:")
    print(f"  qa/full.jsonl:                 {len(qa_new)}  (−{qa_drop})")
    print(f"  graphs/full.jsonl:             {len(graphs_new)}  (−{graphs_drop})")
    print(f"  audit/recall_labels.jsonl:     {len(new_records)}  (−{n_recall - len(new_records)})")
    print(f"  audit/human_verdicts.jsonl:    {len(human_new)}  (−{human_drop})")

    if not args.apply:
        print("\n(dry run — rerun with --apply to write)")
        return

    # Backup + write
    for p in [QA, GRAPHS, RECALL, HUMAN]:
        shutil.copy(p, p.with_suffix(p.suffix + ".prerecall_bak"))
        print(f"  backup: {p}.prerecall_bak")

    write_jsonl(QA, qa_new)
    write_jsonl(GRAPHS, graphs_new)
    write_jsonl(RECALL, new_records)
    write_jsonl(HUMAN, human_new)
    print("\n✔ All files patched in place. Backups created with .prerecall_bak suffix.")


if __name__ == "__main__":
    main()
