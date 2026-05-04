"""Helper: build the recall-label triage case file consumed by stage7_recall_app.py.

Pulls two categories of flagged cases from audit outputs:
  (vf) verified=false claims        — 48 claims across 33 samples
  (us) verified=true, unresolved    — 41 claims across 21 samples

Output: recall_triage_cases.jsonl
        one line per case, ordered: vf first (grouped by sample), then us.

Each case carries everything the Gradio audit form needs to render without
further I/O: sample question/answer, the flagged claim, the paper's title
and full section list (header + text + char count), ar5iv/arxiv URLs, and
(for verified=false) the audit note.
"""
import json
import re
from pathlib import Path

import os
ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("AGENTHOP_DATA", ROOT / "AgentHop_release"))
QA = DATA / "qa/full.jsonl"
POOL = DATA / "paper_pool/papers.jsonl"
RECALL = DATA / "audit/recall_labels.jsonl"
OUT = Path(__file__).resolve().parent / "recall_triage_cases.jsonl"


def normalize_header(h: str) -> str:
    h = h.lower().strip()
    h = re.sub(r"^apdx_[a-z][\d.]*:\s*", "", h)
    h = re.sub(r"^apdx:\s*", "", h)
    h = re.sub(r"^[ivxlc]+-?[a-z]?\s+", "", h)
    h = re.sub(r"^[\d.]+\s*", "", h)
    h = re.sub(r"[:\-–—]", " ", h)
    h = re.sub(r"\s+", " ", h).strip()
    return h


def section_matches(label_sec: str, sections: list[tuple[str, str]]) -> bool:
    if not label_sec:
        return True
    if not sections:
        return False
    lbl_lower = label_sec.lower()
    lbl_norm = normalize_header(label_sec)
    for header, text in sections:
        h_lower = header.lower()
        h_norm = normalize_header(header)
        if lbl_lower in h_lower or h_lower in lbl_lower:
            return True
        if lbl_norm and h_norm and (lbl_norm in h_norm or h_norm in lbl_norm):
            return True
        if lbl_norm and h_norm:
            lw = set(lbl_norm.split())
            hw = set(h_norm.split())
            if len(lw) >= 2 and hw and len(lw & hw) / max(len(lw), 1) >= 0.6:
                return True
        if lbl_norm and len(lbl_norm) >= 5 and text:
            if lbl_norm in text.lower() or lbl_lower in text.lower():
                return True
    return False


def main():
    # ── Load QA ──────────────────────────────────────────────────────────
    qa = {}
    with open(QA) as f:
        for line in f:
            r = json.loads(line)
            qa[r["id"]] = r
    print(f"Loaded {len(qa)} QA samples")

    # ── Load paper pool (arxiv_id -> {title, sections}) ──────────────────
    pool = {}
    title_to_aid = {}
    with open(POOL) as f:
        for line in f:
            r = json.loads(line)
            aid = r.get("arxiv_id", "")
            pool[aid] = {
                "title": r.get("title", ""),
                "sections": [(s.get("header", ""), s.get("text", "")) for s in r.get("sections", [])],
            }
            if r.get("title"):
                title_to_aid[r["title"].strip().lower()] = aid
    print(f"Loaded {len(pool)} papers")

    # ── Walk recall labels, emit cases ───────────────────────────────────
    vf_cases = []  # verified_false
    us_cases = []  # unresolved_section

    with open(RECALL) as f:
        for line in f:
            r = json.loads(line)
            sid = r["sample_id"]
            if sid not in qa:
                continue
            sample = qa[sid]
            analysis = r.get("analysis", {})

            # verified=false claims
            av_claims = analysis.get("answer_validity", {}).get("claims", [])
            for cidx, c in enumerate(av_claims):
                if c.get("verified") is not False:
                    continue
                # Resolve paper title -> arxiv_id (the claim carries paper title, not arxiv_id)
                paper_title = c.get("source_paper", "") or ""
                aid = title_to_aid.get(paper_title.strip().lower(), "")
                # Fallback: fuzzy title prefix match
                if not aid and paper_title:
                    needle = paper_title.strip().lower()[:40]
                    for t, a in title_to_aid.items():
                        if needle and (needle in t or t.startswith(needle)):
                            aid = a
                            break
                sections_payload = []
                if aid and aid in pool:
                    sections_payload = [
                        {"header": h, "text": t, "chars": len(t)}
                        for h, t in pool[aid]["sections"]
                    ]
                vf_cases.append({
                    "case_id": f"vf_{sid}_c{cidx}",
                    "type": "verified_false",
                    "sample_id": sid,
                    "question_type": sample.get("question_type", ""),
                    "question": sample.get("question", ""),
                    "options": sample.get("options", []),
                    "correct_index": sample.get("correct_index", -1),
                    "claim": c.get("claim", ""),
                    "claimed_paper_title": paper_title,
                    "claimed_arxiv_id": aid or "",
                    "claimed_section": c.get("source_section", ""),
                    "audit_note": c.get("note", ""),
                    "verbatim_match": c.get("verbatim_match", None),
                    "sections": sections_payload,
                })

            # verified=true but unresolved section_recall_labels
            for lidx, lbl in enumerate(analysis.get("section_recall_labels", [])):
                if lbl.get("relevance") != "direct":
                    continue
                aid = lbl.get("arxiv_id", "")
                sec = lbl.get("section", "")
                if not aid:
                    continue
                paper_sections = pool.get(aid, {}).get("sections", [])
                if not paper_sections and aid not in pool:
                    # mt_0849 case: arxiv_id contains section text — try to resolve via paper title
                    fallback_title = lbl.get("paper", "")
                    fallback_aid = title_to_aid.get(fallback_title.strip().lower(), "")
                    if fallback_aid:
                        aid = fallback_aid
                        paper_sections = pool[aid]["sections"]
                if section_matches(sec, paper_sections):
                    continue
                # unresolved
                sections_payload = [
                    {"header": h, "text": t, "chars": len(t)}
                    for h, t in paper_sections
                ]
                us_cases.append({
                    "case_id": f"us_{sid}_l{lidx}",
                    "type": "unresolved_section",
                    "sample_id": sid,
                    "question_type": sample.get("question_type", ""),
                    "question": sample.get("question", ""),
                    "options": sample.get("options", []),
                    "correct_index": sample.get("correct_index", -1),
                    "claim": lbl.get("what_it_contains", ""),
                    "claimed_paper_title": lbl.get("paper", "") or pool.get(aid, {}).get("title", ""),
                    "claimed_arxiv_id": aid,
                    "claimed_section": sec,
                    "audit_note": "",
                    "verbatim_match": None,
                    "sections": sections_payload,
                })

    # ── Write output ─────────────────────────────────────────────────────
    all_cases = vf_cases + us_cases
    with open(OUT, "w") as f:
        for c in all_cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\nVerified-false cases:   {len(vf_cases)}")
    print(f"Unresolved-section:     {len(us_cases)}")
    print(f"Total triage cases:     {len(all_cases)}")
    print(f"\nWritten to {OUT}")


if __name__ == "__main__":
    main()
