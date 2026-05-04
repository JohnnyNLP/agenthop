"""Stage 7 (auditor UI): Gradio interface for recall-label triage on flagged claims.

Loads recall_triage_cases.jsonl (produced by
build_recall_triage_cases.py) and presents each flagged case with a two-column
layout: the question + flagged claim + verdict form on the left, and the full
paper sections (foldable) on the right with the claimed section auto-expanded.

Usage:
    python stage7_recall_app.py --port 7872
    python stage7_recall_app.py --cases /path/to/recall_triage_cases.jsonl

Results are saved per-case to audit_results/recall_triage/{auditor}/.
"""
import argparse
import json
import re
import time
from pathlib import Path

import gradio as gr

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_CASES = EVAL_DIR / "recall_triage_cases.jsonl"
DEFAULT_LOG = EVAL_DIR / "audit_results/recall_triage"
OPTION_LABELS = ["A", "B", "C", "D"]

VF_VERDICTS = [
    "Keep (rounding/precision ok)",
    "Repoint section",
    "Drop claim only",
    "Drop sample",
]
US_VERDICTS = [
    "Repoint to parent",
    "Recover via LaTeX source",
    "Demote to supporting",
    "Drop sample",
]


def load_cases(path: Path) -> list[dict]:
    cases = []
    with open(path) as f:
        for line in f:
            cases.append(json.loads(line))
    return cases


def _format_paper_text(text: str) -> str:
    text = re.sub(r"\[\s*(\d+)\s*\]", r"[\1]", text)
    text = re.sub(r"\[\s*([\d\s,]+)\s*\]",
                  lambda m: "[" + ", ".join(m.group(1).split()) + "]", text)
    lines = text.split("\n")
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if len(line) > 500:
            line = re.sub(r"(\. )([A-Z])", r".\n\n\2", line)
        out.append(line)
    return "\n\n".join(out)


def _normalize_header(h: str) -> str:
    h = h.lower().strip()
    h = re.sub(r"^apdx_[a-z][\d.]*:\s*", "", h)
    h = re.sub(r"^apdx:\s*", "", h)
    h = re.sub(r"^[ivxlc]+-?[a-z]?\s+", "", h)
    h = re.sub(r"^[\d.]+\s*", "", h)
    h = re.sub(r"[:\-–—]", " ", h)
    h = re.sub(r"\s+", " ", h).strip()
    return h


def _is_claimed_section(header: str, claimed: str) -> bool:
    """Best-effort claimed-section highlight. Matches when the auditor would
    expect to see this section as 'the one GPT named' — substring either way,
    or 60%+ normalized word overlap."""
    if not claimed:
        return False
    h_lower = header.lower()
    c_lower = claimed.lower()
    if c_lower in h_lower or h_lower in c_lower:
        return True
    hn = _normalize_header(header)
    cn = _normalize_header(claimed)
    if hn and cn and (cn in hn or hn in cn):
        return True
    if hn and cn:
        hw = set(hn.split())
        cw = set(cn.split())
        if len(cw) >= 2 and hw and len(cw & hw) / max(len(cw), 1) >= 0.6:
            return True
    return False


def render_left(case: dict, case_idx: int, total: int) -> tuple[str, str, str, str]:
    """Return (header_md, question_md, options_md, claim_md)."""
    ctype = case["type"]
    type_badge = "🟥 verified=false" if ctype == "verified_false" else "🟧 unresolved section"
    header = (
        f"### Case {case_idx+1} / {total}  ·  `{case['case_id']}`\n"
        f"**Type:** {type_badge}  ·  "
        f"**Sample:** `{case['sample_id']}` ({case.get('question_type','?')})"
    )

    question_md = f"**Question**\n\n{case.get('question', '')}"

    ci = case.get("correct_index", -1)
    opts_lines = []
    for i, opt in enumerate(case.get("options", [])):
        if i == ci:
            opts_lines.append(
                f'<div class="option-correct">\n\n**({OPTION_LABELS[i]}) CORRECT** — {opt}\n\n</div>'
            )
        else:
            opts_lines.append(
                f'<div class="option-wrong">\n\n**({OPTION_LABELS[i]})** {opt}\n\n</div>'
            )
    options_md = "\n".join(opts_lines)

    aid = case.get("claimed_arxiv_id", "")
    ptitle = case.get("claimed_paper_title", "")
    claimed_sec = case.get("claimed_section", "")
    links = ""
    if aid:
        links = (
            f"[ar5iv](https://ar5iv.labs.arxiv.org/html/{aid}) · "
            f"[arxiv](https://arxiv.org/abs/{aid})"
        )
    verbatim = case.get("verbatim_match")
    verb_note = ""
    if verbatim is True:
        verb_note = "  ·  *verbatim match*"
    elif verbatim is False:
        verb_note = "  ·  *paraphrased*"

    note = case.get("audit_note", "")
    note_block = ""
    if note:
        note_block = (
            f"\n\n<div class=\"audit-note\">\n\n"
            f"**GPT audit note:** {note}\n\n</div>"
        )

    claim_md = (
        f"<div class=\"claim-card\">\n\n"
        f"### Flagged {'claim' if ctype=='verified_false' else 'label'}{verb_note}\n\n"
        f"> {case.get('claim', '')}\n\n"
        f"**Paper:** {ptitle}  `{aid}`  \n{links}\n\n"
        f"**Claimed section:** `{claimed_sec}`"
        f"{note_block}\n\n</div>"
    )

    return header, question_md, options_md, claim_md


def render_right(case: dict) -> str:
    aid = case.get("claimed_arxiv_id", "")
    ptitle = case.get("claimed_paper_title", "")
    secs = case.get("sections", [])
    claimed_sec = case.get("claimed_section", "")

    links = ""
    if aid:
        links = (
            f"[ar5iv](https://ar5iv.labs.arxiv.org/html/{aid}) · "
            f"[arxiv](https://arxiv.org/abs/{aid})"
        )

    lines = [f"### {ptitle or aid or 'Paper'}", f"{links}", f"*{len(secs)} sections*"]
    if not secs:
        lines.append("\n\n*(Paper content not found in pool.)*")
        return "\n".join(lines)

    lines.append("")
    for s in secs:
        header = s.get("header", "(untitled)")
        text = s.get("text", "")
        chars = s.get("chars", len(text))
        is_claimed = _is_claimed_section(header, claimed_sec)
        open_attr = " open" if is_claimed else ""
        summary_class = " class=\"claimed-section\"" if is_claimed else ""
        tag = " 🎯" if is_claimed else ""
        body = _format_paper_text(text) if text else "*(empty)*"
        lines.append(
            f"<details{open_attr}><summary{summary_class}>"
            f"<b>{header}</b>{tag} — {chars:,} chars"
            f"</summary>\n\n{body}\n\n</details>"
        )
    return "\n".join(lines)


APP_CSS = """
    .claim-card { background:#fffbe6; padding:16px 20px; border-radius:10px;
                   border-left:4px solid #f0a500; margin:8px 0; }
    .audit-note { background:#fff0f0; padding:10px 14px; border-radius:8px;
                  border-left:3px solid #d32f2f; margin-top:8px; font-size:0.95em; }
    .option-correct { background:#c8e6c9; padding:8px 12px; border-radius:6px;
                      margin:4px 0; }
    .option-wrong { background:#f5f5f5; padding:8px 12px; border-radius:6px;
                    margin:4px 0; }
    summary.claimed-section { background:#fff3cd; padding:4px 8px; border-radius:4px;
                               font-weight:bold; }
    .case-header { font-size:1.1em; font-weight:bold; }
    .paper-column { max-height: 82vh; overflow-y: auto; padding-right: 8px; }
    .paper-column .claim-card { position: sticky; top: 0; z-index: 10;
                                 box-shadow: 0 2px 6px rgba(0,0,0,0.08); }
"""


def build_app(cases: list[dict], log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)

    def resume_idx(auditor: str) -> int:
        """First case without a saved verdict for this auditor."""
        if not auditor:
            return 0
        adir = log_dir / auditor
        if not adir.exists():
            return 0
        done = {p.stem for p in adir.glob("*.json")}
        for i, c in enumerate(cases):
            if c["case_id"] not in done:
                return i
        return len(cases)  # all done

    def load_case(auditor: str, idx: int):
        auditor = (auditor or "").strip()
        if not auditor:
            return (
                "**Enter auditor ID above, then click Load / Resume.**",
                "", "", "", "### (paper will appear here)",
                gr.update(choices=VF_VERDICTS, value=None, visible=False),
                gr.update(value="", visible=False), "",
                {"idx": 0, "auditor": auditor, "start_time": time.time()},
                "",
            )
        if idx >= len(cases):
            return (
                f"## Done! {len(cases)}/{len(cases)} cases reviewed.",
                "", "", "", "### (no more cases)",
                gr.update(choices=VF_VERDICTS, value=None, visible=False),
                gr.update(value="", visible=False), "",
                {"idx": idx, "auditor": auditor, "start_time": time.time()},
                "",
            )
        c = cases[idx]
        header, q, opts, claim = render_left(c, idx, len(cases))
        paper = render_right(c)
        choices = VF_VERDICTS if c["type"] == "verified_false" else US_VERDICTS

        # Check if previously saved
        prev_path = log_dir / auditor / f"{c['case_id']}.json"
        prev_verdict = None
        prev_repoint = ""
        prev_notes = ""
        if prev_path.exists():
            try:
                with open(prev_path) as f:
                    prev = json.load(f)
                prev_verdict = prev.get("verdict")
                prev_repoint = prev.get("repoint_to") or ""
                prev_notes = prev.get("notes") or ""
            except (json.JSONDecodeError, OSError):
                pass

        repoint_visible = prev_verdict in ("Repoint section", "Repoint to parent")
        return (
            header, q, opts, claim, paper,
            gr.update(choices=choices, value=prev_verdict, visible=True),
            gr.update(value=prev_repoint, visible=repoint_visible),
            prev_notes,
            {"idx": idx, "auditor": auditor, "start_time": time.time()},
            f"Loaded `{c['case_id']}`",
        )

    def on_load_click(auditor):
        idx = resume_idx(auditor.strip())
        return load_case(auditor, idx)

    def on_verdict_change(verdict):
        show = verdict in ("Repoint section", "Repoint to parent")
        return gr.update(visible=show)

    def on_save(state, verdict, repoint_to, notes):
        auditor = state.get("auditor", "")
        idx = state.get("idx", 0)
        start = state.get("start_time", time.time())
        if not auditor:
            return (
                gr.update(),) * 10 + ("Enter auditor ID first.",)  # not used
        if idx >= len(cases):
            return load_case(auditor, idx)[:-1] + ("Already done.",)
        c = cases[idx]
        if not verdict:
            # Don't advance; show a nudge
            return load_case(auditor, idx)[:-1] + ("⚠️ Select a verdict before saving.",)

        adir = log_dir / auditor
        adir.mkdir(parents=True, exist_ok=True)
        result = {
            "case_id": c["case_id"],
            "case_type": c["type"],
            "sample_id": c["sample_id"],
            "claimed_arxiv_id": c.get("claimed_arxiv_id", ""),
            "claimed_section": c.get("claimed_section", ""),
            "claim": c.get("claim", ""),
            "verdict": verdict,
            "repoint_to": (repoint_to or "").strip() or None,
            "notes": notes or "",
            "wall_time_s": time.time() - start,
            "auditor": auditor,
        }
        with open(adir / f"{c['case_id']}.json", "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        next_idx = idx + 1
        return load_case(auditor, next_idx)[:-1] + (f"✔ Saved `{c['case_id']}`",)

    def on_prev(state):
        auditor = state.get("auditor", "")
        idx = max(0, state.get("idx", 0) - 1)
        return load_case(auditor, idx)

    def on_jump(state, jump_to):
        auditor = state.get("auditor", "")
        try:
            idx = max(0, min(len(cases), int(jump_to) - 1))
        except (ValueError, TypeError):
            idx = state.get("idx", 0)
        return load_case(auditor, idx)

    with gr.Blocks(title="Recall-Label Triage") as app:
        state = gr.State({"idx": 0, "auditor": "", "start_time": time.time()})

        with gr.Row():
            auditor_box = gr.Textbox(label="Auditor ID", placeholder="e.g. alice", scale=2)
            load_btn = gr.Button("Load / Resume", variant="primary", scale=1)
            jump_box = gr.Number(label="Jump to case #", value=1, precision=0, scale=1)
            jump_btn = gr.Button("Go", scale=1)
            status_md = gr.Markdown("", elem_classes=["case-header"])

        header_md = gr.Markdown("**Enter auditor ID and click Load / Resume.**")

        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                question_md = gr.Markdown()
                options_md = gr.Markdown()

                verdict_radio = gr.Radio(choices=VF_VERDICTS, label="Verdict", visible=False)
                repoint_box = gr.Textbox(
                    label="Repoint target section (if applicable)",
                    placeholder="e.g. 4.2 Results",
                    visible=False,
                )
                notes_box = gr.Textbox(label="Notes", lines=2, placeholder="Optional")

                with gr.Row():
                    prev_btn = gr.Button("◀ Prev")
                    save_btn = gr.Button("Save & Next ▶", variant="primary")

            with gr.Column(scale=7, elem_classes=["paper-column"]):
                claim_md = gr.Markdown()
                paper_md = gr.Markdown("### (load a case)")

        outputs = [
            header_md, question_md, options_md, claim_md, paper_md,
            verdict_radio, repoint_box, notes_box, state, status_md,
        ]

        load_btn.click(on_load_click, [auditor_box], outputs)
        jump_btn.click(on_jump, [state, jump_box], outputs)
        save_btn.click(on_save, [state, verdict_radio, repoint_box, notes_box], outputs)
        prev_btn.click(on_prev, [state], outputs)
        verdict_radio.change(on_verdict_change, [verdict_radio], [repoint_box])

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=str, default=str(DEFAULT_CASES))
    parser.add_argument("--log-dir", type=str, default=str(DEFAULT_LOG))
    parser.add_argument("--port", type=int, default=7872)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    cases = load_cases(Path(args.cases))
    print(f"Loaded {len(cases)} triage cases from {args.cases}")
    vf = sum(1 for c in cases if c["type"] == "verified_false")
    us = sum(1 for c in cases if c["type"] == "unresolved_section")
    print(f"  verified_false: {vf}")
    print(f"  unresolved_section: {us}")

    app = build_app(cases, Path(args.log_dir))
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        css=APP_CSS,
    )


if __name__ == "__main__":
    main()
