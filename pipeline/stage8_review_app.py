"""Stage 8 (post-audit triage): Gradio review of flagged items for the lead-author triage pass.

Matches the audit page layout with fixed bottom bar, skip button, and
loads/overwrites existing audit results.

Sources:
  1. Human-flagged (verdict=Flag from audit_results/, excluding auto-unflagged)
  2. GPT-flagged (warn/fail from gpt_review_list.json)
  3. Zero section match (from gpt_review_list.json)

Usage:
    python stage8_review_app.py --port 7872
    python stage8_review_app.py --port 7872 --auditor alice
"""

import argparse
import json
import glob
import re
import time
from pathlib import Path

import gradio as gr

EVAL_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = EVAL_DIR.parent / "pipeline" / "data" / "benchmark" / "full"
MCQ_VALIDATED = EVAL_DIR.parent / "pipeline" / "data" / "questions" / "mcq_combined_validated.json"
LOG_DIR = EVAL_DIR / "audit_results"
REVIEW_LIST = EVAL_DIR / "gpt_review_list.json"
ANALYSES_EN = EVAL_DIR / "audit_analyses_en_full.json"
OPTION_LABELS = ["A", "B", "C", "D"]


def load_metadata() -> dict:
    with open(MCQ_VALIDATED) as f:
        validated = json.load(f)
    return {v["id"]: v for v in validated}


def load_en_analyses() -> dict:
    if not ANALYSES_EN.exists():
        return {}
    with open(ANALYSES_EN) as f:
        data = json.load(f)
    result = {}
    for item in data:
        if item.get("analysis"):
            result[item["id"]] = item["analysis"]
    return result


def collect_review_items() -> list[dict]:
    """Collect all samples needing review from 3 sources."""
    items = {}  # sid -> {sample_id, reasons[], source[]}

    # 1. Human-flagged
    for auditor_dir in sorted(LOG_DIR.iterdir()):
        if not auditor_dir.is_dir():
            continue
        for f in auditor_dir.glob("*.json"):
            try:
                with open(f) as fh:
                    r = json.load(fh)
                if r.get("verdict") == "Flag" and "Auto-unflagged" not in r.get("notes", ""):
                    sid = r["sample_id"]
                    if sid not in items:
                        items[sid] = {"sample_id": sid, "reasons": [], "sources": [], "human_notes": ""}
                    items[sid]["sources"].append("human_flag")
                    items[sid]["human_notes"] = r.get("notes", "")
                    items[sid]["human_auditor"] = auditor_dir.name
                    items[sid]["human_verdict"] = r.get("verdict")
            except (json.JSONDecodeError, KeyError):
                pass

    # 2+3. GPT-flagged + zero-match
    if REVIEW_LIST.exists():
        with open(REVIEW_LIST) as f:
            gpt_items = json.load(f)
        for entry in gpt_items:
            sid = entry["sample_id"]
            if sid not in items:
                items[sid] = {"sample_id": sid, "reasons": [], "sources": [], "human_notes": ""}
            items[sid]["reasons"].extend(entry.get("reasons", []))
            if "zero_section_match" in entry.get("reasons", []):
                items[sid]["sources"].append("zero_match")
            if any(r for r in entry.get("reasons", []) if r != "zero_section_match"):
                items[sid]["sources"].append("gpt_flag")

    # Deduplicate reasons
    for sid in items:
        items[sid]["reasons"] = list(dict.fromkeys(items[sid]["reasons"]))
        items[sid]["sources"] = list(dict.fromkeys(items[sid]["sources"]))

    return sorted(items.values(), key=lambda x: x["sample_id"])


def load_existing_audit(sid: str) -> dict | None:
    """Load existing audit result from any auditor subdir."""
    for subdir in LOG_DIR.iterdir():
        if subdir.is_dir():
            f = subdir / f"{sid}.json"
            if f.exists():
                with open(f) as fh:
                    return json.load(fh)
    f = LOG_DIR / f"{sid}.json"
    if f.exists():
        with open(f) as fh:
            return json.load(fh)
    return None


def _format_paper_text(text: str, max_chars: int = 8000) -> str:
    if len(text) > max_chars:
        return text[:max_chars] + f"\n\n[...truncated, {len(text) - max_chars} chars omitted]"
    return text


_PAPER_COLORS = [
    ("#1565c0", "#e3f2fd"),  # blue
    ("#c62828", "#ffebee"),  # red
    ("#2e7d32", "#e8f5e9"),  # green
    ("#6a1b9a", "#f3e5f5"),  # purple
    ("#e65100", "#fff3e0"),  # orange
    ("#00838f", "#e0f7fa"),  # teal
]


def _build_paper_colormap(analysis: dict) -> dict:
    """Extract unique paper titles from analysis and assign colors."""
    titles = []
    seen = set()
    for claim in analysis.get("answer_validity", {}).get("claims", []):
        t = claim.get("source_paper", "")
        if t and t not in seen:
            titles.append(t)
            seen.add(t)
    for lbl in analysis.get("section_recall_labels", []):
        t = lbl.get("paper", "")
        if t and t not in seen:
            titles.append(t)
            seen.add(t)
    colormap = {}
    for i, t in enumerate(titles):
        fg, bg = _PAPER_COLORS[i % len(_PAPER_COLORS)]
        colormap[t] = (fg, bg)
    return colormap


def _highlight_paper(title: str, colormap: dict) -> str:
    if title in colormap:
        fg, bg = colormap[title]
        return f'<span style="background:{bg}; color:{fg}; padding:2px 6px; border-radius:4px; font-weight:bold;">{title}</span>'
    return title


def render_en_analysis(analysis: dict, entry: dict) -> str:
    """Render English GPT audit analysis with color-coded paper titles."""
    parts = []

    # Source tags
    sources = entry.get("sources", [])
    reasons = entry.get("reasons", [])
    source_tags = " | ".join(f"`{s}`" for s in sources)
    parts.append(f"**Sources:** {source_tags}")
    if reasons:
        parts.append(f"**GPT reasons:** {', '.join(reasons)}")
    human_notes = entry.get("human_notes", "")
    if human_notes:
        parts.append(f"**Human auditor note:** {human_notes}")

    if not analysis:
        return "\n".join(parts) + "\n\n(No GPT analysis available)"

    # Build paper color map
    colormap = _build_paper_colormap(analysis)

    # Legend
    if colormap:
        legend = " | ".join(_highlight_paper(t, colormap) for t in colormap)
        parts.append(f"\n**Papers:** {legend}")

    # Query quality
    qv = analysis.get("query_quality", {})
    icon = {"pass": "OK", "warn": "WARN", "fail": "FAIL"}.get(qv.get("verdict", ""), "?")
    parts.append(f"\n**Query** [{icon}]: {qv.get('note', '') or 'No issues'}")

    # Answer validity
    av = analysis.get("answer_validity", {})
    icon = {"pass": "OK", "warn": "WARN", "fail": "FAIL"}.get(av.get("verdict", ""), "?")
    parts.append(f"\n**Answer** [{icon}]:")
    for claim in av.get("claims", []):
        check = "OK" if claim.get("verified") else "FAIL"
        verbatim = " (verbatim)" if claim.get("verbatim_match") else ""
        note = f" -- {claim['note']}" if claim.get("note") else ""
        sec = claim.get("source_section", "?")
        paper = claim.get("source_paper", "?")
        parts.append(f"- [{check}]{verbatim} {claim.get('claim', '')}")
        parts.append(f"  -> `{sec}` in {_highlight_paper(paper, colormap)}{note}")

    # Chain coherence (st_)
    cc = analysis.get("chain_coherence")
    if cc and cc.get("verdict"):
        icon = {"pass": "OK", "warn": "WARN", "fail": "FAIL"}.get(cc.get("verdict", ""), "?")
        parts.append(f"\n**Chain** [{icon}]: bridge_necessary={cc.get('bridge_necessary')}")
        if cc.get("seed_to_bridge"):
            parts.append(f"  seed->bridge: {cc['seed_to_bridge']}")
        if cc.get("bridge_to_terminal"):
            parts.append(f"  bridge->terminal: {cc['bridge_to_terminal']}")
        if cc.get("note"):
            parts.append(f"  note: {cc['note']}")

    # Synthesis check (mt_)
    sc = analysis.get("synthesis_check")
    if sc and sc.get("verdict"):
        icon = {"pass": "OK", "warn": "WARN", "fail": "FAIL"}.get(sc.get("verdict", ""), "?")
        parts.append(f"\n**Synthesis** [{icon}]: both_needed={sc.get('both_targets_needed')}")
        if sc.get("target_1_contribution"):
            parts.append(f"  T1: {sc['target_1_contribution']}")
        if sc.get("target_2_contribution"):
            parts.append(f"  T2: {sc['target_2_contribution']}")
        if sc.get("note"):
            parts.append(f"  note: {sc['note']}")

    # Section recall labels (direct only — supporting is too noisy)
    labels = analysis.get("section_recall_labels", [])
    direct_labels = [lbl for lbl in labels if lbl.get("relevance") == "direct"]
    supporting_count = len(labels) - len(direct_labels)
    if direct_labels:
        parts.append(f"\n**Key sections** ({len(direct_labels)} direct, {supporting_count} supporting hidden):")
        for lbl in direct_labels:
            paper = lbl.get("paper", "?")
            parts.append(f"- `{lbl.get('section', '?')}` in {_highlight_paper(paper, colormap)}")
            if lbl.get("what_it_contains"):
                parts.append(f"  {lbl['what_it_contains']}")

    return "\n".join(parts)


def _normalize_for_match(h: str) -> str:
    """Normalize header for matching against GPT recall labels."""
    h = h.lower().strip()
    h = re.sub(r"^apdx_[a-z][\d.]*:\s*", "", h)
    h = re.sub(r"^apdx:\s*", "", h)
    h = re.sub(r"^[ivxlc]+-?[a-z]?\s+", "", h)
    h = re.sub(r"^[\d.]+\s*", "", h)
    h = re.sub(r"[:\-–—]", " ", h)
    h = re.sub(r"\s+", " ", h).strip()
    return h


def _is_recall_section(header: str, section_text: str, recall_labels: list[dict], arxiv_id: str = "") -> str | None:
    """Check if a section matches any direct recall label for this specific paper. Returns relevance or None."""
    h_norm = _normalize_for_match(header)
    h_lower = header.lower()

    # Only match direct labels for the specific paper
    for lbl in [l for l in recall_labels if l.get("relevance") == "direct"]:
        # Filter by paper: match arxiv_id if available
        lbl_arxiv = lbl.get("arxiv_id", "")
        if arxiv_id and lbl_arxiv and lbl_arxiv != arxiv_id:
            continue
        # If no arxiv_id in label, try matching paper title won't work here,
        # so skip paper filtering (backward compat)
        lbl_sec = lbl.get("section", "")
        lbl_norm = _normalize_for_match(lbl_sec)
        lbl_lower = lbl_sec.lower()

        # Strategy 1: substring on raw
        if lbl_lower in h_lower or h_lower in lbl_lower:
            return lbl.get("relevance", "matched")
        # Strategy 2: normalized
        if lbl_norm and h_norm and (lbl_norm in h_norm or h_norm in lbl_norm):
            return lbl.get("relevance", "matched")
        # Strategy 3: keyword overlap
        if lbl_norm and h_norm:
            lbl_words = set(lbl_norm.split())
            h_words = set(h_norm.split())
            if len(lbl_words) >= 2 and h_words:
                if len(lbl_words & h_words) / max(len(lbl_words), 1) >= 0.6:
                    return lbl.get("relevance", "matched")
        # Strategy 4: subsection in body text
        if lbl_norm and len(lbl_norm) >= 5 and section_text:
            if lbl_norm in section_text.lower() or lbl_lower in section_text.lower():
                return lbl.get("relevance", "matched")

    return None


def build_oracle_view(s: dict, meta: dict | None, analysis: dict | None = None) -> str:
    """Build oracle view with paper links, expandable sections, and recall highlights."""
    sections = []
    pp = s.get("paper_pool", {})
    nodes = s.get("graph", {}).get("nodes", {})
    recall_labels = analysis.get("section_recall_labels", []) if analysis else []

    if not meta:
        return "(No metadata available)"

    # Seed paper
    seed_pid = s.get("seed_paper_id", "")
    seed_info = nodes.get(seed_pid, {})
    seed_arxiv = seed_info.get("arxivId", "")
    seed_links = f"[ar5iv](https://ar5iv.labs.arxiv.org/html/{seed_arxiv}) | [arxiv](https://arxiv.org/abs/{seed_arxiv})" if seed_arxiv else ""
    sections.append(f"**Seed:** {s.get('seed_title', '?')}\n{seed_links}")

    # Bridge papers (st_ depth-2)
    chain = meta.get("chain", {})
    if chain:
        path_titles = chain.get("path_titles", [])
        terminal_title = chain.get("terminal", {}).get("title", "")
        for j, btitle in enumerate(path_titles):
            if btitle == terminal_title:
                continue
            b_arxiv = ""
            for pid, node in nodes.items():
                if node.get("title", "") == btitle:
                    b_arxiv = node.get("arxivId", "")
                    break
            links = f"[ar5iv](https://ar5iv.labs.arxiv.org/html/{b_arxiv}) | [arxiv](https://arxiv.org/abs/{b_arxiv})" if b_arxiv else ""
            sections.append(f"\n---\n**Bridge {j+1}:** {btitle}\narxiv: {b_arxiv} {links}")
            if b_arxiv and b_arxiv in pp:
                for sec in pp[b_arxiv].get("sections", []):
                    header = sec.get("header", "Untitled")
                    text = sec.get("text", "")
                    relevance = _is_recall_section(header, text, recall_labels, b_arxiv)
                    if relevance == "direct":
                        tag = '<span style="background:#c8e6c9; color:#2e7d32; padding:2px 6px; border-radius:4px; font-weight:bold;">RECALL: direct</span>'
                        sections.append(f"\n<details open><summary><b>{header}</b> ({len(text):,} chars) {tag}</summary>\n\n{_format_paper_text(text)}\n\n</details>")
                    elif relevance:
                        tag = f'<span style="background:#e3f2fd; color:#1565c0; padding:2px 6px; border-radius:4px;">RECALL: {relevance}</span>'
                        sections.append(f"\n<details><summary><b>{header}</b> ({len(text):,} chars) {tag}</summary>\n\n{_format_paper_text(text)}\n\n</details>")
                    else:
                        sections.append(f"\n<details><summary><b>{header}</b> ({len(text):,} chars)</summary>\n\n{_format_paper_text(text)}\n\n</details>")

    # Terminal (st_)
    if chain and "terminal" in chain:
        t = chain["terminal"]
        tinfo = nodes.get(t.get("paperId", ""), {})
        t_arxiv = tinfo.get("arxivId", "") or t.get("arxivId", "")
        t_links = f"[ar5iv](https://ar5iv.labs.arxiv.org/html/{t_arxiv}) | [arxiv](https://arxiv.org/abs/{t_arxiv})" if t_arxiv else ""
        sections.append(f"\n---\n**Terminal:** {t.get('title', '?')}\narxiv: {t_arxiv} {t_links}")
        if t_arxiv and t_arxiv in pp:
            for sec in pp[t_arxiv].get("sections", []):
                header = sec.get("header", "Untitled")
                text = sec.get("text", "")
                relevance = _is_recall_section(header, text, recall_labels, t_arxiv)
                if relevance == "direct":
                    tag = '<span style="background:#c8e6c9; color:#2e7d32; padding:2px 6px; border-radius:4px; font-weight:bold;">RECALL: direct</span>'
                    sections.append(f"\n<details open><summary><b>{header}</b> ({len(text):,} chars) {tag}</summary>\n\n{_format_paper_text(text)}\n\n</details>")
                elif relevance:
                    tag = f'<span style="background:#e3f2fd; color:#1565c0; padding:2px 6px; border-radius:4px;">RECALL: {relevance}</span>'
                    sections.append(f"\n<details><summary><b>{header}</b> ({len(text):,} chars) {tag}</summary>\n\n{_format_paper_text(text)}\n\n</details>")
                else:
                    sections.append(f"\n<details><summary><b>{header}</b> ({len(text):,} chars)</summary>\n\n{_format_paper_text(text)}\n\n</details>")

    # Targets (mt_)
    group = meta.get("group", {})
    for j, t in enumerate(group.get("targets", [])):
        tpid = t.get("paperId", "")
        tinfo = nodes.get(tpid, {})
        t_arxiv = tinfo.get("arxivId", "") or t.get("arxivId", "")
        t_links = f"[ar5iv](https://ar5iv.labs.arxiv.org/html/{t_arxiv}) | [arxiv](https://arxiv.org/abs/{t_arxiv})" if t_arxiv else ""
        sections.append(f"\n---\n**Target {j+1}:** {t.get('title', '?')}\narxiv: {t_arxiv} {t_links}")
        if t_arxiv and t_arxiv in pp:
            for sec in pp[t_arxiv].get("sections", []):
                header = sec.get("header", "Untitled")
                text = sec.get("text", "")
                relevance = _is_recall_section(header, text, recall_labels, t_arxiv)
                if relevance == "direct":
                    tag = '<span style="background:#c8e6c9; color:#2e7d32; padding:2px 6px; border-radius:4px; font-weight:bold;">RECALL: direct</span>'
                    sections.append(f"\n<details open><summary><b>{header}</b> ({len(text):,} chars) {tag}</summary>\n\n{_format_paper_text(text)}\n\n</details>")
                elif relevance:
                    tag = f'<span style="background:#e3f2fd; color:#1565c0; padding:2px 6px; border-radius:4px;">RECALL: {relevance}</span>'
                    sections.append(f"\n<details><summary><b>{header}</b> ({len(text):,} chars) {tag}</summary>\n\n{_format_paper_text(text)}\n\n</details>")
                else:
                    sections.append(f"\n<details><summary><b>{header}</b> ({len(text):,} chars)</summary>\n\n{_format_paper_text(text)}\n\n</details>")

    return "\n".join(sections) if sections else "(No oracle content)"


def build_app(review_items: list[dict], samples: dict, meta_map: dict,
              analyses: dict, save_dir: Path):

    save_dir.mkdir(parents=True, exist_ok=True)

    _css = """
    .oracle-section { background: #f8f9fa; padding: 16px 20px; border-radius: 10px;
                       border-left: 4px solid #1a73e8; margin: 12px 0; }
    .option-correct { background: #c8e6c9; padding: 8px 12px; border-radius: 6px; margin: 4px 0; }
    .option-wrong { background: #f5f5f5; padding: 8px 12px; border-radius: 6px; margin: 4px 0; }
    .ai-guidance { background: #f3e5f5; padding: 16px 20px; border-radius: 10px;
                   border-left: 4px solid #8e24aa; margin: 0 0 12px 0; }
    .ai-guidance h3 { color: #6a1b9a; margin-top: 0; }
    .existing-audit { background: #fff3e0; padding: 12px 16px; border-radius: 8px;
                      border-left: 4px solid #f57c00; margin: 8px 0; }
    .audit-bar { position: fixed; bottom: 0; left: 0; right: 0; z-index: 999;
                 background: #fff; border-top: 2px solid #1a73e8;
                 padding: 12px 24px; box-shadow: 0 -2px 8px rgba(0,0,0,0.1); }
    .main-content { margin-bottom: 180px; }
    """

    with gr.Blocks(title="AgentHop Review", css=_css) as app:
        state = gr.State(value={"idx": 0, "start_time": time.time()})

        with gr.Column(elem_classes=["main-content"]):
            with gr.Row():
                progress_md = gr.Markdown("Loading...", elem_classes=["audit-header"])
                counts_md = gr.Markdown("", elem_classes=["audit-header"])

            with gr.Row(equal_height=False):
                # LEFT: question + options + existing audit
                with gr.Column(scale=2, min_width=400):
                    info_md = gr.Markdown()
                    question_md = gr.Markdown()
                    options_md = gr.Markdown()
                    existing_md = gr.Markdown()

                # RIGHT: GPT analysis + oracle
                with gr.Column(scale=3):
                    ai_guidance_md = gr.Markdown()
                    oracle_md = gr.Markdown()

        # Fixed bar at bottom
        with gr.Row(elem_classes=["audit-bar"]):
            supported_radio = gr.Radio(
                choices=["Yes", "Partially", "No"],
                label="Answer supported?",
                interactive=True,
                scale=2,
            )
            unambiguous_radio = gr.Radio(
                choices=["Yes", "Partially", "No"],
                label="Q&A unambiguous?",
                interactive=True,
                scale=2,
            )
            verdict_radio = gr.Radio(
                choices=["Pass", "Flag", "Remove"],
                label="Verdict",
                interactive=True,
                scale=1,
            )
            notes_box = gr.Textbox(
                label="Notes",
                placeholder="Any issues?",
                lines=1,
                scale=2,
            )
            submit_btn = gr.Button("Submit & Next", variant="primary", size="lg", scale=1)
            skip_btn = gr.Button("Skip", variant="secondary", size="lg", scale=1)

        def _count_progress():
            done = 0
            verdicts = {"Pass": 0, "Flag": 0, "Remove": 0}
            for f in save_dir.glob("*.json"):
                try:
                    with open(f) as fh:
                        r = json.load(fh)
                    v = r.get("verdict", "")
                    if v in verdicts:
                        verdicts[v] += 1
                    done += 1
                except:
                    pass
            return done, verdicts

        def render_sample(idx, st=None):
            if st is not None:
                st["start_time"] = time.time()
            if idx >= len(review_items):
                done, verdicts = _count_progress()
                return ("# Review Complete!\n\n"
                        f"**Pass**: {verdicts['Pass']} | **Flag**: {verdicts['Flag']} | **Remove**: {verdicts['Remove']} | **Total**: {done}",
                        "", "", "", "", "", "", "",
                        None, None, None, "")

            entry = review_items[idx]
            sid = entry["sample_id"]
            s = samples.get(sid)

            if not s:
                return (f"**{idx+1}/{len(review_items)}** — `{sid}` — sample not found",
                        "", "", "", "", "", "", "",
                        None, None, None, "")

            meta = meta_map.get(sid)
            if not meta:
                old_id = ("q_" + sid[3:]) if sid.startswith("st_") else ("dfs_" + sid[3:]) if sid.startswith("mt_") else sid
                meta = meta_map.get(old_id)

            # Progress
            sources = ", ".join(entry.get("sources", []))
            progress = f"**Sample {idx + 1} / {len(review_items)}** — `{sid}` — [{sources}]"
            done, verdicts = _count_progress()
            counts = f"Done: {done} | Pass: {verdicts['Pass']} | Flag: {verdicts['Flag']} | Remove: {verdicts['Remove']}"

            # Info with seed links
            seed_info = s.get("graph", {}).get("nodes", {}).get(s.get("seed_paper_id", ""), {})
            seed_arxiv = seed_info.get("arxivId", "")
            seed_links = f"[ar5iv](https://ar5iv.labs.arxiv.org/html/{seed_arxiv}) | [arxiv](https://arxiv.org/abs/{seed_arxiv})" if seed_arxiv else ""
            seed_authors = ", ".join(a.get("name", "") for a in seed_info.get("authors", [])[:4])
            info = (
                f"`{s.get('question_type', '?')}` / depth {s.get('depth', '?')} / "
                f"{s.get('consensus_tier', '?')} / {s.get('reasoning_type', '?')} / {s.get('cognitive_skill', '?')}\n\n"
                f"**Seed:** {s.get('seed_title', '?')} — {seed_authors or '?'} ({seed_info.get('year', '?')})\n{seed_links}"
            )

            # Question
            question = f"## Question\n\n{s.get('question', '?')}"

            # Options
            ci = s.get("correct_index", 0)
            opts = s.get("options", [])
            opts_lines = []
            for i, opt in enumerate(opts):
                if i == ci:
                    opts_lines.append(f'<div class="option-correct">\n\n### OK ({OPTION_LABELS[i]}) CORRECT\n\n{opt}\n\n</div>')
                else:
                    opts_lines.append(f'<div class="option-wrong">\n\n**({OPTION_LABELS[i]})** {opt}\n\n</div>')
            options_text = "\n".join(opts_lines)

            # Existing audit
            existing = load_existing_audit(sid)
            if existing:
                ev = existing.get("verdict", "?")
                en = existing.get("notes", "")
                ea = existing.get("_auditor", "") or ""
                auto = " [auto-unflagged]" if "Auto-unflagged" in en else ""
                existing_text = f'<div class="existing-audit">\n\n**Previous:** {ev}{auto} ({ea})\n\n**Notes:** {en}\n\n</div>'
            else:
                existing_text = ""

            # GPT analysis
            ai_data = analyses.get(sid)
            rendered = render_en_analysis(ai_data, entry)
            ai_guidance = f'<div class="ai-guidance">\n\n### GPT-5.4 Audit Guide\n\n{rendered}\n\n</div>'

            # Oracle
            oracle = build_oracle_view(s, meta, ai_data)

            # Pre-fill from existing
            prefill_supported = existing.get("supported") if existing else None
            prefill_unambiguous = existing.get("unambiguous") if existing else None
            prefill_verdict = existing.get("verdict") if existing else None
            prefill_notes = existing.get("notes", "") if existing else ""
            if prefill_notes and "Auto-unflagged" in prefill_notes:
                prefill_notes = ""

            return (progress, counts, info, question, options_text, existing_text,
                    ai_guidance, oracle,
                    prefill_supported, prefill_unambiguous, prefill_verdict, prefill_notes)

        def on_submit(st, supported, unambiguous, verdict, notes):
            idx = st["idx"]
            if idx >= len(review_items):
                return _done_return(st)

            entry = review_items[idx]
            sid = entry["sample_id"]
            s = samples.get(sid, {})

            result = {
                "sample_id": sid,
                "question_type": s.get("question_type"),
                "depth": s.get("depth"),
                "supported": supported,
                "unambiguous": unambiguous,
                "verdict": verdict,
                "notes": notes or "",
                "review_sources": entry.get("sources", []),
                "review_reasons": entry.get("reasons", []),
                "wall_time_s": time.time() - st["start_time"],
            }

            with open(save_dir / f"{sid}.json", "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            st["idx"] = idx + 1
            st["start_time"] = time.time()

            vals = render_sample(st["idx"], st)
            return (st,) + vals

        def on_skip(st):
            idx = st["idx"]
            st["idx"] = idx + 1
            st["start_time"] = time.time()
            vals = render_sample(st["idx"], st)
            return (st,) + vals

        def _done_return(st):
            vals = render_sample(len(review_items), st)
            return (st,) + vals

        def on_load():
            return render_sample(0)

        state.value["start_time"] = time.time()

        all_outputs = [progress_md, counts_md, info_md, question_md, options_md,
                       existing_md, ai_guidance_md, oracle_md,
                       supported_radio, unambiguous_radio, verdict_radio, notes_box]

        app.load(on_load, outputs=all_outputs)

        submit_btn.click(
            on_submit,
            inputs=[state, supported_radio, unambiguous_radio, verdict_radio, notes_box],
            outputs=[state] + all_outputs,
        )

        skip_btn.click(
            on_skip,
            inputs=[state],
            outputs=[state] + all_outputs,
        )

    return app


def main():
    parser = argparse.ArgumentParser(description="Review flagged samples")
    parser.add_argument("--port", type=int, default=7872)
    parser.add_argument("--auditor", type=str, default=None)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    print("Collecting review items...")
    review_items = collect_review_items()
    print(f"  {len(review_items)} samples to review")

    print("Loading benchmark samples (flagged only)...")
    review_sids = {r["sample_id"] for r in review_items}
    samples = {}
    for sid in review_sids:
        fpath = BENCHMARK_DIR / f"{sid}.json"
        if fpath.exists():
            with open(fpath) as f:
                s = json.load(f)
            if not isinstance(s, list):
                samples[s["id"]] = s
    print(f"  {len(samples)} samples loaded")

    print("Loading metadata + analyses...")
    meta_map = load_metadata()
    analyses = load_en_analyses()
    print(f"  {len(meta_map)} gold chains, {len(analyses)} EN analyses")

    save_dir = LOG_DIR / (args.auditor or "review")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Resume: skip already-reviewed
    reviewed = set()
    for f in save_dir.glob("*.json"):
        try:
            with open(f) as fh:
                r = json.load(fh)
            if r.get("review_sources"):
                reviewed.add(r.get("sample_id"))
        except:
            pass

    if reviewed:
        before = len(review_items)
        review_items = [r for r in review_items if r["sample_id"] not in reviewed]
        print(f"  Resuming: skipped {before - len(review_items)}, {len(review_items)} remaining")

    app = build_app(review_items, samples, meta_map, analyses, save_dir)
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
