"""Stage 7 (auditor UI): seven-auditor Gradio interface for the main quality verdict.

Unlike the human evaluation form (which simulates tool-based navigation),
this form shows the gold chain and paper content directly. The auditor
reads and judges whether the question is answerable and well-formed.

Usage:
    python stage7_quality_app.py --data ../pipeline/data/benchmark/lite
    python stage7_quality_app.py --data ../pipeline/data/benchmark/full --port 7871
"""

import argparse
import json
import glob
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import gradio as gr

# Load gold chain metadata
MCQ_VALIDATED = Path(__file__).resolve().parent / "data/questions/mcq_combined_validated.json"
LOG_DIR = Path(__file__).resolve().parent / "audit_results"
OPTION_LABELS = ["A", "B", "C", "D"]


def load_metadata() -> dict:
    """Load gold chain/group info keyed by sample ID."""
    with open(MCQ_VALIDATED) as f:
        validated = json.load(f)
    meta = {}
    for v in validated:
        meta[v["id"]] = v
    return meta


def load_samples(data_path: str, start: int | None = None, end: int | None = None) -> list[dict]:
    """Load benchmark samples, optionally only a range by index.

    start/end are 1-based inclusive. Only the files in the range are loaded,
    so memory usage stays low even for the full 1K+ benchmark.
    """
    p = Path(data_path)
    if p.is_file():
        with open(p) as f:
            data = json.load(f)
        samples = data if isinstance(data, list) else [data]
        if start or end:
            s = (start or 1) - 1
            e = (end - 1) if end else len(samples)
            samples = samples[s:e]
        return samples

    files = sorted([f for f in glob.glob(str(p / "*.json"))
                    if not Path(f).name.startswith("agenthop")])

    # Apply range BEFORE loading — only read the files we need
    if start or end:
        s = (start or 1) - 1
        e = (end - 1) if end else len(files)
        total = len(files)
        files = files[s:e]
        print(f"  Range {s+1}-{end-1 if end else total} of {total} files ({len(files)} to load)")

    samples = []
    for i, fpath in enumerate(files):
        with open(fpath) as f:
            sample = json.load(f)
        if "id" in sample and "question" in sample:
            samples.append(sample)
        if (i + 1) % 50 == 0:
            print(f"  Loaded {i+1}/{len(files)}...")

    return samples


def _format_paper_text(text: str) -> str:
    """Add basic formatting to raw paper text for readability."""
    import re

    # Clean up citation brackets: [ 20 ] → [20]
    text = re.sub(r'\[\s*(\d+)\s*\]', r'[\1]', text)
    # Clean multi-cite: [ 20 , 4 , 24 ] → [20, 4, 24]
    text = re.sub(r'\[\s*([\d\s,]+)\s*\]', lambda m: '[' + ', '.join(m.group(1).split()) + ']', text)

    # Add paragraph breaks at sentence boundaries after long runs
    # Split on ". " followed by uppercase (new sentence) if line is very long
    lines = text.split('\n')
    formatted = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # If line is very long (>500 chars), add paragraph breaks
        if len(line) > 500:
            # Break at ". [A-Z]" patterns (sentence boundaries)
            line = re.sub(r'(\. )([A-Z])', r'.\n\n\2', line)
        formatted.append(line)

    return '\n\n'.join(formatted)


def build_oracle_view(sample: dict, meta: dict | None) -> str:
    """Build the oracle information panel for a sample."""
    nodes = sample["graph"]["nodes"]
    edges = sample["graph"]["edges"]
    pp = sample["paper_pool"]
    seed = sample["seed_paper_id"]
    qtype = sample.get("question_type", "?")
    depth = sample.get("depth", 0)

    parts = []

    if not meta:
        return "*No gold chain metadata available for this sample.*"

    def _section(cls, title, content_lines):
        body = "\n".join(content_lines)
        return f'<div class="{cls}">\n\n## {title}\n\n{body}\n\n</div>\n'

    sections = []

    # ── 1. Key Evidence (what auditors should check first) ─────────────
    evidence_lines = []

    # Source passage — the exact text the answer was derived from
    answer_ctx = meta.get("answer_context", "")
    if answer_ctx:
        evidence_lines.append(f"**Source passage** (the text the correct answer was derived from):\n")
        evidence_lines.append(f"> *{answer_ctx}*\n")

    # For DFS: per-paper facts
    answer_sources = meta.get("answer_sources", {})
    if answer_sources and not answer_ctx:
        for k in sorted(answer_sources.keys()):
            fact = answer_sources[k].get("fact", "")
            if fact:
                evidence_lines.append(f"**{k} verbatim extract** (copied from paper during generation):\n\n> *{fact}*\n")

    # Chain / target info
    if qtype in ("bfs", "single-target"):
        chain = meta.get("chain", {})
        if chain:
            path_titles = chain.get("path_titles", [])
            terminal_pid = chain.get("terminal", {}).get("paperId", "")
            t_info = nodes.get(terminal_pid, {})
            t_authors = ", ".join(a.get("name", "") for a in t_info.get("authors", [])[:4])

            chain_vis = f"**Seed** &rarr; {' &rarr; '.join(f'**{t}**' for t in path_titles)}"
            evidence_lines.append(chain_vis + "\n")

            if depth == 2:
                for ref in edges.get(seed, []):
                    if ref in edges and terminal_pid in edges[ref]:
                        b_info = nodes.get(ref, {})
                        b_authors = ", ".join(a.get("name", "") for a in b_info.get("authors", [])[:3])
                        b_arxiv = b_info.get("arxivId", "")
                        b_links = f" — [ar5iv](https://ar5iv.labs.arxiv.org/html/{b_arxiv}) | [arxiv](https://arxiv.org/abs/{b_arxiv})" if b_arxiv else ""
                        evidence_lines.append(f"**Bridge:** {b_info.get('title', 'Unknown')} — {b_authors or 'Unknown'} ({b_info.get('year', '?')}){b_links}")
                        b_abstract = b_info.get("abstract", "")
                        if b_abstract:
                            evidence_lines.append(f"> {b_abstract[:300]}{'...' if len(b_abstract) > 300 else ''}\n")
                        else:
                            evidence_lines.append("")
                        break

            t_arxiv = t_info.get("arxivId", "")
            t_abstract = t_info.get("abstract", "")
            t_links = f" — [ar5iv](https://ar5iv.labs.arxiv.org/html/{t_arxiv}) | [arxiv](https://arxiv.org/abs/{t_arxiv})" if t_arxiv else ""
            evidence_lines.append(f"**Terminal:** {t_info.get('title', 'Unknown')} — {t_authors or 'Unknown'} ({t_info.get('year', '?')}){t_links}")
            if t_abstract:
                evidence_lines.append(f"> {t_abstract[:300]}{'...' if len(t_abstract) > 300 else ''}\n")
            else:
                evidence_lines.append("")

    elif qtype in ("dfs", "multi-target"):
        group = meta.get("group", {})
        targets = group.get("targets", [])
        for i, t in enumerate(targets):
            t_pid = t.get("paperId", "")
            t_info = nodes.get(t_pid, {})
            t_authors = ", ".join(a.get("name", "") for a in t_info.get("authors", [])[:3])
            t_arxiv = t_info.get("arxivId", "") or t.get("arxivId", "")
            t_abstract = t_info.get("abstract", "")
            t_links = f" — [ar5iv](https://ar5iv.labs.arxiv.org/html/{t_arxiv}) | [arxiv](https://arxiv.org/abs/{t_arxiv})" if t_arxiv else ""
            evidence_lines.append(f"**Target {i+1}:** {t.get('title', t_info.get('title', 'Unknown'))} — {t_authors or 'Unknown'} ({t_info.get('year', '?')}){t_links}")
            if t_abstract:
                evidence_lines.append(f"> {t_abstract[:300]}{'...' if len(t_abstract) > 300 else ''}\n")
            else:
                evidence_lines.append("")

    if evidence_lines:
        sections.append(_section("answer-ground", "Key Evidence", evidence_lines))

    # ── 2. Pipeline Reasoning (why this chain) ────────────────────────
    reasoning_lines = []
    hops = meta.get("hops_required", "")
    if hops:
        reasoning_lines.append(f"{hops}\n")

    seed_detail = meta.get("seed_detail", "")
    if seed_detail:
        reasoning_lines.append(f"<details><summary><b>Seed citation detail</b></summary>\n\n{seed_detail}\n\n</details>\n")

    why = meta.get("why_multi_paper", "")
    if why:
        reasoning_lines.append(f"{why}\n")

    if reasoning_lines:
        sections.append(_section("oracle-section", "Pipeline Reasoning", reasoning_lines))

    # ── 3. Full Paper Sections (optional deep-dive) ───────────────────
    paper_lines = []

    if qtype in ("bfs", "single-target"):
        chain = meta.get("chain", {})
        if chain:
            terminal_pid = chain.get("terminal", {}).get("paperId", "")
            t_info = nodes.get(terminal_pid, {})
            t_arxiv = t_info.get("arxivId", "")
            if t_arxiv and t_arxiv in pp:
                secs = pp[t_arxiv].get("sections", [])
                paper_lines.append(f"**{t_info.get('title', 'Terminal')}** ({len(secs)} sections):\n")
                for sec in secs:
                    header = sec.get("header", "Untitled")
                    text = sec.get("text", "")
                    paper_lines.append(f"<details><summary><b>{header}</b> ({len(text):,} chars)</summary>\n\n{_format_paper_text(text)}\n\n</details>\n")

    elif qtype in ("dfs", "multi-target"):
        group = meta.get("group", {})
        targets = group.get("targets", [])
        for i, t in enumerate(targets):
            t_pid = t.get("paperId", "")
            t_info = nodes.get(t_pid, {})
            t_arxiv = t_info.get("arxivId", "") or t.get("arxivId", "")
            if t_arxiv and t_arxiv in pp:
                secs = pp[t_arxiv].get("sections", [])
                paper_lines.append(f"**Target {i+1}: {t_info.get('title', t.get('title', 'Unknown'))}** ({len(secs)} sections):\n")
                for sec in secs:
                    header = sec.get("header", "Untitled")
                    text = sec.get("text", "")
                    paper_lines.append(f"<details><summary><b>{header}</b> ({len(text):,} chars)</summary>\n\n{_format_paper_text(text)}\n\n</details>\n")

    if paper_lines:
        sections.append(_section("chain-section", "Full Paper Sections (optional)", paper_lines))

    return "\n".join(sections)


def _load_analyses() -> dict:
    """Load pre-generated AI analyses keyed by sample ID.

    Loads the English audit JSON (canonical 6-section schema from Appendix E).
    """
    analyses_path = Path(__file__).resolve().parent / "audit_analyses_en_full.json"
    if not analyses_path.exists():
        return {}
    with open(analyses_path) as f:
        data = json.load(f)
    result = {}
    for item in data:
        analysis = item.get("analysis")
        if not analysis:
            continue
        # The English file stores `analysis` as an already-parsed dict;
        # the legacy Korean file stored it as a JSON string. Handle both.
        if isinstance(analysis, str):
            try:
                analysis = json.loads(analysis)
            except (json.JSONDecodeError, TypeError):
                analysis = {"_raw": analysis}
        result[item["id"]] = analysis
    return result


def _render_analysis(analysis: dict) -> str:
    """Render structured JSON analysis as readable Markdown.

    Renders the canonical six-section English audit schema (Appendix E):
    query_quality, answer_validity, chain_coherence, synthesis_check,
    section_recall_labels, structural_integrity.
    """
    if "_raw" in analysis:
        return analysis["_raw"]

    icon = lambda v: {"pass": "✅", "warn": "⚠️", "fail": "❌"}.get(v or "", "")
    parts = []

    # Query quality
    qq = analysis.get("query_quality", {})
    note = qq.get("note", "")
    parts.append(f"**Query quality** {icon(qq.get('verdict'))}" + (f": {note}" if note else ""))

    # Answer validity (claim-by-claim)
    av = analysis.get("answer_validity", {})
    parts.append(f"\n**Answer validity** {icon(av.get('verdict'))}:")
    for claim in av.get("claims", []):
        check = "✅" if claim.get("verified") else "❌"
        src_paper = (claim.get("source_paper") or "?")[:40]
        src_section = claim.get("source_section") or "?"
        note = f" — {claim['note']}" if claim.get("note") else ""
        parts.append(f"- {check} {claim.get('claim', '')} → `{src_paper} / {src_section}`{note}")

    # Chain coherence (single-target)
    cc = analysis.get("chain_coherence")
    if cc and cc.get("verdict") is not None:
        cc_note = cc.get("note", "")
        parts.append(f"\n**Chain coherence** {icon(cc.get('verdict'))}" + (f": {cc_note}" if cc_note else ""))

    # Synthesis check (multi-target)
    sc = analysis.get("synthesis_check")
    if sc and sc.get("verdict") is not None:
        sc_note = sc.get("note", "")
        parts.append(f"\n**Synthesis check** {icon(sc.get('verdict'))}" + (f": {sc_note}" if sc_note else ""))

    # Section recall labels
    srl = analysis.get("section_recall_labels", [])
    if srl:
        parts.append("\n**Section recall labels:**")
        for label in srl:
            paper = (label.get("paper") or "?")[:40]
            section = label.get("section") or "?"
            relevance = label.get("relevance") or "?"
            note = label.get("what_it_contains") or ""
            parts.append(f"- 📄 **{paper}** → `{section}` (*{relevance}*){': ' + note if note else ''}")

    # Structural integrity
    si = analysis.get("structural_integrity", {})
    si_note = si.get("note", "")
    parts.append(f"\n**Structural integrity** {icon(si.get('verdict'))}" + (f": {si_note}" if si_note else ""))
    missing = si.get("missing_sections", [])
    if missing:
        parts.append(f"  Missing sections: {', '.join(missing)}")

    return "\n".join(parts)


def build_app(samples: list[dict], meta_map: dict, log_dir: Path | None = None):
    save_dir = log_dir or LOG_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load pre-generated AI analyses
    analyses = _load_analyses()

    _css = """
    .oracle-section { background: #f8f9fa; padding: 16px 20px; border-radius: 10px;
                       border-left: 4px solid #1a73e8; margin: 12px 0; }
    .oracle-section h3 { color: #1a73e8; font-size: 1.3em; margin-top: 0; }
    .answer-ground { background: #e8f5e9; padding: 16px 20px; border-radius: 10px;
                     border-left: 4px solid #43a047; margin: 12px 0; }
    .answer-ground h3 { color: #2e7d32; }
    .distractor-section { background: #fff3e0; padding: 16px 20px; border-radius: 10px;
                          border-left: 4px solid #f57c00; margin: 12px 0; }
    .distractor-section h3 { color: #e65100; }
    .chain-section { background: #e3f2fd; padding: 16px 20px; border-radius: 10px;
                     border-left: 4px solid #1565c0; margin: 12px 0; }
    .chain-section h3 { color: #0d47a1; }
    .question-panel { font-size: 1.1em; line-height: 1.7; }
    .question-panel h3 { font-size: 1.4em; }
    .audit-header { font-size: 1.1em; font-weight: bold; }
    .option-correct { background: #c8e6c9; padding: 8px 12px; border-radius: 6px; margin: 4px 0; }
    .option-wrong { background: #f5f5f5; padding: 8px 12px; border-radius: 6px; margin: 4px 0; }
    .ai-guidance { background: #f3e5f5; padding: 16px 20px; border-radius: 10px;
                   border-left: 4px solid #8e24aa; margin: 0 0 12px 0; }
    .ai-guidance h3 { color: #6a1b9a; margin-top: 0; }
    .audit-bar { position: fixed; bottom: 0; left: 0; right: 0; z-index: 999;
                 background: #fff; border-top: 2px solid #1a73e8;
                 padding: 12px 24px; box-shadow: 0 -2px 8px rgba(0,0,0,0.1); }
    .main-content { margin-bottom: 220px; }
    """

    with gr.Blocks(title="AgentHop Quality Audit", css=_css) as app:

        state = gr.State(value={"idx": 0, "start_time": time.time()})

        # Scrollable main content
        with gr.Column(elem_classes=["main-content"]):
            # Header
            with gr.Row():
                progress_md = gr.Markdown("Sample 1 / ?", elem_classes=["audit-header"])
                skip_count_md = gr.Markdown("Flagged: 0", elem_classes=["audit-header"])

            # Main layout: left (question) | right (oracle)
            with gr.Row(equal_height=False):
                # LEFT: Question + correct answer + metadata
                with gr.Column(scale=2, min_width=400):
                    info_md = gr.Markdown()
                    question_md = gr.Markdown(elem_classes=["question-panel"])
                    options_md = gr.Markdown()

                # RIGHT: AI guidance + Oracle view
                with gr.Column(scale=3):
                    ai_guidance_md = gr.Markdown()
                    oracle_md = gr.Markdown()

        # Fixed audit bar at bottom
        with gr.Row(elem_classes=["audit-bar"]):
            supported_radio = gr.Radio(
                choices=["Yes", "Partially", "No"],
                label="Pipeline reasoning legitimate?",
                interactive=True,
                scale=2,
            )
            unambiguous_radio = gr.Radio(
                choices=["Yes", "Partially", "No"],
                label="Gold content reflected in Q&A?",
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

        # Hidden — kept for compatibility with submit handler
        distractors_radio = gr.Radio(choices=["Yes", "No"], visible=False, value="Yes")
        identifiable_radio = gr.Radio(choices=["Yes", "No"], visible=False, value="Yes")

        # Helpers
        def render_sample(idx, st=None):
            """Render sample at idx. Resets start_time on the state if provided."""
            if st is not None:
                st["start_time"] = time.time()
            if idx >= len(samples):
                return ("Done!", "", "", "", "", "", None, None, None, None, None, "")

            s = samples[idx]
            sid = s["id"]
            meta = meta_map.get(sid)
            if not meta:
                old_id = sid
                if sid.startswith("st_"):
                    old_id = "q_" + sid[3:]
                elif sid.startswith("mt_"):
                    old_id = "dfs_" + sid[3:]
                meta = meta_map.get(old_id)

            qtype = s.get("question_type", "?")
            depth = s.get("depth", 0)
            tier = s.get("consensus_tier", "?")

            progress = f"**Sample {idx + 1} / {len(samples)}** — `{sid}`"

            # Seed paper info with abstract
            seed_info = s.get("graph", {}).get("nodes", {}).get(s.get("seed_paper_id", ""), {})
            seed_arxiv = seed_info.get("arxivId", "")
            seed_abstract = seed_info.get("abstract", "")
            seed_links = f"[ar5iv](https://ar5iv.labs.arxiv.org/html/{seed_arxiv}) | [arxiv](https://arxiv.org/abs/{seed_arxiv})" if seed_arxiv else ""
            seed_authors = ", ".join(a.get("name", "") for a in seed_info.get("authors", [])[:4])

            info_parts = [
                f"`{qtype}` / depth {depth} / {tier} / "
                f"{s.get('reasoning_type', '?')} / {s.get('cognitive_skill', '?')} / "
                f"{s.get('venue', '?')}",
                f"\n\n**Seed:** {s.get('seed_title', '?')} — {seed_authors or 'Unknown'} ({seed_info.get('year', '?')})",
            ]
            if seed_links:
                info_parts.append(seed_links)
            if seed_abstract:
                info_parts.append(f"\n\n<details><summary><b>Seed Abstract</b></summary>\n\n{seed_abstract}\n\n</details>")
            info = "\n".join(info_parts)

            # Question
            question = f"## Question\n\n{s['question']}"

            # Options with correct answer highlighted
            ci = s["correct_index"]
            opts_lines = []
            for i, opt in enumerate(s["options"]):
                if i == ci:
                    opts_lines.append(f'<div class="option-correct">\n\n### ✓ ({OPTION_LABELS[i]}) CORRECT\n\n{opt}\n\n</div>')
                else:
                    opts_lines.append(f'<div class="option-wrong">\n\n**({OPTION_LABELS[i]})** {opt}\n\n</div>')
            options_text = "\n".join(opts_lines)

            oracle = build_oracle_view(s, meta)

            # AI guidance (pre-generated structured JSON)
            ai_data = analyses.get(sid)
            if not ai_data:
                old_lookup = ("q_" + sid[3:]) if sid.startswith("st_") else ("dfs_" + sid[3:]) if sid.startswith("mt_") else sid
                ai_data = analyses.get(old_lookup)
            if ai_data:
                rendered = _render_analysis(ai_data)
                ai_guidance = f'<div class="ai-guidance">\n\n### AI Audit Guide\n\n{rendered}\n\n</div>'
            else:
                ai_guidance = ""

            return (progress, question, options_text, info, ai_guidance, oracle,
                    None, None, None, None, None, "")

        def on_submit(st, supported, identifiable, unambiguous, distractors, verdict, notes):
            idx = st["idx"]
            if idx >= len(samples):
                return (st, "Done!", "", "", "", "", "",
                        None, None, None, None, None, "",
                        gr.update())

            s = samples[idx]

            # Save audit result
            result = {
                "sample_id": s["id"],
                "question_type": s.get("question_type"),
                "depth": s.get("depth"),
                "supported": supported,
                "identifiable": identifiable,
                "unambiguous": unambiguous,
                "distractors_ok": distractors,
                "verdict": verdict,
                "notes": notes or "",
                "wall_time_s": time.time() - st["start_time"],
            }

            with open(save_dir / f"{s['id']}.json", "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            # Count flags
            flag_count = len([f for f in save_dir.glob("*.json")
                            if json.load(open(f)).get("verdict") in ("Flag for review", "Remove")])

            # Next sample
            st["idx"] = idx + 1
            st["start_time"] = time.time()

            if st["idx"] >= len(samples):
                total = len(samples)
                passed = sum(1 for f in save_dir.glob("*.json")
                           if json.load(open(f)).get("verdict") == "Pass")
                flagged = sum(1 for f in save_dir.glob("*.json")
                            if json.load(open(f)).get("verdict") == "Flag for review")
                removed = sum(1 for f in save_dir.glob("*.json")
                            if json.load(open(f)).get("verdict") == "Remove")
                done_text = (
                    f"# Audit Complete!\n\n"
                    f"**Pass**: {passed} | **Flag**: {flagged} | **Remove**: {removed} | **Total**: {total}\n\n"
                    f"Results saved to `{save_dir}`"
                )
                return (st, done_text, "", "", "", "", "",
                        None, None, None, None, None, "",
                        f"Flagged: {flag_count}")

            vals = render_sample(st["idx"], st)
            return (st,) + vals + (f"Flagged: {flag_count}",)

        # Initial render
        def on_load():
            vals = render_sample(0)
            return vals

        # Also reset timer when app loads
        state.value["start_time"] = time.time()

        app.load(
            on_load,
            outputs=[progress_md, question_md, options_md, info_md, ai_guidance_md, oracle_md,
                     supported_radio, identifiable_radio, unambiguous_radio,
                     distractors_radio, verdict_radio, notes_box],
        )

        submit_btn.click(
            on_submit,
            inputs=[state, supported_radio, identifiable_radio,
                    unambiguous_radio, distractors_radio, verdict_radio, notes_box],
            outputs=[state, progress_md, question_md, options_md, info_md, ai_guidance_md, oracle_md,
                     supported_radio, identifiable_radio, unambiguous_radio,
                     distractors_radio, verdict_radio, notes_box,
                     skip_count_md],
        )

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AgentHop Quality Audit",
        epilog="""
Multi-auditor usage examples:
  # Auditor 1: samples 1-50
  python stage7_quality_app.py --data ../pipeline/data/benchmark/lite --start 1 --end 50 --auditor alice

  # Auditor 2: samples 51-100
  python stage7_quality_app.py --data ../pipeline/data/benchmark/lite --start 51 --end 100 --auditor alice

  # See sample index listing
  python stage7_quality_app.py --data ../pipeline/data/benchmark/lite --list
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--start", type=int, default=None,
                        help="Start index (1-based, inclusive). For splitting work across auditors.")
    parser.add_argument("--end", type=int, default=None,
                        help="End index (1-based, inclusive)")
    parser.add_argument("--auditor", type=str, default=None,
                        help="Auditor name (used in log filenames to avoid collisions)")
    parser.add_argument("--list", action="store_true",
                        help="List all sample IDs with indices and exit")
    parser.add_argument("--status", action="store_true",
                        help="Show audit progress and exit")
    parser.add_argument("--port", type=int, default=7871)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    # List mode: only read filenames + minimal JSON fields, no full load
    if args.list:
        p = Path(args.data)
        files = sorted([f for f in glob.glob(str(p / "*.json"))
                        if not Path(f).name.startswith("agenthop")])
        print(f"\n{'Idx':>4}  {'ID':<16}  {'Type':<15}  {'Depth':>5}  {'Tier':<6}  {'File'}")
        print("-" * 75)
        for i, fpath in enumerate(files):
            with open(fpath) as f:
                # Read only first ~500 bytes for metadata
                raw = f.read(500)
            import re
            sid = re.search(r'"id"\s*:\s*"([^"]+)"', raw)
            qt = re.search(r'"question_type"\s*:\s*"([^"]+)"', raw)
            dep = re.search(r'"depth"\s*:\s*(\d+)', raw)
            tier = re.search(r'"consensus_tier"\s*:\s*"([^"]+)"', raw)
            print(f"{i+1:>4}  {sid.group(1) if sid else '?':<16}  {qt.group(1) if qt else '?':<15}  {dep.group(1) if dep else '?':>5}  {tier.group(1) if tier else '?':<6}  {Path(fpath).name}")
        print(f"\nTotal: {len(files)} files")
        sys.exit(0)

    # Status mode: show audit progress
    if args.status:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        audited_ids = set()
        verdicts = {}
        for f in LOG_DIR.glob("*.json"):
            with open(f) as fh:
                r = json.load(fh)
            audited_ids.add(r.get("sample_id"))
            v = r.get("verdict", "?")
            verdicts[v] = verdicts.get(v, 0) + 1

        sample_ids = {s["id"] for s in samples}
        done = audited_ids & sample_ids
        remaining = sample_ids - audited_ids

        print(f"\nAudit progress: {len(done)}/{len(samples)} ({len(done)/len(samples)*100:.1f}%)")
        for v, cnt in sorted(verdicts.items()):
            print(f"  {v}: {cnt}")
        print(f"  Remaining: {len(remaining)}")

        if remaining:
            # Find first unaudited index
            for i, s in enumerate(samples):
                if s["id"] in remaining:
                    print(f"\nNext unaudited: index {i+1} ({s['id']})")
                    break
        sys.exit(0)

    # Load only the requested range
    print("Loading samples...")
    samples = load_samples(args.data, start=args.start, end=args.end)
    print(f"Loaded {len(samples)} samples")

    # Set auditor-specific log dir and filter out already-audited
    if args.auditor:
        LOG_DIR_ACTUAL = LOG_DIR / args.auditor
    else:
        LOG_DIR_ACTUAL = LOG_DIR

    # Resume: skip already-audited samples (check root + ALL auditor subdirs)
    LOG_DIR_ACTUAL.mkdir(parents=True, exist_ok=True)
    audited_ids = set()
    # Scan root dir
    if LOG_DIR.exists():
        for f in LOG_DIR.glob("*.json"):
            try:
                with open(f) as fh:
                    r = json.load(fh)
                audited_ids.add(r.get("sample_id"))
            except (json.JSONDecodeError, KeyError):
                pass
    # Scan all auditor subdirs
    if LOG_DIR.exists():
        for subdir in LOG_DIR.iterdir():
            if subdir.is_dir():
                for f in subdir.glob("*.json"):
                    try:
                        with open(f) as fh:
                            r = json.load(fh)
                        audited_ids.add(r.get("sample_id"))
                    except (json.JSONDecodeError, KeyError):
                        pass

    # if audited_ids:
    #     before = len(samples)
    #     samples = [s for s in samples if s["id"] not in audited_ids]
    #     skipped = before - len(samples)
    #     print(f"Resuming: skipped {skipped} already-audited, {len(samples)} remaining")

    # if not samples:
    #     print("All samples in this range have been audited!")
    #     sys.exit(0)

    print("Loading metadata...")
    meta_map = load_metadata()
    print(f"Loaded {len(meta_map)} gold chains")

    app = build_app(samples, meta_map, log_dir=LOG_DIR_ACTUAL)
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
