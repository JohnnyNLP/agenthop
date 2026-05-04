"""Stage 7 (audit prep): build the audit-batch JSONL for the LLM auditor.

Focuses on query quality + correct answer verification + section-level recall labels.
No distractor options shown — audit targets only the query and correct answer.
Includes appendix content (post-rebuild).

Dual purpose:
  1. Human audit proof for NeurIPS reviewers
  2. Section-level recall labels (per-claim → paper/section ground truth)

Usage:
    python stage7_prepare_batch.py
    python stage7_prepare_batch.py --target lite
    python stage7_prepare_batch.py --dry-run
"""

import argparse
import json
import glob
from pathlib import Path

MCQ_VALIDATED = Path(__file__).resolve().parent / "data/questions/mcq_combined_validated.json"
BENCHMARK_DIR = Path(__file__).resolve().parent / "data/benchmark"
OUTPUT_DIR = Path(__file__).resolve().parent

SYSTEM_PROMPT = """You are an audit assistant for AgentHop, a multi-hop scientific reasoning benchmark.

You will receive:
- A benchmark question designed to test citation-chain navigation
- The correct answer to that question
- Full content of the seed paper and target paper(s), including appendix sections

Your job is to verify the QUERY and CORRECT ANSWER only. You are NOT selecting an answer from multiple options.

The sample type will be indicated as "single-target" or "multi-target". Apply the type-specific checks described below.

Return a JSON object with the following structure:

{
  "query_quality": {
    "verdict": "pass" | "warn" | "fail",
    "natural": true | false,
    "unambiguous": true | false,
    "requires_target": true | false,
    "note": "Brief explanation of any issues with the question wording, clarity, or scope. Empty string if none."
  },
  "answer_validity": {
    "verdict": "pass" | "warn" | "fail",
    "claims": [
      {
        "claim": "A specific factual assertion from the correct answer",
        "source_paper": "Which paper contains this claim (seed title or target title)",
        "source_section": "Exact section header where this claim is found",
        "verbatim_match": true | false,
        "verified": true | false,
        "note": "Discrepancy detail if not verified, or empty string"
      }
    ]
  },
  "chain_coherence": {
    "verdict": "pass" | "warn" | "fail",
    "seed_to_bridge": "Does the seed paper's citation context naturally point toward the bridge paper? Explain briefly.",
    "bridge_to_terminal": "Does the bridge paper lead to the terminal? Is the bridge necessary or could you skip it?",
    "bridge_necessary": true | false,
    "note": "Any issues with the citation chain logic. Empty string if none."
  },
  "synthesis_check": {
    "verdict": "pass" | "warn" | "fail",
    "both_targets_needed": true | false,
    "target_1_contribution": "What fact/evidence does target 1 contribute to the answer?",
    "target_2_contribution": "What fact/evidence does target 2 contribute to the answer?",
    "genuine_synthesis": true | false,
    "note": "Is this genuine cross-paper synthesis or just two independent facts? Empty string if fine."
  },
  "section_recall_labels": [
    {
      "paper": "Paper title (as provided in the content headers)",
      "arxiv_id": "arxiv ID if identifiable from the content header, else empty string",
      "section": "Section header containing answer-relevant evidence",
      "relevance": "direct" | "supporting",
      "what_it_contains": "Brief description of what evidence this section provides for the answer"
    }
  ],
  "structural_integrity": {
    "verdict": "pass" | "warn" | "fail",
    "missing_sections": ["Section names that appear to be missing from the provided content, if any"],
    "note": "Brief explanation of structural issues, or empty string"
  }
}

Type-specific instructions:

FOR SINGLE-TARGET SAMPLES:
- Focus on "chain_coherence". The key question is whether the seed → bridge → terminal path is natural and necessary.
- Check: does the seed paper's text contain a citation or reference that would lead a reader toward the bridge paper?
- Check: does the bridge paper connect to the terminal, or could you reach the terminal directly from the seed?
- Check: is the answer found ONLY in the terminal paper, not in the seed or bridge?
- Set "synthesis_check" fields to null — not applicable for single-target.

FOR MULTI-TARGET SAMPLES:
- Focus on "synthesis_check". The key question is whether BOTH target papers are genuinely needed.
- Check: does the answer combine information from both targets, or could one target alone suffice?
- Check: is the synthesis genuine (comparing, contrasting, combining) or superficial (two unrelated facts)?
- Set "chain_coherence" fields to null — not applicable for multi-target.

General guidelines:
- For "claims": decompose the correct answer into individual factual assertions. Each number, comparison, method name, or result is a separate claim.
- For "section_recall_labels": list ALL sections across ALL provided papers that contain evidence relevant to the answer. "direct" = the claim is stated here; "supporting" = provides context needed to interpret the claim.
- For "requires_target": true if the question cannot be answered from the seed paper alone (this should almost always be true).
- For "verbatim_match": true if the claim text appears nearly word-for-word in the source section.
- Be precise with section names — use the exact header as it appears in the provided content.

IMPORTANT: Return ONLY the JSON object. No markdown, no explanation outside the JSON."""


def format_paper_content(paper_pool: dict, arxiv_id: str, label: str) -> str:
    """Format full paper content for the prompt."""
    if not arxiv_id or arxiv_id not in paper_pool:
        return f"[{label}: content not available]"

    sections = paper_pool[arxiv_id].get("sections", [])
    if not sections:
        return f"[{label}: no sections available]"

    APPENDIX_CAP = 15_000  # truncate appendix sections to 15K chars total

    parts = [f"=== {label} (arxiv: {arxiv_id}) ==="]
    apdx_chars = 0
    for sec in sections:
        header = sec["header"]
        text = sec["text"]
        if header.startswith("Apdx"):
            remaining = APPENDIX_CAP - apdx_chars
            if remaining <= 0:
                continue
            if len(text) > remaining:
                text = text[:remaining] + f"\n[...truncated, {len(sec['text']) - remaining} chars omitted]"
            apdx_chars += len(text)
        parts.append(f"\n## {header}\n{text}")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Prepare English audit batch for GPT-5.4")
    parser.add_argument("--target", choices=["lite", "full"], default="full")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    benchmark_dir = BENCHMARK_DIR / args.target
    output_file = OUTPUT_DIR / f"audit_batch_en_{args.target}.jsonl"

    # Load metadata
    print("Loading metadata...")
    with open(MCQ_VALIDATED) as f:
        validated = json.load(f)
    meta_map = {v["id"]: v for v in validated}

    # Process benchmark files
    files = sorted([f for f in glob.glob(str(benchmark_dir / "*.json"))
                     if not Path(f).name.startswith("agenthop")])
    if args.limit:
        files = files[:args.limit]

    print(f"Processing {len(files)} samples from {args.target}...")

    skipped = 0
    written = 0

    out_handle = None if args.dry_run else open(output_file, "w")

    for i, fpath in enumerate(files):
        with open(fpath) as f:
            s = json.load(f)
        if isinstance(s, list):
            continue

        sid = s["id"]
        old_id = ("q_" + sid[3:]) if sid.startswith("st_") else ("dfs_" + sid[3:]) if sid.startswith("mt_") else sid
        meta = meta_map.get(sid) or meta_map.get(old_id) or {}

        nodes = s["graph"]["nodes"]
        pp = s["paper_pool"]
        ci = s["correct_index"]
        correct_answer = s["options"][ci]

        # Seed paper content
        seed_pid = s["seed_paper_id"]
        seed_arxiv = nodes.get(seed_pid, {}).get("arxivId", "")
        seed_title = s.get("seed_title", nodes.get(seed_pid, {}).get("title", "?"))
        seed_content = format_paper_content(pp, seed_arxiv, f"Seed: {seed_title}")

        # Chain papers (st_: bridge + terminal) and target papers (mt_)
        bridge_contents = []
        target_contents = []
        chain = meta.get("chain", {})
        if chain:
            # Bridge papers — resolve from path_titles via graph node title matching
            path_titles = chain.get("path_titles", [])
            terminal_title = chain.get("terminal", {}).get("title", "")
            for j, btitle in enumerate(path_titles):
                # Skip the terminal (last in path_titles for depth-2)
                if btitle == terminal_title:
                    continue
                # Find in graph nodes by title
                b_arxiv = ""
                for pid, node in nodes.items():
                    if node.get("title", "") == btitle:
                        b_arxiv = node.get("arxivId", "")
                        break
                if not b_arxiv:
                    # Fuzzy match (first 30 chars)
                    for pid, node in nodes.items():
                        if btitle.lower()[:30] in node.get("title", "").lower():
                            b_arxiv = node.get("arxivId", "")
                            break
                if b_arxiv:
                    bridge_contents.append(
                        format_paper_content(pp, b_arxiv, f"Bridge {j+1}: {btitle}")
                    )

            # Terminal paper
            if "terminal" in chain:
                tpid = chain["terminal"]["paperId"]
                tinfo = nodes.get(tpid, {})
                t_arxiv = tinfo.get("arxivId", "")
                t_title = tinfo.get("title", chain["terminal"].get("title", "?"))
                target_contents.append(
                    format_paper_content(pp, t_arxiv, f"Terminal: {t_title}")
                )

        group = meta.get("group", {})
        for j, t in enumerate(group.get("targets", [])):
            tpid = t.get("paperId", "")
            tinfo = nodes.get(tpid, {})
            t_arxiv = tinfo.get("arxivId", "") or t.get("arxivId", "")
            t_title = tinfo.get("title", t.get("title", "?"))
            target_contents.append(
                format_paper_content(pp, t_arxiv, f"Target {j+1}: {t_title}")
            )

        # Build user prompt — correct answer only, no other options
        user_prompt = f"""Sample: {sid}
Type: {s.get('question_type')} | Depth: {s.get('depth')} | Reasoning: {s.get('reasoning_type', '?')} | Skill: {s.get('cognitive_skill', '?')}

## Seed Paper Content
{seed_content}

## Bridge Paper Content
{chr(10).join(bridge_contents) if bridge_contents else '[No bridge papers — depth 1 or multi-target sample]'}

## Target Paper Content
{chr(10).join(target_contents) if target_contents else '[No target content available]'}

## Question
{s['question']}

## Correct Answer
{correct_answer}"""

        total_chars = len(SYSTEM_PROMPT) + len(user_prompt)
        if total_chars > 800_000:
            skipped += 1
            if args.dry_run:
                print(f"  SKIP {sid} ({total_chars // 1000}K chars, >800K)")
            continue

        if args.dry_run:
            print(f"  {sid}: {total_chars // 1000}K chars")
            written += 1
            continue

        request = {
            "custom_id": sid,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-5.4",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "max_completion_tokens": 8192,
                "temperature": 0.0,
            },
        }
        out_handle.write(json.dumps(request, ensure_ascii=False) + "\n")
        written += 1

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(files)}]...")

    if out_handle:
        out_handle.close()

    print(f"\nDone: {written} written, {skipped} skipped (>800K chars)")
    if not args.dry_run:
        size_mb = output_file.stat().st_size / 1024 / 1024
        print(f"Output: {output_file} ({size_mb:.1f}MB)")


if __name__ == "__main__":
    main()
