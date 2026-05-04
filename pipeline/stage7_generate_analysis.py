"""Stage 7 (audit prep): run the LLM auditor (GPT-5.4 Batch API).

Step 1: prepare  — create batch JSONL from stage7_prepare_batch.py output
Step 2: submit   — upload to OpenAI Batch API
Step 3: status   — poll for completion
Step 4: download — download results
Step 5: verify   — post-hoc verification of section mappings

Usage:
    python stage7_generate_analysis.py prepare [--target full|lite]
    python stage7_generate_analysis.py submit  [--target full|lite]
    python stage7_generate_analysis.py status <batch_id>
    python stage7_generate_analysis.py download <batch_id> [--target full|lite]
    python stage7_generate_analysis.py verify [--target full|lite]
"""

import argparse
import json
import re
import os
from pathlib import Path
from collections import Counter

_ENV = Path(__file__).resolve().parent.parent / ".env"
if _ENV.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_ENV)
    except ImportError:
        pass

EVAL_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = Path(__file__).resolve().parent / "data" / "benchmark"


def _batch_file(target: str) -> Path:
    return EVAL_DIR / f"audit_batch_en_{target}.jsonl"


def _output_file(target: str) -> Path:
    return EVAL_DIR / f"audit_analyses_en_{target}.json"


# ── Step 1: Prepare ──────────────────────────────────────────────────────────

def prepare(target: str):
    """Create batch JSONL using stage7_prepare_batch.py."""
    import subprocess
    result = subprocess.run(
        ["python3", str(EVAL_DIR / "stage7_prepare_batch.py"), "--target", target],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)


# ── Step 2: Submit ───────────────────────────────────────────────────────────

def submit(target: str):
    """Submit batch to OpenAI."""
    from openai import OpenAI
    client = OpenAI()

    batch_file = _batch_file(target)
    if not batch_file.exists():
        print(f"Batch file not found: {batch_file}")
        print(f"Run: python stage7_generate_analysis.py prepare --target {target}")
        return

    print(f"Uploading {batch_file}...")
    with open(batch_file, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    print(f"Uploaded: {file_obj.id}")

    print("Submitting batch...")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"Batch ID: {batch.id}")
    print(f"Status: {batch.status}")
    print(f"\nNext: python stage7_generate_analysis.py status {batch.id}")


# ── Step 3: Status ───────────────────────────────────────────────────────────

def status(batch_id: str):
    """Check batch status."""
    from openai import OpenAI
    client = OpenAI()
    batch = client.batches.retrieve(batch_id)
    print(f"Batch: {batch.id}")
    print(f"Status: {batch.status}")
    if batch.request_counts:
        print(f"Total: {batch.request_counts.total}")
        print(f"Completed: {batch.request_counts.completed}")
        print(f"Failed: {batch.request_counts.failed}")
    if batch.status == "completed":
        print(f"\nOutput file: {batch.output_file_id}")
        print(f"Next: python stage7_generate_analysis.py download {batch.id}")


# ── Step 4: Download ─────────────────────────────────────────────────────────

def download(batch_id: str, target: str):
    """Download batch results."""
    from openai import OpenAI
    client = OpenAI()
    batch = client.batches.retrieve(batch_id)

    if batch.status != "completed":
        print(f"Batch not complete: {batch.status}")
        return

    print(f"Downloading from {batch.output_file_id}...")
    content = client.files.content(batch.output_file_id)

    results = []
    for line in content.text.strip().split("\n"):
        item = json.loads(line)
        custom_id = item["custom_id"]
        response = item.get("response", {})
        body = response.get("body", {})
        choices = body.get("choices", [])

        analysis_raw = ""
        if choices:
            analysis_raw = choices[0].get("message", {}).get("content", "")

        # Try to parse JSON
        analysis = None
        try:
            analysis = json.loads(analysis_raw)
        except (json.JSONDecodeError, TypeError):
            # Try extracting JSON from markdown code block
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", analysis_raw, re.DOTALL)
            if m:
                try:
                    analysis = json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass

        results.append({
            "id": custom_id,
            "analysis": analysis,
            "raw": analysis_raw if not analysis else None,
            "status": "ok" if analysis else "parse_error",
        })

    output_file = _output_file(target)
    with open(output_file, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"Downloaded {len(results)} results ({ok} ok, {len(results) - ok} parse errors)")
    print(f"Saved to {output_file}")


# ── Step 5: Verify ───────────────────────────────────────────────────────────

def verify(target: str):
    """Post-hoc verification: check GPT-5.4's section mappings against actual content.

    For each claim → section mapping, check if key terms from the claim
    actually appear in the mapped section's text. Flags unreliable mappings.
    """
    output_file = _output_file(target)
    if not output_file.exists():
        print(f"No audit results found: {output_file}")
        print(f"Run download first.")
        return

    with open(output_file) as f:
        results = json.load(f)

    benchmark_dir = BENCHMARK_DIR / target

    stats = Counter()
    flagged = []

    for r in results:
        if r["status"] != "ok" or not r.get("analysis"):
            stats["skip_no_analysis"] += 1
            continue

        sid = r["id"]
        analysis = r["analysis"]

        # Load benchmark sample
        sample_path = benchmark_dir / f"{sid}.json"
        if not sample_path.exists():
            stats["skip_no_sample"] += 1
            continue

        with open(sample_path) as f:
            s = json.load(f)
        if isinstance(s, list):
            continue

        pp = s.get("paper_pool", {})

        # Build section lookup with normalized headers for fuzzy matching
        def _normalize_header(h):
            """Strip numbering, prefixes, punctuation for fuzzy matching."""
            h = h.lower().strip()
            # Strip Apdx_ prefix variants
            h = re.sub(r"^apdx_[a-z][\d.]*:\s*", "", h)
            h = re.sub(r"^apdx:\s*", "", h)
            # Strip Roman numerals (IV-B, III, etc.)
            h = re.sub(r"^[ivxlc]+-?[a-z]?\s+", "", h)
            # Strip Arabic numbering (5.4.2., 3., etc.)
            h = re.sub(r"^[\d.]+\s*", "", h)
            # Strip punctuation and extra whitespace
            h = re.sub(r"[:\-–—]", " ", h)
            h = re.sub(r"\s+", " ", h).strip()
            return h

        section_lookup = {}
        section_lookup_normalized = {}  # normalized_header -> text
        for arxiv_id, paper_data in pp.items():
            for sec in paper_data.get("sections", []):
                header = sec.get("header", "")
                text = sec.get("text", "").lower()
                section_lookup[(arxiv_id, header.lower())] = text
                section_lookup[("*", header.lower())] = text
                # Normalized index
                nh = _normalize_header(header)
                if nh:
                    section_lookup_normalized[nh] = text

        # Check answer_validity claims
        for claim_entry in analysis.get("answer_validity", {}).get("claims", []):
            stats["total_claims"] += 1
            claim_text = claim_entry.get("claim", "")
            source_section = claim_entry.get("source_section", "")
            verified = claim_entry.get("verified", False)
            verbatim = claim_entry.get("verbatim_match", False)

            if not claim_text or not source_section:
                stats["skip_empty"] += 1
                continue

            # Extract key terms: numbers, technical terms (3+ char words)
            numbers = re.findall(r"\d+\.?\d*%?", claim_text)
            words = [w.lower() for w in re.findall(r"[A-Za-z]{4,}", claim_text)]
            key_terms = numbers + words[:5]  # prioritize numbers

            if not key_terms:
                stats["skip_no_terms"] += 1
                continue

            # Find matching section (3 strategies: exact, substring, normalized)
            matched_text = None
            src_lower = source_section.lower()
            src_norm = _normalize_header(source_section)

            # Strategy 1: exact/substring match on raw headers
            for (aid, header), text in section_lookup.items():
                if src_lower in header or header in src_lower:
                    matched_text = text
                    break

            # Strategy 2: normalized header match
            if matched_text is None and src_norm:
                for nh, text in section_lookup_normalized.items():
                    if src_norm in nh or nh in src_norm:
                        matched_text = text
                        break

            # Strategy 3: keyword overlap (>60% of words match)
            if matched_text is None and src_norm:
                src_words = set(src_norm.split())
                if len(src_words) >= 2:
                    best_overlap = 0
                    for nh, text in section_lookup_normalized.items():
                        nh_words = set(nh.split())
                        if not nh_words:
                            continue
                        overlap = len(src_words & nh_words) / max(len(src_words), 1)
                        if overlap > best_overlap and overlap >= 0.6:
                            best_overlap = overlap
                            matched_text = text

            # Strategy 4: subsection search — GPT references a subsection title
            # that's rolled into a parent section's body text
            if matched_text is None and src_norm and len(src_norm) >= 5:
                for (aid, header), text in section_lookup.items():
                    if src_norm in text or src_lower in text:
                        matched_text = text
                        break

            if matched_text is None:
                stats["section_not_found"] += 1
                flagged.append({
                    "sample_id": sid,
                    "claim": claim_text[:100],
                    "mapped_section": source_section,
                    "issue": "section_not_found",
                    "gpt_verified": verified,
                })
                continue

            # Check key terms
            found = sum(1 for t in key_terms if t.lower() in matched_text)
            ratio = found / len(key_terms)

            if ratio >= 0.5:
                stats["confirmed"] += 1
            elif ratio > 0:
                stats["partial"] += 1
                if verified:
                    flagged.append({
                        "sample_id": sid,
                        "claim": claim_text[:100],
                        "mapped_section": source_section,
                        "issue": f"partial_match ({found}/{len(key_terms)} terms)",
                        "gpt_verified": verified,
                        "missing_terms": [t for t in key_terms if t.lower() not in matched_text],
                    })
            else:
                stats["no_match"] += 1
                flagged.append({
                    "sample_id": sid,
                    "claim": claim_text[:100],
                    "mapped_section": source_section,
                    "issue": f"no_terms_found (0/{len(key_terms)})",
                    "gpt_verified": verified,
                    "key_terms": key_terms,
                })

        # Check section_recall_labels (same 3-strategy matching)
        for label in analysis.get("section_recall_labels", []):
            stats["total_labels"] += 1
            section = label.get("section", "")
            sec_lower = section.lower()
            sec_norm = _normalize_header(section)

            matched = False
            # Strategy 1: exact/substring
            if any(sec_lower in header or header in sec_lower for (_, header) in section_lookup.keys()):
                matched = True
            # Strategy 2: normalized
            if not matched and sec_norm:
                if any(sec_norm in nh or nh in sec_norm for nh in section_lookup_normalized.keys()):
                    matched = True
            # Strategy 3: keyword overlap
            if not matched and sec_norm:
                sec_words = set(sec_norm.split())
                if len(sec_words) >= 2:
                    for nh in section_lookup_normalized.keys():
                        nh_words = set(nh.split())
                        if nh_words and len(sec_words & nh_words) / max(len(sec_words), 1) >= 0.6:
                            matched = True
                            break
            # Strategy 4: subsection search in body text
            if not matched and sec_norm and len(sec_norm) >= 5:
                for (_, header), text in section_lookup.items():
                    if sec_norm in text or sec_lower in text:
                        matched = True
                        break

            if matched:
                stats["label_found"] += 1
            else:
                stats["label_not_found"] += 1

    # Report
    print("=== Post-hoc Verification ===\n")
    print(f"Total claims checked: {stats['total_claims']}")
    print(f"  Confirmed (≥50% terms match):  {stats['confirmed']}")
    print(f"  Partial (some terms match):     {stats['partial']}")
    print(f"  No match (0 terms found):       {stats['no_match']}")
    print(f"  Section not found:              {stats['section_not_found']}")
    print(f"  Skipped (empty/no terms):       {stats['skip_empty'] + stats['skip_no_terms']}")
    print()
    print(f"Section recall labels: {stats['total_labels']}")
    print(f"  Found in paper_pool:  {stats['label_found']}")
    print(f"  Not found:            {stats['label_not_found']}")

    if flagged:
        print(f"\n=== Flagged Mappings ({len(flagged)}) ===\n")
        for f_entry in flagged[:30]:
            print(f"  {f_entry['sample_id']}: [{f_entry['issue']}]")
            print(f"    claim: {f_entry['claim']}")
            print(f"    mapped: {f_entry['mapped_section']}")
            if f_entry.get("missing_terms"):
                print(f"    missing: {f_entry['missing_terms'][:5]}")
            print()
        if len(flagged) > 30:
            print(f"  ... and {len(flagged) - 30} more")

    # Save flagged
    flagged_path = EVAL_DIR / f"audit_verify_flagged_{target}.json"
    with open(flagged_path, "w") as f_out:
        json.dump(flagged, f_out, indent=2, ensure_ascii=False)
    print(f"\nFlagged mappings saved to {flagged_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="English audit pipeline")
    parser.add_argument("command", choices=["prepare", "submit", "status", "download", "verify"])
    parser.add_argument("batch_id", nargs="?", default=None)
    parser.add_argument("--target", choices=["lite", "full"], default="full")
    args = parser.parse_args()

    if args.command == "prepare":
        prepare(args.target)
    elif args.command == "submit":
        submit(args.target)
    elif args.command == "status":
        if not args.batch_id:
            print("Usage: python stage7_generate_analysis.py status <batch_id>")
        else:
            status(args.batch_id)
    elif args.command == "download":
        if not args.batch_id:
            print("Usage: python stage7_generate_analysis.py download <batch_id>")
        else:
            download(args.batch_id, args.target)
    elif args.command == "verify":
        verify(args.target)
