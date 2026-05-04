"""Stage 6: Data sanity check (structural-integrity pass).

Checks structural integrity AND content navigability for every sample.

A (Structural):
  - Gold path nodes have titles (no ghosts)
  - Gold path nodes have content in paper_pool
  - For d2 BFS: bridge paper reachable from seed refs
  - Seed has references in graph

B (Content):
  - Terminal paper sections contain keywords from the correct answer
  - For DFS: target papers are readable and contain relevant facts
  - Question author/year references match identifiable nodes

Usage:
    python validate_benchmark.py --target lite
    python validate_benchmark.py --target full
    python validate_benchmark.py --target both --remove   # remove failing samples
"""

import argparse
import json
import glob
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass, field

BENCHMARK_DIR = Path(__file__).resolve().parent / "data/benchmark"
MCQ_VALIDATED = Path(__file__).resolve().parent / "data/questions/mcq_combined_validated.json"


@dataclass
class ValidationResult:
    sample_id: str
    question_type: str
    depth: int
    passed: bool
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)  # non-blocking


def load_metadata() -> dict:
    """Load gold chain / group info from validated MCQs."""
    with open(MCQ_VALIDATED) as f:
        validated = json.load(f)

    meta = {}
    for v in validated:
        entry = {
            "question_type": v["question_type"],
            "depth": v["depth"],
            "correct_index": v["correct_index"],
        }
        if v["question_type"] == "bfs":
            chain = v.get("chain", {})
            if chain and "terminal" in chain:
                entry["terminal_pid"] = chain["terminal"]["paperId"]
                entry["terminal_title"] = chain["terminal"].get("title", "")
                entry["seed_pid"] = chain["seed"]["paperId"]
                entry["path_titles"] = chain.get("path_titles", [])
        elif v["question_type"] == "dfs":
            group = v.get("group", {})
            entry["seed_pid"] = group.get("seed", {}).get("paperId", "")
            targets = group.get("targets", [])
            entry["target_pids"] = [t["paperId"] for t in targets if "paperId" in t]
            entry["target_arxivs"] = [t.get("arxivId", "") for t in targets]
            ans_src = v.get("answer_sources", {})
            entry["answer_sources"] = ans_src

        meta[v["id"]] = entry
    return meta


def extract_answer_keywords(answer_text: str, n: int = 8) -> list[str]:
    """Extract salient keywords from the correct answer for content matching.

    Focuses on numbers, proper nouns, and technical terms.
    """
    keywords = []

    # Numbers (scores, percentages, counts) — strongest signal
    numbers = re.findall(r'\d+\.?\d*%?', answer_text)
    keywords.extend(numbers[:5])

    # Multi-word technical terms (capitalized sequences)
    tech_terms = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', answer_text)
    keywords.extend(tech_terms[:3])

    # Acronyms
    acronyms = re.findall(r'\b[A-Z]{2,}\b', answer_text)
    keywords.extend(acronyms[:3])

    return keywords[:n]


def check_content_contains_keywords(paper_pool: dict, arxiv_id: str, keywords: list[str]) -> tuple[bool, int]:
    """Check if a paper's sections contain any of the answer keywords.

    Returns (any_found, count_found).
    """
    if not arxiv_id or arxiv_id not in paper_pool:
        return False, 0

    paper = paper_pool[arxiv_id]
    sections = paper.get("sections", [])
    all_text = " ".join(sec.get("text", "") for sec in sections if isinstance(sec, dict))

    found = sum(1 for kw in keywords if kw in all_text)
    return found > 0, found


def validate_bfs(sample: dict, meta: dict) -> ValidationResult:
    """Validate a BFS sample."""
    sid = sample["id"]
    nodes = sample["graph"]["nodes"]
    edges = sample["graph"]["edges"]
    pp = sample["paper_pool"]
    seed = sample["seed_paper_id"]
    depth = sample["depth"]
    correct_answer = sample["options"][sample["correct_index"]]

    result = ValidationResult(sid, "bfs", depth, passed=True)

    terminal = meta.get("terminal_pid")
    if not terminal:
        result.issues.append("NO_GOLD_CHAIN: missing terminal in metadata")
        result.passed = False
        return result

    # A1: Terminal exists in graph
    t_info = nodes.get(terminal, {})
    if not t_info:
        result.issues.append(f"TERMINAL_MISSING: {terminal[:16]}... not in graph nodes")
        result.passed = False
        return result

    # A2: Terminal has title
    if not t_info.get("title"):
        result.issues.append(f"TERMINAL_GHOST: {terminal[:16]}... has no title/metadata")
        result.passed = False

    # A3: Terminal has content
    t_arxiv = t_info.get("arxivId", "")
    if not t_arxiv or t_arxiv not in pp:
        result.issues.append(f"TERMINAL_NO_CONTENT: {terminal[:16]}... not in paper_pool")
        result.passed = False

    # A4: Seed has references
    seed_refs = edges.get(seed, [])
    if not seed_refs:
        result.issues.append("SEED_NO_REFS: seed has 0 references in graph")
        result.passed = False
        return result

    # A5: For d2 — bridge paper check
    if depth == 2:
        bridge_found = False
        bridge_ghost = False
        for ref in seed_refs:
            if ref in edges and terminal in edges[ref]:
                bridge_found = True
                b_info = nodes.get(ref, {})
                if not b_info.get("title"):
                    bridge_ghost = True
                    result.issues.append(f"BRIDGE_GHOST: {ref[:16]}... has no title")
                    result.passed = False
                break

        if not bridge_found:
            result.issues.append("NO_BRIDGE_PATH: no seed ref leads to terminal")
            result.passed = False

    # B1: Answer keywords in terminal content
    if t_arxiv and t_arxiv in pp:
        keywords = extract_answer_keywords(correct_answer)
        if keywords:
            found, count = check_content_contains_keywords(pp, t_arxiv, keywords)
            if not found:
                result.warnings.append(f"ANSWER_NOT_IN_CONTENT: 0/{len(keywords)} keywords found in terminal sections")

    # B2 omitted — author/year reference checks are better handled by human audit
    # (too many edge cases for heuristic matching)

    return result


def validate_dfs(sample: dict, meta: dict) -> ValidationResult:
    """Validate a DFS sample."""
    sid = sample["id"]
    nodes = sample["graph"]["nodes"]
    edges = sample["graph"]["edges"]
    pp = sample["paper_pool"]
    seed = sample["seed_paper_id"]
    depth = sample["depth"]
    correct_answer = sample["options"][sample["correct_index"]]

    result = ValidationResult(sid, "dfs", depth, passed=True)

    target_pids = meta.get("target_pids", [])

    # A1: Seed has references
    seed_refs = edges.get(seed, [])
    if not seed_refs:
        result.issues.append("SEED_NO_REFS")
        result.passed = False
        return result

    # A2: Target papers exist and are identifiable
    for tpid in target_pids:
        t_info = nodes.get(tpid, {})
        if not t_info.get("title"):
            result.issues.append(f"TARGET_GHOST: {tpid[:16]}...")
            result.passed = False

    # A3: Target papers have readable content
    targets_readable = 0
    for tpid in target_pids:
        t_info = nodes.get(tpid, {})
        t_arxiv = t_info.get("arxivId", "")
        if t_arxiv and t_arxiv in pp:
            targets_readable += 1
        else:
            result.issues.append(f"TARGET_NO_CONTENT: {tpid[:16]}... ({t_info.get('title','?')[:40]})")
            result.passed = False

    # A4: Targets reachable from seed
    for tpid in target_pids:
        if tpid not in seed_refs:
            # Maybe reachable via d2?
            if depth == 2:
                found_d2 = False
                for ref in seed_refs:
                    if ref in edges and tpid in edges[ref]:
                        found_d2 = True
                        break
                if not found_d2:
                    result.warnings.append(f"TARGET_UNREACHABLE: {tpid[:16]}... not in seed refs or d2")

    # B1: Answer keywords in target content
    keywords = extract_answer_keywords(correct_answer)
    if keywords and target_pids:
        any_content_match = False
        for tpid in target_pids:
            t_arxiv = nodes.get(tpid, {}).get("arxivId", "")
            if t_arxiv:
                found, _ = check_content_contains_keywords(pp, t_arxiv, keywords)
                if found:
                    any_content_match = True
                    break
        if not any_content_match:
            result.warnings.append(f"ANSWER_NOT_IN_TARGETS: 0/{len(keywords)} keywords in any target paper")

    return result


def validate_benchmark(target: str, remove: bool = False, verbose: bool = True) -> dict:
    """Run validation on benchmark files.

    Returns summary dict with results.
    """
    meta_map = load_metadata()
    print(f"Loaded metadata for {len(meta_map)} samples")

    dirs = []
    if target in ("lite", "both"):
        dirs.append(("lite", BENCHMARK_DIR / "lite"))
    if target in ("full", "both"):
        dirs.append(("full", BENCHMARK_DIR / "full"))

    all_results = {}

    for label, dirpath in dirs:
        files = sorted([f for f in glob.glob(str(dirpath / "*.json"))
                        if not Path(f).name.startswith("agenthop")])

        results = []
        for fpath in files:
            with open(fpath) as f:
                sample = json.load(f)

            sid = sample["id"]
            meta = meta_map.get(sid, {})

            if sample["question_type"] == "bfs" and meta:
                r = validate_bfs(sample, meta)
            elif sample["question_type"] == "dfs" and meta:
                r = validate_dfs(sample, meta)
            else:
                # No metadata — can only do basic checks
                r = ValidationResult(sid, sample["question_type"], sample["depth"], passed=True)
                if not sample["graph"]["edges"].get(sample["seed_paper_id"]):
                    r.issues.append("SEED_NO_REFS")
                    r.passed = False

            results.append(r)

        # Summarize
        passed = [r for r in results if r.passed]
        failed = [r for r in results if not r.passed]
        warned = [r for r in passed if r.warnings]

        issue_types = Counter()
        warn_types = Counter()
        for r in results:
            for i in r.issues:
                issue_types[i.split(":")[0]] += 1
            for w in r.warnings:
                warn_types[w.split(":")[0]] += 1

        print(f"\n{'=' * 60}")
        print(f"{label.upper()}: {len(files)} files")
        print(f"{'=' * 60}")
        print(f"  PASSED: {len(passed)}/{len(results)}")
        print(f"  FAILED: {len(failed)}/{len(results)} ({len(failed)/len(results)*100:.1f}%)")
        print(f"  WARNED: {len(warned)} (passed but flagged)")

        if issue_types:
            print(f"\n  Blocking issues:")
            for itype, cnt in issue_types.most_common():
                print(f"    {itype}: {cnt}")

        if warn_types:
            print(f"\n  Warnings (non-blocking):")
            for wtype, cnt in warn_types.most_common():
                print(f"    {wtype}: {cnt}")

        if verbose and failed:
            print(f"\n  Failed samples:")
            for r in sorted(failed, key=lambda x: x.sample_id):
                print(f"    {r.sample_id} ({r.question_type}/d{r.depth}): {', '.join(r.issues)}")

        # Remove failed samples
        if remove and failed:
            failed_ids = {r.sample_id for r in failed}
            removed = 0
            for fpath in files:
                with open(fpath) as f:
                    sample = json.load(f)
                if sample["id"] in failed_ids:
                    Path(fpath).unlink()
                    removed += 1
            print(f"\n  REMOVED {removed} files from {dirpath}")

            # Update combined file if it exists
            combined = dirpath / f"agenthop_{label}.json"
            if combined.exists():
                with open(combined) as f:
                    data = json.load(f)
                data = [s for s in data if s.get("id") not in failed_ids]
                with open(combined, "w") as f:
                    json.dump(data, f, ensure_ascii=False)
                print(f"  Updated {combined.name}: {len(data)} samples remaining")

        all_results[label] = {
            "total": len(results),
            "passed": len(passed),
            "failed": len(failed),
            "warned": len(warned),
            "failed_ids": [r.sample_id for r in failed],
            "warned_ids": [r.sample_id for r in warned],
            "results": results,
        }

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgentHop benchmark validation")
    parser.add_argument("--target", choices=["lite", "full", "both"], default="both")
    parser.add_argument("--remove", action="store_true", help="Remove failing samples")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args()

    validate_benchmark(args.target, remove=args.remove, verbose=not args.quiet)
