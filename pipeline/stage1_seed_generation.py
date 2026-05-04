"""Stage 1: Seed generation.

Selects the seed papers that anchor every citation chain in the benchmark.
Queries Semantic Scholar's bulk-search API for papers from the nine target
venues and stratifies them into three tiers by recency and citation count.

Usage:
    python select_seeds.py                     # full run, save seeds.json
    python select_seeds.py --dry-run           # show counts without saving
    python select_seeds.py --tiers recent      # only run one tier
    python select_seeds.py --pilot 5           # select 5 per tier (quick test)
    python select_seeds.py --exclude data/seeds.json -o data/seeds_r2.json  # incremental
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from config import (
    VENUES,
    VENUE_SHORT,
    SEED_TIERS,
    SEED_MIN_REFERENCES,
    SEED_MAX_REFERENCES,
    SEED_MIN_METADATA_COVERAGE,
    DATA_DIR,
)
from s2_client import S2Client

# Broad query covering virtually all CS papers.
# The venue + year + citation filters do the real work;
# this just satisfies the API's required query parameter.
BROAD_QUERY = (
    "model | learning | method | network | data | system | algorithm "
    "| approach | training | analysis | task | representation | framework"
)

# Title keywords that signal surveys/position papers
SURVEY_KEYWORDS = ["survey", "review", "tutorial", "overview", "position paper"]


def is_survey(paper: dict) -> bool:
    title = (paper.get("title") or "").lower()
    if any(kw in title for kw in SURVEY_KEYWORDS):
        return True
    ref_count = paper.get("referenceCount") or 0
    pub_types = paper.get("publicationTypes") or []
    if "Review" in pub_types:
        return True
    return False


def has_arxiv_id(paper: dict) -> bool:
    ext = paper.get("externalIds") or {}
    return bool(ext.get("ArXiv"))


def get_arxiv_id(paper: dict) -> str:
    ext = paper.get("externalIds") or {}
    return ext.get("ArXiv", "")


def check_metadata_coverage(client: S2Client, paper_id: str) -> float:
    """Check what fraction of a paper's references have S2 citation metadata.

    Returns a float in [0, 1] representing the fraction of references that
    have at least one non-empty citation context.
    """
    refs = client.get_references(paper_id)
    if not refs:
        return 0.0

    with_context = sum(
        1 for ref in refs
        if any(c.strip() for c in (ref.get("contexts") or []))
    )
    return with_context / len(refs)


def validate_seeds(
    client: S2Client,
    seeds: list[dict],
    min_coverage: float = SEED_MIN_METADATA_COVERAGE,
) -> tuple[list[dict], list[dict]]:
    """Validate seeds by checking S2 metadata coverage on their references.

    Returns (passed, failed) lists. Each seed gets a '_metadata_coverage' field.
    """
    passed, failed = [], []
    for i, seed in enumerate(seeds):
        pid = seed["paperId"]
        title = (seed.get("title") or "?")[:50]
        print(f"  [{i+1}/{len(seeds)}] {title}...", end=" ", flush=True)

        coverage = check_metadata_coverage(client, pid)
        seed["_metadata_coverage"] = coverage

        if coverage >= min_coverage:
            passed.append(seed)
            print(f"✓ {coverage:.0%}")
        else:
            failed.append(seed)
            print(f"✗ {coverage:.0%} (below {min_coverage:.0%})")

    return passed, failed


def select_seeds_for_tier(
    client: S2Client,
    tier_name: str,
    tier_config: dict,
    venues: list[str],
    target_override: int = None,
    exclude_ids: set[str] = None,
) -> list[dict]:
    """Select seed papers for one tier."""
    year_lo, year_hi = tier_config["year_range"]
    min_citations = tier_config["min_citations"]
    target = target_override or tier_config["target_count"]
    exclude_ids = exclude_ids or set()

    print(f"\n{'='*70}")
    print(f"Tier: {tier_name} | Year: {year_lo}-{year_hi} | "
          f"Min citations: {min_citations} | Target: {target}")
    print(f"{'='*70}")

    all_papers = []
    seen_ids = set(exclude_ids)
    venue_raw_counts = {}

    for venue in venues:
        short = VENUE_SHORT.get(venue, venue)
        print(f"\n  Searching {short} ...", end=" ", flush=True)

        papers = client.search_papers_bulk(
            query=BROAD_QUERY,
            venue=venue,
            year=f"{year_lo}-{year_hi}",
            min_citation_count=min_citations if min_citations > 0 else None,
            # Note: publicationTypes="Conference" excluded — S2 misclassifies
            # NeurIPS/ICLR as JournalArticle. Venue filter is sufficient.
        )
        venue_raw_counts[venue] = len(papers)

        # Post-filter
        valid = []
        for p in papers:
            pid = p.get("paperId")
            if not pid or pid in seen_ids:
                continue
            if not has_arxiv_id(p):
                continue
            ref_count = p.get("referenceCount") or 0
            if not (SEED_MIN_REFERENCES <= ref_count <= SEED_MAX_REFERENCES):
                continue
            if is_survey(p):
                continue
            seen_ids.add(pid)
            valid.append(p)

        print(f"raw={len(papers)}, valid={len(valid)}")
        all_papers.extend(valid)

    print(f"\n  Total candidates for {tier_name}: {len(all_papers)}")

    # Sample with venue balance
    if len(all_papers) <= target:
        selected = all_papers
    else:
        by_venue = defaultdict(list)
        for p in all_papers:
            v = p.get("venue") or "unknown"
            by_venue[v].append(p)

        # Sort within each venue by citation count (descending) for reproducibility
        for v in by_venue:
            by_venue[v].sort(key=lambda p: -(p.get("citationCount") or 0))

        selected = []
        venue_list = sorted(by_venue.keys())
        per_venue = max(1, target // len(venue_list))

        for v in venue_list:
            selected.extend(by_venue[v][:per_venue])

        # Fill remaining slots from papers not yet selected
        selected_ids = {p["paperId"] for p in selected}
        remaining = [p for p in all_papers if p["paperId"] not in selected_ids]
        remaining.sort(key=lambda p: -(p.get("citationCount") or 0))
        selected.extend(remaining[: target - len(selected)])
        selected = selected[:target]

    print(f"  Selected: {len(selected)}")
    return selected


def main():
    parser = argparse.ArgumentParser(description="Select seed papers for AgentHop")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show counts and examples without saving"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--tiers",
        nargs="+",
        choices=list(SEED_TIERS.keys()),
        default=None,
        help="Only run specific tiers",
    )
    parser.add_argument(
        "--pilot",
        type=int,
        default=None,
        help="Override target count per tier (e.g., --pilot 5 for quick test)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: data/seeds.json)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip S2 metadata coverage validation (faster, but may include bad seeds)",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=SEED_MIN_METADATA_COVERAGE,
        help=f"Min fraction of refs with citation contexts (default: {SEED_MIN_METADATA_COVERAGE})",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="+",
        default=None,
        help="Path(s) to existing seeds JSON files whose paperIds should be excluded",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    client = S2Client()

    # Load paper IDs to exclude from previous runs
    exclude_ids = set()
    if args.exclude:
        for exc_path in args.exclude:
            p = Path(exc_path)
            if not p.exists():
                print(f"Warning: exclude file {exc_path} not found, skipping")
                continue
            with open(p) as f:
                existing = json.load(f)
            ids = {s["paperId"] for s in existing if "paperId" in s}
            exclude_ids |= ids
            print(f"Excluding {len(ids)} paper IDs from {exc_path}")
        print(f"Total exclusions: {len(exclude_ids)}")

    tiers_to_run = args.tiers or list(SEED_TIERS.keys())

    # When validation is enabled, oversample by 40% to compensate for rejections
    oversample = 1.4 if not args.skip_validation else 1.0

    all_seeds = []
    for tier_name in tiers_to_run:
        tier_config = SEED_TIERS[tier_name]
        effective_target = args.pilot
        if effective_target is None and oversample > 1.0:
            effective_target = int(tier_config["target_count"] * oversample)
        seeds = select_seeds_for_tier(
            client, tier_name, tier_config, VENUES,
            target_override=effective_target, exclude_ids=exclude_ids,
        )
        for s in seeds:
            s["_tier"] = tier_name
        all_seeds.extend(seeds)

    # Validate S2 metadata coverage
    if not args.skip_validation:
        print(f"\n{'='*70}")
        print(f"VALIDATING S2 METADATA COVERAGE (min={args.min_coverage:.0%})")
        print(f"{'='*70}")
        passed, failed = validate_seeds(client, all_seeds, args.min_coverage)
        print(f"\n  Passed: {len(passed)} / {len(all_seeds)}")
        if failed:
            print(f"  Rejected seeds:")
            for s in failed:
                v_short = VENUE_SHORT.get(s.get("venue", "?"), s.get("venue", "?"))
                print(f"    [{s['_tier']:12s}] [{v_short:8s}] "
                      f"coverage={s['_metadata_coverage']:.0%} | {(s.get('title') or '?')[:55]}")

        # Trim back to original target per tier
        trimmed = []
        for tier_name in tiers_to_run:
            tier_target = args.pilot or SEED_TIERS[tier_name]["target_count"]
            tier_seeds = [s for s in passed if s["_tier"] == tier_name]
            # Prefer higher coverage seeds
            tier_seeds.sort(key=lambda s: -s.get("_metadata_coverage", 0))
            trimmed.extend(tier_seeds[:tier_target])
        all_seeds = trimmed

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    tier_counts = defaultdict(int)
    venue_counts = defaultdict(int)
    for s in all_seeds:
        tier_counts[s["_tier"]] += 1
        venue_counts[s.get("venue") or "?"] += 1

    print(f"\nBy tier:")
    for tier in tiers_to_run:
        print(f"  {tier}: {tier_counts[tier]}")

    print(f"\nBy venue:")
    for venue, count in sorted(venue_counts.items(), key=lambda x: -x[1]):
        short = VENUE_SHORT.get(venue, venue)
        print(f"  {short:10s}: {count}")

    print(f"\nTotal: {len(all_seeds)}")

    # Show examples
    print(f"\nExamples:")
    for s in all_seeds[:10]:
        v_short = VENUE_SHORT.get(s.get("venue", "?"), s.get("venue", "?"))
        print(
            f"  [{s['_tier']:12s}] [{v_short:8s}] "
            f"cit={s.get('citationCount', 0):5d} ref={s.get('referenceCount', 0):2d} "
            f"| {(s.get('title') or '?')[:65]}"
        )
    if len(all_seeds) > 10:
        print(f"  ... and {len(all_seeds) - 10} more")

    if args.dry_run:
        print("\n(dry run — not saving)")
        return

    # Save
    out_dir = Path(DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = Path(args.output) if args.output else out_dir / "seeds.json"

    output = []
    for s in all_seeds:
        venue_full = s.get("venue", "")
        entry = {
            "paperId": s["paperId"],
            "arxivId": get_arxiv_id(s),
            "title": s.get("title"),
            "year": s.get("year"),
            "venue": VENUE_SHORT.get(venue_full, venue_full),
            "venueFull": venue_full,
            "citationCount": s.get("citationCount"),
            "referenceCount": s.get("referenceCount"),
            "tier": s["_tier"],
        }
        if "_metadata_coverage" in s:
            entry["metadataCoverage"] = round(s["_metadata_coverage"], 3)
        output.append(entry)

    out_file.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nSaved {len(output)} seeds to {out_file}")


if __name__ == "__main__":
    main()
