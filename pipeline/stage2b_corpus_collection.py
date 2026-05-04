"""Stage 2 (chain collection — corpus half): fetch + parse the section-structured paper bodies.

Fetches and parses the section-structured body of every paper in the
expanded citation neighbourhoods. Primary source is ar5iv (arXiv's HTML5
rendering); falls back to arxiv.org/html/ when ar5iv is unavailable.

Usage:
    python 2.fetch_html.py                    # process all chain files
    python 2.fetch_html.py --skip-cached      # only fetch papers not already cached
    python 2.fetch_html.py --parse-only       # re-parse cached HTML (no fetching)
"""

import argparse
import json
import glob
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from config import ARXIV_HTML_URL, AR5IV_HTML_URL, CACHE_DIR, CHAINS_DIR


HTML_CACHE = Path(CACHE_DIR) / "html"
HTML_CACHE.mkdir(parents=True, exist_ok=True)

CONTENT_DIR = Path(CHAINS_DIR) / "content"
CONTENT_DIR.mkdir(parents=True, exist_ok=True)


# ── Fetch ────────────────────────────────────────────────────────────────────

def fetch_paper_html(arxiv_id: str, delay: float = 1.0) -> str | None:
    """Fetch HTML for an arxiv paper. Returns raw HTML or None."""
    if not arxiv_id:
        return None

    cache_file = HTML_CACHE / f"{arxiv_id.replace('/', '_')}.html"
    if cache_file.exists():
        return cache_file.read_text()

    # Try ar5iv first (better HTML quality), then arxiv
    for url_template in [AR5IV_HTML_URL, ARXIV_HTML_URL]:
        url = url_template.format(arxiv_id=arxiv_id)
        try:
            time.sleep(delay)
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200 and len(resp.text) > 1000:
                cache_file.write_text(resp.text)
                return resp.text
        except requests.RequestException as e:
            print(f"  Warning: {url}: {e}")
            continue

    return None


# ── Parse ────────────────────────────────────────────────────────────────────

def _strip_math_annotations(el) -> None:
    """Remove MathML annotation layers from an element in-place.

    ar5iv renders math as MathML with three parallel representations:
      1. Presentation MathML (Unicode symbols) — what we keep
      2. <annotation-xml> (MathML-Content, semantic) — duplicate
      3. <annotation> (raw LaTeX source) — duplicate

    Stripping layers 2-3 eliminates ~34% of noise in math-heavy paragraphs.
    """
    for tag in el.find_all(["annotation", "annotation-xml"]):
        tag.decompose()


_EXCLUDED_SECTIONS = re.compile(
    r"^(references|bibliography|acknowledg[e]?ments?|acknowledg|"
    r"checklist|.*paper checklist|version control|author contrib|contributions|"
    r"ethics|ethics statement|ethics and impact|ethical|"
    r"disclosure of funding|broader impact|impact statement|"
    r"reproducibility|reproducibility statement|"
    r"summary of contributions|"
    r"Список литературы)",  # Russian "References"
    re.IGNORECASE,
)
# Appendix/supplementary sections are included but prefixed for clarity
_APPENDIX_SECTIONS = re.compile(
    r"^(appendix|supplementary|Приложение)",
    re.IGNORECASE,
)


def _is_top_level_section(section_el) -> bool:
    """Check if a <section> tag is top-level (no parent <section>)."""
    parent = section_el.parent
    while parent:
        if parent.name == "section":
            return False
        parent = parent.parent
    return True


def parse_sections(html_or_soup) -> list[dict]:
    """Parse HTML into main-body sections using heading level (h2).

    Uses <h2> tags to identify main sections, rolling all subsection
    content (h3-h6) into the parent. This is more robust than nesting-based
    detection since ar5iv HTML structure varies across papers.

    References and acknowledgements are excluded.
    Appendix/supplementary sections are included with a prefix.
    """
    if isinstance(html_or_soup, str):
        soup = BeautifulSoup(html_or_soup, "html.parser")
        _strip_math_annotations(soup)
    else:
        soup = html_or_soup

    sections = []

    # Strategy: find all <section> elements that have an <h2> header.
    # These are main sections. All content within (including nested subsections)
    # is rolled up into the parent section text.
    # Fallback: if no <h2> sections found, try the old nesting-based approach.

    h2_sections = []
    for section_el in soup.find_all("section"):
        # Direct h2 child (not nested h2 from a child section)
        header_el = section_el.find(re.compile(r"^h[1-6]$"), recursive=False)
        if not header_el:
            # Try first header in immediate children
            for child in section_el.children:
                if hasattr(child, 'name') and child.name and re.match(r'^h[1-6]$', child.name):
                    header_el = child
                    break
        if header_el and header_el.name == "h2":
            h2_sections.append((section_el, header_el))

    # Fallback 1: if h2 yields ≤1 body section, try h3 (some arxiv HTML templates
    # use h3 for main sections, with h2 only for "Part I Appendix")
    if len(h2_sections) <= 1:
        h3_sections = []
        for section_el in soup.find_all("section"):
            header_el = section_el.find(re.compile(r"^h[1-6]$"), recursive=False)
            if not header_el:
                for child in section_el.children:
                    if hasattr(child, 'name') and child.name and re.match(r'^h[1-6]$', child.name):
                        header_el = child
                        break
            if header_el and header_el.name == "h3":
                h3_sections.append((section_el, header_el))
        if len(h3_sections) > len(h2_sections):
            h2_sections = h3_sections

    # Fallback 2: if still no sections, use top-level nesting detection
    if not h2_sections:
        for section_el in soup.find_all("section"):
            if not _is_top_level_section(section_el):
                continue
            header_el = section_el.find(re.compile(r"^h[1-6]$"))
            if header_el:
                h2_sections.append((section_el, header_el))

    for section_el, header_el in h2_sections:
        header = header_el.get_text(strip=True)
        # Clean up header (remove section numbers like "1.", "2.1", "IIntroduction", etc.)
        header = re.sub(r"^[IVXLC]+(?=[A-Z])", "", header)  # Roman numerals glued to title
        header = re.sub(r"^[\d.]+\s*", "", header)

        # Skip non-body sections (references, acknowledgements)
        if _EXCLUDED_SECTIONS.match(header):
            continue

        # Normalize appendix/supplementary headers to unified format
        # Pattern 1: "Appendix A...", "Supplementary Material..."
        if _APPENDIX_SECTIONS.match(header):
            rest = re.sub(r"^(appendix|supplementary|Приложение)\s*:?\s*", "", header, flags=re.IGNORECASE)
            rest = re.sub(r"^[\d.]+\s*", "", rest)  # strip leading numerals (e.g., "0.A...")
            m = re.match(r"^([A-Z])[\s:]*(.*)$", rest)
            if m:
                letter, title = m.group(1), m.group(2).strip()
                header = f"Apdx_{letter}: {title}" if title else f"Apdx_{letter}"
            elif rest.strip():
                header = f"Apdx: {rest.strip()}"
            else:
                header = "Apdx"
        # Pattern 2: "A.1Title", "S.3Details" — appendix subsections at h2 level
        else:
            m_sub = re.match(r"^([AS])\.(\d+)\s*(.*)", header)
            if m_sub:
                letter, num, title = m_sub.group(1), m_sub.group(2), m_sub.group(3).strip()
                header = f"Apdx_{letter}.{num}: {title}" if title else f"Apdx_{letter}.{num}"

        # Remove excluded child sections (e.g. References nested inside Discussion)
        for child_sec in section_el.find_all("section"):
            child_h = child_sec.find(re.compile(r"^h[1-6]$"))
            if child_h:
                child_header = re.sub(r"^[\d.]+\s*", "", child_h.get_text(strip=True))
                if _EXCLUDED_SECTIONS.match(child_header):
                    child_sec.decompose()

        # Get all text from the section (including remaining subsections)
        text = section_el.get_text(separator=" ", strip=True)
        # Remove the header text from the beginning
        raw_header = header_el.get_text(strip=True)
        if text.startswith(raw_header):
            text = text[len(raw_header):].strip()
        paragraphs = [text] if text else []

        if paragraphs:
            sections.append({
                "header": header,
                "text": "\n\n".join(paragraphs),
            })

    # Fallback: if no <section> tags, try splitting by headers
    if not sections:
        for header_el in soup.find_all(re.compile(r"^h[1-6]$")):
            header = header_el.get_text(strip=True)
            header = re.sub(r"^[\d.]+\s*", "", header)

            if _EXCLUDED_SECTIONS.match(header):
                continue

            content_parts = []
            for sibling in header_el.find_next_siblings():
                if sibling.name and re.match(r"^h[1-6]$", sibling.name):
                    break
                text = sibling.get_text(separator=" ", strip=True)
                if text:
                    content_parts.append(text)

            if content_parts:
                sections.append({
                    "header": header,
                    "text": "\n\n".join(content_parts),
                })

    return sections


def extract_tables(html_or_soup) -> list[dict]:
    """Extract tables with their captions."""
    if isinstance(html_or_soup, str):
        soup = BeautifulSoup(html_or_soup, "html.parser")
        _strip_math_annotations(soup)
    else:
        soup = html_or_soup
    tables = []

    for table_container in soup.find_all(class_=re.compile(r"ltx_table|table-wrapper")):
        caption_el = table_container.find(class_=re.compile(r"ltx_caption|caption"))
        caption = caption_el.get_text(strip=True) if caption_el else ""

        table_el = table_container.find("table")
        if not table_el:
            continue

        rows = []
        for tr in table_el.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if cells:
                rows.append(cells)

        if rows:
            tables.append({"caption": caption, "rows": rows})

    # Fallback: plain <table> tags
    if not tables:
        for table_el in soup.find_all("table"):
            rows = []
            for tr in table_el.find_all("tr"):
                cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                if cells:
                    rows.append(cells)
            if rows and len(rows) > 1:
                tables.append({"caption": "", "rows": rows})

    return tables


def parse_paper(html: str) -> dict:
    """Parse a paper's HTML into structured content."""
    soup = BeautifulSoup(html, "html.parser")
    _strip_math_annotations(soup)
    sections = parse_sections(soup)
    tables = extract_tables(soup)
    return {
        "sections": sections,
        "tables": tables,
        "num_sections": len(sections),
        "num_tables": len(tables),
    }


# ── Collect arxiv IDs from chains ────────────────────────────────────────────

def collect_arxiv_ids() -> dict[str, dict]:
    """Scan all chain files and collect unique arxiv IDs with metadata.

    Returns dict: arxiv_id -> {paperId, title, role, depth}
    """
    papers = {}  # arxiv_id -> info

    for fpath in sorted(glob.glob(str(Path(CHAINS_DIR) / "chains_d*.json"))):
        with open(fpath) as f:
            chains = json.load(f)

        for chain in chains:
            # Seed paper (depth 0)
            seed = chain.get("seed", {})
            aid = seed.get("arxivId")
            if aid and aid not in papers:
                papers[aid] = {
                    "paperId": seed.get("paperId"),
                    "title": seed.get("title", ""),
                    "role": "seed",
                    "depth": 0,
                }

            # Path papers (intermediate hops)
            for i, hop in enumerate(chain.get("path", [])):
                paper = hop.get("paper", {})
                aid = paper.get("arxivId")
                if aid and aid not in papers:
                    papers[aid] = {
                        "paperId": paper.get("paperId"),
                        "title": paper.get("title", ""),
                        "role": "path",
                        "depth": i + 1,
                    }

            # Terminal paper
            term = chain.get("terminal", {})
            aid = term.get("arxivId")
            if aid and aid not in papers:
                papers[aid] = {
                    "paperId": term.get("paperId"),
                    "title": term.get("title", ""),
                    "role": "terminal",
                    "depth": chain.get("depth", 0),
                }

    return papers


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fetch arxiv HTML for chain papers")
    parser.add_argument("--skip-cached", action="store_true",
                        help="Only report; don't re-fetch cached papers")
    parser.add_argument("--parse-only", action="store_true",
                        help="Re-parse cached HTML without fetching new papers")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Delay between requests in seconds (default: 1.0)")
    args = parser.parse_args()

    # Collect all arxiv IDs from chains
    papers = collect_arxiv_ids()
    print(f"Unique papers with arxiv IDs: {len(papers)}")

    # Check cache
    cached_ids = set()
    for f in HTML_CACHE.iterdir():
        if f.suffix == ".html":
            cached_ids.add(f.stem.replace("_", "/"))

    already_cached = set(papers.keys()) & cached_ids
    to_fetch = set(papers.keys()) - cached_ids
    print(f"Already cached: {len(already_cached)}")
    print(f"Need to fetch: {len(to_fetch)}")

    # Fetch
    if not args.parse_only:
        failed = []
        for i, arxiv_id in enumerate(sorted(to_fetch)):
            info = papers[arxiv_id]
            print(f"[{i+1}/{len(to_fetch)}] {arxiv_id} — {info['title'][:60]}...", end="", flush=True)
            html = fetch_paper_html(arxiv_id, delay=args.delay)
            if html:
                print(f" OK ({len(html)//1024}KB)")
            else:
                print(" FAILED")
                failed.append(arxiv_id)

        print(f"\nFetch complete: {len(to_fetch) - len(failed)}/{len(to_fetch)} succeeded")
        if failed:
            print(f"Failed ({len(failed)}):")
            for aid in failed:
                print(f"  {aid}: {papers[aid]['title'][:70]}")
    else:
        print("Parse-only mode — skipping fetch")

    # Parse all cached HTML into structured content
    print(f"\nParsing all cached HTML...")
    content = {}
    total_sections = 0
    total_tables = 0
    parse_failed = 0

    for arxiv_id in sorted(papers.keys()):
        cache_file = HTML_CACHE / f"{arxiv_id.replace('/', '_')}.html"
        if not cache_file.exists():
            continue

        html = cache_file.read_text()
        parsed = parse_paper(html)

        if parsed["num_sections"] == 0:
            parse_failed += 1
            continue

        content[arxiv_id] = parsed
        total_sections += parsed["num_sections"]
        total_tables += parsed["num_tables"]

    # Save
    out_file = CONTENT_DIR / "paper_contents.json"
    out_file.write_text(json.dumps(content, indent=2, ensure_ascii=False))

    print(f"\nParsed: {len(content)}/{len(papers)} papers")
    print(f"Total sections: {total_sections} (mean {total_sections/max(len(content),1):.1f}/paper)")
    print(f"Total tables: {total_tables} (mean {total_tables/max(len(content),1):.1f}/paper)")
    if parse_failed:
        print(f"Parse failures (0 sections): {parse_failed}")
    print(f"Saved to: {out_file}")

    # Summary by role
    roles = {"seed": 0, "path": 0, "terminal": 0}
    for aid in content:
        roles[papers[aid]["role"]] += 1
    print(f"\nBy role: seed={roles['seed']}, path={roles['path']}, terminal={roles['terminal']}")


if __name__ == "__main__":
    main()
