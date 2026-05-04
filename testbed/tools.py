"""AgentHop evaluation tools — backed by a benchmark sample JSON.

Each tool operates on a pre-built citation graph and paper content pool.
No external API calls; fully deterministic and reproducible.
"""

import re
from dataclasses import dataclass, field


@dataclass
class ToolResult:
    """Result from a tool call."""
    name: str
    args: dict
    output: str
    cost: int  # budget cost of this call


# ── Section-name matching helpers ────────────────────────────────────────────

def _section_alias(i: int) -> str:
    """Alphabet alias for a 0-based section index: 0→A, 25→Z, 26→AA, 27→AB, ..."""
    if i < 26:
        return chr(ord("A") + i)
    return _section_alias(i // 26 - 1) + chr(ord("A") + (i % 26))


def _alias_to_index(letters: str) -> int:
    """Inverse of _section_alias. Input is upper-case A..ZZ..."""
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


_ALIAS_RE = re.compile(r"^\(?([A-Z]{1,3})\)?$")


def _normalize_header(h: str) -> str:
    """Lowercase and strip leading numbering/punctuation for fuzzy matching."""
    h = h.lower().strip()
    h = re.sub(r"^apdx_[a-z][\d.]*:\s*", "", h)
    h = re.sub(r"^apdx:\s*", "", h)
    h = re.sub(r"^[ivxlc]+-?[a-z]?\s+", "", h)
    h = re.sub(r"^[\d.]+\s*", "", h)
    h = re.sub(r"[:\-–—]", " ", h)
    h = re.sub(r"\s+", " ", h).strip()
    return h


def _match_section(query: str, sections: list[dict]) -> int | None:
    """Resolve a section-name query to an index, or None if ambiguous/missing.

    Resolution order (first match wins):
      1. alphabet alias  — "(A)", "A", "B2" (index 27)
      2. exact header    — case-insensitive
      3. normalized exact — after stripping Apdx_/numbering/punctuation
      4. substring       — query is a substring of the header (unique only)
      5. normalized substring — same but over normalized headers (unique only)
    """
    q = (query or "").strip()
    if not q:
        return None

    # 1. alphabet alias
    m = _ALIAS_RE.match(q.upper())
    if m:
        idx = _alias_to_index(m.group(1))
        if 0 <= idx < len(sections):
            return idx

    # 2. exact header (case-insensitive)
    q_lower = q.lower()
    for i, sec in enumerate(sections):
        if sec.get("header", "").strip().lower() == q_lower:
            return i

    # 3. normalized exact
    q_norm = _normalize_header(q)
    if q_norm:
        for i, sec in enumerate(sections):
            if _normalize_header(sec.get("header", "")) == q_norm:
                return i

    # 4. substring (unique)
    hits = [i for i, sec in enumerate(sections)
            if q_lower in sec.get("header", "").lower()]
    if len(hits) == 1:
        return hits[0]

    # 5. normalized substring (unique)
    if q_norm:
        hits = [i for i, sec in enumerate(sections)
                if q_norm in _normalize_header(sec.get("header", ""))]
        if len(hits) == 1:
            return hits[0]

    return None


# ── Tool cost table ──────────────────────────────────────────────────────────

TOOL_COSTS = {
    "get_paper_info": 1,
    "get_references": 1,
    "search_papers": 1,
    "list_sections": 1,
    "read_section": 5,
    "submit_answer": 0,
    "think": 0,
}

# Tool schemas for LLM function calling
TOOL_SCHEMAS = [
    {
        "name": "get_paper_info",
        "description": "Get metadata for a paper: title, abstract, year, and venue.",
        "parameters": {
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": "The paper ID (Semantic Scholar paper ID).",
                }
            },
            "required": ["paper_id"],
        },
    },
    {
        "name": "get_references",
        "description": "Get the list of papers cited by this paper. Returns each cited paper's ID, title, year, and the citation context (the sentence where it was cited).",
        "parameters": {
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": "The paper ID to get references for.",
                }
            },
            "required": ["paper_id"],
        },
    },
    {
        "name": "search_papers",
        "description": "Search for papers in the available pool by keyword query. Returns the top matching papers with their titles and abstracts.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (keywords).",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5, max: 10).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_sections",
        "description": "List the section names of a paper. Each section is labeled with an alphabet alias like (A), (B), ... that you may pass to read_section in place of the full header.",
        "parameters": {
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": "The paper ID.",
                }
            },
            "required": ["paper_id"],
        },
    },
    {
        "name": "read_section",
        "description": "Read the full text of a specific section of a paper. This is the most expensive tool — use it selectively.",
        "parameters": {
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": "The paper ID.",
                },
                "section": {
                    "type": "string",
                    "description": "Section identifier. Accepts: the alphabet alias from list_sections (e.g. '(A)', 'A'), the full header, or a case-insensitive unique substring.",
                },
            },
            "required": ["paper_id", "section"],
        },
    },
    {
        "name": "think",
        "description": "Free scratchpad to organize your thoughts. Use this to plan your next steps, review what you've learned, or reason about the evidence. No budget cost.",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Your thoughts, strategy, or interim reasoning.",
                },
            },
            "required": ["reasoning"],
        },
    },
    {
        "name": "submit_answer",
        "description": "Submit your final answer. You must choose exactly one option (A, B, C, or D). Include your reasoning.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "enum": ["A", "B", "C", "D"],
                    "description": "Your chosen answer (A, B, C, or D).",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of why you chose this answer.",
                },
            },
            "required": ["answer", "reasoning"],
        },
    },
]


class AgentHopTools:
    """Tool implementations backed by a benchmark sample."""

    def __init__(self, sample: dict):
        """Initialize with a loaded benchmark sample JSON.

        Expected sample structure:
            {
                "graph": {"seed": str, "nodes": {...}, "edges": {...}},
                "paper_pool": {arxivId: {"sections": [...], "tables": [...]}},
            }
        """
        self.graph = sample["graph"]
        self.nodes = self.graph["nodes"]       # paperId -> {title, arxivId, year}
        self.edges = self.graph["edges"]       # paperId -> [cited_paperId, ...]
        self.paper_pool = sample["paper_pool"]  # arxivId -> {sections, tables}

        # Build reverse lookup: arxivId -> paperId
        self._aid_to_pid = {}
        for pid, info in self.nodes.items():
            aid = info.get("arxivId", "")
            if aid:
                self._aid_to_pid[aid] = pid

        # Build BM25-like index for search
        self._search_index = self._build_search_index()

    def _build_search_index(self) -> list[dict]:
        """Build simple TF search index over paper pool."""
        index = []
        for pid, info in self.nodes.items():
            text = f"{info.get('title', '')} {info.get('abstract', '')}"
            tokens = set(re.findall(r'\w+', text.lower()))
            index.append({
                "paper_id": pid,
                "title": info.get("title", ""),
                "abstract": info.get("abstract", ""),
                "year": info.get("year"),
                "tokens": tokens,
            })
        return index

    def _get_content(self, paper_id: str) -> dict | None:
        """Get paper content by paperId."""
        aid = self.nodes.get(paper_id, {}).get("arxivId", "")
        if aid and aid in self.paper_pool:
            return self.paper_pool[aid]
        return None

    def _format_authors(self, authors: list[dict] | None, short: bool = False) -> str:
        """Format author list. short=True gives 'First et al.' style."""
        if not authors:
            return "Unknown"
        names = [a.get("name", "") for a in authors if a.get("name")]
        if not names:
            return "Unknown"
        if short:
            if len(names) == 1:
                return names[0]
            if len(names) == 2:
                return f"{names[0]} and {names[1]}"
            return f"{names[0]} et al."
        return ", ".join(names)

    def get_paper_info(self, paper_id: str) -> str:
        """Get paper metadata."""
        info = self.nodes.get(paper_id)
        if not info:
            return f"Paper '{paper_id}' not found in the available paper pool."

        has_content = self._get_content(paper_id) is not None
        authors = self._format_authors(info.get("authors"))
        return (
            f"Title: {info.get('title', 'Unknown')}\n"
            f"Authors: {authors}\n"
            f"Year: {info.get('year', 'Unknown')}\n"
            f"Abstract: {info.get('abstract') or '(not available)'}\n"
            f"Full text available: {'Yes' if has_content else 'No'}"
        )

    def get_references(self, paper_id: str) -> str:
        """Get papers cited by this paper."""
        if paper_id not in self.nodes:
            return f"Paper '{paper_id}' not found in the available paper pool."

        refs = self.edges.get(paper_id, [])
        if not refs:
            return f"No references found for this paper (or references not available in pool)."

        lines = [f"References for paper '{paper_id}' ({len(refs)} papers):\n"]
        for i, ref_pid in enumerate(refs):
            ref_info = self.nodes.get(ref_pid, {})
            title = ref_info.get("title", "Unknown")
            authors = self._format_authors(ref_info.get("authors"), short=True)
            year = ref_info.get("year", "?")
            has_content = self._get_content(ref_pid) is not None
            lines.append(
                f"  [{i+1}] {ref_pid}\n"
                f"      Title: {title}\n"
                f"      Authors: {authors} ({year})\n"
                f"      Full text: {'Available' if has_content else 'Not available'}"
            )

        return "\n".join(lines)

    def search_papers(self, query: str, top_k: int = 5) -> str:
        """Search papers by keyword."""
        top_k = min(max(1, top_k), 10)
        query_tokens = set(re.findall(r'\w+', query.lower()))
        if not query_tokens:
            return "Empty query. Please provide search keywords."

        # Score by token overlap
        scored = []
        for entry in self._search_index:
            overlap = len(query_tokens & entry["tokens"])
            if overlap > 0:
                scored.append((overlap, entry))

        scored.sort(key=lambda x: -x[0])
        results = scored[:top_k]

        if not results:
            return f"No papers found matching '{query}'."

        lines = [f"Search results for '{query}' ({len(results)} matches):\n"]
        for i, (score, entry) in enumerate(results):
            has_content = self._get_content(entry["paper_id"]) is not None
            authors = self._format_authors(
                self.nodes.get(entry["paper_id"], {}).get("authors"), short=True
            )
            abstract = entry["abstract"] or "(no abstract)"
            if len(abstract) > 300:
                abstract = abstract[:300] + "..."
            lines.append(
                f"  [{i+1}] {entry['paper_id']}\n"
                f"      Title: {entry['title']}\n"
                f"      Authors: {authors} ({entry['year']})\n"
                f"      Abstract: {abstract}\n"
                f"      Full text: {'Available' if has_content else 'Not available'}"
            )

        return "\n".join(lines)

    def list_sections(self, paper_id: str) -> str:
        """List available sections for a paper with alphabet aliases."""
        content = self._get_content(paper_id)
        if content is None:
            if paper_id not in self.nodes:
                return f"Paper '{paper_id}' not found in the available paper pool."
            return f"Full text not available for paper '{paper_id}'. Only abstract is available via get_paper_info."

        sections = content.get("sections", [])
        if not sections:
            return f"No sections found for paper '{paper_id}'."

        lines = [f"Sections for paper '{paper_id}' ({len(sections)} sections):\n"]
        for i, sec in enumerate(sections):
            header = sec.get("header", f"Section {i+1}")
            text_len = len(sec.get("text", ""))
            alias = _section_alias(i)
            lines.append(f"  ({alias}) {header} ({text_len} chars)")

        tables = content.get("tables", [])
        if tables:
            lines.append(f"\n  Tables: {len(tables)} available")

        return "\n".join(lines)

    def read_section(self, paper_id: str, section: str) -> str:
        """Read a section. Accepts alphabet alias, full header, or unique substring."""
        content = self._get_content(paper_id)
        if content is None:
            if paper_id not in self.nodes:
                return f"Paper '{paper_id}' not found in the available paper pool."
            return f"Full text not available for paper '{paper_id}'."

        sections = content.get("sections", [])
        if not sections:
            return f"No sections found for paper '{paper_id}'."

        idx = _match_section(section, sections)
        if idx is not None:
            sec = sections[idx]
            return f"=== {sec['header']} ===\n\n{sec['text']}"

        available = [f"({_section_alias(i)}) {s.get('header','')}" for i, s in enumerate(sections)]
        return (
            f"Section '{section}' not found in paper '{paper_id}'.\n"
            f"Available sections: {'; '.join(available)}"
        )

    def call(self, tool_name: str, args: dict) -> ToolResult:
        """Dispatch a tool call and return the result."""
        if tool_name not in TOOL_COSTS:
            return ToolResult(
                name=tool_name,
                args=args,
                output=f"Unknown tool: {tool_name}",
                cost=0,
            )

        handlers = {
            "get_paper_info": lambda: self.get_paper_info(args["paper_id"]),
            "get_references": lambda: self.get_references(args["paper_id"]),
            "search_papers": lambda: self.search_papers(
                args["query"], args.get("top_k", 5)
            ),
            "list_sections": lambda: self.list_sections(args["paper_id"]),
            "read_section": lambda: self.read_section(
                args["paper_id"], args["section"]
            ),
            "think": lambda: f"Noted.",
            "submit_answer": lambda: f"Answer submitted: {args['answer']}",
        }

        handler = handlers.get(tool_name)
        if not handler:
            output = f"Tool '{tool_name}' not implemented."
        else:
            try:
                output = handler()
            except (KeyError, TypeError) as e:
                output = f"Error calling {tool_name}: Missing or invalid parameter — {e}"

        return ToolResult(
            name=tool_name,
            args=args,
            output=output,
            cost=TOOL_COSTS[tool_name],
        )
