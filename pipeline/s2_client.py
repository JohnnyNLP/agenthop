"""Semantic Scholar API client with rate limiting and caching."""

import hashlib
import json
import os
import time
import requests
from pathlib import Path

from config import S2_API_KEY, S2_BASE_URL, S2_RATE_LIMIT, CACHE_DIR


class S2Client:
    def __init__(self):
        self.session = requests.Session()
        if S2_API_KEY:
            self.session.headers["x-api-key"] = S2_API_KEY
        self.last_request_time = 0
        self.cache_dir = Path(CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < S2_RATE_LIMIT:
            time.sleep(S2_RATE_LIMIT - elapsed)
        self.last_request_time = time.time()

    def _cache_path(self, key: str) -> Path:
        h = hashlib.md5(key.encode()).hexdigest()[:12]
        # Keep a readable prefix (first 60 chars) + hash for uniqueness
        safe_key = key.replace("/", "_").replace("?", "_").replace("&", "_")[:60]
        return self.cache_dir / f"{safe_key}_{h}.json"

    def _get(self, endpoint: str, params: dict = None) -> dict:
        cache_key = endpoint + (json.dumps(params, sort_keys=True) if params else "")
        cache_file = self._cache_path(cache_key)

        if cache_file.exists():
            return json.loads(cache_file.read_text())

        self._rate_limit()
        url = f"{S2_BASE_URL}/{endpoint}"
        resp = self.session.get(url, params=params)

        if resp.status_code == 429:
            self._429_count = getattr(self, "_429_count", 0) + 1
            wait = min(30 * self._429_count, 120)
            print(f"Rate limited (429). Waiting {wait}s before retry (attempt {self._429_count})...")
            time.sleep(wait)
            return self._get(endpoint, params)

        if resp.status_code == 504:
            print(f"Gateway timeout (504). Waiting 10s before retry...")
            time.sleep(10)
            return self._get(endpoint, params)

        if resp.status_code != 200:
            if resp.status_code != 500:  # 500s are expected during field-set fallback
                print(f"Error {resp.status_code}: {resp.text[:200]}")
            return {}

        self._429_count = 0  # reset backoff on success
        data = resp.json()
        cache_file.write_text(json.dumps(data, ensure_ascii=False))
        return data

    def get_paper(self, paper_id: str, fields: str = None) -> dict:
        """Get paper details."""
        default_fields = "paperId,externalIds,title,abstract,year,venue,referenceCount,citationCount,authors"
        params = {"fields": fields or default_fields}
        return self._get(f"paper/{paper_id}", params)

    def get_references(self, paper_id: str) -> list:
        """Get papers that this paper cites, with citation contexts."""
        # Try progressively simpler field sets (S2 returns 500 on heavy requests)
        field_sets = [
            "contexts,intents,isInfluential,citedPaper.paperId,citedPaper.title,citedPaper.abstract,citedPaper.year,citedPaper.venue,citedPaper.citationCount,citedPaper.externalIds",
            "contexts,intents,isInfluential,citedPaper.paperId,citedPaper.title,citedPaper.year,citedPaper.citationCount,citedPaper.externalIds",
            "contexts,intents,isInfluential,citedPaper.paperId,citedPaper.title,citedPaper.year,citedPaper.citationCount",
        ]
        for fields in field_sets:
            params = {"fields": fields, "limit": 500}
            data = self._get(f"paper/{paper_id}/references", params)
            if data.get("data"):
                return data["data"]
        return []

    def get_citations(self, paper_id: str) -> list:
        """Get papers that cite this paper, with citation contexts."""
        field_sets = [
            "contexts,intents,isInfluential,citingPaper.paperId,citingPaper.title,citingPaper.abstract,citingPaper.year,citingPaper.venue,citingPaper.citationCount,citingPaper.externalIds",
            "contexts,intents,isInfluential,citingPaper.paperId,citingPaper.title,citingPaper.year,citingPaper.citationCount,citingPaper.externalIds",
            "contexts,intents,isInfluential,citingPaper.paperId,citingPaper.title,citingPaper.year,citingPaper.citationCount",
        ]
        for fields in field_sets:
            params = {"fields": fields, "limit": 500}
            data = self._get(f"paper/{paper_id}/citations", params)
            if data.get("data"):
                return data["data"]
        return []


    def search_papers_bulk(
        self,
        query: str,
        venue: str = None,
        year: str = None,
        min_citation_count: int = None,
        publication_types: str = None,
        fields_of_study: str = None,
        fields: str = None,
        max_pages: int = 10,
    ) -> list:
        """Search papers using bulk endpoint with token-based pagination.

        Returns up to max_pages * 1000 papers.
        The bulk endpoint supports boolean queries (+, |, -, quotes, wildcards).
        """
        default_fields = "paperId,externalIds,title,abstract,year,venue,referenceCount,citationCount,publicationTypes"
        params = {
            "query": query,
            "fields": fields or default_fields,
        }
        if venue:
            params["venue"] = venue
        if year:
            params["year"] = year
        if min_citation_count is not None:
            params["minCitationCount"] = str(min_citation_count)
        if publication_types:
            params["publicationTypes"] = publication_types
        if fields_of_study:
            params["fieldsOfStudy"] = fields_of_study

        all_papers = []
        for page in range(max_pages):
            data = self._get("paper/search/bulk", params)
            papers = data.get("data", [])
            all_papers.extend(papers)

            token = data.get("token")
            if not token or not papers:
                break

            # New params dict for next page (different cache key)
            params = dict(params)
            params["token"] = token

        return all_papers


if __name__ == "__main__":
    client = S2Client()

    # Test with the user's MIRAGE paper
    paper_id = "2504.17137"  # ArXiv ID for MIRAGE
    print(f"\n=== Paper Details ===")
    paper = client.get_paper(paper_id)
    print(f"Title: {paper.get('title')}")
    print(f"Year: {paper.get('year')}")
    print(f"Citations: {paper.get('citationCount')}")
    print(f"References: {paper.get('referenceCount')}")

    print(f"\n=== References (papers MIRAGE cites) ===")
    refs = client.get_references(paper_id)
    print(f"Total references: {len(refs)}")
    for ref in refs[:5]:
        cited = ref.get("citedPaper", {})
        contexts = ref.get("contexts", [])
        intents = ref.get("intents", [])
        print(f"\n  [{cited.get('year')}] {cited.get('title')}")
        print(f"  Intents: {intents}")
        print(f"  Influential: {ref.get('isInfluential')}")
        if contexts:
            print(f"  Context: {contexts[0][:150]}...")

    print(f"\n=== Citations (papers that cite MIRAGE) ===")
    cites = client.get_citations(paper_id)
    print(f"Total citers: {len(cites)}")
    for cite in cites[:5]:
        citing = cite.get("citingPaper", {})
        contexts = cite.get("contexts", [])
        intents = cite.get("intents", [])
        print(f"\n  [{citing.get('year')}] {citing.get('title')}")
        print(f"  Intents: {intents}")
        if contexts:
            print(f"  Context: {contexts[0][:150]}...")
