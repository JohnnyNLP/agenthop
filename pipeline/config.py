"""AgentHop pipeline configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# Semantic Scholar API
S2_API_KEY = os.getenv("S2_API_KEY", "")
S2_BASE_URL = "https://api.semanticscholar.org/graph/v1"
S2_RATE_LIMIT = 2.0  # seconds between requests (1.2-1.5 triggers 429s with free key)

# Arxiv HTML
ARXIV_HTML_URL = "https://arxiv.org/html/{arxiv_id}"
AR5IV_HTML_URL = "https://ar5iv.labs.arxiv.org/html/{arxiv_id}"

# Target venues (10 top CS conferences across 4 subfields)
# Use S2's stored venue names — short forms work for most, but NeurIPS/ICLR
# need full names for the bulk search endpoint.
VENUES = [
    "ACL", "EMNLP", "NAACL",                              # NLP
    "Neural Information Processing Systems",               # ML (NeurIPS)
    "International Conference on Machine Learning",        # ML (ICML)
    "International Conference on Learning Representations", # ML (ICLR)
    "Computer Vision and Pattern Recognition",             # CV (CVPR)
    "International Conference on Computer Vision",         # CV (ICCV)
    "European Conference on Computer Vision",              # CV (ECCV)
    "Annual International ACM SIGIR Conference on Research and Development in Information Retrieval",  # IR (SIGIR)
]

# Short names for display
VENUE_SHORT = {
    "ACL": "ACL", "EMNLP": "EMNLP", "NAACL": "NAACL",
    "Neural Information Processing Systems": "NeurIPS",
    "International Conference on Machine Learning": "ICML",
    "International Conference on Learning Representations": "ICLR",
    "Computer Vision and Pattern Recognition": "CVPR",
    "International Conference on Computer Vision": "ICCV",
    "European Conference on Computer Vision": "ECCV",
    "Annual International ACM SIGIR Conference on Research and Development in Information Retrieval": "SIGIR",
}

# Seed paper stratification: three tiers for contamination analysis
SEED_TIERS = {
    "recent":      {"year_range": (2024, 2025), "min_citations": 0,   "target_count": 500},
    "established": {"year_range": (2023, 2023), "min_citations": 50,  "target_count": 300},
    "well_known":  {"year_range": (2022, 2022), "min_citations": 300, "target_count": 200},
}

# Seed paper criteria (applied across all tiers)
SEED_MIN_REFERENCES = 15    # enough for chain construction
SEED_MAX_REFERENCES = 50    # excludes surveys

# Citation chain criteria
CHAIN_MIN_CONTEXT_LENGTH = 80       # min chars for citation context to be "substantive"
CHAIN_VALID_INTENTS = ["methodology", "result"]  # skip "background"-only
CHAIN_CITED_YEAR_MIN = 2020         # cited papers should be 2020+ (2 yrs before oldest seed tier)

# Chain quality filter
FILTER_MIN_JACCARD = 0.08       # min reference overlap (Jaccard) for Layer 1
FILTER_LLM_THRESHOLD = 4       # min LLM score (1-5) for Layer 3

# Hierarchical extraction (1.extract_and_filter.py)
EXTRACT_MAX_HOP1 = 30           # top hop1 candidates by S2 signals to screen
EXTRACT_MAX_HOP2 = 30           # top hop2 candidates per surviving hop1
FILTER_HOP1_THRESHOLD = 4       # min score from Phase 1 to explore hop2

# N-hop extraction (generalized)
EXTRACT_CANDIDATES_PER_HOP = [10, 7]      # candidates to screen at each depth (max depth 2)
EXTRACT_MAX_DEPTH = 2                      # hard cap — depth 3 excluded

# Seed validation
SEED_MIN_METADATA_COVERAGE = 0.5  # min fraction of refs with citation contexts

# Question mode skill distribution (weights, must sum to ~1.0)
SKILL_DISTRIBUTION = {
    "retrieve":   0.10,
    "reason":     0.20,
    "contrast":   0.20,
    "assess":     0.15,
    "synthesize": 0.20,
    "justify":    0.15,
}

# Paths — anchored to this file so writes always land next to the pipeline
# code, regardless of the caller's current working directory.
_PIPELINE_DIR = Path(__file__).resolve().parent
DATA_DIR = str(_PIPELINE_DIR / "data")
CACHE_DIR = str(_PIPELINE_DIR / "data" / "cache")
CHAINS_DIR = str(_PIPELINE_DIR / "data" / "chains")
QUESTIONS_DIR = str(_PIPELINE_DIR / "data" / "questions")
