"""Central configuration for the Legal Research Assistant."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_DIR = PROCESSED_DIR / "chunks"
EVAL_DIR = DATA_DIR / "evaluation"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# ── Embedding Model ───────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 64
EMBEDDING_DIMENSION = 384

# ── ChromaDB ──────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME = "indian_legal"

# ── Chunking ──────────────────────────────────────────────────────
CHUNK_TARGET_TOKENS = 450  # 300-600 range
CHUNK_MAX_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 50

# ── Retrieval ─────────────────────────────────────────────────────
SEMANTIC_TOP_K = 10
RERANKED_TOP_K = 4
RRF_K = 60  # Reciprocal Rank Fusion constant

# Weights for final scoring
SEMANTIC_WEIGHT = 0.50
RRF_WEIGHT = 0.25
AUTHORITY_WEIGHT = 0.25

# Court authority weights
COURT_WEIGHTS = {
    "supreme_court": 1.0,
    "high_court": 0.7,
    "tribunal": 0.4,
    "statute": 1.0,
}

# Recency weights (years since decision)
RECENCY_BRACKETS = [
    (2, 1.0),
    (5, 0.8),
    (10, 0.6),
    (20, 0.4),
]
RECENCY_DEFAULT = 0.2  # >20 years

# ── LLM Provider ─────────────────────────────────────────────────
GEMINI_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
GEMINI_TEMPERATURE = 0.1
GEMINI_MAX_TOKENS = 1024
GEMINI_RPM_LIMIT = 60  # OpenRouter API rate limit
LLM_BASE_URL = "https://openrouter.ai/api/v1"

# ── Confidence Scoring ────────────────────────────────────────────
CONFIDENCE_WEIGHTS = {
    "retrieval_quality": 0.40,
    "source_agreement": 0.25,
    "authority_strength": 0.20,
    "coverage_score": 0.15,
}

# ── Scraping ──────────────────────────────────────────────────────
INDIANKANOON_DELAY = 1.0  # seconds between requests
MAX_JUDGEMENTS = 200
