"""Central configuration for the Legal Research Assistant."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_DIR = PROCESSED_DIR / "chunks"
EVAL_DIR = DATA_DIR / "evaluation"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# Embedding Model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 64

# ChromaDB
CHROMA_COLLECTION_NAME = "indian_legal"

# Chunking
CHUNK_TARGET_TOKENS = 450  # 300-600 range
CHUNK_MAX_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 50

# Retrieval
SEMANTIC_TOP_K = 20
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

# LLM provider
# Any OpenAI-compatible endpoint works. The default is DeepSeek V3.2 served
# through NVIDIA NIM. Override with LLM_BASE_URL / LLM_MODEL in .env.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-ai/deepseek-v3.2")
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024
LLM_RPM_LIMIT = 40  # NVIDIA NIM free tier allows 40 requests per minute

# Confidence Scoring
CONFIDENCE_WEIGHTS = {
    "retrieval_quality": 0.40,
    "source_agreement": 0.25,
    "authority_strength": 0.20,
    "coverage_score": 0.15,
}

# IPC to BNS correspondence table
IPC_BNS_MAPPING_PATH = PROJECT_ROOT / "config" / "ipc_bns_mapping.json"

# Scraping
INDIANKANOON_DELAY = 1.0  # seconds between requests
MAX_JUDGEMENTS = 200

# Answer writer. "api" = OpenAI-compatible endpoint above (LLMClient);
# "local" = Nyaya-Gen, the fine-tuned SmolLM2-135M in models/nyaya-gen
# (scripts/train_generator.py). Sources are cut to the length used in training.
LLM_BACKEND = os.getenv("LLM_BACKEND", "api").strip().lower()
LOCAL_GENERATOR_BASE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
LOCAL_GENERATOR_PATH = Path(os.getenv("NYAYA_GEN_PATH", PROJECT_ROOT / "models" / "nyaya-gen"))
LOCAL_GENERATOR_SOURCE_WORDS = 90
LOCAL_GENERATOR_MAX_NEW_TOKENS = 256
