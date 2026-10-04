"""Build the ChromaDB vector store from pre-chunked JSON files.

Reads all ``*.json`` files from ``data/processed/chunks/``, embeds the text
of every chunk, and upserts the results into ChromaDB.

Usage:
    python scripts/build_vectorstore.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap - allow ``python scripts/build_vectorstore.py``
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import CHUNKS_DIR, EMBEDDING_BATCH_SIZE  # noqa: E402
from src.chunking.chunk_models import Chunk  # noqa: E402
from src.embeddings.embedding_service import EmbeddingService  # noqa: E402
from src.embeddings.vectorstore import VectorStore  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("build_vectorstore")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_chunks_from_file(filepath: Path) -> list[Chunk]:
    """Deserialise a JSON file into a list of Chunk models.

    Supports both a top-level JSON array and a single JSON object.
    Returns a *new* list - the file is never modified.
    """
    try:
        with filepath.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to read '%s': %s", filepath, exc)
        return []

    if isinstance(raw, dict):
        raw = [raw]

    if not isinstance(raw, list):
        logger.warning("Unexpected JSON structure in '%s' - skipping.", filepath)
        return []

    chunks: list[Chunk] = []
    for idx, entry in enumerate(raw):
        try:
            chunks.append(Chunk.model_validate(entry))
        except Exception as exc:
            logger.warning(
                "Skipping invalid chunk #%d in '%s': %s", idx, filepath, exc
            )
    return chunks


def _discover_chunk_files(chunks_dir: Path) -> list[Path]:
    """Return all JSON files inside *chunks_dir*, sorted by name."""
    if not chunks_dir.exists():
        logger.warning("Chunks directory does not exist: %s", chunks_dir)
        return []

    files = sorted(chunks_dir.glob("*.json"))
    if not files:
        logger.warning("No JSON files found in %s", chunks_dir)
    return files


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=== Build Vector Store ===")
    start_time = time.time()

    # 1. Discover chunk files
    chunk_files = _discover_chunk_files(CHUNKS_DIR)
    if not chunk_files:
        logger.info("Nothing to process. Exiting.")
        return

    logger.info("Found %d chunk file(s) in %s.", len(chunk_files), CHUNKS_DIR)

    # 2. Load all chunks
    all_chunks: list[Chunk] = []
    for filepath in chunk_files:
        loaded = _load_chunks_from_file(filepath)
        logger.info("  Loaded %d chunk(s) from %s", len(loaded), filepath.name)
        all_chunks.extend(loaded)

    if not all_chunks:
        logger.info("No valid chunks loaded. Exiting.")
        return

    logger.info("Total chunks loaded: %d", len(all_chunks))

    # Duplicate ids (e.g. the same chunk in two files) would make ChromaDB
    # reject the whole upsert batch, so keep the first occurrence only.
    unique: dict[str, Chunk] = {}
    for chunk in all_chunks:
        unique.setdefault(chunk.chunk_id, chunk)
    if len(unique) < len(all_chunks):
        logger.info("Dropped %d duplicate chunk ids.", len(all_chunks) - len(unique))
    all_chunks = list(unique.values())

    # 3. Embed and upsert batch by batch so memory stays flat on large corpora
    embedding_service = EmbeddingService()
    vector_store = VectorStore()
    upsert_batch_size = 1000
    total = len(all_chunks)
    for start in range(0, total, upsert_batch_size):
        end = min(start + upsert_batch_size, total)
        batch_chunks = all_chunks[start:end]
        batch_embeddings = embedding_service.embed_batch(
            [chunk.text for chunk in batch_chunks],
            batch_size=EMBEDDING_BATCH_SIZE,
        )
        vector_store.add_chunks(batch_chunks, batch_embeddings)
        logger.info("  Indexed %d / %d chunks", end, total)

    # 4. Report
    elapsed = time.time() - start_time
    stats = vector_store.get_collection_stats()
    logger.info("=== Build Complete ===")
    logger.info("  Collection count : %d", stats["count"])
    logger.info("  Elapsed time     : %.1f s", elapsed)

    if stats["sample_metadata"]:
        logger.info("  Sample metadata  : %s", stats["sample_metadata"][0])


if __name__ == "__main__":
    main()
