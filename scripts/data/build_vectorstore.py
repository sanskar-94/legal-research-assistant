"""Build the ChromaDB vector store from pre-chunked JSON files.

Reads all ``*.json`` files from ``data/processed/chunks/``, embeds the text
of every chunk, and upserts the results into ChromaDB. Chunk ids are
deterministic, so running this again updates chunks in place. Use
``--reset`` to clear the collection first, e.g. after changing the chunker,
so chunks that no longer exist are dropped.

Usage:
    python scripts/data/build_vectorstore.py
    python scripts/data/build_vectorstore.py --reset
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap - allow ``python scripts/data/build_vectorstore.py``
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
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

_UPSERT_BATCH_SIZE = 1000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_chunks_from_file(filepath: Path) -> list[Chunk]:
    """Deserialise a JSON file into a list of Chunk models.

    Supports both a top-level JSON array and a single JSON object.
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


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--reset",
        action="store_true",
        help="delete every indexed chunk before building",
    )
    args = parser.parse_args()

    start_time = time.time()

    chunk_files = sorted(CHUNKS_DIR.glob("*.json"))
    if not chunk_files:
        logger.error(
            "No chunk files in %s. Run prepare_hf_data.py and "
            "chunk_documents.py first.",
            CHUNKS_DIR,
        )
        sys.exit(1)

    # Duplicate ids (e.g. the same chunk in two files) would make ChromaDB
    # reject the whole upsert batch, so keep the first occurrence only.
    unique: dict[str, Chunk] = {}
    loaded_count = 0
    for filepath in chunk_files:
        loaded = _load_chunks_from_file(filepath)
        logger.info("Loaded %d chunk(s) from %s", len(loaded), filepath.name)
        loaded_count += len(loaded)
        for chunk in loaded:
            unique.setdefault(chunk.chunk_id, chunk)

    if not unique:
        logger.error("No valid chunks loaded.")
        sys.exit(1)

    if len(unique) < loaded_count:
        logger.info("Dropped %d duplicate chunk ids.", loaded_count - len(unique))
    all_chunks = list(unique.values())

    embedding_service = EmbeddingService()
    vector_store = VectorStore()
    if args.reset:
        vector_store.reset()

    # Embed and upsert batch by batch so memory stays flat on large corpora
    total = len(all_chunks)
    for start in range(0, total, _UPSERT_BATCH_SIZE):
        batch_chunks = all_chunks[start:start + _UPSERT_BATCH_SIZE]
        batch_embeddings = embedding_service.embed_batch(
            [chunk.text for chunk in batch_chunks],
            batch_size=EMBEDDING_BATCH_SIZE,
        )
        vector_store.add_chunks(batch_chunks, batch_embeddings)
        logger.info("Indexed %d / %d chunks", start + len(batch_chunks), total)

    logger.info(
        "Build complete: %d chunks in the collection (%.1f s).",
        vector_store.count(),
        time.time() - start_time,
    )


if __name__ == "__main__":
    main()
