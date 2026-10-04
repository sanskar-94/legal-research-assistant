"""Pipeline script: chunk all parsed documents and persist as JSON.

Reads every JSON file in ``data/processed/``, applies the appropriate
chunking strategy based on ``doc_type``, and writes the resulting chunks
to ``data/processed/chunks/``.

Usage:
    python scripts/chunk_documents.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap — allow ``python scripts/chunk_documents.py``
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import CHUNKS_DIR, PROCESSED_DIR  # noqa: E402
from src.chunking.chunk_models import Chunk  # noqa: E402
from src.chunking.legal_chunker import chunk_document  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("chunk_documents")

# ---------------------------------------------------------------------------
# Token estimation (mirrors legal_chunker)
# ---------------------------------------------------------------------------
_WORDS_TO_TOKENS_FACTOR = 1.3


def _estimate_tokens(text: str) -> int:
    return int(len(text.split()) * _WORDS_TO_TOKENS_FACTOR)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_documents(source_dir: Path) -> list[dict]:
    """Load all JSON documents from *source_dir* (non-recursive).

    Each JSON file may contain either a single document dict or a list
    of document dicts.  Returns a flat list of all documents found.
    """
    documents: list[dict] = []
    json_files = sorted(source_dir.glob("*.json"))

    if not json_files:
        logger.warning("No JSON files found in %s", source_dir)
        return documents

    for json_path in json_files:
        try:
            with json_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to read %s: %s", json_path, exc)
            continue

        if isinstance(data, list):
            documents.extend(data)
            logger.info("Loaded %d documents from %s", len(data), json_path.name)
        elif isinstance(data, dict):
            documents.append(data)
            logger.info("Loaded 1 document from %s", json_path.name)
        else:
            logger.warning("Unexpected root type in %s — skipping", json_path.name)

    return documents


def _serialize_chunks(chunks: list[Chunk]) -> list[dict]:
    """Convert a list of Chunk models to plain dicts for JSON output."""
    return [chunk.model_dump() for chunk in chunks]


def _save_chunks(chunks: list[dict], output_path: Path) -> None:
    """Write serialised chunks to a JSON file."""
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(chunks, fh, ensure_ascii=False, indent=2)
    logger.info("  Saved %d chunks → %s", len(chunks), output_path.name)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory ready: %s", CHUNKS_DIR)

    documents = _load_documents(PROCESSED_DIR)
    if not documents:
        logger.warning("No documents to chunk. Exiting.")
        return

    total_chunks = 0
    total_tokens = 0
    doc_success = 0
    doc_failure = 0

    all_chunks: list[dict] = []

    for doc in documents:
        doc_id = doc.get("doc_id", "unknown")
        doc_type = doc.get("doc_type", "").lower()
        text = doc.get("text", "") or doc.get("content", "")

        if not text:
            logger.warning("Document %s has no text — skipping.", doc_id)
            doc_failure += 1
            continue

        if not doc_type:
            logger.warning(
                "Document %s has no doc_type — skipping.", doc_id
            )
            doc_failure += 1
            continue

        # Build metadata dict from document fields (excluding the text body)
        metadata = {
            key: value
            for key, value in doc.items()
            if key not in ("text", "content")
        }

        try:
            chunks = chunk_document(text, doc_type, metadata)
        except ValueError as exc:
            logger.error("Skipping document %s: %s", doc_id, exc)
            doc_failure += 1
            continue

        serialised = _serialize_chunks(chunks)
        all_chunks.extend(serialised)

        chunk_tokens = sum(_estimate_tokens(c.text) for c in chunks)
        total_chunks += len(chunks)
        total_tokens += chunk_tokens
        doc_success += 1

        logger.info(
            "  %s (%s): %d chunks, ~%d tokens",
            doc_id,
            doc_type,
            len(chunks),
            chunk_tokens,
        )

    # Persist all chunks to a single output file
    if all_chunks:
        _save_chunks(all_chunks, CHUNKS_DIR / "all_chunks.json")

    # Also save per-doc-type files for convenience
    by_type: dict[str, list[dict]] = {}
    for chunk in all_chunks:
        dt = chunk.get("metadata", {}).get("doc_type", "unknown")
        by_type.setdefault(dt, []).append(chunk)

    for doc_type, typed_chunks in by_type.items():
        _save_chunks(typed_chunks, CHUNKS_DIR / f"{doc_type}_chunks.json")

    # ── Summary stats ─────────────────────────────────────────────
    avg_tokens = total_tokens / total_chunks if total_chunks else 0
    logger.info("=" * 60)
    logger.info("Chunking pipeline complete.")
    logger.info("  Documents processed : %d", doc_success)
    logger.info("  Documents skipped   : %d", doc_failure)
    logger.info("  Total chunks        : %d", total_chunks)
    logger.info("  Total tokens (est.) : %d", total_tokens)
    logger.info("  Avg tokens/chunk    : %.1f", avg_tokens)
    logger.info("=" * 60)

    if doc_failure > 0 and doc_success == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
