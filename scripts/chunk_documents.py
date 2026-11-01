"""Chunk every parsed document and write the chunks as JSON.

Reads the JSON files under ``data/processed/`` (recursively, skipping the
``chunks/`` output folder). Two input shapes are accepted:

* records from ``prepare_hf_data.py``: ``doc_id``, ``doc_type``, ``text``
  plus flat metadata fields;
* files from ``parse_pdfs.py``: ``full_text`` plus a nested ``metadata``
  dict from the metadata extractor. ``doc_type`` and ``act_name`` are
  inferred from where the source file came from.

All chunks go to ``data/processed/chunks/all_chunks.json``.

Usage:
    python scripts/chunk_documents.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap - allow ``python scripts/chunk_documents.py``
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import CHUNKS_DIR, PROCESSED_DIR  # noqa: E402
from src.chunking.legal_chunker import chunk_document, estimate_tokens  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("chunk_documents")

# Statute slugs written by download_statutes.py
_ACT_NAMES: dict[str, str] = {
    "constitution_of_india": "Constitution of India",
    "ipc_1860": "Indian Penal Code, 1860",
    "bns_2023": "Bharatiya Nyaya Sanhita, 2023",
    "crpc_1973": "Code of Criminal Procedure, 1973",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_parsed_file(doc: dict, json_path: Path, source_dir: Path) -> dict:
    """Convert a ``parse_pdfs.py`` output file into a chunker record."""
    relative = json_path.relative_to(source_dir)
    slug = relative.stem
    parts = {part.lower() for part in relative.parts[:-1]}
    extracted = doc.get("metadata") or {}

    if doc.get("doc_type"):
        doc_type = doc["doc_type"]
    elif "constitution" in slug:
        doc_type = "constitution"
    elif "statutes" in parts:
        doc_type = "statute"
    else:
        doc_type = "judgement"

    citations = extracted.get("citations") or []
    record = {
        "doc_id": doc.get("doc_id") or str(relative.with_suffix("")).replace("/", "_"),
        "doc_type": doc_type,
        "text": doc["full_text"],
        "case_name": extracted.get("case_name"),
        "citation": citations[0] if citations else None,
        "court": extracted.get("court"),
        "date": extracted.get("date_of_judgment"),
        "judges": list(extracted.get("bench") or []) or None,
        "section_refs": list(extracted.get("statutes_cited") or []),
    }
    if doc_type in ("statute", "constitution"):
        record["act_name"] = doc.get("act_name") or _ACT_NAMES.get(slug, slug)
    return record


def _load_documents(source_dir: Path) -> list[dict]:
    """Load all documents under *source_dir*, excluding the chunks folder.

    Each JSON file may contain a single document dict or a list of them.
    """
    documents: list[dict] = []
    json_files = sorted(
        path for path in source_dir.rglob("*.json")
        if CHUNKS_DIR not in path.parents
    )

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
        elif isinstance(data, dict) and data.get("full_text"):
            documents.append(_normalise_parsed_file(data, json_path, source_dir))
            logger.info("Loaded parsed file %s", json_path.relative_to(source_dir))
        elif isinstance(data, dict):
            documents.append(data)
            logger.info("Loaded 1 document from %s", json_path.name)
        else:
            logger.warning("Unexpected root type in %s - skipping", json_path.name)

    return documents


def _save_chunks(chunks: list[dict], output_path: Path) -> None:
    """Write serialised chunks to a JSON file."""
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(chunks, fh, ensure_ascii=False, indent=2)
    logger.info("  Saved %d chunks -> %s", len(chunks), output_path.name)


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
            logger.warning("Document %s has no text - skipping.", doc_id)
            doc_failure += 1
            continue

        if not doc_type:
            logger.warning(
                "Document %s has no doc_type - skipping.", doc_id
            )
            doc_failure += 1
            continue

        # Build metadata dict from document fields (excluding the text body
        # and empty values, which ChunkMetadata fills with defaults)
        metadata = {
            key: value
            for key, value in doc.items()
            if key not in ("text", "content", "full_text") and value is not None
        }

        try:
            chunks = chunk_document(text, doc_type, metadata)
        except ValueError as exc:
            logger.error("Skipping document %s: %s", doc_id, exc)
            doc_failure += 1
            continue

        all_chunks.extend(chunk.model_dump() for chunk in chunks)

        chunk_tokens = sum(estimate_tokens(c.text) for c in chunks)
        total_chunks += len(chunks)
        total_tokens += chunk_tokens
        doc_success += 1

        logger.debug(
            "  %s (%s): %d chunks, ~%d tokens",
            doc_id,
            doc_type,
            len(chunks),
            chunk_tokens,
        )

    # One output file. build_vectorstore.py indexes every JSON file in
    # CHUNKS_DIR, so writing per-type copies here would index chunks twice.
    if all_chunks:
        _save_chunks(all_chunks, CHUNKS_DIR / "all_chunks.json")

    # Summary stats
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
