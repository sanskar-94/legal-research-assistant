"""Convert HuggingFace JSON datasets into the format expected by the chunker.

The chunker expects documents in data/processed/ with fields:
  - doc_id: str
  - doc_type: "judgement" | "statute" | "constitution"
  - text: str
  - (optional metadata fields)

This script reads the downloaded HF datasets from data/raw/ and
converts them into that format.

Usage:
    python scripts/data/prepare_hf_data.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import PROCESSED_DIR, RAW_DIR, CHUNKS_DIR  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("prepare_hf_data")


def _load_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_json(data: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    logger.info("  Saved %d records -> %s", len(data), path.name)


def process_indian_legal(raw_dir: Path) -> list[dict]:
    """Convert ninadn/indian-legal (full judgments with summaries)."""
    docs = []
    for split in ("train", "test"):
        path = raw_dir / f"indian_legal_{split}.json"
        if not path.exists():
            continue
        records = _load_json(path)
        for i, rec in enumerate(records):
            text = rec.get("Text", "")
            if not text or len(text) < 200:
                continue
            docs.append({
                "doc_id": f"indian_legal_{split}_{i}",
                "doc_type": "judgement",
                "text": text,
                "case_name": "",
                "court": "Supreme Court of India",
            })
    logger.info("indian-legal: %d judgement documents", len(docs))
    return docs


def process_sc_judgements_chunked(raw_dir: Path) -> list[dict]:
    """Convert vihaannnn/Indian-Supreme-Court-Judgements-Chunked (pre-chunked).

    These are already chunked - save directly as chunks.
    """
    path = raw_dir / "sc_judgements_chunked_train.json"
    if not path.exists():
        return []
    records = _load_json(path)
    chunks = []
    for i, rec in enumerate(records):
        text = rec.get("text", "")
        if not text or len(text) < 50:
            continue
        chunks.append({
            "chunk_id": f"sc_chunk_{i}",
            "text": text,
            "metadata": {
                "doc_id": f"sc_chunk_{i}",
                "doc_type": "judgement",
                "court": "Supreme Court of India",
                "case_name": "",
                "citation": "",
                "date": "",
                "judges": [],
                "section_refs": [],
                "page_numbers": [],
                "paragraph_numbers": [],
                "chunk_index": i,
                "authority_score": 0.8,
                "act_name": "",
                "section_number": "",
            },
        })
    logger.info("sc-judgements-chunked: %d pre-chunked entries", len(chunks))
    return chunks


def process_legal_dataset_india(raw_dir: Path) -> list[dict]:
    """Convert ShreyasP123/Legal-Dataset-for-india (already chunked texts)."""
    path = raw_dir / "legal_dataset_india_train.json"
    if not path.exists():
        return []
    records = _load_json(path)
    chunks = []
    for i, rec in enumerate(records):
        text = rec.get("text", "")
        if not text or len(text) < 50:
            continue
        # Try to classify doc_type from content
        text_lower = text.lower()
        if any(kw in text_lower for kw in ("article", "constitution", "fundamental right")):
            doc_type = "constitution"
        elif any(kw in text_lower for kw in ("section", "ipc", "bns", "crpc", "penal", "code")):
            doc_type = "statute"
        else:
            doc_type = "judgement"

        chunks.append({
            "chunk_id": rec.get("chunk_id", f"legal_india_{i}"),
            "text": text,
            "metadata": {
                "doc_id": rec.get("chunk_id", f"legal_india_{i}"),
                "doc_type": doc_type,
                "court": "",
                "case_name": "",
                "citation": "",
                "date": "",
                "judges": [],
                "section_refs": [],
                "page_numbers": [],
                "paragraph_numbers": [],
                "chunk_index": i,
                "authority_score": 0.5,
                "act_name": "",
                "section_number": "",
            },
        })
    logger.info("legal-dataset-india: %d pre-chunked entries", len(chunks))
    return chunks


def process_indian_law_dataset(raw_dir: Path) -> list[dict]:
    """Convert viber1/indian-law-dataset (QA pairs) into documents.

    Each instruction+response pair becomes a document about a legal topic.
    """
    path = raw_dir / "indian_law_dataset_train.json"
    if not path.exists():
        return []
    records = _load_json(path)
    docs = []
    for i, rec in enumerate(records):
        instruction = rec.get("Instruction", "")
        response = rec.get("Response", "")
        if not response or len(response) < 100:
            continue

        # Combine Q+A as a document with the question as context
        text = f"Legal Question: {instruction}\n\nLegal Answer: {response}"

        # Classify based on content
        combined_lower = (instruction + response).lower()
        if any(kw in combined_lower for kw in ("article", "constitution", "fundamental", "directive principle")):
            doc_type = "constitution"
        elif any(kw in combined_lower for kw in ("section", "ipc", "bns", "crpc", "penal code", "criminal procedure")):
            doc_type = "statute"
        else:
            doc_type = "judgement"

        docs.append({
            "doc_id": f"indian_law_qa_{i}",
            "doc_type": doc_type,
            "text": text,
        })
    logger.info("indian-law-dataset: %d QA documents", len(docs))
    return docs


def process_constitution_dataset(raw_dir: Path) -> list[dict]:
    """Convert nisaar/Constitution_of_India (QA pairs about Constitution)."""
    path = raw_dir / "constitution_of_india_train.json"
    if not path.exists():
        return []
    records = _load_json(path)
    docs = []
    for i, rec in enumerate(records):
        question = rec.get("question", "")
        answer = rec.get("answer", "")
        if not answer or len(answer) < 50:
            continue

        text = f"Constitutional Question: {question}\n\nAnswer: {answer}"
        docs.append({
            "doc_id": f"constitution_qa_{i}",
            "doc_type": "constitution",
            "text": text,
        })
    logger.info("constitution-of-india: %d QA documents", len(docs))
    return docs


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    # Documents that need chunking (save to data/processed/)
    all_docs: list[dict] = []
    all_docs.extend(process_indian_legal(RAW_DIR))
    all_docs.extend(process_indian_law_dataset(RAW_DIR))
    all_docs.extend(process_constitution_dataset(RAW_DIR))

    if all_docs:
        _save_json(all_docs, PROCESSED_DIR / "hf_documents.json")

    # Pre-chunked data (save directly to data/processed/chunks/)
    all_prechunks: list[dict] = []
    all_prechunks.extend(process_sc_judgements_chunked(RAW_DIR))
    all_prechunks.extend(process_legal_dataset_india(RAW_DIR))

    if all_prechunks:
        _save_json(all_prechunks, CHUNKS_DIR / "prechunked_hf.json")

    logger.info("=" * 60)
    logger.info("Preparation complete.")
    logger.info("  Documents for chunking: %d -> %s", len(all_docs), PROCESSED_DIR)
    logger.info("  Pre-chunked entries:    %d -> %s", len(all_prechunks), CHUNKS_DIR)
    logger.info("=" * 60)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. python scripts/data/chunk_documents.py   (chunk the documents)")
    logger.info("  2. python scripts/data/build_vectorstore.py (embed + index)")


if __name__ == "__main__":
    main()
