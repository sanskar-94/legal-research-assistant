"""ChromaDB collection management for legal document embeddings."""

from __future__ import annotations

import logging
from typing import Any, Optional

from config.settings import CHROMA_COLLECTION_NAME, CHROMA_DIR
from src.chunking.chunk_models import Chunk

logger = logging.getLogger(__name__)

# Metadata fields that are stored in ChromaDB for filtering.
_METADATA_FIELDS: tuple[str, ...] = (
    "doc_id",
    "chunk_index",
    "doc_type",
    "court",
    "date",
    "act_name",
    "section_number",
    "case_name",
    "citation",
    "authority_score",
)

_COLLECTION_METADATA = {"hnsw:space": "cosine"}


def _sanitise_metadata_value(value: Any) -> str | int | float | bool:
    """Convert a value to a ChromaDB-compatible scalar.

    ChromaDB metadata values must be str, int, float, or bool.
    - ``None`` is replaced with an empty string.
    - Lists are serialised to a comma-separated string.
    - Anything else is converted with ``str``.
    """
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def _build_chroma_metadata(chunk: Chunk) -> dict[str, str | int | float | bool]:
    """Extract and sanitise the stored metadata fields of *chunk*."""
    return {
        field_name: _sanitise_metadata_value(getattr(chunk.metadata, field_name, None))
        for field_name in _METADATA_FIELDS
    }


class VectorStore:
    """Manages the persistent ChromaDB collection that holds the corpus."""

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        import chromadb

        resolved_dir = persist_dir or str(CHROMA_DIR)
        self._collection_name = collection_name or CHROMA_COLLECTION_NAME

        logger.info("Opening ChromaDB at '%s' ...", resolved_dir)
        self._client = chromadb.PersistentClient(path=resolved_dir)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata=_COLLECTION_METADATA,
        )
        logger.info(
            "ChromaDB collection '%s' ready (%d existing chunks).",
            self._collection_name,
            self._collection.count(),
        )

    @property
    def collection(self) -> Any:
        """The underlying ChromaDB collection, as used by ``HybridSearch``."""
        return self._collection

    def count(self) -> int:
        """Return the number of chunks in the collection."""
        return self._collection.count()

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Upsert chunks with their embeddings and metadata.

        *embeddings* must be aligned 1-to-1 with *chunks*.
        """
        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings."
            )

        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk.text for chunk in chunks],
            metadatas=[_build_chroma_metadata(chunk) for chunk in chunks],
        )
        logger.debug("Upserted %d chunks into the collection.", len(chunks))

    def reset(self) -> None:
        """Delete every chunk by dropping and re-creating the collection."""
        self._client.delete_collection(name=self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata=_COLLECTION_METADATA,
        )
        logger.info("Collection '%s' cleared.", self._collection_name)
