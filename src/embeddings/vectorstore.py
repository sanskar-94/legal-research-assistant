"""ChromaDB collection management for legal document embeddings."""

from __future__ import annotations

import logging
from typing import Any, Optional

from config.settings import CHROMA_COLLECTION_NAME, CHROMA_DIR

logger = logging.getLogger(__name__)

# Metadata fields that are stored in ChromaDB for filtering.
_METADATA_FIELDS: tuple[str, ...] = (
    "doc_type",
    "court",
    "date",
    "act_name",
    "section_number",
    "case_name",
    "citation",
    "authority_score",
)


def _sanitise_metadata_value(value: Any) -> str | int | float | bool:
    """Convert a value to a ChromaDB-compatible scalar.

    ChromaDB metadata values must be str, int, float, or bool.
    - ``None`` is replaced with an empty string.
    - Lists/dicts are serialised to a comma-separated string.
    - Everything else is left as-is if already a valid type.
    """
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def _build_chroma_metadata(chunk) -> dict[str, str | int | float | bool]:
    """Extract and sanitise metadata from a Chunk's ChunkMetadata model.

    Returns a *new* dict — the original model is never mutated.
    """
    raw_meta = chunk.metadata
    sanitised: dict[str, str | int | float | bool] = {}
    for field_name in _METADATA_FIELDS:
        raw_value = getattr(raw_meta, field_name, None)
        sanitised[field_name] = _sanitise_metadata_value(raw_value)
    return sanitised


class VectorStore:
    """Manages a ChromaDB collection for legal document embeddings.

    Provides semantic and keyword search with optional metadata filtering.
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        resolved_dir = persist_dir or str(CHROMA_DIR)
        resolved_name = collection_name or CHROMA_COLLECTION_NAME

        try:
            import chromadb

            logger.info(
                "Initialising ChromaDB PersistentClient at '%s' ...", resolved_dir
            )
            self._client = chromadb.PersistentClient(path=resolved_dir)
            self._collection = self._client.get_or_create_collection(
                name=resolved_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "ChromaDB collection '%s' ready (%d existing documents).",
                resolved_name,
                self._collection.count(),
            )
        except ImportError as exc:
            logger.error(
                "chromadb is not installed. Install it with: pip install chromadb"
            )
            raise ImportError(
                "chromadb package is required for VectorStore"
            ) from exc
        except Exception as exc:
            logger.error("Failed to initialise ChromaDB: %s", exc)
            raise RuntimeError(
                f"Could not initialise ChromaDB at '{resolved_dir}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: list,
        embeddings: list[list[float]],
    ) -> None:
        """Upsert chunks with their embeddings and metadata.

        Parameters
        ----------
        chunks:
            Chunk pydantic model instances with ``.chunk_id``, ``.text``,
            and ``.metadata`` attributes.
        embeddings:
            Pre-computed embedding vectors aligned 1-to-1 with *chunks*.
        """
        if not chunks:
            logger.warning("add_chunks called with an empty list — nothing to upsert.")
            return

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings."
            )

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [_build_chroma_metadata(chunk) for chunk in chunks]

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("Upserted %d chunks into the collection.", len(chunks))

    # ------------------------------------------------------------------
    # Read / search operations
    # ------------------------------------------------------------------

    def search_semantic(
        self,
        query_embedding: list[float],
        top_k: int = 20,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """Semantic (vector similarity) search.

        Parameters
        ----------
        query_embedding:
            The embedding vector of the user query.
        top_k:
            Maximum number of results.
        filters:
            Optional ChromaDB ``where`` clause dict for metadata filtering.

        Returns
        -------
        list[dict]
            Each dict contains ``id``, ``text``, ``metadata``, and ``score``
            (cosine distance — lower is more similar).
        """
        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
        }
        if filters:
            query_kwargs["where"] = filters

        results = self._collection.query(**query_kwargs)
        return self._format_results(results)

    def search_keyword(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """Keyword search using ChromaDB ``where_document`` contains filter.

        Parameters
        ----------
        query:
            Raw keyword/phrase to search for inside document text.
        top_k:
            Maximum number of results.
        filters:
            Optional ChromaDB ``where`` clause dict for metadata filtering.

        Returns
        -------
        list[dict]
            Each dict contains ``id``, ``text``, ``metadata``, and ``score``.
        """
        query_kwargs: dict[str, Any] = {
            "where_document": {"$contains": query},
            "n_results": top_k,
        }
        if filters:
            query_kwargs["where"] = filters

        results = self._collection.get(
            where_document={"$contains": query},
            where=filters if filters else None,
            limit=top_k,
            include=["documents", "metadatas"],
        )
        return self._format_get_results(results)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self._collection.count()

    def get_collection_stats(self) -> dict:
        """Return collection count and a sample of metadata entries.

        Returns
        -------
        dict
            ``count`` — total documents, ``sample_metadata`` — up to 5
            metadata dicts for quick inspection.
        """
        count = self._collection.count()
        sample_metadata: list[dict] = []

        if count > 0:
            peek = self._collection.peek(limit=min(count, 5))
            sample_metadata = peek.get("metadatas", [])

        return {
            "count": count,
            "sample_metadata": sample_metadata,
        }

    def delete_collection(self) -> None:
        """Delete all documents from the collection for a full rebuild."""
        collection_name = self._collection.name
        self._client.delete_collection(name=collection_name)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection '%s' deleted and re-created.", collection_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_results(results: dict) -> list[dict]:
        """Convert a ChromaDB ``query`` response into a flat list of dicts."""
        formatted: list[dict] = []
        if not results or not results.get("ids"):
            return formatted

        ids = results["ids"][0]
        documents = results["documents"][0] if results.get("documents") else [""] * len(ids)
        metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)
        distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)

        for doc_id, text, meta, dist in zip(ids, documents, metadatas, distances):
            formatted.append(
                {
                    "id": doc_id,
                    "text": text,
                    "metadata": dict(meta) if meta else {},
                    "score": dist,
                }
            )
        return formatted

    @staticmethod
    def _format_get_results(results: dict) -> list[dict]:
        """Convert a ChromaDB ``get`` response into a flat list of dicts."""
        formatted: list[dict] = []
        if not results or not results.get("ids"):
            return formatted

        ids = results["ids"]
        documents = results.get("documents") or [""] * len(ids)
        metadatas = results.get("metadatas") or [{}] * len(ids)

        for doc_id, text, meta in zip(ids, documents, metadatas):
            formatted.append(
                {
                    "id": doc_id,
                    "text": text,
                    "metadata": dict(meta) if meta else {},
                    "score": 0.0,
                }
            )
        return formatted
