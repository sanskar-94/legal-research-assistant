"""End-to-end retrieval pipeline: hybrid search followed by authority re-ranking."""

from __future__ import annotations

import logging
import re
from typing import Any

from config.settings import RERANKED_TOP_K, SEMANTIC_TOP_K
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.reranker import AuthorityReranker

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """Orchestrates hybrid search and authority-based re-ranking.

    Usage::

        pipeline = RetrievalPipeline(vectorstore, embedding_service)
        results  = pipeline.retrieve("What is Section 302 IPC?")

    Each result dict contains:
        ``chunk_id``, ``text``, ``metadata``, ``semantic_score``,
        ``rrf_score``, ``authority_score``, ``final_score``
    """

    def __init__(self, vectorstore: Any, embedding_service: Any) -> None:
        """Create the pipeline.

        Parameters
        ----------
        vectorstore:
            A ChromaDB collection (or compatible wrapper) used by
            :class:`HybridSearch`.
        embedding_service:
            An object exposing ``embed_query(text) -> list[float]``,
            used by :class:`HybridSearch`.
        """
        self._hybrid_search = HybridSearch(vectorstore, embedding_service)
        self._reranker = AuthorityReranker()

    def retrieve(
        self,
        query: str,
        *,
        search_top_k: int = SEMANTIC_TOP_K,
        rerank_top_k: int = RERANKED_TOP_K,
    ) -> list[dict[str, Any]]:
        """Run hybrid search followed by authority re-ranking.

        Parameters
        ----------
        query:
            Natural-language legal research query.
        search_top_k:
            Maximum number of candidates from each search path before
            fusion (default from ``config.settings.SEMANTIC_TOP_K``).
        rerank_top_k:
            Number of results to return after re-ranking (default from
            ``config.settings.RERANKED_TOP_K``).

        Returns
        -------
        list[dict]
            Up to *rerank_top_k* result dicts, each containing:

            - ``chunk_id`` -- unique identifier of the text chunk
            - ``text`` -- the chunk content
            - ``metadata`` -- original metadata dict
            - ``semantic_score`` -- similarity from embedding search
            - ``rrf_score`` -- reciprocal rank fusion score
            - ``authority_score`` -- court / recency / citation authority
            - ``final_score`` -- weighted combination of the above
        """
        if not query or not query.strip():
            logger.warning("Retrieval called with empty query")
            return []

        logger.info("Retrieval pipeline starting for query: %.80s...", query)

        candidates = self._hybrid_search.search(query, top_k=search_top_k)

        if not candidates:
            logger.warning("Hybrid search returned zero candidates")
            return []

        candidates = _drop_duplicate_texts(candidates)

        ranked = self._reranker.rerank(candidates, top_k=rerank_top_k)

        # Guarantee every result has the full set of expected keys
        normalised: list[dict[str, Any]] = []
        for result in ranked:
            normalised.append({
                "chunk_id": result.get("chunk_id", ""),
                "text": result.get("text", ""),
                "metadata": result.get("metadata", {}),
                "semantic_score": result.get("semantic_score", 0.0),
                "rrf_score": result.get("rrf_score", 0.0),
                "authority_score": result.get("authority_score", 0.0),
                "final_score": result.get("final_score", 0.0),
            })

        logger.info(
            "Retrieval pipeline complete - %d results returned", len(normalised),
        )
        return normalised


def _drop_duplicate_texts(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the first of any candidates whose text is identical.

    Some source datasets contain the same record several times; without
    this the top results can be copies of one passage.
    """
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for candidate in candidates:
        key = re.sub(r"\s+", " ", candidate.get("text", "")).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    if len(unique) < len(candidates):
        logger.info("Dropped %d duplicate passages", len(candidates) - len(unique))
    return unique
