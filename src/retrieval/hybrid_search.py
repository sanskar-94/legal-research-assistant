"""Hybrid search combining semantic and keyword retrieval with query classification."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any

from config.settings import RRF_K, SEMANTIC_TOP_K

logger = logging.getLogger(__name__)

# ── Query classification patterns ─────────────────────────────────────
_SECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bSection\s+\d+", re.IGNORECASE),
    re.compile(r"\bArticle\s+\d+", re.IGNORECASE),
    re.compile(r"\bAct[,\s]", re.IGNORECASE),
    re.compile(r"\bCode\s+of\b", re.IGNORECASE),
    re.compile(r"\bIPC\b"),
    re.compile(r"\bCrPC\b"),
    re.compile(r"\bCPC\b"),
)

_CASE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bv[.\s]\s", re.IGNORECASE),
    re.compile(r"\bvs\.?\s", re.IGNORECASE),
    re.compile(r"\bAIR\s+\d{4}\b"),
    re.compile(r"\bSCC\s+\d+\b", re.IGNORECASE),
    re.compile(r"\(\d{4}\)\s+\d+\s+SCC\b"),
)

# Weight presets per classification  (semantic_weight, keyword_weight)
_CLASSIFICATION_WEIGHTS: dict[str, tuple[float, float]] = {
    "section_lookup": (0.3, 0.7),
    "case_lookup": (0.4, 0.6),
    "conceptual": (0.8, 0.2),
    "mixed": (0.5, 0.5),
}


def _extract_keyword_terms(query: str) -> list[str]:
    """Pull out specific legal identifiers for keyword matching.

    Returns a list of distinct search phrases extracted from the query,
    such as section numbers, article numbers, act names, and case
    citation fragments.
    """
    terms: list[str] = []

    # Section / Article numbers  (e.g. "Section 302", "Article 21")
    for match in re.finditer(r"(Section\s+\d+[A-Za-z]*)", query, re.IGNORECASE):
        terms.append(match.group(0))
    for match in re.finditer(r"(Article\s+\d+[A-Za-z]*)", query, re.IGNORECASE):
        terms.append(match.group(0))

    # Common act abbreviations
    for abbr in ("IPC", "CrPC", "CPC"):
        if re.search(rf"\b{abbr}\b", query):
            terms.append(abbr)

    # AIR / SCC citations
    for match in re.finditer(r"(AIR\s+\d{4}\s+\w+\s+\d+)", query):
        terms.append(match.group(0))
    for match in re.finditer(r"(\(\d{4}\)\s+\d+\s+SCC\s+\d+)", query):
        terms.append(match.group(0))
    for match in re.finditer(r"(SCC\s+\d+)", query, re.IGNORECASE):
        terms.append(match.group(0))

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for term in terms:
        normalised = term.strip().lower()
        if normalised not in seen:
            seen.add(normalised)
            unique.append(term.strip())
    return unique


def _reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = RRF_K,
) -> dict[str, float]:
    """Compute RRF scores from multiple ranked result lists.

    Each list contains chunk_ids ordered by descending relevance.
    Returns a mapping of chunk_id -> rrf_score.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] += 1.0 / (k + rank)
    return dict(scores)


class HybridSearch:
    """Dual-path retrieval combining semantic and keyword search with
    query-adaptive weighting and Reciprocal Rank Fusion."""

    def __init__(self, vectorstore: Any, embedding_service: Any) -> None:
        """Initialise hybrid search.

        Parameters
        ----------
        vectorstore:
            A ChromaDB collection (or compatible wrapper) exposing
            ``query`` and ``get`` methods.
        embedding_service:
            An object with an ``embed_query(text) -> list[float]`` method.
        """
        self._vectorstore = vectorstore
        self._embedding_service = embedding_service

    # ── Query classification ──────────────────────────────────────────

    @staticmethod
    def classify_query(query: str) -> str:
        """Classify *query* into a retrieval strategy bucket.

        Returns one of:
            ``"section_lookup"`` | ``"case_lookup"`` |
            ``"conceptual"``     | ``"mixed"``
        """
        has_section = any(p.search(query) for p in _SECTION_PATTERNS)
        has_case = any(p.search(query) for p in _CASE_PATTERNS)

        if has_section and has_case:
            return "mixed"
        if has_section:
            return "section_lookup"
        if has_case:
            return "case_lookup"
        return "conceptual"

    # ── Semantic search ───────────────────────────────────────────────

    def _semantic_search(
        self,
        query: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Embed *query* and retrieve nearest neighbours from ChromaDB."""
        query_embedding: list[float] = self._embedding_service.embed_query(query)
        results = self._vectorstore.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        hits: list[dict[str, Any]] = []
        if not results or not results.get("ids"):
            return hits

        ids = results["ids"][0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for idx, chunk_id in enumerate(ids):
            # ChromaDB L2 distance -> similarity (lower distance = better)
            distance = distances[idx] if idx < len(distances) else 0.0
            semantic_score = 1.0 / (1.0 + distance)

            hits.append({
                "chunk_id": chunk_id,
                "text": documents[idx] if idx < len(documents) else "",
                "metadata": metadatas[idx] if idx < len(metadatas) else {},
                "semantic_score": semantic_score,
            })
        return hits

    # ── Keyword search ────────────────────────────────────────────────

    def _keyword_search(
        self,
        query: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Run keyword-based retrieval via ChromaDB ``where_document``."""
        terms = _extract_keyword_terms(query)
        if not terms:
            # Fall back to the full query as a single keyword phrase
            terms = [query]

        all_hits: dict[str, dict[str, Any]] = {}
        for term in terms:
            try:
                results = self._vectorstore.get(
                    where_document={"$contains": term},
                    include=["documents", "metadatas"],
                    limit=top_k,
                )
            except Exception:
                logger.debug("Keyword search failed for term '%s'", term, exc_info=True)
                continue

            if not results or not results.get("ids"):
                continue

            for idx, chunk_id in enumerate(results["ids"]):
                if chunk_id in all_hits:
                    continue
                all_hits[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": results["documents"][idx] if idx < len(results.get("documents", [])) else "",
                    "metadata": results["metadatas"][idx] if idx < len(results.get("metadatas", [])) else {},
                    "semantic_score": 0.0,
                }

        return list(all_hits.values())[:top_k]

    # ── Public search method ──────────────────────────────────────────

    def search(self, query: str, top_k: int = SEMANTIC_TOP_K) -> list[dict[str, Any]]:
        """Execute hybrid search and return merged results sorted by RRF score.

        Parameters
        ----------
        query:
            Natural-language legal query.
        top_k:
            Maximum number of candidates to retrieve from *each* path
            before fusion.

        Returns
        -------
        list[dict]
            Merged result dicts containing ``chunk_id``, ``text``,
            ``metadata``, ``semantic_score``, and ``rrf_score``, sorted
            descending by weighted RRF score.
        """
        classification = self.classify_query(query)
        sem_weight, kw_weight = _CLASSIFICATION_WEIGHTS[classification]
        logger.info(
            "Query classified as '%s' — semantic=%.1f, keyword=%.1f",
            classification, sem_weight, kw_weight,
        )

        semantic_hits = self._semantic_search(query, top_k)
        keyword_hits = self._keyword_search(query, top_k)

        logger.debug(
            "Semantic hits: %d, keyword hits: %d",
            len(semantic_hits), len(keyword_hits),
        )

        # Build ranked id lists for RRF
        semantic_ranked = [h["chunk_id"] for h in semantic_hits]
        keyword_ranked = [h["chunk_id"] for h in keyword_hits]

        # Weighted RRF: duplicate each list proportionally
        # We simulate weighting by scaling the RRF contribution
        raw_rrf = _reciprocal_rank_fusion([semantic_ranked, keyword_ranked])

        # Collect every hit by chunk_id (prefer semantic result for score)
        hits_by_id: dict[str, dict[str, Any]] = {}
        for hit in semantic_hits:
            hits_by_id[hit["chunk_id"]] = {**hit}
        for hit in keyword_hits:
            if hit["chunk_id"] not in hits_by_id:
                hits_by_id[hit["chunk_id"]] = {**hit}

        # Separate RRF contributions from each ranked list
        semantic_rrf: dict[str, float] = defaultdict(float)
        for rank, cid in enumerate(semantic_ranked, start=1):
            semantic_rrf[cid] = 1.0 / (RRF_K + rank)

        keyword_rrf: dict[str, float] = defaultdict(float)
        for rank, cid in enumerate(keyword_ranked, start=1):
            keyword_rrf[cid] = 1.0 / (RRF_K + rank)

        # Compute weighted RRF score per chunk
        merged: list[dict[str, Any]] = []
        for chunk_id, hit in hits_by_id.items():
            rrf_score = (
                sem_weight * semantic_rrf.get(chunk_id, 0.0)
                + kw_weight * keyword_rrf.get(chunk_id, 0.0)
            )
            merged.append({
                **hit,
                "rrf_score": rrf_score,
            })

        merged.sort(key=lambda r: r["rrf_score"], reverse=True)

        logger.info("Hybrid search returned %d merged candidates", len(merged))
        return merged
