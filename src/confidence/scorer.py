"""Multi-signal confidence scoring for legal research responses.

Computes an overall confidence score from four independent signals:
retrieval quality, source agreement, authority strength, and coverage.
Each signal is weighted according to CONFIDENCE_WEIGHTS in config.settings.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from config.settings import CONFIDENCE_WEIGHTS

logger = logging.getLogger(__name__)

# ── Confidence-level thresholds ──────────────────────────────────────
_CONFIDENCE_LEVELS: tuple[tuple[float, str, str], ...] = (
    (0.80, "High", "green"),
    (0.60, "Medium", "yellow"),
    (0.40, "Low", "orange"),
)
_DEFAULT_LEVEL: tuple[str, str] = ("Very Low", "red")

# ── Retrieval-quality mapping constants ──────────────────────────────
_SIMILARITY_FLOOR = 0.3
_SIMILARITY_CEILING = 0.9


class ConfidenceResult(BaseModel):
    """Immutable result of a confidence scoring pass."""

    overall_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Weighted confidence score in [0, 1].",
    )
    level: str = Field(
        description='Human-readable level: "High", "Medium", "Low", or "Very Low".',
    )
    color: str = Field(
        description='Display colour tag: "green", "yellow", "orange", or "red".',
    )
    breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Per-signal scores keyed by signal name.",
    )

    model_config = {"frozen": True}


class ConfidenceScorer:
    """Calculate a multi-signal confidence score for a generated answer.

    Signals
    -------
    1. **retrieval_quality** (weight 0.40) — average cosine similarity of
       the top retrieved chunks, linearly mapped from [0.3, 0.9] to [0, 1].
    2. **source_agreement** (weight 0.25) — how many unique sources the
       answer cites.  1 → 0.3, 2 → 0.6, 3+ → 1.0.
    3. **authority_strength** (weight 0.20) — mean ``authority_score`` of
       cited chunks (already in [0, 1]).
    4. **coverage_score** (weight 0.15) — proportion of provided chunks
       actually cited in the response.
    """

    # ── Public API ────────────────────────────────────────────────────

    def score(
        self,
        retrieved_chunks: list[dict[str, Any]],
        cited_response: Any,
    ) -> ConfidenceResult:
        """Compute a confidence result from retrieved chunks and the parsed response.

        Parameters
        ----------
        retrieved_chunks:
            Chunk dicts returned by the retrieval pipeline.  Each dict
            should contain at least ``semantic_score`` and ``metadata``
            (which itself should carry ``authority_score`` and ``doc_id``).
        cited_response:
            The parsed response object.  Expected to expose a
            ``cited_sources`` attribute (list) where each source has a
            ``doc_id`` attribute (or is a dict with a ``"doc_id"`` key).

        Returns
        -------
        ConfidenceResult
            An immutable result with the overall score, level, colour,
            and per-signal breakdown.
        """
        retrieval_quality = self._compute_retrieval_quality(retrieved_chunks)
        source_agreement = self._compute_source_agreement(cited_response)
        authority_strength = self._compute_authority_strength(
            retrieved_chunks, cited_response,
        )
        coverage = self._compute_coverage(retrieved_chunks, cited_response)

        breakdown: dict[str, float] = {
            "retrieval_quality": round(retrieval_quality, 4),
            "source_agreement": round(source_agreement, 4),
            "authority_strength": round(authority_strength, 4),
            "coverage_score": round(coverage, 4),
        }

        weights = CONFIDENCE_WEIGHTS
        overall = (
            weights["retrieval_quality"] * retrieval_quality
            + weights["source_agreement"] * source_agreement
            + weights["authority_strength"] * authority_strength
            + weights["coverage_score"] * coverage
        )
        overall = round(min(max(overall, 0.0), 1.0), 4)

        level, color = self.get_confidence_level(overall)

        logger.info(
            "Confidence scored: %.2f (%s) — breakdown=%s",
            overall,
            level,
            breakdown,
        )

        return ConfidenceResult(
            overall_score=overall,
            level=level,
            color=color,
            breakdown=breakdown,
        )

    @staticmethod
    def get_confidence_level(score: float) -> tuple[str, str]:
        """Map a 0-1 score to a human-readable level and display colour.

        Returns
        -------
        tuple[str, str]
            ``(level, colour)`` where level is one of
            ``"High"`` / ``"Medium"`` / ``"Low"`` / ``"Very Low"``
            and colour is one of
            ``"green"`` / ``"yellow"`` / ``"orange"`` / ``"red"``.
        """
        for threshold, level, color in _CONFIDENCE_LEVELS:
            if score >= threshold:
                return level, color
        return _DEFAULT_LEVEL

    # ── Private signal computations ───────────────────────────────────

    @staticmethod
    def _compute_retrieval_quality(chunks: list[dict[str, Any]]) -> float:
        """Average quality score of retrieved chunks mapped to [0, 1].

        Uses ``final_score`` (composite of semantic + RRF + authority)
        when available, falling back to ``semantic_score``.  Values are
        linearly scaled from [``_SIMILARITY_FLOOR``, ``_SIMILARITY_CEILING``]
        to [0, 1] and clamped.
        """
        if not chunks:
            return 0.0

        scores = [
            float(
                chunk.get("final_score", 0.0)
                or chunk.get("semantic_score", 0.0)
            )
            for chunk in chunks
        ]
        avg_score = sum(scores) / len(scores)

        # Linear mapping from [FLOOR, CEILING] → [0, 1]
        span = _SIMILARITY_CEILING - _SIMILARITY_FLOOR
        if span <= 0:
            return 0.0

        mapped = (avg_score - _SIMILARITY_FLOOR) / span
        return min(max(mapped, 0.0), 1.0)

    @staticmethod
    def _compute_source_agreement(cited_response: Any) -> float:
        """Score based on the number of unique sources cited.

        1 source → 0.3, 2 sources → 0.6, 3+ sources → 1.0, 0 → 0.0.
        """
        cited_sources = _extract_cited_sources(cited_response)
        unique_doc_ids: set[str] = set()
        for source in cited_sources:
            doc_id = _get_doc_id(source)
            if doc_id:
                unique_doc_ids.add(doc_id)

        count = len(unique_doc_ids)
        if count == 0:
            return 0.0
        if count == 1:
            return 0.3
        if count == 2:
            return 0.6
        return 1.0

    @staticmethod
    def _compute_authority_strength(
        chunks: list[dict[str, Any]],
        cited_response: Any,
    ) -> float:
        """Average ``authority_score`` of chunks that were actually cited."""
        cited_sources = _extract_cited_sources(cited_response)
        cited_doc_ids: set[str] = set()
        for source in cited_sources:
            doc_id = _get_doc_id(source)
            if doc_id:
                cited_doc_ids.add(doc_id)

        if not cited_doc_ids:
            # Fallback: use all retrieved chunks if nothing was cited
            authority_scores = [
                float(
                    chunk.get("authority_score", 0.0)
                    or chunk.get("metadata", {}).get("authority_score", 0.0)
                )
                for chunk in chunks
            ]
        else:
            authority_scores = [
                float(
                    chunk.get("authority_score", 0.0)
                    or chunk.get("metadata", {}).get("authority_score", 0.0)
                )
                for chunk in chunks
                if _chunk_matches_ids(chunk, cited_doc_ids)
            ]

        if not authority_scores:
            return 0.0

        return sum(authority_scores) / len(authority_scores)

    @staticmethod
    def _compute_coverage(
        chunks: list[dict[str, Any]],
        cited_response: Any,
    ) -> float:
        """Fraction of provided chunks that appear in the cited sources."""
        if not chunks:
            return 0.0

        cited_sources = _extract_cited_sources(cited_response)
        cited_ids: set[str] = set()
        for source in cited_sources:
            doc_id = _get_doc_id(source)
            if doc_id:
                cited_ids.add(doc_id)

        if not cited_ids:
            return 0.0

        cited_count = sum(
            1
            for chunk in chunks
            if _chunk_matches_ids(chunk, cited_ids)
        )

        return cited_count / len(chunks)


# ── Helpers ──────────────────────────────────────────────────────────

def _chunk_matches_ids(chunk: dict, ids: set[str]) -> bool:
    """Return True if any identifier of *chunk* appears in *ids*."""
    return (
        chunk.get("chunk_id", "") in ids
        or chunk.get("metadata", {}).get("doc_id", "") in ids
        or chunk.get("metadata", {}).get("chunk_id", "") in ids
    )


def _extract_cited_sources(cited_response: Any) -> list[Any]:
    """Safely pull the list of cited sources from a response object or dict."""
    if cited_response is None:
        return []
    if isinstance(cited_response, dict):
        return cited_response.get("cited_sources", [])
    return getattr(cited_response, "cited_sources", [])


def _get_doc_id(source: Any) -> str | None:
    """Extract an identifier from a source that may be a dict or object.

    Checks ``chunk_id`` first (used by CitedSource), then ``doc_id``,
    then looks inside nested ``metadata``.
    """
    if isinstance(source, dict):
        return (
            source.get("chunk_id")
            or source.get("doc_id")
            or source.get("metadata", {}).get("doc_id")
        )
    return (
        getattr(source, "chunk_id", None)
        or getattr(source, "doc_id", None)
    )
