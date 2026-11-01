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
from src.generation.response_parser import CitedResponse

logger = logging.getLogger(__name__)

# Confidence-level thresholds
_CONFIDENCE_LEVELS: tuple[tuple[float, str], ...] = (
    (0.80, "High"),
    (0.60, "Medium"),
    (0.40, "Low"),
)
_DEFAULT_LEVEL = "Very Low"

# Retrieval-quality mapping constants
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
    breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Per-signal scores keyed by signal name.",
    )

    model_config = {"frozen": True}


def confidence_level(score: float) -> str:
    """Map a 0-1 score to ``"High"``, ``"Medium"``, ``"Low"`` or ``"Very Low"``."""
    for threshold, level in _CONFIDENCE_LEVELS:
        if score >= threshold:
            return level
    return _DEFAULT_LEVEL


class ConfidenceScorer:
    """Calculate a multi-signal confidence score for a generated answer.

    Signals
    -------
    1. **retrieval_quality** (weight 0.40) - average ``final_score`` of
       the retrieved chunks, linearly mapped from [0.3, 0.9] to [0, 1].
    2. **source_agreement** (weight 0.25) - how many unique sources the
       answer cites.  1 -> 0.3, 2 -> 0.6, 3+ -> 1.0.
    3. **authority_strength** (weight 0.20) - mean ``authority_score`` of
       cited chunks (already in [0, 1]).
    4. **coverage_score** (weight 0.15) - proportion of provided chunks
       actually cited in the response.
    """

    def score(
        self,
        retrieved_chunks: list[dict[str, Any]],
        cited_response: CitedResponse,
    ) -> ConfidenceResult:
        """Compute a confidence result from retrieved chunks and the parsed response.

        Parameters
        ----------
        retrieved_chunks:
            Chunk dicts returned by the retrieval pipeline.  Each dict
            should contain ``chunk_id``, ``final_score`` (or
            ``semantic_score``) and ``authority_score``.
        cited_response:
            The parsed response from ``ResponseParser``.
        """
        cited_ids = {source.chunk_id for source in cited_response.cited_sources}

        retrieval_quality = _retrieval_quality(retrieved_chunks)
        source_agreement = _source_agreement(len(cited_ids))
        authority_strength = _authority_strength(retrieved_chunks, cited_ids)
        coverage = _coverage(retrieved_chunks, cited_ids)

        breakdown: dict[str, float] = {
            "retrieval_quality": round(retrieval_quality, 4),
            "source_agreement": round(source_agreement, 4),
            "authority_strength": round(authority_strength, 4),
            "coverage_score": round(coverage, 4),
        }

        overall = sum(CONFIDENCE_WEIGHTS[name] * value for name, value in breakdown.items())
        overall = round(min(max(overall, 0.0), 1.0), 4)
        level = confidence_level(overall)

        logger.info(
            "Confidence scored: %.2f (%s) - breakdown=%s",
            overall,
            level,
            breakdown,
        )

        return ConfidenceResult(
            overall_score=overall,
            level=level,
            breakdown=breakdown,
        )


# Signal computations


def _retrieval_quality(chunks: list[dict[str, Any]]) -> float:
    """Average quality score of retrieved chunks mapped to [0, 1].

    Uses ``final_score`` (composite of semantic + RRF + authority)
    when available, falling back to ``semantic_score``.  Values are
    linearly scaled from [``_SIMILARITY_FLOOR``, ``_SIMILARITY_CEILING``]
    to [0, 1] and clamped.
    """
    if not chunks:
        return 0.0

    scores = [
        float(chunk.get("final_score", 0.0) or chunk.get("semantic_score", 0.0))
        for chunk in chunks
    ]
    avg_score = sum(scores) / len(scores)
    mapped = (avg_score - _SIMILARITY_FLOOR) / (_SIMILARITY_CEILING - _SIMILARITY_FLOOR)
    return min(max(mapped, 0.0), 1.0)


def _source_agreement(cited_count: int) -> float:
    """Score based on the number of unique sources cited.

    1 source -> 0.3, 2 sources -> 0.6, 3+ sources -> 1.0, 0 -> 0.0.
    """
    if cited_count == 0:
        return 0.0
    if cited_count == 1:
        return 0.3
    if cited_count == 2:
        return 0.6
    return 1.0


def _authority(chunk: dict[str, Any]) -> float:
    return float(
        chunk.get("authority_score", 0.0)
        or chunk.get("metadata", {}).get("authority_score", 0.0)
    )


def _authority_strength(chunks: list[dict[str, Any]], cited_ids: set[str]) -> float:
    """Average ``authority_score`` of the cited chunks.

    Falls back to all retrieved chunks when nothing was cited.
    """
    relevant = [c for c in chunks if c.get("chunk_id") in cited_ids] if cited_ids else chunks
    if not relevant:
        return 0.0
    return sum(_authority(c) for c in relevant) / len(relevant)


def _coverage(chunks: list[dict[str, Any]], cited_ids: set[str]) -> float:
    """Fraction of provided chunks that appear in the cited sources."""
    if not chunks or not cited_ids:
        return 0.0
    return sum(1 for c in chunks if c.get("chunk_id") in cited_ids) / len(chunks)
