"""Authority-aware re-ranking for legal document retrieval results."""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Optional

from config.settings import (
    AUTHORITY_WEIGHT,
    COURT_WEIGHTS,
    RECENCY_BRACKETS,
    RECENCY_DEFAULT,
    RERANKED_TOP_K,
    RRF_K,
    RRF_WEIGHT,
    SEMANTIC_WEIGHT,
)

# Largest RRF score a chunk can get from one list (rank 1). Used to put
# the RRF score on the same 0-1 scale as the other two signals.
_MAX_SINGLE_LIST_RRF = 1.0 / (RRF_K + 1)

logger = logging.getLogger(__name__)


def _years_since(date_str: Optional[str]) -> Optional[int]:
    """Return the number of full years between *date_str* and today.

    Accepts ISO-style dates (``YYYY-MM-DD``, ``YYYY-MM``, ``YYYY``).
    Returns ``None`` when *date_str* is empty or unparseable.
    """
    if not date_str:
        return None

    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            parsed = datetime.strptime(date_str, fmt).date()
            today = date.today()
            # Full calendar years difference
            years = today.year - parsed.year
            if (today.month, today.day) < (parsed.month, parsed.day):
                years -= 1
            return max(years, 0)
        except ValueError:
            continue

    logger.debug("Could not parse date '%s'", date_str)
    return None


def _recency_weight(years: Optional[int]) -> float:
    """Map *years* since decision to a recency weight via configured brackets."""
    if years is None:
        return RECENCY_DEFAULT

    for threshold, weight in RECENCY_BRACKETS:
        if years <= threshold:
            return weight
    return RECENCY_DEFAULT


def _normalise_court(court_raw: Optional[str]) -> str:
    """Normalise a raw court string to a canonical key in ``COURT_WEIGHTS``.

    Handles common variations such as ``"Supreme Court of India"``,
    ``"Delhi High Court"``, ``"NCLAT"``, etc.
    """
    if not court_raw:
        return ""

    lower = court_raw.strip().lower()

    if "supreme" in lower:
        return "supreme_court"
    if "high court" in lower or "high_court" in lower:
        return "high_court"
    if any(t in lower for t in ("tribunal", "nclat", "nclt", "itat", "sat", "drt")):
        return "tribunal"
    if any(t in lower for t in ("statute", "act", "code", "legislation")):
        return "statute"

    return lower


class AuthorityReranker:
    """Re-ranks retrieval candidates by combining semantic similarity,
    RRF score, and a domain-specific authority score that accounts for
    court hierarchy and decision recency."""

    # Authority scoring

    @staticmethod
    def compute_authority_score(metadata: dict[str, Any]) -> float:
        """Derive an authority score in [0, 1] from chunk *metadata*.

        Scoring formula
        ---------------
        * **Statutes** always receive ``1.0``.
        * **Case law** is scored as::

              court_w * 0.4 + recency_w * 0.3 + citation_w * 0.3

        Where ``citation_w`` defaults to ``0.5`` when the citation count
        is unavailable.
        """
        doc_type = str(metadata.get("doc_type", "")).lower()

        if doc_type in ("statute", "constitution", "legislation"):
            return 1.0

        court_key = _normalise_court(metadata.get("court"))
        court_w = COURT_WEIGHTS.get(court_key, 0.3)

        years = _years_since(metadata.get("date"))
        recency_w = _recency_weight(years)

        raw_citations = metadata.get("citation_count")
        if raw_citations is not None:
            try:
                # Normalise to [0, 1] via a soft cap at 100 citations
                citation_w = min(float(raw_citations) / 100.0, 1.0)
            except (TypeError, ValueError):
                citation_w = 0.5
        else:
            citation_w = 0.5

        authority = court_w * 0.4 + recency_w * 0.3 + citation_w * 0.3
        return round(min(max(authority, 0.0), 1.0), 4)

    # Re-ranking

    def rerank(
        self,
        results: list[dict[str, Any]],
        top_k: int = RERANKED_TOP_K,
    ) -> list[dict[str, Any]]:
        """Score and re-rank *results*, returning the top *top_k*.

        Each returned dict is a **new** object containing the original
        fields plus ``authority_score`` and ``final_score``.

        Final score formula::

            final = semantic_score * SEMANTIC_WEIGHT
                  + rrf_norm       * RRF_WEIGHT
                  + authority      * AUTHORITY_WEIGHT

        ``rrf_norm`` is the fused RRF score divided by the rank-1 score of a
        single list, clipped to 1, so a top keyword or semantic hit scores 1.0.
        """
        if not results:
            logger.warning("Reranker received empty results list")
            return []

        scored: list[dict[str, Any]] = []
        for result in results:
            metadata = result.get("metadata") or {}
            authority = self.compute_authority_score(metadata)

            semantic_score = float(result.get("semantic_score", 0.0))
            rrf_score = float(result.get("rrf_score", 0.0))
            rrf_norm = min(rrf_score / _MAX_SINGLE_LIST_RRF, 1.0)

            final_score = (
                semantic_score * SEMANTIC_WEIGHT
                + rrf_norm * RRF_WEIGHT
                + authority * AUTHORITY_WEIGHT
            )

            scored.append({
                **result,
                "authority_score": authority,
                "final_score": round(final_score, 6),
            })

        scored.sort(key=lambda r: r["final_score"], reverse=True)

        top_results = scored[:top_k]
        logger.info(
            "Reranker selected %d / %d candidates (top final_score=%.4f)",
            len(top_results),
            len(scored),
            top_results[0]["final_score"] if top_results else 0.0,
        )
        return top_results
