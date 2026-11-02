"""Authority-aware re-ranking for legal document retrieval results."""

from __future__ import annotations

import logging
import re
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

# Deployed final-score weights (w_s, w_r, w_a) of eq. 6.
DEFAULT_SCORE_WEIGHTS: tuple[float, float, float] = (
    SEMANTIC_WEIGHT,
    RRF_WEIGHT,
    AUTHORITY_WEIGHT,
)

# Court classes of eq. 5. Any other court, or none, gets _UNKNOWN_COURT_WEIGHT.
_JUDICIAL_COURT_KEYS = ("supreme_court", "high_court", "tribunal")
_UNKNOWN_COURT_WEIGHT = 0.3
_TRIBUNAL_PATTERN = re.compile(r"tribunal|\b(?:nclat|nclt|itat|sat|drt|ngt|cat)\b")

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
    # Whole-word abbreviations, so "Satara District Court" is not a tribunal
    if _TRIBUNAL_PATTERN.search(lower):
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

        # eq. 5: Supreme Court 1.0, High Court 0.7, tribunal 0.4, otherwise 0.3
        court_key = _normalise_court(metadata.get("court"))
        court_w = (
            COURT_WEIGHTS.get(court_key, _UNKNOWN_COURT_WEIGHT)
            if court_key in _JUDICIAL_COURT_KEYS
            else _UNKNOWN_COURT_WEIGHT
        )

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
        *,
        weights: Optional[tuple[float, float, float]] = None,
        rrf_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Score and re-rank *results*, returning the top *top_k*.

        Each returned dict is a **new** object containing the original
        fields plus ``authority_score`` and ``final_score``.

        Final score formula (eq. 6)::

            final = semantic_score * w_s
                  + rrf_norm       * w_r
                  + authority      * w_a

        with ``(w_s, w_r, w_a)`` = *weights* (default
        ``(SEMANTIC_WEIGHT, RRF_WEIGHT, AUTHORITY_WEIGHT)`` = (0.50, 0.25,
        0.25)). ``rrf_norm = min(R * (k + 1), 1)`` is the fused RRF score
        divided by the rank-1 score of a single list, so a top keyword or
        semantic hit scores 1.0; *rrf_k* is the ``k`` used in the fusion
        (default ``RRF_K``). The two keyword arguments exist for the
        parameter sweeps in ``scripts/evaluation/retrieval_experiments.py``.
        """
        if not results:
            logger.warning("Reranker received empty results list")
            return []

        w_s, w_r, w_a = weights if weights is not None else DEFAULT_SCORE_WEIGHTS
        max_single_list_rrf = (
            _MAX_SINGLE_LIST_RRF if rrf_k is None else 1.0 / (rrf_k + 1)
        )

        scored: list[dict[str, Any]] = []
        for result in results:
            metadata = result.get("metadata") or {}
            authority = self.compute_authority_score(metadata)

            semantic_score = float(result.get("semantic_score", 0.0))
            rrf_score = float(result.get("rrf_score", 0.0))
            rrf_norm = min(rrf_score / max_single_list_rrf, 1.0)

            final_score = (
                semantic_score * w_s
                + rrf_norm * w_r
                + authority * w_a
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
