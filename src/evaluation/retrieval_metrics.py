"""Retrieval metrics of Section IV-B of the report (no LLM involved).

For a query with retrieved list ``D_K`` (the top ``K`` chunks), expected
keywords ``W`` and expected document types ``T``:

* ``P@K``    (eq. 8)  ``1/K * sum_{c in D_K} 1[type(c) in T or some w in W occurs in c]``
* ``P_kw@K``          ``1/K * sum_{c in D_K} 1[some w in W occurs in c]``
* ``Hit@K``           ``1`` if at least one chunk of ``D_K`` contains a keyword
* ``RR@K``            ``1 / rank`` of the first keyword-relevant chunk in ``D_K``
                      (``0`` if none); its mean over queries is the MRR

Keyword matching is a case-insensitive substring test, as in
``scripts/evaluate.py``. When fewer than ``K`` chunks are retrieved the
missing slots count as irrelevant (the denominator stays ``K``).
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence


def _text(chunk: dict[str, Any]) -> str:
    return str(chunk.get("text", "") or "")


def _doc_type(chunk: dict[str, Any]) -> str:
    metadata = chunk.get("metadata") or {}
    return str(metadata.get("doc_type") or chunk.get("doc_type") or "")


def is_keyword_relevant(chunk: dict[str, Any], expected_keywords: Iterable[str]) -> bool:
    """True when *chunk* contains at least one expected keyword."""
    text = _text(chunk).lower()
    return any(kw and kw.lower() in text for kw in expected_keywords)


def is_relevant(
    chunk: dict[str, Any],
    expected_doc_types: Iterable[str],
    expected_keywords: Iterable[str],
) -> bool:
    """Relevance used by ``P@K`` (eq. 8): expected type OR expected keyword."""
    return _doc_type(chunk) in set(expected_doc_types) or is_keyword_relevant(
        chunk, expected_keywords
    )


def _top(chunks: Sequence[dict[str, Any]], k: Optional[int]) -> tuple[list[dict[str, Any]], int]:
    """Return ``(D_K, K)``. ``k=None`` means "all retrieved chunks"."""
    top = list(chunks) if k is None else list(chunks)[:k]
    return top, (len(top) if k is None else k)


def precision_at_k(
    chunks: Sequence[dict[str, Any]],
    expected_doc_types: Iterable[str],
    expected_keywords: Iterable[str],
    k: Optional[int] = None,
) -> float:
    """``P@K`` of eq. 8."""
    top, denom = _top(chunks, k)
    if denom <= 0:
        return 0.0
    types, keywords = list(expected_doc_types), list(expected_keywords)
    return sum(1 for c in top if is_relevant(c, types, keywords)) / denom


def keyword_precision_at_k(
    chunks: Sequence[dict[str, Any]],
    expected_keywords: Iterable[str],
    k: Optional[int] = None,
) -> float:
    """``P_kw@K``: share of the ``K`` slots holding a chunk with an expected keyword."""
    top, denom = _top(chunks, k)
    if denom <= 0:
        return 0.0
    keywords = list(expected_keywords)
    return sum(1 for c in top if is_keyword_relevant(c, keywords)) / denom


def hit_at_k(
    chunks: Sequence[dict[str, Any]],
    expected_keywords: Iterable[str],
    k: Optional[int] = None,
) -> float:
    """``Hit@K``: 1.0 if any of the top ``K`` chunks contains an expected keyword."""
    top, _ = _top(chunks, k)
    keywords = list(expected_keywords)
    return 1.0 if any(is_keyword_relevant(c, keywords) for c in top) else 0.0


def reciprocal_rank(
    chunks: Sequence[dict[str, Any]],
    expected_keywords: Iterable[str],
    k: Optional[int] = None,
) -> float:
    """``1/rank`` of the first keyword-relevant chunk within the top ``K`` (0 if none)."""
    top, _ = _top(chunks, k)
    keywords = list(expected_keywords)
    for rank, chunk in enumerate(top, start=1):
        if is_keyword_relevant(chunk, keywords):
            return 1.0 / rank
    return 0.0


def retrieval_metrics(
    chunks: Sequence[dict[str, Any]],
    expected_doc_types: Iterable[str],
    expected_keywords: Iterable[str],
    k: Optional[int] = None,
) -> dict[str, float]:
    """All four retrieval metrics for one query, keyed ``P``, ``P_kw``, ``Hit``, ``RR``."""
    types, keywords = list(expected_doc_types), list(expected_keywords)
    return {
        "P": precision_at_k(chunks, types, keywords, k),
        "P_kw": keyword_precision_at_k(chunks, keywords, k),
        "Hit": hit_at_k(chunks, keywords, k),
        "RR": reciprocal_rank(chunks, keywords, k),
    }


def mean_retrieval_metrics(per_query: Sequence[dict[str, float]]) -> dict[str, float]:
    """Average per-query metric dicts; ``RR`` becomes ``MRR``."""
    n = len(per_query)
    if n == 0:
        return {"P": 0.0, "P_kw": 0.0, "Hit": 0.0, "MRR": 0.0, "n": 0}
    return {
        "P": sum(m["P"] for m in per_query) / n,
        "P_kw": sum(m["P_kw"] for m in per_query) / n,
        "Hit": sum(m["Hit"] for m in per_query) / n,
        "MRR": sum(m["RR"] for m in per_query) / n,
        "n": n,
    }
