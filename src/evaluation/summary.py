"""Aggregate per-query evaluation records into Table IV and Table V of the report.

``scripts/evaluate.py`` writes one record per test query. This module turns
those records into the summary (Table IV: one column per run) and the
per-category breakdown (Table V). It accepts both the current record format
and the older one of Run A (``citation_accuracy`` instead of
``citation_validity``; no coverage, support or keyword metrics), so the
recorded runs in ``data/evaluation/`` can be re-summarised without an LLM.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

# Record key -> older name used by Run A (data/evaluation/eval_results.json).
_ALIASES: dict[str, str] = {"citation_validity": "citation_accuracy"}


def _value(record: dict[str, Any], key: str) -> Optional[float]:
    if key in record and record[key] is not None:
        return float(record[key])
    alias = _ALIASES.get(key)
    if alias and record.get(alias) is not None:
        return float(record[alias])
    return None


def _mean(records: Sequence[dict[str, Any]], key: str) -> Optional[float]:
    """Mean of *key* over all records, or ``None`` if no record has it."""
    values = [_value(r, key) for r in records]
    if not records or all(v is None for v in values):
        return None
    return round(sum(v or 0.0 for v in values) / len(records), 4)


def _share(records: Sequence[dict[str, Any]], predicate) -> float:
    if not records:
        return 0.0
    return round(sum(1 for r in records if predicate(r)) / len(records), 4)


def mean_latency(records: Sequence[dict[str, Any]]) -> Optional[float]:
    """Mean ``elapsed_seconds`` over the queries that ran the pipeline.

    The empty test query is rejected before retrieval in 0 s, so it is left
    out; Table IV's mean latencies (25.0 / 15.4 / 17.5 s) are means over the
    49 non-empty queries.
    """
    ran = [
        r for r in records
        if str(r.get("query", "")).strip() and r.get("elapsed_seconds") is not None
    ]
    if not ran:
        return None
    return round(sum(float(r["elapsed_seconds"]) for r in ran) / len(ran), 3)


def summarise(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Table IV metrics over *records* (all queries, the empty one included).

    Keys missing from every record (e.g. citation support in Run A) are
    ``None``.
    """
    return {
        "total_queries": len(records),
        "citation_validity_mean": _mean(records, "citation_validity"),
        "cited_claim_coverage_mean": _mean(records, "cited_claim_coverage"),
        "citation_support_score_mean": _mean(records, "citation_support_score"),
        "citation_quality_score_mean": _mean(records, "citation_quality_score"),
        "hallucination_rate": _share(records, lambda r: bool(r.get("hallucinations"))),
        "response_rate": _share(records, lambda r: bool(r.get("is_substantive"))),
        "average_confidence": _mean(records, "confidence"),
        "retrieval_precision_mean": _mean(records, "retrieval_precision"),
        "keyword_precision_mean": _mean(records, "keyword_precision"),
        "hit_rate": _mean(records, "hit"),
        "mrr": _mean(records, "reciprocal_rank"),
        "mean_latency_seconds": mean_latency(records),
        "errors": sum(1 for r in records if r.get("error")),
    }


def category_breakdown(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Table V: the summary metrics per query category, in first-seen order."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record.get("category", "unknown"), []).append(record)
    return {category: summarise(items) for category, items in groups.items()}
