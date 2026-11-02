"""Tests for the answer-quality metrics in src/evaluation/citation_metrics.py."""

from __future__ import annotations

from src.evaluation.citation_metrics import (
    compute_citation_quality,
    compute_citation_validity,
    detect_hallucinations,
    is_substantive_response,
)


def test_citation_validity_only_checks_source_indices() -> None:
    chunks = [
        {
            "chunk_id": "ipc_302",
            "text": "Section 302 IPC: Whoever commits murder shall be punished with death or life imprisonment.",
        }
    ]
    response = "Section 302 IPC deals with theft. [SOURCE_1]"

    assert compute_citation_validity(response, chunks) == 1.0


def test_citation_support_penalizes_unsupported_claim() -> None:
    chunks = [
        {
            "chunk_id": "ipc_302",
            "text": "Section 302 IPC: Whoever commits murder shall be punished with death or life imprisonment.",
        }
    ]
    response = "Section 302 IPC deals with theft. [SOURCE_1]"

    quality = compute_citation_quality(response, chunks)

    assert quality["citation_validity"] == 1.0
    assert quality["cited_claim_coverage"] == 1.0
    assert quality["citation_support_score"] == 0.0
    assert quality["citation_quality_score"] == 0.6


def test_citation_support_rewards_supported_claim() -> None:
    chunks = [
        {
            "chunk_id": "ipc_302",
            "text": "Section 302 IPC: Whoever commits murder shall be punished with death or life imprisonment.",
        }
    ]
    response = (
        "Section 302 IPC provides punishment for murder with death or life "
        "imprisonment. [SOURCE_1]"
    )

    quality = compute_citation_quality(response, chunks)

    assert quality["citation_validity"] == 1.0
    assert quality["cited_claim_coverage"] == 1.0
    assert quality["citation_support_score"] == 1.0
    assert quality["citation_quality_score"] == 1.0


SOURCES = [
    {
        "chunk_id": "sc_1",
        "text": "In State of Maharashtra v. Prakash the court relied on AIR 1984 SC 1622.",
    }
]


def test_hallucination_flags_unknown_case() -> None:
    response = "As held in Ramesh Kumar v. State of Punjab, the appeal fails. [SOURCE_1]"
    flagged = detect_hallucinations(response, SOURCES)
    assert any("Ramesh Kumar" in item for item in flagged)


def test_hallucination_ignores_case_in_sources() -> None:
    response = "In State of Maharashtra v. Prakash, the conviction stood. [SOURCE_1]"
    assert detect_hallucinations(response, SOURCES) == []


def test_hallucination_flags_unsourced_citation() -> None:
    flagged = detect_hallucinations("See AIR 1999 SC 1234. [SOURCE_1]", SOURCES)
    assert flagged == ["Citation: AIR 1999 SC 1234"]


def test_substantive_requires_valid_citation() -> None:
    assert is_substantive_response("Murder is punished under Section 302. [SOURCE_1]", 1)
    assert not is_substantive_response("Murder is punished under Section 302.", 1)
    assert not is_substantive_response("Murder is punished. [SOURCE_4]", 1)


def test_partial_answer_still_substantive() -> None:
    response = (
        "Section 302 IPC punishes murder with death or life imprisonment [SOURCE_1]. "
        "Not covered by the retrieved sources: the BNS equivalent."
    )
    assert is_substantive_response(response, 1)


def test_refusal_is_not_substantive() -> None:
    assert not is_substantive_response("The retrieved sources do not address this question.", 3)


# Retrieval metrics (Section IV-B, eq. 8) and the Table IV / V aggregation

import importlib.util  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402

from src.evaluation.retrieval_metrics import (  # noqa: E402
    hit_at_k,
    keyword_precision_at_k,
    mean_retrieval_metrics,
    precision_at_k,
    reciprocal_rank,
    retrieval_metrics,
)
from src.evaluation.summary import category_breakdown, mean_latency, summarise  # noqa: E402

RETRIEVED = [
    {"chunk_id": "a", "text": "Article 14 equality", "metadata": {"doc_type": "statute"}},
    {"chunk_id": "b", "text": "Whoever commits MURDER", "metadata": {"doc_type": "judgement"}},
    {"chunk_id": "c", "text": "Unrelated text", "metadata": {"doc_type": "judgement"}},
    {"chunk_id": "d", "text": "Section 302 IPC", "metadata": {"doc_type": "statute"}},
]
TYPES = ["statute"]
KEYWORDS = ["Section 302", "murder"]


def test_precision_counts_type_or_keyword() -> None:
    # a: type; b: keyword (case-insensitive); c: neither; d: both
    assert precision_at_k(RETRIEVED, TYPES, KEYWORDS, k=4) == 0.75


def test_keyword_precision_ignores_type() -> None:
    assert keyword_precision_at_k(RETRIEVED, KEYWORDS, k=4) == 0.5


def test_precision_denominator_is_k() -> None:
    # eq. 8 divides by K, so a short list is not rewarded
    assert precision_at_k(RETRIEVED[:2], TYPES, KEYWORDS, k=4) == 0.5
    assert keyword_precision_at_k([], KEYWORDS, k=4) == 0.0
    # without K the denominator is the number retrieved (previous behaviour)
    assert precision_at_k(RETRIEVED[:2], TYPES, KEYWORDS) == 1.0


def test_hit_and_reciprocal_rank() -> None:
    assert hit_at_k(RETRIEVED, KEYWORDS, k=4) == 1.0
    assert hit_at_k(RETRIEVED, KEYWORDS, k=1) == 0.0
    assert reciprocal_rank(RETRIEVED, KEYWORDS, k=4) == 0.5
    assert reciprocal_rank(RETRIEVED, KEYWORDS, k=1) == 0.0
    assert reciprocal_rank(RETRIEVED, ["nothing"], k=4) == 0.0


def test_mean_reciprocal_rank() -> None:
    per_query = [
        retrieval_metrics(RETRIEVED, TYPES, KEYWORDS, k=4),
        retrieval_metrics(RETRIEVED[3:], TYPES, KEYWORDS, k=4),
        retrieval_metrics([], TYPES, KEYWORDS, k=4),
    ]
    mean = mean_retrieval_metrics(per_query)
    assert mean["MRR"] == pytest.approx((0.5 + 1.0 + 0.0) / 3)
    assert mean["Hit"] == pytest.approx(2 / 3)
    assert mean["P_kw"] == pytest.approx((0.5 + 0.25 + 0.0) / 3)
    assert mean["n"] == 3


def _record(query: str, category: str, **values: object) -> dict:
    base = {
        "query": query, "category": category, "citation_validity": 1.0,
        "cited_claim_coverage": 0.5, "citation_support_score": 0.5,
        "citation_quality_score": 0.65, "hallucinations": [], "is_substantive": True,
        "confidence": 0.8, "retrieval_precision": 1.0, "keyword_precision": 0.75,
        "hit": 1.0, "reciprocal_rank": 1.0, "elapsed_seconds": 10.0, "error": None,
    }
    return {**base, **values}


def test_mean_latency_skips_empty_query() -> None:
    records = [
        _record("q1", "a", elapsed_seconds=10.0),
        _record("q2", "a", elapsed_seconds=20.0),
        _record("", "edge", elapsed_seconds=0.0),
    ]
    assert mean_latency(records) == 15.0


def test_summary_and_breakdown() -> None:
    records = [
        _record("q1", "section_lookup"),
        _record("q2", "section_lookup", hallucinations=["Case: X v Y"], hit=0.0, reciprocal_rank=0.0),
        _record("", "edge_case", citation_validity=0.0, is_substantive=False, hit=0.0,
                reciprocal_rank=0.0, keyword_precision=0.0, retrieval_precision=0.0,
                elapsed_seconds=0.0, error="Query must not be empty."),
    ]
    summary = summarise(records)
    assert summary["citation_validity_mean"] == pytest.approx(0.6667, abs=1e-4)
    assert summary["hallucination_rate"] == pytest.approx(0.3333, abs=1e-4)
    assert summary["response_rate"] == pytest.approx(0.6667, abs=1e-4)
    assert summary["hit_rate"] == pytest.approx(0.3333, abs=1e-4)
    assert summary["mrr"] == pytest.approx(0.3333, abs=1e-4)
    assert summary["mean_latency_seconds"] == 10.0
    assert summary["errors"] == 1
    breakdown = category_breakdown(records)
    assert list(breakdown) == ["section_lookup", "edge_case"]
    assert breakdown["section_lookup"]["hit_rate"] == 0.5
    assert breakdown["edge_case"]["mean_latency_seconds"] is None


def test_summary_reads_run_a_format() -> None:
    old = {"query": "q", "category": "c", "citation_accuracy": 0.9, "hallucinations": [],
           "is_substantive": False, "confidence": 0.7, "retrieval_precision": 0.5,
           "elapsed_seconds": 25.0}
    summary = summarise([old])
    assert summary["citation_validity_mean"] == 0.9
    assert summary["citation_support_score_mean"] is None
    assert summary["mrr"] is None


def _load_evaluate_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "evaluation" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("evaluate_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_evaluate_query_record_has_retrieval_metrics() -> None:
    evaluate = _load_evaluate_script()
    sources = [
        {"chunk_id": "s1", "text": "Section 302 IPC: punishment for murder.", "metadata": {"doc_type": "statute"}},
        {"chunk_id": "s2", "text": "Unrelated", "metadata": {"doc_type": "judgement"}},
        {"chunk_id": "ipc_bns_mapping", "text": "IPC 302 -> BNS 103", "metadata": {"doc_type": "mapping"}},
    ]
    result = SimpleNamespace(
        answer_text="Section 302 IPC punishes murder with death or life imprisonment. [SOURCE_1]",
        sources=sources,
        confidence=SimpleNamespace(overall_score=0.7),
        ipc_bns_mappings=[{"ipc_section": "302"}],
        error=None,
    )
    test_query = {
        "query": "What is Section 302 IPC?", "category": "section_lookup",
        "expected_doc_types": ["statute"], "expected_keywords": ["Section 302", "murder"],
    }
    record = evaluate.query_record(test_query, result, elapsed=12.3456, top_k=2)
    # the IPC/BNS table is a prompt source but not a retrieved chunk
    assert record["num_chunks_retrieved"] == 2
    assert record["retrieval_precision"] == 0.5
    assert record["keyword_precision"] == 0.5
    assert record["hit"] == 1.0
    assert record["reciprocal_rank"] == 1.0
    assert record["elapsed_seconds"] == 12.346
    assert record["citation_validity"] == 1.0
    assert record["is_substantive"] is True
    assert record["ipc_bns_matches"] == ["302"]
