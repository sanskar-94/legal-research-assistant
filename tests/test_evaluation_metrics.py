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
