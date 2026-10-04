"""Tests for evaluation citation-quality metrics."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.evaluate import (
    _compute_citation_quality,
    _compute_citation_validity,
)


def test_citation_validity_only_checks_source_indices() -> None:
    chunks = [
        {
            "chunk_id": "ipc_302",
            "text": "Section 302 IPC: Whoever commits murder shall be punished with death or life imprisonment.",
        }
    ]
    response = "Section 302 IPC deals with theft. [SOURCE_1]"

    assert _compute_citation_validity(response, chunks) == 1.0


def test_citation_support_penalizes_unsupported_claim() -> None:
    chunks = [
        {
            "chunk_id": "ipc_302",
            "text": "Section 302 IPC: Whoever commits murder shall be punished with death or life imprisonment.",
        }
    ]
    response = "Section 302 IPC deals with theft. [SOURCE_1]"

    quality = _compute_citation_quality(response, chunks)

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

    quality = _compute_citation_quality(response, chunks)

    assert quality["citation_validity"] == 1.0
    assert quality["cited_claim_coverage"] == 1.0
    assert quality["citation_support_score"] == 1.0
    assert quality["citation_quality_score"] == 1.0
