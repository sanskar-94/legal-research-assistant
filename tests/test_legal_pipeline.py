"""Tests for src/pipeline.py with every external service mocked."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.confidence.scorer import ConfidenceScorer
from src.generation.prompt_builder import PromptBuilder
from src.generation.response_parser import ResponseParser
from src.pipeline import LegalResearchPipeline
from src.retrieval.ipc_bns import MAPPING_CHUNK_ID, IpcBnsMapper


def _chunk(chunk_id: str, text: str, **metadata: Any) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "metadata": metadata,
        "semantic_score": 0.8,
        "rrf_score": 0.01,
        "authority_score": 1.0,
        "final_score": 0.75,
    }


RETRIEVED = [
    _chunk(
        "ipc::302",
        "Section 302. Punishment for murder. Whoever commits murder shall be "
        "punished with death, or imprisonment for life, and shall also be liable to fine.",
        doc_type="statute",
        act_name="Indian Penal Code, 1860",
        section_number="302",
    ),
    _chunk(
        "sc::1",
        "The conviction under Section 302 IPC was upheld by this Court.",
        doc_type="judgement",
        case_name="State of Maharashtra v. Prakash",
        court="Supreme Court of India",
    ),
]


def _build(retrieved: list[dict], answer: str, mapper: Any = "default") -> tuple[LegalResearchPipeline, MagicMock, MagicMock]:
    """Create a pipeline without running __init__ (which loads models)."""
    pipeline = LegalResearchPipeline.__new__(LegalResearchPipeline)
    retrieval = MagicMock()
    retrieval.retrieve.return_value = retrieved
    llm = MagicMock()
    llm.generate.return_value = answer
    pipeline._retrieval_pipeline = retrieval
    pipeline._llm_client = llm
    pipeline._ipc_bns_mapper = IpcBnsMapper() if mapper == "default" else mapper
    pipeline._prompt_builder = PromptBuilder()
    pipeline._response_parser = ResponseParser()
    pipeline._confidence_scorer = ConfidenceScorer()
    pipeline._vectorstore = MagicMock()
    return pipeline, retrieval, llm


class TestLegalResearchPipeline:

    def test_empty_query(self) -> None:
        pipeline, retrieval, _ = _build(RETRIEVED, "")
        result = pipeline.query("   ")
        assert result.error == "Query must not be empty."
        retrieval.retrieve.assert_not_called()

    def test_ipc_query_adds_mapping_source(self) -> None:
        answer = (
            "Murder is punished under Section 302 IPC [SOURCE_1], which now "
            "corresponds to Section 103 BNS [SOURCE_3]. [SOURCE_2]"
        )
        pipeline, retrieval, llm = _build(RETRIEVED, answer)
        result = pipeline.query("What is Section 302 IPC and its BNS equivalent?")

        assert result.error is None
        # Retrieval ran on the expanded query
        sent_query = retrieval.retrieve.call_args.args[0]
        assert "Section 103 BNS" in sent_query
        # The mapping table is the last numbered source in the prompt
        user_prompt = llm.generate.call_args.args[0]
        assert "[SOURCE_3]" in user_prompt
        assert "BNS Section 103" in user_prompt
        assert result.retrieved_chunks[-1]["chunk_id"] == MAPPING_CHUNK_ID
        assert [m["ipc_section"] for m in result.ipc_bns_mappings] == ["302"]
        assert result.retrieved_count == 2
        assert result.cited_count == 3

    def test_metadata_reaches_prompt(self) -> None:
        pipeline, _, llm = _build(RETRIEVED, "Answer [SOURCE_1].")
        pipeline.query("What is the punishment for murder?")
        user_prompt = llm.generate.call_args.args[0]
        assert "Act: Indian Penal Code, 1860 | Section: 302" in user_prompt
        assert "Case: State of Maharashtra v. Prakash" in user_prompt

    def test_no_mapping_for_constitutional_query(self) -> None:
        pipeline, retrieval, _ = _build(RETRIEVED, "Answer [SOURCE_1].")
        result = pipeline.query("What does Article 21 guarantee?")
        assert result.ipc_bns_mappings == []
        assert retrieval.retrieve.call_args.args[0] == "What does Article 21 guarantee?"
        assert all(c["chunk_id"] != MAPPING_CHUNK_ID for c in result.retrieved_chunks)

    def test_nothing_retrieved(self) -> None:
        pipeline, _, llm = _build([], "unused")
        result = pipeline.query("What does Article 21 guarantee?")
        assert result.error == "No relevant documents found for the query."
        llm.generate.assert_not_called()

    def test_mapping_alone_is_enough_to_answer(self) -> None:
        pipeline, _, _ = _build([], "Section 302 IPC is now Section 103 BNS [SOURCE_1].")
        result = pipeline.query("Section 302 IPC in BNS?")
        assert result.error is None
        assert result.cited_count == 1

    def test_generation_failure_is_reported(self) -> None:
        pipeline, _, llm = _build(RETRIEVED, "")
        llm.generate.side_effect = RuntimeError("endpoint down")
        result = pipeline.query("What is Section 302 IPC?")
        assert "Generation stage failed" in result.error
        assert result.retrieved_count == 2

    def test_works_without_mapper(self) -> None:
        pipeline, _, _ = _build(RETRIEVED, "Answer [SOURCE_1].", mapper=None)
        result = pipeline.query("What is Section 302 IPC?")
        assert result.error is None
        assert result.ipc_bns_mappings == []

    def test_confidence_uses_retrieved_chunks_only(self) -> None:
        pipeline, _, _ = _build(RETRIEVED, "A [SOURCE_1] B [SOURCE_2] C [SOURCE_3].")
        result = pipeline.query("What is Section 302 IPC?")
        assert result.confidence is not None
        # both retrieved chunks were cited
        assert result.confidence.breakdown["coverage_score"] == pytest.approx(1.0)

    def test_without_llm_returns_sources(self) -> None:
        pipeline, _, _ = _build(RETRIEVED, "unused")
        pipeline._llm_client = None
        result = pipeline.query("What is Section 302 IPC?")
        assert "NVIDIA_API_KEY" in result.error
        assert result.answer_text == ""
        assert len(result.retrieved_chunks) == 3  # two passages + IPC/BNS table
