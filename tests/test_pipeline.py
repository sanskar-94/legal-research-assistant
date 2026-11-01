"""Tests for src/pipeline.py and src/confidence/scorer.py.

``TestLegalResearchPipeline`` mocks the retrieval stage to check how the
pipeline assembles sources, prompts and results. ``TestEndToEnd`` runs the
real retrieval, prompt, parsing and scoring code with only the vector
store, embedding model and LLM mocked.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.confidence.scorer import ConfidenceScorer, confidence_level
from src.generation.response_parser import CitedResponse, CitedSource
from src.pipeline import LegalResearchPipeline
from src.retrieval.ipc_bns import MAPPING_CHUNK_ID, IpcBnsMapper
from src.retrieval.retrieval_pipeline import RetrievalPipeline


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


def _build(
    retrieved: list[dict],
    answer: str,
    mapper: Any = "default",
) -> tuple[LegalResearchPipeline, MagicMock, MagicMock]:
    retrieval = MagicMock()
    retrieval.retrieve.return_value = retrieved
    llm = MagicMock()
    llm.generate.return_value = answer
    pipeline = LegalResearchPipeline(
        retrieval,
        llm,
        IpcBnsMapper() if mapper == "default" else mapper,
    )
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
        assert result.sources[-1]["chunk_id"] == MAPPING_CHUNK_ID
        assert [m["ipc_section"] for m in result.ipc_bns_mappings] == ["302"]
        assert result.retrieved_count == 2
        assert len(result.cited_sources) == 3

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
        assert all(c["chunk_id"] != MAPPING_CHUNK_ID for c in result.sources)

    def test_nothing_retrieved(self) -> None:
        pipeline, _, llm = _build([], "unused")
        result = pipeline.query("What does Article 21 guarantee?")
        assert result.error == "No relevant documents found for the query."
        llm.generate.assert_not_called()

    def test_mapping_alone_is_enough_to_answer(self) -> None:
        pipeline, _, _ = _build([], "Section 302 IPC is now Section 103 BNS [SOURCE_1].")
        result = pipeline.query("Section 302 IPC in BNS?")
        assert result.error is None
        assert len(result.cited_sources) == 1

    def test_retrieval_failure_is_reported(self) -> None:
        pipeline, retrieval, llm = _build(RETRIEVED, "unused")
        retrieval.retrieve.side_effect = RuntimeError("Connection refused")
        result = pipeline.query("What is Section 302 IPC?")
        assert result.error == "Retrieval failed: Connection refused"
        llm.generate.assert_not_called()

    def test_generation_failure_keeps_sources(self) -> None:
        pipeline, _, llm = _build(RETRIEVED, "")
        llm.generate.side_effect = RuntimeError("endpoint down")
        result = pipeline.query("What is Section 302 IPC?")
        assert result.error == "Generation failed: endpoint down"
        assert result.retrieved_count == 2
        assert len(result.sources) == 3

    def test_works_without_mapper(self) -> None:
        pipeline, _, _ = _build(RETRIEVED, "Answer [SOURCE_1].", mapper=None)
        result = pipeline.query("What is Section 302 IPC?")
        assert result.error is None
        assert result.ipc_bns_mappings == []

    def test_invalid_citations_are_reported(self) -> None:
        pipeline, _, _ = _build(RETRIEVED, "Answer [SOURCE_1] [SOURCE_9].", mapper=None)
        result = pipeline.query("What is the punishment for murder?")
        assert result.invalid_citations == ["[SOURCE_9]"]

    def test_confidence_uses_retrieved_chunks_only(self) -> None:
        pipeline, _, _ = _build(RETRIEVED, "A [SOURCE_1] B [SOURCE_2] C [SOURCE_3].")
        result = pipeline.query("What is Section 302 IPC?")
        assert result.confidence is not None
        # both retrieved chunks were cited
        assert result.confidence.breakdown["coverage_score"] == pytest.approx(1.0)

    def test_without_llm_returns_sources(self) -> None:
        retrieval = MagicMock()
        retrieval.retrieve.return_value = RETRIEVED
        pipeline = LegalResearchPipeline(retrieval, None, IpcBnsMapper())
        assert not pipeline.can_generate
        result = pipeline.query("What is Section 302 IPC?")
        assert "NVIDIA_API_KEY" in result.error
        assert result.answer_text == ""
        assert len(result.sources) == 3  # two passages + IPC/BNS table


class TestEndToEnd:

    @pytest.fixture()
    def collection(self) -> MagicMock:
        """A ChromaDB collection mock with realistic legal chunks."""
        store = MagicMock()
        store.query.return_value = {
            "ids": [["c1", "c2", "c3"]],
            "documents": [[
                "Section 302 IPC: Whoever commits murder shall be punished with death or life imprisonment.",
                "In State of Maharashtra v. Prakash, the conviction under Section 302 was upheld.",
                "Article 21 of the Constitution guarantees right to life and personal liberty.",
            ]],
            "metadatas": [[
                {"doc_type": "statute", "section_number": "302", "act_name": "IPC"},
                {"doc_type": "judgement", "court": "Supreme Court of India", "date": "2020-03-15"},
                {"doc_type": "constitution", "section_number": "21"},
            ]],
            "distances": [[0.1, 0.2, 0.4]],
        }
        return store

    def test_answer_with_citations_and_confidence(
        self,
        collection: MagicMock,
        mock_embedding_service: MagicMock,
    ) -> None:
        llm = MagicMock()
        llm.generate.return_value = (
            "Section 302 punishes murder [SOURCE_1]. The Supreme Court upheld "
            "a conviction under it [SOURCE_2]."
        )
        pipeline = LegalResearchPipeline(
            RetrievalPipeline(collection, mock_embedding_service), llm, IpcBnsMapper()
        )
        result = pipeline.query("What is Section 302 IPC?")

        assert result.error is None
        assert result.retrieved_count == 3
        scores = [c["final_score"] for c in result.sources[:3]]
        assert scores == sorted(scores, reverse=True)
        assert result.sources[-1]["chunk_id"] == MAPPING_CHUNK_ID
        assert {s.source_index for s in result.cited_sources} == {1, 2}
        assert result.confidence is not None
        assert 0.0 < result.confidence.overall_score <= 1.0

    def test_embedding_failure_is_reported(self, collection: MagicMock) -> None:
        embedder = MagicMock()
        embedder.embed_query.side_effect = RuntimeError("Model load failed")
        pipeline = LegalResearchPipeline(RetrievalPipeline(collection, embedder), MagicMock())
        result = pipeline.query("What is Section 302 IPC?")
        assert result.error == "Retrieval failed: Model load failed"


class TestConfidenceScorer:

    @staticmethod
    def _cited(*chunk_ids: str) -> CitedResponse:
        return CitedResponse(
            answer_text="",
            cited_sources=[
                CitedSource(source_index=i, chunk_id=cid, text="", metadata={})
                for i, cid in enumerate(chunk_ids, start=1)
            ],
        )

    def test_no_chunks_scores_zero(self) -> None:
        result = ConfidenceScorer().score([], self._cited())
        assert result.overall_score == 0.0
        assert result.level == "Very Low"

    def test_more_citations_score_higher(self) -> None:
        scorer = ConfidenceScorer()
        all_cited = scorer.score(RETRIEVED, self._cited("ipc::302", "sc::1"))
        one_cited = scorer.score(RETRIEVED, self._cited("ipc::302"))
        assert all_cited.overall_score > one_cited.overall_score
        assert all_cited.breakdown["coverage_score"] == 1.0
        assert one_cited.breakdown["coverage_score"] == 0.5

    def test_authority_uses_cited_chunks(self) -> None:
        chunks = [
            {**RETRIEVED[0], "authority_score": 1.0},
            {**RETRIEVED[1], "authority_score": 0.2},
        ]
        result = ConfidenceScorer().score(chunks, self._cited("sc::1"))
        assert result.breakdown["authority_strength"] == pytest.approx(0.2)

    @pytest.mark.parametrize(
        ("score", "level"),
        [(0.95, "High"), (0.8, "High"), (0.6, "Medium"), (0.45, "Low"), (0.1, "Very Low")],
    )
    def test_levels(self, score: float, level: str) -> None:
        assert confidence_level(score) == level
