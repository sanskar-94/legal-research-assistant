"""Tests for src/generation/ - prompt building, response parsing,
citation validation, and CitedResponse model construction.

All LLM API calls are mocked.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from config.prompts import SYSTEM_PROMPT
from src.generation.prompt_builder import PromptBuilder, source_label
from src.generation.response_parser import (
    CitedResponse,
    CitedSource,
    ResponseParser,
)


# Fixtures

@pytest.fixture()
def prompt_builder() -> PromptBuilder:
    return PromptBuilder()


@pytest.fixture()
def response_parser() -> ResponseParser:
    return ResponseParser()


@pytest.fixture()
def judgement_chunk() -> dict[str, Any]:
    return {
        "chunk_id": "j_001",
        "text": "The court held that Section 302 IPC applies to this case.",
        "doc_type": "judgement",
        "case_name": "State v. Accused",
        "citation": "AIR 2020 SC 1234",
        "court": "Supreme Court of India",
        "date": "2020-05-15",
    }


@pytest.fixture()
def statute_chunk() -> dict[str, Any]:
    return {
        "chunk_id": "s_001",
        "text": "Section 302: Whoever commits murder shall be punished with death or life imprisonment.",
        "doc_type": "statute",
        "act_name": "Indian Penal Code",
        "section_number": "302",
    }


@pytest.fixture()
def constitution_chunk() -> dict[str, Any]:
    return {
        "chunk_id": "c_001",
        "text": "Article 21: No person shall be deprived of his life or personal liberty.",
        "doc_type": "constitution",
        "section_number": "21",
    }


# Tests: prompt building

class TestPromptBuilder:

    def test_returns_system_and_user_prompt(
        self,
        prompt_builder: PromptBuilder,
        sample_retrieved_chunks: list[dict],
    ) -> None:
        sys_prompt, user_prompt = prompt_builder.build_prompt(
            "What is Section 302?", sample_retrieved_chunks
        )
        assert sys_prompt == SYSTEM_PROMPT
        assert isinstance(user_prompt, str)
        assert len(user_prompt) > 0

    def test_sources_included_in_prompt(
        self,
        prompt_builder: PromptBuilder,
        sample_retrieved_chunks: list[dict],
    ) -> None:
        _, user_prompt = prompt_builder.build_prompt(
            "test query", sample_retrieved_chunks
        )
        assert "[SOURCE_1]" in user_prompt
        assert "[SOURCE_2]" in user_prompt
        assert "[SOURCE_3]" in user_prompt

    def test_query_included_in_prompt(
        self,
        prompt_builder: PromptBuilder,
        sample_retrieved_chunks: list[dict],
    ) -> None:
        query = "What is Section 302 of the Indian Penal Code?"
        _, user_prompt = prompt_builder.build_prompt(query, sample_retrieved_chunks)
        assert query in user_prompt

    def test_empty_query_raises(self, prompt_builder: PromptBuilder) -> None:
        with pytest.raises(ValueError, match="Query must not be empty"):
            prompt_builder.build_prompt("", [])

    def test_no_chunks_warning(self, prompt_builder: PromptBuilder) -> None:
        sys_prompt, user_prompt = prompt_builder.build_prompt("test query", [])
        assert sys_prompt == SYSTEM_PROMPT
        assert "test query" in user_prompt

    def test_judgement_metadata_formatting(
        self,
        prompt_builder: PromptBuilder,
        judgement_chunk: dict,
    ) -> None:
        _, user_prompt = prompt_builder.build_prompt("test", [judgement_chunk])
        assert "Case:" in user_prompt
        assert "State v. Accused" in user_prompt
        assert "Court:" in user_prompt

    def test_statute_metadata_formatting(
        self,
        prompt_builder: PromptBuilder,
        statute_chunk: dict,
    ) -> None:
        _, user_prompt = prompt_builder.build_prompt("test", [statute_chunk])
        assert "Act:" in user_prompt
        assert "Indian Penal Code" in user_prompt
        assert "Section:" in user_prompt

    def test_constitution_metadata_formatting(
        self,
        prompt_builder: PromptBuilder,
        constitution_chunk: dict,
    ) -> None:
        _, user_prompt = prompt_builder.build_prompt("test", [constitution_chunk])
        assert "Article:" in user_prompt
        assert "21" in user_prompt

    def test_unknown_doc_type_fallback(
        self, prompt_builder: PromptBuilder
    ) -> None:
        chunk = {
            "chunk_id": "u_001",
            "text": "Some text",
            "doc_type": "regulation",
        }
        _, user_prompt = prompt_builder.build_prompt("test", [chunk])
        assert "Document type:" in user_prompt

    def test_source_numbering_sequential(
        self, prompt_builder: PromptBuilder
    ) -> None:
        chunks = [
            {"chunk_id": f"c{i}", "text": f"text {i}", "doc_type": "statute"}
            for i in range(5)
        ]
        _, user_prompt = prompt_builder.build_prompt("test", chunks)
        for i in range(1, 6):
            assert f"[SOURCE_{i}]" in user_prompt

    @pytest.mark.parametrize(
        "doc_type,expected_label",
        [
            ("judgement", "judgement"),
            ("statute", "statute"),
            ("constitution", "constitution"),
        ],
    )
    def test_doc_type_in_source_block(
        self,
        prompt_builder: PromptBuilder,
        doc_type: str,
        expected_label: str,
    ) -> None:
        chunk = {"chunk_id": "x", "text": "content", "doc_type": doc_type}
        _, user_prompt = prompt_builder.build_prompt("test", [chunk])
        assert f"Type: {expected_label}" in user_prompt


# Tests: response parsing - extracting [SOURCE_N] references

class TestResponseParsing:

    def test_extracts_valid_references(
        self, response_parser: ResponseParser
    ) -> None:
        text = "The answer is [SOURCE_1] and also [SOURCE_2]."
        chunks = [
            {"chunk_id": "c1", "text": "t1", "doc_type": "statute"},
            {"chunk_id": "c2", "text": "t2", "doc_type": "judgement"},
        ]
        result = response_parser.parse_response(text, chunks)
        assert len(result.cited_sources) == 2
        assert len(result.cited_sources) == 2

    def test_deduplicates_repeated_references(
        self, response_parser: ResponseParser
    ) -> None:
        text = "[SOURCE_1] says this. [SOURCE_1] confirms it."
        chunks = [{"chunk_id": "c1", "text": "t", "doc_type": "statute"}]
        result = response_parser.parse_response(text, chunks)
        assert len(result.cited_sources) == 1

    def test_no_references_found(
        self, response_parser: ResponseParser
    ) -> None:
        text = "No citations here."
        chunks = [{"chunk_id": "c1", "text": "t", "doc_type": "statute"}]
        result = response_parser.parse_response(text, chunks)
        assert len(result.cited_sources) == 0

    def test_answer_text_preserved(
        self, response_parser: ResponseParser
    ) -> None:
        text = "The answer is [SOURCE_1]."
        chunks = [{"chunk_id": "c1", "text": "t", "doc_type": "statute"}]
        result = response_parser.parse_response(text, chunks)
        assert result.answer_text == text

    def test_grouped_references_are_expanded(
        self, response_parser: ResponseParser
    ) -> None:
        text = "Both apply [SOURCE_1, SOURCE_3]; see also [SOURCE_2;SOURCE_3]."
        chunks = [
            {"chunk_id": f"c{i}", "text": f"t{i}", "doc_type": "statute"}
            for i in range(1, 4)
        ]
        result = response_parser.parse_response(text, chunks)
        assert result.answer_text == (
            "Both apply [SOURCE_1] [SOURCE_3]; see also [SOURCE_2] [SOURCE_3]."
        )
        assert [s.source_index for s in result.cited_sources] == [1, 3, 2]
        assert result.citation_validity == 1.0


# Tests: citation validation

class TestCitationValidation:

    def test_valid_reference(self, response_parser: ResponseParser) -> None:
        text = "See [SOURCE_1]."
        chunks = [{"chunk_id": "c1", "text": "t", "doc_type": "statute"}]
        result = response_parser.parse_response(text, chunks)
        assert len(result.invalid_citations) == 0
        assert len(result.cited_sources) == 1

    def test_invalid_reference_out_of_range(
        self, response_parser: ResponseParser
    ) -> None:
        text = "See [SOURCE_5]."
        chunks = [{"chunk_id": "c1", "text": "t", "doc_type": "statute"}]
        result = response_parser.parse_response(text, chunks)
        assert "[SOURCE_5]" in result.invalid_citations
        assert len(result.cited_sources) == 0

    def test_mixed_valid_and_invalid(
        self, response_parser: ResponseParser
    ) -> None:
        text = "See [SOURCE_1] and [SOURCE_99]."
        chunks = [
            {"chunk_id": "c1", "text": "t1", "doc_type": "statute"},
            {"chunk_id": "c2", "text": "t2", "doc_type": "judgement"},
        ]
        result = response_parser.parse_response(text, chunks)
        assert len(result.cited_sources) == 1
        assert "[SOURCE_99]" in result.invalid_citations
        assert result.citation_validity == 0.5

    def test_zero_index_is_invalid(
        self, response_parser: ResponseParser
    ) -> None:
        text = "See [SOURCE_0]."
        chunks = [{"chunk_id": "c1", "text": "t", "doc_type": "statute"}]
        result = response_parser.parse_response(text, chunks)
        assert "[SOURCE_0]" in result.invalid_citations
        assert result.citation_validity == 0.0


# Tests: CitedResponse model

class TestCitedResponseModel:

    def test_construction(self) -> None:
        resp = CitedResponse(
            answer_text="Answer",
            cited_sources=[
                CitedSource(
                    source_index=1,
                    chunk_id="c1",
                    text="source text",
                    metadata={"doc_type": "statute"},
                )
            ],
        )
        assert len(resp.cited_sources) == 1
        assert resp.cited_sources[0].chunk_id == "c1"

    def test_defaults(self) -> None:
        resp = CitedResponse(answer_text="A")
        assert resp.cited_sources == []
        assert resp.invalid_citations == []
        assert resp.citation_validity == 0.0

    def test_only_cited_sources_listed(
        self, response_parser: ResponseParser
    ) -> None:
        text = "Only [SOURCE_1] matters."
        chunks = [
            {"chunk_id": "c1", "text": "used", "doc_type": "statute"},
            {"chunk_id": "c2", "text": "unused", "doc_type": "judgement"},
            {"chunk_id": "c3", "text": "also unused", "doc_type": "constitution"},
        ]
        result = response_parser.parse_response(text, chunks)
        assert [s.chunk_id for s in result.cited_sources] == ["c1"]

    def test_cited_source_keeps_document_metadata(
        self, response_parser: ResponseParser
    ) -> None:
        chunks = [{"chunk_id": "c1", "text": "t", "metadata": {"doc_type": "statute", "act_name": "IPC"}}]
        result = response_parser.parse_response("See [SOURCE_1].", chunks)
        assert result.cited_sources[0].metadata == {"doc_type": "statute", "act_name": "IPC"}


# Tests: source labels

class TestSourceLabel:

    def test_judgement_label(self) -> None:
        label = source_label({
            "doc_type": "judgement",
            "case_name": "State v. X",
            "citation": "AIR 2020 SC 100",
            "court": "Supreme Court of India",
        })
        assert label == "Case: State v. X | Citation: AIR 2020 SC 100 | Court: Supreme Court of India"

    def test_statute_label_from_nested_metadata(self) -> None:
        label = source_label({
            "text": "...",
            "metadata": {"doc_type": "statute", "act_name": "IPC", "section_number": "302"},
        })
        assert label == "Act: IPC | Section: 302"

    def test_constitution_label(self) -> None:
        assert source_label({"doc_type": "constitution", "section_number": "21"}) == "Article: 21"

    def test_unknown_doc_type_label(self) -> None:
        assert source_label({"doc_type": "regulation"}) == "Document type: regulation"


# Tests: mock LLM API

class TestLLMClientMocked:

    def _mock_completion(self, client: Any, response_text: str) -> None:
        """Set up mock for OpenAI-compatible chat completions."""
        mock_message = MagicMock()
        mock_message.content = response_text
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        client._client.chat.completions.create.return_value = mock_completion

    def test_generate_returns_text(self) -> None:
        from src.generation.llm_client import LLMClient

        client = LLMClient(api_key="test-key")
        self._mock_completion(client, "According to [SOURCE_1], Section 302 IPC...")

        result = client.generate("What is Section 302?")
        assert result == "According to [SOURCE_1], Section 302 IPC..."

    def test_generate_with_system_prompt(self) -> None:
        from src.generation.llm_client import LLMClient

        client = LLMClient(api_key="test-key")
        self._mock_completion(client, "response text")

        result = client.generate("query", system_prompt="You are a legal expert.")
        assert result == "response text"

    def test_generate_strips_thinking_tags(self) -> None:
        from src.generation.llm_client import LLMClient

        client = LLMClient(api_key="test-key")
        self._mock_completion(
            client,
            "<think>I should reason privately.</think>\n"
            "Section 302 IPC punishes murder. [SOURCE_1]",
        )

        result = client.generate("query")
        assert result == "Section 302 IPC punishes murder. [SOURCE_1]"

    def test_generate_strips_visible_reasoning_prefix(self) -> None:
        from src.generation.llm_client import LLMClient

        client = LLMClient(api_key="test-key")
        self._mock_completion(
            client,
            "Reasoning: I will compare the retrieved sources first.\n\n"
            "Final Answer: Section 302 IPC punishes murder. [SOURCE_1]",
        )

        result = client.generate("query")
        assert result == "Section 302 IPC punishes murder. [SOURCE_1]"

    def test_empty_prompt_raises(self) -> None:
        from src.generation.llm_client import LLMClient, LLMClientError

        client = LLMClient(api_key="test-key")
        with pytest.raises(LLMClientError, match="Prompt must not be empty"):
            client.generate("")

    def test_no_api_key_raises(self) -> None:
        with patch.dict("os.environ", {}, clear=True), patch(
            "src.generation.llm_client.LLM_BASE_URL", "https://integrate.api.nvidia.com/v1"
        ):
            from src.generation.llm_client import LLMClient, LLMClientError
            with pytest.raises(LLMClientError, match="No API key"):
                LLMClient()

    def test_local_server_needs_no_api_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True), patch(
            "src.generation.llm_client.LLM_BASE_URL", "http://127.0.0.1:8080/v1"
        ):
            from src.generation.llm_client import LLMClient
            client = LLMClient()
            self._mock_completion(client, "Section 302 IPC punishes murder. [SOURCE_1]")
            assert client.generate("query") == "Section 302 IPC punishes murder. [SOURCE_1]"


class TestFlattenChunk:

    def test_nested_metadata_lifted(self) -> None:
        from src.generation.prompt_builder import flatten_chunk

        chunk = {"text": "t", "final_score": 0.5, "metadata": {"case_name": "A v. B", "final_score": 0.1}}
        flat = flatten_chunk(chunk)
        assert flat["case_name"] == "A v. B"
        assert flat["final_score"] == 0.5  # top-level value wins
        assert "case_name" not in chunk  # input untouched

    def test_nested_metadata_used_in_prompt(self, prompt_builder: PromptBuilder) -> None:
        chunk = {
            "text": "Whoever commits murder shall be punished.",
            "metadata": {"doc_type": "statute", "act_name": "Indian Penal Code, 1860", "section_number": "302"},
        }
        _, user_prompt = prompt_builder.build_prompt("What is murder?", [chunk])
        assert "Type: statute" in user_prompt
        assert "Act: Indian Penal Code, 1860 | Section: 302" in user_prompt

    def test_mapping_source_formatted(self, prompt_builder: PromptBuilder) -> None:
        from src.retrieval.ipc_bns import IpcBnsMapper

        mapper = IpcBnsMapper()
        chunk = mapper.as_source_chunk(mapper.find("Section 302 IPC"))
        _, user_prompt = prompt_builder.build_prompt("Section 302 IPC?", [chunk])
        assert "Type: mapping" in user_prompt
        assert "Correspondence table" in user_prompt
