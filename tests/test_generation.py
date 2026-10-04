"""Tests for src/generation/ — prompt building, response parsing,
citation validation, and CitedResponse model construction.

All Gemini API calls are mocked.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ── Path bootstrap ──────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Mock heavy third-party deps before importing src.generation ─────
# google.generativeai is imported at module level by gemini_client.py
# (which is re-exported from src.generation.__init__).  We install a
# lightweight stub so the tests can run without the real SDK.
if "google" not in sys.modules:
    _google = ModuleType("google")
    _google_genai = ModuleType("google.generativeai")
    _google_genai_types = ModuleType("google.generativeai.types")
    _google_genai.types = _google_genai_types  # type: ignore[attr-defined]
    _google_genai.configure = lambda **kw: None  # type: ignore[attr-defined]
    _google_genai.GenerativeModel = MagicMock  # type: ignore[attr-defined]
    _google_genai_types.GenerationConfig = MagicMock  # type: ignore[attr-defined]
    sys.modules["google"] = _google
    sys.modules["google.generativeai"] = _google_genai
    sys.modules["google.generativeai.types"] = _google_genai_types

# Stub the new google.genai SDK so GeminiClient.__init__ can use it
if "google.genai" not in sys.modules:
    _google_mod = sys.modules["google"]
    _google_genai_new = ModuleType("google.genai")
    _google_genai_new_types = ModuleType("google.genai.types")
    _google_genai_new.Client = MagicMock  # type: ignore[attr-defined]
    _google_genai_new.types = _google_genai_new_types  # type: ignore[attr-defined]
    _google_genai_new_types.GenerateContentConfig = MagicMock  # type: ignore[attr-defined]
    _google_mod.genai = _google_genai_new  # type: ignore[attr-defined]
    sys.modules["google.genai"] = _google_genai_new
    sys.modules["google.genai.types"] = _google_genai_new_types

if "dotenv" not in sys.modules:
    _dotenv = ModuleType("dotenv")
    _dotenv.load_dotenv = lambda *a, **kw: None  # type: ignore[attr-defined]
    sys.modules["dotenv"] = _dotenv

# Stub openai SDK so GeminiClient can construct without real SDK
if "openai" not in sys.modules:
    _openai = ModuleType("openai")
    _openai.OpenAI = MagicMock  # type: ignore[attr-defined]
    sys.modules["openai"] = _openai

from config.prompts import SOURCE_FORMAT_TEMPLATE, SYSTEM_PROMPT
from src.generation.prompt_builder import PromptBuilder
from src.generation.response_parser import (
    CitedResponse,
    CitedSource,
    ResponseParser,
    _tooltip_for,
)


# ── Fixtures ────────────────────────────────────────────────────────

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


# ── Tests: prompt building ──────────────────────────────────────────

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


# ── Tests: response parsing — extracting [SOURCE_N] references ──────

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
        assert result.citation_count == 2
        assert len(result.cited_sources) == 2

    def test_deduplicates_repeated_references(
        self, response_parser: ResponseParser
    ) -> None:
        text = "[SOURCE_1] says this. [SOURCE_1] confirms it."
        chunks = [{"chunk_id": "c1", "text": "t", "doc_type": "statute"}]
        result = response_parser.parse_response(text, chunks)
        assert result.citation_count == 1

    def test_no_references_found(
        self, response_parser: ResponseParser
    ) -> None:
        text = "No citations here."
        chunks = [{"chunk_id": "c1", "text": "t", "doc_type": "statute"}]
        result = response_parser.parse_response(text, chunks)
        assert result.citation_count == 0
        assert len(result.uncited_sources) == 1

    def test_answer_text_preserved(
        self, response_parser: ResponseParser
    ) -> None:
        text = "The answer is [SOURCE_1]."
        chunks = [{"chunk_id": "c1", "text": "t", "doc_type": "statute"}]
        result = response_parser.parse_response(text, chunks)
        assert result.answer_text == text


# ── Tests: citation validation ──────────────────────────────────────

class TestCitationValidation:

    def test_valid_reference(self, response_parser: ResponseParser) -> None:
        text = "See [SOURCE_1]."
        chunks = [{"chunk_id": "c1", "text": "t", "doc_type": "statute"}]
        result = response_parser.parse_response(text, chunks)
        assert len(result.invalid_citations) == 0
        assert result.citation_count == 1

    def test_invalid_reference_out_of_range(
        self, response_parser: ResponseParser
    ) -> None:
        text = "See [SOURCE_5]."
        chunks = [{"chunk_id": "c1", "text": "t", "doc_type": "statute"}]
        result = response_parser.parse_response(text, chunks)
        assert "[SOURCE_5]" in result.invalid_citations
        assert result.citation_count == 0

    def test_mixed_valid_and_invalid(
        self, response_parser: ResponseParser
    ) -> None:
        text = "See [SOURCE_1] and [SOURCE_99]."
        chunks = [
            {"chunk_id": "c1", "text": "t1", "doc_type": "statute"},
            {"chunk_id": "c2", "text": "t2", "doc_type": "judgement"},
        ]
        result = response_parser.parse_response(text, chunks)
        assert result.citation_count == 1
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


# ── Tests: CitedResponse model ──────────────────────────────────────

class TestCitedResponseModel:

    def test_construction(self) -> None:
        resp = CitedResponse(
            answer_text="Answer",
            answer_html="<p>Answer</p>",
            cited_sources=[
                CitedSource(
                    source_index=1,
                    chunk_id="c1",
                    text="source text",
                    metadata={"doc_type": "statute"},
                )
            ],
            citation_count=1,
        )
        assert resp.citation_count == 1
        assert len(resp.cited_sources) == 1
        assert resp.cited_sources[0].chunk_id == "c1"

    def test_defaults(self) -> None:
        resp = CitedResponse(answer_text="A", answer_html="A")
        assert resp.cited_sources == []
        assert resp.uncited_sources == []
        assert resp.invalid_citations == []
        assert resp.citation_count == 0

    def test_uncited_sources_tracked(
        self, response_parser: ResponseParser
    ) -> None:
        text = "Only [SOURCE_1] matters."
        chunks = [
            {"chunk_id": "c1", "text": "used", "doc_type": "statute"},
            {"chunk_id": "c2", "text": "unused", "doc_type": "judgement"},
            {"chunk_id": "c3", "text": "also unused", "doc_type": "constitution"},
        ]
        result = response_parser.parse_response(text, chunks)
        assert len(result.uncited_sources) == 2


# ── Tests: HTML generation ──────────────────────────────────────────

class TestHTMLGeneration:

    def test_valid_citations_become_badges(
        self, response_parser: ResponseParser
    ) -> None:
        text = "See [SOURCE_1]."
        chunks = [{"chunk_id": "c1", "text": "t", "doc_type": "statute"}]
        result = response_parser.parse_response(text, chunks)
        assert "citation-badge" in result.answer_html
        assert 'data-source-index="1"' in result.answer_html

    def test_invalid_citations_left_as_text(
        self, response_parser: ResponseParser
    ) -> None:
        text = "See [SOURCE_99]."
        chunks = [{"chunk_id": "c1", "text": "t", "doc_type": "statute"}]
        result = response_parser.parse_response(text, chunks)
        assert "[SOURCE_99]" in result.answer_html
        assert "citation-badge" not in result.answer_html


# ── Tests: tooltip generation ───────────────────────────────────────

class TestTooltips:

    def test_judgement_tooltip(self) -> None:
        chunk = {
            "doc_type": "judgement",
            "case_name": "State v. X",
            "citation": "AIR 2020 SC 100",
            "court": "Supreme Court of India",
        }
        tip = _tooltip_for(chunk)
        assert "State v. X" in tip
        assert "AIR 2020 SC 100" in tip

    def test_statute_tooltip(self) -> None:
        chunk = {
            "doc_type": "statute",
            "act_name": "IPC",
            "section_number": "302",
        }
        tip = _tooltip_for(chunk)
        assert "IPC" in tip
        assert "Section 302" in tip

    def test_constitution_tooltip(self) -> None:
        chunk = {"doc_type": "constitution", "section_number": "21"}
        tip = _tooltip_for(chunk)
        assert "Article 21" in tip

    def test_unknown_doc_type_tooltip(self) -> None:
        chunk = {"doc_type": "regulation"}
        tip = _tooltip_for(chunk)
        assert tip == "regulation"


# ── Tests: mock Gemini API ──────────────────────────────────────────

class TestGeminiClientMocked:

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
        from src.generation.gemini_client import GeminiClient

        client = GeminiClient(api_key="test-key")
        self._mock_completion(client, "According to [SOURCE_1], Section 302 IPC...")

        result = client.generate("What is Section 302?")
        assert result == "According to [SOURCE_1], Section 302 IPC..."

    def test_generate_with_system_prompt(self) -> None:
        from src.generation.gemini_client import GeminiClient

        client = GeminiClient(api_key="test-key")
        self._mock_completion(client, "response text")

        result = client.generate("query", system_prompt="You are a legal expert.")
        assert result == "response text"

    def test_generate_strips_thinking_tags(self) -> None:
        from src.generation.gemini_client import GeminiClient

        client = GeminiClient(api_key="test-key")
        self._mock_completion(
            client,
            "<think>I should reason privately.</think>\n"
            "Section 302 IPC punishes murder. [SOURCE_1]",
        )

        result = client.generate("query")
        assert result == "Section 302 IPC punishes murder. [SOURCE_1]"

    def test_generate_strips_visible_reasoning_prefix(self) -> None:
        from src.generation.gemini_client import GeminiClient

        client = GeminiClient(api_key="test-key")
        self._mock_completion(
            client,
            "Reasoning: I will compare the retrieved sources first.\n\n"
            "Final Answer: Section 302 IPC punishes murder. [SOURCE_1]",
        )

        result = client.generate("query")
        assert result == "Section 302 IPC punishes murder. [SOURCE_1]"

    def test_empty_prompt_raises(self) -> None:
        from src.generation.gemini_client import GeminiClient, GeminiClientError

        client = GeminiClient(api_key="test-key")
        with pytest.raises(GeminiClientError, match="Prompt must not be empty"):
            client.generate("")

    def test_no_api_key_raises(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            from src.generation.gemini_client import GeminiClient, GeminiClientError
            with pytest.raises(GeminiClientError, match="No API key"):
                GeminiClient()
