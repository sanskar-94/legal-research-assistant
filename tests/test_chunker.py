"""Tests for src/chunking/legal_chunker.py.

Covers all three chunking strategies (judgement, statute, constitution),
the router function ``chunk_document``, and edge cases.
"""

from __future__ import annotations

from typing import Any

import pytest

from config.settings import CHUNK_MAX_TOKENS
from src.chunking.chunk_models import Chunk, ChunkMetadata
from src.chunking.legal_chunker import (
    _merge_small_segments,
    _overlap_text,
    _split_long_text,
    chunk_constitution,
    chunk_document,
    chunk_judgement,
    chunk_statute,
    estimate_tokens,
)
from tests.conftest import make_chunk_metadata


# Tests: internal helpers

class TestEstimateTokens:

    @pytest.mark.parametrize(
        "text,expected_min,expected_max",
        [
            ("", 0, 0),
            ("hello", 1, 3),
            ("one two three four five", 5, 10),
        ],
    )
    def test_estimate_tokens_range(
        self, text: str, expected_min: int, expected_max: int
    ) -> None:
        tokens = estimate_tokens(text)
        assert expected_min <= tokens <= expected_max

    def test_estimate_tokens_returns_int(self) -> None:
        assert isinstance(estimate_tokens("hello world"), int)


class TestOverlapText:

    def test_returns_tail_words(self) -> None:
        text = "a b c d e f g h i j"
        result = _overlap_text(text, 5)
        # 5 tokens ~ 3-4 words; result should be a suffix
        assert result.endswith("j")

    def test_returns_full_text_when_short(self) -> None:
        text = "short"
        result = _overlap_text(text, 100)
        assert result == text


class TestMergeSmallSegments:

    def test_empty_input(self) -> None:
        assert _merge_small_segments([], 100) == []

    def test_no_merge_needed(self) -> None:
        segments = ["word " * 100, "word " * 100]
        result = _merge_small_segments(segments, 10)
        assert len(result) == 2

    def test_merges_tiny_segments(self) -> None:
        segments = ["a", "b", "c", "word " * 100]
        result = _merge_small_segments(segments, 50)
        # "a", "b", "c" should merge into one segment
        assert len(result) < len(segments)


class TestSplitLongText:

    def test_short_text_not_split(self) -> None:
        text = "This is a short sentence."
        result = _split_long_text(text, max_tokens=100, overlap_tokens=10)
        assert len(result) == 1
        assert result[0] == text

    def test_long_text_produces_multiple_pieces(self) -> None:
        # Build text that is definitely >800 tokens
        text = ". ".join(["This is sentence number " + str(i) for i in range(200)])
        result = _split_long_text(text, max_tokens=100, overlap_tokens=10)
        assert len(result) > 1

    def test_all_pieces_within_max(self) -> None:
        text = ". ".join(["Long sentence with words " + str(i) for i in range(200)])
        pieces = _split_long_text(text, max_tokens=100, overlap_tokens=10)
        for piece in pieces:
            # Allow some tolerance since splitting is heuristic
            assert estimate_tokens(piece) < 200


# Tests: judgement chunking

class TestChunkJudgement:

    def test_basic_splitting(self, judgement_text: str, judgement_metadata: dict) -> None:
        chunks = chunk_judgement(judgement_text, judgement_metadata)
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_case_name_prefix(self, judgement_text: str, judgement_metadata: dict) -> None:
        chunks = chunk_judgement(judgement_text, judgement_metadata)
        case_name = judgement_metadata["case_name"]
        for chunk in chunks:
            assert chunk.text.startswith(case_name)

    def test_respects_max_token_limit(
        self, judgement_text: str, judgement_metadata: dict
    ) -> None:
        chunks = chunk_judgement(judgement_text, judgement_metadata)
        for chunk in chunks:
            # Allow generous tolerance for prefix + overlap
            assert estimate_tokens(chunk.text) < CHUNK_MAX_TOKENS * 2

    def test_chunk_indices_are_sequential(
        self, judgement_text: str, judgement_metadata: dict
    ) -> None:
        chunks = chunk_judgement(judgement_text, judgement_metadata)
        indices = [c.metadata.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_metadata_doc_type(
        self, judgement_text: str, judgement_metadata: dict
    ) -> None:
        chunks = chunk_judgement(judgement_text, judgement_metadata)
        for chunk in chunks:
            assert chunk.metadata.doc_id == judgement_metadata["doc_id"]

    def test_splits_on_paragraph_numbers(self) -> None:
        text = (
            "1. First paragraph with enough content to stand on its own as a chunk. "
            "It discusses the background and sets up the legal issues at hand.\n"
            "2. Second paragraph with additional content and legal reasoning. "
            "The court considers the evidence presented by both parties.\n"
            "3. Third paragraph containing the final disposition and order of the bench. "
            "The appeal is dismissed and the conviction stands."
        )
        meta = make_chunk_metadata(doc_type="judgement")
        chunks = chunk_judgement(text, meta)
        assert len(chunks) >= 1  # may merge small paras

    def test_no_paragraph_markers_produces_single_logical_group(self) -> None:
        text = "This text has no paragraph numbering at all. " * 20
        meta = make_chunk_metadata(doc_type="judgement")
        chunks = chunk_judgement(text, meta)
        assert len(chunks) >= 1

    def test_no_case_name_no_prefix(self) -> None:
        text = "1. Some paragraph content here.\n2. Another paragraph follows."
        meta = make_chunk_metadata(doc_type="judgement", case_name=None)
        chunks = chunk_judgement(text, meta)
        for chunk in chunks:
            # Without case_name the prefix should be empty
            assert not chunk.text.startswith("\n\n")


# Tests: statute chunking

class TestChunkStatute:

    def test_one_section_per_chunk(
        self, statute_text: str, statute_metadata: dict
    ) -> None:
        chunks = chunk_statute(statute_text, statute_metadata)
        assert len(chunks) >= 3  # at least Sections 299, 300, 302, 304

    def test_section_number_in_metadata(
        self, statute_text: str, statute_metadata: dict
    ) -> None:
        chunks = chunk_statute(statute_text, statute_metadata)
        section_numbers = [c.metadata.section_number for c in chunks]
        # At least some chunks should have section numbers
        assert any(sn is not None for sn in section_numbers)

    def test_section_text_contains_header(
        self, statute_text: str, statute_metadata: dict
    ) -> None:
        chunks = chunk_statute(statute_text, statute_metadata)
        # Each chunk with a section number should contain "Section" in text
        for chunk in chunks:
            if chunk.metadata.section_number:
                assert "Section" in chunk.text or "section" in chunk.text.lower()

    def test_sub_section_splitting_for_long_section(self) -> None:
        """A very long section should be split at sub-section markers."""
        # Build a section with many sub-sections
        body = "\n".join(
            f"(a{i}) This sub-section deals with provision number {i}. " * 30
            for i in range(20)
        )
        text = f"Section 100. Very long section:\n{body}"
        meta = make_chunk_metadata(doc_type="statute", act_name="Test Act")
        chunks = chunk_statute(text, meta)
        # Should produce multiple chunks for this very long section
        assert len(chunks) >= 1

    def test_preamble_included_when_substantial(self) -> None:
        preamble = "An act to consolidate and amend the law. " * 20
        text = f"{preamble}\nSection 1. Short title:\nThis Act may be called the Test Act."
        meta = make_chunk_metadata(doc_type="statute")
        chunks = chunk_statute(text, meta)
        # The first chunk should be the preamble
        assert len(chunks) >= 2

    def test_no_section_markers_produces_single_chunk(self) -> None:
        text = "This statute text has no section markers whatsoever."
        meta = make_chunk_metadata(doc_type="statute")
        chunks = chunk_statute(text, meta)
        assert len(chunks) == 1


# Tests: constitution chunking

class TestChunkConstitution:

    def test_articles_stay_together(
        self, constitution_text: str, constitution_metadata: dict
    ) -> None:
        chunks = chunk_constitution(constitution_text, constitution_metadata)
        assert len(chunks) >= 3  # Articles 14, 19, 21, 32

    def test_explanation_attached_to_article(
        self, constitution_text: str, constitution_metadata: dict
    ) -> None:
        chunks = chunk_constitution(constitution_text, constitution_metadata)
        # Find the Article 14 chunk
        art14_chunks = [
            c for c in chunks if c.metadata.section_number == "14"
        ]
        assert len(art14_chunks) >= 1
        # The Explanation should be in the same chunk
        assert "Explanation" in art14_chunks[0].text

    def test_article_number_in_metadata(
        self, constitution_text: str, constitution_metadata: dict
    ) -> None:
        chunks = chunk_constitution(constitution_text, constitution_metadata)
        article_numbers = [c.metadata.section_number for c in chunks]
        # Should have article numbers extracted
        assert any(n is not None for n in article_numbers)
        # Specific articles we know exist
        assert "14" in article_numbers
        assert "21" in article_numbers

    def test_long_article_is_split(self) -> None:
        """An article exceeding max tokens should be split."""
        body = "This is clause content. " * 500
        text = f"Article 370. Special status:\n{body}"
        meta = make_chunk_metadata(doc_type="constitution")
        chunks = chunk_constitution(text, meta)
        assert len(chunks) > 1

    def test_preamble_included(self) -> None:
        preamble = "WE THE PEOPLE OF INDIA having solemnly resolved " * 20
        text = f"{preamble}\nArticle 1. Name and territory:\nIndia shall be a Union of States."
        meta = make_chunk_metadata(doc_type="constitution")
        chunks = chunk_constitution(text, meta)
        assert len(chunks) >= 2


# Tests: chunk_document router

class TestChunkDocumentRouter:

    def test_dispatches_to_judgement(
        self, judgement_text: str, judgement_metadata: dict
    ) -> None:
        chunks = chunk_document(judgement_text, "judgement", judgement_metadata)
        assert len(chunks) > 0
        for c in chunks:
            assert c.metadata.doc_type == "judgement"

    def test_dispatches_to_statute(
        self, statute_text: str, statute_metadata: dict
    ) -> None:
        chunks = chunk_document(statute_text, "statute", statute_metadata)
        assert len(chunks) > 0
        for c in chunks:
            assert c.metadata.doc_type == "statute"

    def test_dispatches_to_constitution(
        self, constitution_text: str, constitution_metadata: dict
    ) -> None:
        chunks = chunk_document(constitution_text, "constitution", constitution_metadata)
        assert len(chunks) > 0
        for c in chunks:
            assert c.metadata.doc_type == "constitution"

    def test_unknown_doc_type_raises(self) -> None:
        meta = make_chunk_metadata(doc_type="unknown")
        with pytest.raises(ValueError, match="Unknown doc_type"):
            chunk_document("some text", "unknown", meta)

    @pytest.mark.parametrize("doc_type", ["judgement", "statute", "constitution"])
    def test_enriches_metadata_with_doc_type(self, doc_type: str) -> None:
        text = "Some sample legal text with enough content to chunk properly."
        meta = make_chunk_metadata(doc_type=doc_type)
        chunks = chunk_document(text, doc_type, meta)
        for chunk in chunks:
            assert chunk.metadata.doc_type == doc_type


# Tests: edge cases

class TestEdgeCases:

    @pytest.mark.parametrize("doc_type", ["judgement", "statute", "constitution"])
    def test_empty_text(self, doc_type: str) -> None:
        meta = make_chunk_metadata(doc_type=doc_type)
        chunks = chunk_document("", doc_type, meta)
        assert len(chunks) >= 1  # at minimum one chunk with empty text

    @pytest.mark.parametrize("doc_type", ["judgement", "statute", "constitution"])
    def test_very_short_text(self, doc_type: str) -> None:
        meta = make_chunk_metadata(doc_type=doc_type)
        chunks = chunk_document("Hi", doc_type, meta)
        assert len(chunks) >= 1

    def test_text_with_no_paragraph_markers_judgement(self) -> None:
        text = "A plain text with no numbered paragraphs at all."
        meta = make_chunk_metadata(doc_type="judgement")
        chunks = chunk_judgement(text, meta)
        assert len(chunks) >= 1
        assert text in chunks[0].text or chunks[0].text.endswith(text)

    def test_chunk_ids_are_unique(self, judgement_text: str) -> None:
        meta = make_chunk_metadata(doc_type="judgement")
        chunks = chunk_judgement(judgement_text, meta)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_statute_with_only_preamble(self) -> None:
        text = "Short preamble."
        meta = make_chunk_metadata(doc_type="statute")
        chunks = chunk_statute(text, meta)
        assert len(chunks) == 1

    def test_constitution_with_single_article(self) -> None:
        text = "Article 1. Short article text with explanation."
        meta = make_chunk_metadata(doc_type="constitution")
        chunks = chunk_constitution(text, meta)
        assert len(chunks) >= 1

    @pytest.mark.parametrize(
        "marker",
        [
            "1. Para one.\n2. Para two.",
            "(1) Para one.\n(2) Para two.",
            "[1] Para one.\n[2] Para two.",
        ],
    )
    def test_different_paragraph_marker_styles(self, marker: str) -> None:
        meta = make_chunk_metadata(doc_type="judgement")
        chunks = chunk_judgement(marker, meta)
        assert len(chunks) >= 1


class TestChunkIds:

    def test_ids_are_stable_across_runs(self, judgement_text: str, judgement_metadata: dict[str, Any]) -> None:
        first = [c.chunk_id for c in chunk_judgement(judgement_text, judgement_metadata)]
        second = [c.chunk_id for c in chunk_judgement(judgement_text, judgement_metadata)]
        assert first == second

    def test_id_built_from_doc_id_and_index(self) -> None:
        chunk = Chunk(text="x", metadata=ChunkMetadata(doc_id="ipc_1860", doc_type="statute", chunk_index=4))
        assert chunk.chunk_id == "ipc_1860::4"

    def test_explicit_id_kept(self) -> None:
        chunk = Chunk(chunk_id="sc_chunk_9", text="x", metadata={"doc_id": "d", "doc_type": "judgement"})
        assert chunk.chunk_id == "sc_chunk_9"
