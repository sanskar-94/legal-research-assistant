"""Tests for src/retrieval/ - query classification, RRF scoring,
authority scoring, weight adjustment, and hybrid search.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# Path bootstrap
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import RRF_K
from src.retrieval.hybrid_search import (
    HybridSearch,
    _CLASSIFICATION_WEIGHTS,
    _extract_keyword_terms,
    _reciprocal_rank_fusion,
)
from src.retrieval.reranker import (
    AuthorityReranker,
    _normalise_court,
    _recency_weight,
    _years_since,
)


# Tests: query classification

class TestQueryClassification:

    @pytest.mark.parametrize(
        "query,expected",
        [
            ("What is Section 302 IPC?", "section_lookup"),
            ("Define Section 498A of IPC", "section_lookup"),
            ("What does Article 21 say?", "section_lookup"),
            ("Explain Article 14 of the Constitution", "section_lookup"),
            ("Section 375 of the IPC", "section_lookup"),
            ("What is the Code of Criminal Procedure?", "section_lookup"),
        ],
    )
    def test_section_lookup(self, query: str, expected: str) -> None:
        assert HybridSearch.classify_query(query) == expected

    @pytest.mark.parametrize(
        "query,expected",
        [
            ("Summarize Kesavananda Bharati v. State of Kerala", "case_lookup"),
            ("What was held in Maneka Gandhi vs Union of India?", "case_lookup"),
            ("AIR 2017 SC 4161 ruling", "case_lookup"),
            ("Explain the case (2017) 10 SCC 1", "case_lookup"),
        ],
    )
    def test_case_lookup(self, query: str, expected: str) -> None:
        assert HybridSearch.classify_query(query) == expected

    @pytest.mark.parametrize(
        "query,expected",
        [
            ("Explain doctrine of basic structure", "conceptual"),
            ("What is the right to privacy in Indian law?", "conceptual"),
            ("Define natural justice", "conceptual"),
            ("Explain judicial review", "conceptual"),
            ("What are fundamental rights?", "conceptual"),
        ],
    )
    def test_conceptual(self, query: str, expected: str) -> None:
        assert HybridSearch.classify_query(query) == expected

    @pytest.mark.parametrize(
        "query,expected",
        [
            (
                "What was held about Section 302 IPC in State v. Accused?",
                "mixed",
            ),
            (
                "How does Article 21 apply in Maneka Gandhi vs Union case?",
                "mixed",
            ),
        ],
    )
    def test_mixed(self, query: str, expected: str) -> None:
        assert HybridSearch.classify_query(query) == expected


# Tests: keyword extraction

class TestKeywordExtraction:

    def test_extracts_section_numbers(self) -> None:
        terms = _extract_keyword_terms("What is Section 302 of IPC?")
        assert any("Section 302" in t for t in terms)
        # "IPC" alone would match most of the corpus, so it is dropped here
        assert "IPC" not in terms

    def test_act_abbreviation_used_when_no_section(self) -> None:
        assert _extract_keyword_terms("Offences against women under the IPC") == ["IPC"]

    def test_extracts_article_numbers(self) -> None:
        terms = _extract_keyword_terms("Article 21 of the Constitution")
        assert any("Article 21" in t for t in terms)

    def test_extracts_citations(self) -> None:
        terms = _extract_keyword_terms("See AIR 2017 SC 4161")
        assert any("AIR" in t for t in terms)

    def test_deduplicates(self) -> None:
        terms = _extract_keyword_terms("Section 302 IPC and Section 302 IPC again")
        section_terms = [t for t in terms if "Section 302" in t]
        assert len(section_terms) == 1

    def test_empty_query(self) -> None:
        assert _extract_keyword_terms("") == []

    def test_no_legal_terms(self) -> None:
        terms = _extract_keyword_terms("tell me about something random")
        assert terms == []


# Tests: Reciprocal Rank Fusion

class TestRRFScoreCalculation:

    def test_single_list(self) -> None:
        scores = _reciprocal_rank_fusion([["a", "b", "c"]])
        assert scores["a"] == pytest.approx(1.0 / (RRF_K + 1))
        assert scores["b"] == pytest.approx(1.0 / (RRF_K + 2))
        assert scores["c"] == pytest.approx(1.0 / (RRF_K + 3))

    def test_two_lists_overlap(self) -> None:
        scores = _reciprocal_rank_fusion([
            ["a", "b", "c"],
            ["b", "a", "d"],
        ])
        # "a" appears at rank 1 in list 1 and rank 2 in list 2
        expected_a = 1.0 / (RRF_K + 1) + 1.0 / (RRF_K + 2)
        assert scores["a"] == pytest.approx(expected_a)

    def test_empty_lists(self) -> None:
        scores = _reciprocal_rank_fusion([])
        assert scores == {}

    def test_disjoint_lists(self) -> None:
        scores = _reciprocal_rank_fusion([["a", "b"], ["c", "d"]])
        assert "a" in scores
        assert "c" in scores
        assert len(scores) == 4

    def test_higher_rank_gets_higher_score(self) -> None:
        scores = _reciprocal_rank_fusion([["a", "b", "c"]])
        assert scores["a"] > scores["b"] > scores["c"]

    def test_custom_k_value(self) -> None:
        scores = _reciprocal_rank_fusion([["a"]], k=10)
        assert scores["a"] == pytest.approx(1.0 / (10 + 1))


# Tests: authority scoring

class TestAuthorityScoring:

    def test_statute_gets_max_score(self) -> None:
        score = AuthorityReranker.compute_authority_score({"doc_type": "statute"})
        assert score == 1.0

    def test_constitution_gets_max_score(self) -> None:
        score = AuthorityReranker.compute_authority_score({"doc_type": "constitution"})
        assert score == 1.0

    def test_supreme_court_higher_than_high_court(self) -> None:
        sc_score = AuthorityReranker.compute_authority_score({
            "doc_type": "judgement",
            "court": "Supreme Court of India",
            "date": "2024-01-01",
        })
        hc_score = AuthorityReranker.compute_authority_score({
            "doc_type": "judgement",
            "court": "Delhi High Court",
            "date": "2024-01-01",
        })
        assert sc_score > hc_score

    def test_recent_case_higher_than_old(self) -> None:
        recent = AuthorityReranker.compute_authority_score({
            "doc_type": "judgement",
            "court": "Supreme Court of India",
            "date": "2025-01-01",
        })
        old = AuthorityReranker.compute_authority_score({
            "doc_type": "judgement",
            "court": "Supreme Court of India",
            "date": "1990-01-01",
        })
        assert recent > old

    def test_score_is_bounded(self) -> None:
        score = AuthorityReranker.compute_authority_score({
            "doc_type": "judgement",
            "court": "Unknown Court",
            "date": "1900-01-01",
        })
        assert 0.0 <= score <= 1.0

    def test_empty_metadata(self) -> None:
        score = AuthorityReranker.compute_authority_score({})
        assert 0.0 <= score <= 1.0

    @pytest.mark.parametrize(
        "court_raw,expected_key",
        [
            ("Supreme Court of India", "supreme_court"),
            ("Delhi High Court", "high_court"),
            ("High Court of Bombay", "high_court"),
            ("NCLT", "tribunal"),
            ("National Green Tribunal", "tribunal"),
            ("", ""),
            (None, ""),
        ],
    )
    def test_court_normalisation(
        self, court_raw: str | None, expected_key: str
    ) -> None:
        assert _normalise_court(court_raw) == expected_key


# Tests: recency weighting

class TestRecencyWeight:

    def test_very_recent(self) -> None:
        w = _recency_weight(1)
        assert w == 1.0

    def test_moderate_age(self) -> None:
        w = _recency_weight(7)
        assert 0.5 <= w <= 0.7

    def test_old(self) -> None:
        w = _recency_weight(25)
        assert w == 0.2

    def test_none_age(self) -> None:
        w = _recency_weight(None)
        assert w == 0.2  # RECENCY_DEFAULT


# Tests: weight adjustment based on query type

class TestWeightAdjustment:

    def test_section_lookup_favours_keyword(self) -> None:
        sem_w, kw_w = _CLASSIFICATION_WEIGHTS["section_lookup"]
        assert kw_w > sem_w

    def test_conceptual_favours_semantic(self) -> None:
        sem_w, kw_w = _CLASSIFICATION_WEIGHTS["conceptual"]
        assert sem_w > kw_w

    def test_mixed_is_balanced(self) -> None:
        sem_w, kw_w = _CLASSIFICATION_WEIGHTS["mixed"]
        assert sem_w == kw_w

    def test_case_lookup_weights(self) -> None:
        sem_w, kw_w = _CLASSIFICATION_WEIGHTS["case_lookup"]
        assert kw_w > sem_w

    def test_all_weights_sum_to_one(self) -> None:
        for classification, (sem_w, kw_w) in _CLASSIFICATION_WEIGHTS.items():
            assert sem_w + kw_w == pytest.approx(1.0), (
                f"Weights for '{classification}' do not sum to 1.0"
            )


# Tests: reranker

class TestReranker:

    def test_rerank_empty_list(self) -> None:
        reranker = AuthorityReranker()
        assert reranker.rerank([]) == []

    def test_rerank_returns_limited_results(self) -> None:
        reranker = AuthorityReranker()
        results = [
            {
                "chunk_id": f"c{i}",
                "text": f"text {i}",
                "metadata": {"doc_type": "judgement", "court": "Supreme Court of India"},
                "semantic_score": 0.9 - i * 0.1,
                "rrf_score": 0.01,
            }
            for i in range(20)
        ]
        reranked = reranker.rerank(results, top_k=5)
        assert len(reranked) == 5

    def test_rerank_adds_authority_and_final_score(self) -> None:
        reranker = AuthorityReranker()
        results = [
            {
                "chunk_id": "c1",
                "text": "test",
                "metadata": {"doc_type": "statute"},
                "semantic_score": 0.8,
                "rrf_score": 0.01,
            }
        ]
        reranked = reranker.rerank(results)
        assert "authority_score" in reranked[0]
        assert "final_score" in reranked[0]

    def test_rerank_preserves_original_fields(self) -> None:
        reranker = AuthorityReranker()
        results = [
            {
                "chunk_id": "c1",
                "text": "test",
                "metadata": {"doc_type": "statute"},
                "semantic_score": 0.8,
                "rrf_score": 0.01,
                "extra_field": "preserved",
            }
        ]
        reranked = reranker.rerank(results)
        assert reranked[0]["extra_field"] == "preserved"

    def test_rerank_sorted_descending(self) -> None:
        reranker = AuthorityReranker()
        results = [
            {
                "chunk_id": "low",
                "text": "t",
                "metadata": {"doc_type": "judgement", "court": "District Court"},
                "semantic_score": 0.1,
                "rrf_score": 0.001,
            },
            {
                "chunk_id": "high",
                "text": "t",
                "metadata": {"doc_type": "statute"},
                "semantic_score": 0.9,
                "rrf_score": 0.02,
            },
        ]
        reranked = reranker.rerank(results, top_k=10)
        assert reranked[0]["final_score"] >= reranked[1]["final_score"]


# Tests: hybrid search (mocked vectorstore)

class TestHybridSearch:

    def test_search_returns_results(
        self,
        mock_vectorstore: MagicMock,
        mock_embedding_service: MagicMock,
    ) -> None:
        hs = HybridSearch(mock_vectorstore, mock_embedding_service)
        results = hs.search("What is Section 302 IPC?")
        assert len(results) > 0

    def test_search_results_have_rrf_score(
        self,
        mock_vectorstore: MagicMock,
        mock_embedding_service: MagicMock,
    ) -> None:
        hs = HybridSearch(mock_vectorstore, mock_embedding_service)
        results = hs.search("Section 302 IPC")
        for r in results:
            assert "rrf_score" in r
            assert isinstance(r["rrf_score"], float)

    def test_search_sorted_by_rrf_score(
        self,
        mock_vectorstore: MagicMock,
        mock_embedding_service: MagicMock,
    ) -> None:
        hs = HybridSearch(mock_vectorstore, mock_embedding_service)
        results = hs.search("Section 302 IPC")
        scores = [r["rrf_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_with_empty_results(
        self,
        mock_embedding_service: MagicMock,
    ) -> None:
        empty_store = MagicMock()
        empty_store.query.return_value = {"ids": [[]]}
        empty_store.get.return_value = {"ids": []}

        hs = HybridSearch(empty_store, mock_embedding_service)
        results = hs.search("some query")
        assert results == []

    def test_search_calls_embedding_service(
        self,
        mock_vectorstore: MagicMock,
        mock_embedding_service: MagicMock,
    ) -> None:
        hs = HybridSearch(mock_vectorstore, mock_embedding_service)
        hs.search("test query")
        mock_embedding_service.embed_query.assert_called_once_with("test query")

    def test_keyword_search_failure_handled(
        self,
        mock_embedding_service: MagicMock,
    ) -> None:
        store = MagicMock()
        store.query.return_value = {
            "ids": [["c1"]],
            "documents": [["text"]],
            "metadatas": [[{"doc_type": "statute"}]],
            "distances": [[0.2]],
        }
        store.get.side_effect = Exception("keyword search error")

        hs = HybridSearch(store, mock_embedding_service)
        results = hs.search("Section 302 IPC")
        # Should still return semantic results even if keyword fails
        assert len(results) > 0


class TestHybridSearchScoring:

    def _store(self) -> MagicMock:
        store = MagicMock()
        store.query.return_value = {
            "ids": [["sem_1"]],
            "documents": [["Article 21 protects life and personal liberty."]],
            "metadatas": [[{"doc_type": "constitution"}]],
            "distances": [[0.25]],
        }

        def fake_get(where_document, include, limit):
            if where_document["$contains"] == "Section 302":
                return {
                    "ids": ["kw_1"],
                    "documents": ["Section 302. Punishment for murder."],
                    "metadatas": [{"doc_type": "statute"}],
                    "embeddings": [[1.0, 0.0]],
                }
            return {"ids": [], "documents": [], "metadatas": [], "embeddings": []}

        store.get.side_effect = fake_get
        return store

    def _embedder(self) -> MagicMock:
        embedder = MagicMock()
        embedder.embed_query.return_value = [0.6, 0.8]
        return embedder

    def test_cosine_distance_converted_to_similarity(self) -> None:
        results = HybridSearch(self._store(), self._embedder()).search("right to life")
        sem = next(r for r in results if r["chunk_id"] == "sem_1")
        assert sem["semantic_score"] == pytest.approx(0.75)

    def test_keyword_hit_gets_semantic_score(self) -> None:
        results = HybridSearch(self._store(), self._embedder()).search("section 302 ipc")
        kw = next(r for r in results if r["chunk_id"] == "kw_1")
        # cosine([0.6, 0.8], [1, 0]) = 0.6
        assert kw["semantic_score"] == pytest.approx(0.6)

    def test_lowercase_query_matches_title_case_text(self) -> None:
        store = self._store()
        HybridSearch(store, self._embedder()).search("section 302")
        searched = [c.kwargs["where_document"]["$contains"] for c in store.get.call_args_list]
        assert "Section 302" in searched

    def test_conceptual_query_skips_keyword_path(self) -> None:
        store = self._store()
        HybridSearch(store, self._embedder()).search("explain the doctrine of basic structure")
        store.get.assert_not_called()


class TestRerankerRrfNormalisation:

    def test_top_ranked_rrf_counts_fully(self) -> None:
        reranker = AuthorityReranker()
        top = {"chunk_id": "a", "semantic_score": 0.0, "rrf_score": 1.0 / (RRF_K + 1), "metadata": {}}
        none = {"chunk_id": "b", "semantic_score": 0.0, "rrf_score": 0.0, "metadata": {}}
        ranked = reranker.rerank([none, top], top_k=2)
        assert ranked[0]["chunk_id"] == "a"
        assert ranked[0]["final_score"] - ranked[1]["final_score"] == pytest.approx(0.25)
