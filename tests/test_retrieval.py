"""Tests for src/retrieval/ - query classification, RRF scoring,
authority scoring, weight adjustment, and hybrid search.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

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
            ("Explain the ruling in Vishaka v State of Rajasthan", "case_lookup"),
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

    def test_section_suffix_normalised(self) -> None:
        assert _extract_keyword_terms("what is section 304a ipc") == ["Section 304A"]

    @pytest.mark.parametrize(
        "query,expected",
        [
            ("What was held in Maneka Gandhi v Union of India?", ["Maneka Gandhi"]),
            ("Summarize Kesavananda Bharati v. State of Kerala", ["Kesavananda Bharati"]),
            ("What was held in Indira Nehru Gandhi v Raj Narain?", ["Indira Nehru Gandhi", "Raj Narain"]),
            ("Explain the MC Mehta v Union of India case", ["MC Mehta", "M.C. Mehta"]),
            ("K.S. Puttaswamy versus Union of India", ["K.S. Puttaswamy"]),
        ],
    )
    def test_case_party_names(self, query: str, expected: list[str]) -> None:
        assert _extract_keyword_terms(query) == expected

    def test_generic_parties_only(self) -> None:
        assert _extract_keyword_terms("State of Punjab v Union of India") == []

    def test_statute_comparison_is_not_a_case(self) -> None:
        query = "Compare defamation provisions in IPC vs BNS"
        assert _extract_keyword_terms(query) == ["IPC", "BNS"]
        assert HybridSearch.classify_query(query) == "section_lookup"



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
        semantic = {
            "ids": [["c1"]],
            "documents": [["text"]],
            "metadatas": [[{"doc_type": "statute"}]],
            "distances": [[0.2]],
        }

        def fake_query(**kwargs):
            if "where_document" in kwargs:
                raise Exception("keyword search error")
            return semantic

        store = MagicMock()
        store.query.side_effect = fake_query

        hs = HybridSearch(store, mock_embedding_service)
        results = hs.search("Section 302 IPC")
        # Should still return semantic results even if keyword fails
        assert [r["chunk_id"] for r in results] == ["c1"]


class TestHybridSearchScoring:

    @staticmethod
    def _result(chunk_id: str, text: str, doc_type: str, distance: float) -> dict:
        return {
            "ids": [[chunk_id]],
            "documents": [[text]],
            "metadatas": [[{"doc_type": doc_type}]],
            "distances": [[distance]],
        }

    def _store(self) -> MagicMock:
        """Semantic search finds sem_1; an exact match on "Section 302" finds
        kw_1, and one on "Section 30" finds a passage that only mentions
        Section 302, which must be dropped."""
        empty = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        def fake_query(query_embeddings, n_results, include, where_document=None):
            if where_document is None:
                return self._result("sem_1", "Article 21 protects life and personal liberty.", "constitution", 0.25)
            term = where_document["$contains"]
            if term == "Section 302":
                return self._result("kw_1", "Section 302. Punishment for murder.", "statute", 0.4)
            if term == "Section 30":
                return self._result("kw_2", "Section 302. Punishment for murder.", "statute", 0.4)
            return empty

        store = MagicMock()
        store.query.side_effect = fake_query
        return store

    def _embedder(self) -> MagicMock:
        embedder = MagicMock()
        embedder.embed_query.return_value = [0.6, 0.8]
        return embedder

    def _searched_terms(self, store: MagicMock) -> list[str]:
        return [
            c.kwargs["where_document"]["$contains"]
            for c in store.query.call_args_list
            if "where_document" in c.kwargs
        ]

    def test_cosine_distance_converted_to_similarity(self) -> None:
        results = HybridSearch(self._store(), self._embedder()).search("right to life")
        sem = next(r for r in results if r["chunk_id"] == "sem_1")
        assert sem["semantic_score"] == pytest.approx(0.75)

    def test_keyword_hit_gets_semantic_score(self) -> None:
        results = HybridSearch(self._store(), self._embedder()).search("section 302 ipc")
        kw = next(r for r in results if r["chunk_id"] == "kw_1")
        assert kw["semantic_score"] == pytest.approx(0.6)

    def test_partial_identifier_match_dropped(self) -> None:
        results = HybridSearch(self._store(), self._embedder()).search("Section 30 IPC")
        assert all(r["chunk_id"] != "kw_2" for r in results)

    def test_lowercase_query_matches_title_case_text(self) -> None:
        store = self._store()
        HybridSearch(store, self._embedder()).search("section 304a")
        assert self._searched_terms(store) == ["Section 304A", "section 304A"]

    def test_case_name_searched_as_written(self) -> None:
        store = self._store()
        HybridSearch(store, self._embedder()).search("What was held in Maneka Gandhi v Union of India?")
        assert self._searched_terms(store) == ["Maneka Gandhi"]

    def test_conceptual_query_skips_keyword_path(self) -> None:
        store = self._store()
        HybridSearch(store, self._embedder()).search("explain the doctrine of basic structure")
        assert self._searched_terms(store) == []


class TestRerankerRrfNormalisation:

    def test_top_ranked_rrf_counts_fully(self) -> None:
        reranker = AuthorityReranker()
        top = {"chunk_id": "a", "semantic_score": 0.0, "rrf_score": 1.0 / (RRF_K + 1), "metadata": {}}
        none = {"chunk_id": "b", "semantic_score": 0.0, "rrf_score": 0.0, "metadata": {}}
        ranked = reranker.rerank([none, top], top_k=2)
        assert ranked[0]["chunk_id"] == "a"
        assert ranked[0]["final_score"] - ranked[1]["final_score"] == pytest.approx(0.25)


class TestDuplicateRemoval:

    def test_identical_texts_collapsed(self) -> None:
        from src.retrieval.retrieval_pipeline import _drop_duplicate_texts

        candidates = [
            {"chunk_id": "a", "text": "Article 21 protects life."},
            {"chunk_id": "b", "text": "Article 21  protects\nlife."},
            {"chunk_id": "c", "text": "Article 22 protects against arrest."},
        ]
        assert [c["chunk_id"] for c in _drop_duplicate_texts(candidates)] == ["a", "c"]


class TestWholeTermMatch:

    def test_longer_number_not_matched(self) -> None:
        from src.retrieval.hybrid_search import _whole_term_pattern

        pattern = _whole_term_pattern("Article 21")
        assert pattern.search("under article 21, every person")
        assert not pattern.search("Article 210 deals with language")
        assert not pattern.search("Article 21A on education")


class TestFuseHits:

    def test_weighted_rrf_formula(self) -> None:
        from src.retrieval.hybrid_search import fuse_hits

        sem = [{"chunk_id": "a", "semantic_score": 0.9}, {"chunk_id": "b", "semantic_score": 0.8}]
        kw = [{"chunk_id": "b", "semantic_score": 0.8}, {"chunk_id": "c", "semantic_score": 0.5}]
        fused = {h["chunk_id"]: h["rrf_score"] for h in fuse_hits(sem, kw, 0.3, 0.7, k=60)}
        assert fused["a"] == pytest.approx(0.3 / 61)
        assert fused["b"] == pytest.approx(0.3 / 62 + 0.7 / 61)
        assert fused["c"] == pytest.approx(0.7 / 62)

    def test_search_uses_fuse_hits(self, mock_vectorstore: MagicMock, mock_embedding_service: MagicMock) -> None:
        hs = HybridSearch(mock_vectorstore, mock_embedding_service)
        paths = hs.search_paths("Section 302 IPC", top_k=20)
        assert paths["classification"] == "section_lookup"
        assert [h["chunk_id"] for h in paths["semantic_hits"]] == ["chunk_1", "chunk_2", "chunk_3"]
        results = hs.search("Section 302 IPC")
        assert results[0]["rrf_score"] == pytest.approx(0.3 / (RRF_K + 1) + 0.7 / (RRF_K + 1))


class TestRerankerParameters:

    def test_custom_weights(self) -> None:
        chunk = {"chunk_id": "a", "semantic_score": 0.8, "rrf_score": 0.0, "metadata": {"doc_type": "statute"}}
        default = AuthorityReranker().rerank([chunk], top_k=1)[0]["final_score"]
        assert default == pytest.approx(0.5 * 0.8 + 0.25 * 1.0)
        semantic_only = AuthorityReranker().rerank([chunk], top_k=1, weights=(1.0, 0.0, 0.0))
        assert semantic_only[0]["final_score"] == pytest.approx(0.8)

    def test_rrf_rescaled_with_given_k(self) -> None:
        chunk = {"chunk_id": "a", "semantic_score": 0.0, "rrf_score": 0.5 / 11, "metadata": {"doc_type": "statute"}}
        scored = AuthorityReranker().rerank([chunk], top_k=1, weights=(0.0, 1.0, 0.0), rrf_k=10)
        assert scored[0]["final_score"] == pytest.approx(0.5)


class TestCourtWeights:

    def test_unknown_court_class_gets_default(self) -> None:
        # eq. 5 has four court classes; a court string that is none of them is "unknown"
        acts = AuthorityReranker.compute_authority_score(
            {"doc_type": "judgement", "court": "Appellate Authority under the Act"}
        )
        unknown = AuthorityReranker.compute_authority_score({"doc_type": "judgement", "court": ""})
        assert acts == unknown == pytest.approx(0.4 * 0.3 + 0.3 * 0.2 + 0.3 * 0.5)

    def test_tribunal_abbreviation_needs_whole_word(self) -> None:
        assert _normalise_court("Satara District Court") != "tribunal"
        assert _normalise_court("ITAT Mumbai") == "tribunal"

    @pytest.mark.parametrize(
        "court,expected",
        [("Supreme Court of India", 1.0), ("Delhi High Court", 0.7), ("NCLT", 0.4), (None, 0.3)],
    )
    def test_court_weight_values(self, court: str | None, expected: float) -> None:
        score = AuthorityReranker.compute_authority_score(
            {"doc_type": "judgement", "court": court, "citation_count": 100}
        )
        # undated -> recency 0.2; 100 citations -> 1.0
        assert score == pytest.approx(0.4 * expected + 0.3 * 0.2 + 0.3 * 1.0)
