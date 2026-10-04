"""Integration tests for the full pipeline flow.

Since there is no standalone ``src/pipeline.py`` module yet, these tests
exercise the end-to-end pipeline by composing the retrieval, generation,
and parsing stages together with all external services mocked.
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

if "dotenv" not in sys.modules:
    _dotenv = ModuleType("dotenv")
    _dotenv.load_dotenv = lambda *a, **kw: None  # type: ignore[attr-defined]
    sys.modules["dotenv"] = _dotenv

from config.settings import CONFIDENCE_WEIGHTS
from src.generation.prompt_builder import PromptBuilder
from src.generation.response_parser import CitedResponse, ResponseParser
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.reranker import AuthorityReranker


# ── Pipeline helper ─────────────────────────────────────────────────

def run_pipeline(
    query: str,
    vectorstore: Any,
    embedding_service: Any,
    gemini_generate: Any,
    top_k: int = 6,
) -> dict[str, Any]:
    """Execute the full retrieval-generation pipeline.

    This function mimics what a ``pipeline.py`` module would do:
    1. Hybrid search
    2. Re-rank
    3. Build prompt
    4. Generate response (via Gemini)
    5. Parse citations
    6. Compute confidence
    """
    # Stage 1: retrieval
    hybrid = HybridSearch(vectorstore, embedding_service)
    raw_results = hybrid.search(query, top_k=20)

    # Stage 2: re-rank
    reranker = AuthorityReranker()
    reranked = reranker.rerank(raw_results, top_k=top_k)

    # Stage 3: build prompt
    builder = PromptBuilder()
    sys_prompt, user_prompt = builder.build_prompt(query, reranked)

    # Stage 4: generate
    raw_response = gemini_generate(user_prompt, system_prompt=sys_prompt)

    # Stage 5: parse
    parser = ResponseParser()
    cited_response = parser.parse_response(raw_response, reranked)

    # Stage 6: confidence scoring
    confidence = _compute_confidence(cited_response, reranked)

    return {
        "query": query,
        "cited_response": cited_response,
        "reranked_chunks": reranked,
        "confidence": confidence,
        "raw_response": raw_response,
    }


def _compute_confidence(
    cited_response: CitedResponse,
    reranked: list[dict[str, Any]],
) -> float:
    """Compute an aggregate confidence score in [0, 1].

    Weights are drawn from ``config.settings.CONFIDENCE_WEIGHTS``:
    - retrieval_quality:  mean of final_scores from reranked chunks
    - source_agreement:   fraction of sources that were cited
    - authority_strength: mean authority_score of cited sources
    - coverage_score:     1.0 if any citations exist, else 0.0
    """
    if not reranked:
        return 0.0

    retrieval_quality = sum(
        r.get("final_score", 0.0) for r in reranked
    ) / len(reranked)

    total_sources = len(reranked)
    cited_count = cited_response.citation_count
    source_agreement = cited_count / total_sources if total_sources else 0.0

    authority_scores = [
        r.get("authority_score", 0.0) for r in reranked
        if r.get("chunk_id") in {cs.chunk_id for cs in cited_response.cited_sources}
    ]
    authority_strength = (
        sum(authority_scores) / len(authority_scores) if authority_scores else 0.0
    )

    coverage = 1.0 if cited_count > 0 else 0.0

    score = (
        CONFIDENCE_WEIGHTS["retrieval_quality"] * retrieval_quality
        + CONFIDENCE_WEIGHTS["source_agreement"] * source_agreement
        + CONFIDENCE_WEIGHTS["authority_strength"] * authority_strength
        + CONFIDENCE_WEIGHTS["coverage_score"] * coverage
    )
    return round(min(max(score, 0.0), 1.0), 4)


# ── Fixtures ────────────────────────────────────────────────────────

def _make_rich_vectorstore() -> MagicMock:
    """Build a vectorstore mock with realistic legal chunks."""
    store = MagicMock()
    store.query.return_value = {
        "ids": [["c1", "c2", "c3", "c4"]],
        "documents": [[
            "Section 302 IPC: Whoever commits murder shall be punished with death or life imprisonment.",
            "In State of Maharashtra v. Prakash, the conviction under Section 302 was upheld by the Supreme Court.",
            "Article 21 of the Constitution guarantees right to life and personal liberty.",
            "The doctrine of basic structure was established in Kesavananda Bharati v State of Kerala.",
        ]],
        "metadatas": [[
            {"doc_type": "statute", "section_number": "302", "act_name": "IPC"},
            {"doc_type": "judgement", "court": "Supreme Court of India", "case_name": "State of Maharashtra v. Prakash", "date": "2020-03-15"},
            {"doc_type": "constitution", "section_number": "21"},
            {"doc_type": "judgement", "court": "Supreme Court of India", "case_name": "Kesavananda Bharati v State of Kerala", "date": "1973-04-24"},
        ]],
        "distances": [[0.1, 0.2, 0.4, 0.5]],
    }
    store.get.return_value = {
        "ids": ["c1", "c2"],
        "documents": [
            "Section 302 IPC: Whoever commits murder shall be punished with death or life imprisonment.",
            "In State of Maharashtra v. Prakash, the conviction under Section 302 was upheld.",
        ],
        "metadatas": [
            {"doc_type": "statute", "section_number": "302", "act_name": "IPC"},
            {"doc_type": "judgement", "court": "Supreme Court of India"},
        ],
    }
    return store


@pytest.fixture()
def rich_vectorstore() -> MagicMock:
    return _make_rich_vectorstore()


@pytest.fixture()
def empty_vectorstore() -> MagicMock:
    store = MagicMock()
    store.query.return_value = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    store.get.return_value = {"ids": [], "documents": [], "metadatas": []}
    return store


@pytest.fixture()
def mock_gemini() -> MagicMock:
    def fake_generate(prompt: str, system_prompt: str | None = None) -> str:
        return (
            "According to [SOURCE_1], Section 302 of IPC provides punishment "
            "for murder which is death or life imprisonment. "
            "The Supreme Court in [SOURCE_2] upheld the conviction. "
            "Article 21 [SOURCE_3] protects the right to life."
        )
    return MagicMock(side_effect=fake_generate)


# ── Tests: full pipeline flow ───────────────────────────────────────

class TestFullPipelineFlow:

    def test_pipeline_returns_expected_keys(
        self,
        rich_vectorstore: MagicMock,
        mock_embedding_service: MagicMock,
        mock_gemini: MagicMock,
    ) -> None:
        result = run_pipeline(
            "What is Section 302 IPC?",
            rich_vectorstore,
            mock_embedding_service,
            mock_gemini,
        )
        assert "query" in result
        assert "cited_response" in result
        assert "reranked_chunks" in result
        assert "confidence" in result
        assert "raw_response" in result

    def test_pipeline_produces_citations(
        self,
        rich_vectorstore: MagicMock,
        mock_embedding_service: MagicMock,
        mock_gemini: MagicMock,
    ) -> None:
        result = run_pipeline(
            "What is Section 302 IPC?",
            rich_vectorstore,
            mock_embedding_service,
            mock_gemini,
        )
        cited = result["cited_response"]
        assert isinstance(cited, CitedResponse)
        assert cited.citation_count > 0

    def test_pipeline_cited_response_has_html(
        self,
        rich_vectorstore: MagicMock,
        mock_embedding_service: MagicMock,
        mock_gemini: MagicMock,
    ) -> None:
        result = run_pipeline(
            "What is Section 302 IPC?",
            rich_vectorstore,
            mock_embedding_service,
            mock_gemini,
        )
        assert "citation-badge" in result["cited_response"].answer_html

    def test_pipeline_reranked_sorted(
        self,
        rich_vectorstore: MagicMock,
        mock_embedding_service: MagicMock,
        mock_gemini: MagicMock,
    ) -> None:
        result = run_pipeline(
            "What is Section 302 IPC?",
            rich_vectorstore,
            mock_embedding_service,
            mock_gemini,
        )
        scores = [r["final_score"] for r in result["reranked_chunks"]]
        assert scores == sorted(scores, reverse=True)


# ── Tests: pipeline with empty vectorstore ──────────────────────────

class TestPipelineEmptyVectorstore:

    def test_empty_results_zero_confidence(
        self,
        empty_vectorstore: MagicMock,
        mock_embedding_service: MagicMock,
    ) -> None:
        def no_sources_response(prompt: str, system_prompt: str | None = None) -> str:
            return "The provided sources do not contain sufficient information."

        result = run_pipeline(
            "What is Section 99999 IPC?",
            empty_vectorstore,
            mock_embedding_service,
            MagicMock(side_effect=no_sources_response),
        )
        assert result["confidence"] == 0.0
        assert result["cited_response"].citation_count == 0

    def test_empty_reranked_list(
        self,
        empty_vectorstore: MagicMock,
        mock_embedding_service: MagicMock,
    ) -> None:
        result = run_pipeline(
            "test",
            empty_vectorstore,
            mock_embedding_service,
            MagicMock(return_value="No info"),
        )
        assert result["reranked_chunks"] == []


# ── Tests: confidence scoring ───────────────────────────────────────

class TestConfidenceScoring:

    def test_confidence_between_zero_and_one(
        self,
        rich_vectorstore: MagicMock,
        mock_embedding_service: MagicMock,
        mock_gemini: MagicMock,
    ) -> None:
        result = run_pipeline(
            "What is Section 302 IPC?",
            rich_vectorstore,
            mock_embedding_service,
            mock_gemini,
        )
        assert 0.0 <= result["confidence"] <= 1.0

    def test_higher_citation_coverage_means_higher_confidence(
        self,
        rich_vectorstore: MagicMock,
        mock_embedding_service: MagicMock,
    ) -> None:
        def all_cited(prompt: str, system_prompt: str | None = None) -> str:
            return "[SOURCE_1] [SOURCE_2] [SOURCE_3] [SOURCE_4]"

        def few_cited(prompt: str, system_prompt: str | None = None) -> str:
            return "[SOURCE_1] only."

        result_all = run_pipeline(
            "Section 302 IPC?",
            rich_vectorstore,
            mock_embedding_service,
            MagicMock(side_effect=all_cited),
        )
        # Rebuild the vectorstore mock so it has fresh call counts
        result_few = run_pipeline(
            "Section 302 IPC?",
            _make_rich_vectorstore(),
            mock_embedding_service,
            MagicMock(side_effect=few_cited),
        )
        assert result_all["confidence"] >= result_few["confidence"]

    def test_confidence_zero_for_empty_results(self) -> None:
        resp = CitedResponse(answer_text="no data", answer_html="no data")
        score = _compute_confidence(resp, [])
        assert score == 0.0

    def test_confidence_with_authority(self) -> None:
        from src.generation.response_parser import CitedSource

        resp = CitedResponse(
            answer_text="[SOURCE_1]",
            answer_html="[SOURCE_1]",
            cited_sources=[
                CitedSource(
                    source_index=1,
                    chunk_id="c1",
                    text="Section 302 IPC",
                    metadata={"doc_type": "statute"},
                ),
            ],
            citation_count=1,
        )
        reranked = [
            {
                "chunk_id": "c1",
                "final_score": 0.8,
                "authority_score": 1.0,
            },
        ]
        score = _compute_confidence(resp, reranked)
        assert 0.0 < score <= 1.0


# ── Tests: error handling at each stage ─────────────────────────────

class TestPipelineErrorHandling:

    def test_embedding_service_failure(
        self,
        rich_vectorstore: MagicMock,
    ) -> None:
        bad_embedder = MagicMock()
        bad_embedder.embed_query.side_effect = RuntimeError("Model load failed")

        with pytest.raises(RuntimeError, match="Model load failed"):
            run_pipeline(
                "test",
                rich_vectorstore,
                bad_embedder,
                MagicMock(return_value="ok"),
            )

    def test_vectorstore_query_failure(
        self,
        mock_embedding_service: MagicMock,
    ) -> None:
        bad_store = MagicMock()
        bad_store.query.side_effect = RuntimeError("Connection refused")

        with pytest.raises(RuntimeError, match="Connection refused"):
            run_pipeline(
                "test",
                bad_store,
                mock_embedding_service,
                MagicMock(return_value="ok"),
            )

    def test_gemini_failure(
        self,
        rich_vectorstore: MagicMock,
        mock_embedding_service: MagicMock,
    ) -> None:
        bad_gemini = MagicMock(
            side_effect=RuntimeError("Gemini API unavailable")
        )
        with pytest.raises(RuntimeError, match="Gemini API unavailable"):
            run_pipeline(
                "What is Section 302 IPC?",
                rich_vectorstore,
                mock_embedding_service,
                bad_gemini,
            )

    def test_prompt_builder_empty_query(
        self,
        rich_vectorstore: MagicMock,
        mock_embedding_service: MagicMock,
    ) -> None:
        with pytest.raises(ValueError, match="Query must not be empty"):
            run_pipeline(
                "",
                rich_vectorstore,
                mock_embedding_service,
                MagicMock(return_value="ok"),
            )
