"""End-to-end query pipeline.

Wires together IPC/BNS lookup, retrieval, generation, citation parsing and
confidence scoring behind a single ``query()`` call.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.confidence.scorer import ConfidenceResult, ConfidenceScorer

logger = logging.getLogger(__name__)


# Result model


class PipelineResult(BaseModel):
    """Immutable result returned by :pymeth:`LegalResearchPipeline.query`."""

    query: str
    answer_text: str = ""
    answer_html: str = ""
    cited_sources: list[Any] = Field(default_factory=list)
    retrieved_chunks: list[Any] = Field(default_factory=list)
    confidence: Optional[ConfidenceResult] = None
    ipc_bns_mappings: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_count: int = 0
    cited_count: int = 0
    error: Optional[str] = None

    model_config = {"frozen": True}


# Pipeline


class LegalResearchPipeline:
    """Orchestrate the full legal-research query lifecycle.

    Stages
    ------
    0. **Map** - look up IPC/BNS section references in the query and expand
       the retrieval query with the counterpart section.
    1. **Retrieve** - hybrid search plus authority re-ranking.
    2. **Build prompt** - number the sources; the IPC/BNS table, if matched,
       is added as the last source so the model can cite it.
    3. **Generate** - call the LLM through ``LLMClient``.
    4. **Parse** - validate ``[SOURCE_N]`` citations with ``ResponseParser``.
    5. **Score** - compute confidence with ``ConfidenceScorer``.
    """

    def __init__(self) -> None:
        logger.info("Initialising LegalResearchPipeline...")

        self._embedding_service = _init_embedding_service()
        self._vectorstore = _init_vectorstore()
        self._retrieval_pipeline = _init_retrieval_pipeline(
            self._vectorstore,
            self._embedding_service,
        )
        self._ipc_bns_mapper = _init_ipc_bns_mapper()
        self._llm_client = _init_llm_client()
        self._prompt_builder = _init_prompt_builder()
        self._response_parser = _init_response_parser()
        self._confidence_scorer = ConfidenceScorer()

        logger.info("LegalResearchPipeline initialised successfully.")

    # Public API

    def query(self, user_query: str) -> PipelineResult:
        """Run the full pipeline for *user_query* and return a result.

        If any stage fails the pipeline returns a ``PipelineResult`` whose
        ``error`` field describes the failure while still carrying whatever
        partial data was gathered up to that point.

        Parameters
        ----------
        user_query:
            The natural-language legal question.

        Returns
        -------
        PipelineResult
            An immutable result containing the answer, citations,
            confidence score, and (optionally) an error message.
        """
        if not user_query or not user_query.strip():
            return PipelineResult(
                query=user_query or "",
                error="Query must not be empty.",
            )

        logger.info("Pipeline processing query: %s", user_query[:120])

        # 0. IPC/BNS lookup
        mappings: list[dict[str, Any]] = []
        retrieval_query = user_query
        if self._ipc_bns_mapper is not None:
            mappings = self._ipc_bns_mapper.find(user_query)
            if mappings:
                retrieval_query = self._ipc_bns_mapper.expand_query(user_query, mappings)
                logger.info("[0/5] IPC/BNS matches: %s", [m["ipc_section"] for m in mappings])

        # 1. Retrieve
        retrieved_chunks: list[dict[str, Any]] = []
        try:
            logger.info("[1/5] Retrieving relevant chunks ...")
            retrieved_chunks = self._retrieval_pipeline.retrieve(retrieval_query)
            logger.info(
                "[1/5] Retrieved %d chunks.",
                len(retrieved_chunks),
            )
        except Exception as exc:
            logger.error("[1/5] Retrieval failed: %s", exc, exc_info=True)
            return PipelineResult(
                query=user_query,
                error=f"Retrieval stage failed: {exc}",
            )

        mapping_chunk = (
            self._ipc_bns_mapper.as_source_chunk(mappings)
            if self._ipc_bns_mapper is not None
            else None
        )
        source_chunks = retrieved_chunks + ([mapping_chunk] if mapping_chunk else [])

        if not source_chunks:
            logger.warning("[1/5] No chunks retrieved, stopping.")
            return PipelineResult(
                query=user_query,
                error="No relevant documents found for the query.",
                retrieved_count=0,
            )

        # 2. Build prompt
        system_prompt: str = ""
        user_prompt: str = ""
        try:
            logger.info("[2/5] Building LLM prompt...")
            system_prompt, user_prompt = self._prompt_builder.build_prompt(
                user_query,
                source_chunks,
            )
            logger.info("[2/5] Prompt built (%d chars).", len(user_prompt))
        except Exception as exc:
            logger.error("[2/5] Prompt building failed: %s", exc, exc_info=True)
            return PipelineResult(
                query=user_query,
                retrieved_count=len(retrieved_chunks),
                error=f"Prompt-building stage failed: {exc}",
            )

        # 3. Generate
        if self._llm_client is None:
            return PipelineResult(
                query=user_query,
                retrieved_chunks=source_chunks,
                ipc_bns_mappings=mappings,
                retrieved_count=len(retrieved_chunks),
                error=(
                    "No LLM API key configured, so only the retrieved sources are "
                    "shown. Set NVIDIA_API_KEY in .env and restart the app."
                ),
            )

        raw_response: str = ""
        try:
            logger.info("[3/5] Generating response via LLM...")
            raw_response = self._llm_client.generate(
                user_prompt,
                system_prompt=system_prompt,
            )
            logger.info(
                "[3/5] Response generated (%d chars).",
                len(raw_response),
            )
        except Exception as exc:
            logger.error("[3/5] Generation failed: %s", exc, exc_info=True)
            return PipelineResult(
                query=user_query,
                retrieved_count=len(retrieved_chunks),
                error=f"Generation stage failed: {exc}",
            )

        # 4. Parse citations
        parsed_response: Any = None
        try:
            logger.info("[4/5] Parsing citations...")
            parsed_response = self._response_parser.parse_response(
                response_text=raw_response,
                source_chunks=source_chunks,
            )
            logger.info("[4/5] Citations parsed successfully.")
        except Exception as exc:
            logger.error("[4/5] Parsing failed: %s", exc, exc_info=True)
            # Graceful degradation: return the raw answer without citations
            return PipelineResult(
                query=user_query,
                answer_text=raw_response,
                answer_html=raw_response,
                retrieved_chunks=source_chunks,
                ipc_bns_mappings=mappings,
                retrieved_count=len(retrieved_chunks),
                error=f"Citation-parsing stage failed: {exc}",
            )

        # 5. Score confidence (on retrieved chunks only; the mapping table
        # has no retrieval score and would skew retrieval quality)
        confidence: Optional[ConfidenceResult] = None
        try:
            logger.info("[5/5] Scoring confidence...")
            confidence = self._confidence_scorer.score(
                retrieved_chunks=retrieved_chunks,
                cited_response=parsed_response,
            )
            logger.info(
                "[5/5] Confidence: %.2f (%s).",
                confidence.overall_score,
                confidence.level,
            )
        except Exception as exc:
            logger.error(
                "[5/5] Confidence scoring failed: %s",
                exc,
                exc_info=True,
            )
            # Non-fatal - we can still return the answer

        # Assemble final result
        answer_text = _extract_text(parsed_response, raw_response)
        answer_html = _extract_html(parsed_response, raw_response)
        cited_sources = _extract_cited_sources(parsed_response)

        result = PipelineResult(
            query=user_query,
            answer_text=answer_text,
            answer_html=answer_html,
            cited_sources=cited_sources,
            retrieved_chunks=source_chunks,
            confidence=confidence,
            ipc_bns_mappings=mappings,
            retrieved_count=len(retrieved_chunks),
            cited_count=len(cited_sources),
        )

        logger.info(
            "Pipeline complete - %d chunks retrieved, %d sources cited, confidence=%s.",
            result.retrieved_count,
            result.cited_count,
            confidence.level if confidence else "N/A",
        )
        return result

    def document_count(self) -> int | None:
        """Return the number of documents in the vector store, or ``None`` on error."""
        try:
            return self._vectorstore.count()
        except Exception as exc:
            logger.error("document_count failed: %s", exc, exc_info=True)
            return None

    def is_ready(self) -> bool:
        """Return ``True`` when the vector store contains at least one document."""
        try:
            count = self._vectorstore.count()
            ready = count > 0
            logger.info(
                "Pipeline readiness check: %d documents -> %s.",
                count,
                "ready" if ready else "not ready",
            )
            return ready
        except Exception as exc:
            logger.error("Readiness check failed: %s", exc, exc_info=True)
            return False


# Component initialisation helpers
# Each helper isolates the import so a missing module produces a clear
# error rather than crashing the entire import of this file.


def _init_embedding_service() -> Any:
    """Create and return an ``EmbeddingService`` instance."""
    from src.embeddings.embedding_service import EmbeddingService

    logger.info("Initialising EmbeddingService...")
    return EmbeddingService()


def _init_vectorstore() -> Any:
    """Create and return a ``VectorStore`` instance."""
    from src.embeddings.vectorstore import VectorStore

    logger.info("Initialising VectorStore...")
    return VectorStore()


def _init_retrieval_pipeline(vectorstore: Any, embedding_service: Any) -> Any:
    """Create and return a ``RetrievalPipeline`` instance."""
    from src.retrieval.retrieval_pipeline import RetrievalPipeline

    logger.info("Initialising RetrievalPipeline...")
    # HybridSearch expects the raw ChromaDB collection, not the VectorStore wrapper
    collection = getattr(vectorstore, "_collection", vectorstore)
    return RetrievalPipeline(
        vectorstore=collection,
        embedding_service=embedding_service,
    )


def _init_llm_client() -> Any:
    """Create an ``LLMClient``, or return ``None`` when no API key is set.

    Without a key the pipeline still retrieves and shows sources; it just
    cannot write an answer.
    """
    from src.generation.llm_client import LLMClient, LLMClientError

    logger.info("Initialising LLMClient ...")
    try:
        return LLMClient()
    except LLMClientError as exc:
        logger.warning("LLM disabled: %s", exc)
        return None


def _init_ipc_bns_mapper() -> Any:
    """Load the IPC/BNS table. The pipeline still works if the file is missing."""
    from src.retrieval.ipc_bns import IpcBnsMapper

    try:
        return IpcBnsMapper()
    except (OSError, ValueError) as exc:
        logger.warning("IPC/BNS mapping unavailable: %s", exc)
        return None


def _init_prompt_builder() -> Any:
    """Create and return a ``PromptBuilder`` instance."""
    from src.generation.prompt_builder import PromptBuilder

    logger.info("Initialising PromptBuilder...")
    return PromptBuilder()


def _init_response_parser() -> Any:
    """Create and return a ``ResponseParser`` instance."""
    from src.generation.response_parser import ResponseParser

    logger.info("Initialising ResponseParser...")
    return ResponseParser()


# Response extraction helpers


def _extract_text(parsed_response: Any, fallback: str) -> str:
    """Pull plain-text answer from a parsed response, falling back to *fallback*."""
    if parsed_response is None:
        return fallback
    if isinstance(parsed_response, dict):
        return parsed_response.get("answer_text", fallback)
    return getattr(parsed_response, "answer_text", fallback)


def _extract_html(parsed_response: Any, fallback: str) -> str:
    """Pull HTML answer from a parsed response, falling back to *fallback*."""
    if parsed_response is None:
        return fallback
    if isinstance(parsed_response, dict):
        return parsed_response.get("answer_html", fallback)
    return getattr(parsed_response, "answer_html", fallback)


def _extract_cited_sources(parsed_response: Any) -> list[Any]:
    """Pull the list of cited sources from a parsed response."""
    if parsed_response is None:
        return []
    if isinstance(parsed_response, dict):
        return list(parsed_response.get("cited_sources", []))
    return list(getattr(parsed_response, "cited_sources", []))
