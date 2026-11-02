"""End-to-end query pipeline.

Wires together IPC/BNS lookup, retrieval, generation, citation parsing and
confidence scoring behind a single ``query()`` call.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

from config.settings import LLM_BACKEND
from src.confidence.scorer import ConfidenceResult, ConfidenceScorer
from src.generation.llm_client import LLMClient, LLMClientError
from src.generation.prompt_builder import PromptBuilder
from src.generation.response_parser import CitedSource, ResponseParser
from src.retrieval.ipc_bns import IpcBnsMapper
from src.retrieval.retrieval_pipeline import RetrievalPipeline

logger = logging.getLogger(__name__)


class PipelineResult(BaseModel):
    """Immutable result returned by :meth:`LegalResearchPipeline.query`.

    ``sources`` holds every passage numbered in the prompt, in order, so
    ``[SOURCE_N]`` in ``answer_text`` refers to ``sources[N - 1]``. When the
    IPC/BNS table matched, it is the last entry.
    """

    query: str
    answer_text: str = ""
    sources: list[dict[str, Any]] = Field(default_factory=list)
    cited_sources: list[CitedSource] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    confidence: Optional[ConfidenceResult] = None
    ipc_bns_mappings: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_count: int = 0
    error: Optional[str] = None

    model_config = {"frozen": True}


class LegalResearchPipeline:
    """Orchestrate the full legal-research query lifecycle.

    Stages
    ------
    0. **Map** - look up IPC/BNS section references in the query and expand
       the retrieval query with the counterpart section.
    1. **Retrieve** - hybrid search plus authority re-ranking.
    2. **Build prompt** - number the sources; the IPC/BNS table, if matched,
       is added as the last source so the model can cite it.
    3. **Generate** - call the LLM. Without an LLM client the pipeline stops
       here and returns the sources only.
    4. **Parse** - validate ``[SOURCE_N]`` citations.
    5. **Score** - compute confidence.
    """

    def __init__(
        self,
        retrieval_pipeline: Any,
        llm_client: Any = None,
        ipc_bns_mapper: Optional[IpcBnsMapper] = None,
    ) -> None:
        self._retrieval_pipeline = retrieval_pipeline
        self._llm_client = llm_client
        self._ipc_bns_mapper = ipc_bns_mapper
        self._prompt_builder = PromptBuilder()
        self._response_parser = ResponseParser()
        self._confidence_scorer = ConfidenceScorer()

    @classmethod
    def from_settings(cls) -> LegalResearchPipeline:
        """Build the pipeline from ``config.settings``: load the embedding
        model, open the vector store and, if an API key is set, the LLM.

        Raises ``RuntimeError`` when the vector store has not been built.
        """
        from src.embeddings.embedding_service import EmbeddingService
        from src.embeddings.vectorstore import VectorStore

        vectorstore = VectorStore()
        if vectorstore.count() == 0:
            raise RuntimeError(
                "The vector store is empty. Build it first (see README, "
                "'Building the knowledge base')."
            )
        retrieval = RetrievalPipeline(vectorstore.collection, EmbeddingService())

        llm_client: Any
        if LLM_BACKEND == "local":
            from src.generation.local_generator import LocalGenerator, LocalGeneratorError

            try:
                llm_client = LocalGenerator()
            except LocalGeneratorError as exc:
                logger.warning("Local LLM disabled: %s", exc)
                llm_client = None
        else:
            try:
                llm_client = LLMClient()
            except LLMClientError as exc:
                logger.warning("LLM disabled: %s", exc)
                llm_client = None

        try:
            mapper: Optional[IpcBnsMapper] = IpcBnsMapper()
        except (OSError, ValueError) as exc:
            logger.warning("IPC/BNS mapping unavailable: %s", exc)
            mapper = None

        return cls(retrieval, llm_client, mapper)

    @property
    def can_generate(self) -> bool:
        """``False`` when no LLM is configured and only sources are returned."""
        return self._llm_client is not None

    def query(self, user_query: str) -> PipelineResult:
        """Run the full pipeline for *user_query* and return a result.

        Retrieval and generation failures are reported in ``error`` along
        with whatever was gathered up to that point.
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
                logger.info("IPC/BNS matches: %s", [m["ipc_section"] for m in mappings])

        # 1. Retrieve
        try:
            retrieved_chunks = self._retrieval_pipeline.retrieve(retrieval_query)
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc, exc_info=True)
            return PipelineResult(
                query=user_query,
                ipc_bns_mappings=mappings,
                error=f"Retrieval failed: {exc}",
            )

        mapping_chunk = (
            self._ipc_bns_mapper.as_source_chunk(mappings)
            if self._ipc_bns_mapper is not None
            else None
        )
        sources = retrieved_chunks + ([mapping_chunk] if mapping_chunk else [])
        partial = {
            "query": user_query,
            "sources": sources,
            "ipc_bns_mappings": mappings,
            "retrieved_count": len(retrieved_chunks),
        }

        if not sources:
            return PipelineResult(**partial, error="No relevant documents found for the query.")

        if self._llm_client is None:
            return PipelineResult(
                **partial,
                error=(
                    "No LLM API key configured, so only the retrieved sources are "
                    "returned. Set NVIDIA_API_KEY in .env to get answers."
                ),
            )

        # 2. Build prompt and 3. generate
        system_prompt, user_prompt = self._prompt_builder.build_prompt(user_query, sources)
        try:
            raw_response = self._llm_client.generate(user_prompt, system_prompt=system_prompt)
        except Exception as exc:
            logger.error("Generation failed: %s", exc, exc_info=True)
            return PipelineResult(**partial, error=f"Generation failed: {exc}")

        # 4. Parse citations
        parsed = self._response_parser.parse_response(raw_response, sources)

        # 5. Score confidence on retrieved chunks only; the mapping table has
        # no retrieval score and would skew retrieval quality.
        confidence = self._confidence_scorer.score(retrieved_chunks, parsed)

        logger.info(
            "Pipeline complete - %d chunks retrieved, %d sources cited, confidence=%s.",
            len(retrieved_chunks),
            len(parsed.cited_sources),
            confidence.level,
        )
        return PipelineResult(
            **partial,
            answer_text=parsed.answer_text,
            cited_sources=parsed.cited_sources,
            invalid_citations=parsed.invalid_citations,
            confidence=confidence,
        )
