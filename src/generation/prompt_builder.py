"""Citation-grounded prompt construction.

Converts a user query and a list of retrieved chunks into a
(system_prompt, user_prompt) pair ready for the LLM, with every
chunk formatted as a numbered source using the templates defined
in ``config.prompts``.
"""

from __future__ import annotations

import logging

from config.prompts import (
    SOURCE_FORMAT_TEMPLATE,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)


def flatten_chunk(chunk: dict) -> dict:
    """Return a copy of *chunk* with its nested ``metadata`` lifted to the top level.

    Retrieval results keep document fields (``case_name``, ``court``,
    ``act_name``...) under ``chunk["metadata"]``, while the formatters below
    read them from the top level. Top-level keys win on conflict, so
    scores set by the retriever are not overwritten.
    """
    metadata = chunk.get("metadata")
    if not isinstance(metadata, dict):
        return dict(chunk)
    flat = {**metadata, **{k: v for k, v in chunk.items() if k != "metadata"}}
    flat["metadata"] = metadata
    return flat


# Metadata formatters keyed by doc_type


def _judgement_metadata(chunk: dict) -> str:
    """Build the metadata line for a judgement chunk."""
    parts: list[str] = []
    if chunk.get("case_name"):
        parts.append(f"Case: {chunk['case_name']}")
    if chunk.get("citation"):
        parts.append(f"Citation: {chunk['citation']}")
    if chunk.get("court"):
        parts.append(f"Court: {chunk['court']}")
    if chunk.get("date"):
        parts.append(f"Date: {chunk['date']}")
    return " | ".join(parts) if parts else "Judgement (details unavailable)"


def _statute_metadata(chunk: dict) -> str:
    """Build the metadata line for a statute chunk."""
    parts: list[str] = []
    if chunk.get("act_name"):
        parts.append(f"Act: {chunk['act_name']}")
    if chunk.get("section_number"):
        parts.append(f"Section: {chunk['section_number']}")
    return " | ".join(parts) if parts else "Statute (details unavailable)"


def _constitution_metadata(chunk: dict) -> str:
    """Build the metadata line for a constitution chunk."""
    if chunk.get("section_number"):
        return f"Article: {chunk['section_number']}"
    return "Constitution (article number unavailable)"


def _mapping_metadata(chunk: dict) -> str:
    """Build the metadata line for the IPC/BNS correspondence table."""
    return f"Correspondence table: {chunk.get('act_name', 'IPC / BNS')}"


def _default_metadata(chunk: dict) -> str:
    """Fallback metadata line for unknown doc_types."""
    return f"Document type: {chunk.get('doc_type', 'unknown')}"


_METADATA_FORMATTERS = {
    "judgement": _judgement_metadata,
    "statute": _statute_metadata,
    "constitution": _constitution_metadata,
    "mapping": _mapping_metadata,
}


def source_label(chunk: dict) -> str:
    """One-line description of a retrieved chunk, e.g. ``"Act: IPC | Section: 302"``."""
    flat = flatten_chunk(chunk)
    formatter = _METADATA_FORMATTERS.get(flat.get("doc_type"), _default_metadata)
    return formatter(flat)


class PromptBuilder:
    """Builds citation-grounded prompts from retrieved chunks.

    The builder holds no state; the same inputs always give the same prompt.
    """

    def build_prompt(
        self,
        query: str,
        retrieved_chunks: list[dict],
    ) -> tuple[str, str]:
        """Assemble the full (system_prompt, user_prompt) pair.

        Args:
            query:            The user's legal research question.
            retrieved_chunks: Ordered list of chunk dicts, each
                              containing at least ``text`` and
                              ``doc_type`` keys plus optional
                              metadata fields.

        Returns:
            A ``(system_prompt, user_prompt)`` tuple ready for the LLM client.
        """
        if not query:
            raise ValueError("Query must not be empty.")

        if not retrieved_chunks:
            logger.warning("No chunks provided - LLM will have no sources.")

        formatted_sources = "\n".join(
            SOURCE_FORMAT_TEMPLATE.format(
                index=i + 1,
                doc_type=flatten_chunk(chunk).get("doc_type") or "unknown",
                metadata_line=source_label(chunk),
                text=chunk.get("text", ""),
            )
            for i, chunk in enumerate(retrieved_chunks)
        )

        user_prompt = USER_PROMPT_TEMPLATE.format(
            formatted_sources=formatted_sources,
            query=query,
        )

        logger.debug(
            "Prompt built - %d sources, user prompt length %d",
            len(retrieved_chunks),
            len(user_prompt),
        )

        return SYSTEM_PROMPT, user_prompt
