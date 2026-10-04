"""Citation-grounded prompt construction for Gemini.

Converts a user query and a list of retrieved chunks into a
(system_prompt, user_prompt) pair ready for the LLM, with every
chunk formatted as a numbered source using the templates defined
in ``config.prompts``.
"""

from __future__ import annotations

import logging
from typing import Optional

from config.prompts import (
    SOURCE_FORMAT_TEMPLATE,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)

# ── Metadata formatters keyed by doc_type ─────────────────────────

_METADATA_BUILDERS: dict[str, object] = {}  # populated below


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


def _default_metadata(chunk: dict) -> str:
    """Fallback metadata line for unknown doc_types."""
    return f"Document type: {chunk.get('doc_type', 'unknown')}"


class PromptBuilder:
    """Builds citation-grounded prompts from retrieved chunks.

    All methods are pure functions that return new values — no
    internal state is mutated.
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
            A ``(system_prompt, user_prompt)`` tuple ready for the
            Gemini client.
        """
        if not query:
            raise ValueError("Query must not be empty.")

        if not retrieved_chunks:
            logger.warning("No chunks provided — LLM will have no sources.")

        formatted_sources = "\n".join(
            self._format_source(index=i + 1, chunk=chunk)
            for i, chunk in enumerate(retrieved_chunks)
        )

        user_prompt = USER_PROMPT_TEMPLATE.format(
            formatted_sources=formatted_sources,
            query=query,
        )

        logger.debug(
            "Prompt built — %d sources, user prompt length %d",
            len(retrieved_chunks),
            len(user_prompt),
        )

        return SYSTEM_PROMPT, user_prompt

    # ── Internal helpers ──────────────────────────────────────────

    def _format_source(self, index: int, chunk: dict) -> str:
        """Format a single chunk as a numbered SOURCE block.

        Args:
            index: 1-based source number.
            chunk: Dict with ``text``, ``doc_type``, and optional
                   metadata fields (``case_name``, ``citation``,
                   ``court``, ``date``, ``act_name``,
                   ``section_number``).

        Returns:
            A string matching ``SOURCE_FORMAT_TEMPLATE``.
        """
        doc_type = chunk.get("doc_type", "unknown")
        metadata_line = self._build_metadata_line(doc_type, chunk)
        text = chunk.get("text", "")

        return SOURCE_FORMAT_TEMPLATE.format(
            index=index,
            doc_type=doc_type,
            metadata_line=metadata_line,
            text=text,
        )

    @staticmethod
    def _build_metadata_line(doc_type: str, chunk: dict) -> str:
        """Dispatch to the correct metadata formatter by doc_type."""
        formatters: dict[str, object] = {
            "judgement": _judgement_metadata,
            "statute": _statute_metadata,
            "constitution": _constitution_metadata,
        }
        formatter = formatters.get(doc_type, _default_metadata)
        return formatter(chunk)
