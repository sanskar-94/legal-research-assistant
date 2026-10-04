"""Parse and validate LLM citation references.

Extracts ``[SOURCE_N]`` markers from the generated response,
maps them back to the original retrieved chunks, and produces
a structured ``CitedResponse`` with citation metadata.
"""

from __future__ import annotations

import html
import logging
import re
from typing import Optional

from pydantic import BaseModel, Field

from src.generation.prompt_builder import flatten_chunk

logger = logging.getLogger(__name__)

# Regex for [SOURCE_N] markers
_SOURCE_REF_PATTERN: re.Pattern[str] = re.compile(
    r"\[SOURCE_(\d+)\]"
)


# Pydantic v2 models

class CitedSource(BaseModel):
    """A single source that was cited in the LLM response."""

    source_index: int
    chunk_id: str
    text: str
    metadata: dict


class CitedResponse(BaseModel):
    """Structured representation of an LLM answer with citations."""

    answer_text: str
    answer_html: str
    cited_sources: list[CitedSource] = Field(default_factory=list)
    uncited_sources: list[dict] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
    citation_count: int = 0
    citation_validity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of [SOURCE_N] tags that point to provided chunks.",
    )


# Parser

class ResponseParser:
    """Extracts, validates, and structures citation data from LLM output.

    The parser holds no state.
    """

    def parse_response(
        self,
        response_text: str,
        source_chunks: list[dict],
    ) -> CitedResponse:
        """Parse ``[SOURCE_N]`` references from the LLM response.

        Args:
            response_text:  Raw text returned by the LLM.
            source_chunks:  The same ordered list of chunk dicts that
                            was passed to ``PromptBuilder.build_prompt``.
                            Each dict should contain at least ``text``,
                            ``doc_type``, and ideally ``chunk_id``.

        Returns:
            A fully populated ``CitedResponse`` model.
        """
        # 1. Find all [SOURCE_N] references
        raw_refs = _SOURCE_REF_PATTERN.findall(response_text)
        referenced_indices: list[int] = []
        for ref in raw_refs:
            try:
                referenced_indices.append(int(ref))
            except ValueError:
                continue

        # Deduplicate while preserving first-occurrence order
        seen: set[int] = set()
        unique_indices: list[int] = []
        for idx in referenced_indices:
            if idx not in seen:
                seen.add(idx)
                unique_indices.append(idx)

        # 2. Classify each unique reference as valid or invalid for rendering.
        # Citation validity below is computed from raw refs, so repeated bad
        # references still lower the metric.
        valid_indices: list[int] = []
        invalid_citations: list[str] = []

        for idx in unique_indices:
            if 1 <= idx <= len(source_chunks):
                valid_indices.append(idx)
            else:
                tag = f"[SOURCE_{idx}]"
                invalid_citations.append(tag)
                logger.warning("Invalid citation reference: %s", tag)

        valid_ref_count = sum(
            1 for idx in referenced_indices if 1 <= idx <= len(source_chunks)
        )
        citation_validity = (
            valid_ref_count / len(referenced_indices) if referenced_indices else 0.0
        )

        # 3. Build CitedSource objects for valid references
        cited_sources: list[CitedSource] = []
        cited_index_set: set[int] = set()

        for idx in valid_indices:
            chunk = source_chunks[idx - 1]  # 1-based -> 0-based
            cited_index_set.add(idx)

            # Extract the document metadata dict (flat, not nested)
            raw_meta = chunk.get("metadata", {})
            if not isinstance(raw_meta, dict):
                raw_meta = {}

            # Merge in scoring data so the UI has access if needed
            metadata = {
                **raw_meta,
                "chunk_id": chunk.get("chunk_id", f"chunk_{idx}"),
                "semantic_score": chunk.get("semantic_score"),
                "final_score": chunk.get("final_score"),
            }

            cited_sources.append(
                CitedSource(
                    source_index=idx,
                    chunk_id=chunk.get("chunk_id", f"chunk_{idx}"),
                    text=chunk.get("text", ""),
                    metadata=metadata,
                )
            )

        # 4. Identify uncited sources
        uncited_sources: list[dict] = [
            {**chunk, "_source_index": i + 1}
            for i, chunk in enumerate(source_chunks)
            if (i + 1) not in cited_index_set
        ]

        if uncited_sources:
            logger.info(
                "%d of %d provided sources were not cited.",
                len(uncited_sources),
                len(source_chunks),
            )

        # 5. Build HTML with citation badges
        answer_html = self._build_html(
            response_text, source_chunks,
        )

        return CitedResponse(
            answer_text=response_text,
            answer_html=answer_html,
            cited_sources=cited_sources,
            uncited_sources=uncited_sources,
            invalid_citations=invalid_citations,
            citation_count=len(cited_sources),
            citation_validity=round(citation_validity, 4),
        )

    # Internal helpers

    @staticmethod
    def _build_html(
        response_text: str,
        source_chunks: list[dict],
    ) -> str:
        """Replace ``[SOURCE_N]`` markers with clickable HTML badges.

        Valid references become styled ``<span>`` badges;
        invalid references are left as plain text.
        """

        def _replacer(match: re.Match[str]) -> str:
            idx = int(match.group(1))
            if 1 <= idx <= len(source_chunks):
                chunk = source_chunks[idx - 1]
                tooltip = html.escape(_tooltip_for(chunk), quote=True)
                return (
                    f'<span class="citation-badge" '
                    f'data-source-index="{idx}" '
                    f'title="{tooltip}">'
                    f"[SOURCE_{idx}]"
                    f"</span>"
                )
            # Invalid reference - leave unchanged
            return match.group(0)

        return _SOURCE_REF_PATTERN.sub(_replacer, response_text)


def _tooltip_for(chunk: dict) -> str:
    """Generate a human-readable tooltip for a citation badge."""
    chunk = flatten_chunk(chunk)
    doc_type = chunk.get("doc_type", "unknown")

    if doc_type == "judgement":
        parts = [
            chunk.get("case_name", ""),
            chunk.get("citation", ""),
            chunk.get("court", ""),
        ]
        return " | ".join(p for p in parts if p)

    if doc_type == "statute":
        parts = [
            chunk.get("act_name", ""),
            f"Section {chunk['section_number']}"
            if chunk.get("section_number")
            else "",
        ]
        return " | ".join(p for p in parts if p)

    if doc_type == "constitution":
        section = chunk.get("section_number", "")
        return f"Article {section}" if section else "Constitution"

    if doc_type == "mapping":
        return "IPC / BNS correspondence table"

    return doc_type
