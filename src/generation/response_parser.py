"""Parse and validate LLM citation references.

Extracts ``[SOURCE_N]`` markers from the generated response,
maps them back to the original retrieved chunks, and produces
a structured ``CitedResponse`` with citation metadata.
"""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Regex for [SOURCE_N] markers
SOURCE_REF_PATTERN: re.Pattern[str] = re.compile(r"\[SOURCE_(\d+)\]")

# Grouped markers such as [SOURCE_2, SOURCE_4] or [SOURCE_1; SOURCE_3]
_GROUPED_REF_PATTERN: re.Pattern[str] = re.compile(
    r"\[\s*(SOURCE_\d+(?:\s*[,;]\s*SOURCE_\d+)+)\s*\]"
)


def expand_grouped_citations(text: str) -> str:
    """Rewrite ``[SOURCE_2, SOURCE_4]`` as ``[SOURCE_2] [SOURCE_4]``.

    The prompt asks for one marker per source, but models sometimes group
    them; left as is, the group would not be counted.
    """
    def _expand(match: re.Match[str]) -> str:
        refs = re.split(r"\s*[,;]\s*", match.group(1))
        return " ".join(f"[{ref}]" for ref in refs)

    return _GROUPED_REF_PATTERN.sub(_expand, text)


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
    cited_sources: list[CitedSource] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)
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
        response_text = expand_grouped_citations(response_text)
        referenced_indices = [int(ref) for ref in SOURCE_REF_PATTERN.findall(response_text)]

        def _is_valid(idx: int) -> bool:
            return 1 <= idx <= len(source_chunks)

        # Validity counts every reference, so repeated bad references still
        # lower the metric; the lists below hold each index once.
        valid_ref_count = sum(1 for idx in referenced_indices if _is_valid(idx))
        citation_validity = (
            valid_ref_count / len(referenced_indices) if referenced_indices else 0.0
        )

        cited_sources: list[CitedSource] = []
        invalid_citations: list[str] = []

        for idx in dict.fromkeys(referenced_indices):
            if not _is_valid(idx):
                tag = f"[SOURCE_{idx}]"
                invalid_citations.append(tag)
                logger.warning("Invalid citation reference: %s", tag)
                continue

            chunk = source_chunks[idx - 1]  # 1-based -> 0-based
            metadata = chunk.get("metadata", {})
            cited_sources.append(
                CitedSource(
                    source_index=idx,
                    chunk_id=chunk.get("chunk_id", f"chunk_{idx}"),
                    text=chunk.get("text", ""),
                    metadata=metadata if isinstance(metadata, dict) else {},
                )
            )

        uncited = len(source_chunks) - len(cited_sources)
        if uncited:
            logger.info(
                "%d of %d provided sources were not cited.",
                uncited,
                len(source_chunks),
            )

        return CitedResponse(
            answer_text=response_text,
            cited_sources=cited_sources,
            invalid_citations=invalid_citations,
            citation_validity=round(citation_validity, 4),
        )
