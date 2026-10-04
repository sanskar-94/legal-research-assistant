"""Parser for Indian statutes, splitting by section or article markers.

Handles IPC/BNS (Section N), Constitution (Article N), and other
Indian statutes. Each section/article becomes a standalone unit
with metadata for downstream chunking and retrieval.
"""

from __future__ import annotations

import logging
import re
from typing import Final

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class StatuteSection(BaseModel):
    """A single section or article from a statute."""

    act_name: str = Field(description="Name of the statute or act")
    section_number: str = Field(
        description="Section or Article number (e.g. '302', '21A', '14(1)')"
    )
    section_title: str = Field(
        default="",
        description="Title/heading of the section, if available",
    )
    text: str = Field(description="Full text of the section or article")

    model_config = {"frozen": True}


# ── Section Marker Patterns ─────────────────────────────────────────

# Pattern for "Section <number>" style (IPC, BNS, CrPC, etc.)
_SECTION_MARKER: Final[re.Pattern[str]] = re.compile(
    r"(?:^|\n)"
    r"(?:"
    r"(?:Section|SECTION|Sec\.?|SEC\.?)\s+"   # "Section 302" / "Sec. 302"
    r"(\d+[A-Za-z]?(?:\([A-Za-z0-9]+\))?)"   # number with optional sub-clause
    r"|"
    r"(\d+[A-Za-z]?)\.\s+"                     # "302. Punishment for murder"
    r")"
    r"\.?\s*[-—.]?\s*"
    r"(.*?)$",                                  # optional title on same line
    re.MULTILINE,
)

# Pattern for "Article <number>" style (Constitution)
_ARTICLE_MARKER: Final[re.Pattern[str]] = re.compile(
    r"(?:^|\n)"
    r"(?:Article|ARTICLE|Art\.?|ART\.?)\s+"
    r"(\d+[A-Za-z]?(?:\([A-Za-z0-9]+\))?)"
    r"\.?\s*[-—.]?\s*"
    r"(.*?)$",
    re.MULTILINE,
)

# Known Constitution-related act names
_CONSTITUTION_NAMES: Final[frozenset[str]] = frozenset({
    "constitution",
    "constitution of india",
    "indian constitution",
    "the constitution of india",
})


def _is_constitution(act_name: str) -> bool:
    """Check if the act name refers to the Constitution."""
    return act_name.strip().lower() in _CONSTITUTION_NAMES


def _split_by_pattern(
    text: str,
    pattern: re.Pattern[str],
    act_name: str,
    is_article: bool,
) -> list[StatuteSection]:
    """Split text into sections using a regex pattern.

    Finds all marker positions, then assigns each section the text
    between its marker and the next marker (or end of document).

    Args:
        text: Full statute text.
        pattern: Compiled regex for section/article markers.
        act_name: Name of the act for metadata.
        is_article: True if parsing articles, False for sections.

    Returns:
        List of StatuteSection models.
    """
    matches = list(pattern.finditer(text))
    if not matches:
        logger.debug(
            "No %s markers found for '%s'",
            "article" if is_article else "section",
            act_name,
        )
        return []

    sections: list[StatuteSection] = []

    for i, match in enumerate(matches):
        if is_article:
            section_number = match.group(1)
            title = match.group(2).strip() if match.group(2) else ""
        else:
            # Section marker has two capture groups for the number
            section_number = match.group(1) or match.group(2)
            title = match.group(3).strip() if match.group(3) else ""

        # Extract text from end of this marker line to start of next marker
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()

        # Combine title line text and body
        full_text = f"{title}\n{body}".strip() if title else body

        if not full_text:
            logger.debug(
                "Skipping empty %s %s of %s",
                "Article" if is_article else "Section",
                section_number,
                act_name,
            )
            continue

        sections.append(
            StatuteSection(
                act_name=act_name,
                section_number=section_number,
                section_title=title,
                text=full_text,
            )
        )

    return sections


def parse_statute(text: str, act_name: str) -> list[StatuteSection]:
    """Parse a statute into individual sections or articles.

    Automatically selects the appropriate marker pattern:
    - Constitution: Article markers
    - IPC/BNS/CrPC/other: Section markers

    If the primary pattern yields no results, falls back to the
    alternate pattern (e.g., a Constitution amendment referencing sections).

    Args:
        text: Full text of the statute.
        act_name: Name of the act (e.g., "Indian Penal Code", "Constitution of India").

    Returns:
        List of StatuteSection models, one per section/article found.
        Returns an empty list if no markers are detected.
    """
    if not text:
        logger.warning("Empty text received for statute parsing.")
        return []

    if not act_name:
        raise ValueError("act_name must not be empty")

    logger.info("Parsing statute: %s (%d chars)", act_name, len(text))

    is_constitution = _is_constitution(act_name)

    if is_constitution:
        # Try articles first, fall back to sections
        sections = _split_by_pattern(text, _ARTICLE_MARKER, act_name, is_article=True)
        if not sections:
            logger.debug(
                "No Article markers found for '%s', trying Section markers",
                act_name,
            )
            sections = _split_by_pattern(
                text, _SECTION_MARKER, act_name, is_article=False
            )
    else:
        # Try sections first, fall back to articles
        sections = _split_by_pattern(
            text, _SECTION_MARKER, act_name, is_article=False
        )
        if not sections:
            logger.debug(
                "No Section markers found for '%s', trying Article markers",
                act_name,
            )
            sections = _split_by_pattern(
                text, _ARTICLE_MARKER, act_name, is_article=True
            )

    logger.info(
        "Parsed %d sections/articles from '%s'",
        len(sections), act_name,
    )
    return sections
