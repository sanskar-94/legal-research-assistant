"""Legal text normalization and cleaning utilities.

Handles whitespace normalization, OCR artifact correction,
section reference standardization, and citation format normalization
for Indian legal documents.
"""

from __future__ import annotations

import logging
import re
from typing import Final

logger = logging.getLogger(__name__)

# ── OCR Artifact Corrections ────────────────────────────────────────
# Ordered pairs: (pattern, replacement)
# Applied in sequence; order matters for correctness.
_OCR_CORRECTIONS: Final[tuple[tuple[str, str], ...]] = (
    # Common ligature / glyph misreads
    (r"\brn\b", "m"),               # "rn" misread as separate chars instead of "m"
    (r"(?<=[A-Z])0(?=[A-Z])", "O"), # digit zero between uppercase letters → letter O
    (r"(?<=[a-z])0(?=[a-z])", "o"), # digit zero between lowercase letters → letter o
    (r"(?<=\s)l(?=\d)", "1"),       # lowercase L before digit → digit 1
    (r"(?<=\d)l(?=\d)", "1"),       # lowercase L between digits → digit 1
    (r"(?<=\d)l(?=\s)", "1"),       # lowercase L after digit before space → digit 1
    (r"\bII\b(?=\s+\d)", "11"),     # "II" before numbers → "11" (common in section refs)
    (r"(?<=\w)ff(?=\s)", "ff"),     # keep "ff" at word end (not a misread)
    (r"\btbe\b", "the"),            # common OCR misread
    (r"\bwitb\b", "with"),          # common OCR misread
    (r"\bwbich\b", "which"),        # common OCR misread
    (r"\bjudgrnent\b", "judgment"), # "rn" in judgment
    (r"\bgovemment\b", "government"),
    (r"\bparliarnent\b", "parliament"),
    (r"\bamnendment\b", "amendment"),
    (r"\bcomrnission\b", "commission"),
)

# ── Section Reference Patterns ──────────────────────────────────────
# All common abbreviations for "Section"
_SECTION_ABBREVS: Final[re.Pattern[str]] = re.compile(
    r"\b(?:Sec\.|sec\.|S\.|s\.)\s*(\d+)",
    re.IGNORECASE,
)

# "Article" abbreviations
_ARTICLE_ABBREVS: Final[re.Pattern[str]] = re.compile(
    r"\b(?:Art\.|art\.)\s*(\d+)",
    re.IGNORECASE,
)

# ── Citation Format Patterns ────────────────────────────────────────
# AIR citation: normalize spacing and format
_AIR_CITATION: Final[re.Pattern[str]] = re.compile(
    r"A\.?\s*I\.?\s*R\.?\s*(\d{4})\s+(SC|S\.?\s*C\.?|"
    r"Bom|Del|Mad|Cal|All|Ker|Kar|Pat|AP|Guj|HP|J&K|MP|Ori|P&H|Raj|NOC)\s+(\d+)",
    re.IGNORECASE,
)

# SCC citation: normalize spacing and format
_SCC_CITATION: Final[re.Pattern[str]] = re.compile(
    r"\((\d{4})\)\s*(\d+)\s*S\.?\s*C\.?\s*C\.?\s*(\d+)",
    re.IGNORECASE,
)

# ── Header / Footer Patterns ───────────────────────────────────────
_PAGE_HEADER_FOOTER: Final[re.Pattern[str]] = re.compile(
    r"(?m)"
    r"(?:"
    r"^\s*-\s*\d+\s*-\s*$"       # lines like "- 12 -"
    r"|^\s*Page\s+\d+\s*(?:of\s+\d+)?\s*$"  # "Page 3 of 10"
    r"|^\s*\d+\s*/\s*\d+\s*$"    # "3/10"
    r"|^\s*(?:www\.|http).*$"     # URL lines (often in footers)
    r"|^\s*(?:REPORTABLE|NON-REPORTABLE|NOT\s+REPORTABLE)\s*$"
    r"|^\s*IN\s+THE\s+SUPREME\s+COURT\s+OF\s+INDIA\s*$"  # repeated header
    r")",
    re.IGNORECASE,
)

# ── Whitespace Patterns ────────────────────────────────────────────
_MULTIPLE_BLANK_LINES: Final[re.Pattern[str]] = re.compile(r"\n{3,}")
_MULTIPLE_SPACES: Final[re.Pattern[str]] = re.compile(r"[ \t]{2,}")
_TRAILING_SPACES: Final[re.Pattern[str]] = re.compile(r"[ \t]+$", re.MULTILINE)


def _normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces, tabs, and blank lines."""
    result = _TRAILING_SPACES.sub("", text)
    result = _MULTIPLE_SPACES.sub(" ", result)
    result = _MULTIPLE_BLANK_LINES.sub("\n\n", result)
    return result.strip()


def _fix_ocr_artifacts(text: str) -> str:
    """Apply common OCR misread corrections."""
    result = text
    for pattern, replacement in _OCR_CORRECTIONS:
        result = re.sub(pattern, replacement, result)
    return result


def _standardize_section_references(text: str) -> str:
    """Normalize 'Sec. 302', 'S. 302', 'sec. 302' to 'Section 302'."""
    result = _SECTION_ABBREVS.sub(r"Section \1", text)
    result = _ARTICLE_ABBREVS.sub(r"Article \1", result)
    return result


def _standardize_citations(text: str) -> str:
    """Normalize AIR and SCC citation formats."""

    def _normalize_air(match: re.Match[str]) -> str:
        year = match.group(1)
        court = match.group(2).strip().replace(".", "").replace(" ", "")
        page = match.group(3)
        return f"AIR {year} {court.upper()} {page}"

    def _normalize_scc(match: re.Match[str]) -> str:
        year = match.group(1)
        volume = match.group(2)
        page = match.group(3)
        return f"({year}) {volume} SCC {page}"

    result = _AIR_CITATION.sub(_normalize_air, text)
    result = _SCC_CITATION.sub(_normalize_scc, result)
    return result


def _remove_headers_footers(text: str) -> str:
    """Remove common page headers and footers from legal documents."""
    return _PAGE_HEADER_FOOTER.sub("", text)


def clean_legal_text(text: str) -> str:
    """Clean and normalize Indian legal text.

    Applies the following transformations in order:
    1. Remove page headers/footers
    2. Fix common OCR artifacts
    3. Standardize section/article references
    4. Standardize citation formats (AIR, SCC)
    5. Normalize whitespace

    Args:
        text: Raw legal text to clean.

    Returns:
        A new string with all normalizations applied.
    """
    if not text:
        logger.debug("Empty text received, returning empty string.")
        return ""

    logger.debug("Cleaning legal text of length %d", len(text))

    result = _remove_headers_footers(text)
    result = _fix_ocr_artifacts(result)
    result = _standardize_section_references(result)
    result = _standardize_citations(result)
    result = _normalize_whitespace(result)

    logger.debug("Cleaned text length: %d", len(result))
    return result
