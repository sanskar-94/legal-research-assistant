"""Regex-based metadata extraction for Indian legal documents.

Extracts case names, citations, judgment dates, court names,
bench/judge names, and statutes/sections cited from legal text.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Final, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Pydantic Models


class DocumentMetadata(BaseModel):
    """Metadata extracted from a legal document.

    All fields are optional because not every document contains
    every piece of metadata.
    """

    case_name: Optional[str] = Field(
        default=None,
        description="Case name, e.g. 'State of Maharashtra v. Xyz'",
    )
    citations: Tuple[str, ...] = Field(
        default=(),
        description="All citations found (AIR, SCC, etc.)",
    )
    date_of_judgment: Optional[date] = Field(
        default=None,
        description="Date the judgment was delivered",
    )
    court: Optional[str] = Field(
        default=None,
        description="Court that issued the judgment",
    )
    bench: Tuple[str, ...] = Field(
        default=(),
        description="Names of judges on the bench",
    )
    statutes_cited: Tuple[str, ...] = Field(
        default=(),
        description="Statutes and sections cited in the document",
    )

    model_config = {"frozen": True}


# Case Name Patterns
# Matches patterns like "X v. Y", "X vs. Y", "X versus Y"
_CASE_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?P<case_name>"
    r"(?:[A-Z][A-Za-z.\s&,'()-]+?)"  # petitioner
    r"\s+(?:v\.?s?\.?|versus)\s+"     # separator
    r"(?:[A-Z][A-Za-z.\s&,'()-]+?)"   # respondent
    r")"
    r"(?=\s*[\n,;(]|\s*$)",
    re.MULTILINE,
)

# Citation Patterns
_AIR_CITATION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"AIR\s+(\d{4})\s+(SC|Bom|Del|Mad|Cal|All|Ker|Kar|Pat|AP|Guj|HP|"
    r"J&K|MP|Ori|P&H|Raj|NOC)\s+(\d+)",
    re.IGNORECASE,
)

_SCC_CITATION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\((\d{4})\)\s+(\d+)\s+SCC\s+(\d+)",
    re.IGNORECASE,
)

_SCR_CITATION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\((\d{4})\)\s+(\d+)\s+SCR\s+(\d+)",
    re.IGNORECASE,
)

# Date Patterns
_MONTH_NAMES: Final[str] = (
    "January|February|March|April|May|June|July|August|"
    "September|October|November|December"
)

_DATE_LONG_PATTERN: Final[re.Pattern[str]] = re.compile(
    rf"(?:dated?\s*[:.]?\s*)?(\d{{1,2}})\s*(?:st|nd|rd|th)?\s+"
    rf"({_MONTH_NAMES}),?\s+(\d{{4}})",
    re.IGNORECASE,
)

_DATE_SHORT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(\d{1,2})[./-](\d{1,2})[./-](\d{4})"
)

_JUDGMENT_DATE_LABEL: Final[re.Pattern[str]] = re.compile(
    rf"(?:Date\s+of\s+(?:Judgment|Judgement|Order|Decision)\s*[:.]?\s*)"
    rf"(\d{{1,2}})\s*(?:st|nd|rd|th)?\s+({_MONTH_NAMES}),?\s+(\d{{4}})",
    re.IGNORECASE,
)

_MONTH_MAP: Final[dict[str, int]] = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

# Court Patterns
_COURT_PATTERNS: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (re.compile(r"Supreme\s+Court\s+of\s+India", re.IGNORECASE), "Supreme Court of India"),
    (re.compile(r"(?:Hon'?ble\s+)?Supreme\s+Court", re.IGNORECASE), "Supreme Court of India"),
    (re.compile(r"High\s+Court\s+of\s+([\w\s&]+?)(?:\s+at\s+\w+)?(?=[,.\n])", re.IGNORECASE), ""),
    (re.compile(r"([\w]+)\s+High\s+Court", re.IGNORECASE), ""),
    (re.compile(r"District\s+Court", re.IGNORECASE), "District Court"),
    (re.compile(r"Sessions\s+Court", re.IGNORECASE), "Sessions Court"),
    (re.compile(r"National\s+Company\s+Law\s+Tribunal|NCLT", re.IGNORECASE), "NCLT"),
    (re.compile(r"National\s+Consumer\s+Disputes\s+Redressal\s+Commission|NCDRC", re.IGNORECASE), "NCDRC"),
    (re.compile(r"National\s+Green\s+Tribunal|NGT", re.IGNORECASE), "NGT"),
)

# Bench / Judge Patterns
_BENCH_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?:(?:Hon'?ble|HON'?BLE)\s+)?"
    r"(?:(?:Mr|Mrs|Ms|Smt|Dr|Justice|JUSTICE|Chief\s+Justice|CJI)\.?\s+)+"
    r"([A-Z][A-Za-z.\s]+?)(?=\s*[,;)\n]|\s+and\s+|\s+JJ|\s+J\.)",
    re.MULTILINE,
)

_CORAM_SECTION: Final[re.Pattern[str]] = re.compile(
    r"(?:CORAM|Coram|BENCH|Bench)\s*[:.]?\s*\n?(.*?)(?:\n\n|\n(?=[A-Z]{2,}))",
    re.DOTALL | re.IGNORECASE,
)

# Statute Patterns
_STATUTE_SECTION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?:Section|Sec\.|S\.)\s*(\d+[A-Za-z]?(?:\(\d+\))?)"
    r"(?:\s+(?:of|,)\s+(?:the\s+)?)"
    r"((?:Indian\s+Penal\s+Code|IPC|"
    r"Bharatiya\s+Nyaya\s+Sanhita|BNS|"
    r"Code\s+of\s+Criminal\s+Procedure|CrPC|Cr\.P\.C\.|"
    r"Code\s+of\s+Civil\s+Procedure|CPC|"
    r"Constitution(?:\s+of\s+India)?|"
    r"Evidence\s+Act|"
    r"Indian\s+Contract\s+Act|"
    r"Transfer\s+of\s+Property\s+Act|"
    r"Limitation\s+Act|"
    r"Negotiable\s+Instruments\s+Act|"
    r"Companies\s+Act|"
    r"Arbitration\s+(?:and\s+Conciliation\s+)?Act|"
    r"Information\s+Technology\s+Act|"
    r"Motor\s+Vehicles\s+Act|"
    r"(?:[\w\s]+Act(?:,?\s*\d{4})?))"
    r")",
    re.IGNORECASE,
)

_ARTICLE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"Article\s+(\d+[A-Za-z]?(?:\(\d+\))?)\s+of\s+(?:the\s+)?"
    r"(Constitution(?:\s+of\s+India)?)",
    re.IGNORECASE,
)


def _extract_case_name(text: str) -> str | None:
    """Extract the primary case name from legal text."""
    # Check first ~2000 chars (case name is usually near the top)
    header_text = text[:2000]
    match = _CASE_NAME_PATTERN.search(header_text)
    if match:
        name = match.group("case_name").strip()
        # Clean up excessive whitespace within the name
        name = re.sub(r"\s+", " ", name)
        logger.debug("Extracted case name: %s", name)
        return name
    return None


def _extract_citations(text: str) -> tuple[str, ...]:
    """Extract all AIR, SCC, and SCR citations."""
    citations: list[str] = []

    for match in _AIR_CITATION_PATTERN.finditer(text):
        year, court, page = match.group(1), match.group(2), match.group(3)
        citations.append(f"AIR {year} {court.upper()} {page}")

    for match in _SCC_CITATION_PATTERN.finditer(text):
        year, volume, page = match.group(1), match.group(2), match.group(3)
        citations.append(f"({year}) {volume} SCC {page}")

    for match in _SCR_CITATION_PATTERN.finditer(text):
        year, volume, page = match.group(1), match.group(2), match.group(3)
        citations.append(f"({year}) {volume} SCR {page}")

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for citation in citations:
        if citation not in seen:
            seen.add(citation)
            unique.append(citation)

    return tuple(unique)


def _parse_date_from_match(
    day_str: str, month_str: str, year_str: str
) -> date | None:
    """Safely parse a date from string components."""
    try:
        if month_str.isdigit():
            month = int(month_str)
        else:
            month = _MONTH_MAP.get(month_str.lower())
            if month is None:
                return None
        return date(int(year_str), month, int(day_str))
    except (ValueError, TypeError):
        logger.debug(
            "Failed to parse date: day=%s month=%s year=%s",
            day_str, month_str, year_str,
        )
        return None


def _extract_date(text: str) -> date | None:
    """Extract the date of judgment."""
    # Try labeled date first (most reliable)
    match = _JUDGMENT_DATE_LABEL.search(text)
    if match:
        result = _parse_date_from_match(
            match.group(1), match.group(2), match.group(3)
        )
        if result is not None:
            return result

    # Try long date format
    match = _DATE_LONG_PATTERN.search(text[:3000])
    if match:
        result = _parse_date_from_match(
            match.group(1), match.group(2), match.group(3)
        )
        if result is not None:
            return result

    # Try short date format (DD/MM/YYYY or DD.MM.YYYY)
    match = _DATE_SHORT_PATTERN.search(text[:3000])
    if match:
        result = _parse_date_from_match(
            match.group(1), match.group(2), match.group(3)
        )
        if result is not None:
            return result

    return None


def _extract_court(text: str) -> str | None:
    """Extract the court name from legal text."""
    header_text = text[:3000]
    for pattern, default_name in _COURT_PATTERNS:
        match = pattern.search(header_text)
        if match:
            if default_name:
                return default_name
            # For High Courts, construct the full name
            court_region = match.group(1).strip()
            return f"High Court of {court_region}"
    return None


def _extract_bench(text: str) -> tuple[str, ...]:
    """Extract judge names from the bench/coram section."""
    judges: list[str] = []

    # Try to find a CORAM section first
    coram_match = _CORAM_SECTION.search(text[:5000])
    search_text = coram_match.group(1) if coram_match else text[:5000]

    for match in _BENCH_PATTERN.finditer(search_text):
        name = match.group(1).strip()
        # Clean up the name
        name = re.sub(r"\s+", " ", name)
        # Skip very short or obviously wrong matches
        if len(name) > 3 and not name.isupper():
            judges.append(name)

    # Deduplicate
    seen: set[str] = set()
    unique: list[str] = []
    for judge in judges:
        if judge not in seen:
            seen.add(judge)
            unique.append(judge)

    return tuple(unique)


def _extract_statutes(text: str) -> tuple[str, ...]:
    """Extract statutes and sections cited."""
    statutes: list[str] = []

    for match in _STATUTE_SECTION_PATTERN.finditer(text):
        section = match.group(1)
        act = match.group(2).strip()
        statutes.append(f"Section {section} of {act}")

    for match in _ARTICLE_PATTERN.finditer(text):
        article = match.group(1)
        act = match.group(2).strip()
        statutes.append(f"Article {article} of {act}")

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for statute in statutes:
        normalized = statute.lower()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(statute)

    return tuple(unique)


def extract_metadata(text: str) -> DocumentMetadata:
    """Extract structured metadata from Indian legal text.

    Performs regex-based extraction of:
    - Case name (e.g., "State of Maharashtra v. Xyz")
    - Citations (AIR, SCC, SCR formats)
    - Date of judgment
    - Court name
    - Bench / judge names
    - Statutes and sections cited

    Args:
        text: The legal document text to extract metadata from.

    Returns:
        An immutable DocumentMetadata model with all extracted fields.
    """
    if not text:
        logger.warning("Empty text received for metadata extraction.")
        return DocumentMetadata()

    logger.debug("Extracting metadata from text of length %d", len(text))

    return DocumentMetadata(
        case_name=_extract_case_name(text),
        citations=_extract_citations(text),
        date_of_judgment=_extract_date(text),
        court=_extract_court(text),
        bench=_extract_bench(text),
        statutes_cited=_extract_statutes(text),
    )
