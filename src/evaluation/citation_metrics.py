"""Answer-quality metrics shared by the evaluation script and the UI.

Citation validity only checks whether ``[SOURCE_N]`` tags point to provided
source chunks. Citation support is a conservative local proxy for whether the
cited source text supports the attached legal claim. Hallucination detection
flags case names and reporter citations that appear in no source.
"""

from __future__ import annotations

import re
from typing import Any

SOURCE_REF_PATTERN: re.Pattern[str] = re.compile(r"\[SOURCE_(\d+)\]")
SENTENCE_SPLIT_PATTERN: re.Pattern[str] = re.compile(r"(?<=[.!?])\s+")
LEGAL_ANCHOR_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:section|sec\.?|article|act|ipc|crpc|bns|constitution|court|"
    r"judg(?:e)?ment|case|conviction|murder|theft|bail|appeal|right|"
    r"punishment|imprisonment|fine|liability|offence|offense)\b",
    re.IGNORECASE,
)
SECTION_OR_ARTICLE_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:section|sec\.?|article)\s+\d+[A-Za-z-]*\b",
    re.IGNORECASE,
)
LEGAL_CONCEPT_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:murder|theft|bail|appeal|conviction|right|punish(?:ment|ed)?|"
    r"imprisonment|fine|liability|offence|offense)\b",
    re.IGNORECASE,
)
STOPWORDS: set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "under",
    "with",
}


def compute_citation_validity(
    response_text: str,
    source_chunks: list[dict[str, Any]],
) -> float:
    """Return the fraction of ``[SOURCE_N]`` references that map to real chunks."""
    refs = SOURCE_REF_PATTERN.findall(response_text)
    if not refs:
        return 0.0

    valid = sum(1 for ref in refs if 1 <= int(ref) <= len(source_chunks))
    return valid / len(refs)


def claim_sentences(response_text: str) -> list[str]:
    """Extract substantive legal claim sentences from an answer."""
    cleaned = response_text.replace("\n", " ")
    raw_sentences = [
        part.strip()
        for part in SENTENCE_SPLIT_PATTERN.split(cleaned)
        if part.strip()
    ]
    sentences: list[str] = []
    for part in raw_sentences:
        if SOURCE_REF_PATTERN.fullmatch(part.strip()) and sentences:
            sentences[-1] = f"{sentences[-1]} {part}"
        else:
            sentences.append(part)

    claims: list[str] = []
    for sentence in sentences:
        without_refs = SOURCE_REF_PATTERN.sub("", sentence).strip()
        if len(without_refs) < 25:
            continue
        if LEGAL_ANCHOR_PATTERN.search(without_refs) or SECTION_OR_ARTICLE_PATTERN.search(
            without_refs
        ):
            claims.append(sentence)
    return claims


def compute_cited_claim_coverage(response_text: str) -> float:
    """Return the fraction of substantive legal claims with nearby citations."""
    claims = claim_sentences(response_text)
    if not claims:
        return 0.0

    cited = sum(1 for claim in claims if SOURCE_REF_PATTERN.search(claim))
    return cited / len(claims)


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z][A-Za-z0-9]+", text.lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def _legal_anchors(text: str) -> set[str]:
    anchors = {
        match.group(0).lower()
        for match in SECTION_OR_ARTICLE_PATTERN.finditer(text)
    }
    anchors.update(
        token.lower()
        for token in re.findall(r"\b(?:ipc|crpc|bns|constitution)\b", text, re.I)
    )
    return anchors


def _legal_concepts(text: str) -> set[str]:
    concepts: set[str] = set()
    for match in LEGAL_CONCEPT_PATTERN.finditer(text):
        concept = match.group(0).lower()
        if concept.startswith("punish"):
            concept = "punish"
        concepts.add(concept)
    return concepts


def claim_supported_by_source(claim: str, source_text: str) -> bool:
    """Return whether a cited source appears to support a legal claim."""
    claim_without_refs = SOURCE_REF_PATTERN.sub("", claim)
    claim_tokens = _content_tokens(claim_without_refs)
    source_tokens = _content_tokens(source_text)
    if not claim_tokens or not source_tokens:
        return False

    claim_anchors = _legal_anchors(claim_without_refs)
    source_anchors = _legal_anchors(source_text)
    if claim_anchors and not claim_anchors.issubset(source_anchors):
        return False

    claim_concepts = _legal_concepts(claim_without_refs)
    source_concepts = _legal_concepts(source_text)
    if claim_concepts and not claim_concepts.issubset(source_concepts):
        return False

    overlap = claim_tokens & source_tokens
    overlap_rate = len(overlap) / len(claim_tokens)
    return overlap_rate >= 0.35 or len(overlap) >= 5


def compute_claim_support_rate(
    response_text: str,
    source_chunks: list[dict[str, Any]],
) -> float:
    """Return supported cited legal claims divided by all cited legal claims."""
    cited_claims = [
        claim for claim in claim_sentences(response_text) if SOURCE_REF_PATTERN.search(claim)
    ]
    if not cited_claims:
        return 0.0

    supported = 0
    for claim in cited_claims:
        refs = [int(ref) for ref in SOURCE_REF_PATTERN.findall(claim)]
        valid_sources = [
            source_chunks[ref - 1]
            for ref in refs
            if 1 <= ref <= len(source_chunks)
        ]
        if any(
            claim_supported_by_source(claim, chunk.get("text", ""))
            for chunk in valid_sources
        ):
            supported += 1

    return supported / len(cited_claims)


def compute_citation_quality(
    response_text: str,
    source_chunks: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute citation validity, coverage, support, and blended quality."""
    validity = compute_citation_validity(response_text, source_chunks)
    coverage = compute_cited_claim_coverage(response_text)
    support = compute_claim_support_rate(response_text, source_chunks)
    blended = 0.30 * validity + 0.30 * coverage + 0.40 * support
    return {
        "citation_validity": round(validity, 4),
        "cited_claim_coverage": round(coverage, 4),
        "citation_support_score": round(support, 4),
        "citation_quality_score": round(blended, 4),
    }


# Well-known cases the model may name from general knowledge. Mentioning
# one of these without a source is not counted as a fabrication.
_KNOWN_CASE_FRAGMENTS: tuple[str, ...] = (
    "Kesavananda Bharati",
    "Maneka Gandhi",
    "Vishaka",
    "ADM Jabalpur",
    "Navtej Singh Johar",
    "Puttaswamy",
    "MC Mehta",
    "Indira Gandhi",
    "Shreya Singhal",
    "Golaknath",
    "Minerva Mills",
    "S.R. Bommai",
    "Menaka Gandhi",
    "State of Maharashtra v. Prakash",
    "Sharad Birdhichand",
    "Joseph Shine",
)

_CITATION_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:AIR|SCC|SCR)\s+\d{4}\s+\w+\s+\d+",
    re.IGNORECASE,
)

_CASE_NAME_PATTERN: re.Pattern[str] = re.compile(
    r"([A-Z][A-Za-z.\s]+?)\s+(?:v\.?s?\.?|versus)\s+([A-Z][A-Za-z.\s]+?)(?=\s*[\n,;(.\[]|$)",
    re.MULTILINE,
)

_REFUSAL_MARKERS: tuple[str, ...] = (
    "retrieved sources do not address this question",
    "do not contain sufficient information",
    "cannot answer",
)


def detect_hallucinations(
    response_text: str,
    source_chunks: list[dict[str, Any]],
) -> list[str]:
    """Return case names and citations in the answer that no source contains."""
    source_text = " ".join(chunk.get("text", "") for chunk in source_chunks).lower()
    flagged: list[str] = []

    for match in _CASE_NAME_PATTERN.finditer(response_text):
        name = match.group(0).strip()
        if any(frag.lower() in name.lower() for frag in _KNOWN_CASE_FRAGMENTS):
            continue
        if name.lower() not in source_text:
            flagged.append(f"Case: {name}")

    for match in _CITATION_PATTERN.finditer(response_text):
        citation = match.group(0)
        if citation.lower() not in source_text:
            flagged.append(f"Citation: {citation}")

    return flagged


def is_substantive_response(response_text: str, num_sources: int) -> bool:
    """An answer counts if it cites a real source and is not a refusal.

    Partial answers that end with "Not covered by the retrieved sources: ..."
    still count, because the covered part is answered with citations.
    """
    lower = response_text.lower()
    if any(marker in lower for marker in _REFUSAL_MARKERS):
        return False
    refs = [int(ref) for ref in SOURCE_REF_PATTERN.findall(response_text)]
    return any(1 <= ref <= num_sources for ref in refs)
