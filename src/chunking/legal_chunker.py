"""Chunking strategies for Indian legal documents.

Provides three document-type-specific strategies (judgement, statute,
constitution) and a router function that dispatches to the correct one.

All functions are pure: they return *new* Chunk objects and never mutate
their inputs.
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

from config.settings import CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS, CHUNK_TARGET_TOKENS
from src.chunking.chunk_models import Chunk, ChunkMetadata

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

_WORDS_TO_TOKENS_FACTOR = 1.3


def _estimate_tokens(text: str) -> int:
    """Estimate token count using a simple word-count heuristic.

    Uses ``len(text.split()) * 1.3`` as an approximation.
    """
    return int(len(text.split()) * _WORDS_TO_TOKENS_FACTOR)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_metadata(base: dict, *, chunk_index: int) -> ChunkMetadata:
    """Create a *new* ChunkMetadata with the given chunk_index."""
    return ChunkMetadata(**{**base, "chunk_index": chunk_index})


def _overlap_text(text: str, token_count: int) -> str:
    """Return the last *token_count* estimated-tokens worth of text.

    Used to prepend overlap context to the next chunk.
    """
    words = text.split()
    # Convert token budget back to approximate word count
    overlap_words = int(token_count / _WORDS_TO_TOKENS_FACTOR)
    if overlap_words >= len(words):
        return text
    return " ".join(words[-overlap_words:])


def _merge_small_segments(
    segments: Sequence[str],
    min_tokens: int,
) -> list[str]:
    """Merge consecutive segments that are smaller than *min_tokens*.

    Returns a *new* list without mutating the input.
    """
    if not segments:
        return []

    merged: list[str] = []
    buffer = segments[0]

    for segment in segments[1:]:
        if _estimate_tokens(buffer) < min_tokens:
            buffer = buffer.rstrip() + "\n" + segment
        else:
            merged.append(buffer)
            buffer = segment

    # Always flush the remaining buffer
    merged.append(buffer)
    return merged


def _split_long_text(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """Split *text* into pieces of at most *max_tokens* with overlap.

    Splitting is done on sentence boundaries when possible, falling back
    to word boundaries.
    """
    if _estimate_tokens(text) <= max_tokens:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces: list[str] = []
    current = ""

    for sentence in sentences:
        candidate = (current + " " + sentence).strip() if current else sentence
        if _estimate_tokens(candidate) > max_tokens and current:
            pieces.append(current)
            overlap = _overlap_text(current, overlap_tokens)
            current = (overlap + " " + sentence).strip()
        else:
            current = candidate

    if current:
        pieces.append(current)

    return pieces


# ---------------------------------------------------------------------------
# Judgement chunking
# ---------------------------------------------------------------------------

_PARA_NUMBER_RE = re.compile(
    r"(?:^|\n)"           # start of text or newline
    r"(?:"
    r"\d{1,4}\.\s"        # "1. ", "23. "
    r"|"
    r"\(\d{1,4}\)\s"      # "(1) ", "(23) "
    r"|"
    r"\[\d{1,4}\]\s"      # "[1] ", "[23] "
    r")"
)


def chunk_judgement(text: str, metadata: dict) -> list[Chunk]:
    """Chunk a court judgement by paragraph-number markers.

    * Splits on ``^\\d+.``, ``^(\\d+)``, ``^[\\d+]`` markers.
    * Targets CHUNK_TARGET_TOKENS per chunk with CHUNK_OVERLAP_TOKENS overlap.
    * Prefixes every chunk with the case name for retrieval context.
    * Merges very small paragraphs with their neighbours.
    """
    case_name = metadata.get("case_name", "")
    prefix = f"{case_name}\n\n" if case_name else ""

    # Split on paragraph markers, keeping the marker with the text that follows
    splits = _PARA_NUMBER_RE.split(text)
    # The regex split produces empty strings and marker-less leading text
    raw_paragraphs = [s.strip() for s in splits if s and s.strip()]

    if not raw_paragraphs:
        raw_paragraphs = [text]

    # Merge tiny paragraphs (< half of target)
    min_para_tokens = CHUNK_TARGET_TOKENS // 2
    paragraphs = _merge_small_segments(raw_paragraphs, min_para_tokens)

    chunks: list[Chunk] = []
    chunk_index = 0
    overlap_carry = ""

    for paragraph in paragraphs:
        # Prepend overlap from previous chunk
        base_text = (
            (overlap_carry + " " + paragraph).strip()
            if overlap_carry
            else paragraph
        )

        # If still too large, break further
        sub_pieces = _split_long_text(
            base_text,
            max_tokens=CHUNK_MAX_TOKENS,
            overlap_tokens=CHUNK_OVERLAP_TOKENS,
        )

        for piece in sub_pieces:
            chunk_text = prefix + piece
            chunk = Chunk(
                text=chunk_text,
                metadata=_build_metadata(metadata, chunk_index=chunk_index),
            )
            chunks.append(chunk)
            chunk_index += 1

        # Carry overlap into next iteration
        overlap_carry = _overlap_text(paragraph, CHUNK_OVERLAP_TOKENS)

    logger.info(
        "Judgement chunked: %d chunks from %d estimated tokens",
        len(chunks),
        _estimate_tokens(text),
    )
    return chunks


# ---------------------------------------------------------------------------
# Statute chunking
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(
    r"(?:^|\n)"
    r"(Section\s+\d+[A-Za-z]*"  # "Section 302", "Section 34A"
    r"(?:\s*\([^)]*\))?)"       # optional sub-section indicator "(1)"
    r"[.:\-—\s]",
    re.IGNORECASE,
)


def chunk_statute(text: str, metadata: dict) -> list[Chunk]:
    """Chunk a statute so that each section becomes one chunk.

    If a section exceeds CHUNK_MAX_TOKENS it is further split at
    sub-section markers.
    """
    parts = _SECTION_RE.split(text)

    # Re-assemble: parts alternates between inter-section text and
    # captured group (the section header).  We pair each header with
    # the text that follows it.
    sections: list[tuple[str, str]] = []
    preamble = parts[0].strip() if parts else ""
    idx = 1
    while idx < len(parts) - 1:
        header = parts[idx].strip()
        body = parts[idx + 1].strip()
        sections.append((header, f"{header} {body}"))
        idx += 2

    # If no section markers found, treat entire text as one section
    if not sections:
        sections = [("", text)]

    # Include preamble if non-trivial
    if preamble and _estimate_tokens(preamble) > 30:
        sections = [("Preamble", preamble)] + list(sections)

    act_name = metadata.get("act_name", "")
    chunks: list[Chunk] = []
    chunk_index = 0

    for header, section_text in sections:
        # Extract section number for metadata
        sec_match = re.search(r"Section\s+(\d+[A-Za-z]*)", header, re.IGNORECASE)
        section_number = sec_match.group(1) if sec_match else None

        per_chunk_meta = {
            **metadata,
            "section_number": section_number or metadata.get("section_number"),
        }

        if _estimate_tokens(section_text) <= CHUNK_MAX_TOKENS:
            chunk = Chunk(
                text=section_text,
                metadata=_build_metadata(per_chunk_meta, chunk_index=chunk_index),
            )
            chunks.append(chunk)
            chunk_index += 1
        else:
            # Split at sub-section markers: (1), (a), (i), etc.
            sub_parts = re.split(r"\n\s*\([a-z0-9]+\)\s", section_text)
            sub_parts = _merge_small_segments(
                [s.strip() for s in sub_parts if s.strip()],
                CHUNK_TARGET_TOKENS // 2,
            )
            for sub in sub_parts:
                prefix = f"{act_name} {header}\n" if act_name else f"{header}\n"
                chunk = Chunk(
                    text=prefix + sub,
                    metadata=_build_metadata(per_chunk_meta, chunk_index=chunk_index),
                )
                chunks.append(chunk)
                chunk_index += 1

    logger.info(
        "Statute chunked: %d chunks from %d estimated tokens",
        len(chunks),
        _estimate_tokens(text),
    )
    return chunks


# ---------------------------------------------------------------------------
# Constitution chunking
# ---------------------------------------------------------------------------

_ARTICLE_RE = re.compile(
    r"(?:^|\n)"
    r"(Article\s+\d+[A-Za-z]*)"  # "Article 14", "Article 21A"
    r"[.:\-—\s]",
    re.IGNORECASE,
)

_EXPLANATION_RE = re.compile(
    r"(?:Explanation|Proviso|Exception)\b",
    re.IGNORECASE,
)


def chunk_constitution(text: str, metadata: dict) -> list[Chunk]:
    """Chunk a constitutional document so that each Article is one chunk.

    Explanations and provisos remain attached to their parent Article.
    Articles exceeding CHUNK_MAX_TOKENS are split at clause boundaries.
    """
    parts = _ARTICLE_RE.split(text)

    articles: list[tuple[str, str]] = []
    preamble = parts[0].strip() if parts else ""
    idx = 1
    while idx < len(parts) - 1:
        header = parts[idx].strip()
        body = parts[idx + 1].strip()
        articles.append((header, f"{header}. {body}"))
        idx += 2

    if not articles:
        articles = [("", text)]

    if preamble and _estimate_tokens(preamble) > 30:
        articles = [("Preamble", preamble)] + list(articles)

    chunks: list[Chunk] = []
    chunk_index = 0

    for header, article_text in articles:
        sec_match = re.search(r"Article\s+(\d+[A-Za-z]*)", header, re.IGNORECASE)
        article_number = sec_match.group(1) if sec_match else None

        per_chunk_meta = {
            **metadata,
            "section_number": article_number or metadata.get("section_number"),
        }

        if _estimate_tokens(article_text) <= CHUNK_MAX_TOKENS:
            chunk = Chunk(
                text=article_text,
                metadata=_build_metadata(per_chunk_meta, chunk_index=chunk_index),
            )
            chunks.append(chunk)
            chunk_index += 1
        else:
            # Split at clause boundaries but keep explanations attached
            sub_pieces = _split_long_text(
                article_text,
                max_tokens=CHUNK_MAX_TOKENS,
                overlap_tokens=CHUNK_OVERLAP_TOKENS,
            )
            for piece in sub_pieces:
                chunk = Chunk(
                    text=piece,
                    metadata=_build_metadata(per_chunk_meta, chunk_index=chunk_index),
                )
                chunks.append(chunk)
                chunk_index += 1

    logger.info(
        "Constitution chunked: %d chunks from %d estimated tokens",
        len(chunks),
        _estimate_tokens(text),
    )
    return chunks


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

_STRATEGY_MAP = {
    "judgement": chunk_judgement,
    "statute": chunk_statute,
    "constitution": chunk_constitution,
}


def chunk_document(text: str, doc_type: str, metadata: dict) -> list[Chunk]:
    """Route a document to the appropriate chunking strategy.

    Parameters
    ----------
    text:
        Full document text.
    doc_type:
        One of ``"judgement"``, ``"statute"``, ``"constitution"``.
    metadata:
        Dictionary of document-level metadata fields.  Must contain at
        least ``doc_id`` and ``doc_type``.

    Returns
    -------
    list[Chunk]
        Ordered list of chunks produced by the selected strategy.

    Raises
    ------
    ValueError
        If *doc_type* is not recognised.
    """
    strategy = _STRATEGY_MAP.get(doc_type)
    if strategy is None:
        supported = ", ".join(sorted(_STRATEGY_MAP))
        raise ValueError(
            f"Unknown doc_type '{doc_type}'. Supported types: {supported}"
        )

    enriched_metadata = {**metadata, "doc_type": doc_type}
    return strategy(text, enriched_metadata)
