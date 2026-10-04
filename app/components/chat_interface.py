"""Chat UI component for the Legal Research Assistant.

Renders conversation history and formatted answers with minimal,
monospace-styled citation badges and a source summary section.
"""

from __future__ import annotations

import re
from typing import Optional

import streamlit as st


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------


def _esc(value: str) -> str:
    """HTML-escape a string to prevent XSS."""
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


# ---------------------------------------------------------------------------
# Metadata helpers for source summary
# ---------------------------------------------------------------------------


def _resolve_meta(source: dict) -> dict:
    """Flatten metadata from source dict, handling nested structures."""
    meta = source.get("metadata", {})
    if not isinstance(meta, dict):
        meta = {}
    inner = meta.get("metadata")
    if isinstance(inner, dict) and "doc_type" in inner:
        resolved = {**inner}
        for key in (
            "semantic_score",
            "final_score",
            "rrf_score",
            "authority_score",
            "chunk_id",
        ):
            if key in meta:
                resolved[key] = meta[key]
        return resolved
    return meta


def _source_origin(meta: dict, text: str) -> str:
    """Return the human-readable origin name (court, act, or institution)."""
    doc_type = meta.get("doc_type", "unknown")

    if doc_type == "judgement":
        court = meta.get("court", "")
        if not court and text:
            text_lower = text[:500].lower()
            if "supreme court" in text_lower:
                court = "Supreme Court of India"
            elif "high court" in text_lower:
                hc = re.search(r"([\w\s]+high\s+court)", text_lower)
                court = hc.group(1).title() if hc else "High Court"
            elif "tribunal" in text_lower:
                court = "Tribunal"
        if court:
            return court.replace("_", " ").title() if "_" in court else court
        return "Judgement"

    if doc_type == "statute":
        return meta.get("act_name") or "Statute"

    if doc_type == "constitution":
        return "Constitution of India"

    return (
        doc_type.replace("_", " ").title() if doc_type != "unknown" else "Legal Source"
    )


def _source_title(meta: dict, text: str) -> str:
    """Short descriptive title for a source in the summary."""
    case_name = meta.get("case_name", "")
    if case_name:
        return case_name

    act_name = meta.get("act_name", "")
    section = meta.get("section_number", "")
    if act_name and section:
        return f"{act_name}, S. {section}"
    if act_name:
        return act_name
    if section:
        return f"Section {section}"

    if text:
        first_line = text.strip().split("\n")[0].strip()
        if len(first_line) > 60:
            return first_line[:57] + "\u2026"
        if len(first_line) > 15:
            return first_line

    return "Source"


def _source_url(meta: dict) -> str:
    """Build URL to the original document if ``doc_id`` is available."""
    doc_id = str(meta.get("doc_id", ""))
    if doc_id and doc_id.isdigit():
        return f"https://indiankanoon.org/doc/{doc_id}/"
    return ""


# ---------------------------------------------------------------------------
# Citation badges
# ---------------------------------------------------------------------------


def format_citation_badge(source_index: int) -> str:
    """Return an inline-HTML badge for ``[SOURCE_N]``.

    Uses a unified warm accent style with monospace font for a
    clean, minimal look that's consistent with the dark theme.

    Parameters
    ----------
    source_index:
        1-based source number.

    Returns
    -------
    str
        An HTML ``<span>`` styled as a subtle pill badge.
    """
    return f'<span class="cite-badge">S{source_index}</span>'


def _replace_citation_tags(text: str) -> str:
    """Replace all ``[SOURCE_N]`` markers in *text* with styled badges.

    The text is HTML-escaped first so that any injected HTML in the
    LLM output is neutralised before rendering with ``unsafe_allow_html``.
    ``[SOURCE_N]`` markers survive escaping (no HTML-special characters)
    and are then replaced with badge HTML.
    """
    escaped = _esc(text)

    def _replacer(match: re.Match[str]) -> str:
        index = int(match.group(1))
        return format_citation_badge(index)

    return re.sub(r"\[SOURCE_(\d+)\]", _replacer, escaped)


# ---------------------------------------------------------------------------
# Source summary (rendered below the answer)
# ---------------------------------------------------------------------------


def render_source_summary(sources: list[dict]) -> None:
    """Render a compact source summary with names and clickable links.

    Each entry shows the source badge, the origin name (e.g.
    "Supreme Court of India") and a short title that links to the
    original document when a URL can be constructed.
    """
    if not sources:
        return

    items_html: list[str] = []
    for src in sources:
        try:
            idx = int(src.get("source_index", 0))
        except (TypeError, ValueError):
            idx = 0
        meta = _resolve_meta(src)
        text = src.get("text", "")

        origin = _esc(_source_origin(meta, text))
        title = _esc(_source_title(meta, text))
        url = _source_url(meta)

        if url:
            title_html = (
                f'<a href="{_esc(url)}" target="_blank" '
                f'rel="noopener noreferrer" class="source-link-text">'
                f"{title}</a>"
            )
        else:
            title_html = f'<span class="source-link-text-plain">{title}</span>'

        items_html.append(
            f'<div class="source-summary-item">'
            f'  <span class="source-summary-badge">S{idx}</span>'
            f'  <span class="source-summary-origin">{origin}</span>'
            f'  <span class="source-summary-sep">\u2014</span>'
            f"  {title_html}"
            f"</div>"
        )

    html = (
        '<div class="source-summary">'
        '  <div class="source-summary-header">Sources Referenced</div>'
        + "".join(items_html)
        + "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main renderers
# ---------------------------------------------------------------------------


def render_answer(
    answer_text: str,
    sources: Optional[list[dict]] = None,
) -> None:
    """Display an answer with citation badges and a source summary.

    Parameters
    ----------
    answer_text:
        The raw answer string potentially containing ``[SOURCE_N]`` markers.
    sources:
        Optional list of cited source dicts. When provided, a compact
        "Sources Referenced" section is appended below the answer.
    """
    formatted = _replace_citation_tags(answer_text)
    st.markdown(formatted, unsafe_allow_html=True)


from app.components.confidence_bar import render_confidence_bar


def render_chat_history(messages: list[dict]) -> None:
    """Render the full conversation history.

    Parameters
    ----------
    messages:
        A list of dicts with ``"role"``, ``"content"``, and optional
        ``"sources"``, ``"confidence"``, and ``"retrieved_chunks"`` keys.
    """
    for msg in messages:
        role = msg["role"]
        with st.chat_message(role):
            if role == "assistant":
                render_answer(msg["content"], msg.get("sources"))

                confidence = msg.get("confidence")
                retrieved_chunks = msg.get("retrieved_chunks")

                if confidence is not None:
                    render_confidence_bar(
                        confidence,
                        answer_text=msg["content"],
                        retrieved_chunks=retrieved_chunks,
                    )
            else:
                st.markdown(msg["content"])
