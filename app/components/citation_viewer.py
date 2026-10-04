"""Citation panel component for the Legal Research Assistant.

Renders cited sources as clean dark cards with metadata and
expandable excerpts. Designed for the right column of the layout.

When stored metadata fields are empty (common in this dataset),
falls back to extracting meaningful information from the chunk
text itself — section references, case names, act names, and
Q&A-style headings.
"""

from __future__ import annotations

import re

import streamlit as st


def _esc(value: str) -> str:
    """HTML-escape a string to prevent XSS in unsafe_allow_html output."""
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


_TYPE_LABELS: dict[str, str] = {
    "judgement": "Judgement",
    "statute": "Statute",
    "constitution": "Constitution",
}

# ---------------------------------------------------------------------------
# Text-based metadata extraction (fallback when stored metadata is empty)
# ---------------------------------------------------------------------------

# Matches patterns like "Section 302 IPC", "Section 154 of CrPC",
# "Section 34 of the Indian Penal Code", "Article 21"
_SECTION_RE = re.compile(
    r"(?:Section|Article|S\.)\s+(\d+[A-Za-z]?)"
    r"(?:\s+(?:of\s+(?:the\s+)?)?(?:IPC|BNS|CrPC|BNSS|Indian Penal Code|"
    r"Bharatiya Nyaya Sanhita|Code of Criminal Procedure|Constitution))?",
    re.IGNORECASE,
)

# Matches "v." or "vs." case citation patterns
_CASE_NAME_RE = re.compile(
    r"([A-Z][A-Za-z\.\s]+?)\s+v[s]?\.\s+([A-Z][A-Za-z\.\s]+?)(?:\s*[\(\[,]|\s*$)",
    re.MULTILINE,
)

# Matches "Legal Question: ..." style from HuggingFace Q&A data
_LEGAL_QA_RE = re.compile(
    r"Legal\s+Question:\s*(.+?)(?:\n|Legal\s+Answer:)",
    re.IGNORECASE | re.DOTALL,
)

# Act name patterns
_ACT_RE = re.compile(
    r"(?:Indian Penal Code|IPC|Bharatiya Nyaya Sanhita|BNS|"
    r"Code of Criminal Procedure|CrPC|BNSS|"
    r"Indian Constitution|Constitution of India|"
    r"Evidence Act|Contract Act|"
    r"(?:the\s+)?[\w\s]+Act(?:,?\s*\d{4})?)",
    re.IGNORECASE,
)


def _extract_title_from_text(text: str, doc_type: str) -> str:
    """Derive a meaningful title from chunk text when metadata is empty."""
    if not text:
        return ""

    # 1. Check for Q&A format (HuggingFace legal datasets)
    qa_match = _LEGAL_QA_RE.search(text)
    if qa_match:
        question = qa_match.group(1).strip()
        # Clean up and truncate
        question = question.rstrip("?").strip()
        if len(question) > 80:
            question = question[:77] + "..."
        return question

    # 2. Check for case name (e.g. "State of Kerala v. Rajesh")
    case_match = _CASE_NAME_RE.search(text[:500])
    if case_match:
        party1 = case_match.group(1).strip()
        party2 = case_match.group(2).strip()
        name = f"{party1} v. {party2}"
        if len(name) > 80:
            name = name[:77] + "..."
        return name

    # 3. Extract section references
    section_matches = _SECTION_RE.findall(text[:400])
    if section_matches:
        unique_sections = list(dict.fromkeys(section_matches))[:3]
        sections_str = ", ".join(f"S. {s}" for s in unique_sections)
        if doc_type == "statute":
            # Try to find which act
            act_match = _ACT_RE.search(text[:300])
            if act_match:
                act = act_match.group(0).strip()
                return f"{act} \u2014 {sections_str}"
        return f"Sections {', '.join(unique_sections)}"

    # 4. Use first meaningful sentence as title
    first_line = text.strip().split("\n")[0].strip()
    # Skip very short or numeric-only lines
    if len(first_line) > 15:
        if len(first_line) > 80:
            first_line = first_line[:77] + "..."
        return first_line

    return ""


def _extract_sections_from_text(text: str) -> list[str]:
    """Extract section/article references from text."""
    if not text:
        return []
    matches = _SECTION_RE.findall(text[:600])
    return list(dict.fromkeys(matches))[:5]


def _extract_court_from_text(text: str) -> str:
    """Try to identify the court from text content."""
    text_lower = text[:500].lower()
    if "supreme court" in text_lower:
        return "Supreme Court of India"
    if "high court" in text_lower:
        # Try to get specific high court
        hc_match = re.search(r"([\w\s]+high\s+court)", text_lower)
        if hc_match:
            return hc_match.group(1).title()
        return "High Court"
    if "tribunal" in text_lower:
        return "Tribunal"
    return ""


def _build_source_url(metadata: dict) -> str:
    """Build a URL to the original document when a doc_id is available."""
    doc_id = str(metadata.get("doc_id", ""))
    if doc_id and doc_id.isdigit():
        return f"https://indiankanoon.org/doc/{doc_id}/"
    return ""


def _get_source_origin(metadata: dict, text: str) -> str:
    """Return the human-readable origin (court, act name, or institution)."""
    doc_type = metadata.get("doc_type", "unknown")

    if doc_type == "judgement":
        court = metadata.get("court", "")
        if not court:
            court = _extract_court_from_text(text)
        if court:
            return court.replace("_", " ").title() if "_" in court else court
        return "Court (unspecified)"

    if doc_type == "statute":
        return metadata.get("act_name") or "Statute"

    if doc_type == "constitution":
        return "Constitution of India"

    return doc_type.replace("_", " ").title() if doc_type != "unknown" else "Legal Source"


def _resolve_metadata(source: dict) -> dict:
    """Resolve metadata from source dict, handling nested structures.

    The response parser may produce metadata in several shapes:
    - ``source["metadata"]`` is a flat dict with doc fields
    - ``source["metadata"]["metadata"]`` is the actual doc metadata (legacy nesting)

    This function always returns a flat dict with the document fields.
    """
    meta = source.get("metadata", {})
    if not isinstance(meta, dict):
        meta = {}

    # Handle double-nesting: if meta contains a nested "metadata" dict
    # with doc_type inside it, that's the real metadata
    inner = meta.get("metadata")
    if isinstance(inner, dict) and "doc_type" in inner:
        # Merge inner metadata with any scoring fields from outer
        resolved = {**inner}
        for key in ("semantic_score", "final_score", "rrf_score", "authority_score", "chunk_id"):
            if key in meta:
                resolved[key] = meta[key]
        return resolved

    return meta


# ---------------------------------------------------------------------------
# Card builder
# ---------------------------------------------------------------------------

def _build_source_card_html(source: dict, index: int) -> str:
    """Build HTML for a single source card.

    Uses stored metadata when available, falls back to text extraction.
    """
    metadata = _resolve_metadata(source)
    doc_type = metadata.get("doc_type", "unknown")
    text = source.get("text", "")

    type_label = _esc(_TYPE_LABELS.get(doc_type, doc_type.replace("_", " ").title()))

    # --- Title resolution (stored metadata → text extraction) ---
    case_name = metadata.get("case_name", "")
    act_name = metadata.get("act_name", "")
    section_number = metadata.get("section_number", "")

    if case_name:
        title = _esc(case_name)
    elif act_name and section_number:
        title = _esc(f"{act_name}, S. {section_number}")
    elif act_name:
        title = _esc(act_name)
    else:
        # Fallback: extract from text
        extracted_title = _extract_title_from_text(text, doc_type)
        title = _esc(extracted_title) if extracted_title else f"{type_label} Source"

    # --- Metadata items ---
    meta_items: list[str] = []

    # Court (stored or extracted)
    court = metadata.get("court", "")
    if not court:
        court = _extract_court_from_text(text)
    if court:
        display_court = _esc(court.replace("_", " ").title() if "_" in court else court)
        meta_items.append(
            f"<span class='source-meta-item'><strong>Court:</strong> {display_court}</span>"
        )

    # Citation
    citation = metadata.get("citation", "")
    if citation:
        meta_items.append(
            f"<span class='source-meta-item'><strong>Cite:</strong> {_esc(citation)}</span>"
        )

    # Date
    date = metadata.get("date", "")
    if date:
        meta_items.append(
            f"<span class='source-meta-item'><strong>Date:</strong> {_esc(date)}</span>"
        )

    # Sections (stored or extracted from text)
    sections_to_show: list[str] = []
    if section_number:
        sections_to_show = [section_number]
    else:
        sections_to_show = _extract_sections_from_text(text)

    if sections_to_show:
        sec_str = _esc(", ".join(f"S. {s}" for s in sections_to_show))
        meta_items.append(
            f"<span class='source-meta-item'><strong>Sections:</strong> {sec_str}</span>"
        )

    # Relevance score (from retrieval)
    final_score = metadata.get("final_score")
    if final_score is not None:
        score_pct = int(float(final_score) * 100)
        meta_items.append(
            f"<span class='source-meta-item'><strong>Relevance:</strong> {score_pct}%</span>"
        )

    # --- Source origin + URL ---
    origin = _esc(_get_source_origin(metadata, text))
    url = _build_source_url(metadata)

    # --- Build card HTML ---
    meta_html = ""
    if meta_items:
        meta_html = f'<div class="source-meta">{"".join(meta_items)}</div>'

    excerpt_html = ""
    if text:
        # For Q&A format, show the answer part more prominently
        display_text = text
        qa_match = _LEGAL_QA_RE.search(text)
        if qa_match:
            # Skip past "Legal Question: ..." to show the answer
            answer_start = text.find("Legal Answer:")
            if answer_start != -1:
                display_text = text[answer_start + len("Legal Answer:"):].strip()

        safe_text = _esc(display_text[:500])
        if len(display_text) > 500:
            safe_text += "..."
        excerpt_html = f'<div class="source-excerpt">{safe_text}</div>'

    # Make title clickable when a URL is available
    if url:
        title_rendered = (
            f'<a href="{_esc(url)}" target="_blank" '
            f'rel="noopener noreferrer" class="source-title-link">{title}</a>'
        )
    else:
        title_rendered = title

    # Source origin label (e.g. "Supreme Court of India")
    origin_html = f'<div class="source-origin-label">{origin}</div>'

    return (
        f'<div class="source-card">'
        f'  <div class="source-card-header">'
        f'    <span class="source-num">S{index}</span>'
        f'    <span class="source-type">{type_label}</span>'
        f"  </div>"
        f'  <div class="source-title">{title_rendered}</div>'
        f"  {origin_html}"
        f"  {meta_html}"
        f"  {excerpt_html}"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Panel renderer
# ---------------------------------------------------------------------------

def render_citation_panel(cited_sources: list[dict]) -> None:
    """Display all cited sources as styled cards.

    Parameters
    ----------
    cited_sources:
        Ordered list of source dicts with ``"text"`` and ``"metadata"``.
    """
    count = len(cited_sources)
    count_display = str(count) if count > 0 else "\u2014"

    st.markdown(
        f'<div class="sources-header">'
        f'  <span class="sources-title">Sources</span>'
        f'  <span class="sources-count">{count_display}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )

    if not cited_sources:
        st.markdown(
            '<div class="empty-state">'
            '  <div class="empty-state-icon">\U0001f4c4</div>'
            '  <div class="empty-state-text">'
            "    Sources will appear here<br>when you ask a question."
            "  </div>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    for idx, source in enumerate(cited_sources, start=1):
        source_idx = source.get("source_index", idx)
        card_html = _build_source_card_html(source, source_idx)
        st.markdown(card_html, unsafe_allow_html=True)
