"""Confidence score display component for the Legal Research Assistant.

Renders a minimal indicator with a thin
progress bar and optional expandable breakdown that now includes evaluation metrics.
"""

from __future__ import annotations

from typing import Any, Optional

import streamlit as st

from src.evaluation.citation_metrics import (
    compute_citation_quality,
    detect_hallucinations,
    is_substantive_response,
)


def _level_label(score: float) -> str:
    """Return a human-readable confidence level label."""
    if score >= 0.80:
        return "High"
    if score >= 0.60:
        return "Medium"
    if score >= 0.40:
        return "Low"
    return "Very Low"


def _level_color(score: float) -> str:
    """Return a hex colour appropriate for the score."""
    if score >= 0.80:
        return "#4ade80"  # green
    if score >= 0.60:
        return "#fbbf24"  # amber
    if score >= 0.40:
        return "#fb923c"  # orange
    return "#f87171"  # red


def render_confidence_bar(
    confidence_result: Any,
    answer_text: Optional[str] = None,
    retrieved_chunks: Optional[list[Any]] = None,
) -> None:
    """Render a minimal confidence indicator with evaluation metrics in the breakdown.

    Parameters
    ----------
    confidence_result:
        A ``float``, ``dict`` with ``overall_score``/``breakdown``, or
        an object with those attributes.
    answer_text:
        The generated text to run evaluation metrics on.
    retrieved_chunks:
        The context chunks used for evaluation metrics.
    """
    if isinstance(confidence_result, (int, float)):
        overall = float(confidence_result)
        breakdown: dict[str, float] = {}
    elif isinstance(confidence_result, dict):
        overall = float(confidence_result.get("overall_score", 0.0))
        breakdown = confidence_result.get("breakdown", {})
    else:
        overall = float(getattr(confidence_result, "overall_score", 0.0))
        breakdown = getattr(confidence_result, "breakdown", {})
        if not isinstance(breakdown, dict):
            breakdown = {}

    overall = max(0.0, min(1.0, overall))
    pct = int(overall * 100)
    label = _level_label(overall)
    color = _level_color(overall)
    width_pct = max(2, pct)  # minimum visual width

    # Build the breakdown rows HTML if data is available
    breakdown_html = ""
    rows = []

    if breakdown:
        for signal_name, raw_value in breakdown.items():
            signal_value = max(0.0, min(1.0, float(raw_value)))
            nice_name = signal_name.replace("_", " ").title()
            s_pct = int(signal_value * 100)
            s_color = _level_color(signal_value)
            s_width = max(2, s_pct)
            rows.append(
                f'<div style="display:flex;align-items:center;gap:10px;margin:6px 0;">'
                f'  <span style="font-size:0.78rem;color:var(--text-secondary);min-width:120px;">{nice_name}</span>'
                f'  <div style="flex:1;height:3px;background:var(--bg-overlay);border-radius:2px;overflow:hidden;">'
                f'    <div style="width:{s_width}%;height:100%;background:{s_color};border-radius:2px;"></div>'
                f"  </div>"
                f'  <span style="font-family:var(--font-mono);font-size:0.72rem;color:var(--text-tertiary);min-width:30px;text-align:right;">{s_pct}%</span>'
                f"</div>"
            )

    # Append Evaluation Metrics if text and chunks are provided
    if answer_text is not None and retrieved_chunks is not None:
        citation_quality = compute_citation_quality(answer_text, retrieved_chunks)
        hallucinations = detect_hallucinations(answer_text, retrieved_chunks)
        is_substantive = is_substantive_response(answer_text, len(retrieved_chunks))

        hallucination_text = (
            "None detected" if not hallucinations else f"{len(hallucinations)} found"
        )
        hallucination_color = "#4ade80" if not hallucinations else "#f87171"

        substantive_text = "Yes" if is_substantive else "No"
        substantive_color = "#4ade80" if is_substantive else "#fbbf24"

        if rows:
            # Separator between confidence breakdown and evaluation metrics
            rows.append(
                f'<div style="margin: 12px 0 8px; padding-top: 10px; border-top: 1px dashed var(--border-subtle);">'
                f'  <span style="font-size:0.68rem;font-weight:600;color:var(--text-tertiary);text-transform:uppercase;letter-spacing:0.05em;">Evaluation Metrics</span>'
                f"</div>"
            )

        metric_labels = {
            "citation_validity": "Citation Validity",
            "cited_claim_coverage": "Claim Coverage",
            "citation_support_score": "Citation Support",
            "citation_quality_score": "Citation Quality",
        }
        for metric_name, metric_label in metric_labels.items():
            metric_value = citation_quality[metric_name]
            metric_pct = int(metric_value * 100)
            metric_color = _level_color(metric_value)
            rows.append(
                f'<div style="display:flex;align-items:center;gap:10px;margin:6px 0;">'
                f'  <span style="font-size:0.78rem;color:var(--text-secondary);min-width:120px;">{metric_label}</span>'
                f'  <div style="flex:1;height:3px;background:var(--bg-overlay);border-radius:2px;overflow:hidden;">'
                f'    <div style="width:{max(2, metric_pct)}%;height:100%;background:{metric_color};border-radius:2px;"></div>'
                f"  </div>"
                f'  <span style="font-family:var(--font-mono);font-size:0.72rem;color:var(--text-tertiary);min-width:30px;text-align:right;">{metric_pct}%</span>'
                f"</div>"
            )
        rows.append(
            f'<div style="display:flex;align-items:center;gap:10px;margin:6px 0;">'
            f'  <span style="font-size:0.78rem;color:var(--text-secondary);min-width:120px;">Hallucinations</span>'
            f'  <div style="flex:1;text-align:right;">'
            f'    <span style="font-size:0.78rem;color:{hallucination_color};">{hallucination_text}</span>'
            f"  </div>"
            f"</div>"
        )
        rows.append(
            f'<div style="display:flex;align-items:center;gap:10px;margin:6px 0;">'
            f'  <span style="font-size:0.78rem;color:var(--text-secondary);min-width:120px;">Substantive</span>'
            f'  <div style="flex:1;text-align:right;">'
            f'    <span style="font-size:0.78rem;color:{substantive_color};">{substantive_text}</span>'
            f"  </div>"
            f"</div>"
        )

    if rows:
        breakdown_html = (
            f'<details style="border:1px solid var(--border-subtle);border-radius:var(--radius-md);'
            f'background:var(--bg-surface);margin-top:8px;" open>'
            f'<summary style="color:var(--text-primary);font-size:0.88rem;padding:10px 14px;cursor:pointer;'
            f'list-style:none;display:flex;align-items:center;gap:6px;">'
            f'<span style="font-size:0.7rem;transition:transform 0.2s;">&#9660;</span> Score breakdown</summary>'
            f'<div style="padding:8px 14px 12px;border-top:1px solid var(--border-subtle);">'
            + "".join(rows)
            + "</div></details>"
        )

    # Render everything as a single HTML block to avoid
    # Streamlit inserting wrapper elements between widgets.
    st.markdown(
        f'<div class="confidence-wrap">'
        f'  <span class="confidence-score" style="color:{color};">{pct}%</span>'
        f'  <div class="confidence-bar-track">'
        f'    <div class="confidence-bar-fill" style="width:{width_pct}%;background:{color};"></div>'
        f"  </div>"
        f'  <span class="confidence-label">{label}</span>'
        f"</div>"
        f"{breakdown_html}",
        unsafe_allow_html=True,
    )
