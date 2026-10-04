"""Main Streamlit application for the Indian Legal Research Assistant.

Run from the project root with:

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import logging

import streamlit as st

from app.components.chat_interface import render_answer, render_chat_history

logger = logging.getLogger(__name__)
from app.components.citation_viewer import render_citation_panel
from app.components.confidence_bar import render_confidence_bar
from app.components.export_handler import render_export_button

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Nyaya - Legal Research",
    page_icon="\u2696\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Master CSS - complete dark theme overhaul
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
/* ═══════════════════════════════════════════════════════════════════
   IMPORT FONTS
   ═══════════════════════════════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=Questrial&family=JetBrains+Mono:wght@400;500&display=swap');

/* ═══════════════════════════════════════════════════════════════════
   CSS VARIABLES
   ═══════════════════════════════════════════════════════════════════ */
:root {
    --bg-base: #f9fafb;
    --bg-raised: #ffffff;
    --bg-surface: #ffffff;
    --bg-overlay: #f3f4f6;
    --border-subtle: #e5e7eb;
    --border-default: #d1d5db;
    --border-hover: #9ca3af;
    --text-primary: #111827;
    --text-secondary: #4b5563;
    --text-tertiary: #6b7280;
    --accent: #b45309;
    --accent-hover: #92400e;
    --accent-dim: rgba(180, 83, 9, 0.10);
    --accent-glow: rgba(180, 83, 9, 0.06);
    --success: #16a34a;
    --warning: #d97706;
    --error: #dc2626;
    --info: #2563eb;
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --radius-xl: 20px;
    --font-body: 'Questrial', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'JetBrains Mono', 'SF Mono', monospace;
}

/* ═══════════════════════════════════════════════════════════════════
   GLOBAL RESETS
   ═══════════════════════════════════════════════════════════════════ */
.stApp {
    background-color: var(--bg-base) !important;
    font-family: var(--font-body) !important;
}

/* Hide streamlit branding */
#MainMenu, footer {
    display: none !important;
}
/* Hide most of the header but keep the sidebar toggle button visible */
[data-testid="stHeader"] {
    background: transparent !important;
}
[data-testid="stHeader"] .stAppDeployButton,
[data-testid="stHeader"] [data-testid="stHeaderActionElements"] {
    display: none !important;
}

/* ── Kill Streamlit tooltip / floating overlays ────────────── */
[data-testid="stTooltipHoverTarget"],
.stTooltipIcon,
div[data-floating-ui-portal] {
    display: none !important;
    visibility: hidden !important;
    pointer-events: none !important;
}
/* Prevent native title-tooltip hover text on all widgets */
[data-testid="stExpander"] summary,
section[data-testid="stSidebar"] button,
[data-testid="stChatInput"] button {
    pointer-events: auto;
}
/* Hide the small "?" help icons Streamlit adds */
.stElementToolbar, [data-testid="stElementToolbar"] {
    display: none !important;
}
/* Hide hover toolbars on elements */
[data-testid="StyledFullScreenButton"],
[data-testid="stElementToolbarButton"] {
    display: none !important;
}

/* Remove default padding */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 100px !important;
    max-width: 100% !important;
}

/* Selection colors */
::selection {
    background: var(--accent) !important;
    color: #000 !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: var(--border-default);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--border-hover);
}

/* ═══════════════════════════════════════════════════════════════════
   TYPOGRAPHY
   ═══════════════════════════════════════════════════════════════════ */
h1, h2, h3, h4, h5, h6,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: var(--font-body) !important;
    color: var(--text-primary) !important;
    font-weight: 400 !important;
    letter-spacing: -0.01em;
}

p, li, label {
    font-family: var(--font-body) !important;
}

code, .stCode {
    font-family: var(--font-mono) !important;
}

/* ═══════════════════════════════════════════════════════════════════
   SIDEBAR
   ═══════════════════════════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background-color: var(--bg-raised) !important;
    border-right: 1px solid var(--border-subtle) !important;
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.5rem !important;
}

section[data-testid="stSidebar"] .stMarkdown p {
    color: var(--text-secondary) !important;
    font-size: 0.88rem !important;
    line-height: 1.6 !important;
}

section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: 1px solid var(--border-subtle) !important;
    color: var(--text-secondary) !important;
    border-radius: var(--radius-md) !important;
    padding: 10px 14px !important;
    font-size: 0.85rem !important;
    font-family: var(--font-body) !important;
    text-align: left !important;
    width: 100% !important;
    white-space: normal !important;
    height: auto !important;
    transition: all 0.2s ease !important;
    line-height: 1.4 !important;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--accent-dim) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* Sidebar dividers */
section[data-testid="stSidebar"] hr {
    border-color: var(--border-subtle) !important;
    margin: 1rem 0 !important;
}

/* ═══════════════════════════════════════════════════════════════════
   CHAT MESSAGES
   ═══════════════════════════════════════════════════════════════════ */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 1rem 0 !important;
}

/* User messages */
[data-testid="stChatMessage"][data-testid-type="user"] {
    background: transparent !important;
}

/* Assistant messages */
[data-testid="stChatMessage"][data-testid-type="assistant"] {
    background: transparent !important;
}

/* Chat avatars */
[data-testid="stChatMessage"] [data-testid="stAvatar"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
}

/* ═══════════════════════════════════════════════════════════════════
   CHAT INPUT
   ═══════════════════════════════════════════════════════════════════ */
[data-testid="stChatInput"] {
    background-color: var(--bg-base) !important;
    border-top: 1px solid var(--border-subtle) !important;
}

[data-testid="stChatInput"] textarea {
    background-color: var(--bg-raised) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-lg) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
    font-size: 0.95rem !important;
    padding: 14px 18px !important;
}

[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent-dim) !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: var(--text-tertiary) !important;
}

[data-testid="stChatInput"] button {
    background: var(--accent) !important;
    color: #0c0c10 !important;
    border-radius: var(--radius-md) !important;
    border: none !important;
}

[data-testid="stChatInput"] button:hover {
    background: var(--accent-hover) !important;
}

/* ═══════════════════════════════════════════════════════════════════
   BUTTONS (GENERAL)
   ═══════════════════════════════════════════════════════════════════ */
.stButton > button {
    font-family: var(--font-body) !important;
    border-radius: var(--radius-md) !important;
    transition: all 0.2s ease !important;
    font-weight: 400 !important;
}

/* Primary action buttons */
.stDownloadButton > button {
    background: transparent !important;
    border: 1px solid var(--border-default) !important;
    color: var(--text-secondary) !important;
    border-radius: var(--radius-md) !important;
    font-family: var(--font-body) !important;
    font-size: 0.85rem !important;
    padding: 8px 16px !important;
}

.stDownloadButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: var(--accent-dim) !important;
}

/* ═══════════════════════════════════════════════════════════════════
   EXPANDERS
   ═══════════════════════════════════════════════════════════════════ */
.streamlit-expanderHeader {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
    font-size: 0.88rem !important;
}

.streamlit-expanderContent {
    background: var(--bg-raised) !important;
    border: 1px solid var(--border-subtle) !important;
    border-top: none !important;
    border-radius: 0 0 var(--radius-md) var(--radius-md) !important;
}

details {
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    background: var(--bg-surface) !important;
}

details summary {
    color: var(--text-primary) !important;
    font-size: 0.88rem !important;
}

details[open] summary {
    border-bottom: 1px solid var(--border-subtle) !important;
}

/* ═══════════════════════════════════════════════════════════════════
   PROGRESS BAR
   ═══════════════════════════════════════════════════════════════════ */
.stProgress > div > div {
    background-color: var(--bg-overlay) !important;
    border-radius: 4px !important;
}

/* ═══════════════════════════════════════════════════════════════════
   METRICS
   ═══════════════════════════════════════════════════════════════════ */
[data-testid="stMetric"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    padding: 12px 16px !important;
}

[data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-family: var(--font-body) !important;
}

[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
}

/* ═══════════════════════════════════════════════════════════════════
   ALERTS / INFO / WARNING
   ═══════════════════════════════════════════════════════════════════ */
[data-testid="stAlert"] {
    background-color: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-secondary) !important;
}

/* ═══════════════════════════════════════════════════════════════════
   SPINNER
   ═══════════════════════════════════════════════════════════════════ */
.stSpinner > div {
    border-top-color: var(--accent) !important;
}

/* ═══════════════════════════════════════════════════════════════════
   CUSTOM COMPONENTS
   ═══════════════════════════════════════════════════════════════════ */

/* Brand */
.nyaya-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 4px;
}
.nyaya-brand-icon {
    font-size: 1.8rem;
    line-height: 1;
}
.nyaya-brand-text {
    font-family: var(--font-body);
    font-size: 1.4rem;
    color: var(--text-primary);
    letter-spacing: 0.08em;
    font-weight: 400;
}
.nyaya-brand-sub {
    font-family: var(--font-body);
    font-size: 0.72rem;
    color: var(--text-tertiary);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 2px;
}

/* Sidebar section labels */
.sidebar-label {
    font-family: var(--font-body);
    font-size: 0.68rem;
    color: var(--text-tertiary);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 8px;
    margin-top: 4px;
}

/* Welcome hero */
.welcome-hero {
    text-align: center;
    padding: 4rem 2rem 2rem;
}
.welcome-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
    opacity: 0.6;
}
.welcome-title {
    font-family: var(--font-body);
    font-size: 1.8rem;
    color: var(--text-primary);
    font-weight: 400;
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
}
.welcome-subtitle {
    font-family: var(--font-body);
    font-size: 0.95rem;
    color: var(--text-tertiary);
    max-width: 480px;
    margin: 0 auto 2.5rem;
    line-height: 1.6;
}

/* Suggestion cards */
.suggestion-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    max-width: 640px;
    margin: 0 auto;
}
.suggestion-card {
    background: var(--bg-raised);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 14px 16px;
    font-family: var(--font-body);
    font-size: 0.84rem;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.2s ease;
    text-align: left;
    line-height: 1.45;
}
.suggestion-card:hover {
    border-color: var(--accent);
    color: var(--accent);
    background: var(--accent-dim);
}

/* Disclaimer - minimal */
.disclaimer-bar {
    background: var(--bg-raised);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 10px 16px;
    font-family: var(--font-body);
    font-size: 0.78rem;
    color: var(--text-tertiary);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.disclaimer-bar .disc-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--warning);
    flex-shrink: 0;
}

/* Source card */
.source-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
    padding: 16px;
    margin-bottom: 10px;
    transition: border-color 0.2s ease;
}
.source-card:hover {
    border-color: var(--border-hover);
}
.source-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 10px;
}
.source-num {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--accent);
    background: var(--accent-dim);
    padding: 3px 8px;
    border-radius: 6px;
    letter-spacing: 0.05em;
}
.source-type {
    font-family: var(--font-body);
    font-size: 0.68rem;
    color: var(--text-tertiary);
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.source-title {
    font-family: var(--font-body);
    font-size: 0.92rem;
    color: var(--text-primary);
    margin-bottom: 8px;
    line-height: 1.4;
}
.source-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 6px 14px;
    margin-bottom: 10px;
}
.source-meta-item {
    font-family: var(--font-body);
    font-size: 0.78rem;
    color: var(--text-tertiary);
}
.source-meta-item strong {
    color: var(--text-secondary);
    font-weight: 400;
}
.source-excerpt {
    font-family: var(--font-body);
    font-size: 0.82rem;
    color: var(--text-tertiary);
    line-height: 1.6;
    border-left: 2px solid var(--border-default);
    padding-left: 12px;
    margin-top: 10px;
}

/* Citation badge (inline) */
.cite-badge {
    display: inline-flex;
    align-items: center;
    background: var(--accent-dim);
    color: var(--accent);
    font-family: var(--font-mono);
    font-size: 0.7rem;
    padding: 1px 7px;
    border-radius: 5px;
    margin: 0 2px;
    font-weight: 500;
    letter-spacing: 0.02em;
    border: 1px solid rgba(196, 154, 108, 0.18);
    cursor: default;
    vertical-align: middle;
}

/* Source origin label in card */
.source-origin-label {
    font-family: var(--font-body);
    font-size: 0.78rem;
    color: var(--accent);
    font-weight: 500;
    margin-bottom: 8px;
    letter-spacing: 0.01em;
}

/* Clickable source title in card */
.source-title-link {
    color: var(--text-primary);
    text-decoration: none;
    border-bottom: 1px dashed var(--border-hover);
    transition: color 0.15s ease, border-color 0.15s ease;
}
.source-title-link:hover {
    color: var(--accent);
    border-bottom-color: var(--accent);
}

/* Source summary (below answer in chat) */
.source-summary {
    margin-top: 16px;
    padding: 14px 16px;
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
}
.source-summary-header {
    font-family: var(--font-body);
    font-size: 0.75rem;
    color: var(--text-tertiary);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 10px;
}
.source-summary-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 0;
    flex-wrap: wrap;
}
.source-summary-item + .source-summary-item {
    border-top: 1px solid var(--border-subtle);
}
.source-summary-badge {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--accent);
    background: var(--accent-dim);
    padding: 2px 7px;
    border-radius: 5px;
    letter-spacing: 0.04em;
    flex-shrink: 0;
}
.source-summary-origin {
    font-family: var(--font-body);
    font-size: 0.8rem;
    color: var(--accent);
    font-weight: 500;
    flex-shrink: 0;
}
.source-summary-sep {
    color: var(--text-tertiary);
    font-size: 0.75rem;
    flex-shrink: 0;
}
.source-link-text {
    font-family: var(--font-body);
    font-size: 0.8rem;
    color: var(--text-secondary);
    text-decoration: none;
    border-bottom: 1px dashed var(--border-hover);
    transition: color 0.15s ease, border-color 0.15s ease;
}
.source-link-text:hover {
    color: var(--accent);
    border-bottom-color: var(--accent);
}
.source-link-text-plain {
    font-family: var(--font-body);
    font-size: 0.8rem;
    color: var(--text-secondary);
}

/* Confidence display */
.confidence-wrap {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 12px;
    padding: 10px 14px;
    background: var(--bg-raised);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-md);
}
.confidence-score {
    font-family: var(--font-mono);
    font-size: 0.88rem;
    font-weight: 500;
}
.confidence-label {
    font-family: var(--font-body);
    font-size: 0.78rem;
    color: var(--text-tertiary);
}
.confidence-bar-track {
    flex: 1;
    height: 4px;
    background: var(--bg-overlay);
    border-radius: 2px;
    overflow: hidden;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.4s ease;
}

/* Sources panel header */
.sources-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border-subtle);
}
.sources-title {
    font-family: var(--font-body);
    font-size: 0.78rem;
    color: var(--text-tertiary);
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.sources-count {
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--text-tertiary);
    background: var(--bg-overlay);
    padding: 2px 8px;
    border-radius: 6px;
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 3rem 1.5rem;
}
.empty-state-icon {
    font-size: 2rem;
    margin-bottom: 0.8rem;
    opacity: 0.3;
}
.empty-state-text {
    font-family: var(--font-body);
    font-size: 0.85rem;
    color: var(--text-tertiary);
    line-height: 1.5;
}

/* Status indicator */
.status-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    margin-right: 6px;
}
.status-dot.active { background: var(--success); }
.status-dot.inactive { background: var(--text-tertiary); }

/* Sources column — left border + sticky + aligned to top */
[data-testid="stColumn"]:last-child {
    border-left: 1px solid var(--border-subtle) !important;
    padding-left: 1rem !important;
}
[data-testid="stColumn"]:last-child > [data-testid="stVerticalBlockBorderWrapper"] {
    position: sticky !important;
    top: 1rem !important;
    max-height: 90vh !important;
    overflow-y: auto !important;
}

/* Chat input — always pinned at bottom, full width */
[data-testid="stBottom"] {
    background: var(--bg-base) !important;
    border-top: 1px solid var(--border-subtle) !important;
    z-index: 100 !important;
}


/* Markdown in chat */
[data-testid="stChatMessage"] .stMarkdown {
    color: var(--text-primary) !important;
    font-size: 0.92rem !important;
    line-height: 1.7 !important;
}

[data-testid="stChatMessage"] .stMarkdown p {
    color: var(--text-primary) !important;
}

[data-testid="stChatMessage"] .stMarkdown blockquote {
    border-left: 2px solid var(--border-default) !important;
    color: var(--text-secondary) !important;
    padding-left: 12px !important;
}

[data-testid="stChatMessage"] .stMarkdown code {
    background: var(--bg-surface) !important;
    color: var(--accent) !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.84rem !important;
}

/* Stop button */
.stop-btn-wrap {
    display: flex;
    justify-content: center;
    margin: 12px 0;
}
.stop-btn-wrap .stButton > button {
    background: transparent !important;
    border: 1px solid var(--error) !important;
    color: var(--error) !important;
    border-radius: var(--radius-lg) !important;
    padding: 6px 24px !important;
    font-size: 0.82rem !important;
    font-family: var(--font-body) !important;
    transition: all 0.2s ease !important;
}
.stop-btn-wrap .stButton > button:hover {
    background: rgba(248, 113, 113, 0.1) !important;
}

/* subtle separator */
.sep-line {
    border: none;
    border-top: 1px solid var(--border-subtle);
    margin: 12px 0;
}

/* Custom Animated Spinner */
.nyaya-spinner {
    display: inline-block;
    width: 14px;
    height: 14px;
    border: 2px solid var(--border-subtle);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: nyaya-spin 0.8s linear infinite;
    vertical-align: middle;
    margin-right: 8px;
}
@keyframes nyaya-spin {
    to { transform: rotate(360deg); }
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Example queries
# ---------------------------------------------------------------------------
EXAMPLE_QUERIES: list[str] = [
    "What is Section 302 of IPC and its BNS equivalent?",
    "Explain the Kesavananda Bharati case and its significance",
    "What are the fundamental rights under the Indian Constitution?",
    "Explain the doctrine of basic structure",
    "What is the punishment for theft under IPC?",
]

RESEARCH_TOPICS: list[dict[str, str]] = [
    {
        "icon": "\U0001f4dc",
        "label": "Constitutional Law",
        "query": "What are the fundamental rights under the Indian Constitution?",
    },
    {
        "icon": "\u2696\ufe0f",
        "label": "Criminal Law",
        "query": "What is Section 302 of IPC and its BNS equivalent?",
    },
    {
        "icon": "\U0001f3db\ufe0f",
        "label": "Landmark Cases",
        "query": "Explain the Kesavananda Bharati case and its significance",
    },
    {
        "icon": "\U0001f4d6",
        "label": "Legal Doctrine",
        "query": "Explain the doctrine of basic structure",
    },
]

# ---------------------------------------------------------------------------
# Pipeline initialisation
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Initialising research engine...")
def _load_pipeline_v2():
    """Attempt to import and initialise the LegalResearchPipeline."""
    try:
        from src.pipeline import LegalResearchPipeline

        return LegalResearchPipeline()
    except ImportError:
        logger.warning("LegalResearchPipeline not available — running in UI-only mode.")
        return None
    except Exception:
        logger.exception("Pipeline initialisation failed.")
        return None


pipeline = _load_pipeline_v2()

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
for key, default in {
    "messages": [],
    "current_sources": [],
    "current_confidence": None,
    "current_retrieved_chunks": None,
    "current_query": "",
    "pending_example": None,
    "gen_active": None,  # dict with thread + result state, or None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Sidebar - minimal redesign
# ---------------------------------------------------------------------------
with st.sidebar:
    # Brand
    st.markdown(
        """
        <div class="nyaya-brand">
            <div class="nyaya-brand-icon">\u2696\ufe0f</div>
            <div>
                <div class="nyaya-brand-text">Nyaya</div>
                <div class="nyaya-brand-sub">Legal Research Assistant</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<hr class="sep-line">', unsafe_allow_html=True)

    # New research button
    if st.button("\u2728  New Research", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_sources = []
        st.session_state.current_confidence = None
        st.session_state.current_retrieved_chunks = None
        st.session_state.current_query = ""
        st.rerun()

    st.markdown('<hr class="sep-line">', unsafe_allow_html=True)

    # Research topics
    st.markdown(
        '<div class="sidebar-label">Explore Topics</div>', unsafe_allow_html=True
    )

    for topic in RESEARCH_TOPICS:
        if st.button(
            f"{topic['icon']}  {topic['label']}",
            key=f"topic_{hash(topic['label'])}",
            use_container_width=True,
        ):
            st.session_state.pending_example = topic["query"]

    st.markdown('<hr class="sep-line">', unsafe_allow_html=True)

    # Pipeline status
    pipeline_active = pipeline is not None
    dot_class = "active" if pipeline_active else "inactive"
    status_text = "Engine online" if pipeline_active else "Engine offline"
    st.markdown(
        f'<div style="display:flex;align-items:center;padding:4px 0;">'
        f'<span class="status-dot {dot_class}"></span>'
        f'<span style="font-size:0.78rem;color:var(--text-tertiary);">{status_text}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main layout: Chat (75%) | Sources (25%)
# ---------------------------------------------------------------------------
col_chat, col_sources = st.columns([3, 1], gap="medium")

# -- Chat column -----------------------------------------------------------
with col_chat:
    # Minimal disclaimer
    st.markdown(
        '<div class="disclaimer-bar">'
        '<span class="disc-dot"></span>'
        "Research tool only \u2014 verify sources and consult a qualified advocate."
        "</div>",
        unsafe_allow_html=True,
    )

    # Pipeline readiness
    if pipeline is None:
        st.markdown(
            '<div style="background:var(--bg-raised);border:1px solid var(--border-subtle);'
            "border-radius:var(--radius-md);padding:14px 18px;color:var(--text-tertiary);"
            'font-size:0.85rem;margin-bottom:1rem;">'
            "The research pipeline is not available. The vector database may not be populated, "
            "or required dependencies are missing. You can explore the UI."
            "</div>",
            unsafe_allow_html=True,
        )

    # Render existing conversation
    if st.session_state.messages:
        render_chat_history(st.session_state.messages)

    # Welcome state when no messages
    if not st.session_state.messages:
        st.markdown(
            """
            <div class="welcome-hero">
                <div class="welcome-icon">\u2696\ufe0f</div>
                <div class="welcome-title">What would you like to research?</div>
                <div class="welcome-subtitle">
                    Ask about Indian law \u2014 statutes, landmark judgements,
                    constitutional provisions, and legal doctrines.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Suggestion cards
        suggestion_cols = st.columns(2)
        for idx, query in enumerate(EXAMPLE_QUERIES):
            with suggestion_cols[idx % 2]:
                if st.button(query, key=f"sug_{hash(query)}", use_container_width=True):
                    st.session_state.pending_example = query

# -- Sources column --------------------------------------------------------
with col_sources:
    # Always display the retrieved chunks in the sources panel to ensure users can see all context,
    # even if the LLM fails to formally cite them with [SOURCE_N] tags. Fallback to cited sources.
    sources_to_display = (
        st.session_state.current_retrieved_chunks
        or st.session_state.current_sources
        or []
    )
    render_citation_panel(sources_to_display)

    if (
        (st.session_state.current_retrieved_chunks or st.session_state.current_sources)
        and st.session_state.current_query
        and st.session_state.messages
    ):
        st.markdown('<hr class="sep-line">', unsafe_allow_html=True)
        latest_answer = ""
        for msg in reversed(st.session_state.messages):
            if msg["role"] == "assistant":
                latest_answer = msg["content"]
                break

        render_export_button(
            query=st.session_state.current_query,
            answer=latest_answer,
            sources=st.session_state.current_retrieved_chunks
            or st.session_state.current_sources,
            confidence=st.session_state.current_confidence,
        )

# ---------------------------------------------------------------------------
# Chat input — at page root level so Streamlit pins it to viewport bottom
# ---------------------------------------------------------------------------
user_input = st.chat_input("Ask about Indian law...")

# Resolve active query
active_query = None
if st.session_state.pending_example is not None:
    active_query = st.session_state.pending_example
    st.session_state.pending_example = None
elif user_input:
    active_query = user_input

# ---------------------------------------------------------------------------
# Generation state machine
# ---------------------------------------------------------------------------
import threading
import time as _time

# Step 1: Launch generation when a new query arrives
if active_query and st.session_state.gen_active is None:
    st.session_state.messages.append({"role": "user", "content": active_query})
    st.session_state.current_query = active_query

    if pipeline is not None:
        gen_state = {"done": False, "result": None, "error": None}

        def _run_query(q=active_query):
            try:
                gen_state["result"] = pipeline.query(q)
            except Exception as exc:
                gen_state["error"] = exc
            finally:
                gen_state["done"] = True

        thread = threading.Thread(target=_run_query, daemon=True)
        thread.start()
        st.session_state.gen_active = gen_state
        st.rerun()
    else:
        # Pipeline not available
        with col_chat:
            with st.chat_message("user"):
                st.markdown(active_query)
            with st.chat_message("assistant"):
                placeholder = (
                    f"The research pipeline is not yet available. "
                    f"Once implemented and indexed, this will retrieve relevant "
                    f"legal sources and generate cited answers."
                    f"\n\n**Your query:** {active_query}"
                )
                st.info(placeholder)
                st.session_state.messages.append(
                    {"role": "assistant", "content": placeholder}
                )
                st.session_state.current_sources = []
                st.session_state.current_confidence = None
                st.session_state.current_retrieved_chunks = None
        st.rerun()

# Step 2: Poll active generation (runs on each rerun)
gen = st.session_state.gen_active
if gen is not None:
    with col_chat:
        if gen["done"]:
            # --- Generation complete: render result ---
            if gen["error"] is not None:
                logger.exception(
                    "Query processing failed for: %s",
                    st.session_state.current_query,
                )
                with st.chat_message("assistant"):
                    st.error(
                        "An error occurred while processing your query. "
                        "Please try again."
                    )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "An error occurred. Please try again.",
                    }
                )
            else:
                result = gen["result"]
                answer_text = result.answer_text or "No answer generated."
                sources = [
                    s.model_dump() if hasattr(s, "model_dump") else s
                    for s in result.cited_sources
                ]
                confidence = result.confidence
                error = result.error
                retrieved_chunks = getattr(result, "retrieved_chunks", [])

                # --- Renumbering logic to ensure perfect sequential numbering (S1, S2, ...) ---
                display_list = retrieved_chunks if retrieved_chunks else sources
                if display_list:
                    import re

                    index_map = {}
                    for new_idx, item in enumerate(display_list, start=1):
                        # Safely handle if item is a dict or an object
                        if isinstance(item, dict):
                            old_idx = item.get("source_index")
                            if old_idx is not None:
                                try:
                                    index_map[int(old_idx)] = new_idx
                                except (ValueError, TypeError):
                                    pass
                            item["source_index"] = new_idx
                        else:
                            old_idx = getattr(item, "source_index", None)
                            if old_idx is not None:
                                try:
                                    index_map[int(old_idx)] = new_idx
                                except (ValueError, TypeError):
                                    pass
                            try:
                                setattr(item, "source_index", new_idx)
                            except AttributeError:
                                pass

                    if retrieved_chunks and sources:
                        for src in sources:
                            if isinstance(src, dict):
                                old_idx = src.get("source_index")
                                try:
                                    if (
                                        old_idx is not None
                                        and int(old_idx) in index_map
                                    ):
                                        src["source_index"] = index_map[int(old_idx)]
                                except (ValueError, TypeError):
                                    pass
                            else:
                                old_idx = getattr(src, "source_index", None)
                                try:
                                    if (
                                        old_idx is not None
                                        and int(old_idx) in index_map
                                    ):
                                        setattr(
                                            src, "source_index", index_map[int(old_idx)]
                                        )
                                except (ValueError, TypeError, AttributeError):
                                    pass

                    if index_map and answer_text:

                        def replace_tag(match):
                            old_idx = int(match.group(1))
                            new_idx = index_map.get(old_idx, old_idx)
                            return f"[SOURCE_{new_idx}]"

                        answer_text = re.sub(
                            r"\[SOURCE_(\d+)\]", replace_tag, answer_text
                        )
                # -----------------------------------------------------------------------------

                with st.chat_message("assistant"):
                    if error and not answer_text.strip():
                        st.warning(error)
                    render_answer(answer_text, sources)
                    if confidence is not None:
                        render_confidence_bar(
                            confidence,
                            answer_text=answer_text,
                            retrieved_chunks=retrieved_chunks,
                        )

                st.session_state.current_sources = sources
                st.session_state.current_confidence = confidence
                st.session_state.current_retrieved_chunks = retrieved_chunks
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer_text,
                        "sources": sources,
                        "confidence": confidence,
                        "retrieved_chunks": retrieved_chunks,
                    }
                )

            # Clear generation state and refresh
            st.session_state.gen_active = None
            st.rerun()

        else:
            # --- Still generating: show spinner + stop button ---
            st.markdown(
                '<div style="color:var(--text-tertiary);font-size:0.88rem;'
                'padding:1rem 0 0.5rem; display:flex; align-items:center;">'
                '<div class="nyaya-spinner"></div>'
                "Searching legal documents and generating answer..."
                "</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="stop-btn-wrap">',
                unsafe_allow_html=True,
            )
            if st.button("\u23f9  Stop generating", key="stop_gen"):
                # Cancel: remove the pending user message
                if (
                    st.session_state.messages
                    and st.session_state.messages[-1].get("role") == "user"
                ):
                    st.session_state.messages.pop()
                st.session_state.gen_active = None
                st.session_state.current_sources = []
                st.session_state.current_confidence = None
                st.session_state.current_retrieved_chunks = None
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

            # Poll: wait briefly then rerun to check again
            _time.sleep(1)
            st.rerun()
