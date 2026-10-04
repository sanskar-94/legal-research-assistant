"""Ingestion modules for parsing, cleaning, and extracting legal documents."""

from __future__ import annotations

from src.ingestion.metadata_extractor import DocumentMetadata, extract_metadata
from src.ingestion.pdf_parser import ParsedPage, parse_pdf
from src.ingestion.statute_parser import StatuteSection, parse_statute
from src.ingestion.text_cleaner import clean_legal_text

__all__ = [
    "clean_legal_text",
    "DocumentMetadata",
    "extract_metadata",
    "ParsedPage",
    "parse_pdf",
    "StatuteSection",
    "parse_statute",
]
