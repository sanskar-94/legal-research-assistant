"""Two-pass PDF parsing for Indian legal documents.

Pass 1: PyMuPDF (fitz) for digital/native PDFs.
Pass 2: pytesseract OCR fallback for scanned pages when text
        extraction yields fewer than OCR_CHAR_THRESHOLD characters.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Minimum characters per page to consider digital extraction successful.
# Pages with fewer characters trigger the OCR fallback.
OCR_CHAR_THRESHOLD: Final[int] = 50


class ParsedPage(BaseModel):
    """A single page extracted from a PDF document."""

    page_number: int = Field(description="1-based page number")
    text: str = Field(description="Extracted text content")
    is_ocr: bool = Field(
        default=False,
        description="True if text was obtained via OCR fallback",
    )

    model_config = {"frozen": True}


def _extract_text_fitz(pdf_path: Path) -> list[tuple[int, str]]:
    """Extract text from all pages using PyMuPDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of (1-based page number, extracted text) tuples.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required for PDF parsing. "
            "Install it with: pip install PyMuPDF"
        ) from exc

    pages: list[tuple[int, str]] = []
    doc = fitz.open(str(pdf_path))
    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            text = page.get_text("text") or ""
            pages.append((page_index + 1, text))
    finally:
        doc.close()

    return pages


def _ocr_page(pdf_path: Path, page_index_0based: int) -> str:
    """Run OCR on a single PDF page using pytesseract.

    Renders the page as a high-resolution image and feeds it
    to Tesseract for text extraction.

    Args:
        pdf_path: Path to the PDF file.
        page_index_0based: Zero-based page index.

    Returns:
        OCR-extracted text for the page.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required for PDF parsing. "
            "Install it with: pip install PyMuPDF"
        ) from exc

    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "pytesseract and Pillow are required for OCR. "
            "Install them with: pip install pytesseract Pillow"
        ) from exc

    doc = fitz.open(str(pdf_path))
    try:
        page = doc[page_index_0based]
        # Render at 300 DPI for good OCR quality
        matrix = fitz.Matrix(300 / 72, 300 / 72)
        pixmap = page.get_pixmap(matrix=matrix)

        # Convert pixmap to PIL Image without writing to disk
        img = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)

        # Run Tesseract with English language
        text = pytesseract.image_to_string(img, lang="eng")
        return text or ""
    finally:
        doc.close()


def parse_pdf(pdf_path: Path) -> list[ParsedPage]:
    """Parse a PDF document using a two-pass strategy.

    Pass 1: Extract text using PyMuPDF (fast, works for digital PDFs).
    Pass 2: For pages where Pass 1 yields fewer than 50 characters,
            fall back to OCR via pytesseract.

    Args:
        pdf_path: Path to the PDF file to parse.

    Returns:
        List of ParsedPage objects, one per page, ordered by page number.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ImportError: If required libraries are not installed.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not pdf_path.suffix.lower() == ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {pdf_path.suffix}")

    logger.info("Parsing PDF: %s", pdf_path)

    # Pass 1: digital text extraction
    raw_pages = _extract_text_fitz(pdf_path)
    logger.info(
        "Pass 1 complete: extracted %d pages from %s",
        len(raw_pages), pdf_path.name,
    )

    parsed_pages: list[ParsedPage] = []
    ocr_count = 0

    for page_number, text in raw_pages:
        stripped = text.strip()

        if len(stripped) >= OCR_CHAR_THRESHOLD:
            # Digital extraction succeeded
            parsed_pages.append(
                ParsedPage(
                    page_number=page_number,
                    text=stripped,
                    is_ocr=False,
                )
            )
        else:
            # Pass 2: OCR fallback
            logger.debug(
                "Page %d has only %d chars, falling back to OCR",
                page_number, len(stripped),
            )
            try:
                ocr_text = _ocr_page(pdf_path, page_number - 1)
                ocr_count += 1
                parsed_pages.append(
                    ParsedPage(
                        page_number=page_number,
                        text=ocr_text.strip(),
                        is_ocr=True,
                    )
                )
            except ImportError:
                logger.warning(
                    "pytesseract not available; skipping OCR for page %d. "
                    "Install pytesseract and Tesseract OCR engine for "
                    "scanned PDF support.",
                    page_number,
                )
                # Keep whatever minimal text we got from Pass 1
                parsed_pages.append(
                    ParsedPage(
                        page_number=page_number,
                        text=stripped,
                        is_ocr=False,
                    )
                )
            except Exception:
                logger.exception(
                    "OCR failed for page %d of %s",
                    page_number, pdf_path.name,
                )
                parsed_pages.append(
                    ParsedPage(
                        page_number=page_number,
                        text=stripped,
                        is_ocr=False,
                    )
                )

    if ocr_count > 0:
        logger.info(
            "Pass 2 complete: OCR applied to %d/%d pages",
            ocr_count, len(raw_pages),
        )

    return parsed_pages
