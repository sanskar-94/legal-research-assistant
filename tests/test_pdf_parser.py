"""Tests for src/ingestion/pdf_parser.py.

Covers digital PDF text extraction, OCR fallback detection, handling
of non-existent files, and empty PDF behaviour.  All heavy dependencies
(PyMuPDF, pytesseract, Pillow) are mocked.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.ingestion.pdf_parser import OCR_CHAR_THRESHOLD, ParsedPage, parse_pdf


# Helpers

def _make_fitz_page(text: str) -> MagicMock:
    """Create a mock fitz page that returns *text* from get_text."""
    page = MagicMock()
    page.get_text.return_value = text
    page.get_pixmap.return_value = MagicMock(
        width=100, height=100, samples=b"\x00" * (100 * 100 * 3),
    )
    return page


def _make_fitz_doc(pages_text: list[str]) -> MagicMock:
    """Create a mock fitz.Document with pages returning *pages_text*."""
    mock_pages = [_make_fitz_page(t) for t in pages_text]
    doc = MagicMock()
    doc.__len__ = lambda self: len(mock_pages)
    doc.__getitem__ = lambda self, idx: mock_pages[idx]
    doc.close = MagicMock()
    return doc


# Fixtures

@pytest.fixture()
def pdf_path(tmp_path: Path) -> Path:
    """Return a path to a dummy .pdf file (just needs to exist)."""
    p = tmp_path / "test_document.pdf"
    p.write_bytes(b"%PDF-1.4 dummy")
    return p


@pytest.fixture()
def nonexistent_pdf(tmp_path: Path) -> Path:
    return tmp_path / "does_not_exist.pdf"


# Tests: digital PDF text extraction

class TestDigitalPDFExtraction:
    """Tests for the happy path where all pages have sufficient text."""

    @patch("src.ingestion.pdf_parser.fitz", create=True)
    def test_single_page_digital_pdf(
        self, mock_fitz: MagicMock, pdf_path: Path
    ) -> None:
        page_text = "This is a legal document with more than fifty characters of meaningful content."
        mock_doc = _make_fitz_doc([page_text])
        mock_fitz.open.return_value = mock_doc

        with patch("src.ingestion.pdf_parser._extract_text_fitz") as mock_extract:
            mock_extract.return_value = [(1, page_text)]
            result = parse_pdf(pdf_path)

        assert len(result) == 1
        assert isinstance(result[0], ParsedPage)
        assert result[0].page_number == 1
        assert result[0].text == page_text.strip()
        assert result[0].is_ocr is False

    @patch("src.ingestion.pdf_parser._extract_text_fitz")
    def test_multi_page_digital_pdf(
        self, mock_extract: MagicMock, pdf_path: Path
    ) -> None:
        texts = [
            "First page with well over fifty characters of meaningful legal content here.",
            "Second page discussing Section 302 of IPC and its interpretation by courts.",
            "Third page with the final order and disposition of the appeal by the bench.",
        ]
        mock_extract.return_value = [(i + 1, t) for i, t in enumerate(texts)]

        result = parse_pdf(pdf_path)

        assert len(result) == 3
        for i, page in enumerate(result):
            assert page.page_number == i + 1
            assert page.is_ocr is False
            assert page.text == texts[i].strip()

    @patch("src.ingestion.pdf_parser._extract_text_fitz")
    def test_page_text_is_stripped(
        self, mock_extract: MagicMock, pdf_path: Path
    ) -> None:
        raw = "   Plenty of whitespace around this long legal text content.   "
        mock_extract.return_value = [(1, raw)]

        result = parse_pdf(pdf_path)
        assert result[0].text == raw.strip()


# Tests: OCR fallback detection

class TestOCRFallback:
    """Verify OCR is triggered when a page has fewer than OCR_CHAR_THRESHOLD chars."""

    @patch("src.ingestion.pdf_parser._ocr_page")
    @patch("src.ingestion.pdf_parser._extract_text_fitz")
    def test_ocr_triggered_for_short_text(
        self,
        mock_extract: MagicMock,
        mock_ocr: MagicMock,
        pdf_path: Path,
    ) -> None:
        short_text = "ab"  # well under 50 chars
        ocr_text = "Full OCR result with more than fifty characters from the scanned page image."
        mock_extract.return_value = [(1, short_text)]
        mock_ocr.return_value = ocr_text

        result = parse_pdf(pdf_path)

        assert len(result) == 1
        assert result[0].is_ocr is True
        assert result[0].text == ocr_text.strip()
        mock_ocr.assert_called_once_with(pdf_path, 0)

    @patch("src.ingestion.pdf_parser._ocr_page")
    @patch("src.ingestion.pdf_parser._extract_text_fitz")
    def test_no_ocr_for_sufficient_text(
        self,
        mock_extract: MagicMock,
        mock_ocr: MagicMock,
        pdf_path: Path,
    ) -> None:
        long_text = "A" * (OCR_CHAR_THRESHOLD + 10)
        mock_extract.return_value = [(1, long_text)]

        result = parse_pdf(pdf_path)

        assert result[0].is_ocr is False
        mock_ocr.assert_not_called()

    @patch("src.ingestion.pdf_parser._ocr_page")
    @patch("src.ingestion.pdf_parser._extract_text_fitz")
    def test_ocr_exactly_at_threshold(
        self,
        mock_extract: MagicMock,
        mock_ocr: MagicMock,
        pdf_path: Path,
    ) -> None:
        """Text with exactly OCR_CHAR_THRESHOLD chars should NOT trigger OCR."""
        exact_text = "A" * OCR_CHAR_THRESHOLD
        mock_extract.return_value = [(1, exact_text)]

        result = parse_pdf(pdf_path)

        assert result[0].is_ocr is False
        mock_ocr.assert_not_called()

    @patch("src.ingestion.pdf_parser._ocr_page")
    @patch("src.ingestion.pdf_parser._extract_text_fitz")
    def test_mixed_digital_and_scanned_pages(
        self,
        mock_extract: MagicMock,
        mock_ocr: MagicMock,
        pdf_path: Path,
    ) -> None:
        mock_extract.return_value = [
            (1, "Sufficient digital text content that has more than the threshold."),
            (2, "x"),  # triggers OCR
            (3, "Another page with plenty of text content above the threshold."),
        ]
        mock_ocr.return_value = "OCR text for the scanned page with plenty of text content."

        result = parse_pdf(pdf_path)

        assert len(result) == 3
        assert result[0].is_ocr is False
        assert result[1].is_ocr is True
        assert result[2].is_ocr is False
        mock_ocr.assert_called_once_with(pdf_path, 1)

    @patch("src.ingestion.pdf_parser._ocr_page")
    @patch("src.ingestion.pdf_parser._extract_text_fitz")
    def test_ocr_import_error_graceful_fallback(
        self,
        mock_extract: MagicMock,
        mock_ocr: MagicMock,
        pdf_path: Path,
    ) -> None:
        """When pytesseract is not installed, OCR should fail gracefully."""
        mock_extract.return_value = [(1, "short")]
        mock_ocr.side_effect = ImportError("pytesseract not installed")

        result = parse_pdf(pdf_path)

        assert len(result) == 1
        assert result[0].is_ocr is False
        assert result[0].text == "short"

    @patch("src.ingestion.pdf_parser._ocr_page")
    @patch("src.ingestion.pdf_parser._extract_text_fitz")
    def test_ocr_runtime_error_graceful_fallback(
        self,
        mock_extract: MagicMock,
        mock_ocr: MagicMock,
        pdf_path: Path,
    ) -> None:
        """When OCR fails at runtime, should fall back to digital text."""
        mock_extract.return_value = [(1, "tiny")]
        mock_ocr.side_effect = RuntimeError("Tesseract binary not found")

        result = parse_pdf(pdf_path)

        assert len(result) == 1
        assert result[0].is_ocr is False
        assert result[0].text == "tiny"


# Tests: non-existent and invalid files

class TestFileErrors:

    def test_nonexistent_file_raises(self, nonexistent_pdf: Path) -> None:
        with pytest.raises(FileNotFoundError, match="PDF file not found"):
            parse_pdf(nonexistent_pdf)

    def test_non_pdf_extension_raises(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "document.txt"
        txt_file.write_text("not a pdf")
        with pytest.raises(ValueError, match="Expected a .pdf file"):
            parse_pdf(txt_file)


# Tests: empty PDF

class TestEmptyPDF:

    @patch("src.ingestion.pdf_parser._ocr_page")
    @patch("src.ingestion.pdf_parser._extract_text_fitz")
    def test_empty_pdf_no_pages(
        self,
        mock_extract: MagicMock,
        mock_ocr: MagicMock,
        pdf_path: Path,
    ) -> None:
        mock_extract.return_value = []

        result = parse_pdf(pdf_path)

        assert result == []
        mock_ocr.assert_not_called()

    @patch("src.ingestion.pdf_parser._ocr_page")
    @patch("src.ingestion.pdf_parser._extract_text_fitz")
    def test_pdf_with_all_blank_pages(
        self,
        mock_extract: MagicMock,
        mock_ocr: MagicMock,
        pdf_path: Path,
    ) -> None:
        mock_extract.return_value = [(1, ""), (2, "   "), (3, "\n\t")]
        mock_ocr.return_value = ""

        result = parse_pdf(pdf_path)

        assert len(result) == 3
        # All pages should have attempted OCR since stripped text is empty
        assert mock_ocr.call_count == 3


# Tests: ParsedPage model

class TestParsedPageModel:

    def test_parsed_page_is_frozen(self) -> None:
        page = ParsedPage(page_number=1, text="hello", is_ocr=False)
        with pytest.raises(ValidationError):
            page.text = "modified"

    def test_parsed_page_defaults(self) -> None:
        page = ParsedPage(page_number=5, text="content")
        assert page.is_ocr is False
        assert page.page_number == 5

    @pytest.mark.parametrize(
        "page_num,text,is_ocr",
        [
            (1, "digital text", False),
            (2, "ocr text", True),
            (100, "", False),
        ],
    )
    def test_parsed_page_construction(
        self, page_num: int, text: str, is_ocr: bool
    ) -> None:
        page = ParsedPage(page_number=page_num, text=text, is_ocr=is_ocr)
        assert page.page_number == page_num
        assert page.text == text
        assert page.is_ocr is is_ocr
