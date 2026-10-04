"""Pipeline script to parse all documents in data/raw/ and save to data/processed/.

Walks the raw data directory for PDF, HTML, and text files, runs the
appropriate parser, and writes structured JSON output.

Usage:
    python -m scripts.parse_pdfs
    python scripts/parse_pdfs.py
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date
from pathlib import Path

# Ensure project root is on sys.path so imports work when running as a script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import PROCESSED_DIR, RAW_DIR
from src.ingestion.metadata_extractor import extract_metadata
from src.ingestion.pdf_parser import parse_pdf
from src.ingestion.text_cleaner import clean_legal_text

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Supported Extensions
PDF_EXTENSIONS = frozenset({".pdf"})
TEXT_EXTENSIONS = frozenset({".txt", ".text"})
HTML_EXTENSIONS = frozenset({".html", ".htm"})


class _DateEncoder(json.JSONEncoder):
    """JSON encoder that serializes date objects to ISO format strings."""

    def default(self, obj: object) -> object:
        if isinstance(obj, date):
            return obj.isoformat()
        return super().default(obj)


def _read_text_file(file_path: Path) -> str:
    """Read a text or HTML file with encoding fallback."""
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    logger.error("Failed to decode file with any encoding: %s", file_path)
    return ""


def _extract_html_text(html_content: str) -> str:
    """Extract visible text from HTML content."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, "lxml")

        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        text = soup.get_text(separator="\n", strip=True)
        return text
    except ImportError:
        logger.warning(
            "beautifulsoup4/lxml not installed. "
            "Install with: pip install beautifulsoup4 lxml"
        )
        # Crude fallback: strip HTML tags with regex
        import re
        text = re.sub(r"<[^>]+>", " ", html_content)
        return re.sub(r"\s+", " ", text).strip()


def _output_path_for(source_path: Path, output_dir: Path) -> Path:
    """Compute the output JSON path preserving subdirectory structure."""
    try:
        relative = source_path.relative_to(RAW_DIR)
    except ValueError:
        relative = Path(source_path.name)
    return output_dir / relative.with_suffix(".json")


def process_pdf(file_path: Path, output_dir: Path) -> bool:
    """Parse a PDF and save structured output as JSON.

    Returns True on success, False on failure.
    """
    logger.info("Processing PDF: %s", file_path)
    try:
        pages = parse_pdf(file_path)
        full_text = "\n\n".join(page.text for page in pages)

        cleaned_text = clean_legal_text(full_text)
        metadata = extract_metadata(cleaned_text)

        output = {
            "source_file": str(file_path.name),
            "source_type": "pdf",
            "total_pages": len(pages),
            "ocr_pages": sum(1 for p in pages if p.is_ocr),
            "metadata": metadata.model_dump(),
            "pages": [page.model_dump() for page in pages],
            "full_text": cleaned_text,
        }

        out_path = _output_path_for(file_path, output_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False, cls=_DateEncoder),
            encoding="utf-8",
        )
        logger.info("Saved: %s", out_path)
        return True

    except Exception:
        logger.exception("Failed to process PDF: %s", file_path)
        return False


def process_text(file_path: Path, output_dir: Path) -> bool:
    """Parse a text file and save structured output as JSON.

    Returns True on success, False on failure.
    """
    logger.info("Processing text file: %s", file_path)
    try:
        raw_text = _read_text_file(file_path)
        if not raw_text:
            logger.warning("Empty or unreadable file: %s", file_path)
            return False

        cleaned_text = clean_legal_text(raw_text)
        metadata = extract_metadata(cleaned_text)

        output = {
            "source_file": str(file_path.name),
            "source_type": "text",
            "metadata": metadata.model_dump(),
            "full_text": cleaned_text,
        }

        out_path = _output_path_for(file_path, output_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False, cls=_DateEncoder),
            encoding="utf-8",
        )
        logger.info("Saved: %s", out_path)
        return True

    except Exception:
        logger.exception("Failed to process text file: %s", file_path)
        return False


def process_html(file_path: Path, output_dir: Path) -> bool:
    """Parse an HTML file and save structured output as JSON.

    Returns True on success, False on failure.
    """
    logger.info("Processing HTML file: %s", file_path)
    try:
        raw_html = _read_text_file(file_path)
        if not raw_html:
            logger.warning("Empty or unreadable file: %s", file_path)
            return False

        raw_text = _extract_html_text(raw_html)
        cleaned_text = clean_legal_text(raw_text)
        metadata = extract_metadata(cleaned_text)

        output = {
            "source_file": str(file_path.name),
            "source_type": "html",
            "metadata": metadata.model_dump(),
            "full_text": cleaned_text,
        }

        out_path = _output_path_for(file_path, output_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False, cls=_DateEncoder),
            encoding="utf-8",
        )
        logger.info("Saved: %s", out_path)
        return True

    except Exception:
        logger.exception("Failed to process HTML file: %s", file_path)
        return False


def process_statute_sections(file_path: Path, output_dir: Path) -> bool:
    """Convert a section-by-section statute crawl into one text document.

    ``download_statutes.py`` saves India Code crawls as JSON with a
    ``sections`` list. Each section is written as "Section N. Title" followed
    by its text so the statute chunker can split on those headers again.

    Returns True when a document was written, False when the file had no
    sections (generic downloads are handled through their .txt copy).
    """
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("Failed to read statute JSON: %s", file_path)
        return False

    sections = [s for s in data.get("sections", []) if s.get("text")]
    if not sections:
        return False

    body = "\n\n".join(
        f"Section {s['section_no']}. {s.get('title', '')}\n{s['text']}".strip()
        for s in sections
    )
    output = {
        "source_file": file_path.name,
        "source_type": "statute_sections",
        "act_name": data.get("name"),
        "metadata": {},
        "full_text": clean_legal_text(body),
    }
    out_path = _output_path_for(file_path, output_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved %d sections: %s", len(sections), out_path)
    return True


def run_pipeline() -> None:
    """Walk data/raw/ and process all supported files."""
    if not RAW_DIR.exists():
        logger.error("Raw data directory does not exist: %s", RAW_DIR)
        logger.info("Create %s and add PDF/HTML/text files to process.", RAW_DIR)
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all files
    all_files = sorted(RAW_DIR.rglob("*"))
    supported_files = [
        f for f in all_files
        if f.is_file() and f.suffix.lower() in (
            PDF_EXTENSIONS | TEXT_EXTENSIONS | HTML_EXTENSIONS
        )
    ]

    logger.info(
        "Found %d PDF/HTML/text files to process in %s",
        len(supported_files), RAW_DIR,
    )

    success_count = 0
    failure_count = 0

    # Statutes crawled section by section from India Code
    for file_path in sorted((RAW_DIR / "statutes").glob("*.json")):
        if process_statute_sections(file_path, PROCESSED_DIR):
            success_count += 1

    for file_path in supported_files:
        ext = file_path.suffix.lower()

        if ext in PDF_EXTENSIONS:
            ok = process_pdf(file_path, PROCESSED_DIR)
        elif ext in HTML_EXTENSIONS:
            ok = process_html(file_path, PROCESSED_DIR)
        elif ext in TEXT_EXTENSIONS:
            ok = process_text(file_path, PROCESSED_DIR)
        else:
            continue

        if ok:
            success_count += 1
        else:
            failure_count += 1

    logger.info(
        "Pipeline complete: %d succeeded, %d failed out of %d files",
        success_count, failure_count, len(supported_files),
    )


if __name__ == "__main__":
    run_pipeline()
