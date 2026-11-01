"""Parse every document in data/raw/ and save it to data/processed/.

Walks the raw data directory for PDF, HTML, and text files, runs the
appropriate parser, and writes structured JSON output. Statutes crawled
section by section by ``download_statutes.py`` are converted as well.

Usage:
    python scripts/parse_pdfs.py
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any, Callable

# Ensure project root is on sys.path so imports work when running as a script
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import PROCESSED_DIR, RAW_DIR  # noqa: E402
from src.ingestion.metadata_extractor import extract_metadata  # noqa: E402
from src.ingestion.pdf_parser import parse_pdf  # noqa: E402
from src.ingestion.text_cleaner import clean_legal_text  # noqa: E402

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("parse_pdfs")


def _json_default(obj: object) -> object:
    """Serialise ``date`` values (from the metadata extractor) as ISO strings."""
    if isinstance(obj, date):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _read_text_file(file_path: Path) -> str:
    """Read a text or HTML file with encoding fallback."""
    for encoding in ("utf-8", "cp1252"):
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    # latin-1 maps every byte, so this never fails
    return file_path.read_text(encoding="latin-1")


def _extract_html_text(html_content: str) -> str:
    """Extract visible text from HTML content."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning("beautifulsoup4 is not installed; stripping tags with a regex.")
        text = re.sub(r"<[^>]+>", " ", html_content)
        return re.sub(r"\s+", " ", text).strip()

    soup = BeautifulSoup(html_content, "lxml")
    for element in soup(["script", "style", "nav", "footer", "header"]):
        element.decompose()
    return soup.get_text(separator="\n", strip=True)


def _write_output(source_path: Path, output_dir: Path, output: dict[str, Any]) -> None:
    """Write *output* as JSON, mirroring the source's path under ``data/raw/``."""
    try:
        relative = source_path.relative_to(RAW_DIR)
    except ValueError:
        relative = Path(source_path.name)
    out_path = output_dir / relative.with_suffix(".json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    logger.info("Saved: %s", out_path)


def _document_output(source_path: Path, source_type: str, raw_text: str) -> dict[str, Any]:
    """Clean *raw_text*, extract its metadata and build the output record."""
    cleaned_text = clean_legal_text(raw_text)
    return {
        "source_file": source_path.name,
        "source_type": source_type,
        "metadata": extract_metadata(cleaned_text).model_dump(),
        "full_text": cleaned_text,
    }


def process_pdf(file_path: Path) -> dict[str, Any] | None:
    pages = parse_pdf(file_path)
    output = _document_output(
        file_path, "pdf", "\n\n".join(page.text for page in pages)
    )
    output["total_pages"] = len(pages)
    output["ocr_pages"] = sum(1 for p in pages if p.is_ocr)
    return output


def process_text(file_path: Path) -> dict[str, Any] | None:
    raw_text = _read_text_file(file_path)
    return _document_output(file_path, "text", raw_text) if raw_text.strip() else None


def process_html(file_path: Path) -> dict[str, Any] | None:
    raw_html = _read_text_file(file_path)
    if not raw_html.strip():
        return None
    return _document_output(file_path, "html", _extract_html_text(raw_html))


_PROCESSORS: dict[str, Callable[[Path], dict[str, Any] | None]] = {
    ".pdf": process_pdf,
    ".txt": process_text,
    ".text": process_text,
    ".html": process_html,
    ".htm": process_html,
}


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
    _write_output(file_path, output_dir, {
        "source_file": file_path.name,
        "source_type": "statute_sections",
        "act_name": data.get("name"),
        "metadata": {},
        "full_text": clean_legal_text(body),
    })
    return True


def run_pipeline() -> None:
    """Walk data/raw/ and process all supported files."""
    if not RAW_DIR.exists():
        logger.error("Raw data directory does not exist: %s", RAW_DIR)
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    supported_files = [
        f for f in sorted(RAW_DIR.rglob("*"))
        if f.is_file() and f.suffix.lower() in _PROCESSORS
    ]
    logger.info(
        "Found %d PDF/HTML/text files to process in %s",
        len(supported_files), RAW_DIR,
    )

    success_count = 0
    failure_count = 0

    for file_path in sorted((RAW_DIR / "statutes").glob("*.json")):
        if process_statute_sections(file_path, PROCESSED_DIR):
            success_count += 1

    for file_path in supported_files:
        logger.info("Processing: %s", file_path)
        try:
            output = _PROCESSORS[file_path.suffix.lower()](file_path)
        except Exception:
            logger.exception("Failed to process: %s", file_path)
            output = None

        if output is None:
            failure_count += 1
            continue
        _write_output(file_path, PROCESSED_DIR, output)
        success_count += 1

    logger.info(
        "Done: %d succeeded, %d failed.",
        success_count, failure_count,
    )


if __name__ == "__main__":
    run_pipeline()
