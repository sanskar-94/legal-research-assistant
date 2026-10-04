"""Download Indian statutes and save as text / JSON files.

Targets:
  - Constitution of India
  - Indian Penal Code (IPC) — all 511 sections
  - Bharatiya Nyaya Sanhita (BNS, 2023)
  - Code of Criminal Procedure (CrPC)

Primary source: India Code portal (https://www.indiacode.nic.in/).
Fallback sources: legislative.gov.in, other public mirrors.

Rate-limited to 1 request per second.

Usage:
    python scripts/download_statutes.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import RAW_DIR  # noqa: E402

import requests  # noqa: E402
from bs4 import BeautifulSoup  # noqa: E402
from tenacity import (  # noqa: E402
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("download_statutes")

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
STATUTES_DIR = RAW_DIR / "statutes"

# ---------------------------------------------------------------------------
# Rate-limiter
# ---------------------------------------------------------------------------
_last_request_time: float = 0.0
REQUEST_DELAY = 1.0  # seconds


def _rate_limited_get(url: str, **kwargs) -> requests.Response:
    """GET with at least REQUEST_DELAY seconds between calls."""
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    if elapsed < REQUEST_DELAY:
        time.sleep(REQUEST_DELAY - elapsed)
    response = _fetch(url, **kwargs)
    _last_request_time = time.monotonic()
    return response


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    retry=retry_if_exception_type(
        (requests.ConnectionError, requests.Timeout, requests.HTTPError)
    ),
    reraise=True,
)
def _fetch(url: str, **kwargs) -> requests.Response:
    """Fetch URL with retries on transient network errors."""
    headers = kwargs.pop("headers", {})
    headers.setdefault(
        "User-Agent",
        (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    )
    resp = requests.get(url, headers=headers, timeout=30, **kwargs)
    resp.raise_for_status()
    return resp


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StatuteSource:
    """Immutable descriptor for a statute to download."""

    name: str
    slug: str
    urls: tuple[str, ...]  # primary URL first, then fallbacks
    act_id: str = ""  # India Code act identifier (if applicable)
    sections: tuple[str, ...] = ()  # specific section pages to crawl


# ---------------------------------------------------------------------------
# Statute registry — India Code portal IDs + fallback URLs
# ---------------------------------------------------------------------------
STATUTES: list[StatuteSource] = [
    StatuteSource(
        name="Constitution of India",
        slug="constitution_of_india",
        urls=(
            "https://www.indiacode.nic.in/handle/123456789/2013",
            "https://legislative.gov.in/constitution-of-india/",
            "https://cdnbbsr.s3waas.gov.in/s380537a945c7aaa788f04571e18c5d6ce/uploads/2024/01/2024012290.pdf",
        ),
    ),
    StatuteSource(
        name="Indian Penal Code, 1860",
        slug="ipc_1860",
        urls=(
            "https://www.indiacode.nic.in/handle/123456789/2263",
            "https://legislative.gov.in/actsofparliamentfromtheyear/indian-penal-code-1860/",
            "https://www.iitk.ac.in/wc/data/IPC_186045.pdf",
        ),
        act_id="186045",
    ),
    StatuteSource(
        name="Bharatiya Nyaya Sanhita, 2023",
        slug="bns_2023",
        urls=(
            "https://www.indiacode.nic.in/handle/123456789/20062",
            "https://legislative.gov.in/actsofparliamentfromtheyear/bharatiya-nyaya-sanhita-2023/",
            "https://egazette.gov.in/WriteReadData/2023/248045.pdf",
        ),
    ),
    StatuteSource(
        name="Code of Criminal Procedure, 1973",
        slug="crpc_1973",
        urls=(
            "https://www.indiacode.nic.in/handle/123456789/1611",
            "https://legislative.gov.in/actsofparliamentfromtheyear/code-of-criminal-procedure-1973-2/",
            "https://www.iitk.ac.in/wc/data/CrPC_1974.pdf",
        ),
        act_id="197402",
    ),
]

# ---------------------------------------------------------------------------
# India Code section-list URL builder
# ---------------------------------------------------------------------------
INDIACODE_SECTION_LIST = (
    "https://www.indiacode.nic.in/show-data"
    "?actid={act_id}&sectionId=&sectionno=&orderno="
)

INDIACODE_SECTION_DETAIL = (
    "https://www.indiacode.nic.in/show-data"
    "?actid={act_id}&sectionId={section_id}&sectionno={section_no}&orderno={order_no}"
)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------
def _parse_indiacode_sections_page(html: str) -> list[dict]:
    """Extract section entries from India Code section-list page.

    Returns a list of dicts with keys: section_id, section_no, order_no, title.
    """
    soup = BeautifulSoup(html, "lxml")
    sections: list[dict] = []

    rows = soup.select("table tbody tr")
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 2:
            continue

        link = cells[0].find("a")
        if link is None:
            continue

        href = link.get("href", "")
        section_no = link.get_text(strip=True)
        title = cells[1].get_text(strip=True) if len(cells) > 1 else ""

        # Try to extract parameters from href
        section_entry = {
            "section_no": section_no,
            "title": title,
            "href": href,
        }
        sections.append(section_entry)

    return sections


def _parse_indiacode_section_text(html: str) -> str:
    """Extract the section body text from an India Code section-detail page."""
    soup = BeautifulSoup(html, "lxml")

    # Primary: the #akoma-ntoso or .akn-section containers
    content = soup.select_one("#akoma-ntoso, .akn-section, .section-content, #sectionContent")
    if content:
        return content.get_text(separator="\n", strip=True)

    # Fallback: main content area
    main = soup.select_one("main, .main-content, #content, .container")
    if main:
        return main.get_text(separator="\n", strip=True)

    # Last resort
    body = soup.find("body")
    return body.get_text(separator="\n", strip=True) if body else ""


def _parse_generic_page(html: str) -> str:
    """Extract readable text from a generic statute page."""
    soup = BeautifulSoup(html, "lxml")

    # Remove script / style tags
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # Try common content containers
    for selector in (
        "article",
        "main",
        ".content",
        "#content",
        ".main-content",
        ".container",
    ):
        elem = soup.select_one(selector)
        if elem and len(elem.get_text(strip=True)) > 200:
            return elem.get_text(separator="\n", strip=True)

    body = soup.find("body")
    return body.get_text(separator="\n", strip=True) if body else ""


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
def _download_indiacode_act(statute: StatuteSource) -> dict:
    """Attempt to download a full act from India Code by section crawling.

    Returns a dict: { 'name', 'slug', 'sections': [ { 'section_no', 'title', 'text' } ] }
    """
    result: dict = {
        "name": statute.name,
        "slug": statute.slug,
        "source": "indiacode",
        "sections": [],
    }

    if not statute.act_id:
        logger.info("  No act_id for '%s' — skipping India Code section crawl.", statute.name)
        return result

    list_url = INDIACODE_SECTION_LIST.format(act_id=statute.act_id)
    logger.info("  Fetching section list from India Code: %s", list_url)

    try:
        resp = _rate_limited_get(list_url)
    except Exception:
        logger.warning("  Could not fetch section list for '%s'.", statute.name)
        return result

    section_entries = _parse_indiacode_sections_page(resp.text)
    logger.info("  Found %d section entries for '%s'.", len(section_entries), statute.name)

    for entry in section_entries:
        href = entry.get("href", "")
        if not href:
            continue

        section_url = href if href.startswith("http") else f"https://www.indiacode.nic.in{href}"

        try:
            section_resp = _rate_limited_get(section_url)
            text = _parse_indiacode_section_text(section_resp.text)
        except Exception:
            logger.warning(
                "  Failed to fetch section %s of '%s'.",
                entry.get("section_no", "?"),
                statute.name,
            )
            text = ""

        section_record = {
            "section_no": entry["section_no"],
            "title": entry.get("title", ""),
            "text": text,
        }
        result["sections"].append(section_record)

    return result


def _download_generic_page(statute: StatuteSource) -> str:
    """Try each URL in order and return the first successful page text."""
    for url in statute.urls:
        # Skip PDF URLs — we only handle HTML here
        if url.lower().endswith(".pdf"):
            logger.info("  Skipping PDF URL (not handled in text scraper): %s", url)
            continue

        try:
            resp = _rate_limited_get(url)
            text = _parse_generic_page(resp.text)
            if len(text) > 200:
                logger.info("  Got %d chars from %s", len(text), url)
                return text
            logger.info("  Page too short (%d chars) from %s — trying next URL.", len(text), url)
        except Exception:
            logger.warning("  Failed to fetch %s — trying next URL.", url)

    return ""


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------
def _save_json(data: dict | list, path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    logger.info("  Saved → %s", path)


def _save_text(text: str, path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write(text)
    logger.info("  Saved → %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    STATUTES_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", STATUTES_DIR)

    success_count = 0

    for statute in STATUTES:
        logger.info("Processing '%s' …", statute.name)

        # Strategy 1: India Code section-by-section crawl (IPC, CrPC)
        if statute.act_id:
            act_data = _download_indiacode_act(statute)
            if act_data["sections"]:
                _save_json(act_data, STATUTES_DIR / f"{statute.slug}.json")
                success_count += 1
                continue
            logger.info(
                "  Section crawl yielded nothing — falling back to generic page download."
            )

        # Strategy 2: Generic page text extraction
        text = _download_generic_page(statute)
        if text:
            _save_text(text, STATUTES_DIR / f"{statute.slug}.txt")

            # Also save a structured JSON wrapper for consistency
            structured = {
                "name": statute.name,
                "slug": statute.slug,
                "source": "generic_web",
                "text": text,
            }
            _save_json(structured, STATUTES_DIR / f"{statute.slug}.json")
            success_count += 1
        else:
            logger.error(
                "  Could not download '%s' from any source.", statute.name
            )

    logger.info("Done. Statutes saved: %d / %d.", success_count, len(STATUTES))


if __name__ == "__main__":
    main()
