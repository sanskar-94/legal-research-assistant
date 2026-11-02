"""Scrape ~200 landmark Supreme Court judgements from IndianKanoon.

Downloads full judgement HTML pages and saves them under data/raw/judgements/.
Rate-limited to 1 request per second.

Usage:
    python scripts/data/download_indiankanoon.py
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote_plus

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import INDIANKANOON_DELAY, MAX_JUDGEMENTS, RAW_DIR  # noqa: E402

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
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("download_indiankanoon")

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
JUDGEMENTS_DIR = RAW_DIR / "judgements"

# ---------------------------------------------------------------------------
# IndianKanoon constants
# ---------------------------------------------------------------------------
BASE_URL = "https://indiankanoon.org"
SEARCH_URL = f"{BASE_URL}/search/"
DOC_URL_PREFIX = f"{BASE_URL}/doc/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# ---------------------------------------------------------------------------
# Landmark case URLs (direct doc IDs on IndianKanoon)
#
# These cover major constitutional, criminal, civil, and administrative law
# landmarks from the Supreme Court of India.
# ---------------------------------------------------------------------------
LANDMARK_DOC_IDS: list[dict] = [
    # Constitutional / Fundamental Rights
    {"id": "1363234", "name": "Kesavananda Bharati v State of Kerala"},
    {"id": "257876", "name": "Maneka Gandhi v Union of India"},
    {"id": "1199182", "name": "Golaknath v State of Punjab"},
    {"id": "1735815", "name": "Minerva Mills v Union of India"},
    {"id": "513797", "name": "S R Bommai v Union of India"},
    {"id": "68051886", "name": "K S Puttaswamy v Union of India (Right to Privacy)"},
    {"id": "127517806", "name": "Navtej Singh Johar v Union of India (Section 377)"},
    {"id": "91938676", "name": "Shayara Bano v Union of India (Triple Talaq)"},
    {"id": "1916344", "name": "Vishaka v State of Rajasthan"},
    {"id": "78536", "name": "Bandhua Mukti Morcha v Union of India"},
    {"id": "1515893", "name": "Indra Sawhney v Union of India (Mandal Commission)"},
    {"id": "68662874", "name": "NALSA v Union of India (Transgender Rights)"},
    {"id": "1644805", "name": "Olga Tellis v Bombay Municipal Corporation"},
    {"id": "1378441", "name": "M C Mehta v Union of India (Oleum Gas Leak)"},
    {"id": "69aborl", "name": "Shreya Singhal v Union of India (Section 66A)"},
    {"id": "501198", "name": "A K Gopalan v State of Madras"},
    {"id": "1894410", "name": "S P Gupta v Union of India"},

    # Criminal Law
    {"id": "1250004", "name": "Bachan Singh v State of Punjab (Death Penalty)"},
    {"id": "1233776", "name": "Machhi Singh v State of Punjab (Rarest of Rare)"},
    {"id": "1631284", "name": "Hussainara Khatoon v Home Secretary Bihar"},
    {"id": "1789980", "name": "D K Basu v State of West Bengal (Custodial Rights)"},
    {"id": "1122681", "name": "K M Nanavati v State of Maharashtra"},
    {"id": "1419591", "name": "Kehar Singh v Union of India"},
    {"id": "36aborl", "name": "Lalita Kumari v Govt of UP (FIR Registration)"},
    {"id": "1072aborl", "name": "Arnesh Kumar v State of Bihar (Arrest Guidelines)"},
    {"id": "116699920", "name": "Joseph Shine v Union of India (Adultery)"},
    {"id": "809587", "name": "Rattan Lal v State of Punjab"},
    {"id": "544079", "name": "State of Maharashtra v Mayer Hans George"},
    {"id": "1569253", "name": "Nandini Satpathy v PL Dani (Right Against Self-Incrimination)"},

    # Civil / Property / Contract
    {"id": "1380625", "name": "Mohori Bibee v Dharmodas Ghose"},
    {"id": "1353689", "name": "Satyabrata Ghose v Mugneeram"},
    {"id": "595025", "name": "Keshavlal v Union of India"},
    {"id": "1137014", "name": "V Tulasamma v V Sesha Reddi"},

    # Administrative Law
    {"id": "1310996", "name": "Tata Cellular v Union of India"},
    {"id": "753224", "name": "Balco Employees Union v Union of India"},
    {"id": "180425", "name": "E P Royappa v State of Tamil Nadu"},
    {"id": "1517037", "name": "Ajay Hasia v Khalid Mujib"},

    # Company / Commercial
    {"id": "1155028", "name": "Salomon v A Salomon and Co (Indian Reference)"},
    {"id": "513804", "name": "LIC v Escorts Ltd"},
    {"id": "1150498", "name": "Tata Consultancy Services v State of AP"},

    # Environmental Law
    {"id": "547093", "name": "Vellore Citizens Welfare Forum v Union of India"},
    {"id": "1199419", "name": "Indian Council for Enviro Legal Action v Union of India"},
    {"id": "1514672", "name": "M C Mehta v Kamal Nath"},
    {"id": "921735", "name": "Rural Litigation and Entitlement Kendra v State of UP"},

    # Labour Law
    {"id": "1328852", "name": "Bangalore Water Supply v A Rajappa"},
    {"id": "1375843", "name": "ONGC v Petroleum Workers Union"},

    # Family Law
    {"id": "68661071", "name": "Shilpa Sailesh v Varun Sreenivasan (Mutual Divorce)"},
    {"id": "544023", "name": "Shah Bano Begum v Mohd Ahmed Khan"},
    {"id": "1118403", "name": "Sarla Mudgal v Union of India (Bigamy)"},
    {"id": "127768357", "name": "Indian Young Lawyers Assn v State of Kerala (Sabarimala)"},

    # Media / Free Speech
    {"id": "1042214", "name": "Bennett Coleman v Union of India"},
    {"id": "1937481", "name": "Indian Express Newspapers v Union of India"},
    {"id": "1306932", "name": "Romesh Thappar v State of Madras"},
    {"id": "1085799", "name": "Brij Bhushan v State of Delhi"},

    # Election Law
    {"id": "1376116", "name": "Indira Nehru Gandhi v Raj Narain"},
    {"id": "1538336", "name": "Kuldip Nayar v Union of India"},

    # Tax Law
    {"id": "761aborl", "name": "Vodafone International v Union of India"},
]

# ---------------------------------------------------------------------------
# Search queries to find additional landmark cases
# ---------------------------------------------------------------------------
SEARCH_QUERIES: list[str] = [
    "landmark supreme court constitution",
    "supreme court fundamental rights Article 21",
    "supreme court criminal law landmark",
    "supreme court IPC 302 murder",
    "supreme court IPC 376 rape",
    "supreme court IPC 420 cheating",
    "supreme court bail conditions",
    "supreme court dowry death 304B",
    "supreme court NDPS Act",
    "supreme court POCSO Act",
    "supreme court arbitration",
    "supreme court environmental PIL",
    "supreme court writ jurisdiction",
    "supreme court land acquisition",
    "supreme court consumer protection",
    "supreme court cyber crime IT Act",
    "supreme court contempt of court",
    "supreme court judicial review",
    "supreme court reservation quota",
    "supreme court preventive detention",
]

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
_last_request_time: float = 0.0


def _rate_limited_get(url: str) -> requests.Response:
    """GET with at least INDIANKANOON_DELAY seconds between requests."""
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    if elapsed < INDIANKANOON_DELAY:
        time.sleep(INDIANKANOON_DELAY - elapsed)
    response = _fetch(url)
    _last_request_time = time.monotonic()
    return response


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=3, max=20),
    retry=retry_if_exception_type(
        (requests.ConnectionError, requests.Timeout, requests.HTTPError)
    ),
    reraise=True,
)
def _fetch(url: str) -> requests.Response:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------
def _extract_doc_ids_from_search(html: str) -> list[str]:
    """Extract document IDs from an IndianKanoon search results page."""
    soup = BeautifulSoup(html, "lxml")
    doc_ids: list[str] = []

    for link in soup.find_all("a", href=True):
        href = link["href"]
        match = re.search(r"/doc/(\d+)", href)
        if match:
            doc_id = match.group(1)
            if doc_id not in doc_ids:
                doc_ids.append(doc_id)

    return doc_ids


def _extract_judgement_text(html: str) -> str:
    """Extract the judgement body from an IndianKanoon doc page."""
    soup = BeautifulSoup(html, "lxml")

    # Remove unwanted elements
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    # Primary: the judgement div
    judgement_div = soup.select_one("#judgments, .judgments, .judgment_content")
    if judgement_div:
        return judgement_div.get_text(separator="\n", strip=True)

    # Fallback: content area
    content = soup.select_one(".result, .doc_content, #content, main")
    if content:
        return content.get_text(separator="\n", strip=True)

    body = soup.find("body")
    return body.get_text(separator="\n", strip=True) if body else ""


def _extract_title_from_page(html: str) -> str:
    """Pull the <title> or first heading as case title."""
    soup = BeautifulSoup(html, "lxml")
    title_tag = soup.find("title")
    if title_tag:
        return title_tag.get_text(strip=True)
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)
    return "Unknown"


# ---------------------------------------------------------------------------
# Sanitise filename
# ---------------------------------------------------------------------------
def _safe_filename(name: str, max_len: int = 80) -> str:
    """Convert a case name to a filesystem-safe string."""
    safe = re.sub(r"[^a-zA-Z0-9_\- ]", "", name)
    safe = re.sub(r"\s+", "_", safe.strip())
    return safe[:max_len] if safe else "unknown"


# ---------------------------------------------------------------------------
# Download logic
# ---------------------------------------------------------------------------
def _download_doc(doc_id: str, name: str, output_dir: Path) -> bool:
    """Download a single judgement and save as HTML + metadata JSON.

    Returns True on success, False otherwise.
    """
    safe_name = _safe_filename(name)
    html_path = output_dir / f"{doc_id}_{safe_name}.html"
    meta_path = output_dir / f"{doc_id}_{safe_name}.json"

    # Skip if already downloaded
    if html_path.exists():
        logger.info("  Already exists - skipping doc %s (%s)", doc_id, name)
        return True

    url = f"{DOC_URL_PREFIX}{doc_id}/"
    try:
        resp = _rate_limited_get(url)
    except Exception:
        logger.warning("  Failed to download doc %s (%s).", doc_id, name)
        return False

    html_content = resp.text
    title = _extract_title_from_page(html_content)
    judgement_text = _extract_judgement_text(html_content)

    # Save raw HTML
    html_path.write_text(html_content, encoding="utf-8")

    # Save metadata JSON
    metadata = {
        "doc_id": doc_id,
        "name": name,
        "title": title,
        "url": url,
        "text_length": len(judgement_text),
        "source": "indiankanoon",
    }
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)

    logger.info(
        "  Saved doc %s - %s (%d chars)", doc_id, name[:50], len(judgement_text)
    )
    return True


def _collect_doc_ids_from_searches(
    existing_ids: set[str],
    target_count: int,
) -> list[dict]:
    """Run search queries to discover additional document IDs."""
    additional: list[dict] = []
    seen = set(existing_ids)

    for query in SEARCH_QUERIES:
        if len(seen) >= target_count:
            break

        search_url = f"{SEARCH_URL}?formInput={quote_plus(query)}"
        logger.info("  Searching: %s", query)

        try:
            resp = _rate_limited_get(search_url)
        except Exception:
            logger.warning("  Search failed for query: %s", query)
            continue

        doc_ids = _extract_doc_ids_from_search(resp.text)
        for doc_id in doc_ids:
            if doc_id not in seen and len(seen) < target_count:
                seen.add(doc_id)
                additional.append({"id": doc_id, "name": f"Search result: {query}"})

    return additional


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    JUDGEMENTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", JUDGEMENTS_DIR)
    logger.info("Target: up to %d judgements.", MAX_JUDGEMENTS)

    # Phase 1: Download known landmark cases
    downloaded = 0
    failed = 0
    known_ids: set[str] = set()

    # Filter out entries with non-numeric IDs (placeholders)
    valid_landmarks = [
        entry for entry in LANDMARK_DOC_IDS if entry["id"].isdigit()
    ]

    logger.info(
        "Phase 1: Downloading %d known landmark cases ...", len(valid_landmarks)
    )

    for entry in valid_landmarks:
        if downloaded >= MAX_JUDGEMENTS:
            break

        doc_id = entry["id"]
        name = entry["name"]
        known_ids.add(doc_id)

        success = _download_doc(doc_id, name, JUDGEMENTS_DIR)
        if success:
            downloaded += 1
        else:
            failed += 1

    logger.info(
        "Phase 1 complete. Downloaded: %d, Failed: %d.", downloaded, failed
    )

    # Phase 2: Search-based discovery to reach target count
    if downloaded < MAX_JUDGEMENTS:
        remaining = MAX_JUDGEMENTS - downloaded
        logger.info(
            "Phase 2: Searching for up to %d additional judgements ...", remaining
        )
        additional = _collect_doc_ids_from_searches(known_ids, MAX_JUDGEMENTS)

        for entry in additional:
            if downloaded >= MAX_JUDGEMENTS:
                break
            success = _download_doc(entry["id"], entry["name"], JUDGEMENTS_DIR)
            if success:
                downloaded += 1
            else:
                failed += 1

    # Save a manifest of everything downloaded
    manifest_path = JUDGEMENTS_DIR / "_manifest.json"
    manifest_entries: list[dict] = []
    for json_file in sorted(JUDGEMENTS_DIR.glob("*.json")):
        if json_file.name.startswith("_"):
            continue
        try:
            with json_file.open("r", encoding="utf-8") as fh:
                manifest_entries.append(json.load(fh))
        except Exception:
            pass

    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {"total": len(manifest_entries), "judgements": manifest_entries},
            fh,
            ensure_ascii=False,
            indent=2,
        )

    logger.info(
        "Done. Total downloaded: %d, Failed: %d. Manifest: %s",
        downloaded,
        failed,
        manifest_path,
    )


if __name__ == "__main__":
    main()
