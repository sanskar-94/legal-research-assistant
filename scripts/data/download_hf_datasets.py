"""Download the Indian-law datasets from Hugging Face into data/raw/.

Each split of each dataset in ``DATASETS`` is saved as
``data/raw/<slug>_<split>.json``.

Usage:
    python scripts/data/download_hf_datasets.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path bootstrap – allow ``python scripts/data/download_hf_datasets.py``
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import RAW_DIR  # noqa: E402

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
logger = logging.getLogger("download_hf_datasets")

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
# Each entry: (HuggingFace dataset ID, optional config name, friendly slug)
DATASETS: list[tuple[str, str | None, str]] = [
    # 24K QA pairs on IPC, CrPC, Constitution, legal terminology
    ("viber1/indian-law-dataset", None, "indian_law_dataset"),
    # 7K court judgments + summaries
    ("ninadn/indian-legal", None, "indian_legal"),
    # Pre-chunked SC judgments (great for RAG)
    ("vihaannnn/Indian-Supreme-Court-Judgements-Chunked", None, "sc_judgements_chunked"),
    # 1470 rows of chunked legal texts for RAG
    ("ShreyasP123/Legal-Dataset-for-india", None, "legal_dataset_india"),
    # Structured Constitution articles
    ("nisaar/Constitution_of_India", None, "constitution_of_india"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ensure_dirs() -> None:
    """Create output directories if they do not already exist."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory ready: %s", RAW_DIR)


def _rows_to_serialisable(dataset_split) -> list[dict]:
    """Convert a HuggingFace Dataset split into a plain list of dicts.

    Returns a *new* list - the original dataset is never mutated.
    """
    rows: list[dict] = []
    for row in dataset_split:
        serialisable = {}
        for key, value in row.items():
            # Handle non-serialisable types gracefully
            try:
                json.dumps(value)
                serialisable[key] = value
            except (TypeError, ValueError):
                serialisable[key] = str(value)
        rows.append(serialisable)
    return rows


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=30),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    reraise=True,
)
def _download_dataset(
    dataset_id: str,
    config_name: str | None,
) -> Any:
    """Download a dataset with automatic retries on network errors."""
    from datasets import load_dataset  # imported here to defer heavy load

    logger.info(
        "Downloading dataset '%s' (config=%s) ...",
        dataset_id,
        config_name or "default",
    )
    kwargs: dict = {}
    if config_name is not None:
        kwargs["name"] = config_name

    dataset = load_dataset(dataset_id, **kwargs)
    return dataset


def _save_split(rows: list[dict], output_path: Path) -> None:
    """Write rows to a JSON file."""
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)
    logger.info("  Saved %d records -> %s", len(rows), output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    _ensure_dirs()

    success_count = 0
    failure_count = 0

    for dataset_id, config_name, slug in DATASETS:
        try:
            dataset_dict = _download_dataset(dataset_id, config_name)
        except Exception:
            logger.exception(
                "Failed to download '%s' after retries - skipping.", dataset_id
            )
            failure_count += 1
            continue

        for split_name, split_data in dataset_dict.items():
            rows = _rows_to_serialisable(split_data)
            filename = f"{slug}_{split_name}.json"
            _save_split(rows, RAW_DIR / filename)

        logger.info("Finished dataset '%s'.", dataset_id)
        success_count += 1

    logger.info(
        "Done. Datasets downloaded: %d, failed: %d.",
        success_count,
        failure_count,
    )

    if failure_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
