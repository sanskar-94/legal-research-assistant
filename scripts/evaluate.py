`"""Run the evaluation set through the full pipeline and report metrics.

Loads ``data/evaluation/test_queries.json``, sends each query through
``LegalResearchPipeline`` (IPC/BNS lookup, retrieval, generation, citation
parsing, confidence), prints a summary table and writes per-query results
to ``data/evaluation/eval_results_latest.json``.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --limit 10

Requires NVIDIA_API_KEY (or LLM_API_KEY) in .env and a populated vector store.

Metrics
-------
citation_validity     share of [SOURCE_N] tags that point to a source that
                      was actually given to the model ("citation accuracy"
                      in the project report)
cited_claim_coverage  share of legal claim sentences that carry a citation
citation_support      share of cited claims whose source shares the claim's
                      section numbers, legal terms and enough vocabulary
hallucination_rate    share of answers that mention a case name or an
                      AIR/SCC/SCR citation found in none of the sources
response_rate         share of answers that cite at least one valid source
                      and are not a refusal
retrieval_precision   share of retrieved chunks of an expected document type
                      or containing an expected keyword
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import EVAL_DIR  # noqa: E402
from src.evaluation.citation_metrics import (  # noqa: E402
    compute_citation_quality,
    detect_hallucinations,
    is_substantive_response,
)
from src.pipeline import LegalResearchPipeline  # noqa: E402
from src.retrieval.ipc_bns import MAPPING_CHUNK_ID  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("evaluate")


def _compute_retrieval_precision(
    retrieved_chunks: list[dict[str, Any]],
    expected_doc_types: list[str],
    expected_keywords: list[str],
) -> float:
    """Share of retrieved chunks matching an expected doc type or keyword."""
    if not retrieved_chunks:
        return 0.0

    relevant = 0
    for chunk in retrieved_chunks:
        doc_type = chunk.get("metadata", {}).get("doc_type", "")
        text = chunk.get("text", "").lower()
        type_match = doc_type in expected_doc_types
        keyword_match = any(kw.lower() in text for kw in expected_keywords)
        if type_match or keyword_match:
            relevant += 1

    return relevant / len(retrieved_chunks)


def evaluate(test_queries_path: Path | None = None, limit: int | None = None) -> dict[str, Any]:
    """Run every test query and return ``{"summary": ..., "details": ...}``."""
    if test_queries_path is None:
        test_queries_path = EVAL_DIR / "test_queries.json"

    with test_queries_path.open("r", encoding="utf-8") as fh:
        test_queries: list[dict[str, Any]] = json.load(fh)
    if limit:
        test_queries = test_queries[:limit]

    logger.info("Loaded %d test queries from %s", len(test_queries), test_queries_path)

    try:
        pipeline = LegalResearchPipeline.from_settings()
    except Exception as exc:
        logger.error("Failed to initialise pipeline: %s", exc)
        return {"summary": {}, "details": [], "error": str(exc)}
    if not pipeline.can_generate:
        return {
            "summary": {},
            "details": [],
            "error": "No LLM API key configured. Set NVIDIA_API_KEY in .env.",
        }

    details: list[dict[str, Any]] = []

    for i, tq in enumerate(test_queries):
        query = tq["query"]
        category = tq.get("category", "unknown")
        logger.info("[%d/%d] (%s) %s", i + 1, len(test_queries), category, query[:80])

        start_time = time.monotonic()
        result = pipeline.query(query)
        elapsed = time.monotonic() - start_time

        answer = result.answer_text or ""
        sources = result.sources
        retrieved = [c for c in sources if c.get("chunk_id") != MAPPING_CHUNK_ID]

        details.append({
            "query": query,
            "category": category,
            **compute_citation_quality(answer, sources),
            "hallucinations": detect_hallucinations(answer, sources),
            "is_substantive": is_substantive_response(answer, len(sources)),
            "confidence": result.confidence.overall_score if result.confidence else 0.0,
            "retrieval_precision": round(
                _compute_retrieval_precision(
                    retrieved,
                    tq.get("expected_doc_types", []),
                    tq.get("expected_keywords", []),
                ),
                4,
            ),
            "ipc_bns_matches": [m["ipc_section"] for m in result.ipc_bns_mappings],
            "num_chunks_retrieved": len(retrieved),
            "response_length": len(answer),
            "elapsed_seconds": round(elapsed, 2),
            "error": result.error,
        })

    n = len(details) or 1

    def _mean(key: str) -> float:
        return round(sum(d[key] for d in details) / n, 4)

    summary: dict[str, Any] = {
        "total_queries": len(details),
        "citation_validity_mean": _mean("citation_validity"),
        "cited_claim_coverage_mean": _mean("cited_claim_coverage"),
        "citation_support_score_mean": _mean("citation_support_score"),
        "citation_quality_score_mean": _mean("citation_quality_score"),
        "hallucination_rate": round(sum(1 for d in details if d["hallucinations"]) / n, 4),
        "response_rate": round(sum(1 for d in details if d["is_substantive"]) / n, 4),
        "average_confidence": _mean("confidence"),
        "retrieval_precision_mean": _mean("retrieval_precision"),
        "errors": sum(1 for d in details if d["error"]),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return {"summary": summary, "details": details}




def _print_summary_table(summary: dict[str, Any]) -> None:
    """Print aggregate metrics as a formatted table."""
    border = "+" + "-" * 38 + "+" + "-" * 14 + "+"
    header = f"| {'Metric':<36} | {'Value':>12} |"

    print("\n" + "=" * 56)
    print("  EVALUATION RESULTS")
    print("=" * 56)
    print(border)
    print(header)
    print(border)

    rows: list[tuple[str, str]] = [
        ("Total queries", str(summary.get("total_queries", 0))),
        (
            "Citation validity (mean)",
            f"{summary.get('citation_validity_mean', 0):.2%}",
        ),
        (
            "Cited claim coverage (mean)",
            f"{summary.get('cited_claim_coverage_mean', 0):.2%}",
        ),
        (
            "Citation support score (mean)",
            f"{summary.get('citation_support_score_mean', 0):.2%}",
        ),
        (
            "Citation quality score (mean)",
            f"{summary.get('citation_quality_score_mean', 0):.2%}",
        ),
        (
            "Hallucination rate",
            f"{summary.get('hallucination_rate', 0):.2%}",
        ),
        (
            "Response rate (substantive)",
            f"{summary.get('response_rate', 0):.2%}",
        ),
        (
            "Average confidence score",
            f"{summary.get('average_confidence', 0):.4f}",
        ),
        (
            "Retrieval precision (mean)",
            f"{summary.get('retrieval_precision_mean', 0):.2%}",
        ),
    ]

    for label, value in rows:
        print(f"| {label:<36} | {value:>12} |")

    print(border)
    print(f"  Evaluated at: {summary.get('timestamp', 'N/A')}")
    print()


def _print_category_breakdown(details: list[dict[str, Any]]) -> None:
    """Print per-category breakdown."""
    categories: dict[str, list[dict[str, Any]]] = {}
    for d in details:
        cat = d.get("category", "unknown")
        categories.setdefault(cat, []).append(d)

    border = (
        "+"
        + "-" * 22
        + "+"
        + "-" * 8
        + "+"
        + "-" * 12
        + "+"
        + "-" * 12
        + "+"
        + "-" * 12
        + "+"
        + "-" * 12
        + "+"
        + "-" * 12
        + "+"
    )
    header = f"| {'Category':<20} | {'Count':>6} | {'Validity':>10} | {'Support':>10} | {'Quality':>10} | {'Resp Rate':>10} | {'Ret Prec':>10} |"

    print("  PER-CATEGORY BREAKDOWN")
    print(border)
    print(header)
    print(border)

    for cat, items in sorted(categories.items()):
        n = len(items)
        avg_validity = sum(d["citation_validity"] for d in items) / n
        avg_support = sum(d["citation_support_score"] for d in items) / n
        avg_quality = sum(d["citation_quality_score"] for d in items) / n
        resp_rate = sum(1 for d in items if d["is_substantive"]) / n
        avg_ret = sum(d["retrieval_precision"] for d in items) / n
        print(
            f"| {cat:<20} | {n:>6} | {avg_validity:>9.2%} | {avg_support:>9.2%} | {avg_quality:>9.2%} | {resp_rate:>9.2%} | {avg_ret:>9.2%} |"
        )

    print(border)
    print()


# Main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--limit", type=int, default=None, help="only run the first N queries")
    parser.add_argument(
        "--output",
        type=Path,
        default=EVAL_DIR / "eval_results_latest.json",
        help="where to write the JSON results (eval_results.json holds the "
        "run recorded in the project report, so it is not the default)",
    )
    args = parser.parse_args()

    logger.info("Starting evaluation...")
    results = evaluate(limit=args.limit)

    if results.get("error"):
        logger.error("Evaluation aborted: %s", results["error"])
        sys.exit(1)

    summary = results["summary"]
    details = results["details"]

    # Print to terminal
    _print_summary_table(summary)
    _print_category_breakdown(details)

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {"summary": summary, "details": details}

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2, default=str)

    logger.info("Detailed results saved to %s", output_path)


if __name__ == "__main__":
    main()
