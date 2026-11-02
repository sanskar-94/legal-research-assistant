"""Run the evaluation set through the full pipeline and report metrics.

Loads ``data/evaluation/test_queries.json``, sends each query through
``LegalResearchPipeline`` (IPC/BNS lookup, retrieval, generation, citation
parsing, confidence), prints a summary table (Table IV of the report) and a
per-category breakdown (Table V) and writes per-query results to
``data/evaluation/eval_results_latest.json``.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --limit 10
    python scripts/evaluate.py --summarise data/evaluation/eval_results_deepseek.json

Requires NVIDIA_API_KEY (or LLM_API_KEY) in .env and a populated vector store.
``--summarise`` needs neither: it re-prints the tables of a saved run.

Metrics (Section IV-B of the report; all deterministic, no LLM judge)
-------
citation_validity     share of [SOURCE_N] tags with 1 <= N <= n, n the number
                      of sources sent; an answer with no tag scores 0
                      ("citation accuracy" in Run A)
cited_claim_coverage  share of legal claim sentences (>= 25 characters that
                      mention a section, Act, court, offence...) carrying a citation
citation_support      share of cited claims whose source has the same section /
                      article numbers and legal concepts and shares >= 35% of
                      the claim's content words or >= 5 of them
hallucination_rate    share of answers that mention a case name ("X v Y") or an
                      AIR/SCC/SCR citation found in none of the sources
response_rate         share of answers that cite at least one valid source
                      and are not a refusal
retrieval_precision   P@K (eq. 8): share of the K retrieved chunks of an
                      expected document type or containing an expected keyword
keyword_precision     P_kw@K: share of the K chunks containing an expected keyword
hit                   Hit@K: 1 if any retrieved chunk contains an expected keyword
reciprocal_rank       1/rank of the first keyword-relevant chunk (0 if none);
                      its mean is the MRR
elapsed_seconds       per-query latency; the summary's mean_latency_seconds is
                      the mean over the non-empty queries (Table IV)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import EVAL_DIR, LLM_BACKEND, LLM_MODEL, RERANKED_TOP_K  # noqa: E402
from src.evaluation.citation_metrics import (  # noqa: E402
    compute_citation_quality,
    detect_hallucinations,
    is_substantive_response,
)
from src.evaluation.retrieval_metrics import retrieval_metrics  # noqa: E402
from src.evaluation.summary import category_breakdown, summarise  # noqa: E402
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
    k: Optional[int] = None,
) -> float:
    """``P@K`` (eq. 8): share of the K slots holding a chunk of an expected
    doc type or containing an expected keyword. ``k=None`` uses the number
    of retrieved chunks (the same value whenever K chunks were retrieved)."""
    from src.evaluation.retrieval_metrics import precision_at_k

    return precision_at_k(retrieved_chunks, expected_doc_types, expected_keywords, k)


def query_record(
    test_query: dict[str, Any],
    result: Any,
    elapsed: float,
    top_k: int = RERANKED_TOP_K,
) -> dict[str, Any]:
    """Score one pipeline result against its test query (no LLM involved)."""
    answer = result.answer_text or ""
    sources = result.sources
    retrieved = [c for c in sources if c.get("chunk_id") != MAPPING_CHUNK_ID]
    retrieval = retrieval_metrics(
        retrieved,
        test_query.get("expected_doc_types", []),
        test_query.get("expected_keywords", []),
        k=top_k,
    )
    return {
        "query": test_query["query"],
        "category": test_query.get("category", "unknown"),
        **compute_citation_quality(answer, sources),
        "hallucinations": detect_hallucinations(answer, sources),
        "is_substantive": is_substantive_response(answer, len(sources)),
        "confidence": result.confidence.overall_score if result.confidence else 0.0,
        "retrieval_precision": round(retrieval["P"], 4),
        "keyword_precision": round(retrieval["P_kw"], 4),
        "hit": retrieval["Hit"],
        "reciprocal_rank": round(retrieval["RR"], 4),
        "ipc_bns_matches": [m["ipc_section"] for m in result.ipc_bns_mappings],
        "num_chunks_retrieved": len(retrieved),
        "response_length": len(answer),
        "elapsed_seconds": round(elapsed, 3),
        "error": result.error,
    }


def evaluate(test_queries_path: Path | None = None, limit: int | None = None) -> dict[str, Any]:
    """Run every test query and return ``{"summary", "details", "per_category"}``."""
    from src.pipeline import LegalResearchPipeline

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

        details.append(query_record(tq, result, elapsed))

    summary: dict[str, Any] = {
        **summarise(details),
        "llm_model": "nyaya-gen (local)" if LLM_BACKEND == "local" else LLM_MODEL,
        "top_k": RERANKED_TOP_K,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return {
        "summary": summary,
        "details": details,
        "per_category": category_breakdown(details),
    }


def _fmt(value: Optional[float], spec: str = ".2f") -> str:
    return "-" if value is None else format(value, spec)


def _print_summary_table(summary: dict[str, Any]) -> None:
    """Print aggregate metrics in the layout of Table IV."""
    border = "+" + "-" * 38 + "+" + "-" * 14 + "+"
    header = f"| {'Metric':<36} | {'Value':>12} |"

    print("\n" + "=" * 56)
    print("  EVALUATION RESULTS (Table IV)")
    print("=" * 56)
    print(border)
    print(header)
    print(border)

    rows: list[tuple[str, str]] = [
        ("Total queries", str(summary.get("total_queries", 0))),
        ("Citation validity", _fmt(summary.get("citation_validity_mean"))),
        ("Cited claim coverage", _fmt(summary.get("cited_claim_coverage_mean"))),
        ("Citation support", _fmt(summary.get("citation_support_score_mean"))),
        ("Citation quality (blend)", _fmt(summary.get("citation_quality_score_mean"))),
        ("Hallucination rate (lower is better)", _fmt(summary.get("hallucination_rate"))),
        ("Response rate", _fmt(summary.get("response_rate"))),
        ("Retrieval precision P@K", _fmt(summary.get("retrieval_precision_mean"))),
        ("Keyword precision P_kw@K", _fmt(summary.get("keyword_precision_mean"))),
        ("Hit@K", _fmt(summary.get("hit_rate"))),
        ("MRR", _fmt(summary.get("mrr"))),
        ("Average confidence", _fmt(summary.get("average_confidence"))),
        ("Mean latency (s)", _fmt(summary.get("mean_latency_seconds"), ".1f")),
    ]

    for label, value in rows:
        print(f"| {label:<36} | {value:>12} |")

    print(border)
    if summary.get("llm_model"):
        print(f"  Model: {summary['llm_model']}   K = {summary.get('top_k', '-')}")
    print(f"  Evaluated at: {summary.get('timestamp', 'N/A')}")
    print()


def _print_category_breakdown(details: list[dict[str, Any]]) -> None:
    """Print the per-category breakdown in the layout of Table V, plus the
    keyword retrieval metrics and latency."""
    columns = [
        ("Validity", "citation_validity_mean", ".2f"),
        ("Support", "citation_support_score_mean", ".2f"),
        ("Resp.", "response_rate", ".2f"),
        ("Ret. P", "retrieval_precision_mean", ".2f"),
        ("Conf.", "average_confidence", ".2f"),
        ("P_kw", "keyword_precision_mean", ".2f"),
        ("Hit", "hit_rate", ".2f"),
        ("MRR", "mrr", ".2f"),
        ("Lat. s", "mean_latency_seconds", ".1f"),
    ]
    border = "+" + "-" * 22 + "+" + ("-" * 10 + "+") * len(columns)
    header = f"| {'Category (n)':<20} |" + "".join(f" {name:>8} |" for name, _, _ in columns)

    print("  PER-CATEGORY BREAKDOWN (Table V)")
    print(border)
    print(header)
    print(border)

    for category, metrics in category_breakdown(details).items():
        label = f"{category} ({metrics['total_queries']})"
        cells = "".join(f" {_fmt(metrics.get(key), spec):>8} |" for _, key, spec in columns)
        print(f"| {label:<20} |{cells}")

    print(border)
    print()


def _summarise_saved_run(path: Path) -> None:
    """Re-print Tables IV and V for a results file written by this script."""
    with path.open("r", encoding="utf-8") as fh:
        saved = json.load(fh)
    details = saved.get("details", [])
    summary = {**summarise(details), "timestamp": saved.get("summary", {}).get("timestamp")}
    for key in ("llm_model", "top_k"):
        if key in saved.get("summary", {}):
            summary[key] = saved["summary"][key]
    print(f"\n  Saved run: {path}")
    _print_summary_table(summary)
    _print_category_breakdown(details)


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
    parser.add_argument(
        "--summarise",
        type=Path,
        default=None,
        metavar="RESULTS_JSON",
        help="print Tables IV and V for a saved results file and exit "
        "(no index or LLM needed)",
    )
    args = parser.parse_args()

    if args.summarise is not None:
        _summarise_saved_run(args.summarise)
        return

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
    output = {
        "summary": summary,
        "details": details,
        "per_category": results.get("per_category", {}),
    }

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2, default=str)

    logger.info("Detailed results saved to %s", output_path)


if __name__ == "__main__":
    main()
