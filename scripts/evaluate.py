"""Evaluation script for the Legal Research Assistant.

Loads test queries from ``data/evaluation/test_queries.json``, runs each
through the full pipeline, computes metrics, and writes results to both
the terminal (formatted table) and ``data/evaluation/eval_results.json``.

Usage:
    python scripts/evaluate.py

Requires:
    - GEMINI_API_KEY in environment (or .env file)
    - ChromaDB collection populated via the ingestion pipeline
    - sentence-transformers model available
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Path bootstrap ──────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import (
    CONFIDENCE_WEIGHTS,
    EVAL_DIR,
    RERANKED_TOP_K,
    SEMANTIC_TOP_K,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("evaluate")

# ── Known Indian case names (used for hallucination detection) ──────
_KNOWN_CASE_FRAGMENTS: tuple[str, ...] = (
    "Kesavananda Bharati",
    "Maneka Gandhi",
    "Vishaka",
    "ADM Jabalpur",
    "Navtej Singh Johar",
    "Puttaswamy",
    "MC Mehta",
    "Indira Gandhi",
    "Shreya Singhal",
    "Golaknath",
    "Minerva Mills",
    "S.R. Bommai",
    "Menaka Gandhi",
    "State of Maharashtra v. Prakash",
    "Sharad Birdhichand",
)

_FABRICATED_CITATION_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:AIR|SCC|SCR)\s+\d{4}\s+\w+\s+\d+",
    re.IGNORECASE,
)

_CASE_NAME_PATTERN: re.Pattern[str] = re.compile(
    r"([A-Z][A-Za-z.\s]+?)\s+(?:v\.?s?\.?|versus)\s+([A-Z][A-Za-z.\s]+?)(?=\s*[\n,;(.\[]|$)",
    re.MULTILINE,
)


# ── Metric computation helpers ──────────────────────────────────────

_SOURCE_REF_PATTERN: re.Pattern[str] = re.compile(r"\[SOURCE_(\d+)\]")
_SENTENCE_SPLIT_PATTERN: re.Pattern[str] = re.compile(r"(?<=[.!?])\s+")
_LEGAL_ANCHOR_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:section|sec\.?|article|act|ipc|crpc|bns|constitution|court|"
    r"judg(?:e)?ment|case|conviction|murder|theft|bail|appeal|right|"
    r"punishment|imprisonment|fine|liability|offence|offense)\b",
    re.IGNORECASE,
)
_SECTION_OR_ARTICLE_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:section|sec\.?|article)\s+\d+[A-Za-z-]*\b",
    re.IGNORECASE,
)
_LEGAL_CONCEPT_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:murder|theft|bail|appeal|conviction|right|punish(?:ment|ed)?|"
    r"imprisonment|fine|liability|offence|offense)\b",
    re.IGNORECASE,
)
_STOPWORDS: set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "under",
    "with",
}


def _compute_citation_validity(
    response_text: str,
    source_chunks: list[dict[str, Any]],
) -> float:
    """Return the fraction of [SOURCE_N] references that map to real chunks.

    This metric only verifies that citation tags point to provided source
    chunks. It does not prove that the cited source supports the attached
    claim.
    """
    refs = _SOURCE_REF_PATTERN.findall(response_text)
    if not refs:
        return 0.0

    valid = sum(1 for r in refs if 1 <= int(r) <= len(source_chunks))
    return valid / len(refs)


def _compute_citation_accuracy(
    response_text: str,
    source_chunks: list[dict[str, Any]],
) -> float:
    """Backward-compatible alias for source-link validity.

    Prefer ``_compute_citation_validity`` for new code. The old name was
    misleading because it measured source-ID validity, not semantic support.
    """
    return _compute_citation_validity(response_text, source_chunks)


def _claim_sentences(response_text: str) -> list[str]:
    """Extract substantive legal claim sentences from an answer."""
    cleaned = response_text.replace("\n", " ")
    raw_sentences = [
        part.strip()
        for part in _SENTENCE_SPLIT_PATTERN.split(cleaned)
        if part.strip()
    ]
    sentences: list[str] = []
    for part in raw_sentences:
        if _SOURCE_REF_PATTERN.fullmatch(part.strip()) and sentences:
            sentences[-1] = f"{sentences[-1]} {part}"
        else:
            sentences.append(part)

    claims: list[str] = []
    for sentence in sentences:
        without_refs = _SOURCE_REF_PATTERN.sub("", sentence).strip()
        if len(without_refs) < 25:
            continue
        if _LEGAL_ANCHOR_PATTERN.search(without_refs) or _SECTION_OR_ARTICLE_PATTERN.search(
            without_refs
        ):
            claims.append(sentence)
    return claims


def _compute_cited_claim_coverage(response_text: str) -> float:
    """Return the fraction of substantive legal claims with nearby citations."""
    claims = _claim_sentences(response_text)
    if not claims:
        return 0.0

    cited = sum(1 for claim in claims if _SOURCE_REF_PATTERN.search(claim))
    return cited / len(claims)


def _content_tokens(text: str) -> set[str]:
    """Return lower-cased content tokens used by the support proxy."""
    return {
        token
        for token in re.findall(r"[A-Za-z][A-Za-z0-9]+", text.lower())
        if len(token) > 2 and token not in _STOPWORDS
    }


def _legal_anchors(text: str) -> set[str]:
    """Return legal identifiers that should match between claim and source."""
    anchors = {
        match.group(0).lower()
        for match in _SECTION_OR_ARTICLE_PATTERN.finditer(text)
    }
    anchors.update(
        token.lower()
        for token in re.findall(r"\b(?:ipc|crpc|bns|constitution)\b", text, re.I)
    )
    return anchors


def _legal_concepts(text: str) -> set[str]:
    """Return legal concept terms that should be grounded in the source."""
    concepts: set[str] = set()
    for match in _LEGAL_CONCEPT_PATTERN.finditer(text):
        concept = match.group(0).lower()
        if concept.startswith("punish"):
            concept = "punish"
        concepts.add(concept)
    return concepts


def _claim_supported_by_source(claim: str, source_text: str) -> bool:
    """Conservative lexical/legal-anchor proxy for claim support.

    This is not a substitute for an LLM/NLI faithfulness judge. It catches
    obvious unsupported citations better than source-ID validation by requiring
    the cited text to share legal anchors or meaningful content with the claim.
    """
    claim_without_refs = _SOURCE_REF_PATTERN.sub("", claim)
    claim_tokens = _content_tokens(claim_without_refs)
    source_tokens = _content_tokens(source_text)
    if not claim_tokens or not source_tokens:
        return False

    claim_anchors = _legal_anchors(claim_without_refs)
    source_anchors = _legal_anchors(source_text)
    if claim_anchors and not claim_anchors.issubset(source_anchors):
        return False

    claim_concepts = _legal_concepts(claim_without_refs)
    source_concepts = _legal_concepts(source_text)
    if claim_concepts and not claim_concepts.issubset(source_concepts):
        return False

    overlap = claim_tokens & source_tokens
    overlap_rate = len(overlap) / len(claim_tokens)
    return overlap_rate >= 0.35 or len(overlap) >= 5


def _compute_claim_support_rate(
    response_text: str,
    source_chunks: list[dict[str, Any]],
) -> float:
    """Return supported cited legal claims divided by all cited legal claims."""
    cited_claims = [
        claim for claim in _claim_sentences(response_text) if _SOURCE_REF_PATTERN.search(claim)
    ]
    if not cited_claims:
        return 0.0

    supported = 0
    for claim in cited_claims:
        refs = [int(ref) for ref in _SOURCE_REF_PATTERN.findall(claim)]
        valid_sources = [
            source_chunks[ref - 1]
            for ref in refs
            if 1 <= ref <= len(source_chunks)
        ]
        if any(
            _claim_supported_by_source(claim, chunk.get("text", ""))
            for chunk in valid_sources
        ):
            supported += 1

    return supported / len(cited_claims)


def _compute_citation_quality(
    response_text: str,
    source_chunks: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute citation validity, coverage, support, and blended quality."""
    validity = _compute_citation_validity(response_text, source_chunks)
    coverage = _compute_cited_claim_coverage(response_text)
    support = _compute_claim_support_rate(response_text, source_chunks)
    blended = 0.30 * validity + 0.30 * coverage + 0.40 * support
    return {
        "citation_validity": round(validity, 4),
        "cited_claim_coverage": round(coverage, 4),
        "citation_support_score": round(support, 4),
        "citation_quality_score": round(blended, 4),
    }


def _detect_hallucinations(
    response_text: str,
    source_chunks: list[dict[str, Any]],
) -> list[str]:
    """Return a list of potentially fabricated case names or citations.

    A case name is considered *potentially fabricated* if it appears in the
    response but does not match any known case fragment and is not present
    in any of the source chunk texts.
    """
    hallucinations: list[str] = []

    # Check case names in the response
    for match in _CASE_NAME_PATTERN.finditer(response_text):
        full_name = match.group(0).strip()
        # Check against known case fragments
        is_known = any(
            frag.lower() in full_name.lower() for frag in _KNOWN_CASE_FRAGMENTS
        )
        if is_known:
            continue
        # Check if it appears in any source chunk
        in_sources = any(
            full_name.lower() in chunk.get("text", "").lower()
            for chunk in source_chunks
        )
        if not in_sources:
            hallucinations.append(f"Case: {full_name}")

    # Check citations (AIR, SCC, SCR) not grounded in sources
    for match in _FABRICATED_CITATION_PATTERN.finditer(response_text):
        citation = match.group(0)
        in_sources = any(
            citation.lower() in chunk.get("text", "").lower() for chunk in source_chunks
        )
        if not in_sources:
            hallucinations.append(f"Citation: {citation}")

    return hallucinations


def _is_substantive_response(response_text: str) -> bool:
    """Return True if the response provides a substantive answer."""
    insufficient_markers = (
        "do not contain sufficient information",
        "insufficient information",
        "cannot answer",
        "no relevant sources",
        "unable to provide",
    )
    lower = response_text.lower()
    if any(marker in lower for marker in insufficient_markers):
        return False
    # Very short responses are not substantive
    return len(response_text.strip()) > 50


def _compute_retrieval_precision(
    retrieved_chunks: list[dict[str, Any]],
    expected_doc_types: list[str],
    expected_keywords: list[str],
) -> float:
    """Return the fraction of retrieved chunks that are relevant.

    A chunk is considered relevant if its doc_type matches one of the
    expected types OR its text contains at least one expected keyword.
    """
    if not retrieved_chunks:
        return 0.0

    relevant = 0
    for chunk in retrieved_chunks:
        doc_type = chunk.get("doc_type", chunk.get("metadata", {}).get("doc_type", ""))
        text = chunk.get("text", "").lower()

        type_match = doc_type in expected_doc_types if expected_doc_types else False
        keyword_match = (
            any(kw.lower() in text for kw in expected_keywords)
            if expected_keywords
            else False
        )

        if type_match or keyword_match:
            relevant += 1

    return relevant / len(retrieved_chunks)


def _compute_confidence(
    cited_response: Any,
    reranked: list[dict[str, Any]],
) -> float:
    """Compute aggregate confidence score in [0, 1]."""
    if not reranked:
        return 0.0

    retrieval_quality = sum(r.get("final_score", 0.0) for r in reranked) / len(reranked)

    total_sources = len(reranked)
    cited_count = getattr(cited_response, "citation_count", 0)
    source_agreement = cited_count / total_sources if total_sources else 0.0

    authority_scores = [r.get("authority_score", 0.0) for r in reranked]
    authority_strength = (
        sum(authority_scores) / len(authority_scores) if authority_scores else 0.0
    )

    coverage = 1.0 if cited_count > 0 else 0.0

    score = (
        CONFIDENCE_WEIGHTS["retrieval_quality"] * retrieval_quality
        + CONFIDENCE_WEIGHTS["source_agreement"] * source_agreement
        + CONFIDENCE_WEIGHTS["authority_strength"] * authority_strength
        + CONFIDENCE_WEIGHTS["coverage_score"] * coverage
    )
    return round(min(max(score, 0.0), 1.0), 4)


# ── Pipeline runner ─────────────────────────────────────────────────


def _run_query(
    query: str,
    hybrid_search: Any,
    reranker: Any,
    prompt_builder: Any,
    gemini_client: Any,
    response_parser: Any,
) -> dict[str, Any]:
    """Run a single query through the pipeline and return raw results."""
    if not query.strip():
        return {
            "raw_response": "",
            "reranked_chunks": [],
            "cited_response": None,
            "confidence": 0.0,
            "error": "Empty query",
        }

    try:
        raw_results = hybrid_search.search(query, top_k=SEMANTIC_TOP_K)
    except Exception as exc:
        logger.warning("Retrieval failed for query '%s': %s", query[:60], exc)
        return {
            "raw_response": "",
            "reranked_chunks": [],
            "cited_response": None,
            "confidence": 0.0,
            "error": f"Retrieval error: {exc}",
        }

    try:
        reranked = reranker.rerank(raw_results, top_k=RERANKED_TOP_K)
    except Exception as exc:
        logger.warning("Reranking failed for query '%s': %s", query[:60], exc)
        reranked = raw_results[:RERANKED_TOP_K]

    try:
        sys_prompt, user_prompt = prompt_builder.build_prompt(query, reranked)
    except Exception as exc:
        logger.warning("Prompt building failed for query '%s': %s", query[:60], exc)
        return {
            "raw_response": "",
            "reranked_chunks": reranked,
            "cited_response": None,
            "confidence": 0.0,
            "error": f"Prompt error: {exc}",
        }

    try:
        raw_response = gemini_client.generate(user_prompt, system_prompt=sys_prompt)
    except Exception as exc:
        logger.warning("Generation failed for query '%s': %s", query[:60], exc)
        return {
            "raw_response": "",
            "reranked_chunks": reranked,
            "cited_response": None,
            "confidence": 0.0,
            "error": f"Generation error: {exc}",
        }

    try:
        cited_response = response_parser.parse_response(raw_response, reranked)
    except Exception as exc:
        logger.warning("Parsing failed for query '%s': %s", query[:60], exc)
        return {
            "raw_response": raw_response,
            "reranked_chunks": reranked,
            "cited_response": None,
            "confidence": 0.0,
            "error": f"Parse error: {exc}",
        }

    confidence = _compute_confidence(cited_response, reranked)

    return {
        "raw_response": raw_response,
        "reranked_chunks": reranked,
        "cited_response": cited_response,
        "confidence": confidence,
        "error": None,
    }


# ── Evaluation driver ──────────────────────────────────────────────


def evaluate(test_queries_path: Path | None = None) -> dict[str, Any]:
    """Run the full evaluation suite and return aggregate metrics.

    Parameters
    ----------
    test_queries_path:
        Path to the JSON file with test queries.  Defaults to
        ``data/evaluation/test_queries.json``.

    Returns
    -------
    dict
        Contains ``summary`` (aggregate metrics) and ``details``
        (per-query results).
    """
    if test_queries_path is None:
        test_queries_path = EVAL_DIR / "test_queries.json"

    with test_queries_path.open("r", encoding="utf-8") as fh:
        test_queries: list[dict[str, Any]] = json.load(fh)

    logger.info("Loaded %d test queries from %s", len(test_queries), test_queries_path)

    # ── Initialise pipeline components ────────────────────────────
    try:
        from src.embeddings.embedding_service import EmbeddingService
        from src.embeddings.vectorstore import VectorStore
        from src.generation.gemini_client import GeminiClient
        from src.generation.prompt_builder import PromptBuilder
        from src.generation.response_parser import ResponseParser
        from src.retrieval.hybrid_search import HybridSearch
        from src.retrieval.reranker import AuthorityReranker

        embedding_service = EmbeddingService()
        vectorstore = VectorStore()
        hybrid_search = HybridSearch(vectorstore._collection, embedding_service)
        reranker = AuthorityReranker()
        prompt_builder = PromptBuilder()
        gemini_client = GeminiClient()
        response_parser = ResponseParser()
    except Exception as exc:
        logger.error("Failed to initialise pipeline: %s", exc)
        logger.info(
            "Ensure GEMINI_API_KEY is set, ChromaDB is populated, "
            "and all dependencies are installed."
        )
        return {"summary": {}, "details": [], "error": str(exc)}

    # ── Run queries ───────────────────────────────────────────────
    details: list[dict[str, Any]] = []
    citation_validities: list[float] = []
    cited_claim_coverages: list[float] = []
    citation_support_scores: list[float] = []
    citation_quality_scores: list[float] = []
    hallucination_flags: list[bool] = []
    substantive_flags: list[bool] = []
    confidence_scores: list[float] = []
    retrieval_precisions: list[float] = []

    for i, tq in enumerate(test_queries):
        query = tq["query"]
        category = tq.get("category", "unknown")
        expected_doc_types = tq.get("expected_doc_types", [])
        expected_keywords = tq.get("expected_keywords", [])

        logger.info(
            "[%d/%d] Running query (%s): %s",
            i + 1,
            len(test_queries),
            category,
            query[:80],
        )

        start_time = time.monotonic()
        result = _run_query(
            query,
            hybrid_search,
            reranker,
            prompt_builder,
            gemini_client,
            response_parser,
        )
        elapsed = time.monotonic() - start_time

        raw_response = result["raw_response"]
        reranked_chunks = result["reranked_chunks"]
        error = result["error"]

        # Compute per-query metrics. Citation validity only checks that a
        # [SOURCE_N] tag maps to a provided chunk; support is scored separately.
        citation_quality = _compute_citation_quality(raw_response, reranked_chunks)
        hallucinations = _detect_hallucinations(raw_response, reranked_chunks)
        is_substantive = _is_substantive_response(raw_response)
        ret_precision = _compute_retrieval_precision(
            reranked_chunks,
            expected_doc_types,
            expected_keywords,
        )
        confidence = result["confidence"]

        citation_validities.append(citation_quality["citation_validity"])
        cited_claim_coverages.append(citation_quality["cited_claim_coverage"])
        citation_support_scores.append(citation_quality["citation_support_score"])
        citation_quality_scores.append(citation_quality["citation_quality_score"])
        hallucination_flags.append(len(hallucinations) > 0)
        substantive_flags.append(is_substantive)
        confidence_scores.append(confidence)
        retrieval_precisions.append(ret_precision)

        detail_entry: dict[str, Any] = {
            "query": query,
            "category": category,
            **citation_quality,
            "hallucinations": hallucinations,
            "is_substantive": is_substantive,
            "confidence": confidence,
            "retrieval_precision": round(ret_precision, 4),
            "num_chunks_retrieved": len(reranked_chunks),
            "response_length": len(raw_response),
            "elapsed_seconds": round(elapsed, 2),
            "error": error,
        }
        details.append(detail_entry)

    # ── Aggregate metrics ─────────────────────────────────────────
    n = len(test_queries) or 1
    summary: dict[str, Any] = {
        "total_queries": len(test_queries),
        "citation_validity_mean": round(sum(citation_validities) / n, 4),
        "cited_claim_coverage_mean": round(sum(cited_claim_coverages) / n, 4),
        "citation_support_score_mean": round(sum(citation_support_scores) / n, 4),
        "citation_quality_score_mean": round(sum(citation_quality_scores) / n, 4),
        "hallucination_rate": round(sum(hallucination_flags) / n, 4),
        "response_rate": round(sum(substantive_flags) / n, 4),
        "average_confidence": round(sum(confidence_scores) / n, 4),
        "retrieval_precision_mean": round(sum(retrieval_precisions) / n, 4),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return {"summary": summary, "details": details}


# ── Output formatting ──────────────────────────────────────────────


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


# ── Main ────────────────────────────────────────────────────────────


def main() -> None:
    logger.info("Starting evaluation...")
    results = evaluate()

    if results.get("error"):
        logger.error("Evaluation aborted: %s", results["error"])
        sys.exit(1)

    summary = results["summary"]
    details = results["details"]

    # Print to terminal
    _print_summary_table(summary)
    _print_category_breakdown(details)

    # Save to JSON
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EVAL_DIR / "eval_results.json"

    # Convert details for JSON serialisation (strip non-serialisable objects)
    serialisable_details = []
    for d in details:
        entry = {k: v for k, v in d.items()}
        serialisable_details.append(entry)

    output = {
        "summary": summary,
        "details": serialisable_details,
    }

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2, default=str)

    logger.info("Detailed results saved to %s", output_path)


if __name__ == "__main__":
    main()
