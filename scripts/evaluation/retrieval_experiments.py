"""Reproduce the retrieval experiments of the report (Tables III, VI, VII, VIII).

No LLM is involved. For each of the non-empty test queries the script runs
the deployed IPC/BNS query expansion and then the dense path and the
exact-match path ONCE with a large pool (default 40 per path), caching the
results in ``data/evaluation/retrieval_cache.json``. Every configuration is
then evaluated from that cache, so all of them see the same candidates:

* Table III   (a) dense only, (b) exact match only, (c) hybrid with fixed RRF
              0.5/0.5, (d) hybrid with adaptive RRF, (e) (d) + re-rank without
              authority, (f) proposed: (e) + authority; plus P_kw@4 per category
* Table VI    K in {2, 4, 6, 8, 10}
* Table VII   RRF k in {1, 10, 30, 60, 100}; pool per path in {5, 10, 20, 30, 40}
* Table VIII  final-score weights (w_s, w_r, w_a) and fixed vs adaptive fusion

The ranking code is the deployed code: ``HybridSearch.classify_query`` and
``search_paths``, ``fuse_hits`` (weighted RRF), ``_drop_duplicate_texts``
(dedupe after fusion), ``AuthorityReranker.rerank`` (authority and final
score) and ``IpcBnsMapper`` (query expansion). Unless a table varies them,
settings are the deployed ones: pool 20 per path, K = 4, k = 60, adaptive
fusion weights and (w_s, w_r, w_a) = (0.50, 0.25, 0.25).

Usage:
    python scripts/evaluation/retrieval_experiments.py                # build cache if needed, print tables
    python scripts/evaluation/retrieval_experiments.py --refresh      # re-run both search paths

Building the cache needs the ChromaDB index and the embedding model; once it
exists, the tables are computed from it alone.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import (  # noqa: E402
    AUTHORITY_WEIGHT,
    EMBEDDING_MODEL_NAME,
    EVAL_DIR,
    RERANKED_TOP_K,
    RRF_K,
    RRF_WEIGHT,
    SEMANTIC_TOP_K,
    SEMANTIC_WEIGHT,
)
from src.evaluation.retrieval_metrics import (  # noqa: E402
    mean_retrieval_metrics,
    retrieval_metrics,
)
from src.retrieval.hybrid_search import (  # noqa: E402
    _CLASSIFICATION_WEIGHTS,
    HybridSearch,
    _extract_keyword_terms,
    fuse_hits,
)
from src.retrieval.reranker import AuthorityReranker  # noqa: E402
from src.retrieval.retrieval_pipeline import _drop_duplicate_texts  # noqa: E402

logger = logging.getLogger("retrieval_experiments")

DEFAULT_CACHE_PATH = EVAL_DIR / "retrieval_cache.json"
DEFAULT_OUTPUT_PATH = EVAL_DIR / "retrieval_experiments.json"
DEFAULT_QUERIES_PATH = EVAL_DIR / "test_queries.json"
DEFAULT_CACHE_POOL = 40
CACHE_VERSION = 1

DEPLOYED_WEIGHTS: tuple[float, float, float] = (SEMANTIC_WEIGHT, RRF_WEIGHT, AUTHORITY_WEIGHT)
# Row (e): authority removed and its 0.25 moved to the RRF term, (0.5, 0.5, 0).
# This is the setting whose numbers equal row (e) in the report (Table VIII
# left, P_kw 0.842 = row (d)/(e), and "the authority prior costs 3.1 points").
NO_AUTHORITY_WEIGHTS: tuple[float, float, float] = (SEMANTIC_WEIGHT, RRF_WEIGHT + AUTHORITY_WEIGHT, 0.0)
# Row (e'): the same with (w_s, w_r) renormalised proportionally, (2/3, 1/3, 0).
_SR = SEMANTIC_WEIGHT + RRF_WEIGHT
PROPORTIONAL_NO_AUTHORITY_WEIGHTS: tuple[float, float, float] = (
    SEMANTIC_WEIGHT / _SR,
    RRF_WEIGHT / _SR,
    0.0,
)

TABLE_VI_K = (2, 4, 6, 8, 10)
TABLE_VII_RRF_K = (1, 10, 30, 60, 100)
TABLE_VII_POOL = (5, 10, 20, 30, 40)
TABLE_VIII_WEIGHTS: tuple[tuple[float, float, float], ...] = (
    (1.0, 0.0, 0.0),
    (0.7, 0.3, 0.0),
    (0.5, 0.5, 0.0),
    (0.6, 0.2, 0.2),
    (0.4, 0.4, 0.2),
    (0.34, 0.33, 0.33),
    (0.5, 0.25, 0.25),
)
TABLE_VIII_FUSION: tuple[Any, ...] = (
    (1.0, 0.0),
    (0.8, 0.2),
    (0.5, 0.5),
    (0.2, 0.8),
    (0.0, 1.0),
    "adaptive",
)

# Configurations


@dataclass(frozen=True)
class Config:
    """One retrieval configuration evaluated from the cache.

    ``mode`` is ``"dense"`` (dense list only), ``"exact"`` (exact-match list
    only), ``"fused"`` (weighted RRF, ranked by ``R``) or ``"rerank"``
    (weighted RRF, then ranked by the final score ``S``). ``fusion`` is
    ``"adaptive"`` (per query type) or a fixed ``(alpha, beta)``.
    """

    mode: str = "rerank"
    fusion: Any = "adaptive"
    rrf_k: int = RRF_K
    pool: int = SEMANTIC_TOP_K
    weights: tuple[float, float, float] = DEPLOYED_WEIGHTS
    top_k: int = RERANKED_TOP_K
    dedupe: bool = True


PROPOSED = Config()

TABLE_III_ROWS: tuple[tuple[str, str, Config], ...] = (
    ("a", "(a) Dense only (standard RAG)", Config(mode="dense")),
    ("b", "(b) Exact match only", Config(mode="exact")),
    ("c", "(c) Hybrid, fixed RRF (0.5/0.5)", Config(mode="fused", fusion=(0.5, 0.5))),
    ("d", "(d) Hybrid, adaptive RRF", Config(mode="fused")),
    ("e", "(e) (d) + re-rank, no authority", Config(weights=NO_AUTHORITY_WEIGHTS)),
    ("f", "(f) Proposed: (e) + authority", PROPOSED),
)
# Not in the report's table: the proportional reading of row (e), and dense
# retrieval without de-duplication (Section IV-C, Hit@4 0.776 -> 0.796).
EXTRA_ROWS: tuple[tuple[str, str, Config], ...] = (
    ("e'", "(e') (d) + re-rank, no authority, (2/3, 1/3, 0)",
     Config(weights=PROPORTIONAL_NO_AUTHORITY_WEIGHTS)),
    ("a0", "(a0) Dense only, no de-duplication", Config(mode="dense", dedupe=False)),
)

# Cache


def _hit_list(hits: Sequence[dict[str, Any]]) -> list[list[Any]]:
    return [[h["chunk_id"], float(h.get("semantic_score", 0.0))] for h in hits]


def build_cache(
    test_queries: Sequence[dict[str, Any]],
    hybrid_search: Any,
    mapper: Any = None,
    pool: int = DEFAULT_CACHE_POOL,
    meta: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Run the IPC/BNS expansion and both search paths once per query.

    *hybrid_search* needs ``search_paths(query, top_k)`` (``HybridSearch``);
    *mapper* is an ``IpcBnsMapper`` or ``None``. Empty queries are skipped,
    as in the report (49 of the 50 test queries).
    """
    chunks: dict[str, dict[str, Any]] = {}
    entries: list[dict[str, Any]] = []
    for index, tq in enumerate(test_queries):
        query = tq.get("query", "")
        if not query or not query.strip():
            continue

        matches = mapper.find(query) if mapper is not None else []
        retrieval_query = mapper.expand_query(query, matches) if matches else query

        start = time.perf_counter()
        paths = hybrid_search.search_paths(retrieval_query, top_k=pool)
        elapsed = time.perf_counter() - start

        for hit in [*paths["semantic_hits"], *paths["keyword_hits"]]:
            chunks.setdefault(hit["chunk_id"], {
                "text": hit.get("text", ""),
                "metadata": hit.get("metadata") or {},
            })

        entries.append({
            "index": index,
            "query": query,
            "category": tq.get("category", "unknown"),
            "expected_doc_types": tq.get("expected_doc_types", []),
            "expected_keywords": tq.get("expected_keywords", []),
            "retrieval_query": retrieval_query,
            "classification": paths["classification"],
            "keyword_terms": _extract_keyword_terms(retrieval_query),
            "ipc_bns_matches": [m["ipc_section"] for m in matches],
            "dense": _hit_list(paths["semantic_hits"]),
            "exact": _hit_list(paths["keyword_hits"]),
            "search_seconds": round(elapsed, 4),
        })
        logger.info(
            "[%d] %s: %d dense, %d exact (%.1fs) %.60s",
            index, paths["classification"], len(paths["semantic_hits"]),
            len(paths["keyword_hits"]), elapsed, query,
        )

    return {
        "meta": {
            "version": CACHE_VERSION,
            "pool": pool,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "created": datetime.now(timezone.utc).isoformat(),
            **(meta or {}),
        },
        "queries": entries,
        "chunks": chunks,
    }


def check_cache(cache: dict[str, Any], test_queries: Sequence[dict[str, Any]], pool: int) -> None:
    """Raise ``ValueError`` when *cache* does not cover *test_queries* at *pool*."""
    if cache.get("meta", {}).get("version") != CACHE_VERSION:
        raise ValueError("cache version differs; rebuild it with --refresh")
    if cache["meta"].get("pool", 0) < pool:
        raise ValueError(
            f"cache pool {cache['meta'].get('pool')} < requested {pool}; rebuild with --refresh"
        )
    cached = [e["query"] for e in cache["queries"]]
    wanted = [q["query"] for q in test_queries if q.get("query", "").strip()]
    if cached != wanted:
        raise ValueError("cached queries differ from the test set; rebuild with --refresh")


# Ranking from the cache


def _hits(entry: dict[str, Any], path: str, chunks: dict[str, Any], pool: int) -> list[dict[str, Any]]:
    """Rebuild the top-*pool* hit dicts of one path, as ``HybridSearch`` returns them."""
    hits = []
    for chunk_id, score in entry[path][:pool]:
        chunk = chunks[chunk_id]
        hits.append({
            "chunk_id": chunk_id,
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "semantic_score": score,
        })
    return hits


def classification(entry: dict[str, Any]) -> str:
    """Query type from the deployed classifier, on the expanded query."""
    return HybridSearch.classify_query(entry["retrieval_query"])


def fusion_weights(entry: dict[str, Any], fusion: Any) -> tuple[float, float]:
    if fusion == "adaptive":
        return _CLASSIFICATION_WEIGHTS[classification(entry)]
    alpha, beta = fusion
    return float(alpha), float(beta)


def candidate_pool(entry: dict[str, Any], chunks: dict[str, Any], cfg: Config) -> list[dict[str, Any]]:
    """The candidate list of *cfg* before re-ranking and the top-K cut.

    Dense or exact list, or the weighted-RRF fusion of both (sorted by R),
    with identical normalised texts removed after fusion.
    """
    dense = _hits(entry, "dense", chunks, cfg.pool)
    exact = _hits(entry, "exact", chunks, cfg.pool)
    if cfg.mode == "dense":
        ranked = dense
    elif cfg.mode == "exact":
        ranked = exact
    elif cfg.mode in ("fused", "rerank"):
        alpha, beta = fusion_weights(entry, cfg.fusion)
        ranked = fuse_hits(dense, exact, alpha, beta, k=cfg.rrf_k)
    else:
        raise ValueError(f"unknown mode {cfg.mode!r}")
    return _drop_duplicate_texts(ranked) if cfg.dedupe else ranked


_RERANKER = AuthorityReranker()


def score_candidates(candidates: list[dict[str, Any]], cfg: Config) -> list[dict[str, Any]]:
    """All candidates ranked by the final score S of *cfg* (eq. 6)."""
    if not candidates:
        return []
    return _RERANKER.rerank(
        candidates, top_k=len(candidates), weights=cfg.weights, rrf_k=cfg.rrf_k,
    )


def rank_query(entry: dict[str, Any], chunks: dict[str, Any], cfg: Config) -> list[dict[str, Any]]:
    """The top-K chunks that *cfg* passes to the generator for *entry*."""
    candidates = candidate_pool(entry, chunks, cfg)
    if cfg.mode == "rerank":
        candidates = score_candidates(candidates, cfg)
    return candidates[:cfg.top_k]


def _metrics(entry: dict[str, Any], ranked: list[dict[str, Any]], k: int) -> dict[str, float]:
    return retrieval_metrics(ranked, entry["expected_doc_types"], entry["expected_keywords"], k=k)


def summarise_rankings(
    cache: dict[str, Any],
    rankings: Sequence[list[dict[str, Any]]],
    k: int,
) -> dict[str, Any]:
    """Mean P@K, P_kw@K, Hit@K and MRR overall and per category."""
    per_query = [_metrics(e, r, k) for e, r in zip(cache["queries"], rankings)]
    by_category: dict[str, list[dict[str, float]]] = {}
    for entry, metrics in zip(cache["queries"], per_query):
        by_category.setdefault(entry["category"], []).append(metrics)
    return {
        **mean_retrieval_metrics(per_query),
        "per_category": {c: mean_retrieval_metrics(m) for c, m in by_category.items()},
        "per_query": per_query,
    }


def evaluate_config(cache: dict[str, Any], cfg: Config) -> dict[str, Any]:
    """Evaluate one configuration over every cached query."""
    rankings = [rank_query(e, cache["chunks"], cfg) for e in cache["queries"]]
    result = summarise_rankings(cache, rankings, cfg.top_k)
    result["top_chunk_ids"] = [[c["chunk_id"] for c in r] for r in rankings]
    return result


def duplicate_slot_share(cache: dict[str, Any], k: int = RERANKED_TOP_K, pool: int = SEMANTIC_TOP_K) -> float:
    """Share of dense top-K slots that repeat the text of an earlier slot."""
    slots = dupes = 0
    for entry in cache["queries"]:
        top = _hits(entry, "dense", cache["chunks"], pool)[:k]
        slots += len(top)
        dupes += len(top) - len(_drop_duplicate_texts(top))
    return dupes / slots if slots else 0.0


# Tables


def _row(label: str, result: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "label": label,
        "P": round(result["P"], 4),
        "P_kw": round(result["P_kw"], 4),
        "Hit": round(result["Hit"], 4),
        "MRR": round(result["MRR"], 4),
        **extra,
    }


def _per_query(cache: dict[str, Any], result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"query": e["query"], **{k: round(v, 4) for k, v in m.items()}, "top": ids}
        for e, m, ids in zip(cache["queries"], result["per_query"], result["top_chunk_ids"])
    ]


def table_iii(cache: dict[str, Any]) -> dict[str, Any]:
    rows, per_category, extra, per_query = [], {}, [], {}
    for row_id, label, cfg in TABLE_III_ROWS:
        result = evaluate_config(cache, cfg)
        rows.append(_row(label, result, id=row_id))
        per_category[row_id] = {c: round(m["P_kw"], 4) for c, m in result["per_category"].items()}
        per_query[row_id] = _per_query(cache, result)
    for row_id, label, cfg in EXTRA_ROWS:
        extra.append(_row(label, evaluate_config(cache, cfg), id=row_id))
    return {
        "rows": rows,
        "per_category_P_kw": per_category,
        "extra_rows": extra,
        "dense_duplicate_slot_share": round(duplicate_slot_share(cache), 4),
        "per_query": per_query,
    }


def table_vi(cache: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _row(f"K={k}", evaluate_config(cache, replace(PROPOSED, top_k=k)), K=k)
        for k in TABLE_VI_K
    ]


def table_vii(cache: dict[str, Any]) -> dict[str, Any]:
    max_pool = cache["meta"].get("pool", DEFAULT_CACHE_POOL)
    pools = [p for p in TABLE_VII_POOL if p <= max_pool]
    if len(pools) < len(TABLE_VII_POOL):
        logger.warning("Cache pool %d: pool sizes above it are skipped", max_pool)
    return {
        "rrf_k": [
            _row(f"k={k}", evaluate_config(cache, replace(PROPOSED, rrf_k=k)), k=k)
            for k in TABLE_VII_RRF_K
        ],
        "pool": [
            _row(f"pool={p}", evaluate_config(cache, replace(PROPOSED, pool=p)), pool=p)
            for p in pools
        ],
    }


def table_viii(cache: dict[str, Any]) -> dict[str, Any]:
    return {
        "score_weights": [
            _row(str(w), evaluate_config(cache, replace(PROPOSED, weights=w)), weights=list(w))
            for w in TABLE_VIII_WEIGHTS
        ],
        "fusion": [
            _row(
                "adaptive" if f == "adaptive" else f"fixed ({f[0]:.1f}, {f[1]:.1f})",
                evaluate_config(cache, replace(PROPOSED, fusion=f)),
                fusion=f if f == "adaptive" else list(f),
            )
            for f in TABLE_VIII_FUSION
        ],
    }


# Printing


def _fmt(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else str(value)


def _print_table(
    title: str,
    header: Sequence[str],
    rows: Sequence[Sequence[Any]],
    first_width: int = 40,
    width: int = 9,
) -> None:
    print(f"\n{title}")
    line = f"{header[0]:<{first_width}}" + "".join(f"{h:>{width}}" for h in header[1:])
    print(line)
    print("-" * len(line))
    for row in rows:
        print(f"{str(row[0]):<{first_width}}" + "".join(f"{_fmt(v):>{width}}" for v in row[1:]))


def print_report(results: dict[str, Any]) -> None:
    n = results["meta"]["queries"]
    t3 = results["table_III"]
    rows = t3["rows"] + t3["extra_rows"]
    _print_table(
        f"TABLE III. Retrieval comparison with the baseline methods ({n} queries).",
        ["Method", "P@4", "P_kw@4", "Hit@4", "MRR"],
        [[r["label"], r["P"], r["P_kw"], r["Hit"], r["MRR"]] for r in rows],
        first_width=52,
    )
    print(f"Dense top-4 slots repeating an earlier passage (no de-duplication): "
          f"{t3['dense_duplicate_slot_share']:.1%}")

    per_cat = dict(t3["per_category_P_kw"])
    categories = list(next(iter(per_cat.values())).keys()) if per_cat else []
    _print_table(
        "Keyword precision P_kw@4 per category (rows of Table III).",
        ["Row", *categories],
        [[row_id, *[values.get(c, 0.0) for c in categories]] for row_id, values in per_cat.items()],
        first_width=6,
        width=17,
    )

    _print_table(
        "TABLE VI. Effect of the number of passages K given to the LLM.",
        ["K", "P@K", "P_kw@K", "Hit@K", "MRR"],
        [[r["K"], r["P"], r["P_kw"], r["Hit"], r["MRR"]] for r in results["table_VI"]],
        first_width=6,
    )

    t7 = results["table_VII"]
    _print_table(
        "TABLE VII (left). Effect of the RRF constant k.",
        ["RRF k", "P_kw@4", "Hit@4"],
        [[r["k"], r["P_kw"], r["Hit"]] for r in t7["rrf_k"]],
        first_width=8,
    )
    _print_table(
        "TABLE VII (right). Effect of the candidate pool size per path.",
        ["Pool", "P_kw@4", "MRR"],
        [[r["pool"], r["P_kw"], r["MRR"]] for r in t7["pool"]],
        first_width=8,
    )

    t8 = results["table_VIII"]
    _print_table(
        "TABLE VIII (left). Effect of the final score weights (w_s, w_r, w_a).",
        ["(w_s, w_r, w_a)", "P_kw", "Hit"],
        [[str(tuple(r["weights"])), r["P_kw"], r["Hit"]] for r in t8["score_weights"]],
        first_width=22,
    )
    _print_table(
        "TABLE VIII (right). Fixed versus adaptive fusion weights (alpha, beta).",
        ["Fusion (alpha, beta)", "P_kw", "Hit"],
        [[r["label"], r["P_kw"], r["Hit"]] for r in t8["fusion"]],
        first_width=22,
    )

    latency = results.get("search_latency_seconds")
    if latency is not None:
        print(f"\nMean time of both search paths per query (cache build, pool "
              f"{results['meta']['cache_pool']}): {latency:.2f} s")


# Main


def run_experiments(cache: dict[str, Any]) -> dict[str, Any]:
    """Compute every table from *cache*."""
    entries = cache["queries"]
    searched = [e["search_seconds"] for e in entries if e.get("search_seconds") is not None]
    results: dict[str, Any] = {
        "meta": {
            "queries": len(entries),
            "cache_pool": cache["meta"].get("pool"),
            "cache_created": cache["meta"].get("created"),
            "defaults": {
                "pool": PROPOSED.pool, "top_k": PROPOSED.top_k, "rrf_k": PROPOSED.rrf_k,
                "weights": list(PROPOSED.weights), "fusion": "adaptive",
                "fusion_weights": {t: list(w) for t, w in _CLASSIFICATION_WEIGHTS.items()},
            },
            "classifications": {e["query"]: classification(e) for e in entries},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "table_III": table_iii(cache),
        "table_VI": table_vi(cache),
        "table_VII": table_vii(cache),
        "table_VIII": table_viii(cache),
        "search_latency_seconds": round(sum(searched) / len(searched), 3) if searched else None,
    }
    return results


def _load_or_build_cache(args: argparse.Namespace, test_queries: list[dict[str, Any]]) -> dict[str, Any]:
    if args.cache.exists() and not args.refresh:
        with args.cache.open("r", encoding="utf-8") as fh:
            cache = json.load(fh)
        check_cache(cache, test_queries, args.pool)
        logger.info("Loaded cache %s (%d queries)", args.cache, len(cache["queries"]))
        return cache

    from src.embeddings.embedding_service import EmbeddingService
    from src.embeddings.vectorstore import VectorStore
    from src.retrieval.ipc_bns import IpcBnsMapper

    store = VectorStore()
    if store.count() == 0:
        raise SystemExit("The vector store is empty. Build the index first (see README).")
    hybrid = HybridSearch(store.collection, EmbeddingService())
    cache = build_cache(
        test_queries, hybrid, IpcBnsMapper(), pool=args.pool,
        meta={"collection_count": store.count(), "queries_file": str(args.queries)},
    )
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    with args.cache.open("w", encoding="utf-8") as fh:
        json.dump(cache, fh, ensure_ascii=False)
    logger.info("Wrote cache %s", args.cache)
    return cache


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES_PATH)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--pool", type=int, default=DEFAULT_CACHE_POOL,
                        help="candidates cached per search path (default 40)")
    parser.add_argument("--refresh", action="store_true", help="re-run the search paths")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    logger.setLevel(logging.INFO)

    with args.queries.open("r", encoding="utf-8") as fh:
        test_queries = json.load(fh)

    cache = _load_or_build_cache(args, test_queries)

    results = run_experiments(cache)
    print_report(results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
