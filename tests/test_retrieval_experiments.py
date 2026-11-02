"""Tests for scripts/retrieval_experiments.py on a small hand-made cache.

No index or embedding model is needed: the cache below stands in for the
cached dense and exact-match results, and the expected values are worked
out by hand from eq. 3-6 of the report (k = 60, adaptive fusion weights,
(w_s, w_r, w_a) = (0.5, 0.25, 0.25)).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "retrieval_experiments.py"
_spec = importlib.util.spec_from_file_location("retrieval_experiments", _SCRIPT)
assert _spec is not None and _spec.loader is not None
rx = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = rx  # dataclasses look the module up while it loads
_spec.loader.exec_module(rx)

SC = {"doc_type": "judgement", "court": "Supreme Court of India", "date": ""}  # A = 0.61
NO_COURT = {"doc_type": "judgement", "court": "", "date": ""}  # A = 0.33
STATUTE = {"doc_type": "statute"}  # A = 1.0
CONSTITUTION = {"doc_type": "constitution"}  # A = 1.0

CHUNKS: dict[str, dict[str, Any]] = {
    "d1": {"text": "Article 21 protects life and liberty.", "metadata": SC},
    "d2": {"text": "Article 21  protects life and\nliberty.", "metadata": SC},  # copy of d1
    "d3": {"text": "Section 302. Punishment for murder.", "metadata": STATUTE},
    "d4": {"text": "An unrelated passage on taxation.", "metadata": NO_COURT},
    "d5": {"text": "The murder trial was held in 1990.", "metadata": SC},
    "e1": {"text": "Section 302 IPC prescribes the punishment.", "metadata": STATUTE},
    "e2": {"text": "Convicted under Section 302 by the trial court.", "metadata": NO_COURT},
    "n1": {"text": "The principles of natural justice include audi alteram partem.", "metadata": NO_COURT},
    "n2": {"text": "Article 14 guarantees equality before law.", "metadata": CONSTITUTION},
    "n3": {"text": "Natural justice requires a fair hearing.", "metadata": SC},
}

QUERIES = [
    {
        "query": "What is Section 302 IPC?",
        "category": "section_lookup",
        "expected_doc_types": ["statute"],
        "expected_keywords": ["Section 302", "murder"],
    },
    {"query": "", "category": "edge_case", "expected_doc_types": [], "expected_keywords": []},
    {
        "query": "Explain the principle of natural justice",
        "category": "conceptual",
        "expected_doc_types": ["judgement"],
        "expected_keywords": ["natural justice"],
    },
]

DENSE = {
    "What is Section 302 IPC?": [("d1", 0.80), ("d2", 0.79), ("d3", 0.70), ("d4", 0.60), ("d5", 0.55)],
    "Explain the principle of natural justice": [("n1", 0.90), ("n2", 0.80), ("n3", 0.70)],
}
EXACT = {
    "What is Section 302 IPC?": [("d3", 0.70), ("e1", 0.50), ("e2", 0.45)],
    "Explain the principle of natural justice": [],
}


def _hit(chunk_id: str, score: float) -> dict[str, Any]:
    return {"chunk_id": chunk_id, **CHUNKS[chunk_id], "semantic_score": score}


class FakeHybridSearch:
    """Stands in for ``HybridSearch.search_paths``; keyed by the original query."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def search_paths(self, query: str, top_k: int) -> dict[str, Any]:
        self.calls.append((query, top_k))
        original = next(q for q in DENSE if query.startswith(q))
        return {
            "classification": rx.HybridSearch.classify_query(query),
            "semantic_hits": [_hit(c, s) for c, s in DENSE[original]][:top_k],
            "keyword_hits": [_hit(c, s) for c, s in EXACT[original]][:top_k],
        }


@pytest.fixture()
def cache() -> dict[str, Any]:
    from src.retrieval.ipc_bns import IpcBnsMapper

    return rx.build_cache(QUERIES, FakeHybridSearch(), IpcBnsMapper(), pool=40)


def _ids(ranked: list[dict[str, Any]]) -> list[str]:
    return [c["chunk_id"] for c in ranked]


def _entry(cache: dict[str, Any], i: int) -> dict[str, Any]:
    return cache["queries"][i]


# Cache


class TestCache:

    def test_empty_query_skipped(self, cache: dict[str, Any]) -> None:
        assert [e["query"] for e in cache["queries"]] == [QUERIES[0]["query"], QUERIES[2]["query"]]
        assert [e["index"] for e in cache["queries"]] == [0, 2]

    def test_ipc_bns_expansion_used_for_retrieval(self, cache: dict[str, Any]) -> None:
        entry = _entry(cache, 0)
        assert entry["ipc_bns_matches"] == ["302"]
        assert "Section 103 BNS" in entry["retrieval_query"]
        assert entry["classification"] == "section_lookup"
        assert _entry(cache, 1)["retrieval_query"] == QUERIES[2]["query"]
        assert _entry(cache, 1)["classification"] == "conceptual"

    def test_each_path_searched_once_with_pool(self) -> None:
        from src.retrieval.ipc_bns import IpcBnsMapper

        fake = FakeHybridSearch()
        rx.build_cache(QUERIES, fake, IpcBnsMapper(), pool=40)
        assert len(fake.calls) == 2
        assert all(top_k == 40 for _, top_k in fake.calls)

    def test_chunks_stored_once(self, cache: dict[str, Any]) -> None:
        assert set(cache["chunks"]) == set(CHUNKS)
        assert _entry(cache, 0)["dense"][0] == ["d1", 0.80]

    def test_check_cache(self, cache: dict[str, Any]) -> None:
        rx.check_cache(cache, QUERIES, pool=40)
        with pytest.raises(ValueError):
            rx.check_cache(cache, QUERIES, pool=50)
        with pytest.raises(ValueError):
            rx.check_cache(cache, QUERIES[:1], pool=20)

    def test_pool_truncates_each_path(self, cache: dict[str, Any]) -> None:
        cfg = rx.Config(mode="dense", pool=2, dedupe=False)
        assert _ids(rx.candidate_pool(_entry(cache, 0), cache["chunks"], cfg)) == ["d1", "d2"]


# Table III rows on the first query (hand-computed)


class TestRankings:

    def test_dense_only_dedupes(self, cache: dict[str, Any]) -> None:
        cfg = rx.Config(mode="dense")
        assert _ids(rx.rank_query(_entry(cache, 0), cache["chunks"], cfg)) == ["d1", "d3", "d4", "d5"]

    def test_dense_without_dedupe(self, cache: dict[str, Any]) -> None:
        cfg = rx.Config(mode="dense", dedupe=False)
        assert _ids(rx.rank_query(_entry(cache, 0), cache["chunks"], cfg)) == ["d1", "d2", "d3", "d4"]

    def test_exact_only(self, cache: dict[str, Any]) -> None:
        cfg = rx.Config(mode="exact")
        assert _ids(rx.rank_query(_entry(cache, 0), cache["chunks"], cfg)) == ["d3", "e1", "e2"]
        assert rx.rank_query(_entry(cache, 1), cache["chunks"], cfg) == []

    def test_adaptive_rrf_uses_section_weights(self, cache: dict[str, Any]) -> None:
        # (0.3, 0.7): d3 is in both lists; e1, e2 rank 2 and 3 in the exact list
        ranked = rx.rank_query(_entry(cache, 0), cache["chunks"], rx.Config(mode="fused"))
        assert _ids(ranked) == ["d3", "e1", "e2", "d1"]
        assert ranked[0]["rrf_score"] == pytest.approx(0.3 / 63 + 0.7 / 61)

    def test_fixed_rrf(self, cache: dict[str, Any]) -> None:
        # (0.5, 0.5): d1 (dense rank 1) now beats e1 (exact rank 2); d2 ties
        # with e1 and comes first, then is removed as a copy of d1
        cfg = rx.Config(mode="fused", fusion=(0.5, 0.5))
        assert _ids(rx.rank_query(_entry(cache, 0), cache["chunks"], cfg)) == ["d3", "d1", "e1", "e2"]

    def test_proposed_final_score(self, cache: dict[str, Any]) -> None:
        ranked = rx.rank_query(_entry(cache, 0), cache["chunks"], rx.PROPOSED)
        assert _ids(ranked) == ["d3", "e1", "d1", "d5"]
        # S = 0.5 * 0.70 + 0.25 * min((0.3/63 + 0.7/61) * 61, 1) + 0.25 * 1.0
        expected = 0.5 * 0.70 + 0.25 * ((0.3 / 63 + 0.7 / 61) * 61) + 0.25 * 1.0
        assert ranked[0]["final_score"] == pytest.approx(expected, abs=1e-6)
        # judgement, Supreme Court, undated, citations unknown: 0.4*1 + 0.3*0.2 + 0.3*0.5
        assert ranked[2]["authority_score"] == pytest.approx(0.61)

    def test_no_authority_row(self, cache: dict[str, Any]) -> None:
        assert rx.NO_AUTHORITY_WEIGHTS == (0.5, 0.5, 0.0)
        cfg = rx.Config(weights=rx.NO_AUTHORITY_WEIGHTS)
        assert _ids(rx.rank_query(_entry(cache, 0), cache["chunks"], cfg)) == ["d3", "e1", "e2", "d1"]
        assert _ids(rx.rank_query(_entry(cache, 1), cache["chunks"], cfg)) == ["n1", "n2", "n3"]

    def test_authority_promotes_constitution(self, cache: dict[str, Any]) -> None:
        assert _ids(rx.rank_query(_entry(cache, 1), cache["chunks"], rx.PROPOSED)) == ["n2", "n1", "n3"]

    def test_proportional_weights(self) -> None:
        assert rx.PROPORTIONAL_NO_AUTHORITY_WEIGHTS == pytest.approx((2 / 3, 1 / 3, 0.0))

    def test_rrf_k_changes_rescaling(self, cache: dict[str, Any]) -> None:
        cfg = rx.Config(rrf_k=1)
        ranked = rx.rank_query(_entry(cache, 0), cache["chunks"], cfg)
        scored = rx.score_candidates(rx.candidate_pool(_entry(cache, 0), cache["chunks"], cfg), cfg)
        # e1 is rank 2 of the exact list: R = 0.7 / (1 + 2), rescaled by (k + 1) = 2
        e1 = next(c for c in scored if c["chunk_id"] == "e1")
        assert e1["final_score"] == pytest.approx(0.5 * 0.5 + 0.25 * (0.7 / 3 * 2) + 0.25, abs=1e-6)
        assert ranked[0]["chunk_id"] == "d3"


# Metrics and tables


class TestTables:

    def test_table_iii_values(self, cache: dict[str, Any]) -> None:
        rows = {r["id"]: r for r in rx.table_iii(cache)["rows"]}
        assert [r for r in rows] == ["a", "b", "c", "d", "e", "f"]
        # (a): Q1 [d1, d3, d4, d5] -> P_kw 2/4, RR 1/2; Q2 [n1, n2, n3] -> P_kw 2/4, RR 1
        assert rows["a"]["P_kw"] == pytest.approx(0.5)
        assert rows["a"]["MRR"] == pytest.approx(0.75)
        assert rows["a"]["Hit"] == pytest.approx(1.0)
        # (b): Q1 three exact hits out of K = 4 slots; Q2 nothing
        assert rows["b"]["P"] == pytest.approx(0.375)
        assert rows["b"]["P_kw"] == pytest.approx(0.375)
        assert rows["b"]["Hit"] == pytest.approx(0.5)
        # (e) vs (f): the authority prior moves n2 above n1 for Q2
        assert rows["e"]["MRR"] == pytest.approx(1.0)
        assert rows["f"]["MRR"] == pytest.approx(0.75)
        assert rows["f"]["P_kw"] == pytest.approx(0.625)

    def test_table_iii_extras(self, cache: dict[str, Any]) -> None:
        t3 = rx.table_iii(cache)
        extra = {r["id"]: r for r in t3["extra_rows"]}
        assert extra["a0"]["P_kw"] == pytest.approx((0.25 + 0.5) / 2)
        # Q1 has one copied slot (d2) among its 4 dense slots, Q2 none among 3
        assert t3["dense_duplicate_slot_share"] == pytest.approx(1 / 7, abs=1e-4)
        assert t3["per_category_P_kw"]["a"] == {"section_lookup": 0.5, "conceptual": 0.5}
        assert len(t3["per_query"]["f"]) == 2

    def test_deployed_setting_consistent_across_tables(self, cache: dict[str, Any]) -> None:
        f = next(r for r in rx.table_iii(cache)["rows"] if r["id"] == "f")
        t6 = next(r for r in rx.table_vi(cache) if r["K"] == 4)
        t7 = rx.table_vii(cache)
        t8 = rx.table_viii(cache)
        same = [
            t6,
            next(r for r in t7["rrf_k"] if r["k"] == 60),
            next(r for r in t7["pool"] if r["pool"] == 20),
            next(r for r in t8["score_weights"] if r["weights"] == [0.5, 0.25, 0.25]),
            next(r for r in t8["fusion"] if r["fusion"] == "adaptive"),
        ]
        for row in same:
            assert (row["P_kw"], row["Hit"], row["MRR"]) == (f["P_kw"], f["Hit"], f["MRR"])

    def test_table_vi_k(self, cache: dict[str, Any]) -> None:
        rows = rx.table_vi(cache)
        assert [r["K"] for r in rows] == [2, 4, 6, 8, 10]
        k2 = rows[0]
        # Q1 top 2 [d3, e1] both relevant; Q2 top 2 [n2, n1] one relevant
        assert k2["P_kw"] == pytest.approx(0.75)

    def test_table_vii_and_viii_grids(self, cache: dict[str, Any]) -> None:
        t7 = rx.table_vii(cache)
        assert [r["k"] for r in t7["rrf_k"]] == [1, 10, 30, 60, 100]
        assert [r["pool"] for r in t7["pool"]] == [5, 10, 20, 30, 40]
        t8 = rx.table_viii(cache)
        assert len(t8["score_weights"]) == 7
        assert [r["label"] for r in t8["fusion"]] == [
            "fixed (1.0, 0.0)", "fixed (0.8, 0.2)", "fixed (0.5, 0.5)",
            "fixed (0.2, 0.8)", "fixed (0.0, 1.0)", "adaptive",
        ]

    def test_pool_above_cache_skipped(self, cache: dict[str, Any]) -> None:
        cache["meta"]["pool"] = 20
        assert [r["pool"] for r in rx.table_vii(cache)["pool"]] == [5, 10, 20]


# Nyaya-Rerank rows


def _fake_scores(query: str, texts: list[str]) -> list[float]:
    return [1.0 if ("natural justice" in t.lower() or "section 302" in t.lower()) else 0.0 for t in texts]


class TestRerankerRows:

    def test_lambda_zero_equals_proposed(self, cache: dict[str, Any]) -> None:
        out = rx.reranker_rows(cache, _fake_scores, weight=0.0)
        g = out["rows"][0]
        f = next(r for r in rx.table_iii(cache)["rows"] if r["id"] == "f")
        assert (g["P_kw"], g["MRR"]) == (f["P_kw"], f["MRR"])

    def test_reranker_alone(self, cache: dict[str, Any]) -> None:
        out = rx.reranker_rows(cache, _fake_scores, weight=1.0)
        g, h = out["rows"]
        assert h["id"] == "h" and g["id"] == "g"
        # Q1: [d3, e1, e2, d1] -> 3/4; Q2: [n1, n3, n2] -> 2/4, RR 1
        assert h["P_kw"] == pytest.approx(0.625)
        assert h["MRR"] == pytest.approx(1.0)
        assert g["MRR"] == pytest.approx(1.0)
        assert g["mean_added_latency_seconds"] >= 0.0
        assert h["mean_added_latency_seconds"] >= g["mean_added_latency_seconds"]

    def test_candidates_limit(self, cache: dict[str, Any]) -> None:
        out = rx.reranker_rows(cache, _fake_scores, weight=1.0, candidates=2)
        per_query = out["per_category_P_kw"]["g"]
        # Q1 keeps only [d3, e1] (2 of 4 slots); Q2 keeps [n2, n1] -> n1 first
        assert per_query["section_lookup"] == pytest.approx(0.5)
        assert per_query["conceptual"] == pytest.approx(0.25)

    def test_query_field(self, cache: dict[str, Any]) -> None:
        seen: list[str] = []

        def score(query: str, texts: list[str]) -> list[float]:
            seen.append(query)
            return [0.5] * len(texts)

        rx.reranker_rows(cache, score, weight=0.5, query_field="query")
        assert QUERIES[0]["query"] in seen
        assert all("Section 103 BNS" not in q for q in seen)


# End to end through main()


def test_main_writes_all_tables(tmp_path: Path, cache: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    queries_path = tmp_path / "queries.json"
    cache_path = tmp_path / "cache.json"
    output_path = tmp_path / "out.json"
    queries_path.write_text(json.dumps(QUERIES))
    cache_path.write_text(json.dumps(cache))

    rx.main(["--queries", str(queries_path), "--cache", str(cache_path), "--output", str(output_path)])

    printed = capsys.readouterr().out
    for title in ("TABLE III", "TABLE VI", "TABLE VII", "TABLE VIII", "per category"):
        assert title in printed
    results = json.loads(output_path.read_text())
    assert results["meta"]["queries"] == 2
    assert {"table_III", "table_VI", "table_VII", "table_VIII"} <= set(results)
    assert "reranker" not in results
