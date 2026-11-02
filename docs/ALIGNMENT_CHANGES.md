# Report and code alignment

This table checks each formula, constant and metric in the project report (`Citation Grounded Indian Legal Research Assistant`, Team 12) against the code. Line numbers refer to the code after this change.

Status values:
- **matched**: the code already did what the report says.
- **fixed in code**: the code was changed in this pass.
- **report should change**: the code is correct, the recorded numbers come from it, and the report text is wrong or incomplete.

## Claim-by-claim check

| Report claim | Code location (file:line) | Status | Note |
|---|---|---|---|
| **III-C Chunking** | | | |
| T̂(c) = ⌊1.3·\|W(c)\|⌋ (eq. 1) | `src/chunking/legal_chunker.py:23,31` | matched | `int()` of a non-negative number is the floor. |
| Target / max / overlap = 450 / 800 / 50 tokens | `config/settings.py:27-29` | matched | |
| Statutes: one section per chunk, long sections split at sub-section markers | `src/chunking/legal_chunker.py:200-270` (split at `:251`) | report should change | The split happens once, at `(1)`, `(a)`, `(i)` markers, and is followed by a merge. A section with no markers stays whole and can be longer than 800 tokens, so "split again and again" overstates it. The code was left alone because the index (112,555 chunks) comes from it. |
| Constitution: one article per chunk, explanations and provisos kept | `src/chunking/legal_chunker.py:275-350` | matched | Articles are split only at `Article N` line starts, so explanations stay attached. An article over 800 tokens is split at sentence boundaries. |
| Judgments: split at numbered paragraphs, short ones merged, case name prefixed | `src/chunking/legal_chunker.py:115-186` (merge `:147-148`, prefix `:136`) | report should change (wording) | Paragraphs are merged until they reach target/2 = 225 tokens. Pieces over 800 tokens are split at sentences with a 50-token overlap. Nothing is "packed to the target length" of 450. |
| Deterministic ids `doc_id` + index | `src/chunking/chunk_models.py:53` | matched | The format is `f"{doc_id}::{index}"` (the PDF text lost the `::`). |
| **III-D Embeddings** | | | |
| all-MiniLM-L6-v2, 384-d, batch 64 | `config/settings.py:20-21`, `src/embeddings/embedding_service.py:36-43` | matched | |
| ChromaDB HNSW, cosine space | `src/embeddings/vectorstore.py:27` | matched | |
| s(q,c) = 1 − d_cos (eq. 2) | `src/retrieval/hybrid_search.py:165` | report should change (minor) | The code uses `max(0, 1 − d)`, so negative cosine is clipped to 0. This never matters for top-40 MiniLM hits. |
| **III-E IPC/BNS** | | | |
| 63 rows; IPC 376→BNS 64, IPC 420→BNS 318 | `config/ipc_bns_mapping.json` | matched | 63 rows. IPC 420 maps to `318(4)`, a sub-section, which the report allows for. |
| Regexes in both orders ("Section 302 IPC", "IPC 302", "s. 302 of Indian Penal Code") | `src/retrieval/ipc_bns.py:26-48` | matched | |
| Offence names detected | `src/retrieval/ipc_bns.py:139-159` | report should change (precision) | Offence-name (topic) matching runs only when the query names both IPC and BNS and gives no section number. It returns up to 3 rows. |
| Matched rows expand the query and become an extra, last source | `src/retrieval/ipc_bns.py:164-212`, `src/pipeline.py:139-160` | matched | |
| **III-F Classification and hybrid retrieval** | | | |
| Section/Article/Act/IPC/CrPC/CPC/BNS gives section lookup; AIR/SCC or "X v Y" gives case lookup; both gives mixed; neither gives conceptual | `src/retrieval/hybrid_search.py:35-51,250-268` | matched | The query is classified after IPC/BNS expansion. |
| Dense path returns 20 | `config/settings.py:32`, `hybrid_search.py:272-283` | matched | |
| Canonical identifiers ("section 304a" → "Section 304A") | `hybrid_search.py:101-103` | matched | |
| AIR / SCC citations | `hybrid_search.py:105-111` | matched | |
| Distinctive party names; skip "Union of India", "State of …" | `hybrid_search.py:14-32,67-89` | matched | |
| Dotted initials ("MC Mehta" → "M.C. Mehta") | `hybrid_search.py:84-88` | matched | |
| Up to 100 chunks per identifier, ordered by query similarity | `hybrid_search.py:56,286-325` | matched | Exact hits come from a filtered query with the same query embedding, so they carry s(q,c). |
| Word-boundary check ("Article 21" not in "Article 210") | `hybrid_search.py:142-144` | matched | |
| 20 nearest kept | `hybrid_search.py:324-325` | matched | |
| (not in report) bare Act abbreviation as an identifier | `hybrid_search.py:113-118` | report should change (minor) | When a query has no section, citation or party name, `IPC`, `CrPC`, `CPC` or `BNS` is searched as the identifier. |
| **Fusion** | | | |
| RRF_L(c) = 1/(k + r), k = 60 (eq. 3) | `hybrid_search.py:170-183`, `config/settings.py:34` | matched | |
| R(c) = α_τ RRF_sem + β_τ RRF_kw, (0.3,0.7)/(0.2,0.8)/(0.8,0.2)/(0.5,0.5) | `hybrid_search.py:59-64,186-224` | fixed in code (refactor) | Fusion now lives in `fuse_hits(sem, kw, α, β, k)`, which `search()` calls. Behaviour is unchanged. The parameters exist so the sweeps can reuse the deployed code. |
| Duplicates with identical normalised text removed after fusion | `src/retrieval/retrieval_pipeline.py:128,171` | matched | |
| **III-G Authority-aware re-ranking** | | | |
| A = 1 for statute/Constitution (eq. 5) | `src/retrieval/reranker.py:123-124` | matched | `legislation` is also treated as statute. |
| w_court = 1.0 SC / 0.7 HC / 0.4 tribunal / 0.3 unknown | `reranker.py:31-35,126-132` | fixed in code | The code had a fifth class: a judgment whose court string contained "act", "code" or "statute" scored 1.0. Tribunal abbreviations were matched as substrings, so "Satara District Court" counted as a tribunal. Now only SC/HC/tribunal (whole-word abbreviations) are recognised and everything else is 0.3. The default index is unaffected because its courts are "Supreme Court of India" or empty. |
| w_rec = 1.0/0.8/0.6/0.4 for ≤2/5/10/20 years, else or undated 0.2 | `config/settings.py:50-56`, `reranker.py:66-74` | matched | |
| w_cit = min(n/100, 1), default 0.5 | `reranker.py:136-145` | matched / report should change | `citation_count` is never stored in ChromaDB (`src/embeddings/vectorstore.py:14`), so w_cit is 0.5 for every judgment, not only for "many". The same holds for undated chunks: A = 0.61 for an undated SC chunk and 0.33 for one with no court. |
| A = 0.4 w_court + 0.3 w_rec + 0.3 w_cit | `reranker.py:147` | matched | |
| R̃ = min(R·(k+1), 1) | `reranker.py:184-187,193` | fixed in code (parameter) | `rerank(..., rrf_k=)` rescales with the k used in fusion, which Table VII needs. The default is unchanged (k = 60). |
| S = 0.50 s + 0.25 R̃ + 0.25 A (eq. 6) | `config/settings.py:37-39`, `reranker.py:26-30,197-201` | matched (+ `weights=` parameter) | `rerank(..., weights=(w_s,w_r,w_a))` was added for Table VIII. |
| Exact-only hits scored against the query embedding | `hybrid_search.py:286-325` | matched | |
| Top K = 4 | `config/settings.py:33` | matched | |
| **III-H Generation** | | | |
| Prompt rules (i)–(vi) | `config/prompts.py:8-21` | matched | (i) is rule 1, (ii) rule 2, (iii) rule 3, (iv) rule 5, (v) rule 6, (vi) rule 7. Rules 4 (give section and Act) and 8 (no reasoning text) are extra. |
| Temperature 0.1, max 1,024 tokens, retries, 40 req/min | `config/settings.py:63-65`, `src/generation/llm_client.py:134,170-175,194-195` | matched | 3 attempts with exponential backoff. `top_p = 0.95` is also set but not mentioned in the report. |
| **III-I Confidence** | | | |
| C = 0.40Q + 0.25G + 0.20H + 0.15V | `config/settings.py:68-73`, `src/confidence/scorer.py:105` | matched | |
| Q = clip((S̄ − 0.3)/0.6, 0, 1) | `scorer.py:29-30,126-143` | matched | |
| G = 0/0.3/0.6/1.0 | `scorer.py:146-157` | matched | The IPC/BNS table counts as a cited source. |
| H = mean authority of cited chunks | `scorer.py:167-175` | report should change | When nothing is cited, H is the mean authority of all retrieved chunks. The recorded confidences (Tables IV and V, e.g. edge cases 0.35) depend on this fallback. |
| V = fraction of retrieved chunks cited | `scorer.py:178-182` | matched | |
| High ≥ 0.8, Medium ≥ 0.6, Low ≥ 0.4, else Very Low | `scorer.py:21-26` | matched | |
| **IV-B Metrics** | | | |
| P@K (eq. 8), type in T or keyword | `src/evaluation/retrieval_metrics.py:54-65` | fixed in code | The new module divides by K, as eq. 8 does. `scripts/evaluate.py` previously divided by the number retrieved. Every recorded run retrieved 4 for all 49 non-empty queries, so the values are the same. |
| P_kw@K, Hit@K, MRR | `retrieval_metrics.py:68-119` | fixed in code | These did not exist. They are now per-query fields (`keyword_precision`, `hit`, `reciprocal_rank`) and summary fields in `scripts/evaluate.py:86-118`. MRR is cut off at K, which matches Table VI, where MRR changes with K. |
| Citation validity (1 ≤ N ≤ n; no tag gives 0) | `src/evaluation/citation_metrics.py:39-49` | matched | |
| Cited claim coverage (≥ 25 chars, legal anchor) | `citation_metrics.py:52-86` | matched | |
| Citation support (same section/article and concept, ≥ 35% or ≥ 5 content words) | `citation_metrics.py:119-167` | matched | |
| Hallucination rate ("X v Y" or AIR/SCC/SCR not in sources) | `citation_metrics.py:189-245` | report should change | A whitelist of 16 well-known case names (`_KNOWN_CASE_FRAGMENTS`) is never flagged. The citation pattern only matches `AIR/SCC/SCR <year> <court> <page>`, so "(2017) 10 SCC 1" is missed. |
| Response rate | `citation_metrics.py:248-258` | matched | |
| Average confidence | `src/evaluation/summary.py:58-79` | matched | |
| Mean latency (Table IV) | `summary.py:42-55`, `scripts/evaluate.py` (`elapsed_seconds`) | fixed in code | Mean latency is now in the summary and per category. It is the mean over the non-empty queries, which reproduces 25.0 / 15.4 / 17.5 s exactly. |
| Table V (validity, support, resp., ret. P, conf.) | `src/evaluation/summary.py:82-88`, `scripts/evaluate.py:216-244` | fixed in code | The per-category table had no confidence column. It now prints Table V's columns plus P_kw, Hit, MRR and latency, and is written to the JSON as `per_category`. |
| `scripts/evaluate.py` runs | `scripts/evaluate.py:1` | fixed in code | Line 1 started with a stray backtick, a `SyntaxError`, so the script could not run at all. |
| Tables IV and V from the recorded runs | `scripts/evaluate.py --summarise FILE` | fixed in code (new) | Re-summarising `eval_results.json`, `eval_results_nemotron.json` and `eval_results_deepseek.json` reproduces Table IV columns A, B and C and Table V exactly. |
| **IV-C/D Tables III, VI, VII, VIII** | `scripts/retrieval_experiments.py` (rows `:141-157`, tables `:373-428`) | fixed in code (new) | Both paths are cached once per query (pool 40). Every configuration then reuses the deployed `classify_query`, `fuse_hits`, `_drop_duplicate_texts`, `AuthorityReranker.rerank` and `IpcBnsMapper`. Not run yet: it needs the index. |

## Report should change

1. **Confidence H.** When the answer cites nothing, H falls back to the mean authority of all retrieved chunks. The recorded confidences depend on this.
2. **Hallucination detector.** It exempts 16 well-known case names and only recognises citations of the form `AIR/SCC/SCR <year> <court> <page>`. Bracketed SCC citations are not checked.
3. **Chunking wording.**
   - Judgment paragraphs are merged up to 225 tokens (target/2), not packed to 450.
   - Statute sections are split once at sub-section markers, so a section without markers can exceed 800 tokens.
4. **Authority defaults.** `citation_count` is never indexed and the Hugging Face judgments are undated. Every judgment therefore gets w_cit = 0.5 and w_rec = 0.2, so A is 0.61 (Supreme Court) or 0.33 (no court).
5. **Exact path.** A bare Act abbreviation is searched when the query has no other identifier. Offence-name matching for IPC/BNS only applies to queries that name both codes and give no section number.
6. **Table III row (e).** The report does not say how (w_s, w_r) are renormalised when w_a = 0. Its numbers (0.842 = Table VIII (0.5, 0.5, 0)) imply that w_a's 0.25 moves to the RRF term, i.e. (0.5, 0.5, 0). The script prints that as row (e) and the proportional (2/3, 1/3, 0) as an extra row (e′).
7. **Table VIII text.** The text says adaptive fusion "matches the best of the fixed setting". But adaptive P_kw (0.811) is below fixed (0.0, 1.0) at 0.827 and fixed (0.2, 0.8) at 0.821; only Hit@4 is equal (0.878).
8. **Table IV, Run C label.** Run C is labelled "(Nemotron, final)". Its numbers (0.70 / 0.42 / 0.50 / 0.04 / 0.68 / 0.92 / 0.61 / 17.5 s) are in `data/evaluation/eval_results_deepseek.json`. The README says this run is in `eval_results_latest.json`, which does not exist. The result files do not record the model, so the label cannot be checked. `evaluate.py` now writes `llm_model` into the summary.
9. **Table IV, retrieval precision and latency.**
   - Run A's retrieval precision (0.79) is P@6, because Run A used K = 6. It is not comparable with Runs B and C (P@4).
   - Mean latency is averaged over the 49 non-empty queries.
10. **Run A per-query fields.** `data/evaluation/eval_results.json` holds per-query `P@4`, `Pkw@4` and `Hit@4` values that are not multiples of 0.25 or 0/1, and no code in the repository produces them. They are not valid P@4 or Hit@4 values and should not be cited.
11. **Environment.** The report says Python 3.12 and PyTorch 2.14. The current venv has Python 3.13.5, PyTorch 2.15.0, ChromaDB 1.5.9 and sentence-transformers 6.2.0, and `requirements.txt` has only lower bounds.
12. **Test count.** "250 unit and integration tests" has changed (this pass and the Nyaya-Gen work added tests); recount with `pytest --collect-only -q` before quoting a number.
13. **References.** The text cites [11] to [21] for works that are not in the reference list, which has 14 entries:
    - [11] Izacard & Grave in the text is RAGAS in the list.
    - [12] Liu et al. in the text is LegalBench in the list.
    - [13] Malik in the text is DeepSeek in the list.
    - [14] Kalamkar in the text is the BNS in the list.
    - [15] to [21] are missing from the list.
14. **Minor omissions.**
    - s(q,c) is clipped at 0.
    - Generation also sets top_p = 0.95.
    - Grouped tags such as `[SOURCE_2, SOURCE_4]` are split before validation.
    - The evaluation also reports a blended "citation quality" (0.3 validity + 0.3 coverage + 0.4 support).
    - The 76.4% / 19.9% / 3.7% shares are of the 114,024 chunks before merging duplicate ids, not of 112,555.

## Not verified here

These need the ChromaDB index, which was still being built:
- Table I counts.
- Tables III and VI–VIII: `python scripts/retrieval_experiments.py`.
- The dense-duplicate share (19.4%).
- Whether the dense top-20 inside a top-40 query equals a top-20 query under HNSW.
