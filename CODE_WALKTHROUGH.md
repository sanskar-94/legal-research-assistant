# Code Walkthrough — how to show the project to the teacher

Open the files in this order. Each step lists **what to open**, **what to point at**, and **what to say**. Click a link to jump to the exact line.

---

## Part 0 — Start with a live demo (1 min)

```bash
source .venv/bin/activate
LLM_BACKEND=local python scripts/ask.py "What is Section 302 IPC?"
```

> **Say:** "This runs fully on my laptop with our own trained model. It finds the IPC/BNS match, retrieves 4 sources, writes an answer with [SOURCE_N] citations, and gives a confidence level."

---

## Part 1 — Project layout (1 min)

```
config/     settings + prompt + IPC→BNS table     ← all numbers live here
src/        the system itself (one folder per stage)
  ingestion/   read & clean documents
  chunking/    split into legal chunks
  embeddings/  MiniLM vectors + ChromaDB
  retrieval/   IPC/BNS lookup, hybrid search, re-ranking
  generation/  prompt, API model, OUR MODEL, citation parser
  confidence/  confidence score
  evaluation/  metrics
  pipeline.py  connects every stage
scripts/    things you run: build index, ask, train, evaluate
tests/      300 automated tests
report/     final report PDF
```

> **Say:** "Each folder in `src/` is one stage of the pipeline, in the same order as the architecture diagram in the report."

---

## Part 2 — The heart: one question end to end (3 min)

Open [src/pipeline.py](src/pipeline.py) → `query()` at [src/pipeline.py:122](src/pipeline.py:122). Walk down it:

| Line | Step | Say |
|---|---|---|
| [140](src/pipeline.py:140) | `find()` | "Detect IPC/BNS sections in the question." |
| [142](src/pipeline.py:142) | `expand_query()` | "Add the matching BNS section, so old and new law are both searched." |
| [147](src/pipeline.py:147) | `retrieve()` | "Hybrid search + re-ranking → top 4 passages." |
| [182](src/pipeline.py:182) | `build_prompt()` | "Put the passages in as numbered [SOURCE_1..4]." |
| [184](src/pipeline.py:184) | `generate()` | "The answer writer — API model **or our Nyaya-Gen**." |
| [190](src/pipeline.py:190) | `parse_response()` | "Check every citation is real." |
| [194](src/pipeline.py:194) | `score()` | "Compute the confidence score." |

Then show the **model switch** in `from_settings()`: [src/pipeline.py:94](src/pipeline.py:94).

> **Say:** "If `LLM_BACKEND` is `local`, it loads our trained model; otherwise the cloud API. Nothing else in the pipeline changes — so any difference in results comes only from the writer."

---

## Part 3 — Each stage in detail (5 min, show only what is asked)

### 3.1 Settings — [config/settings.py](config/settings.py)
- K = 4, RRF k = 60: [config/settings.py:32](config/settings.py:32)
- Final-score weights 0.50 / 0.25 / 0.25: [config/settings.py:37](config/settings.py:37)
- Court authority weights: [config/settings.py:42](config/settings.py:42)
- Confidence weights: [config/settings.py:68](config/settings.py:68)
- **Our model's settings**: [config/settings.py:85](config/settings.py:85)

### 3.2 Offline index (built once)
- Structure-aware chunking: `chunk_judgement` [src/chunking/legal_chunker.py:127](src/chunking/legal_chunker.py:127), `chunk_statute` [:200](src/chunking/legal_chunker.py:200), `chunk_constitution` [:288](src/chunking/legal_chunker.py:288)
  > "Judgments are split by paragraph, statutes by section, the Constitution by article — so one chunk is one legal unit."
- Embeddings (MiniLM-L6-v2): [src/embeddings/embedding_service.py:13](src/embeddings/embedding_service.py:13)
- ChromaDB store (112,555 chunks): [src/embeddings/vectorstore.py:55](src/embeddings/vectorstore.py:55)
- Run by: `scripts/data/prepare_hf_data.py` → `scripts/data/chunk_documents.py` → `scripts/data/build_vectorstore.py`

### 3.3 IPC → BNS — [src/retrieval/ipc_bns.py](src/retrieval/ipc_bns.py)
- Find sections in the question: `find()` [src/retrieval/ipc_bns.py:101](src/retrieval/ipc_bns.py:101)
- Expand the query: `expand_query()` [:164](src/retrieval/ipc_bns.py:164)
- Table as a citable source: `as_source_chunk()` [:177](src/retrieval/ipc_bns.py:177)

### 3.4 Hybrid search — [src/retrieval/hybrid_search.py](src/retrieval/hybrid_search.py)
- Query type → fusion weights: `classify_query()` [:250](src/retrieval/hybrid_search.py:250)
- Dense search (meaning): `_semantic_search()` [:272](src/retrieval/hybrid_search.py:272)
- Exact-match search (identifiers): `_keyword_search()` [:286](src/retrieval/hybrid_search.py:286)
- Weighted RRF fusion: `fuse_hits()` [:186](src/retrieval/hybrid_search.py:186)
  > "Dense search understands meaning but misses '498A'; exact match finds it. RRF combines the two rank lists."

### 3.5 Authority re-ranking — [src/retrieval/reranker.py](src/retrieval/reranker.py)
- Authority score (court + recency): [src/retrieval/reranker.py:108](src/retrieval/reranker.py:108)
- Final score S = 0.5·sim + 0.25·RRF + 0.25·authority: [src/retrieval/reranker.py:197](src/retrieval/reranker.py:197)
- Whole retrieval chain: [src/retrieval/retrieval_pipeline.py:51](src/retrieval/retrieval_pipeline.py:51)

### 3.6 Prompt and citation check
- The rules the model must follow: [config/prompts.py:3](config/prompts.py:3)
- Numbered sources: `build_prompt()` [src/generation/prompt_builder.py:103](src/generation/prompt_builder.py:103)
- Citation parsing/validation: `parse_response()` [src/generation/response_parser.py:72](src/generation/response_parser.py:72)
- Cloud API writer (old path): [src/generation/llm_client.py:99](src/generation/llm_client.py:99)

### 3.7 Confidence — [src/confidence/scorer.py:75](src/confidence/scorer.py:75)
> "C = 0.40·retrieval quality + 0.25·source agreement + 0.20·authority + 0.15·citation validity → High/Medium/Low."

---

## Part 4 — OUR MODEL: Nyaya-Gen (main part, 5 min)

> **Opening line:** "We replaced the 550B cloud model with SmolLM2-135M, a small pretrained model that we trained further on our own legal data, so the system runs offline. Three files: data, training, running."

### Step 1 — Training data: [scripts/nyaya_gen/prepare_generator_data.py](scripts/nyaya_gen/prepare_generator_data.py)
| Show | Say |
|---|---|
| Docstring, top of file | "Three kinds of examples: Q&A, IPC/BNS mapping, refusal." |
| `build()` [:111](scripts/nyaya_gen/prepare_generator_data.py:111) | "Builds every example in the exact prompt format the pipeline uses." |
| `make()` [:150](scripts/nyaya_gen/prepare_generator_data.py:150) | "The right passage is hidden at a random position k among 2–4 random distractors; the target cites [SOURCE_k]." |
| Leakage filter [:165](scripts/nyaya_gen/prepare_generator_data.py:165) (`_is_leak` at [:98](scripts/nyaya_gen/prepare_generator_data.py:98)) | "Removes 572 questions similar to our 50 test questions — no cheating." |
| Refusals [:175](scripts/nyaya_gen/prepare_generator_data.py:175) | "~7% have no right source → target is the refusal sentence, so it learns not to invent." |
| Split by record [:203](scripts/nyaya_gen/prepare_generator_data.py:203) | "Train 94% / dev 3% / test 3%." |

Output: `data/generator/train.jsonl` (open one line to show a real example).

### Step 2 — Training: [scripts/nyaya_gen/train_generator.py](scripts/nyaya_gen/train_generator.py)
| Show | Say |
|---|---|
| Base model [:36](scripts/nyaya_gen/train_generator.py:36) | "We start from SmolLM2-135M-Instruct, 134.5M parameters." |
| `encode()` [:45](scripts/nyaya_gen/train_generator.py:45), mask at [:55](scripts/nyaya_gen/train_generator.py:55) | "Prompt tokens get label −100, so the loss is only on the answer — Equation 9 in the report." |
| Gradient checkpointing [:112](scripts/nyaya_gen/train_generator.py:112) | "Saves memory so it fits in 8 GB." |
| AdamW [:129](scripts/nyaya_gen/train_generator.py:129), `lr_at()` [:59](scripts/nyaya_gen/train_generator.py:59) | "lr 1e-4, warm-up then cosine decay." |
| Accumulation [:145](scripts/nyaya_gen/train_generator.py:145)–[:158](scripts/nyaya_gen/train_generator.py:158) | "Gradients from 8 examples are added before each update." |
| Save best [:175](scripts/nyaya_gen/train_generator.py:175) | "Keep the checkpoint with the lowest dev loss — update 50, dev loss 0.833 → 0.033." |

Run with `scripts/nyaya_gen/run_training.sh`; result log: `data/generator/training_log.json`; model: `models/nyaya-gen/`.

### Step 3 — Using it: [src/generation/local_generator.py](src/generation/local_generator.py)
- `LocalGenerator` loads the model on the M2 GPU: [src/generation/local_generator.py:42](src/generation/local_generator.py:42)
- `generate()` — chat template, greedy decoding, max 256 tokens: [:61](src/generation/local_generator.py:61)
- `shorten_sources()` — cuts passages to 90 words like in training: [:31](src/generation/local_generator.py:31)
- Switched on by `LLM_BACKEND=local` → [src/pipeline.py:94](src/pipeline.py:94)

> **Say:** "It has the same `generate()` interface as the API client, so the pipeline didn't need any other change."

---

## Part 5 — Evaluation (3 min)

### 5.1 End-to-end (Table XI) — [scripts/evaluation/evaluate.py](scripts/evaluation/evaluate.py)
- `evaluate()` runs the 50 test questions through the full pipeline: [scripts/evaluation/evaluate.py:121](scripts/evaluation/evaluate.py:121)
- Metrics: [src/evaluation/citation_metrics.py](src/evaluation/citation_metrics.py) — validity [:39](src/evaluation/citation_metrics.py:39), coverage [:79](src/evaluation/citation_metrics.py:79), support [:142](src/evaluation/citation_metrics.py:142), hallucination [:225](src/evaluation/citation_metrics.py:225)
- Command: `LLM_BACKEND=local python scripts/evaluation/evaluate.py --output data/evaluation/run_D_nyaya_gen.json`
- Result file: `data/evaluation/run_D_nyaya_gen.json`

| Metric | API 550B | **Nyaya-Gen** |
|---|---|---|
| Citation validity | 0.70 | **0.74** |
| Cited claim coverage | 0.42 | **0.66** |
| Citation support | 0.50 | **0.56** |
| Hallucination rate | 0.04 | **0.00** |
| Response rate | 0.68 | **0.72** |

### 5.2 Standard LLM metrics (Table XII) — [scripts/evaluation/evaluate_generator_metrics.py](scripts/evaluation/evaluate_generator_metrics.py)
- Token F1 (SQuAD style): [:70](scripts/evaluation/evaluate_generator_metrics.py:70)
- ROUGE-L, BLEU, BERTScore: `score()` [:142](scripts/evaluation/evaluate_generator_metrics.py:142)
- Runs both our model and the untrained base model on 200 held-out examples
- Result file: `data/evaluation/generator_metrics.json`

| Metric | Before training | **Nyaya-Gen** |
|---|---|---|
| Token F1 | 0.42 | **0.93** |
| ROUGE-L | 0.40 | **0.93** |
| BERTScore | 0.83 | **0.98** |

### 5.3 Tests
```bash
python -m pytest -q        # 300 passed
```
Our model's test: [tests/test_local_generator.py](tests/test_local_generator.py)

---

## Part 6 — If asked "what are the limits?"
- Training stopped at update 50 because the 8 GB laptop ran out of memory (planned 323).
- Answers are short and restate one source; it was not taught to combine sources.
- Slower (45 s vs 17.5 s), mostly from the exact-match search under memory pressure; writing takes 11–15 s.
- Future work: longer training on a bigger machine, multi-source answers.
