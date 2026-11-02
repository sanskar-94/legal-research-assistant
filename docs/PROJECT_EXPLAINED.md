# Nyaya: The Whole Project Explained

**Citation-Grounded Indian Legal Research Assistant**
Team 12 · BCA · Bennett University
Sanskar Awasthi (S24BCAU0138) · Rishant Ranjan (S24BCAU0036)

This document explains every part of the project in plain English, so that both of us can explain it in the viva. It is based on the real code in this folder, the project report, and `docs/MODEL_PLAN.md`.

**Signs used in this document**

| Sign | Meaning |
|---|---|
| `file.py:function` | The place in the code where this happens. |
| **(example)** | Numbers made up to explain an idea. They are **not** results. |
| **(new)** | Written on 1–2 November 2026 for Nyaya-Gen or the evaluation. The file exists, but it may not have produced final results yet. |

## Contents

1. The project in 5 sentences
2. The big picture (words to know, offline and online diagrams, where the trained model sits)
3. One query from start to finish
4. Folder-by-folder and file-by-file guide
5. The data
6. Every formula, with a small example
7. Generation and the anti-hallucination layers
8. Evaluation: metrics, test set, baselines, sweeps
9. The trained model: Nyaya-Gen
10. How to run everything
11. Honest limitations
12. 34 likely viva questions with answers
13. One-page cheat sheet
- Appendix A: where the numbers come from
- Appendix B: things to check before the viva

---

## 1. The project in 5 sentences

1. Large language models (LLMs) such as ChatGPT often invent case names, citations and section numbers, and Indian users now also have to switch between the old Indian Penal Code (IPC) and the new Bharatiya Nyaya Sanhita (BNS), which replaced it on 1 July 2024 (the punishment for murder was IPC Section 302 and is now BNS Section 103).
2. Nyaya is a retrieval-augmented generation (RAG) assistant: it first finds the right passages in a local collection of 112,555 pieces of Indian legal text (statute questions and answers, Constitution material and Supreme Court judgments), and only then asks an LLM to write an answer from those passages, with a `[SOURCE_N]` tag after every claim.
3. To find the passages it runs two searches, one by meaning and one by exact legal identifiers such as "Section 302", merges them with weighted Reciprocal Rank Fusion whose weights depend on the type of question, and then re-ranks the result by the legal authority of each source.
4. What is new: structure-aware chunking (one section, article or paragraph per piece), the query-adaptive hybrid search, an IPC-to-BNS table that the model can cite as a source, a check of every citation, a confidence score with each answer, and evaluation with simple rule-based metrics that need no LLM judge.
5. The newest part is **one model we train ourselves** on an M2 MacBook Air (8 GB): **Nyaya-Gen** (SmolLM2-135M-Instruct, 134.5 million parameters) writes the cited answer from the retrieved passages instead of the online LLM, so the whole system can run offline.

---

## 2. The big picture

### 2.1 Words you must know

Each word is also explained again where it is first used.

| Word | Simple meaning | Where in Nyaya |
|---|---|---|
| **LLM** (large language model) | A very large program trained on huge amounts of text. It writes text one word-piece at a time. | DeepSeek V3.2 or NVIDIA Nemotron through NVIDIA NIM (an online service), or on the laptop: Nyaya-Gen (our own fine-tuned 135M model) or LFM3-1.7B. |
| **RAG** (retrieval-augmented generation) | "Search first, then write." The LLM only writes from the passages that the search found. | The whole pipeline. |
| **Hallucination** | When an AI states something false as if it were true, for example a case that does not exist. | What we try to prevent and what we measure. |
| **Token** | A small piece of text that a model reads: a word or part of a word. About 1 word = 1.3 tokens. | Chunk sizes; the LLM may write at most 1,024 tokens. |
| **Chunk** | A small piece of a document that is stored and searched on its own. | 112,555 chunks in the index. |
| **Embedding / vector** | A list of numbers that stands for the meaning of a text. Texts with similar meaning get similar numbers. | MiniLM turns every chunk and every question into 384 numbers. |
| **Cosine similarity** | A number that says how much two vectors point in the same direction. 1 = same direction (same meaning), 0 = unrelated. | The "semantic score" s(q, c). |
| **Vector database** | A database that stores vectors and quickly finds the ones nearest to a given vector. | ChromaDB, stored in `chroma_db/`. |
| **HNSW** | "Hierarchical Navigable Small World": a graph that lets the database find near vectors fast, without comparing with all 112,555 chunks one by one. | ChromaDB's index, set to cosine distance. |
| **Dense (semantic) search** | Search by meaning, using embeddings. | The "dense path". |
| **Exact-match search** | Search for an exact string such as "Section 302". | The "exact-match path" (README calls it "exact-match", the report also says "keyword"). |
| **Rank** | Position in a sorted list. Rank 1 is the best. | Used by RRF. |
| **RRF** (Reciprocal Rank Fusion) | A way to merge two ranked lists using only the positions: score = 1 / (60 + rank). | Merging dense and exact-match results. |
| **Re-ranking** | Sorting the candidates again with a better score. | The authority re-ranker (Section 6.7). |
| **Prompt** | The text sent to the LLM: rules, numbered sources and the question. | `config/prompts.py`. |
| **Temperature** | How random the LLM's word choice is. 0 = always the most likely word. Higher = more creative. | 0.1 (almost no randomness). |
| **Fine-tuning** | Training an already-trained model a little more, on our own task. | Nyaya-Gen. |
| **Causal language model** | A model that writes text one token at a time, each time predicting the next token from all the tokens before it. | Nyaya-Gen (Section 9.2). |
| **Loss** | One number that says how wrong the model is on some training examples. Training tries to make it small. | Next-token cross-entropy on answer tokens (Section 9.4). |
| **Epoch** | One full pass over all the training data. | 1 epoch planned for Nyaya-Gen; the run stopped after about 80 updates (Section 9.6). |
| **Learning rate** | How big each training update step is. | 1e-4 for Nyaya-Gen. |
| **Gradient accumulation** | Adding up the gradients of several small passes before one update, to get a bigger effective batch without more memory. | Nyaya-Gen, 8 examples per update (Section 9.6). |
| **Gradient checkpointing** | Not storing the intermediate results of the forward pass, and recomputing them during the backward pass: less memory, more time. | Nyaya-Gen (Section 9.6). |
| **MPS** | "Metal Performance Shaders": Apple's way of running PyTorch maths on the M-series GPU. | Training on the M2 Air. |
| **fp16 / fp32** | Numbers stored in 16 bits / 32 bits. fp16 uses half the memory and is faster, but is less exact. | fp32 for Nyaya-Gen (bf16, a 16-bit format, ran out of memory). |

### 2.2 Offline part: building the knowledge base (run once, about 15 minutes)

"Offline" means work done once, before anyone asks a question.

```
 OFFLINE INDEXING  (about 15 minutes on an M-series Mac, 11 of them for embedding)

 Hugging Face: 5 public datasets            Optional extra sources
        |                                   India Code statutes, Indian Kanoon judgments, PDFs
        | scripts/download_hf_datasets.py          | scripts/download_statutes.py
        v                                          | scripts/download_indiankanoon.py
 data/raw/*.json                                   v
        |                                   data/raw/... (PDF, HTML, text)
        | scripts/prepare_hf_data.py               | scripts/parse_pdfs.py
        |   (pick text, set doc type, ids)         |   (PyMuPDF, OCR fallback, cleaner,
        v                                          |    metadata extractor)
 data/processed/hf_documents.json  <---------------+
 data/processed/chunks/prechunked_hf.json  (already split by the dataset)
        |
        | scripts/chunk_documents.py   (one section / article / paragraph per chunk)
        v
 data/processed/chunks/all_chunks.json
        |
        | scripts/build_vectorstore.py (MiniLM: 384 numbers per chunk, batches of 64)
        v
 chroma_db/   ChromaDB collection "indian_legal", HNSW index, cosine space
              112,555 chunks, about 1.7 GB
```

### 2.3 Online part: answering one question

"Online" means what happens each time a user asks something.

```
 ONLINE QUERY PIPELINE   (src/pipeline.py : LegalResearchPipeline.query)

 question
   |
   v
 [A] IPC/BNS lookup  (src/retrieval/ipc_bns.py)  -----------------------+
   |   finds "Section 302 IPC", adds "Section 103 BNS" to the search    |
   v                                                                    |
 [B] classify the question: section / case / conceptual / mixed         |
   |   -> weights (alpha for dense, beta for exact-match)                |
   +------------------------+                                           |
   v                        v                                           |
 [C1] dense search       [C2] exact-match search                        |
   MiniLM vector,          "Section 302", "Section 103",                |
   20 nearest chunks       case party names, AIR/SCC citations          |
   +------------+-----------+                                           |
                v                                                       |
 [D] weighted RRF fusion, then remove duplicate texts                   |
                v                                                       |
 [E] authority re-ranking: S = 0.50 s + 0.25 R~ + 0.25 A  -> top 4      |
                v                                                       |
 [F] prompt: [SOURCE_1] ... [SOURCE_4]  +  [SOURCE_5] = IPC/BNS table <-+
                v
 [G] the answer writer: a [SOURCE_N] after each claim
       LLM_BACKEND=api   -> online LLM via NVIDIA NIM (temperature 0.1)
       LLM_BACKEND=local -> Nyaya-Gen on the laptop (Section 2.4)
                v
 [H] citation validation  ->  [I] confidence score C (High/Medium/Low/Very Low)
                v
 answer + numbered sources + IPC/BNS row + confidence breakdown
```

### 2.4 Where Nyaya-Gen sits

Nyaya-Gen replaces the online LLM in step [G]. The prompt it receives is the same prompt the online LLM gets; everything after it (citation check, confidence) is the same.

```
 [F] prompt: system rules + [SOURCE_1] ... [SOURCE_5] + question   (PromptBuilder, unchanged)
        |
        +-- LLM_BACKEND=api   : LLMClient.generate        (NVIDIA NIM, ~550B parameters, needs internet)
        |
        +-- LLM_BACKEND=local : LocalGenerator.generate   (src/generation/local_generator.py)
                 1. cut every source passage to its first 90 words (as in training)
                 2. Nyaya-Gen (SmolLM2-135M-Instruct, fine-tuned by us) on the M2 GPU
                 3. greedy decoding, repetition penalty 1.1, at most 256 new tokens
        |
 [H] citation validation -> [I] confidence     (unchanged)
```

With `LLM_BACKEND=local`, **nothing leaves the laptop**: search (MiniLM + ChromaDB), authority re-ranking and writing (Nyaya-Gen) all run locally.

---

## 3. One query from start to finish

**Query:** "What is Section 302 of IPC and its BNS equivalent?"
**Command:** `python scripts/ask.py "What is Section 302 of IPC and its BNS equivalent?"`

How to read this section:
- The outputs of stages 2, 3 and 6 are **real**: we ran that code on this query.
- The chunks and scores in stages 5 to 9 and the answer are **(example)** values, chosen to look like what the system really returns. Use them to explain the method, not as results.

### Stage 0. Start-up

- **Code:** `scripts/ask.py:main` -> `src/pipeline.py:LegalResearchPipeline.from_settings`
- **What happens:** loads the MiniLM embedding model (`src/embeddings/embedding_service.py:EmbeddingService`), opens ChromaDB (`src/embeddings/vectorstore.py:VectorStore`), creates the LLM client if an API key is set (`src/generation/llm_client.py:LLMClient`), or, with `LLM_BACKEND=local`, loads Nyaya-Gen (`src/generation/local_generator.py:LocalGenerator`), and loads the 63-row IPC/BNS table (`src/retrieval/ipc_bns.py:IpcBnsMapper`).
- **Out:** a ready pipeline. If the index is empty it stops with "Build it first".

### Stage 1. Empty-question check

- **Code:** `LegalResearchPipeline.query`
- **In:** the question text.
- **What happens:** an empty or blank question is rejected at once with "Query must not be empty." (This is how the empty edge-case test query is handled.)

### Stage 2. IPC/BNS lookup and query expansion (real output)

- **Code:** `IpcBnsMapper.find`, then `IpcBnsMapper.expand_query`
- **What happens:** regular expressions (text patterns) look for IPC or BNS section numbers in either order ("Section 302 IPC", "IPC 302", "s. 302 of Indian Penal Code"). "Section 302 of IPC" matches. The row for 302 is found in `config/ipc_bns_mapping.json`.
- **Out (real):**
  - matched row: IPC 302 "Punishment for murder" -> BNS 103 "Punishment for murder", `matched_on = "ipc"`
  - expanded search query: `What is Section 302 of IPC and its BNS equivalent? (Section 103 BNS; Punishment for murder)`

The expanded query is used **only for searching**. The LLM later sees the user's original question.

### Stage 3. Question classification (real output)

- **Code:** `src/retrieval/hybrid_search.py:HybridSearch.classify_query`
- **Rules:** "Section n", "Article n", "Act", "Code of", IPC, CrPC, CPC or BNS -> section signal. A case name "X v Y" (with a non-generic party) or an AIR/SCC citation -> case signal. Both -> `mixed`. Only section -> `section_lookup`. Only case -> `case_lookup`. Neither -> `conceptual`.
- **Out (real):** `section_lookup`, so the weights are alpha = 0.3 (dense) and beta = 0.7 (exact-match).

### Stage 4. Turning the question into a vector

- **Code:** `EmbeddingService.embed_query`
- **In:** the expanded query.
- **Out:** a list of 384 numbers, for example `[0.021, -0.087, 0.044, ...]` **(example)**.

### Stage 5. Dense (meaning) search

- **Code:** `HybridSearch._semantic_search` -> ChromaDB `collection.query(..., n_results=20)`
- **Out:** the 20 chunks whose vectors are nearest to the question vector. For each, ChromaDB gives a cosine distance d, and the code stores `semantic_score = 1 - d` (`_hits_from_query_result`).

### Stage 6. Exact-match search (terms are real output)

- **Code:** `HybridSearch._keyword_search`, `_extract_keyword_terms`, `_case_variants`, `_whole_term_pattern`
- **Terms found (real):** `["Section 302", "Section 103"]`. Each is tried in two spellings: `"Section 302"` and `"section 302"` (ChromaDB's text filter is case-sensitive).
- **What happens:** for each spelling, ChromaDB returns up to 100 chunks that **contain** the text, ordered by closeness to the question vector. A whole-word check then removes false matches ("Section 302" inside "Section 3021" or "Section 302A"). The 20 nearest remaining chunks are kept.
- **Out:** a second ranked list of up to 20 chunks.

### Stage 7. Weighted RRF fusion

- **Code:** `hybrid_search.py:fuse_hits` (called by `HybridSearch.search`), which uses `_reciprocal_rank_fusion`
- **What happens:** each chunk gets `R = 0.3 / (60 + dense rank) + 0.7 / (60 + exact rank)`. A missing rank adds 0.

**(example)** Six of the candidates:

| Chunk | What it is | Dense rank (s) | Exact rank | R |
|---|---|---|---|---|
| c1 | Q&A pair "What is the punishment for murder under Section 302 IPC?" (type statute) | 1 (0.71) | 2 | 0.3/61 + 0.7/62 = 0.016208 |
| c2 | Supreme Court judgment paragraph: conviction under Section 302 IPC upheld | 4 (0.62) | 1 | 0.3/64 + 0.7/61 = 0.016163 |
| c3 | Q&A pair on culpable homicide and murder, Sections 299 and 300 (type statute) | 2 (0.68) | – | 0.3/62 = 0.004839 |
| c4 | Another Supreme Court paragraph mentioning Section 302 | – (0.55) | 3 | 0.7/63 = 0.011111 |
| c5 | An exact copy of c1 from another dataset | 5 | 4 | 0.015553 |
| c6 | Constitution chunk on Article 21 (right to life) | 3 (0.64) | – | 0.3/63 = 0.004762 |

(c4 was found only by exact match. It still has a semantic score, because the exact-match search also measures distance to the question vector.)

### Stage 8. Remove duplicate texts

- **Code:** `src/retrieval/retrieval_pipeline.py:_drop_duplicate_texts`
- **What happens:** texts that are identical after lower-casing and removing extra spaces are dropped; the first one (higher R) stays.
- **Out (example):** c5 is removed, because it is a copy of c1.

### Stage 9. Authority re-ranking (final score S)

- **Code:** `src/retrieval/reranker.py:AuthorityReranker.rerank` and `compute_authority_score`
- **What happens:** for each chunk, `R~ = min(R x 61, 1)`, the authority A is computed, and `S = 0.50 s + 0.25 R~ + 0.25 A`. The top K = 4 by S are kept.

**(example)**

| Chunk | s | R~ | A | S = 0.5 s + 0.25 R~ + 0.25 A | Rank |
|---|---|---|---|---|---|
| c1 | 0.71 | 0.989 | 1.00 (statute) | 0.355 + 0.247 + 0.250 = **0.852** | 1 |
| c2 | 0.62 | 0.986 | 0.61 (SC, no date) | 0.310 + 0.247 + 0.153 = **0.709** | 2 |
| c3 | 0.68 | 0.295 | 1.00 (statute) | 0.340 + 0.074 + 0.250 = **0.664** | 3 |
| c6 | 0.64 | 0.290 | 1.00 (Constitution) | 0.320 + 0.073 + 0.250 = **0.643** | 4 |
| c4 | 0.55 | 0.678 | 0.61 (SC, no date) | 0.275 + 0.170 + 0.153 = **0.597** | 5 (dropped) |

Notice: the Article 21 chunk (c6) beats a judgment that really mentions Section 302 (c4), only because statutes and the Constitution get authority 1.0. This is exactly the effect the report measured: the authority prior costs 3.1 points of keyword precision (Table III, row f vs row e).

### Stage 10. Add the IPC/BNS table as the last source

- **Code:** `IpcBnsMapper.as_source_chunk`
- **Out (real text):**
  ```
  Correspondence between the Indian Penal Code, 1860 and the Bharatiya Nyaya Sanhita, 2023 (in force from 1 July 2024):
  - IPC Section 302 (Punishment for murder) corresponds to BNS Section 103 (Punishment for murder).
  ```
  It becomes `[SOURCE_5]`, with doc type `mapping` and authority 1.0. Sources are now: c1, c2, c3, c6, table.

### Stage 11. Build the prompt

- **Code:** `src/generation/prompt_builder.py:PromptBuilder.build_prompt`, using `config/prompts.py`
- **Out:** a system prompt (the 8 rules, Section 7.1) and a user prompt like this **(example, shortened)**:
  ```
  SOURCES:
  [SOURCE_1]
  Type: statute
  Statute (details unavailable)
  ---
  Legal Question: What is the punishment for murder under Section 302 IPC? ...
  ---
  [SOURCE_2]
  Type: judgement
  Court: Supreme Court of India
  ---
  ... the conviction of the appellant under Section 302 IPC ...
  ---
  ...
  [SOURCE_5]
  Type: mapping
  Correspondence table: Indian Penal Code, 1860 (Act 45 of 1860) / Bharatiya Nyaya Sanhita, 2023 (Act 45 of 2023)
  ---
  Correspondence between the Indian Penal Code, 1860 and ...
  ---

  QUESTION: What is Section 302 of IPC and its BNS equivalent?

  Answer following the rules above, citing sources as [SOURCE_1], [SOURCE_2], etc.
  ```

### Stage 12. The LLM writes the answer

- **Code:** `LLMClient.generate` -> `_call_api` -> `_strip_model_thinking`
- **Settings:** temperature 0.1, at most 1,024 output tokens, top_p 0.95, up to 3 tries with growing waits (2 to 30 s), at most 40 requests per minute.
- **Out (example):**
  > Section 302 of the Indian Penal Code, 1860 sets the punishment for murder: death or imprisonment for life, and a fine [SOURCE_1]. The Supreme Court has upheld convictions and life sentences under Section 302 IPC [SOURCE_2]. Murder is separated from culpable homicide by Sections 299 and 300 [SOURCE_3]. Under the Bharatiya Nyaya Sanhita, 2023, in force from 1 July 2024, IPC Section 302 corresponds to BNS Section 103, Punishment for murder [SOURCE_5]. Not covered by the retrieved sources: the full text of BNS Section 103.

Any `<think>...</think>` block or "Reasoning:" section that a model adds is removed before the answer is used.

**With `LLM_BACKEND=local`** this stage is `LocalGenerator.generate` instead (Section 9.8): the same prompt, sources cut to 90 words, greedy decoding, at most 256 new tokens, no internet. A Nyaya-Gen answer is shorter and plainer, typically one source restated in a few cited sentences.

### Stage 13. Citation validation

- **Code:** `src/generation/response_parser.py:ResponseParser.parse_response`
- **What happens:** grouped tags like `[SOURCE_2, SOURCE_4]` are first split into `[SOURCE_2] [SOURCE_4]`. Every `[SOURCE_N]` is found. It is valid if 1 <= N <= 5 (the number of sources sent).
- **Out (example):** tags 1, 2, 3, 5 -> all valid, citation validity = 4/4 = 1.0, invalid list empty. SOURCE_4 (Article 21) was not cited.

### Stage 14. Confidence score

- **Code:** `src/confidence/scorer.py:ConfidenceScorer.score`
- **Out (example):** C = 0.814, level **High**. The full calculation is in Section 6.8.

### Stage 15. What the user sees

- **Code:** `scripts/ask.py:render_text` (or `--json` for everything)
- **Out:** the answer; a note if any citation was invalid; the line `IPC 302 (Punishment for murder) -> BNS 103`; the 5 sources, with `*` in front of the cited ones and a short snippet of each; and `Confidence: 0.81 (High) - retrieval quality 0.69, source agreement 1.00, authority strength 0.87, coverage score 0.75` **(example)**.

If no API key is set, the pipeline stops after Stage 10 and returns only the sources with a note.

---

## 4. Folder-by-folder and file-by-file guide

### 4.1 Top level

| File | What it is |
|---|---|
| `README.md` | Main description: features, setup, build steps, recorded results, list of fixes since the report. |
| `requirements.txt` | Python packages: python-dotenv, pydantic, openai, tenacity, sentence-transformers, chromadb, PyMuPDF, pytesseract, Pillow, datasets, beautifulsoup4, lxml, requests, pytest. (sentence-transformers also installs PyTorch and transformers, which Nyaya-Gen uses.) |
| `.env.example` | Template for secret settings: `NVIDIA_API_KEY`, and optional `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`. Copy it to `.env`. |
| `.env` | Your own copy with the real key. Never put it in git. |
| `.gitignore` | Keeps out of git: `.env`, `data/raw/`, `data/processed/`, `chroma_db/`, `models/`, model files (`*.pt`, `*.safetensors`, ...). |

### 4.2 `config/` — settings in one place

| File | What it holds |
|---|---|
| `__init__.py` | Empty. Makes `config` a Python package. |
| `settings.py` | All the numbers: paths; embedding model `all-MiniLM-L6-v2`, batch 64; chunk sizes 450 / 800 / 50; `SEMANTIC_TOP_K = 20`, `RERANKED_TOP_K = 4`, `RRF_K = 60`; score weights 0.50 / 0.25 / 0.25; court and recency weights; LLM URL, model, temperature 0.1, max tokens 1,024, 40 requests/min; confidence weights 0.40 / 0.25 / 0.20 / 0.15; Nyaya-Gen settings at the end (`LLM_BACKEND` = `api` or `local`, `LOCAL_GENERATOR_BASE_MODEL`, `LOCAL_GENERATOR_PATH` / `NYAYA_GEN_PATH`, `LOCAL_GENERATOR_SOURCE_WORDS = 90`, `LOCAL_GENERATOR_MAX_NEW_TOKENS = 256`). |
| `prompts.py` | `SYSTEM_PROMPT` (the 8 rules and the answer format), `USER_PROMPT_TEMPLATE` (sources + question), `SOURCE_FORMAT_TEMPLATE` (how one `[SOURCE_N]` block looks). |
| `ipc_bns_mapping.json` | The IPC-to-BNS table: 63 rows, each with `ipc_section`, `ipc_title`, `bns_sections` (a list, because one IPC section can map to several BNS parts) and `bns_title`. |

### 4.3 `src/` — the system itself

| File | What it does |
|---|---|
| `src/__init__.py` | Empty package marker. |
| `src/pipeline.py` | `LegalResearchPipeline`: runs all online stages (IPC/BNS lookup, retrieval, prompt, LLM, citation parsing, confidence) and returns a `PipelineResult`. `from_settings()` builds it from `config/settings.py`. |
| **ingestion/** | Only used for the optional PDF/HTML sources. |
| `ingestion/__init__.py` | Exports the three tools below. |
| `ingestion/pdf_parser.py` | `parse_pdf`: reads PDF text with PyMuPDF. If a page gives fewer than 50 characters (a scanned image), it runs Tesseract OCR (optical character recognition: reading text from an image) on that page. |
| `ingestion/text_cleaner.py` | `clean_legal_text`: removes page headers, footers and URL lines, fixes common OCR mistakes ("tbe" -> "the", "judgrnent" -> "judgment"), normalises spaces, writes "Sec. 302" as "Section 302", standardises AIR/SCC citations. |
| `ingestion/metadata_extractor.py` | `extract_metadata`: regular expressions that find the case name, AIR/SCC citations, date, court, judges and sections cited. |
| **chunking/** | |
| `chunking/__init__.py` | Exports the chunk models and chunkers. |
| `chunking/chunk_models.py` | Pydantic models (Python classes that check data types): `ChunkMetadata` (doc id, type, case name, court, date, act, section number, ...) and `Chunk`. If a chunk has no id, it gets `doc_id::index`, so ids are the same on every rebuild. |
| `chunking/legal_chunker.py` | `estimate_tokens` (words x 1.3), `chunk_statute` (one section per chunk), `chunk_constitution` (one article per chunk), `chunk_judgement` (numbered paragraphs), and `chunk_document`, which picks the right one by doc type. |
| **embeddings/** | |
| `embeddings/__init__.py` | Empty package marker. |
| `embeddings/embedding_service.py` | `EmbeddingService`: loads `all-MiniLM-L6-v2` once; `embed_query` (one text -> 384 numbers) and `embed_batch` (many texts, batches of 64). |
| `embeddings/vectorstore.py` | `VectorStore`: opens the ChromaDB folder, collection `indian_legal` with cosine distance; `add_chunks` (upsert = insert or update), `count`, `reset`. Stores 10 metadata fields per chunk. |
| **retrieval/** | |
| `retrieval/__init__.py` | Exports the retrieval classes. |
| `retrieval/hybrid_search.py` | `HybridSearch`: `classify_query`, dense search, exact-match search (identifiers, case party names, whole-word check), `search_paths` (both lists, used by the experiments), and weighted RRF fusion (`fuse_hits`). |
| `retrieval/reranker.py` | `AuthorityReranker`: `compute_authority_score` (court, recency, citations) and `rerank` (final score S, top K). |
| `retrieval/retrieval_pipeline.py` | `RetrievalPipeline.retrieve`: hybrid search -> remove duplicates -> authority re-rank -> top 4. |
| `retrieval/ipc_bns.py` | `IpcBnsMapper`: `find` (IPC/BNS references or offence names), `expand_query`, `as_source_chunk` (the table as a citable source). |
| **generation/** | |
| `generation/__init__.py` | Exports the generation classes. |
| `generation/llm_client.py` | `LLMClient`: talks to any OpenAI-compatible server (NVIDIA NIM by default, or a local llama.cpp server); rate limit, retries, removes "thinking" text. |
| `generation/local_generator.py` | **(new)** Nyaya-Gen at query time: `LocalGenerator.generate(prompt, system_prompt)` (same call as `LLMClient`), `shorten_sources` (cuts each source to 90 words). Loads the model from `models/nyaya-gen/` on MPS or CPU; raises `LocalGeneratorError` if it is missing. |
| `generation/prompt_builder.py` | `PromptBuilder.build_prompt`: numbers the sources, adds a metadata line per type (`flatten_chunk`, `source_label`). |
| `generation/response_parser.py` | `ResponseParser.parse_response`: finds `[SOURCE_N]` tags, splits grouped tags, checks validity, lists cited and invalid sources. |
| **confidence/** | |
| `confidence/__init__.py` | Empty package marker. |
| `confidence/scorer.py` | `ConfidenceScorer.score`: the four-signal confidence C and its level. |
| **evaluation/** | |
| `evaluation/__init__.py` | One-line description of the package. |
| `evaluation/citation_metrics.py` | Answer metrics: citation validity, cited claim coverage, citation support, hallucination detector, response (substantive answer) check. |
| `evaluation/retrieval_metrics.py` | Retrieval metrics P@K, P_kw@K, Hit@K, reciprocal rank / MRR (newly added). |
| `evaluation/summary.py` | Turns per-query results into Table IV (summary per run) and Table V (per category); also reads the older Run A format; mean latency over the 49 non-empty queries (newly added). |

### 4.4 `scripts/` — things you run

| File | What it does |
|---|---|
| `download_hf_datasets.py` | Downloads the 5 Hugging Face datasets into `data/raw/<name>_<split>.json` (3 tries on network errors). |
| `prepare_hf_data.py` | Converts them: to documents for chunking (`data/processed/hf_documents.json`) or to ready chunks (`data/processed/chunks/prechunked_hf.json`). |
| `download_statutes.py` | Optional. Downloads the IPC, BNS, CrPC and Constitution from India Code (1 request per second). |
| `download_indiankanoon.py` | Optional. Downloads about 200 landmark Supreme Court judgments from Indian Kanoon. |
| `parse_pdfs.py` | Optional. Parses everything in `data/raw/` (PDF, HTML, text) with the ingestion tools and writes JSON to `data/processed/`. |
| `chunk_documents.py` | Chunks every document in `data/processed/` and writes one file, `chunks/all_chunks.json`. |
| `build_vectorstore.py` | Embeds all chunks and upserts them into ChromaDB in batches of 1,000. `--reset` empties the collection first. |
| `ask.py` | Command-line question answering: one question, `--json`, `-v` for logs, or an interactive loop. |
| `evaluate.py` | Runs the 50 test queries through the full pipeline and writes `data/evaluation/eval_results_latest.json` (or `--output`), now including P_kw@K, Hit@K, MRR, latency, a per-category table and the model name. `--limit N` runs only the first N; `--summarise FILE` prints Tables IV and V for a saved run without the index or LLM. |
| `local_llm.sh` | `start` / `stop` / `status` for LFM3-1.7B with llama.cpp on port 8080, for fully offline use. |
| `retrieval_experiments.py` | **(new)** Reproduces Tables III, VI, VII and VIII without the LLM. Runs both search paths once per query with a pool of 40, caches them in `data/evaluation/retrieval_cache.json`, then scores every setting from the cache. |
| `prepare_generator_data.py` | **(new)** Builds the Nyaya-Gen examples (prompt in the pipeline's exact format + cited target answer) into `data/generator/{train,dev,test}.jsonl` plus `data_stats.json`. Needs only the raw datasets, not the index. Section 9.3. |
| `train_generator.py` | **(new)** Fine-tunes SmolLM2-135M-Instruct into Nyaya-Gen: plain PyTorch loop, loss on answer tokens only, fp32 with gradient checkpointing; logs to `data/generator/training_log.json`, saves the best model (lowest dev loss) to `models/nyaya-gen/`. `--max-minutes` stops cleanly; `--max-steps 2 --limit 20` is a smoke test. Section 9.5. |
| `run_training.sh` | **(new)** Prepares the Nyaya-Gen data if it is missing, then trains Nyaya-Gen with the settings that fit in 8 GB (`--grad-accum 8 --eval-every 50 --eval-examples 40 --max-minutes 75`). Meant to run detached: `nohup caffeinate -i scripts/run_training.sh > logs/training.log 2>&1 &`. |
| `run_evaluation.sh` | **(new)** Three steps: 1) Run D, Nyaya-Gen fully local (`LLM_BACKEND=local`) -> `data/evaluation/eval_results_local.json`; 2) `retrieval_experiments.py` for Tables III and VI–VIII; 3) saves the un-fine-tuned SmolLM2 to `models/smollm2-base/` and runs Run D0 with it (`NYAYA_GEN_PATH=models/smollm2-base`) -> `eval_results_local_base.json`. Log: `logs/evaluation.log`. |

### 4.5 `tests/` — automatic checks (250 in the report; recount with `pytest --collect-only -q`)

All tests run without internet or API key; the vector store and the LLM are replaced by fakes ("mocks").

| File | What it tests |
|---|---|
| `__init__.py` | Empty package marker. |
| `conftest.py` | Shared test fixtures (ready-made fake chunks, fake vector store) and path setup. |
| `test_chunker.py` | Token estimate, overlap, merging small paragraphs, splitting long text, the three chunkers, the router, edge cases, stable chunk ids. |
| `test_pdf_parser.py` | Digital PDF text, OCR fallback, missing files, empty PDFs (PyMuPDF and Tesseract are mocked). |
| `test_retrieval.py` | Query classification, identifier extraction, RRF scores, authority and recency scores, weights per type, re-ranker, hybrid search, RRF rescaling, duplicate removal, whole-word matching ("Article 21" is not "Article 210"). |
| `test_ipc_bns.py` | Finding IPC/BNS references, the corrected rows (IPC 376 -> BNS 64, IPC 364 -> BNS 140(1)), query expansion and the source chunk. |
| `test_generation.py` | Prompt building, citation parsing and validation, grouped tags, source labels, the LLM client (mocked), metadata flattening. |
| `test_pipeline.py` | The whole `LegalResearchPipeline` with mocks, an end-to-end run with real retrieval code and a fake store, and the confidence scorer. |
| `test_evaluation_metrics.py` | Citation validity, coverage, support, hallucination and response checks. |
| `test_retrieval_experiments.py` | **(new)** `retrieval_experiments.py` on a small hand-made cache, with expected values worked out by hand from the report's equations. |
| `test_local_generator.py` | **(new)** Nyaya-Gen helpers: cutting sources to 90 words, leaving short prompts unchanged, a clear error when the model folder is missing (no model download). |

### 4.6 `data/evaluation/` — test set and recorded results

| File | What it holds |
|---|---|
| `test_queries.json` | The 50 hand-made test questions: `query`, `category`, `expected_doc_types`, `expected_keywords`. |
| `eval_results.json` | **Run A** (22 March 2026, DeepSeek V3.2, 6 passages, old prompt). Has the old name "citation accuracy". Its per-query `P@4`, `Pkw@4` and `Hit@4` fields come from no code in the project and should not be quoted. |
| `eval_results_nemotron.json` | **Run B** (26 October 2026, Nemotron 3 Ultra 550B, current prompt). |
| `eval_results_deepseek.json` | 27 October 2026. Its numbers (validity 0.70, response 0.68, hallucination 0.04, precision 0.915, confidence 0.607) are exactly **Run C**, although the report and README call Run C a Nemotron run. See Appendix B. |
| `eval_results_latest.json` | Default output of `evaluate.py`. Not in the folder at the moment; it appears after the next run. |

### 4.7 `docs/`

| File | What it is |
|---|---|
| `MODEL_PLAN.md` | The design of the trained model, Nyaya-Gen: why, where it sits, training data, settings, evaluation, what happened in training, files. |
| `REPORT_ADDITIONS.md` | Paste-ready report text, tables and the equation for Nyaya-Gen, with all real numbers filled in. |
| `REPORT_FIXES.md` | Corrections to the existing report (references, equations, numbers). |
| `ALIGNMENT_CHANGES.md` | A claim-by-claim check of the report against the code (written 2 November 2026): what matches, what was fixed in code, and what the report text should change. |
| `PROJECT_EXPLAINED.md` | This document. |

### 4.8 Folders made by the scripts (not in git)

| Folder | Made by | Contents |
|---|---|---|
| `data/raw/` | download scripts | Raw downloaded datasets (about 240 MB). |
| `data/processed/` | prepare / parse / chunk scripts | Documents and chunks as JSON. |
| `chroma_db/` | `build_vectorstore.py` | The vector index, about 1.7 GB. |
| `data/generator/` | `prepare_generator_data.py`, `train_generator.py` | Nyaya-Gen examples (`train/dev/test.jsonl`), `data_stats.json`, `training_log.json`. |
| `models/nyaya-gen/` | `train_generator.py` | The trained Nyaya-Gen with its tokenizer and `training_config.json`, about 540 MB (134.5M weights x 4 bytes). |
| `models/smollm2-base/` | `run_evaluation.sh` | The un-fine-tuned SmolLM2-135M-Instruct, saved for Run D0. |
| `logs/` | `run_training.sh`, `run_evaluation.sh` | `training.log`, `evaluation.log`, and `training_failed_oom.log` (the earlier out-of-memory attempts, Section 9.6). |
| `data/evaluation/retrieval_cache.json` | `retrieval_experiments.py` | Cached search results for the retrieval tables. |

---

## 5. The data

### 5.1 The five Hugging Face datasets (Table I of the report)

Hugging Face is a website that hosts free datasets and models. We use five public Indian-law datasets. The "What is really inside" column comes from `scripts/prepare_hf_data.py` and from opening the downloaded files.

| Dataset | Records | Report says | What is really inside | How we use it |
|---|---|---|---|---|
| `viber1/indian-law-dataset` | 24,607 | Statute Q&A pairs | Question ("Instruction") and answer ("Response") pairs about IPC, CrPC, the Constitution and legal terms. | Pairs with an answer of at least 100 characters (23,006) become documents "Legal Question: ... Legal Answer: ...". Doc type is guessed from words: "article/constitution/fundamental/directive principle" -> constitution; "section/IPC/BNS/CrPC/penal code/criminal procedure" -> statute; otherwise judgement. Then chunked. |
| `ninadn/indian-legal` | 7,130 | SC judgments, summaries | Full Supreme Court judgments ("Text") with a summary (7,030 train + 100 test). | Texts of at least 200 characters become judgment documents with court "Supreme Court of India". Case name is empty (the dataset does not give it). Then chunked. |
| `vihaannnn/Indian-Supreme-Court-Judgements-Chunked` | 20,961 | Pre-split SC judgments | Supreme Court judgment passages, already split. | Passages of at least 50 characters (10,572) are indexed as they are, as judgments. |
| `ShreyasP123/Legal-Dataset-for-india` | 1,469 | Pre-split legal texts | Mixed legal texts, already split, with numeric ids. | Prepared as ready chunks, but see 5.4: they are not actually in the index. |
| `nisaar/Constitution_of_India` | 933 | Constitution articles | Question-and-answer pairs about the Constitution. | Each pair becomes a document "Constitutional Question: ... Answer: ..." with type constitution. Then chunked. |

So the "statute" and "constitution" material in the default build is mostly **questions and answers about** the law, not the bare text of each Act. For the bare text of IPC/BNS sections you must also run `download_statutes.py` (Section 10.4).

### 5.2 What "preparation" does

`scripts/prepare_hf_data.py` reads each raw file and, for every record:
1. picks the text field and skips records that are too short;
2. gives it a `doc_id` (for example `indian_legal_train_15`) and a `doc_type` (statute, constitution or judgement);
3. adds the metadata it can (court for the judgment datasets);
4. writes documents to `data/processed/hf_documents.json` (31,069 documents) and the ready-split passages to `data/processed/chunks/prechunked_hf.json` (12,041 passages).

### 5.3 Structure-aware chunking: what and why

**The problem with fixed-size chunks.** A simple RAG system cuts text every, say, 500 words. In law that cuts a section in half, and an Exception or a Proviso ends up in a different chunk from the rule it changes. The model can then see "Whoever commits murder shall be punished..." without the exception that lowers the offence, or the reverse.

**Our rule: cut where the law itself is cut.** (`src/chunking/legal_chunker.py:chunk_document` picks the method by doc type.)

| Doc type | How it is split | Extra rules |
|---|---|---|
| Statute | One **section** per chunk (`chunk_statute` splits at "Section n" at the start of a line). | A section longer than 800 tokens is split once at sub-section markers like (1), (a), (i), and each piece starts with the Act name and section header. A long section with no such markers stays whole. |
| Constitution | One **article** per chunk (`chunk_constitution` splits at "Article n"). | Explanations and provisos stay with their article. An article over 800 tokens is split at sentence boundaries with a 50-token overlap. |
| Judgment | Split at **numbered paragraphs** "1.", "(1)", "[1]" (`chunk_judgement`). | Paragraphs smaller than 225 tokens are merged with the next. Pieces over 800 tokens are split at sentences. The last 50 tokens of each paragraph are carried into the next chunk (overlap) so a sentence that runs across is not lost. The case name, if known, is put at the start of every chunk. |

Settings (`config/settings.py`): target 450 tokens, maximum 800, overlap 50 (all estimated with words x 1.3, Section 6.1).

**Why it helps.**
- Each chunk is one legal unit, so it can be cited as one source ("Section 302", "Article 21").
- The exact-match search can find "Section 302" and get the whole section, not half of it.
- The case name in every judgment chunk tells the model (and the user) where the paragraph comes from.

**Chunk ids.** Each chunk id is `doc_id::chunk_index` (`chunk_models.py:Chunk._default_chunk_id`). Rebuilding the index updates the same ids instead of adding copies. (Earlier versions used random ids, so every rebuild doubled the index.)

### 5.4 The 112,555 chunks: where the number comes from

We re-ran the chunker on the prepared documents (1 November 2026) and got exactly the numbers in the report:

| Step | Judgment | Constitution | Statute | Total |
|---|---|---|---|---|
| Chunks from chunking 31,069 documents | 75,407 | 22,576 | 4,000 | **101,983** |
| Pre-split passages (SC chunked + ShreyasP) | 11,692 | 78 | 271 | **12,041** |
| Together, before indexing | 87,099 | 22,654 | 4,271 | 114,024 |
| **Indexed in ChromaDB** | **85,979 (76.4%)** | **22,576 (20.1%)** | **4,000 (3.6%)** | **112,555** |

The average chunk is about 468 estimated tokens, close to the 450 target.

**Why 114,024 becomes 112,555 (difference 1,469).** The report says "after duplicate identifiers are merged". We checked the code and the data: the real reason is that all 1,469 ShreyasP passages have **numeric** ids (1, 2, 3, ...), while the `Chunk` model requires text ids. `build_vectorstore.py` skips them as invalid, so 101,983 + 10,572 = 112,555. In other words, the ShreyasP dataset is downloaded but not in the index. If the teacher asks, say this honestly. It is easy to fix (turn the id into text) but the fix changes the index, so the recorded numbers would change.

**The corpus is unbalanced:** three quarters are judgment chunks. Many judgment chunks have no case name, date or citation count, so their authority score falls back to default values (Section 6.5).

**Privacy:** all data is published legal material. Names in it are already public in court records. No other personal data was collected.

---

## 6. Every formula, with a small example

The examples in 6.3 to 6.8 use the same chunks c1 to c6 as the walkthrough in Section 3, so the numbers connect.

### 6.1 Token estimate (report equation 1)

```
T^(c) = floor( 1.3 x |W(c)| )
```
- |W(c)| = number of words in chunk c (words are separated by spaces).
- floor = round down to a whole number.
- Code: `src/chunking/legal_chunker.py:estimate_tokens` (`int(len(text.split()) * 1.3)`).

**Example.** "Whoever commits murder shall be punished with death or imprisonment for life, and shall also be liable to fine." has 19 words. T^ = floor(1.3 x 19) = floor(24.7) = **24 tokens**. A paragraph of 346 words gives floor(449.8) = 449, which is right at the 450 target.

**Why an estimate?** Counting real tokens needs a tokenizer for one specific model and is slower. English text has about 1.3 tokens per word, which is close enough to decide where to cut.

### 6.2 Cosine similarity (report equation 2)

```
s(q, c) = 1 - d_cos(q, c) = (e_q . e_c) / ( ||e_q|| x ||e_c|| )
```
- e_q, e_c = the embedding vectors (384 numbers each) of question q and chunk c.
- e_q . e_c = dot product: multiply number by number and add up.
- ||e|| = length of a vector: square root of the sum of squares.
- d_cos = cosine distance, which ChromaDB returns. Similarity = 1 - distance.
- Code: `hybrid_search.py:_hits_from_query_result` stores `max(0, 1 - distance)`.

**Example with 3 numbers instead of 384.** e_q = (1, 2, 2), e_c = (2, 1, 2).
- Dot product = 1x2 + 2x1 + 2x2 = 2 + 2 + 4 = 8
- ||e_q|| = sqrt(1 + 4 + 4) = 3, ||e_c|| = sqrt(4 + 1 + 4) = 3
- s = 8 / (3 x 3) = 8/9 = **0.889** (distance 0.111)

Cosine looks only at **direction**, not length, so a long and a short text about the same topic can still score high.

(A bug fixed after the report: the code used `1 / (1 + d)`, which is not cosine similarity. It is now `1 - d`.)

### 6.3 Reciprocal Rank Fusion (report equation 3)

```
RRF_L(c) = 1 / (k + r_L(c))        and RRF_L(c) = 0 if c is not in list L
k = 60
```
- L = one ranked list (the dense list or the exact-match list).
- r_L(c) = rank of chunk c in that list (1, 2, 3, ...).
- Code: `hybrid_search.py:_reciprocal_rank_fusion`.

| Rank | 1 | 2 | 3 | 10 | 20 |
|---|---|---|---|---|---|
| 1/(60 + rank) | 0.01639 | 0.01613 | 0.01587 | 0.01429 | 0.01250 |

**Why RRF?** The two searches give scores on different scales; you cannot simply add them. RRF uses only positions, so no scaling is needed. It comes from Cormack et al. (SIGIR 2009) and works well without any training.

**Why k = 60?** k makes the scores flatter. With k = 1, rank 1 scores 1/2 = 0.5 and rank 2 scores 1/3 = 0.333, so the first item of each list dominates. Table VII shows keyword precision falls to 0.750 with k = 1. Between k = 30 and k = 100 the results are within 1.5 points. 60 is the standard value from the original paper.

### 6.4 Weighted fusion with per-type weights (report equation 4)

```
R(c) = alpha_t x RRF_sem(c) + beta_t x RRF_kw(c)
```

| Question type t | alpha (dense) | beta (exact-match) | Reason |
|---|---|---|---|
| section_lookup | 0.3 | 0.7 | "Section 498A" must be matched exactly. |
| case_lookup | 0.2 | 0.8 | A case name is an exact string; its meaning is vague. |
| conceptual | 0.8 | 0.2 | "Explain basic structure" has no identifier; meaning matters. |
| mixed | 0.5 | 0.5 | Both signals. |

Code: `hybrid_search.py:_CLASSIFICATION_WEIGHTS` and `fuse_hits`. alpha + beta = 1 always.

**Example (section_lookup, from Section 3).**
- c1: dense rank 1, exact rank 2: R = 0.3 x 1/61 + 0.7 x 1/62 = 0.004918 + 0.011290 = **0.016208**
- c3: dense rank 2 only: R = 0.3 x 1/62 = **0.004839**
- c4: exact rank 3 only: R = 0.7 x 1/63 = **0.011111**

Being found by **both** searches (c1) gives the highest score. For a section question, being found by exact match only (c4) beats being found by meaning only (c3).

### 6.5 Authority score (report equation 5)

```
A(c) = 1                                          if c is a statute or Constitution chunk
A(c) = 0.4 x w_court + 0.3 x w_rec + 0.3 x w_cit   otherwise (judgments)
```

| Part | Values |
|---|---|
| w_court | Supreme Court 1.0, High Court 0.7, tribunal 0.4, unknown 0.3 |
| w_rec (recency) | decided at most 2 years ago 1.0; 5 years 0.8; 10 years 0.6; 20 years 0.4; older or no date 0.2 |
| w_cit | min(number of citations / 100, 1); 0.5 if unknown |

Code: `src/retrieval/reranker.py:compute_authority_score`, `_recency_weight`, `_normalise_court`; weights in `config/settings.py`.

**Why?** Legal sources do not have equal weight. Statute text binds everyone. A Supreme Court judgment binds all lower courts. A tribunal order is persuasive at most.

**Examples.**
- Statute or Constitution chunk: **1.0**.
- Supreme Court chunk with no date and no citation count (most of our judgments): 0.4 x 1.0 + 0.3 x 0.2 + 0.3 x 0.5 = 0.40 + 0.06 + 0.15 = **0.61**.
- **(example)** High Court judgment from 2019 (7 years old in 2026, so the 10-year bracket: 0.6) with 40 citations (0.4): 0.4 x 0.7 + 0.3 x 0.6 + 0.3 x 0.4 = 0.28 + 0.18 + 0.12 = **0.58**.
- A Q&A pair whose type was guessed as "judgement" and has no court: 0.4 x 0.3 + 0.06 + 0.15 = **0.33**.

**Honest note.** The citation count is not stored in ChromaDB (`vectorstore.py:_METADATA_FIELDS` does not include it), so w_cit is always 0.5 in practice, and most dates are missing. So almost every Supreme Court chunk gets 0.61, and every statute or Constitution chunk, even a Q&A pair, gets 1.0.

### 6.6 Rescaled RRF

```
R~(c) = min( R(c) x (k + 1), 1 )        = min( R(c) x 61, 1 )
```
- Code: `reranker.py` (`rrf_norm = min(rrf_score / _MAX_SINGLE_LIST_RRF, 1.0)`, where `_MAX_SINGLE_LIST_RRF = 1/61`).

**Why?** The largest possible R is about 1/61 = 0.0164. The other two signals (s and A) are between 0 and 1, about 60 times bigger. Without rescaling, RRF had almost no effect on the final score (this was a bug found after the report). Multiplying by 61 makes "rank 1 in a list" worth 1.0.

**Example.** c1: 0.016208 x 61 = **0.989**. c3: 0.004839 x 61 = **0.295**. c4: 0.011111 x 61 = **0.678**.

### 6.7 Final score S

```
S(c) = w_s x s(q, c) + w_r x R~(c) + w_a x A(c)        with (w_s, w_r, w_a) = (0.50, 0.25, 0.25)
```
- Code: `AuthorityReranker.rerank`. The top K = 4 by S go to the LLM.

**Example.** c1: 0.50 x 0.71 + 0.25 x 0.989 + 0.25 x 1.00 = 0.355 + 0.247 + 0.250 = **0.852**.
c2: 0.50 x 0.62 + 0.25 x 0.986 + 0.25 x 0.61 = 0.310 + 0.247 + 0.153 = **0.709**.
The full table of six chunks is in Section 3, Stage 9.

**Why these weights?** Meaning (s) gets half. Fusion and authority get a quarter each. Table VIII shows that (0.5, 0.5, 0) gives slightly higher keyword precision (0.842 vs 0.811). We kept authority on purpose, so that binding sources come first; this is a design choice, not a metric result.

### 6.8 Confidence score C and its levels

```
C = 0.40 x Q + 0.25 x G + 0.20 x H + 0.15 x V
Q = clip( (S_mean - 0.3) / 0.6 , 0, 1 )
```

| Signal | Name in code | Meaning |
|---|---|---|
| Q | retrieval_quality | Average final score S of the retrieved chunks, mapped from the range 0.3–0.9 to 0–1 ("clip" = cut off below 0 and above 1). |
| G | source_agreement | How many **distinct** sources the answer cites: 0 -> 0, 1 -> 0.3, 2 -> 0.6, 3 or more -> 1.0. |
| H | authority_strength | Average authority A of the cited retrieved chunks (of all retrieved chunks if nothing is cited). |
| V | coverage_score | Share of the retrieved chunks that the answer cites. |

| C | Level |
|---|---|
| 0.80 or more | High |
| 0.60 to 0.79 | Medium |
| 0.40 to 0.59 | Low |
| below 0.40 | Very Low |

Code: `src/confidence/scorer.py`. Q, H and V use only the retrieved chunks; the IPC/BNS table counts for G (it is a cited source) but not for the others (`pipeline.py` passes only the retrieved chunks).

**Example 1 (the answer from Section 3).**
- S of the 4 chunks: 0.852, 0.709, 0.664, 0.643 -> mean 0.717 -> Q = (0.717 - 0.3) / 0.6 = **0.695**
- Cited: SOURCE_1, 2, 3 and 5 (table) = 4 distinct -> G = **1.0**
- Cited retrieved chunks c1 (1.0), c2 (0.61), c3 (1.0) -> H = 2.61 / 3 = **0.87**
- 3 of the 4 retrieved chunks cited -> V = **0.75**
- C = 0.40 x 0.695 + 0.25 x 1.0 + 0.20 x 0.87 + 0.15 x 0.75 = 0.278 + 0.250 + 0.174 + 0.113 = **0.814 -> High**

**Example 2: same retrieval, but the model refuses and cites nothing.**
- Q is still 0.695. G = 0. V = 0. H falls back to all 4 retrieved chunks: (1.0 + 0.61 + 1.0 + 1.0) / 4 = 0.90.
- C = 0.278 + 0 + 0.180 + 0 = **0.458 -> Low**

So a refusal or an uncited answer cannot get a high confidence, even with good retrieval.

**What C is not.** It is not a probability that the answer is correct. It is a summary of how strong the evidence and the citing look.

---

## 7. Generation and the anti-hallucination layers

"Generation" is the step where the LLM writes the answer. Our idea: **the LLM is only a writer, not a source of law.** All legal facts must come from the numbered sources.

### 7.1 The prompt rules (`config/prompts.py: SYSTEM_PROMPT`)

The system prompt tells the model:

1. Use only information in the sources labelled `[SOURCE_1]`, `[SOURCE_2]`, ...
2. Put a citation like `[SOURCE_2]` right after **every** factual statement. Cite each source by its own number; never write `[SOURCE_N]` literally.
3. **Never invent** case names, citations (AIR, SCC, SCR), section numbers or quotations. If a detail is not in the sources, leave it out.
4. When citing a statute, give the section number and the name of the Act.
5. If one source is the IPC/BNS table, state **both** the IPC section and the BNS section.
6. If sources disagree, say so and explain which has more authority (Supreme Court over High Court, statute text over commentary).
7. If the sources answer only part of the question, answer that part and end with one sentence starting "Not covered by the retrieved sources:". Only if **no** source is relevant, reply exactly "The retrieved sources do not address this question."
8. Output only the answer: no reasoning notes or "thinking" sections.

Format: a 1–2 sentence direct answer, then 2–4 paragraphs with every statement cited, then a short summary.

Rule 7 is new since Run A. The old prompt asked the model to refuse whenever coverage was incomplete, so many partly answerable questions came back as refusals (Run A response rate 0.30). The new rule more than doubled the response rate (0.68 in Run C).

### 7.2 How each source looks in the prompt

`SOURCE_FORMAT_TEMPLATE` gives each passage a number, its type and one metadata line (`prompt_builder.py:source_label`):

| Type | Metadata line |
|---|---|
| judgement | `Case: ... \| Citation: ... \| Court: ... \| Date: ...` (only the fields that exist) |
| statute | `Act: ... \| Section: ...` |
| constitution | `Article: ...` |
| mapping (IPC/BNS table) | `Correspondence table: Indian Penal Code, 1860 ... / Bharatiya Nyaya Sanhita, 2023 ...` |

`flatten_chunk` lifts the metadata to the top level first. (A bug fixed after the report: before, case names and courts never reached the model.)

### 7.3 The `[SOURCE_N]` tags and citation validation

- **Code:** `src/generation/response_parser.py`.
- The answer is searched with the pattern `\[SOURCE_(\d+)\]`.
- Grouped tags `[SOURCE_2, SOURCE_4]` or `[SOURCE_1; SOURCE_3]` are first rewritten as separate tags (`expand_grouped_citations`), because models sometimes group them.
- A tag is **valid** if its number N is between 1 and the number of sources sent.
- Output: `cited_sources` (each valid source once, with its text and metadata), `invalid_citations` (for example `[SOURCE_7]` when only 5 were sent), and `citation_validity`.

**Example.** 5 sources were sent. The answer has `[SOURCE_1]`, `[SOURCE_2]`, `[SOURCE_1]`, `[SOURCE_7]`. Valid: 3 of 4 tags -> validity 0.75. `[SOURCE_7]` is shown to the user as "Citations that point to no source". Because each tag maps back to a stored passage, a reader can click/scroll to the exact text behind each claim.

### 7.4 The IPC/BNS table as a source

- **Code:** `src/retrieval/ipc_bns.py`; table `config/ipc_bns_mapping.json` (63 rows).
- `find` looks for IPC or BNS references in both orders and with the full names ("Indian Penal Code", "Bharatiya Nyaya Sanhita"). A bare "Section 302" counts only if the question names exactly one of the two Acts. If both Acts are named but no number is given ("Compare IPC and BNS provisions on murder"), it matches by offence name in the titles: here rows IPC 300, 302 and 304.
- It works both ways: BNS 103 -> IPC 302 too.
- `expand_query` adds the counterpart section to the search, so both texts can be found.
- `as_source_chunk` turns the matched rows into one extra source, always the last `[SOURCE_N]`.

**Why a citable source and not just memory?** LLMs trained before 2024 may not know BNS numbers, or may mix them up. With the table in the prompt, the model copies the number from a source and cites it, and the user can check it. Nine wrong rows were found and corrected (for example IPC 376 now maps to BNS 64, not 65; IPC 364 to BNS 140(1), not 139).

### 7.5 All the layers together

| # | Layer | Where | What it stops |
|---|---|---|---|
| 1 | Retrieval grounding | `pipeline.py` | The model only sees retrieved passages, not "everything it remembers". |
| 2 | Strict prompt | `config/prompts.py` | Asks for a citation after each claim and forbids inventing names, citations and numbers. |
| 3 | Low temperature 0.1 (Nyaya-Gen: greedy decoding) | `llm_client.py`, `local_generator.py` | Less random, more repeatable wording. |
| 4 | IPC/BNS table as a source | `ipc_bns.py` | Wrong or invented BNS numbers. |
| 5 | Removing "thinking" text | `llm_client.py:_strip_model_thinking` | Reasoning notes being shown as the answer. |
| 6 | Citation validation | `response_parser.py` | Tags that point to nothing are flagged. |
| 7 | Confidence score | `scorer.py` | Weak or uncited answers are shown as Low / Very Low. |
| 8 | Measurement | `citation_metrics.py` | Invented case names or citations are counted in the evaluation (hallucination rate), and citation support is measured. |
| 9 | A writer trained only to restate and cite (Nyaya-Gen, results pending) | `local_generator.py` | In fine-tuning it was only ever rewarded for copying from a numbered source with its tag, or for refusing when no source fits. (It can still copy the wrong passage; layers 6–8 check it like any LLM.) |

**What it does not do.** At answer time, nothing checks that a cited passage really **supports** the sentence; the citation support metric does this only roughly, during evaluation. A wrong statement with a real, valid tag is still possible.

---

## 8. Evaluation

### 8.1 The 50-query test set (Table II)

No public benchmark covers cited answers over Indian statutes, the Constitution and the BNS, so we made 50 questions by hand (`data/evaluation/test_queries.json`). Each has a category, the expected document types and expected keywords. Example:

```json
{ "query": "What is Section 302 IPC?",
  "category": "section_lookup",
  "expected_doc_types": ["statute"],
  "expected_keywords": ["Section 302", "IPC", "murder", "punishment"] }
```

| Category | Queries | Example |
|---|---|---|
| Section lookup | 15 | What is Section 302 IPC? |
| Case lookup | 10 | Summarize the case of Navtej Singh Johar v Union of India |
| Conceptual | 10 | Explain the doctrine of basic structure in Indian constitutional law |
| Cross reference | 10 | Compare IPC and BNS on murder provisions |
| Edge case | 5 | "What is Section 99999 of IPC?", an empty query, Lorem ipsum, Hinglish ("Dhara 302 ke tahat sazaa kya hai?"), symbols around "section 302" |

Retrieval tables use the **49 non-empty** queries. End-to-end tables use all **50**.

### 8.2 Retrieval metrics (no LLM needed)

Code: `src/evaluation/retrieval_metrics.py`. D_K = the top K chunks given to the LLM. Keyword matching ignores upper/lower case.

| Metric | Meaning | Formula |
|---|---|---|
| **P@K** (retrieval precision, report eq. 8) | Share of the K chunks that are of an expected type **or** contain an expected keyword. | (1/K) x number of such chunks |
| **P_kw@K** (keyword precision) | Stricter: share of the K chunks that contain an expected keyword. | (1/K) x number of keyword chunks |
| **Hit@K** | 1 if at least one of the K chunks contains a keyword, else 0. Averaged over queries. | |
| **MRR** (mean reciprocal rank) | For each query, 1 / rank of the first keyword chunk (0 if none). Then the average. | (1/\|Q\|) x sum of 1/rank_q |

**Mini example for one query** "What is Section 302 IPC?" with K = 4:

| Rank | Chunk | Type statute? | Has a keyword? |
|---|---|---|---|
| 1 | Q&A about Section 302 IPC | yes | yes ("Section 302") |
| 2 | Judgment: "convicted for murder" | no | yes ("murder") |
| 3 | Statute chunk on Section 299, culpable homicide, no keyword | yes | no |
| 4 | Judgment quoting "Section 302" | no | yes |

- P@4 = 4/4 = **1.00** (every chunk is a statute or has a keyword)
- P_kw@4 = 3/4 = **0.75** (rank 3 has no keyword)
- Hit@4 = **1** (at least one keyword chunk)
- Reciprocal rank = 1/1 = **1.0**

**MRR over 3 queries (example):** first keyword chunk at rank 1, rank 2, and none -> MRR = (1 + 0.5 + 0) / 3 = **0.50**.

**Why both P@4 and P_kw@4?** The type condition is easy to meet (many chunks are "statute"), so P@4 is high for every method. P_kw@4 shows the real differences.

**Precision vs recall.** Precision = "of what I returned, how much is relevant?" Recall = "of everything relevant that exists, how much did I return?" Example: there are 10 relevant chunks in the index; we return 4 and 3 of them are relevant. Precision = 3/4 = 0.75; recall = 3/10 = 0.30. We cannot measure true recall, because we do not know every relevant chunk among 112,555. Hit@K is a simple recall-like measure ("did we find at least one?").

### 8.3 Answer metrics (end to end)

Code: `src/evaluation/citation_metrics.py`, run by `scripts/evaluate.py`. All are rule-based (deterministic: the same answer always gives the same score). No LLM judges another LLM.

| Metric | Meaning | Mini example |
|---|---|---|
| **Citation validity** | Share of `[SOURCE_N]` tags with 1 <= N <= n (n = sources sent). An answer with **no** tag scores 0. Called "citation accuracy" in Run A. | Tags 1, 2, 1, 7 with n = 5 -> 3/4 = 0.75. |
| **Cited claim coverage** | Share of "legal claim sentences" that carry a citation. A claim sentence is at least 25 characters and mentions something legal (section, article, Act, court, case, murder, bail, punishment, ...). | 4 claim sentences, 3 with a tag -> 0.75. |
| **Citation support** | Share of cited claims where the cited source (a) contains the same section/article numbers and Act names, (b) contains the same legal concepts (murder, bail, punishment...), and (c) shares at least 35% of the claim's content words, or at least 5 of them. | "Section 302 IPC punishes murder with imprisonment for life [SOURCE_1]" is supported only if SOURCE_1 contains "section 302", "ipc", "murder", "punish", "imprisonment" and enough of the other words. |
| **Hallucination rate** | Share of answers that name a case ("X v Y") or an AIR/SCC/SCR citation found in **none** of the sources. A list of 16 famous cases (Kesavananda Bharati, Maneka Gandhi, ...) is not flagged, because users ask about them by name. | 2 of 50 answers flagged -> 0.04. |
| **Response rate** | Share of answers that cite at least one valid source and are not a refusal ("retrieved sources do not address this question", "cannot answer", ...). A partial answer with "Not covered by the retrieved sources: ..." still counts. | 34 of 50 -> 0.68. |
| **Average confidence** | Mean of C over all answers. | |
| **Latency** | Seconds from question to final answer, averaged over the 49 non-empty queries (`summary.py:mean_latency`). | |
| Citation quality (extra) | 0.30 x validity + 0.30 x coverage + 0.40 x support; printed by `evaluate.py`, not in the report tables. | |

### 8.4 Baselines and ablation (Table III)

A **baseline** is a simpler method we compare against. An **ablation** removes one part of our method to see how much that part adds. All rows use the same index, the same MiniLM model, the same IPC/BNS expansion, duplicate removal, and K = 4, on 49 queries.

| Method | P@4 | P_kw@4 | Hit@4 | MRR |
|---|---|---|---|---|
| (a) Dense only (standard RAG) | 0.867 | 0.656 | 0.796 | 0.719 |
| (b) Exact match only | 0.714 | 0.694 | 0.694 | 0.694 |
| (c) Hybrid, fixed RRF (0.5/0.5) | 0.913 | 0.776 | 0.878 | 0.837 |
| (d) Hybrid, adaptive RRF | 0.939 | 0.842 | 0.878 | 0.878 |
| (e) (d) + re-rank, no authority | 0.939 | 0.842 | 0.878 | 0.878 |
| (f) **Proposed** = (e) + authority | 0.934 | 0.811 | 0.878 | 0.878 |

**What it shows.**
- **Hybrid beats either path alone.** Keyword precision goes from 0.656 (dense only) to 0.811 (proposed), +15.5 points; Hit@4 from 0.796 to 0.878 (39 to 43 of 49 queries).
- **The two paths fail in opposite places.** Dense search is good on conceptual and cross-reference questions (keyword precision 0.925 and 0.975) but collapses on case lookups (0.225): the vector for "Maneka Gandhi v Union of India" is closer to any passage about personal liberty than to the judgment itself. Exact match is perfect on section lookups (1.000) but almost useless on conceptual questions (0.100), which contain no identifiers.
- **Adaptive weights help:** +6.6 points keyword precision and +4.1 points MRR over fixed 0.5/0.5, mostly on case lookups (0.475 -> 0.600) and section lookups (0.883 -> 1.000).
- **Duplicate removal** does not change precision, but lifts dense Hit@4 from 0.776 to 0.796, because 19.4% of the dense top-4 slots were copies of passages already shown.
- **The authority prior costs 3.1 points** of keyword precision (f vs e) and does not change Hit@4 or MRR. It pushes statute and Constitution chunks (authority 1.0) above judgments with missing metadata (0.61), which sometimes replaces a relevant judgment paragraph with a less specific Constitution passage (see c6 vs c4 in Section 3). We keep it because it orders conflicting sources by binding force, which this metric cannot see. It should be re-tuned once judgment metadata is more complete.

### 8.5 End-to-end runs (Tables IV and V)

| Metric | Run A (DeepSeek, old prompt, K = 6) | Run B (Nemotron) | Run C (Nemotron, final) | Run D0 (base SmolLM2-135M, not fine-tuned, local) | Run D (Nyaya-Gen, fully local) |
|---|---|---|---|---|---|
| Citation validity | 0.94 | 0.66 | 0.70 | not run (time) | 0.74 |
| Cited claim coverage | – | 0.42 | 0.42 | not run (time) | 0.66 |
| Citation support | – | 0.50 | 0.50 | not run (time) | 0.56 |
| Hallucination rate (lower is better) | 0.02 | 0.02 | 0.04 | not run (time) | 0.00 |
| Response rate | 0.30 | 0.66 | 0.68 | not run (time) | 0.72 |
| Retrieval precision | 0.79 | 0.91 | 0.92 | not run (time) | 0.92 |
| Average confidence | 0.69 | 0.57 | 0.61 | not run (time) | 0.45 |
| Mean latency (s) | 25.0 | 15.4 | 17.5 | not run (time) | 45.0 |

Run C = Run B + case names and nearest-match exact search. Nemotron = `nvidia/nemotron-3-ultra-550b-a55b` on NVIDIA NIM. Run A's retrieval precision (0.79) is P@6, because Run A used 6 passages, so it is not directly comparable with the P@4 of Runs B and C. Latency is the mean over the 49 non-empty queries.

**The new columns.** Run D is made by `scripts/run_evaluation.sh` and runs with no internet, using Nyaya-Gen. D vs C = what we lose by writing with a 135M model on the laptop instead of a ~550B model online; D uses the same retrieval as Run C, so their retrieval precision should match. Run D0 (the same small model *before* fine-tuning) is in the script but was **not run** for the submission because of time; if the teacher asks, say it is the natural next experiment. See Section 9.9.

**Why did validity fall from 0.94 to 0.70 while the system got better?** Validity is averaged over **all** answers, and an answer with no tag scores 0. Run A's high validity came mostly from refusals that still carried a valid tag: 33 of its 47 fully valid answers were refusals, and 35 of its 50 answers were not substantive. In Run C, 35 answers have validity 1.0 and 15 have no tag at all (0), which gives 0.70. The new prompt more than doubled the response rate (0.30 -> 0.68) while hallucinations stayed at 1–2 answers in 50. So validity now moves with the response rate.

**Latency.** Run A used 6 passages; Runs B and C use 4, so the prompt is shorter. The model also changed, so not all of the drop is due to K.

**Table V: Run C by category**

| Category | Validity | Support | Response | Retrieval P | Confidence |
|---|---|---|---|---|---|
| Section lookup (15) | 0.93 | 0.64 | 0.93 | 0.98 | 0.78 |
| Case lookup (10) | 0.50 | 0.32 | 0.50 | 1.00 | 0.46 |
| Conceptual (10) | 0.90 | 0.80 | 0.80 | 1.00 | 0.74 |
| Cross reference (10) | 0.60 | 0.35 | 0.60 | 1.00 | 0.49 |
| Edge case (5) | 0.20 | 0.10 | 0.20 | 0.20 | 0.35 |

- Section lookups and conceptual questions work well.
- **Case lookups are weakest.** The corpus often has only fragments of a named judgment, without its case name, so the model correctly declines to say what "X v Y" held.
- **Cross-reference questions** suffer from missing source texts: the BNSS and the BNS provisions on organised crime and sedition are not in the default corpus, and the model says so.
- **Edge cases behave as intended:** the empty query is rejected, and "Section 99999" gets a refusal instead of an invented provision.
- **The 2 flagged hallucinations** in Run C: one answer wrote out its reasoning ("The user is asking what was held in ..."), and the other contained a phrase like "Federal Court vs Privy Council" from legal history. Neither is an invented case, but the detector cannot tell.

**Comparison with commercial tools.** Magesh et al. found 17–33% hallucination in commercial legal AI tools, but on US law with expert checking. Our 4% is measured by a pattern detector that misses wrong statements attributed to real sources, so the numbers are not directly comparable, and the true error rate is higher. Citation support (0.50) is the more careful indicator.

### 8.6 Parameter sweeps (Tables VI–VIII)

Each setting was changed **one at a time** around the deployed configuration, on cached search results for the 49 queries.

**Table VI: number of passages K given to the LLM**

| K | P@K | P_kw@K | Hit@K | MRR |
|---|---|---|---|---|
| 2 | 0.939 | 0.867 | 0.878 | 0.878 |
| **4** | 0.934 | 0.811 | 0.878 | 0.878 |
| 6 | 0.932 | 0.776 | 0.898 | 0.882 |
| 8 | 0.929 | 0.758 | 0.898 | 0.882 |
| 10 | 0.931 | 0.754 | 0.898 | 0.882 |

Smaller K = purer context but fewer distinct sources (and the confidence score needs 3 cited sources for full source agreement). Beyond K = 6 there are no new hits; the prompt just gets longer and the relevant passage sits in the middle, where LLMs use information less well (Liu et al., "lost in the middle"). **K = 4** is the smallest value that allows multi-source answers while keeping keyword precision above 0.8.

**Table VII: RRF constant k and candidate pool size**

| RRF k | P_kw@4 | Hit@4 | | Pool per path | P_kw@4 | MRR |
|---|---|---|---|---|---|---|
| 1 | 0.750 | 0.878 | | 5 | 0.815 | 0.878 |
| 10 | 0.791 | 0.878 | | 10 | 0.801 | 0.878 |
| 30 | 0.801 | 0.878 | | **20** | 0.811 | 0.878 |
| **60** | 0.811 | 0.878 | | 30 | 0.806 | 0.867 |
| 100 | 0.816 | 0.878 | | 40 | 0.806 | 0.867 |

A very small k hurts (rank 1 of each list dominates). k from 30 to 100 is within 1.5 points. The pool size matters little; above 20 it lets in more weak candidates and MRR falls slightly, so 20 was kept.

**Table VIII: final score weights (left) and fixed vs adaptive fusion (right)**

| (w_s, w_r, w_a) | P_kw | Hit | | Fusion (alpha, beta) | P_kw | Hit |
|---|---|---|---|---|---|---|
| (1.0, 0, 0) | 0.668 | 0.837 | | fixed (1.0, 0.0) | 0.597 | 0.755 |
| (0.7, 0.3, 0) | 0.827 | 0.878 | | fixed (0.8, 0.2) | 0.643 | 0.796 |
| (0.5, 0.5, 0) | 0.842 | 0.878 | | fixed (0.5, 0.5) | 0.719 | 0.837 |
| (0.6, 0.2, 0.2) | 0.786 | 0.878 | | fixed (0.2, 0.8) | 0.821 | 0.878 |
| (0.4, 0.4, 0.2) | 0.827 | 0.878 | | fixed (0.0, 1.0) | 0.827 | 0.878 |
| (0.34, 0.33, 0.33) | 0.827 | 0.878 | | | | |
| **(0.5, 0.25, 0.25)** | 0.811 | 0.878 | | **adaptive** | 0.811 | 0.878 |

- Ranking by semantic similarity alone loses the benefit of fusion (0.668). Any real RRF weight brings it back.
- The best keyword precision is 0.842 at (0.5, 0.5, 0); our (0.5, 0.25, 0.25) gives up 3.1 points to prefer binding sources.
- On the right: giving the exact-match path more weight helps, because most test queries contain identifiers. To be honest, the keyword-heavy fixed settings (0.2, 0.8) and (0.0, 1.0) are slightly higher on this test set (0.821, 0.827) than adaptive (0.811). Adaptive keeps a high semantic weight for conceptual queries, which have no identifiers, so it is the safer choice for real users.

**Important caveat.** There was no separate validation set: these settings were chosen on the **same** 50 test queries. So Tables VI–VIII are a **sensitivity analysis** (how much results change), not an unbiased estimate for new questions. Nyaya-Gen avoids this for its own choices: its checkpoint is chosen on a separate dev split, and training questions similar to the test queries were removed (Section 9.3).

### 8.7 Where the numbers come from

| Numbers | Produced by |
|---|---|
| Tables III, VI, VII, VIII | `scripts/retrieval_experiments.py` (new), using `src/evaluation/retrieval_metrics.py` and the deployed ranking code |
| Tables IV, V | `scripts/evaluate.py` per run, summarised by `src/evaluation/summary.py`; `python scripts/evaluate.py --summarise <file>` prints them for a saved run without the index or LLM |
| Stored runs | `data/evaluation/eval_results*.json` (Section 4.6) |

---

## 9. The trained model: Nyaya-Gen

The report version of Nyaya had **no trainable parts**: MiniLM and the LLM were used as they are ("frozen"). We now add one model that **we train ourselves**, on our own laptop. Nyaya-Gen is the model that **writes the answer**. It replaces the online LLM (NVIDIA NIM), so with Nyaya-Gen the whole system runs on the laptop with no internet and no API key. Design: `docs/MODEL_PLAN.md`. Code: `scripts/prepare_generator_data.py` (data), `scripts/train_generator.py` (training), `src/generation/local_generator.py` (use at query time), `scripts/run_training.sh` and `scripts/run_evaluation.sh` (the long runs).

### 9.1 What it is, and why a 135M model can do this job

- **Base model:** `HuggingFaceTB/SmolLM2-135M-Instruct` (Allal et al. 2025). A small **decoder-only** ("GPT-style") language model: 30 layers, hidden size 576, 9 attention heads, a vocabulary of 49,152 tokens, **134.5 million parameters**. "Instruct" means its makers already fine-tuned it to follow chat instructions.
- **What we do:** **full fine-tuning**: every one of the 134.5M weights is trained (no adapters, nothing frozen), on examples of our own prompt with a correct cited answer.
- **Why we did it:** we wanted the answer writer itself to be a model we trained, and we wanted the whole system to run offline.

**Why it can work.** Nyaya-Gen is **not asked to know the law**. A 135M model knows very little about Indian law and would invent things if asked from memory. But in our pipeline every legal fact is already in the prompt, in the numbered sources. The model's job is **grounded generation**: find the source that answers the question, restate it in a few sentences, and put that source's tag `[SOURCE_k]` at the end of each sentence; if no source fits, say "The retrieved sources do not address this question." That is mostly *selecting and copying*, a pattern that a small model can learn from a few thousand examples.

**What it still does worse than a 550B model (say this honestly):**
- Its answers are **shorter and plainer**: a few sentences restating one source.
- It was only trained to restate **one** source per answer, so it does not combine several sources, compare them by authority (prompt rule 6) or write the "Not covered by the retrieved sources:" sentence (rule 7).
- It reads only the **first 90 words** of each source.
- A small model can still **copy the wrong passage** or garble a sentence. The same citation validation, confidence score and metrics check it, exactly as they check the online LLM.

So our claim is **not** "a 135M model is as good as a 550B model". It is: "a model we trained on an 8 GB laptop can write cited answers from retrieved sources, and here is, measured on the same 50 questions and six metrics, how far it is from the 550B model."

### 9.2 What a causal language model is: next-token prediction

A **causal language model** writes text one **token** at a time. At each step it looks at all the tokens so far (the prompt plus what it has already written) and gives a probability for **every** token in its vocabulary (49,152 of them) being the next one. One token is picked, added to the text, and the step repeats. "Causal" means it can only look backwards, never at future tokens.

**(example)** The prompt ends, and the model has written "Theft is punished with". For the next token it might give: "imprisonment" 0.62, "a" 0.15, "fine" 0.08, ... (all 49,152 probabilities add up to 1). Greedy decoding picks "imprisonment".

**Training** this kind of model means: show it a correct text and, at every position, make the probability of the token that really comes next as high as possible. The loss at one position is -ln(probability of the correct next token): 0 if the model was sure and right, large if it gave the right token a small probability. For "imprisonment" above: -ln 0.62 = **0.478**.

### 9.3 How the training data is built

`scripts/prepare_generator_data.py`. **No new data**, no hand labelling, and the vector index is not needed. It uses the statute Q&A (`viber1/indian-law-dataset`) and Constitution Q&A (`nisaar/Constitution_of_India`) records, 23,939 in all, processed **exactly as they are indexed** (it reuses `prepare_hf_data.py`).

**Every example = one prompt in exactly the pipeline's format + one target answer.** The prompt is made by the real `PromptBuilder` (system rules + numbered sources + question), so the model is trained on precisely what it will see at query time.

**(example, shortened)**

```
SYSTEM : the 8 rules of Section 7.1
USER   : SOURCES:
         [SOURCE_1] Type: judgement ... (a random judgment passage, cut to 90 words)
         [SOURCE_2] Type: statute ... Legal Question: What is the punishment for theft?
                    Legal Answer: Section 379 of the IPC provides ... (the record itself)
         [SOURCE_3] Type: statute ... (another random Q&A record)
         QUESTION: What is the punishment for theft?
TARGET : Section 379 of the IPC provides that theft is punishable with imprisonment of
         up to three years, or with a fine, or with both [SOURCE_2]. ...
```

**Three kinds of example:**

| Kind | What the prompt has | What the target answer is | Why |
|---|---|---|---|
| `qa` | The record's question; the record hidden at a **random position** among 2–4 random distractor passages | Up to 4 sentences (and 120 words) of the record's answer, each ending with `[SOURCE_k]`, k = the record's position | Teaches "find the right source and cite its number". The random position stops it from always writing `[SOURCE_1]`. |
| `mapping` | An IPC section question from 3 templates ("What is the BNS equivalent of Section 420 IPC?" ...), with the IPC/BNS table chunk among the sources | A fixed sentence giving both the IPC and BNS sections and titles, cited, plus "The BNS replaced the IPC from 1 July 2024" | Teaches prompt rule 5 (always state both sections). |
| `refusal` (~7%) | The question, but its own record is **left out** | "The retrieved sources do not address this question." | Teaches rule 7: refuse when nothing fits, instead of inventing. |

Every passage is cut to **90 words**, so that a prompt with up to 5 sources fits in memory during training.

**No leakage into the test set.** A question is removed if its normalised text equals one of the 50 test queries, or if its **content-word Jaccard similarity** to one is 0.5 or more. Content words = the words left after removing small common words (the, of, what, is, under, law, ...). Jaccard = (words in both) / (words in either). **(example)** "What is the punishment for murder under Section 302 IPC?" -> {punishment, murder, section, 302, ipc}; test query "What is Section 302 IPC?" -> {section, 302, ipc}; Jaccard = 3 / 5 = 0.6 >= 0.5, so it is removed. **572 questions were removed.** IPC/BNS table rows whose IPC section number appears in a test query are skipped too.

**Splits by document: 94% train, 3% dev, 3% test.** Training is capped at 4,000 examples (all mapping examples are kept). Real counts (`data/generator/data_stats.json`):

| Split | qa | mapping | refusal | Total |
|---|---|---|---|---|
| train | 3,590 | 141 | 269 | 4,000 |
| dev | 179 | 0 | 21 | 200 |
| test | 181 | 0 | 19 | 200 (held out, not used yet) |

At training time every example longer than **1,024 tokens** is dropped (Section 9.6), which leaves **2,585** training examples.

### 9.4 The loss: next-token cross-entropy on the answer tokens only

For one example, the prompt x and the answer y are joined into one token sequence: chat template (system, user) + answer + the end-of-turn token `<|im_end|>`. The **labels** (the "correct next token" at each position) are the tokens themselves, except that every **prompt** position gets the label **-100**, which PyTorch's cross-entropy ignores (`train_generator.py:encode`).

So the loss of one example is the average of -ln p(correct next token) over the **answer tokens only** (report equation (13) in `docs/REPORT_ADDITIONS.md`):

J_G = average over answer tokens of -ln p_theta(y_t | prompt, y_1 ... y_(t-1))

**Worked example (example).** An answer of 4 tokens; the model gives the correct tokens probabilities 0.9, 0.6, 0.8 and 0.5.
- -ln 0.9 = 0.105, -ln 0.6 = 0.511, -ln 0.8 = 0.223, -ln 0.5 = 0.693
- loss = (0.105 + 0.511 + 0.223 + 0.693) / 4 = **0.383**

A useful reading: e^(-loss) is the (geometric) average probability given to the correct token. Loss 0.383 -> 0.68; loss 0.8 -> 0.45.

**Why only the answer tokens are scored:**
1. **The prompt is most of the text.** The median example is about 930 tokens, and most of it is sources. If the prompt counted, most of the loss would be about predicting random distractor passages and the question, which is not the skill we want.
2. **We do not want it to learn to write sources or questions.** It should learn only "given these sources and this question, write this answer".
3. **The citation tags are in the answer**, so the loss directly rewards putting the right `[SOURCE_k]` in the right place.

The prompt still goes through the model (the answer tokens need it as context); it just is not a target.

### 9.5 The training loop, step by step

`scripts/train_generator.py`, a plain PyTorch loop (no training library), written to be read.

```
 encode every example; drop it if longer than 1,024 tokens        (2,585 of 4,000 kept)
 dev loss on the dev examples that fit (24 of the first 40)       (update 0 = the base model)
 for each training example, shuffled with seed 42:
   1. forward    ONE example -> loss on its answer tokens, divided by 8
   2. backward   add its gradients to the running sum
                 (gradient checkpointing: layer activations are recomputed here)
   every 8 examples (gradient accumulation):
   3. schedule   set the learning rate for this update
   4. clip       scale the total gradient down to norm 1.0 if it is bigger
   5. AdamW      update all 134.5M weights, then reset the gradients
   every 50 updates, and at the end:
   6. dev check  dev loss; if it is the lowest so far, save to models/nyaya-gen/
   stop after 1 epoch (323 updates) or after --max-minutes 75
```

**Learning rate.** Peak 1e-4 (0.0001), a usual value for a short run on a small model whose every weight is trained. **Warm-up:** over the first 20 updates it rises in a straight line: update 1 uses 5e-6, update 5 uses 2.5e-5 (exactly what `logs/training.log` shows), update 20 uses 1e-4. **Cosine decay:** afterwards it follows half a cosine wave down to 0 at update 323: about 5e-5 half way (around update 171), and very small near the end.

**AdamW.** Adam keeps, for each weight, a running average of recent gradients (the "momentum") and of their squares, and gives each weight its own step size: about learning rate x (average gradient) / sqrt(average squared gradient). The **W** is "decoupled weight decay" (0.01): every update also shrinks each weight a little, which discourages extreme weights and over-fitting. We use betas (0.9, 0.95): the running average of squared gradients uses 0.95 instead of the usual 0.999, so it adapts faster, which is common for training language models and keeps short runs stable. **Gradient clipping** at 1.0: if the total gradient is larger than 1.0 it is scaled down, so one bad example cannot cause a sudden jump.

**Checkpoint by dev loss.** Every 50 updates the dev loss is measured on the same 24 dev examples; the model is saved only when it is the lowest so far. If the model starts to over-fit, the earlier, better checkpoint is kept.

**Logs.** `data/generator/training_log.json`: the settings, the loss and learning rate of every update, every dev evaluation, the best dev loss and the minutes taken. `models/nyaya-gen/training_config.json` stores the settings with the model.

### 9.6 Why the 8 GB limit shaped every setting

The M2 Air has 8 GB of **unified memory** shared by macOS, the CPU and the GPU. This, not taste, decided most of the settings.

**Memory estimate (rough).** 134.5M weights x 4 bytes (fp32) = 0.54 GB. Training also needs a gradient per weight (0.54 GB) and two Adam values per weight (1.08 GB): about **2.2 GB** before any data. Then the **activations**: for every one of the 30 layers, the intermediate results for every token of a ~1,000-token sequence, plus the output scores over the 49,152-token vocabulary at every position (1,024 x 49,152 x 4 bytes = 0.2 GB for one copy alone). Without special care, the activations of one long example take several more GB.

**What happened (engineering finding).** Two earlier attempts ran **out of GPU memory** (`logs/training_failed_oom.log`): training with **bf16 autocast** (16-bit maths), and **fp32 without gradient checkpointing**. What fits is:

| Setting | What it saves |
|---|---|
| **Gradient checkpointing** | Normally, the forward pass stores every layer's intermediate results, because the backward pass needs them. With checkpointing, only each layer's input is stored; during the backward pass each layer's forward computation is simply **run again** to get its intermediate results. Memory falls from "all intermediates of 30 layers" to "the inputs of 30 layers + one layer's intermediates". The cost is about one extra forward pass, so training is slower (roughly 30%). |
| **1 example per forward pass + gradient accumulation 8** | Only one sequence's activations are in memory at a time. The gradients of 8 examples are added up before each update, so each update still "sees" 8 examples, a less noisy direction than 1 example would give. |
| **At most 1,024 tokens per example** | Activation memory grows with sequence length. This dropped 1,415 of the 4,000 training examples (the longest prompts), leaving 2,585. (An earlier attempt allowed 1,536 tokens.) |
| **Sources cut to 90 words** | Keeps most prompts under 1,024 tokens. The same cut is applied at query time, so training and use match. |
| **fp32 (no mixed precision)** | bf16 autocast did not fit and fp32 with checkpointing does; fp32 is also the safest for small updates (16-bit numbers can round tiny updates to zero). |

**The price: what actually happened.** Training is slow: about 0.8 minutes per update. The run completed about 80 of the planned 323 updates in about 65 minutes, and then the 8 GB Mac ran out of memory (the swap space was full) and the run stopped, before the 75-minute limit. Dev checks happen every 50 updates, so the last saved (and best) checkpoint is the one at **update 50**: the Nyaya-Gen we evaluate has seen only **50 updates x 8 = about 400 examples**, about 15% of one epoch. In that time:

| | Value |
|---|---|
| Dev loss before training (the base model) | 0.8325 |
| Dev loss at update 50 (best, saved) | 0.0334 |
| Training loss | 0.98 at update 1 -> about 0.005–0.02 at updates 50–80 |

Say this honestly in the viva: the training was limited by the hardware and time, not by choice. The loss fell very fast because the answer format is fixed and easy to learn; a low loss does not prove the answers are correct, which is why Run D on the 50 test questions is the real test.

### 9.7 All settings in one table

| Setting | Value | Simple reason |
|---|---|---|
| Base model | `HuggingFaceTB/SmolLM2-135M-Instruct` (134.5M, 30 layers, hidden 576) | Small enough to fully fine-tune in 8 GB; already follows chat instructions. |
| What is trained | All weights (full fine-tuning) | Simple; no extra methods to explain. |
| Input | Chat template: system rules + sources (90 words each) + question + answer + `<|im_end|>`, max 1,024 tokens | Exactly the pipeline's prompt; fits memory. |
| Loss | Next-token cross-entropy on answer tokens only (prompt labels -100) | Learn to write the cited answer, not to copy the prompt. |
| Batch | 1 example per pass, gradient accumulation 8 | Fits in 8 GB. |
| Optimiser | AdamW, lr 1e-4, betas (0.9, 0.95), weight decay 0.01 | Standard for small language models. |
| Schedule | 20 warm-up updates, then cosine decay to 0 | Safe start, gentle finish. |
| Epochs | planned 1 (2,585 examples = 323 updates); done: about 80 updates, kept update 50 | Stopped early by lack of memory (Section 9.6). |
| Gradient clipping | 1.0 | No sudden jumps. |
| Precision | fp32 with gradient checkpointing | The only setting that fitted (Section 9.6). |
| Dev check | Every 50 updates, on 24 dev examples; keep the lowest dev loss | Protects against over-fitting. |
| Time limit | `--max-minutes 75` (the run stopped at about 65 minutes, out of memory) | Keeps the run within one evening. |
| Seed | 42 | Reproducible order. |

### 9.8 How it plugs into the pipeline

- `config/settings.py`: `LLM_BACKEND = os.getenv("LLM_BACKEND", "api")`. Set `LLM_BACKEND=local` (in `.env` or in front of a command) to use Nyaya-Gen.
- `src/pipeline.py:LegalResearchPipeline.from_settings` then creates `LocalGenerator()` instead of `LLMClient()`. Both have the same `generate(prompt, system_prompt)` call, so nothing else in the pipeline changes.
- `LocalGenerator.generate`:
  1. puts the system prompt and the user prompt into SmolLM2's chat template;
  2. cuts every source passage to its first 90 words (`shorten_sources`), as in training;
  3. runs the model on MPS (or CPU) with **greedy decoding** (always the most probable next token; no randomness, so the same prompt always gives the same answer), a **repetition penalty of 1.1** (tokens that already appear in the text get their scores divided by 1.1, so the model is less likely to loop), and at most **256 new tokens** (training answers are at most 120 words);
  4. stops at `<|im_end|>` and returns the text.
- The answer then goes through the same citation validation and confidence score.
- The model folder is `models/nyaya-gen/`, or `NYAYA_GEN_PATH` if set. Run D0 uses `NYAYA_GEN_PATH=models/smollm2-base` to run the same code with the un-fine-tuned model. If the folder has no model, the pipeline logs a warning and returns only the sources.

### 9.9 How we evaluate it (results to be filled in)

**1. Dev loss** (from `data/generator/training_log.json`): before training, i.e. the base SmolLM2-135M-Instruct, 0.8325; best 0.0334 at update 50; training ran about 65 minutes (about 80 updates) before it stopped. This only shows that training worked on examples like the training ones.

**2. End to end on the 50 test questions** (Table IV, Section 8.5), all six answer metrics + latency, made by `scripts/run_evaluation.sh`:
- **Run D** = Nyaya-Gen, fully local -> `data/evaluation/eval_results_local.json` (step 1 of the script).
- **Run D0** = base SmolLM2-135M-Instruct (not fine-tuned) -> `data/evaluation/eval_results_local_base.json` (step 3). Not run for the submission because of time.

**How to read them:**
- **D0 vs D** = what our fine-tuning adds (same model size, same retrieval, same prompt). The base model has never seen our prompt format or the `[SOURCE_N]` tags, so we expect D to cite far more validly. (Not measured yet: D0 was not run.)
- **D vs C** = what we lose by writing with a 135M model on the laptop instead of a ~550B model online (about 4,000 times smaller), and what we gain: no internet, no API key, no per-request cost. Retrieval is the same in C, D0 and D, so the difference comes from the writer.
- **Retrieval precision** should be close to Run C's (same retrieval).
- **Only 50 updates.** If D is weak, part of the reason may be that training stopped at about 15% of the epoch.
- **The metrics do not see everything.** They do not reward combining sources or comparing their authority, which Nyaya-Gen was never trained to do, so the real gap to the 550B model may be bigger than Table IV shows.

**If the results are weak:** that is still a valid finding, and the report says so. The system keeps both writers: the API model for the best answers, Nyaya-Gen when it must run offline.

---

## 10. How to run everything

All commands are run from the project folder. The folder name has spaces, so use quotes: `cd "/Users/sanskarawasthi/RAG with model"`.

### 10.1 Setup (once)

```bash
# get the code (skip if you already have the folder)
git clone https://github.com/sanskar-94/legal-research-assistant.git
cd legal-research-assistant

# Python environment (Python 3.10 to 3.13 work)
python3.13 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then put your NVIDIA_API_KEY in .env
```

A free API key is available at build.nvidia.com. If `deepseek-ai/deepseek-v3.2` is not available on your key, set `LLM_MODEL` in `.env` to another chat model from the catalogue. Any OpenAI-compatible server works with `LLM_BASE_URL` and `LLM_MODEL`. Without a key, the assistant still returns the retrieved sources but cannot write an answer.

### 10.2 Build the knowledge base (about 15 minutes)

```bash
python scripts/download_hf_datasets.py   # 5 public Indian-law datasets from Hugging Face
python scripts/prepare_hf_data.py        # convert to the chunker's input format
python scripts/chunk_documents.py        # structure-aware chunking
python scripts/build_vectorstore.py      # embed and index into chroma_db/ (112,555 chunks)
```

Running `build_vectorstore.py` again is safe (same ids, updated in place). After changing the chunking, run `python scripts/build_vectorstore.py --reset`.

### 10.3 Ask questions

```bash
python scripts/ask.py "What is Section 302 of IPC and its BNS equivalent?"
python scripts/ask.py --json "What does Article 21 guarantee?"   # full result as JSON
python scripts/ask.py                                             # interactive; models load once
python scripts/ask.py -v "Define Section 498A of IPC"             # with pipeline logs
```

From Python:

```python
from src.pipeline import LegalResearchPipeline

pipeline = LegalResearchPipeline.from_settings()
result = pipeline.query("Explain the doctrine of basic structure")
print(result.answer_text)          # cites passages as [SOURCE_1], [SOURCE_2], ...
print(result.sources[0]["text"])   # [SOURCE_N] is result.sources[N - 1]
print(result.confidence.level)
```

### 10.4 Optional extra sources (full Acts and landmark judgments)

```bash
brew install tesseract                     # only needed for scanned PDFs (OCR)
python scripts/download_statutes.py        # IPC, BNS, CrPC, Constitution from India Code
python scripts/download_indiankanoon.py    # landmark Supreme Court judgments
python scripts/parse_pdfs.py               # parse them (and any PDFs you add to data/raw/)
python scripts/chunk_documents.py
python scripts/build_vectorstore.py
```

### 10.5 Another offline answer writer (LFM3-1.7B with llama.cpp, not trained by us)

Our own offline writer is Nyaya-Gen (Section 10.9). This older option runs a ready-made 1.7B model through llama.cpp:

```bash
brew install llama.cpp             # 0.8.0 or newer
mkdir -p ~/models && curl -L -o ~/models/LFM3-1.7B-Q4_K_M.gguf \
  https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/resolve/main/LFM3-1.7B-Q4_K_M.gguf
# in .env: LLM_BASE_URL=http://127.0.0.1:8080/v1 and LLM_MODEL=LFM3-1.7B
scripts/local_llm.sh start         # loads the model in the background (about 1.5 GB of memory)
python scripts/ask.py "What is Section 302 of IPC?"
scripts/local_llm.sh stop          # frees the memory
```

### 10.6 Tests

```bash
pytest tests/          # no API key or internet needed
```

### 10.7 Evaluation (Tables IV and V)

```bash
python scripts/evaluate.py                     # all 50 queries -> data/evaluation/eval_results_latest.json
python scripts/evaluate.py --limit 10          # only the first 10
python scripts/evaluate.py --output data/evaluation/my_run.json
python scripts/evaluate.py --summarise data/evaluation/eval_results_nemotron.json   # print Tables IV/V of a saved run, no LLM needed
```

### 10.8 Nyaya-Gen (data and training)

No index is needed, only the raw datasets from 10.2. The first run downloads SmolLM2-135M-Instruct (about 270 MB).

```bash
# 1. training data (a few minutes) -> data/generator/
python scripts/prepare_generator_data.py --max-train 4000

# 2. training -> models/nyaya-gen/   (the exact command run_training.sh uses)
python scripts/train_generator.py --max-steps 2 --limit 20 --out /tmp/ng   # optional smoke test
python scripts/train_generator.py --grad-accum 8 --eval-every 50 --eval-examples 40 --max-minutes 75
```

### 10.9 The long runs, and using the fully local system

Nyaya-Gen is trained, and then everything is evaluated, by two scripts that are safe to leave running (`caffeinate -i` stops the Mac from sleeping). Close other apps first: 8 GB is tight.

```bash
mkdir -p logs
# prepares the Nyaya-Gen data if needed, then trains Nyaya-Gen
nohup caffeinate -i scripts/run_training.sh > logs/training.log 2>&1 &
# start it after training has finished: 1) Run D (Nyaya-Gen), 2) retrieval tables III, VI-VIII,
# 3) Run D0 (base SmolLM2, not fine-tuned)
nohup caffeinate -i scripts/run_evaluation.sh > logs/evaluation.log 2>&1 &
tail -f logs/training.log          # watch progress (Ctrl+C stops watching, not training)
```

Ask questions with the fully local system (no internet, no API key):

```bash
LLM_BACKEND=local python scripts/ask.py "What is Section 302 of IPC and its BNS equivalent?"
# or put LLM_BACKEND=local in .env; LLM_BACKEND=api (the default) goes back to NVIDIA NIM
LLM_BACKEND=local NYAYA_GEN_PATH=models/smollm2-base python scripts/ask.py "..."   # the base model, as in Run D0
```

Do not run tests or other heavy Python while these scripts are running: they share the 8 GB.

---

## 11. Honest limitations

Saying these clearly in the viva shows that we understand the system.

**Evaluation**
1. **The test set is small (50 questions) and hand-made**, and there was no separate validation set, so Tables VI–VIII were tuned on the same queries. They are a sensitivity analysis, not an unbiased estimate. (Nyaya-Gen uses its own dev split to avoid this.)
2. **Relevance is judged by keywords, not by people.** A chunk containing "murder" counts as relevant even if it is about something else; a relevant chunk that uses other words counts as irrelevant.
3. **The hallucination detector only looks for invented case names and AIR/SCC/SCR citations.** It misses a wrong statement attributed to a real source, 16 famous case names are excused, and it only recognises citations shaped like "AIR 1978 SC 597", so a bracketed one like "(2017) 10 SCC 1" is not checked. So the true error rate is higher than 4%. Citation support (0.50) is a rough word-overlap check, not real understanding.
4. **One run per setting.** LLM output changes a little between runs even at temperature 0.1 (the README notes one changed answer was just model variance).

**Data**
5. **The default corpus has no bare text of every IPC/BNS section, and no BNSS.** Much "statute" and "constitution" material is question-and-answer pairs. Run `download_statutes.py` for the full Acts.
6. **Document types of Q&A records are guessed from keywords**, so a Q&A pair can be labelled "statute" and get authority 1.0.
7. **Judgment metadata is mostly missing** (case name, date), and the citation count is not stored in the index at all, so almost all Supreme Court chunks get the same authority (0.61). The authority prior costs 3.1 points of keyword precision.
8. **The 1,469 ShreyasP passages are not indexed** (numeric ids rejected; Section 5.4), although the report counts them as merged duplicates.
9. **The IPC/BNS table covers 63 common sections**, not the whole code, and it is hand-made (9 wrong rows were already found and fixed, so more checking is wise).

**Models and speed**
10. **MiniLM is a general-purpose model**, and by default it reads only the first 256 word pieces of a text, while our chunks average about 468 estimated tokens; the end of a long chunk does not affect its vector.
11. **The exact-match search takes seconds per query** (it scans text inside ChromaDB). Mean answer time is 15–18 s, most of it the LLM.
12. **For the best answers we still depend on an external LLM service.** Models move between free and paid tiers, and results differ between models (Run B vs C). The offline options (Nyaya-Gen, or LFM3-1.7B) write plainer answers.
13. **One in three questions still gets no cited answer** (response rate 0.68); case lookups are weakest (0.50).

**Nyaya-Gen**
14. **It is ~4,000 times smaller than the 550B API model** and does not know the law; its answers are only as good as the passages it is given, and shorter and plainer than the API model's.
15. **Its training targets restate one source** in at most 4 sentences, so it never learned to combine sources, compare their authority, or write a partial answer with "Not covered by the retrieved sources:".
16. **It sees only the first 90 words of each source** and at most 1,024 tokens in all; 1,415 of the 4,000 training examples were dropped for length, and the training answers are cut-down dataset answers, not answers written by lawyers.
17. **Only 50 effective updates (about 400 examples, 15% of one epoch), one seed, a dev loss on only 24 examples.** The run stopped early because the 8 GB Mac ran out of memory; its variance is unknown, and longer training might change the results.
18. **It is judged only by the rule-based metrics** (no human or expert check), and those metrics do not reward the things it cannot do (point 15).

**Use**
19. **It is a research and study tool, not legal advice.** English only: the Hinglish test question is an edge case.

---

## 12. 34 likely viva questions with short answers

**Basics**

**Q1. Explain your project in one or two lines.**
Nyaya is a RAG assistant for Indian law. It answers only from passages it retrieves from 112,555 chunks, tags every claim with `[SOURCE_N]`, checks those tags, maps IPC sections to BNS, and gives a confidence score. We also trained one small model on our laptop: SmolLM2-135M fine-tuned as an offline answer writer (Nyaya-Gen).

**Q2. What is RAG and why did you use it?**
Retrieval-augmented generation: first search a trusted collection, then let the LLM write only from what was found. It makes answers checkable and reduces invented facts, because the model copies from sources instead of relying on memory.

**Q3. Why not just use ChatGPT?**
ChatGPT answers from memory. Studies found 69–88% hallucination when LLMs were asked about real US cases (Dahl et al.). It gives no source for each sentence, may not know the 2024 BNS numbers, and cannot be checked. Our system answers only from numbered passages, every claim points to one, and the IPC/BNS table is a source. The LLM is only the writer, and we can swap it (NVIDIA NIM or a local model).

**Q4. What is an embedding, and why 384 numbers?**
A list of numbers that represents the meaning of a text; similar meanings give similar vectors. 384 is simply the output size of `all-MiniLM-L6-v2`.

**Q5. What is cosine similarity? Give an example.**
It measures the angle between two vectors: (a . b) / (||a|| x ||b||). For (1, 2, 2) and (2, 1, 2): 8 / (3 x 3) = 0.889. ChromaDB gives the distance d, and we use 1 - d.

**Q6. Why do you split documents into chunks, and why by section, article and paragraph?**
The LLM can only read a limited amount, and smaller pieces make search precise. Cutting at legal units keeps a rule together with its exceptions and provisos, and each chunk can be cited as one unit ("Section 302", "Article 21").

**Q7. How many chunks are there and where do they come from?**
112,555: 101,983 from chunking 31,069 documents (Q&A pairs, full judgments, Constitution Q&A), plus 10,572 pre-split Supreme Court passages. About 76% are judgment chunks. (The 1,469 ShreyasP passages are prepared but skipped because their ids are numbers.)

**Design choices**

**Q8. Why hybrid search instead of only embeddings?**
Dense search understands meaning but is weak on exact identifiers: on case lookups its keyword precision was only 0.225, and it may confuse Article 21 with Article 210. Exact match is perfect on section lookups (1.000) but useless on conceptual questions (0.100). Together: keyword precision 0.811 vs 0.656 for dense only (+15.5 points), Hit@4 0.878 vs 0.796.

**Q9. Is your exact-match search BM25?**
No. We extract legal identifiers (section and article numbers, AIR/SCC citations, distinctive party names) and ask ChromaDB for chunks that contain them, ordered by closeness to the question vector, with a whole-word check. BM25 is a word-frequency ranking method that we cite but did not use.

**Q10. What is RRF and why k = 60?**
Reciprocal Rank Fusion gives each chunk 1 / (k + rank) in each list and adds them. It needs no score scaling. k = 60 is the standard value from Cormack et al.; a small k (k = 1) lets the top item dominate and keyword precision dropped to 0.750; from 30 to 100 results were within 1.5 points.

**Q11. How do you decide the question type and why does it matter?**
A rule-based classifier with regular expressions: "Section n", "Article n", Act names, IPC/CrPC/CPC/BNS -> section lookup; "X v Y" or AIR/SCC -> case lookup; both -> mixed; neither -> conceptual. The type sets the fusion weights (dense, exact): (0.3, 0.7), (0.2, 0.8), (0.8, 0.2), (0.5, 0.5). Adaptive weights gave +6.6 points keyword precision and +4.1 points MRR over fixed 0.5/0.5.

**Q12. Why do you give only 4 passages to the LLM?**
Table VI: fewer passages give a purer context, more give a few extra hits. Beyond 6 there were no new hits, and long prompts put the useful passage in the middle where LLMs use it less. K = 4 is the smallest value that still allows answers from several sources with keyword precision above 0.8. It also shortened the prompt and the latency.

**Q13. Why temperature 0.1?**
Low randomness: the model sticks to the most likely wording, which is better for factual, cited answers and makes runs more repeatable.

**Q14. How does the IPC-to-BNS mapping work, and why is it a source?**
A 63-row table. Regular expressions find IPC or BNS references in the question (both directions, or by offence name when both Acts are named). The counterpart section is added to the search, and the matched rows are added as the last `[SOURCE_N]`. The model must state both numbers and cite the table, so a BNS number is copied from a checkable source instead of recalled from memory.

**Formulas**

**Q15. Explain the final score S with numbers.**
S = 0.50 s + 0.25 R~ + 0.25 A. Example: s = 0.71, R~ = 0.989, A = 1.0 gives 0.355 + 0.247 + 0.250 = 0.852. R~ = min(61 x R, 1) puts the RRF score on a 0–1 scale; without it, RRF was about 60 times too small to matter.

**Q16. What is the authority score, and why did it lower precision?**
1.0 for statutes and the Constitution; for judgments 0.4 x court + 0.3 x recency + 0.3 x citations. Because most judgment chunks lack date and citation data, they get about 0.61, so Constitution passages sometimes push out relevant judgment paragraphs: keyword precision fell 3.1 points (0.842 -> 0.811). We kept it because binding sources should come first when sources conflict; it is a design choice and should be re-tuned with better metadata.

**Q17. What is the confidence score? Does "High" mean the answer is correct?**
C = 0.40 Q (retrieval quality) + 0.25 G (number of distinct cited sources) + 0.20 H (authority of cited sources) + 0.15 V (share of passages used). High >= 0.8, Medium >= 0.6, Low >= 0.4, else Very Low. No: it measures the strength of the evidence and citing, not correctness. An uncited or refusal answer cannot score high.

**Results**

**Q18. What is the difference between precision and recall?**
Precision: of what we returned, how much is relevant. Recall: of all relevant items that exist, how much we returned. Example: 10 relevant chunks exist, we return 4, 3 are relevant: precision 0.75, recall 0.30. We cannot measure true recall (we do not know every relevant chunk among 112,555), so we use Hit@K as a recall-like measure.

**Q19. What are P@4, P_kw@4, Hit@4 and MRR?**
P@4: share of the 4 passages that have the expected type or a keyword. P_kw@4: share that contain an expected keyword (stricter). Hit@4: 1 if at least one passage has a keyword. MRR: average of 1 / rank of the first keyword passage; for ranks 1, 2 and none it is (1 + 0.5 + 0) / 3 = 0.5.

**Q20. How do you know your system does not hallucinate?**
We cannot promise zero. We reduce it with several layers (retrieval only, strict prompt, low temperature, IPC/BNS source, citation validation, confidence) and we measure it: the detector flagged 2 of 50 answers (0.04), and on inspection neither was an invented case (one was reasoning text, one a historical phrase). The detector cannot catch a wrong statement with a real citation, so the true rate is higher; citation support 0.50 is the more careful number. Every claim can be checked by the user against its `[SOURCE_N]`.

**Q21. Why did citation validity fall from 0.94 (Run A) to 0.70 (Run C)?**
Validity is averaged over all answers and an answer with no tag counts 0. Run A's old prompt refused often: 35 of 50 answers were not substantive, and 33 of its fully valid answers were refusals. The new prompt answers the covered part, so the response rate rose from 0.30 to 0.68, and validity now moves with it. Hallucination stayed at 1–2 answers in 50.

**Q22. Why is the response rate only 0.68? Which questions are hardest?**
Some needed texts are not in the default corpus (BNSS, BNS organised crime and sedition), and for case lookups the corpus often has fragments of a judgment without its name, so the model correctly declines. Case lookups have response rate 0.50, cross-reference 0.60, section lookups 0.93.

**Q23. What was wrong with tuning on the test set, and what did you do about it?**
If you choose settings on the same questions you test on, the scores look better than they would on new questions. We say so in the report (Tables VI–VIII are a sensitivity analysis). For Nyaya-Gen we choose the checkpoint on a separate dev split and remove training questions that resemble the 50 test queries.

**Limitations and practical questions**

**Q24. What are the main limitations?**
Small hand-made test set also used for tuning; keyword-based relevance; a pattern-based hallucination detector; missing bare Acts, BNSS and judgment metadata; a 63-row IPC/BNS table; a general-purpose embedding model that reads only 256 word pieces; the best answers still need an external LLM, and our offline Nyaya-Gen writes plainer, single-source answers; response rate 0.68. It is a study tool, not legal advice.

**Q25. How long does it take to build and to answer, and how did you test the code?**
Building the index takes about 15 minutes (11 for embedding). An answer takes about 15–18 seconds on average, mostly the LLM. Automatic pytest tests (250 in the report, more now) cover chunking, parsing, retrieval scoring, IPC/BNS, prompts, citations, confidence, metrics, the retrieval experiments, the local generator helpers and the full pipeline, with the store and LLM mocked, so they run without internet. Training Nyaya-Gen ran about 65 minutes (limit 75) before it stopped.

**The trained model: Nyaya-Gen**

**Q26. What is Nyaya-Gen and why did you train it?**
SmolLM2-135M-Instruct (134.5M parameters), fully fine-tuned by us to write the answer from the numbered sources with a `[SOURCE_N]` tag after each sentence. It replaces the online LLM, so the whole system runs on the laptop with no internet or API key. We wanted the answer writer itself to be a model we trained.

**Q27. A 135M model cannot know Indian law. Why should it work?**
It is not asked to know the law. Every fact is in the sources in the prompt; its job is grounded generation: pick the source that answers, restate it, and cite its number, or refuse if nothing fits. That is mostly selecting and copying, which a small model can learn from a few thousand examples. It is still worse than a 550B model: shorter, plainer answers from one source, no comparison of sources. We measure that gap on the same 50 questions (Run D vs Run C).

**Q28. What is a causal language model and next-token prediction?**
A model that writes one token at a time, each time giving a probability to every token in its vocabulary (49,152) from all the tokens before it, and never looking ahead. Training makes the probability of the true next token high; the loss at one position is -ln p(true token). Example: p = 0.62 gives loss 0.478.

**Q29. How did you make the training data, and how did you avoid leakage?**
From the statute and Constitution Q&A records, without the index. Each example is a prompt in exactly the pipeline's format, made by our own PromptBuilder: the record hidden at a random position among 2–4 random passages, and as target up to 4 sentences of its answer, each ending with `[SOURCE_k]`. Also IPC/BNS table questions, and about 7% refusals where the right record is left out. 572 questions with content-word Jaccard >= 0.5 to a test query were removed, and table rows whose section appears in a test query were skipped. 4,000 training examples, 2,585 after the 1,024-token limit.

**Q30. Why is the loss computed only on the answer tokens?**
The prompt is most of each example (about 930 tokens, mostly sources). If it counted, the model would mostly be trained to predict random source text and questions. With the prompt labels set to -100 (ignored), the loss only rewards writing the right answer and putting the right `[SOURCE_k]` tag in it. The prompt is still read as context.

**Q31. What are gradient checkpointing and gradient accumulation?**
Checkpointing: instead of storing every layer's intermediate results for the backward pass, store only each layer's input and recompute the rest during the backward pass; much less memory, about one extra forward pass of time. Accumulation: run 1 example at a time, add up the gradients of 8 examples, then do one update, so each update sees 8 examples without 8 examples in memory.

**Q32. How did the 8 GB limit shape the settings?**
Weights, gradients and Adam values alone are about 2.2 GB in fp32, and the activations of a ~1,000-token example take several GB more. bf16 autocast and fp32 without checkpointing both ran out of GPU memory. What fits: fp32 with gradient checkpointing, 1 example per pass with accumulation 8, at most 1,024 tokens (which dropped 1,415 long examples), and sources cut to 90 words, also at query time. It is slow (about 0.8 minutes per update): after about 80 of 323 updates (about 65 minutes) the Mac ran out of memory and the run stopped, so the best saved checkpoint, at update 50, is used.

**Q33. How do you evaluate Nyaya-Gen, and how do you switch it on?**
Dev loss before and after training, then the 50 test questions with all six answer metrics and latency: Run D vs Run C (550B via API) shows the cost of writing locally. Run D uses the same retrieval and prompt as Run C. Switch: `LLM_BACKEND=local` (default `api`); `scripts/run_evaluation.sh` runs D, then (optionally, not run for the submission) the retrieval tables and D0.

**Q34. Your training stopped early. Is the model still valid?**
Yes, but it is a limited result and we say so. One epoch would be 323 updates; on the 8 GB M2 the run did about 80 updates in about 65 minutes and then ran out of memory (swap full). Checkpoints are saved every 50 updates by dev loss, so the model we use is the one at update 50, which has seen about 400 examples. The dev loss had already fallen from 0.8325 to 0.0334, because the answer format is easy to learn. Whether that is enough is measured on the 50 test questions (Run D vs D0), not assumed. With more memory or time we would finish the epoch.

---

## 13. One-page cheat sheet

**The idea:** search first, then let the LLM write only from numbered sources; check every citation; map IPC to BNS; score confidence; optionally write the answer offline with our own fine-tuned SmolLM2-135M (Nyaya-Gen).

**Key numbers**

| Topic | Number |
|---|---|
| Datasets | 5 from Hugging Face (Table I) |
| Documents chunked | 31,069 -> 101,983 chunks |
| Pre-split passages indexed | 10,572 (of 12,041 prepared; 1,469 ShreyasP skipped) |
| Chunks in the index | **112,555** (judgment 76.4%, Constitution 20.1%, statute 3.6%) |
| Chunk sizes | target 450, max 800, overlap 50 estimated tokens; tokens = floor(1.3 x words) |
| Embedding | all-MiniLM-L6-v2, 384 numbers, batches of 64; ChromaDB, HNSW, cosine |
| Search | 20 per path; RRF k = 60 |
| Fusion weights (dense, exact) | section 0.3/0.7 · case 0.2/0.8 · conceptual 0.8/0.2 · mixed 0.5/0.5 |
| Final score | S = 0.50 s + 0.25 R~ + 0.25 A; R~ = min(61 R, 1) |
| Authority | 1.0 statute/Constitution; else 0.4 court + 0.3 recency + 0.3 citations (SC without data = 0.61) |
| Passages to LLM | K = 4 (+ IPC/BNS table as the last source when matched) |
| LLM | DeepSeek V3.2 default; Nemotron 3 Ultra in Runs B and C; temperature 0.1; 1,024 tokens; 40 requests/min; offline: Nyaya-Gen (`LLM_BACKEND=local`) or LFM3-1.7B |
| Confidence | C = 0.40 Q + 0.25 G + 0.20 H + 0.15 V; High >= 0.8, Medium >= 0.6, Low >= 0.4 |
| IPC/BNS table | 63 rows; 302 -> 103, 376 -> 64, 420 -> 318(4), 498A -> 85, 124A -> 152 |
| Test set | 50 queries: 15 section, 10 case, 10 conceptual, 10 cross-reference, 5 edge |
| Retrieval (Table III) | P_kw@4: dense 0.656 -> proposed 0.811 (+15.5 points); Hit@4 0.796 -> 0.878; adaptive vs fixed +6.6 points; authority -3.1 points |
| Run C (final) | validity 0.70 · coverage 0.42 · support 0.50 · hallucination 0.04 · response 0.68 · retrieval precision 0.92 · confidence 0.61 · 17.5 s |
| Run A (old) | validity 0.94 · hallucination 0.02 · response 0.30 · precision 0.79 · confidence 0.69 · 25.0 s |
| Tests | 250 pytest tests in the report; more now (recount before the viva) |
| Build time | about 15 minutes (11 for embedding) |

**Nyaya-Gen in numbers**

| Topic | Value |
|---|---|
| Model | SmolLM2-135M-Instruct, decoder-only causal LM, 134.5M parameters, 30 layers, hidden 576, vocabulary 49,152; full fine-tuning; about 540 MB |
| Job | write the answer from the numbered sources, `[SOURCE_k]` after each sentence; replaces the API LLM (`LLM_BACKEND=local`) |
| Data | statute + Constitution Q&A, prompts made by the real PromptBuilder; qa 3,590 / mapping 141 / refusal 269 = 4,000 train (2,585 within 1,024 tokens); dev 200, test 200; split 94/3/3 by document; 572 removed by the leakage filter (Jaccard >= 0.5) |
| Training | next-token cross-entropy on answer tokens only; 1 example per pass x accumulation 8; AdamW lr 1e-4, betas (0.9, 0.95), weight decay 0.01; 20 warm-up updates then cosine; 1 epoch = 323 updates planned; clip 1.0; fp32 + gradient checkpointing on MPS; dev loss every 50 updates; max 75 minutes |
| What happened | about 0.8 min per update; about 80 updates in about 65 minutes, then out of memory (swap full); kept checkpoint = update 50 (about 400 examples seen) |
| 8 GB finding | bf16 autocast and fp32 without checkpointing ran out of GPU memory |
| Use | sources cut to 90 words; greedy; repetition penalty 1.1; max 256 new tokens |
| Results | dev loss 0.8325 -> 0.0334 (update 50); training loss 0.98 -> about 0.005–0.02; Run D validity 0.74, response rate 0.72 (Run C: 0.70, 0.68) |

**One-line definitions**

- **RAG:** retrieve passages first, then generate an answer only from them.
- **Hallucination:** a false statement presented as true (fake case, wrong section).
- **Embedding:** numbers that represent meaning; similar meaning, similar numbers.
- **Cosine similarity:** (a . b) / (||a|| ||b||); 1 = same direction.
- **Chunk:** one searchable piece; for us one section, article or judgment paragraph group.
- **HNSW:** graph index for fast nearest-vector search.
- **RRF:** 1 / (60 + rank), summed over lists; merges rankings without score scaling.
- **Authority:** how binding a source is (statute and Supreme Court high).
- **Re-ranking:** sorting candidates again with a better score.
- **Fine-tuning:** extra training of a pre-trained model on our task.
- **Leakage:** test data seen in training; makes results look too good.
- **Loss:** a number that says how wrong the model is; training lowers it.
- **Epoch:** one pass over all training data.
- **Learning rate:** size of each update step.
- **Warm-up:** learning rate rises slowly at the start.
- **AdamW:** optimiser with a per-weight step size plus weight decay.
- **Gradient clipping:** limit the size of an update.
- **Checkpoint:** a saved copy of the model; we keep the best on dev.
- **Causal language model:** writes one token at a time, predicting each next token from the ones before.
- **Next-token cross-entropy:** -ln of the probability given to the true next token, averaged.
- **Masked loss (labels -100):** prompt tokens are read but not scored; only the answer is learned.
- **Grounded generation:** writing only from the given sources, citing them.
- **Gradient accumulation:** add up gradients of several small passes, then update once.
- **Gradient checkpointing:** recompute activations in the backward pass instead of storing them; less memory, more time.
- **Greedy decoding:** always take the most probable next token.
- **MPS:** Apple GPU backend for PyTorch.
- **fp32 / bf16:** 32-bit / 16-bit numbers; 16-bit uses half the memory but is less exact.
- **Precision / recall:** relevant share of what we returned / returned share of what is relevant.
- **MRR:** average of 1 / rank of the first relevant result.
- **Citation validity:** share of `[SOURCE_N]` tags that point to a source we sent.
- **Response rate:** share of answers that cite a valid source and are not refusals.

---

## Appendix A. Where the numbers come from

Training: `data/generator/training_log.json`. Run D (Nyaya-Gen, fully local): `data/evaluation/eval_results_local.json` — citation validity 0.74, cited claim coverage 0.66, citation support 0.56, hallucination rate 0.00, response rate 0.72, retrieval precision 0.92, average confidence 0.45, mean latency 45.0 s. Run D0 and the re-computed retrieval tables were not run because of time.

## Appendix B. Things to check before the viva

1. **Which file is Run C?** `data/evaluation/eval_results_deepseek.json` (27 October 2026) holds exactly the Run C numbers, but the report and README describe Run C as a Nemotron run saved as `eval_results_latest.json` (which does not exist). The old result files do not record the model name, so the label cannot be proved from them. Confirm which model produced it, and make the file name and the text agree. (`evaluate.py` now writes the model name into every new results file.)
2. **The 112,555 explanation.** The report says the 1,469 difference comes from "duplicate identifiers". The code shows the 1,469 ShreyasP passages are rejected for numeric ids (Section 5.4). Decide whether to correct the report text, or fix the ids and rebuild (which changes the numbers).
3. **Report corrections already listed.** `docs/ALIGNMENT_CHANGES.md` lists places where the report text should change (for example: confidence H falls back to all retrieved chunks when nothing is cited; the 16-name exemption in the hallucination detector; adaptive fusion is slightly below the best fixed setting on P_kw; Run A's precision is P@6; the reference numbers [11]–[21] in the text do not match the 14-entry reference list). Know these before the teacher finds them.
4. **Training did not finish the epoch.** Nyaya-Gen stopped after about 80 of 323 updates (about 65 minutes) when the 8 GB Mac ran out of memory; the checkpoint at update 50 is used. Say this plainly; `docs/REPORT_ADDITIONS.md` already describes it this way.
5. **Re-read Sections 8.4, 8.5 and 9.9** with the real numbers (Appendix A), so that what you say matches the final report.
