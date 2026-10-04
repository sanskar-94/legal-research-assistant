# Nyaya: Indian Legal Research Assistant

Nyaya answers questions about Indian law using a local collection of legal texts: statutes, Constitution articles and Supreme Court judgments. Every answer cites the passages it relies on, so you can check each claim against its source. Under the hood it is a retrieval-augmented generation (RAG) pipeline. The language model only writes the answer from passages the system has already retrieved. It is not used as a source of legal knowledge.

It is a research and study tool. It does not give legal advice.

## What it does

- **Hybrid retrieval.** Every query is classified as a section lookup, a case lookup, a conceptual question or a mix of these. The system then runs two searches. Semantic search uses `all-MiniLM-L6-v2` embeddings in ChromaDB. Exact-match search looks for legal identifiers such as "Section 302", "Article 21" or an AIR/SCC citation. The two result lists are merged with weighted Reciprocal Rank Fusion, and the weights depend on the query type.
- **Authority re-ranking.** Statutes and the Constitution score 1.0. Judgments are scored by court (Supreme Court > High Court > Tribunal), by how recent they are and by how often they are cited.
- **IPC to BNS lookup.** If a query mentions an IPC section (for example "Section 302 IPC"), the matching Bharatiya Nyaya Sanhita section is added to the search. Both numbers then go to the model as a citable source. This also works from BNS back to IPC, and by offence name for questions like "compare IPC and BNS on murder".
- **Citation checking.** The model must tag each claim with `[SOURCE_N]`. Every tag is checked against the passages that were actually sent to the model, and tags that point nowhere are flagged.
- **Confidence score.** Each answer gets a score built from four signals: retrieval quality (40%), number of distinct sources cited (25%), authority of the cited sources (20%) and the share of retrieved passages actually used (15%).
- **Streamlit interface.** A chat window sits next to a sources panel. Citations are shown as clickable badges, and each answer comes with a confidence breakdown and can be exported as Markdown.

## How a query flows

```
question
  │
  ├─ IPC/BNS lookup ──────────────┐ (matched rows become an extra source)
  │                               │
  ├─ classify query               │
  ├─ semantic search (ChromaDB)   │
  ├─ exact-match search           │
  ├─ weighted RRF + authority     │
  │  re-ranking → top 4 passages  │
  │                               ▼
  ├─ prompt with numbered sources [SOURCE_1] … [SOURCE_n]
  ├─ LLM (DeepSeek V3.2 via NVIDIA NIM, temperature 0.1)
  ├─ citation validation
  └─ confidence score → answer + sources in the UI
```

## Ways it guards against invented answers

| Layer | What it does |
|---|---|
| Retrieval grounding | The model only sees retrieved passages |
| Strict prompt | Answer only from sources, never invent case names, citations or section numbers |
| Low temperature | Generation runs at 0.1 |
| Citation validation | Each `[SOURCE_N]` is checked against the passages that were sent |
| Confidence score | Weak retrieval or thin citation shows up as Low / Very Low |

## Setup

Requires Python 3.10 to 3.13.

```bash
git clone https://github.com/sanskar-94/legal-research-assistant.git
cd legal-research-assistant
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then put your NVIDIA_API_KEY in .env
```

A free API key is available at [build.nvidia.com](https://build.nvidia.com). NVIDIA moves models between free and partner endpoints from time to time. If `deepseek-ai/deepseek-v3.2` is not available on your key, set `LLM_MODEL` to another chat model from the catalogue. Any other OpenAI-compatible endpoint also works if you set `LLM_BASE_URL` and `LLM_MODEL` in `.env`.

Without a key the app still starts and shows the retrieved sources for each question, but it cannot write an answer.

## Building the knowledge base

The vector store is not committed because it is about 1.7 GB. Build it locally:

```bash
python scripts/download_hf_datasets.py   # 5 public Indian-law datasets from Hugging Face
python scripts/prepare_hf_data.py        # convert to the chunker's input format
python scripts/chunk_documents.py        # legal-aware chunking
python scripts/build_vectorstore.py      # embed and index into chroma_db/
```

This gives **112,555 chunks**: 101,983 from chunking about 31,000 documents, plus 12,041 chunks that come pre-split in two of the datasets. On an Apple M-series laptop the whole build takes about 15 minutes, 11 of them for embedding.

Optional extra sources:

```bash
python scripts/download_statutes.py      # IPC, BNS, CrPC, Constitution from India Code
python scripts/download_indiankanoon.py  # landmark Supreme Court judgments
python scripts/parse_pdfs.py             # parse those (and any PDFs you add to data/raw/)
python scripts/chunk_documents.py
python scripts/build_vectorstore.py
```

Running `build_vectorstore.py` again is safe. Chunk ids are deterministic, so existing chunks are updated in place instead of being added twice. Scanned PDFs need the `tesseract` binary for OCR (`brew install tesseract` or `apt install tesseract-ocr`).

### Chunking

Chunks follow the structure of the document rather than a fixed character count:

- **Statutes** are split one section per chunk. A long section is split again at its sub-section markers.
- **The Constitution** is split one article per chunk, and explanations and provisos stay with their article.
- **Judgments** are split at numbered paragraphs, aiming for about 450 tokens with a 50-token overlap. Very short paragraphs are merged, and the case name is added at the start of each chunk.

## Running the app

```bash
streamlit run app/streamlit_app.py
```

Then open http://localhost:8501. The sidebar shows how many chunks are indexed.

Example questions:

- What is Section 302 of IPC and its BNS equivalent?
- Define Section 498A of IPC
- What does Article 21 of the Constitution guarantee?
- Explain the doctrine of basic structure
- Compare IPC and BNS provisions on murder

## Tests

```bash
pytest tests/
```

There are 241 unit and integration tests. They cover chunking, PDF parsing, hybrid search scoring, re-ranking, the IPC/BNS lookup, prompt building, citation parsing, confidence scoring, the evaluation metrics and the full `LegalResearchPipeline` with the vector store and the LLM mocked. They need no API key or network access.

## Evaluation

```bash
python scripts/evaluate.py            # all 50 queries in data/evaluation/test_queries.json
python scripts/evaluate.py --limit 10
```

The test set has 50 questions: section lookups, case lookups, conceptual questions and mixed ones. The script reports:

| Metric | Meaning |
|---|---|
| Citation validity | Share of `[SOURCE_N]` tags that point to a passage the model was actually given (called "citation accuracy" in the project report) |
| Cited claim coverage | Share of legal claim sentences that carry a citation |
| Citation support | Share of cited claims whose source contains the same section numbers, legal terms and enough of the same wording |
| Hallucination rate | Share of answers that name a case or an AIR/SCC/SCR citation found in none of the sources |
| Response rate | Share of answers that cite at least one valid source and are not a refusal |
| Retrieval precision | Share of retrieved passages of the expected document type or containing an expected keyword |

### Recorded results

`data/evaluation/eval_results.json` contains the run reported in the project report (22 March 2026, 50 queries):

| Metric | Value |
|---|---|
| Citation accuracy (validity) | 0.94 |
| Hallucination rate | 0.02 |
| Retrieval precision | 0.79 |
| Average confidence | 0.69 |
| Response rate | 0.30 |

Two caveats apply to these numbers:

1. Citation validity only checks that a tag points to a real passage. It does not check that the passage supports the claim, which is why the evaluator now reports citation support separately.
2. The response rate of 0.30 was the main weakness of that run. The earlier prompt asked the model to say "the provided sources do not contain sufficient information" whenever coverage was incomplete. Many partly answerable questions therefore came back as refusals, and any answer containing that phrase was scored as non-substantive. The prompt now asks the model to answer what the sources cover and to state separately what is missing, and the response-rate check now counts cited partial answers. Retrieval scoring was also corrected (see below). Run `scripts/evaluate.py` with an API key to get current figures. The recorded run predates these changes, and its numbers should not be read as results for the current code.

## Changes since the report

The project report (section 8.3) listed this as remaining work. All of it is now done:

- **IPC to BNS mapping wired into the query pipeline** (`src/retrieval/ipc_bns.py`). Nine wrong rows in `config/ipc_bns_mapping.json` were also corrected, for example IPC 376 now maps to BNS 64 instead of 65, and IPC 364 to BNS 140(1) instead of 139.
- **LLM naming cleaned up.** `GeminiClient` is now `LLMClient`, and the settings point at DeepSeek V3.2 on NVIDIA NIM, which is the model the report describes.
- **Metadata flattened before prompt building**, so case names, courts, act names and section numbers now actually reach the model.
- **`full_text` and `text` handled consistently** in the parse and chunk scripts, so files from `parse_pdfs.py` are chunked too.
- **README and `requirements.txt` restored.**

Fixes found along the way:

- Semantic scores used `1 / (1 + d)` on a cosine distance. They now use `1 - d`.
- The RRF score was about 60 times smaller than the other ranking signals, so it had almost no effect. It is now scaled to 0 to 1 before weighting.
- Exact-match search was case-sensitive, so "section 302" missed "Section 302". Passages found only by exact match also had a semantic score of 0, which pushed them out of the top 4. They are now scored against the query embedding.
- Exact-match search took the first 10 matches in storage order, and "Section 302" appears in thousands of judgments. It now takes 200 candidates and ranks them by similarity to the query. Matches must also be whole identifiers, so "Article 21" no longer matches "Article 210".
- Some datasets contain the same record several times, and the copies used to fill the top-4 results. Duplicate passages are now removed before re-ranking.
- Chunk ids were random UUIDs, so rebuilding the index duplicated every chunk. `chunk_documents.py` also wrote each chunk to two files, so each one was embedded twice.

## Project layout

```
config/
  settings.py              paths, model names, retrieval and scoring weights
  prompts.py               system prompt and source format
  ipc_bns_mapping.json     IPC → BNS correspondence table (63 sections)
src/
  ingestion/               PDF parsing with OCR fallback, text cleaning, metadata extraction
  chunking/                section / article / paragraph chunkers
  embeddings/              sentence-transformers wrapper, ChromaDB store
  retrieval/               query classification, hybrid search, re-ranking, IPC/BNS lookup
  generation/              LLM client, prompt builder, citation parser
  confidence/              four-signal confidence score
  evaluation/              citation, hallucination and response metrics
  pipeline.py              LegalResearchPipeline: ties the stages together
app/
  streamlit_app.py         UI
  components/              chat, sources panel, confidence bar, export
scripts/                   data download, parsing, chunking, indexing, evaluation
tests/                     pytest suite
data/evaluation/           test queries and recorded results
```

## Limitations

- The embedding model is general-purpose. A legal-domain model such as InLegalBERT would probably retrieve better, but that has not been tested here.
- Much of the Hugging Face data is question-and-answer pairs or judgments without metadata. Many judgment chunks have no case name, date or citation count, so their authority score falls back to defaults.
- The IPC/BNS table covers 63 commonly used sections, not the whole code.
- The default corpus comes from Hugging Face datasets and does not include the bare text of every IPC/BNS section. For section-level questions, run `download_statutes.py` too, which adds the full Acts from India Code.
- The hallucination check is pattern-based. It catches invented case names and reporter citations, but not a wrong statement attributed to a real source. The citation-support metric is a rough check against that.

## References

The design follows the literature reviewed in the project report, in particular:

- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS 2020
- Cormack, Clarke and Büttcher, *Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods*, SIGIR 2009
- Magesh et al., *Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools*, 2025
- Paul et al., *Pre-trained Language Models for the Legal Domain: A Case Study on Indian Law* (InLegalBERT), 2023
