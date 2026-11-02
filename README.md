# Nyaya: Indian Legal Research Assistant

Nyaya answers questions about Indian law using a local collection of legal texts: statutes, Constitution articles and Supreme Court judgments. Every answer cites the passages it relies on, so you can check each claim against its source. Under the hood it is a retrieval-augmented generation (RAG) pipeline. The language model only writes the answer from passages the system has already retrieved. It is not used as a source of legal knowledge.

It is a research and study tool. It does not give legal advice.

## What it does

- **Hybrid retrieval.** Every query is classified as a section lookup, a case lookup, a conceptual question or a mix of these. The system then runs two searches. Semantic search uses `all-MiniLM-L6-v2` embeddings in ChromaDB. Exact-match search looks for legal identifiers such as "Section 302", "Article 21", an AIR/SCC citation or a party name from a case title ("Maneka Gandhi" in "Maneka Gandhi v Union of India"), and takes the passages containing them that are closest to the query. The two result lists are merged with weighted Reciprocal Rank Fusion, and the weights depend on the query type.
- **Authority re-ranking.** Statutes and the Constitution score 1.0. Judgments are scored by court (Supreme Court > High Court > Tribunal), by how recent they are and by how often they are cited.
- **IPC to BNS lookup.** If a query mentions an IPC section (for example "Section 302 IPC"), the matching Bharatiya Nyaya Sanhita section is added to the search. Both numbers then go to the model as a citable source. This also works from BNS back to IPC, and by offence name for questions like "compare IPC and BNS on murder".
- **Citation checking.** The model must tag each claim with `[SOURCE_N]`. Every tag is checked against the passages that were actually sent to the model, and tags that point nowhere are flagged.
- **Confidence score.** Each answer gets a score built from four signals: retrieval quality (40%), number of distinct sources cited (25%), authority of the cited sources (20%) and the share of retrieved passages actually used (15%).
- **Command line and Python API.** `scripts/ask.py` prints the answer, the numbered sources (marking the ones cited), the IPC/BNS match and the confidence breakdown, or the full result as JSON. `LegalResearchPipeline` gives the same result to Python code.

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
  ├─ LLM: API model via NVIDIA NIM (LLM_BACKEND=api)
  │   or Nyaya-Gen on the laptop (LLM_BACKEND=local)
  ├─ citation validation
  └─ confidence score → answer + sources
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

A free API key is available at [build.nvidia.com](https://build.nvidia.com). NVIDIA moves models between free and partner endpoints from time to time. If `nvidia/nemotron-3-ultra-550b-a55b` is not available on your key, set `LLM_MODEL` to another chat model from the catalogue. Any other OpenAI-compatible endpoint also works if you set `LLM_BASE_URL` and `LLM_MODEL` in `.env`.

Without a key the assistant still returns the retrieved sources for each question, but it cannot write an answer.

### Nyaya-Gen: the answer writer trained for this project

Nyaya-Gen is the one model trained in this project. It is `HuggingFaceTB/SmolLM2-135M-Instruct` (134.5M parameters), fully fine-tuned to write the answer from the numbered sources with `[SOURCE_N]` tags, in exactly the prompt format of the pipeline. It replaces the API model, so retrieval, generation and citation checking all run on the laptop with no key and no network. The rest of the pipeline is unchanged; citation validation and the confidence score check its answers exactly as they check the API model's.

Select it with two settings (in `.env` or the environment):

| Variable | Meaning |
|---|---|
| `LLM_BACKEND` | `api` (default: OpenAI-compatible API, e.g. Nemotron on NVIDIA NIM) or `local` (Nyaya-Gen) |
| `NYAYA_GEN_PATH` | folder of the model to load when `LLM_BACKEND=local` (default `models/nyaya-gen/`) |

```bash
LLM_BACKEND=local python scripts/ask.py "What is Section 302 of IPC?"
```

At inference each source is cut to its first 90 words (as in training), decoding is greedy with repetition penalty 1.1, and at most 256 new tokens are written. If the model folder is missing, the pipeline logs a warning and returns the sources only.

Training and evaluation (sized for an 8 GB M2; close other apps first):

| Script | What it does |
|---|---|
| `scripts/nyaya_gen/prepare_generator_data.py` | builds `data/generator/{train,dev,test}.jsonl` and `data_stats.json` from the statute and Constitution Q&A datasets (qa, IPC/BNS mapping and refusal examples, with test queries filtered out) |
| `scripts/nyaya_gen/train_generator.py` | fine-tunes SmolLM2-135M-Instruct (fp32, gradient checkpointing, AdamW lr 1e-4, gradient accumulation 8), keeps the checkpoint with the lowest dev loss in `models/nyaya-gen/` and logs to `data/generator/training_log.json` |
| `scripts/nyaya_gen/run_training.sh` | prepares the data if it is missing, then runs `train_generator.py` with the settings used for the report |
| `scripts/evaluation/run_evaluation.sh` | 1) Run D: Nyaya-Gen on the 50 test queries (`data/evaluation/run_D_nyaya_gen.json`); 2) `scripts/evaluation/retrieval_experiments.py` for the retrieval tables; 3) Run D0: the base SmolLM2-135M-Instruct, not fine-tuned (`run_D0_base.json`) |

```bash
nohup caffeinate -i scripts/nyaya_gen/run_training.sh > logs/training.log 2>&1 &
nohup caffeinate -i scripts/evaluation/run_evaluation.sh > logs/evaluation.log 2>&1 &
```

What the recorded training run did: 2,585 of the 4,000 built training examples fit in 1,024 tokens; one epoch would be 323 updates of 8 examples. At about 0.8 minutes per update the run completed about 80 updates in about 65 minutes before the 8 GB Mac ran out of memory, so the kept checkpoint is the one at update 50 (about 400 examples seen). The dev loss fell from 0.833 before training to 0.033 at that checkpoint.

## Building the knowledge base

The vector store is not committed because it is about 1.7 GB. Build it locally:

```bash
python scripts/data/download_hf_datasets.py   # 5 public Indian-law datasets from Hugging Face
python scripts/data/prepare_hf_data.py        # convert to the chunker's input format
python scripts/data/chunk_documents.py        # legal-aware chunking
python scripts/data/build_vectorstore.py      # embed and index into chroma_db/
```

This gives **112,555 chunks**: 101,983 from chunking about 31,000 documents, plus 12,041 chunks that come pre-split in two of the datasets. On an Apple M-series laptop the whole build takes about 15 minutes, 11 of them for embedding.

Optional extra sources:

```bash
python scripts/data/download_statutes.py      # IPC, BNS, CrPC, Constitution from India Code
python scripts/data/download_indiankanoon.py  # landmark Supreme Court judgments
python scripts/data/parse_pdfs.py             # parse those (and any PDFs you add to data/raw/)
python scripts/data/chunk_documents.py
python scripts/data/build_vectorstore.py
```

Running `build_vectorstore.py` again is safe. Chunk ids are deterministic, so existing chunks are updated in place instead of being added twice. After changing the chunking, run it with `--reset` so chunks that no longer exist are removed. Scanned PDFs need the `tesseract` binary for OCR (`brew install tesseract` or `apt install tesseract-ocr`).

### Chunking

Chunks follow the structure of the document rather than a fixed character count:

- **Statutes** are split one section per chunk. A long section is split again at its sub-section markers.
- **The Constitution** is split one article per chunk, and explanations and provisos stay with their article.
- **Judgments** are split at numbered paragraphs, aiming for about 450 tokens with a 50-token overlap. Very short paragraphs are merged, and the case name is added at the start of each chunk.

## Asking questions

```bash
python scripts/ask.py "What is Section 302 of IPC and its BNS equivalent?"
python scripts/ask.py --json "What does Article 21 guarantee?"   # full result as JSON
python scripts/ask.py                                             # interactive; models load once
```

Add `-v` to see the pipeline logs. Loading the embedding model takes a few seconds, and the LLM call usually takes most of the time.

From Python:

```python
from src.pipeline import LegalResearchPipeline

pipeline = LegalResearchPipeline.from_settings()
result = pipeline.query("Explain the doctrine of basic structure")
print(result.answer_text)          # cites passages as [SOURCE_1], [SOURCE_2], ...
print(result.sources[0]["text"])   # [SOURCE_N] is result.sources[N - 1]
print(result.confidence.level)
```

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

There are 300 unit and integration tests. They cover chunking, PDF parsing, hybrid search scoring, re-ranking, the IPC/BNS lookup, prompt building, citation parsing, confidence scoring, the evaluation metrics and the full `LegalResearchPipeline` with the vector store and the LLM mocked. They need no API key or network access.

## Evaluation

```bash
python scripts/evaluation/evaluate.py            # all 50 queries in data/evaluation/test_queries.json
python scripts/evaluation/evaluate.py --limit 10
```

Results go to `data/evaluation/eval_results_latest.json` (choose another file with `--output`). The test set has 50 questions: section lookups, case lookups, conceptual questions and mixed ones. The script reports:

| Metric | Meaning |
|---|---|
| Citation validity | Share of `[SOURCE_N]` tags that point to a passage the model was actually given (called "citation accuracy" in the project report) |
| Cited claim coverage | Share of legal claim sentences that carry a citation |
| Citation support | Share of cited claims whose source contains the same section numbers, legal terms and enough of the same wording |
| Hallucination rate | Share of answers that name a case or an AIR/SCC/SCR citation found in none of the sources |
| Response rate | Share of answers that cite at least one valid source and are not a refusal |
| Retrieval precision | Share of retrieved passages of the expected document type or containing an expected keyword |

### Recorded results

These are the runs of Table IV and Table XI of the report (50 test queries each). Runs A–C use an API model; Run D uses Nyaya-Gen, fully local.

| File | Run | Citation validity | Cited claim coverage | Citation support | Hallucination rate | Response rate |
|---|---|---|---|---|---|---|
| `data/evaluation/run_A.json` | A: DeepSeek V3.2, earlier prompt, K = 6 | 0.94 | – | – | 0.02 | 0.30 |
| `data/evaluation/run_B.json` | B: Nemotron | 0.66 | 0.42 | 0.50 | 0.02 | 0.66 |
| `data/evaluation/run_C.json` | C: Nemotron, final | 0.70 | 0.42 | 0.50 | 0.04 | 0.68 |
| `data/evaluation/run_D_nyaya_gen.json` | D: Nyaya-Gen (local) | **0.74** | **0.66** | **0.56** | **0.00** | **0.72** |

Standard metrics for Nyaya-Gen on its 200 held-out examples (Table XII, `data/evaluation/generator_metrics.json`): token F1 0.42 → 0.93, ROUGE-L 0.40 → 0.93, BERTScore 0.83 → 0.98 (before → after training).

## Project layout

```
config/
  settings.py              paths, model names, retrieval and scoring weights
  prompts.py               system prompt and source format
  ipc_bns_mapping.json     IPC → BNS correspondence table (63 sections)
src/                       the system, one folder per pipeline stage
  ingestion/               PDF parsing, text cleaning, metadata extraction
  chunking/                section / article / paragraph chunkers
  embeddings/              MiniLM embeddings, ChromaDB store
  retrieval/               IPC/BNS lookup, query classification, hybrid search, authority re-ranking
  generation/              prompt builder, API client, Nyaya-Gen (local_generator.py), citation parser
  confidence/              confidence score
  evaluation/              citation, hallucination, response and retrieval metrics
  pipeline.py              LegalResearchPipeline: ties the stages together
scripts/
  ask.py                   ask a question from the terminal
  data/                    download datasets, parse, chunk, build the index
  nyaya_gen/               Nyaya-Gen: build training data, train (run_training.sh)
  evaluation/              end-to-end evaluation, retrieval tables, standard metrics (run_evaluation.sh)
tests/                     pytest suite (300 tests)
data/evaluation/           test queries and the results of Runs A–D
data/generator/            Nyaya-Gen training data, training log and test predictions
report/                    final report PDF
models/                    trained Nyaya-Gen weights (not in git)
```

## Limitations

- The embedding model is general-purpose. A legal-domain model such as InLegalBERT would probably retrieve better, but that has not been tested here.
- Much of the Hugging Face data is question-and-answer pairs or judgments without metadata. Many judgment chunks have no case name, date or citation count, so their authority score falls back to defaults.
- The IPC/BNS table covers 63 commonly used sections, not the whole code.
- The default corpus comes from Hugging Face datasets and does not include the bare text of every IPC/BNS section. For section-level questions, run `download_statutes.py` too, which adds the full Acts from India Code.
- Nyaya-Gen was trained for only 50 effective updates (about 400 examples) because of the 8 GB memory limit, and it was trained to restate one source per answer, so it does not combine or compare sources.
- The hallucination check is pattern-based. It catches invented case names and reporter citations, but not a wrong statement attributed to a real source. The citation-support metric is a rough check against that.

## References

The design follows the literature reviewed in the project report, in particular:

- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS 2020
- Cormack, Clarke and Büttcher, *Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods*, SIGIR 2009
- Magesh et al., *Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools*, 2025
- Paul et al., *Pre-trained Language Models for the Legal Domain: A Case Study on Indian Law* (InLegalBERT), 2023
