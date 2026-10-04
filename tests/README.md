# Indian Legal Research Assistant

An Explainable RAG-based legal research assistant for Indian law. Answers legal questions grounded in real Indian legal documents (Supreme Court judgements, IPC/BNS, CrPC, Constitution) with paragraph-level citations, confidence scores, and authority-based ranking.

The system is designed to reduce hallucinated cases and citations by grounding answers in retrieved legal sources, validating source links, and reporting confidence signals.

## Features

- **Citation-grounded answers**: Answers cite `[SOURCE_N]` tags mapped to retrieved legal documents
- **5-layer anti-hallucination defense**: Retrieval grounding, strict prompts, low temperature, citation validation, confidence scoring
- **Hybrid search**: Semantic (embeddings) + keyword search with Reciprocal Rank Fusion
- **Authority-based ranking**: Supreme Court > High Court > Tribunal, with recency weighting
- **Confidence scoring**: 4-signal scoring (retrieval quality, source agreement, authority strength, coverage)
- **IPC-to-BNS mapping**: Cross-reference old IPC sections with new BNS equivalents
- **Interactive UI**: Two-column Streamlit app with clickable citation viewer and export

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| LLM | Google Gemini 2.0 Flash (free tier) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB |
| Document Processing | PyMuPDF + pytesseract (OCR) |
| Data Models | Pydantic v2 |

## Quick Start

### 1. Install Dependencies

```bash
cd legal-research-assistant
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
cp .env.example .env
# Edit .env and add your Gemini API key
# Get one free at: https://aistudio.google.com/app/apikey
```

### 3. Build the Knowledge Base

```bash
# Step 1: Download legal datasets
python3 scripts/download_hf_datasets.py

# Step 2: Parse all raw documents
python3 scripts/parse_pdfs.py

# Step 3: Chunk documents using legal-aware strategies
python3 scripts/chunk_documents.py

# Step 4: Build the vector store (~15-30 min on CPU)
python3 scripts/build_vectorstore.py
```

### 4. Launch the App

```bash
streamlit run app/streamlit_app.py
```

Open http://localhost:8501 in your browser.

## Project Structure

```
legal-research-assistant/
├── config/
│   ├── settings.py                 # Paths, model names, thresholds
│   ├── prompts.py                  # All LLM prompt templates
│   └── ipc_bns_mapping.json        # IPC-to-BNS section mapping
├── data/
│   ├── raw/                        # Downloaded PDFs, HTML, datasets
│   │   ├── statutes/               # IPC, BNS, CrPC, Constitution
│   │   └── judgements/             # Supreme Court judgements
│   ├── processed/chunks/           # JSON chunks with metadata
│   └── evaluation/test_queries.json
├── scripts/
│   ├── download_hf_datasets.py     # Pull HuggingFace datasets
│   ├── download_statutes.py        # IPC/BNS/CrPC/Constitution text
│   ├── download_indiankanoon.py    # SC judgements (HTML)
│   ├── parse_pdfs.py               # PDF-to-text pipeline
│   ├── chunk_documents.py          # Run chunking
│   ├── build_vectorstore.py        # Embed + populate ChromaDB
│   └── evaluate.py                 # Run eval suite
├── src/
│   ├── ingestion/                  # PDF parsing, OCR, text cleaning
│   ├── chunking/                   # Legal-aware chunking (3 strategies)
│   ├── embeddings/                 # sentence-transformers + ChromaDB
│   ├── retrieval/                  # Hybrid search + authority reranking
│   ├── generation/                 # Gemini client + citation parsing
│   ├── confidence/                 # Multi-signal confidence scoring
│   └── pipeline.py                 # End-to-end orchestration
├── app/
│   ├── streamlit_app.py            # Main Streamlit app
│   └── components/                 # UI components
├── tests/                          # 179 unit + integration tests
└── requirements.txt
```

## How It Works

1. **Query Classification**: Determines if the query is about a specific section, case, or legal concept
2. **Hybrid Retrieval**: Runs both semantic search (embedding similarity) and keyword search (exact terms), merges via Reciprocal Rank Fusion
3. **Authority Reranking**: Re-ranks results by court hierarchy (SC > HC > Tribunal) and recency
4. **Grounded Generation**: Sends top sources to Gemini with strict instructions to only cite provided documents
5. **Citation Validation**: Verifies every `[SOURCE_N]` maps to a provided source chunk
6. **Confidence Scoring**: Computes 4-signal confidence score to warn users of uncertain answers

## Anti-Hallucination Defenses

| Layer | Mechanism |
|-------|-----------|
| 1. Retrieval grounding | LLM only sees real document chunks |
| 2. Strict prompt | System prompt forbids fabrication |
| 3. Low temperature | 0.1 temperature reduces randomness |
| 4. Citation validation | Post-generation source-link validity check for [SOURCE_N] tags |
| 5. Confidence scoring | Warns users when answer quality is uncertain |

## Running Tests

```bash
source venv/bin/activate
pytest tests/ -v
```

## Running Evaluation

```bash
source venv/bin/activate
python3 scripts/evaluate.py
```

Measures citation validity, cited-claim coverage, citation support, blended citation quality, hallucination rate, retrieval precision, and response rate. Citation validity means source-link validity; it does not by itself prove that a cited source supports the attached legal claim.

## Example Queries

- "What is Section 302 of IPC and its BNS equivalent?"
- "Explain the Kesavananda Bharati case and its significance"
- "What are the fundamental rights under the Indian Constitution?"
- "Explain the doctrine of basic structure"
- "What is the punishment for theft under IPC?"
- "Compare IPC and BNS provisions on murder"
- "What was held in Maneka Gandhi v. Union of India?"

## Team

- Sanskar
- Rishant
- Rahul

Bennett University | May 2026

## Disclaimer

This tool is for research and educational purposes only. Please verify original documents and consult a qualified advocate before relying on any legal information.
