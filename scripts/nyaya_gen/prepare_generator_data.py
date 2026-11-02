#!/usr/bin/env python3
"""Build the fine-tuning data for Nyaya-Gen, the small local answer writer.

Each example is one prompt in exactly the format the pipeline sends to the LLM
(``PromptBuilder``: system rules + numbered sources + question) and one target
answer that cites the right source as ``[SOURCE_k]``. Three kinds of examples:

* ``qa``      - a question from the statute / Constitution Q&A datasets. Its own
                record (formatted as it is indexed) is hidden among random
                distractor passages; the target restates the record's answer
                sentence by sentence, each sentence cited.
* ``mapping`` - a question about an IPC section; the IPC/BNS table chunk (built by
                ``IpcBnsMapper.as_source_chunk``) is one of the sources.
* ``refusal`` - a question whose own record is left out, so no source is relevant;
                the target is the refusal sentence from the system prompt.

Questions that resemble one of the 50 evaluation queries are dropped, and
mapping rows whose IPC section appears in an evaluation query are skipped, so
the test set is never seen in training.

Output: ``data/generator/{train,dev,test}.jsonl`` and ``data_stats.json``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import DATA_DIR, EVAL_DIR, RAW_DIR  # noqa: E402
from src.generation.prompt_builder import PromptBuilder  # noqa: E402
from src.retrieval.ipc_bns import IpcBnsMapper  # noqa: E402

OUT_DIR = DATA_DIR / "generator"
REFUSAL = "The retrieved sources do not address this question."
SOURCE_WORDS = 90    # each passage is cut to this many words so training fits in 8 GB
ANSWER_SENTENCES = 4
ANSWER_WORDS = 120

_STOP = set("""a an the of to in on for and or is are was were be by with what which who
whom how does do did under as at from that this it its into about between explain define
describe give state india indian law legal""".split())


def _load_prepare_module():
    """Import scripts/data/prepare_hf_data.py so the Q&A records match the index exactly."""
    path = _PROJECT_ROOT / "scripts" / "data" / "prepare_hf_data.py"
    spec = importlib.util.spec_from_file_location("prepare_hf_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 1}


def _truncate(text: str, words: int) -> str:
    parts = text.split()
    return text if len(parts) <= words else " ".join(parts[:words]) + " ..."


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 15]


def _cite(sentences: list[str], k: int) -> str:
    """Attach ``[SOURCE_k]`` before the final punctuation of every sentence."""
    out = []
    for s in sentences:
        s = s.rstrip()
        if s[-1] in ".!?":
            out.append(f"{s[:-1]} [SOURCE_{k}]{s[-1]}")
        else:
            out.append(f"{s} [SOURCE_{k}].")
    return " ".join(out)


def _answer_part(doc_text: str) -> str:
    for marker in ("\n\nLegal Answer: ", "\n\nAnswer: "):
        if marker in doc_text:
            return doc_text.split(marker, 1)[1]
    return doc_text


def _question_part(doc_text: str) -> str:
    first = doc_text.split("\n\n", 1)[0]
    return first.split(": ", 1)[1] if ": " in first else first


def _is_leak(question: str, test_word_sets: list[set[str]], test_norm: set[str]) -> bool:
    norm = " ".join(re.findall(r"[a-z0-9]+", question.lower()))
    if norm in test_norm:
        return True
    words = _content_words(question)
    if not words:
        return False
    for tw in test_word_sets:
        if tw and len(words & tw) / len(words | tw) >= 0.5:
            return True
    return False


def build(seed: int, max_train: int) -> None:
    rng = random.Random(seed)
    prep = _load_prepare_module()
    builder = PromptBuilder()
    mapper = IpcBnsMapper()

    qa_docs = prep.process_indian_law_dataset(RAW_DIR) + prep.process_constitution_dataset(RAW_DIR)
    distractor_docs = (
        qa_docs
        + prep.process_sc_judgements_chunked(RAW_DIR)
        + prep.process_legal_dataset_india(RAW_DIR)
    )
    # Pre-split datasets keep doc_id / doc_type under "metadata"; lift them to the top level.
    distractor_docs = [
        {
            "doc_id": d.get("doc_id") or d.get("metadata", {}).get("doc_id") or d.get("chunk_id"),
            "doc_type": d.get("doc_type") or d.get("metadata", {}).get("doc_type", "judgement"),
            "text": d.get("text") or "",
        }
        for d in distractor_docs
    ]
    distractor_docs = [d for d in distractor_docs if len(d["text"].split()) >= 30]

    test_queries = [q["query"] for q in json.loads((EVAL_DIR / "test_queries.json").read_text()) if q["query"].strip()]
    test_word_sets = [_content_words(q) for q in test_queries]
    test_norm = {" ".join(re.findall(r"[a-z0-9]+", q.lower())) for q in test_queries}
    test_sections = {s for q in test_queries for s in re.findall(r"\b(\d{2,3}[A-Z]?)\b", q)}

    def as_chunk(doc: dict) -> dict:
        return {"text": _truncate(doc["text"], SOURCE_WORDS), "metadata": {"doc_type": doc["doc_type"], "doc_id": doc["doc_id"]}}

    def distractors(n: int, exclude: str) -> list[dict]:
        picked = []
        while len(picked) < n:
            d = rng.choice(distractor_docs)
            if d["doc_id"] != exclude:
                picked.append(as_chunk(d))
        return picked

    def make(question: str, positives: list[dict], answer_for_k, kind: str, doc_id: str) -> dict:
        n_sources = rng.choice([3, 4, 4, 4, 5])
        sources = distractors(n_sources - len(positives), exclude=doc_id)
        k = rng.randint(1, len(sources) + 1)
        for p in positives:
            sources.insert(k - 1, p)
        system, user = builder.build_prompt(question, sources)
        return {"kind": kind, "doc_id": doc_id, "system": system, "user": user,
                "answer": answer_for_k(k) if positives else REFUSAL, "positive_index": k if positives else None}

    stats = {"qa_records": len(qa_docs), "removed_by_leakage_filter": 0}
    examples: list[dict] = []
    rng.shuffle(qa_docs)
    for doc in qa_docs:
        question = _question_part(doc["text"])
        if _is_leak(question, test_word_sets, test_norm):
            stats["removed_by_leakage_filter"] += 1
            continue
        chunk = as_chunk(doc)
        sents = _sentences(_answer_part(chunk["text"]).rstrip(" ."))[:ANSWER_SENTENCES]
        if not sents:
            continue
        while len(" ".join(sents).split()) > ANSWER_WORDS and len(sents) > 1:
            sents = sents[:-1]
        if rng.random() < 0.08:
            examples.append(make(question, [], None, "refusal", doc["doc_id"]))
        else:
            examples.append(make(question, [chunk], lambda k, s=sents: _cite(s, k), "qa", doc["doc_id"]))

    mapping_rows = [r for r in json.loads(mapper_path().read_text())["mappings"]]
    for row in mapping_rows:
        if row["ipc_section"] in test_sections:
            continue
        chunk = mapper.as_source_chunk([row])
        bns = ", ".join(row.get("bns_sections", []))
        for template in (
            "What is the BNS equivalent of Section {s} IPC?",
            "Which section of the Bharatiya Nyaya Sanhita replaced IPC Section {s}?",
            "Explain Section {s} of the IPC and the corresponding BNS provision.",
        ):
            def answer(k, row=row, bns=bns):
                if bns:
                    return (f"Section {row['ipc_section']} of the Indian Penal Code ({row['ipc_title']}) corresponds to "
                            f"Section {bns} of the Bharatiya Nyaya Sanhita, 2023 ({row['bns_title']}) [SOURCE_{k}]. "
                            f"The BNS replaced the IPC from 1 July 2024 [SOURCE_{k}].")
                return (f"Section {row['ipc_section']} of the Indian Penal Code ({row['ipc_title']}) has no equivalent "
                        f"in the Bharatiya Nyaya Sanhita: {row['bns_title']} [SOURCE_{k}].")
            examples.append(make(template.format(s=row["ipc_section"]), [chunk], answer, "mapping", f"map_{row['ipc_section']}"))

    # Split by document so no record appears in two splits.
    doc_ids = sorted({e["doc_id"] for e in examples})
    rng.shuffle(doc_ids)
    n = len(doc_ids)
    dev_ids, test_ids = set(doc_ids[: int(0.03 * n)]), set(doc_ids[int(0.03 * n): int(0.06 * n)])
    splits = {"train": [], "dev": [], "test": []}
    for e in examples:
        split = "dev" if e["doc_id"] in dev_ids else "test" if e["doc_id"] in test_ids else "train"
        splits[split].append(e)
    # Keep every IPC/BNS table example (there are few), fill the rest with Q&A and refusals.
    mapping_train = [e for e in splits["train"] if e["kind"] == "mapping"]
    others = [e for e in splits["train"] if e["kind"] != "mapping"]
    rng.shuffle(others)
    splits["train"] = mapping_train + others[: max_train - len(mapping_train)]
    rng.shuffle(splits["train"])
    splits["dev"], splits["test"] = splits["dev"][:200], splits["test"][:200]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with open(OUT_DIR / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats[name] = {kind: sum(r["kind"] == kind for r in rows) for kind in ("qa", "mapping", "refusal")}
        stats[name]["total"] = len(rows)
    (OUT_DIR / "data_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def mapper_path() -> Path:
    from config.settings import IPC_BNS_MAPPING_PATH
    return IPC_BNS_MAPPING_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train", type=int, default=6000, help="cap on training examples")
    args = parser.parse_args()
    build(args.seed, args.max_train)


if __name__ == "__main__":
    main()
