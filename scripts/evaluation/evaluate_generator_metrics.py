#!/usr/bin/env python3
"""Standard LLM evaluation metrics for Nyaya-Gen on its held-out test split.

The 200 examples of ``data/generator/test.jsonl`` were never seen in training
and each has a gold answer, so the usual reference-based metrics apply:

* text quality  - Exact Match and token F1 (SQuAD style), ROUGE-1/2/L, BLEU and
                  chrF (sacreBLEU), BERTScore F1 and MiniLM cosine similarity.
                  Computed with the [SOURCE_k] tags removed, so they measure
                  the wording only.
* language model - perplexity of the gold answer (teacher forced, answer tokens only).
* citations     - format validity, correct-source accuracy and citation
                  precision / recall / F1 against the gold source k.
* refusals      - accuracy, precision, recall and F1 of refusing exactly when
                  no source is relevant.

Both Nyaya-Gen and the base SmolLM2-135M-Instruct (not fine-tuned) are run,
so the effect of fine-tuning is visible. Predictions are cached in
``data/generator/predictions_<name>.jsonl`` so a stopped run resumes.

Usage:
    python scripts/evaluation/evaluate_generator_metrics.py            # both models, 200 examples
    python scripts/evaluation/evaluate_generator_metrics.py --limit 50
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import math
import re
import string
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import DATA_DIR, LOCAL_GENERATOR_BASE_MODEL, LOCAL_GENERATOR_PATH  # noqa: E402

GEN_DIR = DATA_DIR / "generator"
BASE_PATH = _PROJECT_ROOT / "models" / "smollm2-base"
TAG = re.compile(r"\[SOURCE_(\d+)\]")
REFUSAL_HINTS = ("do not address", "does not address", "not covered by the retrieved sources",
                 "do not contain", "does not contain", "no information")


def _load_train_module():
    spec = importlib.util.spec_from_file_location("train_generator", _PROJECT_ROOT / "scripts" / "nyaya_gen" / "train_generator.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Text normalisation (SQuAD)
def normalize(text: str) -> str:
    text = TAG.sub(" ", text).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def strip_tags(text: str) -> str:
    return " ".join(TAG.sub("", text).split())


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = collections.Counter(p) & collections.Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def n_sources(user_prompt: str) -> int:
    return len(re.findall(r"^\[SOURCE_\d+\]", user_prompt, flags=re.M)) or len(set(TAG.findall(user_prompt)))


def is_refusal(answer: str) -> bool:
    low = answer.lower()
    return not TAG.search(answer) and any(h in low for h in REFUSAL_HINTS)


def generate_all(name: str, model_path: Path, examples: list[dict]) -> list[dict]:
    from src.generation.local_generator import LocalGenerator

    cache = GEN_DIR / f"predictions_{name}.jsonl"
    done = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            row = json.loads(line)
            done[row["i"]] = row
    todo = [i for i in range(len(examples)) if i not in done]
    if todo:
        gen = LocalGenerator(model_path=model_path)
        with open(cache, "a") as f:
            for n, i in enumerate(todo, 1):
                ex = examples[i]
                start = time.perf_counter()
                answer = gen.generate(ex["user"], system_prompt=ex["system"])
                row = {"i": i, "prediction": answer, "seconds": round(time.perf_counter() - start, 2)}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                done[i] = row
                if n % 10 == 0 or n == len(todo):
                    print(f"  [{name}] {n}/{len(todo)} generated", flush=True)
        del gen
    return [done[i] for i in range(len(examples))]


def perplexity(model_path: Path, examples: list[dict]) -> float:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    train = _load_train_module()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device).eval()
    nll, count = 0.0, 0
    with torch.inference_mode():
        for ex in examples:
            item = train.encode(tok, ex, 4096)
            if item is None:
                continue
            ids = item["input_ids"].unsqueeze(0).to(device)
            labels = item["labels"].unsqueeze(0).to(device)
            n = int((labels[:, 1:] != -100).sum())
            loss = model(input_ids=ids, labels=labels).loss
            nll += float(loss) * n
            count += n
    del model
    return math.exp(nll / count)


def score(examples: list[dict], preds: list[dict]) -> dict:
    import sacrebleu
    from bert_score import score as bert_score
    from rouge_score import rouge_scorer
    from sentence_transformers import SentenceTransformer, util

    rs = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    pred_txt = [strip_tags(p["prediction"]) for p in preds]
    gold_txt = [strip_tags(e["answer"]) for e in examples]

    out: dict = {"examples": len(examples)}
    out["exact_match"] = sum(normalize(p) == normalize(g) for p, g in zip(pred_txt, gold_txt)) / len(examples)
    out["token_f1"] = sum(token_f1(p, g) for p, g in zip(pred_txt, gold_txt)) / len(examples)
    for key in ("rouge1", "rouge2", "rougeL"):
        out[f"{key}_f1"] = sum(rs.score(g, p)[key].fmeasure for p, g in zip(pred_txt, gold_txt)) / len(examples)
    out["bleu"] = sacrebleu.corpus_bleu(pred_txt, [gold_txt]).score
    out["chrf"] = sacrebleu.corpus_chrf(pred_txt, [gold_txt]).score
    _, _, f1 = bert_score([p or "." for p in pred_txt], gold_txt, model_type="distilbert-base-uncased",
                          lang="en", verbose=False)
    out["bertscore_f1"] = float(f1.mean())
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb_p = st.encode(pred_txt, convert_to_tensor=True, normalize_embeddings=True)
    emb_g = st.encode(gold_txt, convert_to_tensor=True, normalize_embeddings=True)
    out["semantic_similarity"] = float(util.cos_sim(emb_p, emb_g).diagonal().mean())

    # Citations, on the examples that have a relevant source
    tp = fp = fn = valid = correct = answerable = 0
    for ex, p in zip(examples, preds):
        if ex["kind"] == "refusal":
            continue
        answerable += 1
        gold = {ex["positive_index"]}
        cited = {int(k) for k in TAG.findall(p["prediction"])}
        limit = n_sources(ex["user"])
        if cited and all(1 <= k <= limit for k in cited):
            valid += 1
        if cited == gold:
            correct += 1
        tp += len(cited & gold)
        fp += len(cited - gold)
        fn += len(gold - cited)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    out["citation_format_validity"] = valid / answerable
    out["correct_source_accuracy"] = correct / answerable
    out["citation_precision"] = prec
    out["citation_recall"] = rec
    out["citation_f1"] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    # Refusal as a binary decision (positive = should refuse)
    tp = fp = fn = tn = 0
    for ex, p in zip(examples, preds):
        gold, pred = ex["kind"] == "refusal", is_refusal(p["prediction"])
        tp += gold and pred
        fp += (not gold) and pred
        fn += gold and not pred
        tn += (not gold) and not pred
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    out["refusal_accuracy"] = (tp + tn) / len(examples)
    out["refusal_precision"] = prec
    out["refusal_recall"] = rec
    out["refusal_f1"] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    out["refusal_confusion"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}

    out["mean_answer_words"] = sum(len(t.split()) for t in pred_txt) / len(examples)
    out["mean_generation_seconds"] = sum(p["seconds"] for p in preds) / len(preds)
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--models", nargs="+", default=["nyaya-gen", "base"], choices=["nyaya-gen", "base"])
    parser.add_argument("--output", default=str(DATA_DIR / "evaluation" / "generator_metrics.json"))
    args = parser.parse_args()

    examples = [json.loads(line) for line in (GEN_DIR / "test.jsonl").read_text().splitlines()]
    examples = examples[: args.limit] if args.limit else examples
    if "base" in args.models and not (BASE_PATH / "config.json").exists():
        from transformers import AutoModelForCausalLM, AutoTokenizer
        AutoTokenizer.from_pretrained(LOCAL_GENERATOR_BASE_MODEL).save_pretrained(BASE_PATH)
        AutoModelForCausalLM.from_pretrained(LOCAL_GENERATOR_BASE_MODEL).save_pretrained(BASE_PATH)

    paths = {"nyaya-gen": Path(LOCAL_GENERATOR_PATH), "base": BASE_PATH}
    results = {"test_split": str(GEN_DIR / "test.jsonl"),
               "kinds": dict(collections.Counter(e["kind"] for e in examples)), "models": {}}
    for name in args.models:
        print(f"[{time.strftime('%H:%M:%S')}] {name}: generating", flush=True)
        preds = generate_all(name, paths[name], examples)
        print(f"[{time.strftime('%H:%M:%S')}] {name}: perplexity", flush=True)
        ppl = perplexity(paths[name], examples)
        print(f"[{time.strftime('%H:%M:%S')}] {name}: scoring", flush=True)
        results["models"][name] = {"perplexity": round(ppl, 3), **score(examples, preds)}
        print(json.dumps(results["models"][name], indent=1), flush=True)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
