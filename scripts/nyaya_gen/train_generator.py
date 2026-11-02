#!/usr/bin/env python3
"""Fine-tune Nyaya-Gen: SmolLM2-135M-Instruct taught to answer from numbered sources.

Plain PyTorch loop, written to be read:

1. Each example = chat prompt (system rules + sources + question) + target answer.
2. Only the answer tokens are scored: the prompt positions get label -100,
   so the model is never trained to reproduce the sources or the question.
3. Loss = mean cross-entropy of the next token over the answer tokens.
4. AdamW, linear warm-up then cosine decay, gradient clipping, gradient
   accumulation (one example per forward pass, N passes per update).
5. Every ``--eval-every`` updates the dev loss is measured and the best
   checkpoint is saved to ``models/nyaya-gen/``.

Runs on the Apple GPU (MPS) when available, otherwise on the CPU.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import DATA_DIR, PROJECT_ROOT  # noqa: E402

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
END_OF_TURN = "<|im_end|>"


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def encode(tok, example: dict, max_len: int) -> dict | None:
    """Token ids for prompt + answer, with the prompt masked out of the labels."""
    messages = [{"role": "system", "content": example["system"]}, {"role": "user", "content": example["user"]}]
    prompt_text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    prompt_ids = tok(prompt_text, add_special_tokens=False).input_ids
    answer_ids = tok(example["answer"] + END_OF_TURN, add_special_tokens=False).input_ids
    if len(prompt_ids) + len(answer_ids) > max_len:
        return None
    return {
        "input_ids": torch.tensor(prompt_ids + answer_ids),
        "labels": torch.tensor([-100] * len(prompt_ids) + answer_ids),
    }


def lr_at(step: int, total: int, warmup: int, peak: float) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def forward_loss(model, item: dict, device: str, autocast_dtype) -> torch.Tensor:
    input_ids = item["input_ids"].unsqueeze(0).to(device)
    labels = item["labels"].unsqueeze(0).to(device)
    if autocast_dtype is None:
        return model(input_ids=input_ids, labels=labels).loss
    with torch.autocast(device_type=device, dtype=autocast_dtype):
        return model(input_ids=input_ids, labels=labels).loss


@torch.no_grad()
def dev_loss(model, dev: list[dict], device: str, autocast_dtype) -> float:
    model.eval()
    losses = [forward_loss(model, item, device, autocast_dtype).item() for item in dev]
    model.train()
    if device == "mps":
        torch.mps.empty_cache()
    return sum(losses) / len(losses)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base", default=BASE_MODEL)
    p.add_argument("--data", type=Path, default=DATA_DIR / "generator")
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "models" / "nyaya-gen")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad-accum", type=int, default=16, help="examples per optimiser update")
    p.add_argument("--warmup", type=int, default=20, help="warm-up updates")
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--eval-every", type=int, default=60, help="updates between dev evaluations")
    p.add_argument("--eval-examples", type=int, default=80)
    p.add_argument("--max-minutes", type=float, default=100)
    p.add_argument("--max-steps", type=int, default=None, help="stop after this many updates (smoke tests)")
    p.add_argument("--limit", type=int, default=None, help="use only the first N training examples")
    p.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="fp32")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[args.precision]

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float32, attn_implementation="sdpa").to(device)
    # Recompute activations in the backward pass instead of storing them: slower, but fits in 8 GB.
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Base model {args.base}: {n_params / 1e6:.1f}M parameters, device {device}, precision {args.precision}")

    train_rows = load_jsonl(args.data / "train.jsonl")[: args.limit]
    dev_rows = load_jsonl(args.data / "dev.jsonl")[: args.eval_examples]
    train = [e for e in (encode(tok, r, args.max_len) for r in train_rows) if e is not None]
    dev = [e for e in (encode(tok, r, args.max_len) for r in dev_rows) if e is not None]
    lengths = sorted(len(e["input_ids"]) for e in train)
    print(f"Train examples {len(train)} (dropped {len(train_rows) - len(train)} over {args.max_len} tokens), "
          f"dev {len(dev)}, median length {lengths[len(lengths) // 2]} tokens")

    total_updates = args.epochs * len(train) // args.grad_accum
    if args.max_steps:
        total_updates = min(total_updates, args.max_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

    log = {"config": {k: str(v) for k, v in vars(args).items()}, "params": n_params, "device": device,
           "train_examples": len(train), "dev_examples": len(dev), "steps": [], "evals": []}
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.data / "training_log.json"

    initial = dev_loss(model, dev, device, autocast_dtype)
    log["evals"].append({"update": 0, "dev_loss": initial})
    print(f"Dev loss before training: {initial:.4f}")
    best = float("inf")

    start = time.time()
    update, running, seen = 0, 0.0, 0
    order = [i for _ in range(args.epochs) for i in random.sample(range(len(train)), len(train))]
    for idx in order:
        loss = forward_loss(model, train[idx], device, autocast_dtype) / args.grad_accum
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss; rerun with --precision fp32")
        loss.backward()
        running += loss.item()
        seen += 1
        if seen % args.grad_accum:
            continue

        lr = lr_at(update, total_updates, args.warmup, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update += 1

        elapsed = time.time() - start
        log["steps"].append({"update": update, "loss": running, "lr": lr, "elapsed_s": round(elapsed, 1)})
        if update % 5 == 0 or update == 1:
            eta = elapsed / update * (total_updates - update)
            print(f"update {update}/{total_updates}  loss {running:.4f}  lr {lr:.2e}  "
                  f"{elapsed / 60:.1f} min elapsed, ~{eta / 60:.0f} min left", flush=True)
        running = 0.0

        stop = update >= total_updates or elapsed > args.max_minutes * 60
        if update % args.eval_every == 0 or stop:
            current = dev_loss(model, dev, device, autocast_dtype)
            log["evals"].append({"update": update, "dev_loss": current, "elapsed_s": round(elapsed, 1)})
            marker = ""
            if current < best:
                best = current
                model.save_pretrained(args.out)
                tok.save_pretrained(args.out)
                marker = "  (best, saved)"
            print(f"  dev loss {current:.4f}{marker}", flush=True)
            log_path.write_text(json.dumps(log, indent=1))
        if stop:
            break

    log["best_dev_loss"] = best
    log["minutes"] = round((time.time() - start) / 60, 1)
    log_path.write_text(json.dumps(log, indent=1))
    (args.out / "training_config.json").write_text(json.dumps(log["config"] | {"best_dev_loss": best, "params": n_params}, indent=2))
    print(f"Done in {log['minutes']} min, best dev loss {best:.4f}, model in {args.out}")


if __name__ == "__main__":
    main()
