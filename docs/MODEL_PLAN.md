# Nyaya's locally trained model: plan

This is the design the project follows. We train **one** small model on an M2 MacBook Air (8 GB), from the
same public datasets as the index, with every choice made on a dev split, never on the 50 test queries:

| | Nyaya-Gen |
|---|---|
| Job | writes the answer from the numbered sources, with `[SOURCE_N]` citations |
| Base model | `HuggingFaceTB/SmolLM2-135M-Instruct` (decoder-only, 134.5M parameters) |
| Replaces | the NVIDIA NIM API model (Nemotron), so the whole system runs offline |
| Switch | `LLM_BACKEND=api/local` (model folder: `NYAYA_GEN_PATH`, default `models/nyaya-gen/`) |

It is trained by `scripts/run_training.sh` and evaluated by `scripts/run_evaluation.sh`. Everything else in
the pipeline (MiniLM embeddings, hybrid search, RRF, authority re-ranking, citation validation, confidence) is
unchanged and untrained.

---

## 1. What we train and why

**Model:** `HuggingFaceTB/SmolLM2-135M-Instruct` (Allal et al. 2025), a decoder-only (causal) language model:
30 layers, hidden size 576, 9 attention heads (3 key-value heads), vocabulary 49,152, **134.5M parameters**.
It is **fully fine-tuned** (every weight is trained, no adapters) to write the answer from the numbered sources,
ending each sentence with the `[SOURCE_N]` tag of the source it came from.

**Why:** the students wanted the answer writer itself to be a model they trained locally. With Nyaya-Gen, the
whole system (search, authority re-ranking, writing, citation checking) runs on the laptop with no API key and
no network.

**Why it can work at 135M (the honest framing):**
- Nyaya-Gen is **not asked to know the law**. Every legal fact must come from the passages in the prompt.
  Its job is grounded generation: pick the passage that answers the question, restate it in a few sentences,
  and attach that passage's number. This is mostly selecting and copying, a pattern a small model can learn
  from a few thousand examples.
- What it cannot do as well as a 550B model: its answers are shorter and plainer, it was only trained to
  restate **one** source per answer, so it does not compare or combine sources (prompt rule 6), and a 135M
  model can still copy the wrong passage or garble a sentence. The citation validation and confidence score
  check its output exactly as they check the API model's.
- So the claim is not "a 135M model matches a 550B model". The claim is: "a model we trained on the laptop
  can write cited answers from retrieved sources, and here is how far it is from the 550B model on the same
  six answer metrics."

## 2. Where it sits in the pipeline

```
... -> top K = 4 (+ IPC/BNS table) -> PromptBuilder (system rules + numbered sources + question)
   -> LLM_BACKEND=api   : LLMClient (NVIDIA NIM, temperature 0.1, 1,024 tokens)
   -> LLM_BACKEND=local : LocalGenerator (Nyaya-Gen; each source cut to 90 words; greedy; 256 tokens)
   -> citation validation -> confidence score
```
Same interface: `LocalGenerator.generate(prompt, system_prompt)` is a drop-in replacement for
`LLMClient.generate` (`src/pipeline.py` picks one from `config/settings.py: LLM_BACKEND`). The prompt is the
pipeline's own prompt; the only change is that each source passage is cut to 90 words, as in training.

## 3. Training data (`scripts/prepare_generator_data.py`, no new data, no vector index needed)

Built from the statute Q&A (viber1/indian-law-dataset) and Constitution Q&A (nisaar/Constitution_of_India)
datasets: 23,939 Q&A records, processed exactly as they are indexed. Each example = one prompt in **exactly**
the pipeline's format (`PromptBuilder`: system rules + numbered sources + question) + one target answer.

| Kind | Prompt | Target answer |
|---|---|---|
| `qa` | the record's question; its own record (formatted as indexed) hidden at a random position among 2-4 random distractor passages (Q&A records and judgment passages) | up to 4 sentences / 120 words of the record's answer, each sentence ending with `[SOURCE_k]`, k = the record's position |
| `mapping` | an IPC section question from 3 templates ("What is the BNS equivalent of Section {s} IPC?", ...); the IPC/BNS table chunk from `IpcBnsMapper.as_source_chunk` among the sources | a fixed sentence giving both section numbers and titles, cited, plus "The BNS replaced the IPC from 1 July 2024" |
| `refusal` (~7%) | a question whose own record is left out | "The retrieved sources do not address this question." |

- Every passage is cut to 90 words, so prompts fit in 8 GB during training.
- **Leakage filter:** a question is dropped if its normalised text equals one of the 50 test queries or its
  content-word Jaccard similarity to one is >= 0.5: **572 removed**. Mapping rows whose IPC section number
  appears in a test query are skipped.
- **Split by document:** 94% train, 3% dev, 3% test (a record never appears in two splits).

| Split | qa | mapping | refusal | total |
|---|---|---|---|---|
| train (capped at 4,000; all mapping examples kept) | 3,590 | 141 | 269 | 4,000 |
| dev | 179 | 0 | 21 | 200 |
| test (held out, not used yet) | 181 | 0 | 19 | 200 |

(From `data/generator/data_stats.json`. At training time, examples longer than 1,024 tokens are dropped:
**2,585 of the 4,000** train examples are used.)

## 4. Training setup (`scripts/train_generator.py`, as run by `scripts/run_training.sh`)

| Setting | Value |
|---|---|
| Base model | SmolLM2-135M-Instruct (134.5M params, 30 layers, hidden 576), full fine-tuning |
| Input | chat template: system rules + user prompt (sources + question) + target answer + `<|im_end|>`, at most 1,024 tokens |
| Loss | next-token cross-entropy on the **answer tokens only** (prompt labels = -100), averaged over answer tokens |
| Batch | 1 example per forward pass, gradient accumulation 8 (8 examples per update) |
| Optimiser | AdamW, lr 1e-4, betas (0.9, 0.95), weight decay 0.01 |
| Schedule | 20 linear warm-up updates, then cosine decay to 0 |
| Epochs | 1 (2,585 examples = 323 updates), gradient clipping 1.0, seed 42 |
| Precision | fp32 on MPS with **gradient checkpointing** |
| Dev check | dev loss on the first 40 dev examples (24 fit in 1,024 tokens), before training and every 50 updates |
| Checkpoint | best dev loss saved to `models/nyaya-gen/` |
| Time limit | `--max-minutes 75` (expected ~70 min); if reached, it evaluates, keeps the best checkpoint and stops |

**Engineering finding (8 GB limit).** Two earlier attempts ran out of GPU memory on the 8 GB M2
(`logs/training_failed_oom.log`): bf16 autocast, and fp32 without gradient checkpointing. What fits is fp32
with gradient checkpointing (activations are recomputed in the backward pass instead of being stored: slower,
but much less memory), one example per forward pass, prompts of at most 1,024 tokens and passages cut to 90
words. These limits come from the hardware, not from a design preference, and are reported as such.

## 5. Inference (`src/generation/local_generator.py`)

- `LocalGenerator.generate(prompt, system_prompt)`: same call as `LLMClient`; selected with `LLM_BACKEND=local`
  (in `.env` or the environment); model folder from `NYAYA_GEN_PATH` (default `models/nyaya-gen/`).
- Sources cut to 90 words (`LOCAL_GENERATOR_SOURCE_WORDS`), as in training.
- Greedy decoding (always the most probable next token), repetition penalty 1.1, at most 256 new tokens
  (`LOCAL_GENERATOR_MAX_NEW_TOKENS`), stops at `<|im_end|>`.
- Runs on MPS (or CPU). If the model folder is missing, the pipeline logs a warning and returns sources only.

## 6. Evaluation (`scripts/run_evaluation.sh`)

1. **Dev loss** before training (= the un-fine-tuned SmolLM2-135M-Instruct) and of the kept checkpoint
   (`data/generator/training_log.json`).
2. **Run D:** Nyaya-Gen, fully local (`LLM_BACKEND=local`), the 50 test queries, all six answer metrics
   + latency (`data/evaluation/eval_results_local.json`).
3. **Retrieval tables** III and VI-VIII re-computed by `scripts/retrieval_experiments.py`
   (`data/evaluation/retrieval_experiments.json`); they do not depend on the answer writer.
4. **Run D0:** the base SmolLM2-135M-Instruct, not fine-tuned (`NYAYA_GEN_PATH=models/smollm2-base`), same
   queries and metrics (`data/evaluation/eval_results_local_base.json`). Shows what the base model does
   without our training.
5. Compare D with D0 (effect of fine-tuning) and with Run C from the report (Nemotron 550B via API; the cost
   of writing locally with a model ~4,000 times smaller). Retrieval is the same in C, D0 and D, so the
   difference comes from the writer.

## 7. What actually happened in training

| Quantity | Value |
|---|---|
| Training examples built / used | 4,000 built (3,590 qa, 141 mapping, 269 refusal); 2,585 used, 1,415 dropped for exceeding 1,024 tokens (median length 931 tokens) |
| Dev examples for the dev loss | 24 (of the first 40) |
| Planned | 1 epoch = 323 updates of 8 examples |
| Speed | about 0.8 min per update (fp32, gradient checkpointing, MPS) |
| Done | about 80 updates in about 65 minutes; then the 8 GB Mac ran out of memory (swap full) and the run stopped |
| Kept checkpoint | update 50 (dev checks are every 50 updates), i.e. about 400 examples seen, ~15% of the epoch |
| Dev loss | 0.8325 before training -> 0.0334 at update 50 (best, saved) |
| Training loss | 0.98 at update 1 -> about 0.005-0.02 at updates 50-80 |

Honest reading: the training was limited by the hardware and time, not by design. The model learned the fixed
answer format very quickly (the dev loss is already very low at update 50), but it has seen only ~400 examples,
so the end-to-end runs D / D0 are the real test.

## 8. Files

| File | Purpose |
|---|---|
| `scripts/prepare_generator_data.py` | builds `data/generator/{train,dev,test}.jsonl` and `data_stats.json` |
| `scripts/train_generator.py` | fine-tunes SmolLM2-135M-Instruct, logs to `data/generator/training_log.json` |
| `src/generation/local_generator.py` | `LocalGenerator` used by the pipeline when `LLM_BACKEND=local` |
| `config/settings.py` (end) | `LLM_BACKEND`, `LOCAL_GENERATOR_*`, `NYAYA_GEN_PATH` |
| `scripts/run_training.sh` | builds the data if missing, then trains Nyaya-Gen (detached) |
| `scripts/run_evaluation.sh` | Run D, retrieval tables (III, VI-VIII), then Run D0 |
| `models/nyaya-gen/` | the trained model (git-ignored, ~540 MB in fp32) |
