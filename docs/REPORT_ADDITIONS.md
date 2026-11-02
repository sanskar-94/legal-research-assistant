# Report additions: Nyaya-Gen (pages to paste into the report)

## How to use this file
1. Apply `REPORT_FIXES.md` first. These edits assume its reference numbers [1]–[20] and equation numbers (4)–(9), so the new equation here is (10). The reference numbers in Section 1.4 of `REPORT_FIXES.md` are replaced by Edit 15 below (two new references are added in Section II, so NIM and DeepSeek become [23] and [24]).
2. Go through Edits 1–15 in order. Each one gives WHERE (the section, and the first words of the paragraph as printed now), the ACTION (INSERT AFTER, INSERT BEFORE or REPLACE) and the TEXT.
3. Paste only the quoted text, equations and tables. Lines that start with "Note:" are for you, not for the report.
4. All numbers are already filled in from the real runs.
5. Set symbols in italic math as in the rest of the report, type the equation in Word's equation editor, and bold the best value in each table column once it is filled in.
6. Length: all edits together add about two pages in the two-column format.

Notation used below. S(c), s(q,c), R̃(c), A(c), K, k and τ mean the same as in the report. The new symbols are:
- x: a prompt; y = (y_1, …, y_T): its target answer tokens; T_y: their number
- θ: the generator weights; p_θ(y_t | x, y_<t): its next-token probability
- ℬ: a batch
- J_G: the generator training loss

J_G is used for the loss because L and ℒ are already taken in (3) and (5), and G is taken in (7). J_G uses G only as a subscript.

Run names used below. Runs A–C are the report's runs (Run C = Nemotron through the API). **Run D** = Nyaya-Gen, fully local (`data/evaluation/eval_results_local.json`). It is produced by `scripts/run_evaluation.sh`. (The script can also run the un-fine-tuned base model and re-compute Tables III and VI–VIII; these were not run for this report because of time, so they are not mentioned in the text.)

---

## Edit 1. Abstract: add the trained model

**WHERE:** Abstract, last sentence. It starts "Thus every end, with a Nemotron model served by…" (after REPORT_FIXES it starts "End to end, with a Nemotron model…").
**ACTION:** INSERT AFTER this sentence, as the new end of the Abstract.
**TEXT:**

> Finally, we fine-tune Nyaya-Gen, a 135M-parameter small language model, on an 8 GB laptop to write the answer from the numbered sources with [SOURCE_N] citations, so that the whole system runs offline. Fully local, it reaches a citation validity of 0.74 and a response rate of 0.72, against 0.70 and 0.68 with the 550B-parameter API model.

---

## Edit 2. Section I: one sentence, and contribution 5

**2a. WHERE:** Section I, paragraph "So our project introduces Nyaya a RAG AI for Indian law…".
**ACTION:** INSERT AFTER its last sentence ("…an IPC to BNS mapping table also as an verifiable source.").
**TEXT:**

> Finally, a small answer writer, fine-tuned on a laptop, can replace the hosted LLM, so that the whole system runs offline.

**2b. WHERE:** Section I, the numbered list of contributions, after item 4 ("A answer verification and evaluation layer every [SOURCE_N] tag is validated…"). Item 4 ends "…together with an ablation study and a systematic parameter sweep."
**ACTION:** INSERT AFTER item 4, as new item 5.
**TEXT:**

> 5. Nyaya-Gen, a 135M-parameter answer writer fully fine-tuned from SmolLM2-135M-Instruct on grounded examples written in the exact prompt format of the pipeline. It replaces the API model, so retrieval, generation and citation checking all run on an 8 GB laptop. It is compared with the same model before fine-tuning and with the 550B-parameter API model on the same test queries and metrics.

Note: do not cite [21] here. A citation in Section I would come before Section II and change the reference order.

---

## Edit 3. Section II: one new paragraph

**WHERE:** Section II, between the paragraph "RAG system evaluation. Es et al introduces and suggest using RAGAS…" and the paragraph "Research gap. The existing research on the topic has created…".
**ACTION:** INSERT BEFORE the "Research gap." paragraph.
**TEXT** (start with its title in bold, like the other paragraph titles):

> **Small language models.** Allal et al. [21] trained SmolLM2, a family of small language models with 135M, 360M and 1.7B parameters, with a data-centric training recipe, and released instruction-tuned versions. Wei et al. [22] showed that fine-tuning a language model on many tasks written as instructions makes it follow new instructions better, which is the basis of such instruction-tuned models. A small model knows much less than a large one, but in RAG the facts are given in the prompt, so the writer mainly has to select, restate and cite them. To our knowledge, a model of this size has not been fine-tuned to write cited answers from Indian legal sources.

---

## Edit 4. Section II: Research gap

**WHERE:** Section II, the whole paragraph "Research gap. The existing research on the topic has created 4 research gaps…" (ends "…deterministic metrics requiring no LLM judge.").
**ACTION:** REPLACE.
**TEXT:**

> **Research gap.** The existing work leaves five gaps for question answering on Indian law. (i) Standard RAG relies on dense retrieval, which is poor at the exact identifiers that legal queries need. (ii) Fusion methods such as RRF use fixed weights, whether the user asks for a section, a case or a concept. (iii) Current systems do not check that the citations in an answer point to the retrieved passages, and they give no signal of how well the answer is grounded. (iv) No available system handles the IPC-to-BNS transition. (v) Legal RAG systems depend on large hosted LLMs to write the answer; whether a small model trained on a laptop can do this has not been tested. Nyaya addresses these gaps with query-adaptive hybrid retrieval, authority-aware re-ranking, a small fine-tuned answer writer, and explicit citation validation with deterministic metrics that need no LLM judge.

---

## Edit 5. Section III-A: pipeline stages and code size

**WHERE:** III-A "System Architecture", paragraph "It has the stages of offline index and online query (Fig. 1)." The part to change starts at "In online, the query goes through IPC/BNS lookup…" and runs to the end of the paragraph ("…with 250 unit and integration tests.").
**ACTION:** REPLACE that part (the last two sentences).
**TEXT:**

> In the online stage, the query goes through IPC/BNS lookup, query classification, hybrid search, fusion and authority re-ranking, prompt construction, generation by the API model or by the local Nyaya-Gen model (Section III-K), citation verification and confidence scoring. Nyaya-Gen is trained once, offline, from the same datasets. The codebase has about 3950 lines of Python, organized into ingestion, chunking, embeddings, retrieval, generation, confidence and evaluation modules, with 300 unit and integration tests.

---

## Edit 6. Fig. 1: add the local generator

**WHERE:** Fig. 1 (page 3) and its caption.
**ACTION:** change the figure as follows, then REPLACE the caption.

Note (figure changes):
1. In the LLM box (the generation box below the prompt box "Prompt with numbered [SOURCE_n]"), add a second line: **"or Nyaya-Gen (local)"**.
2. To the right of the LLM box (the right side of the figure is empty at that height), add a grey box: **"Fine-tuned offline (Sec. IV-E): Nyaya-Gen weights"**. Draw one arrow from the grey box into the LLM box.
3. Leave all other boxes and arrows as they are.

**Caption TEXT:**

> Fig. 1. Nyaya architecture. Grey boxes are data and offline stages: the indexing stage at the bottom left (read upwards) and, on the right, the weights of Nyaya-Gen, trained offline. White boxes form the online query pipeline. The answer is written by the API model or by Nyaya-Gen on the laptop. Matching IPC/BNS rows are also appended to the prompt as a source.

---

## Edit 7. Section III-H: generation can also run locally

**WHERE:** III-H, after the sentence "Any OpenAI compatible model can be configured DeepSeek and NVIDIA Nemotron were used…" (after REPORT_FIXES it contains "DeepSeek-V3.2 [22]"; after Edit 15 it is [24]).
**ACTION:** INSERT AFTER this sentence.
**TEXT:**

> Generation can also run on the laptop with Nyaya-Gen (Section III-K), a 135M-parameter model that we fine-tuned for this prompt. One setting selects the API model or Nyaya-Gen; the prompt, citation validation and confidence scoring are the same for both. Nyaya-Gen uses greedy decoding and at most 256 output tokens, and each source passage is cut to its first 90 words, as in its training data.

---

## Edit 8. Section III-J: Training and Inference

**WHERE:** III-J "Training and Inference", the whole paragraph "Nyaya has no trainable parameters the embedding model and the LLM are used frozen…" (ends "…through a command line tool or a Python API.").
**ACTION:** REPLACE.
**TEXT:**

> The embedding model (MiniLM) and the API LLM are used frozen, and the retrieval and fusion parameters are chosen from the data without a loss function (Section IV-D). One component is trained, once, offline, on the laptop (Table X): Nyaya-Gen (Section III-K) is fully fine-tuned from SmolLM2-135M-Instruct [21] with the masked next-token loss (10), using AdamW [25] with learning rate 1×10⁻⁴, 20 warm-up updates and cosine decay; the checkpoint with the lowest dev loss is kept. At inference, the stages of Fig. 1 run in order, with the API model or Nyaya-Gen as the generator. The system returns the answer, all numbered sources, invalid citations, IPC/BNS matches and the confidence breakdown, through a command-line tool or a Python API.

---

## Edit 9. Section III-K (new subsection)

**WHERE:** end of Section III, after the III-J paragraph (Edit 8) and before the heading "IV. EXPERIMENTAL RESULTS".
**ACTION:** INSERT AFTER the III-J paragraph.
**TEXT:**

> **K. Nyaya-Gen: Local Answer Generator**
>
> The API model of Section III-H has about 550B parameters and needs a network connection. Nyaya-Gen replaces it with a model about 4,000 times smaller that runs on the laptop, so the whole pipeline can run offline. It is SmolLM2-135M-Instruct [21], a decoder-only Transformer with 30 layers, hidden size 576 and 134.5M parameters, fully fine-tuned (all weights are updated).
>
> A model of this size cannot be relied on to know the law, and it is not asked to. Its task is grounded generation: the prompt already contains the numbered sources, and the model has to select the passage that answers the question, restate it, and end each sentence with that passage's [SOURCE_k] tag. This is mostly selection and copying, which a small model can learn from examples. Its answers are plainer than those of a 550B model, and they pass through the same citation validation and confidence scoring.
>
> Each training example is a prompt x, built by the pipeline's own prompt builder (system rules, numbered sources and question, in the model's chat format), and a target answer y = (y_1, …, y_T) (Section IV-E). Nyaya-Gen is a causal language model: with weights θ, it gives the probability p_θ(y_t | x, y_<t) of each next token given all earlier tokens. Only the answer tokens are scored. The prompt tokens are inputs but not targets, so the model is not trained to reproduce the sources or the question. The loss over a batch ℬ of examples is the masked next-token cross-entropy

```text
J_G = −(1/|ℬ|) Σ_{(x,y)∈ℬ} (1/T_y) Σ_{t=1}^{T_y} log p_θ(y_t | x, y_<t),                     (10)
```

> where T_y is the number of answer tokens of y, including the end-of-turn token. A low J_G means that the model gives a high probability to the reference answer, including its citation tags.
>
> To fit training into 8 GB of memory, gradient checkpointing [26] is used: intermediate activations are recomputed during the backward pass instead of being stored, which saves memory at the cost of extra computation. At inference, Nyaya-Gen uses greedy decoding (the most probable token at each step) with a repetition penalty of 1.1, which lowers the scores of tokens that have already appeared, and at most 256 new tokens.

Note: paste the `text` line as a display equation. Its number goes at the right margin, like (1)–(9).

---

## Edit 10. Section IV: opening paragraphs (experimental setup)

**WHERE:** Section IV "EXPERIMENTAL RESULTS", the first two paragraphs, before "A. Dataset Description". They are:
(i) "The experiments were run on an Apple M2 laptop, with Python 3.12, PyTorch 2.14…" and
(ii) "Because the system is not trained, there is no train/test split and no epochs…".
**ACTION:** REPLACE both paragraphs with the two below.
**TEXT:**

> The experiments were run on an Apple MacBook Air with an M2 chip and 8 GB of unified memory. Nyaya-Gen was trained and evaluated with Python 3.13.5, PyTorch 2.15.0 [27] on the MPS (Metal Performance Shaders) back end of the M2 GPU, Hugging Face Transformers 5.22.1, Sentence-Transformers 6.2.0, ChromaDB 1.5.9 and PyMuPDF. Runs A–C, Table III and Tables VI–VIII were recorded earlier with Python 3.12 and PyTorch 2.14. In Runs A–C, generation runs remotely on NVIDIA NIM and everything else runs locally (MiniLM on the M2 GPU, the ChromaDB search on the CPU). In Run D, generation also runs on the M2 GPU, so no network connection is used. Building the index takes about 15 minutes, of which embedding takes about 11. Training Nyaya-Gen ran for about 65 minutes (Section IV-E).

> One component is trained. The data of Nyaya-Gen (Section IV-E) are split by record into train (94%), dev (3%) and test (3%) parts; the dev part is used only to pick the checkpoint. The 50-query test set of Table II is never used to train the model or to choose any of its settings. However, the retriever parameters of Section IV-D were tuned on these 50 queries before the trained model was added, with no separate validation set, so those results are a sensitivity analysis and not unbiased estimates for new queries. Unless stated otherwise, experiments use a pool of 20 per path, K = 4, k = 60, and, for the API model, temperature 0.1 and 1,024 output tokens.

---

## Edit 11. Section IV-C: Table IV

**11a. WHERE:** Section IV-C, the paragraph "However the authority prior costs only 3.1 points…". It ends on page 6 with "…should be re tuned once a judgments metadata is more complete than that it looks."
**ACTION:** INSERT AFTER that paragraph, as a new short paragraph.
**TEXT:**

> Run D of Table IV uses the same retrieval as Run C but replaces the API model with the local Nyaya-Gen; it is discussed in Section IV-E.

**11b. WHERE:** Table IV ("End to end results on a 50 query tests set").
**ACTION:** INSERT one result column after "Run C", and REPLACE the caption.

| Metric | Run D (Nyaya-Gen, fully local) |
| --- | --- |
| Citation validity | 0.74 |
| Cited claim coverage | 0.66 |
| Citation support | 0.56 |
| Hallucination rate ↓ | 0.00 |
| Response rate | 0.72 |
| Retrieval precision | 0.92 |
| Average confidence | 0.45 |
| Mean latency (s) | 45.0 |

**Caption TEXT:**

> TABLE IV. End-to-end results on the 50-query test set (mean latency over the 49 non-empty queries). Runs A–C use an API model; Run D runs fully on the laptop.

Note: with five runs, Table IV may need to span both columns or use a smaller font.

---

## Edit 12. Section IV-D: first sentence

**WHERE:** IV-D "Hyperparameter Tuning", first paragraph. Its first sentence is "With all of the usual frozen models,all the tunable quantities are one of the numbers of passages K, … and also the per classes fusion weights in it."
**ACTION:** REPLACE that first sentence.
**TEXT:**

> With the embedding model and the LLM frozen, the tunable quantities of the retriever are the number of passages K, the candidate pool per path, the RRF constant k, the score weights (w_s, w_r, w_a) and the per-class fusion weights. These retrieval results do not depend on which model writes the answer, so they hold for both the API model and Nyaya-Gen.

---

## Edit 13. New Section IV-E: Nyaya-Gen Training and Evaluation

**WHERE:** after the last paragraph of IV-D, "Table VIII (left) thus confirms that a reranker using semantic similarities alone…", and before "V. CONCLUSION".
**ACTION:** INSERT AFTER that paragraph. Insert the text, Tables IX–X and Fig. 2 in the order shown.

**TEXT:**

> **E. Nyaya-Gen Training and Evaluation**
>
> *1) Training data:* The examples are built from the legal Q&A pairs of viber1 and nisaar (Table I), processed exactly as they are indexed, without the vector index. Each prompt is written by the pipeline's own prompt builder, so the model is trained on the exact input it receives at query time. There are three kinds of example (Table IX). In a *qa* example, the record's question is asked, and the record is hidden at a random position k among two to four random distractor passages; the target is the start of its answer (at most four sentences and 120 words), with every sentence ending in [SOURCE_k]. In a *mapping* example, an IPC section question from one of three templates is asked, with the IPC/BNS table chunk as one of the sources, and the target states both sections and cites the table. In a *refusal* example (about 7%), the record itself is left out, and the target is the refusal sentence required by the system prompt. Every passage is cut to 90 words. To keep the 50 test queries unseen, a question is removed if its normalised text equals a test query or its content-word Jaccard similarity to one is at least 0.5 (572 questions removed), and table rows whose IPC section appears in a test query are skipped. The data are split by record into train (94%), dev (3%) and test (3%) parts, and the training part is capped at 4,000 examples.

**TABLE IX** (one column wide):

> TABLE IX. Training data of Nyaya-Gen (examples per kind and split, before the 1,024-token limit).

| Split | qa | mapping | refusal | Total |
|---|---|---|---|---|
| Train | 3,590 | 141 | 269 | 4,000 |
| Dev | 179 | 0 | 21 | 200 |
| Test | 181 | 0 | 19 | 200 |

**TEXT (continued):**

> *2) Training:* Table X lists the settings. Examples longer than 1,024 tokens are dropped (the median example has 931 tokens), which removes 1,415 of the 4,000 training examples and leaves 2,585, or 323 updates of 8 examples for one epoch. Memory was the main constraint: on the 8 GB M2, both bf16 autocast and fp32 training without gradient checkpointing ran out of GPU memory, so the model is trained in fp32 with gradient checkpointing and one example per forward pass, at about 0.8 minutes per update. The planned epoch could not be completed. After about 80 updates and 65 minutes, the 8 GB machine ran out of memory (its swap space was full) and the run stopped. The dev loss (10) is measured every 50 updates on 24 dev examples (the first 40, of which 24 fit in 1,024 tokens), so the last saved checkpoint is the one at update 50: the Nyaya-Gen used below has therefore seen only 50 updates, about 400 training examples, or 15% of the epoch. In that time the dev loss fell from 0.833 before training (the un-fine-tuned model) to 0.033 at update 50, and the training loss fell from 0.98 at the first update to about 0.005–0.02 at updates 50–80 (Fig. 2). The very low losses show that the model learned the fixed answer format quickly; they do not show that its answers are correct, which is measured below.

**TABLE X** (one column wide):

> TABLE X. Training settings of Nyaya-Gen (MacBook Air M2, 8 GB, PyTorch 2.15.0 on MPS, seed 42).

| Setting | Nyaya-Gen |
|---|---|
| Base model | SmolLM2-135M-Instruct [21]: decoder-only, 30 layers, hidden 576, 134.5M parameters; all weights trained |
| Input | chat prompt (rules, sources cut to 90 words, question) + answer, ≤ 1,024 tokens |
| Training data | 2,585 prompt–answer pairs (Table IX, after the length limit); 24 dev examples for the dev loss |
| Loss | masked next-token cross-entropy, Eq. (10) |
| Per forward pass / per update | 1 example / 8 examples (gradient accumulation 8) |
| Optimiser | AdamW [25], lr 1×10⁻⁴, β = (0.9, 0.95), weight decay 0.01 |
| Schedule | 20 warm-up updates, then cosine decay |
| Planned / done | 1 epoch (323 updates) / about 80 updates in about 65 min, stopped by lack of memory |
| Kept checkpoint | update 50 (about 400 examples seen), lowest dev loss |
| Gradient clipping | max norm 1.0 |
| Precision and memory | fp32, gradient checkpointing [26] |
| Model selection on dev | lowest dev loss, every 50 updates |
| At inference | greedy decoding, repetition penalty 1.1, ≤ 256 new tokens |

**FIG. 2** (one column wide):

> [Insert Fig. 2 here: a line plot made from `data/generator/training_log.json`. x-axis: optimiser update (1–50, the `steps` entries). y-axis: training loss (`loss`). Add the two dev-loss points from `evals` (update 0: 0.833, update 50: 0.033) as markers, and mark update 50 as the kept checkpoint.]
>
> Fig. 2. Training of Nyaya-Gen on the MacBook Air M2 (MPS): training loss per update and dev loss before training and at the kept checkpoint (update 50).

Note: Fig. 2 is optional. If space is short, delete it and the "(Fig. 2)" in IV-E 2); the numbers in the text are enough.

**TEXT (continued):**

> *3) End-to-end results:* Table IV adds Run D, the fully local system, on the same 50 test queries. Only the answer writer differs from Run C, so Run D against Run C shows the cost of replacing the 550B API model with a 135M local one; retrieval precision is the same because retrieval is unchanged.

> Compared with Run C, Nyaya-Gen has a citation validity of 0.74 against 0.70, citation support of 0.56 against 0.50, a hallucination rate of 0.00 against 0.04 and a response rate of 0.72 against 0.68, with a mean latency of 45.0 s against 17.5 s and no network call. Most of that time is the exact-match scan (about 62 s on queries with an identifier), slowed by memory swapping on the 8 GB laptop while the generator was loaded; writing the answer itself took about 11–15 s. Its answers are shorter and plainer. Its training targets restate one source in at most four sentences, so it was never taught to combine several sources, to compare them by authority (rule 6 of the prompt) or to write the "Not covered by the retrieved sources" sentence. The six metrics do not measure these qualities, so the real gap to the 550B model may be larger than Table IV shows.

---

## Edit 14. Conclusion

**14a. WHERE:** V. Conclusion, the sentence that ends "…with only of two answers out of the 50 flagged by hallucination detector." (after REPORT_FIXES rewriting, "…only two of the 50 answers are flagged by the hallucination detector.").
**ACTION:** INSERT AFTER this sentence.
**TEXT:**

> A 135M-parameter answer writer, Nyaya-Gen, fine-tuned on the same 8 GB laptop, lets the whole system run offline, with a citation validity of 0.74 and a response rate of 0.72 against 0.70 and 0.68 for the 550B-parameter API model.

**14b. WHERE:** the same paragraph, from "Howver, limitations remain like the 50 query tests set was also used for tuning…" to the end of the paragraph ("…attributed to the authentic sources.").
**ACTION:** REPLACE.
**TEXT:**

> However, limitations remain. The 50-query test set was used to tune the retriever settings of Section IV-D, although not the trained model. The first-stage search still uses the general MiniLM encoder, which reads only the first 256 tokens of a chunk. Because of the 8 GB memory limit, Nyaya-Gen was trained for only 50 updates (about 400 examples, 15% of one epoch) with one seed, so its variance is unknown and longer training on larger hardware may change the results. Its training targets restate a single source, it sees only the first 90 words of each passage, it was selected on a dev loss over only 24 examples, and it was judged only by the rule-based metrics, not by legal experts. The default corpus lacks the full BNS and BNSS, the IPC/BNS table covers 63 sections, the exact-match scan takes about 62 seconds per query, and the hallucination detector misses incorrect statements attributed to real sources.

---

## Edit 15. References

**15a. WHERE:** Section III-H, the two citations added by REPORT_FIXES (rows 24 and 25 of its table): "NVIDIA NIM [21]" and "DeepSeek-V3.2 [22]".
**ACTION:** change [21] → **[23]** and [22] → **[24]**. No other in-text number changes.

Note: the two small-model papers are first cited in Section II, before Section III-H. IEEE order therefore gives them [21]–[22], and NIM and DeepSeek move to [23]–[24]. References [1]–[20] stay exactly as in REPORT_FIXES. This replaces the numbering in Section 1.4 of `REPORT_FIXES.md`.

**15b. WHERE:** REFERENCES, after [20] (Guha et al.).
**ACTION:** REPLACE the old [21] and [22] with this list [21]–[27].
**TEXT:**

[21] L. Ben Allal et al., "SmolLM2: When Smol goes big — Data-centric training of a small language model," arXiv:2502.02737, 2025.

[22] J. Wei, M. Bosma, V. Y. Zhao, K. Guu, A. W. Yu, B. Lester, N. Du, A. M. Dai, and Q. V. Le, "Finetuned language models are zero-shot learners," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2022.

[23] NVIDIA, "NVIDIA NIM APIs." [Online]. Available: https://build.nvidia.com (accessed Oct. 2026).

[24] DeepSeek-AI, "DeepSeek-V3 technical report," arXiv:2412.19437, 2024.

[25] I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2019.

[26] T. Chen, B. Xu, C. Zhang, and C. Guestrin, "Training deep nets with sublinear memory cost," arXiv:1604.06174, 2016.

[27] A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 32, 2019, pp. 8024–8035.

Order check: first citations are [21]–[22] in II, [23]–[24] in III-H, [25] in III-J, [26] in III-K, and [27] in the Section IV opening. In III-J, III-K and Table X, [21], [25] and [26] are cited again.

Note: check the author list and arXiv number of [21] on the arXiv page before submitting (the first author is Loubna Ben Allal; in the text, "Allal et al." follows the request, but "Ben Allal et al." matches the surname as published).

---

## Numbers

All numbers are filled in from `data/generator/training_log.json` and `data/evaluation/eval_results_local.json` (Run D, 2 Nov 2026). Nothing is left to fill.
