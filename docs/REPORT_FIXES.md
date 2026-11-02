# Errors in the current report, and how to fix them

Report checked: "Citation Grounded Indian Legal Research Assistant", Team 12 (7 pages), on 2 November 2026.
It was checked against the code in this folder, `README.md`, `data/evaluation/*.json`, and the freshly built data in `data/raw/` and `data/processed/`.
Items are ordered by importance. Quotes show the text exactly as printed. Note that the PDF has lost most of its hyphens, for example "re ranking" and "all MiniLM L6 v2" (see Section 5).

## 0. Fix these first

1. **The reference list is broken** (Section 1). The text cites [1]–[21], but the list has only 14 entries, and from [10] onwards the list numbers do not match the papers cited in the text.
2. **One of the "five datasets" is not in the index at all** (Section 3.1). All 1,469 ShreyasP123 passages are rejected when the index is built. The 112,555 count is right, but the explanation of it is wrong.
3. **`nisaar/Constitution_of_India` contains question–answer pairs, not "Constitution articles"** (Section 3.2). The default index has no bare text of the Constitution or of any Act: every chunk typed "statute" or "constitution" is a Q&A pair.
4. **Equation numbers skip (4), (6) and (7), and the MRR equation has no number** (Section 2).
5. **Section IV-D says the adaptive weights "match the best fixed setting", but Table VIII shows 0.811 against 0.827** (Section 3.4).
6. **A number is missing in the Conclusion**: "the exact same match scan takes the  seconds per query" (Section 3.8).
7. **Check which LLM Run C used.** Its result file is named `eval_results_deepseek.json`, but the report says Nemotron (Section 4.1).

---

## 1. References

### 1.1 What is wrong

- The text cites [1]–[21], but the list stops at [14].
- List entries [1]–[9] match the text. From [10] onwards they do not. The list's [10] is BM25, which the text never cites. The list's [11] is RAGAS, but the text's [11] is Izacard and Grave. The list's [12] is LegalBench, but the text's [12] is Liu et al. The list's [13] is DeepSeek, but the text's [13] is Malik et al. The list's [14] is the BNS, but the text's [14] is Kalamkar et al.
- Seven cited papers are missing from the list: Izacard & Grave, Liu et al., Malik et al., Kalamkar et al., FIRE AILA, Gao et al. (ALCE) and Ji et al.
- "Es et al." (RAGAS) has no number in the text. Dahl et al. and Magesh et al. are named in Section I without a number.
- The DeepSeek report is in the list but is never cited in the text.
- IEEE numbers references in order of first citation. The text starts with [21] (Section I), so the BNS must become [1].
- Several titles are wrong. "for legal field" should be "for the legal domain". "open field question answering" should be "open-domain question answering". "LEGAL BERT straight out of law school" is missing "The muppets". Several entries also have no year, and "arXiv preprint arX." has no number.

### 1.2 Renumber the citations in the text

Change every citation in one pass. A safe way is to first write each new number in braces, for example {4}, and then replace all braces with square brackets at the end. This stops a changed number from being changed again.

| # | Section | Sentence in the report (first words) | Now | Change to |
|---|---|---|---|---|
| 1 | I | "...the Code of Criminal Procedure and the Evidence Act from 1 July 2024 [21]." | [21] | [1] |
| 2 | I | "Dahl et al.measured that hallucination rates between 69% and 88%..." | (none) | add [2] after "Dahl et al." |
| 3 | I | "and Magesh et al. founded that a commercial legal research AI..." | (none) | add [3] after "Magesh et al." |
| 4 | II | "Retrieval augmented generation. Lewis etal.[1] combined..." | [1] | [4] |
| 5 | II | "Izacard and Grave [11] showed us that with Fusion in Decoder..." | [11] | [5] |
| 6 | II | "while Liu et al. [12] already showed that LLMs use information..." | [12] | [6] |
| 7 | II | "Dense and lexical retrieval. Karpukhin et al. [7] trained the dual BERT encoder..." | [7] | [7] (no change) |
| 8 | II | "...and beat the BM25 performance by a large margin..." | (none) | add [8] after "BM25" |
| 9 | II | "Reimers and Gurevych [8] proposed Sentence BERT..." | [8] | [9] |
| 10 | II | "...and Wang et al. [9] distilled such models into very compact MiniLM..." | [9] | [10] |
| 11 | II | "Cormack et al. [2] employed Reciprocal Rank Fusion..." | [2] | [11] |
| 12 | II | "Chalkidis et al. [6] pre-trained LEGAL BERT on English legislation..." | [6] | [12] |
| 13 | II | "Paul et al. [5] pre-trained InLegalBERT and InCaseLawBERT..." | [5] | [13] |
| 14 | II | "Malik et al. [13] created the ILDC corpus..." | [13] | [14] |
| 15 | II | "while Kalamkar et al. [14] annotated roles in Indian judgments." | [14] | [15] |
| 16 | II | "The FIRE AILA track [15] benchmarked the retrieval..." | [15] | [16] |
| 17 | II | "Legal hallucinations and attribution. According to Dahl et al. [4]..." | [4] | [2] |
| 18 | II | "Also, according to Magesh et al. [3], they tested it..." | [3] | [3] (no change) |
| 19 | II | "Gao et al [16] introduced the ALCE benchmark..." | [16] | [17] |
| 20 | II | "Then Ji et al. [17] surveyed the hallucination..." | [17] | [18] |
| 21 | II | "RAG system evaluation. Es et al introduces and suggest using RAGAS..." | (none) | add [19] after "Es et al." |
| 22 | II | "...and Guha et al. [19] actually built LegalBench..." | [19] | [20] |
| 23 | III-F | "The two ranked lists are fused with weighted RRF [2]." | [2] | [11] |
| 24 | III-H | "Generation also uses an OpenAI compatible client against NVIDIA NIM with temperature 0.1..." | (none) | add [21] after "NVIDIA NIM" |
| 25 | III-H | "Any OpenAI compatible model can be configured DeepSeek and NVIDIA Nemotron were used..." | (none) | write "DeepSeek-V3.2 [22]" |
| 26 | IV-C | "...dense only retrieval as in a standard RAG [1], [7]..." | [1], [7] | [4], [7] |
| 27 | IV-C | "...hybrid retrieval fused with a standard fixed type of weight RRF [2]." | [2] | [11] |
| 28 | IV-C | "Ussually Commercial legal tools cannot be compared using numbers, since Magesh et al. [3]..." | [3] | [3] (no change) |
| 29 | IV-D | "...where LLMs use them less consistenly [12]." | [12] | [6] |

### 1.3 Corrected reference list (IEEE, in order of first citation)

Replace the whole REFERENCES section with the list below. Where a detail such as the page numbers was not certain, it has been left out rather than guessed.

[1] *The Bharatiya Nyaya Sanhita, 2023* (Act No. 45 of 2023). New Delhi, India: Ministry of Law and Justice, Gazette of India, Dec. 2023.

[2] M. Dahl, V. Magesh, M. Suzgun, and D. E. Ho, "Large legal fictions: Profiling legal hallucinations in large language models," *J. Legal Anal.*, vol. 16, no. 1, 2024.

[3] V. Magesh, F. Surani, M. Dahl, M. Suzgun, C. D. Manning, and D. E. Ho, "Hallucination-free? Assessing the reliability of leading AI legal research tools," *J. Empirical Legal Stud.*, 2025.

[4] P. Lewis et al., "Retrieval-augmented generation for knowledge-intensive NLP tasks," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 33, 2020, pp. 9459–9474.

[5] G. Izacard and E. Grave, "Leveraging passage retrieval with generative models for open domain question answering," in *Proc. 16th Conf. Eur. Chapter Assoc. Comput. Linguistics (EACL)*, 2021.

[6] N. F. Liu et al., "Lost in the middle: How language models use long contexts," *Trans. Assoc. Comput. Linguistics*, vol. 12, pp. 157–173, 2024.

[7] V. Karpukhin et al., "Dense passage retrieval for open-domain question answering," in *Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP)*, 2020, pp. 6769–6781.

[8] S. Robertson and H. Zaragoza, "The probabilistic relevance framework: BM25 and beyond," *Found. Trends Inf. Retr.*, vol. 3, no. 4, pp. 333–389, 2009.

[9] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using Siamese BERT-networks," in *Proc. Conf. Empirical Methods Natural Lang. Process. and 9th Int. Joint Conf. Natural Lang. Process. (EMNLP-IJCNLP)*, 2019, pp. 3982–3992.

[10] W. Wang, F. Wei, L. Dong, H. Bao, N. Yang, and M. Zhou, "MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 33, 2020.

[11] G. V. Cormack, C. L. A. Clarke, and S. Büttcher, "Reciprocal rank fusion outperforms Condorcet and individual rank learning methods," in *Proc. 32nd Int. ACM SIGIR Conf. Res. Develop. Inf. Retr.*, 2009, pp. 758–759.

[12] I. Chalkidis, M. Fergadiotis, P. Malakasiotis, N. Aletras, and I. Androutsopoulos, "LEGAL-BERT: The muppets straight out of law school," in *Findings Assoc. Comput. Linguistics: EMNLP 2020*, 2020.

[13] S. Paul, A. Mandal, P. Goyal, and S. Ghosh, "Pre-trained language models for the legal domain: A case study on Indian law," in *Proc. 19th Int. Conf. Artif. Intell. Law (ICAIL)*, 2023.

[14] V. Malik et al., "ILDC for CJPE: Indian legal documents corpus for court judgment prediction and explanation," in *Proc. 59th Annu. Meeting Assoc. Comput. Linguistics and 11th Int. Joint Conf. Natural Lang. Process. (ACL-IJCNLP)*, 2021.

[15] P. Kalamkar et al., "Corpus for automatic structuring of legal documents," in *Proc. 13th Lang. Resour. Eval. Conf. (LREC)*, 2022.

[16] P. Bhattacharya et al., "Overview of the FIRE 2019 AILA track: Artificial intelligence for legal assistance," in *Working Notes of FIRE 2019 – Forum for Information Retrieval Evaluation*, CEUR Workshop Proceedings, 2019.

[17] T. Gao, H. Yen, J. Yu, and D. Chen, "Enabling large language models to generate text with citations," in *Proc. Conf. Empirical Methods Natural Lang. Process. (EMNLP)*, 2023.

[18] Z. Ji et al., "Survey of hallucination in natural language generation," *ACM Comput. Surv.*, vol. 55, no. 12, Art. no. 248, 2023.

[19] S. Es, J. James, L. Espinosa-Anke, and S. Schockaert, "RAGAS: Automated evaluation of retrieval augmented generation," in *Proc. 18th Conf. Eur. Chapter Assoc. Comput. Linguistics: Syst. Demonstrations*, 2024.

[20] N. Guha et al., "LegalBench: A collaboratively built benchmark for measuring legal reasoning in large language models," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), Datasets and Benchmarks Track*, 2023.

[21] NVIDIA, "NVIDIA NIM APIs." [Online]. Available: https://build.nvidia.com (accessed Oct. 2026).

[22] DeepSeek-AI, "DeepSeek-V3 technical report," arXiv:2412.19437, 2024.

Notes:
- [22] is the technical report of the DeepSeek-V3 family. Run A used DeepSeek-V3.2, so write "DeepSeek-V3.2, a model of the DeepSeek-V3 family [22]".
- The BNS reference [1] also supports the sentence about the BNSS and the Bharatiya Sakshya Adhiniyam. To be complete, you may add those two Acts (Act No. 46 and Act No. 47 of 2023) as extra references.

### 1.4 If you also add the Nyaya-Gen pages

`REPORT_ADDITIONS.md` adds a "Small language models" paragraph to Section II. Its two papers are cited before Section III-H, so in IEEE order they become [21]–[22]. NIM and DeepSeek then move from [21] and [22] to [23] and [24]. This changes only the two citations added in rows 24–25 of the table above. New references [25]–[27] follow. Edit 15 of `REPORT_ADDITIONS.md` gives these entries.

---

## 2. Equation numbering

At present the equations are numbered (1), (2), (3), then left unnumbered, then (5), then two more are unnumbered, then (8), and the MRR equation is unnumbered. A teacher will see the gaps.

| Section | Equation | Now | Fix |
|---|---|---|---|
| III-C | T̂(c) = ⌊1.3 \|W(c)\|⌋ | (1) | keep |
| III-D | s(q,c) = 1 − d_cos(q,c) = ... | (2) | keep |
| III-F | RRF_L(c) = 1/(k + r_L(c)) | (3), but the label sits under the equation | keep (3); put the label on the same line, right-aligned |
| III-F | R(c) = α_τ RRF_sem(c) + β_τ RRF_kw(c) | none | number it **(4)** |
| III-G | A(c) = 1 if c ∈ ℒ, else 0.4 w_court + 0.3 w_rec + 0.3 w_cit | (5), but "is" is glued to the equation and the label sits below it | keep (5). End the sentence with "The authority score of chunk c is" and put the equation on its own line |
| III-G | R̃(c) = min(R(c)(k+1), 1) | inline | can stay inline |
| III-G | S(c) = w_s s(q,c) + w_r R̃(c) + w_a A(c) | none | number it **(6)** |
| III-I | C = 0.40 Q + 0.25 G + 0.20 H + 0.15 V | none | number it **(7)** |
| IV-B | P@K = ... | (8) | keep |
| IV-B | MRR = ... | none, and words are written inside the equation | number it **(9)**. Write it as MRR = (1/\|𝒬\|) Σ_{q∈𝒬} 1/rank_q, and explain in the text that rank_q is the rank of the first keyword-relevant chunk, and that the term is 0 when there is none in the top K |

Symbol clashes to fix at the same time:
- **Q** means retrieval quality in (7), but the query set in the MRR equation. Use 𝒬 for the query set.
- The text after (5) says "where L is the set of statute and Constitution chunks", but the equation uses ℒ. Use ℒ in both places. Note that L is also the ranked list in (3).

---

## 3. Numbers that disagree with a table, another part of the report, or the data

### 3.1 ShreyasP123 is not in the index, and the 112,555 explanation is wrong (serious)

Section IV-A says "101,983 chunked and 12,041 pre split passages giving 112,555 unique chunks after duplication identifiers are merged". Now 101,983 + 12,041 = 114,024, and 114,024 − 112,555 = 1,469. This is exactly the size of ShreyasP123 in Table I. It is not a coincidence:

- `prepare_hf_data.py` copies the `chunk_id` field of `ShreyasP123/Legal-Dataset-for-india` unchanged, and in that dataset it is an integer (1, 2, 3, ...).
- The chunk schema (`src/chunking/chunk_models.py`) requires a string. So `build_vectorstore.py` rejects all 1,469 of these chunks with "Skipping invalid chunk". I checked this with the project's own `Chunk` model: 10,572 pre-split chunks pass and 1,469 fail, and every failure is a ShreyasP123 chunk.
- Also, 10,389 of the 20,961 `vihaannnn` records are shorter than 50 characters and are dropped. So 10,572 pre-split passages are indexed, not 12,041.
- The correct sum is therefore 101,983 + 10,572 = 112,555, with no merging of duplicate identifiers.

So the index holds chunks from **four** of the five datasets. This affects the Abstract ("from the 5 of the public Hugging Face dataset"), contribution 1 ("from 5 public Indian law datasets"), Table I, Section IV-A and the Conclusion ("of the 5 available public datasets").

Fix (text only; all results stay valid):
- **IV-A, replace** "Table I lists all the five dataset. ... after duplication identifiers are merged." with:
  "Table I lists the five datasets. Records that arrive pre-split are indexed as they are, and all other records go through the structure-aware chunkers. Chunking about 31,000 documents gives 101,983 chunks. Of the 22,430 pre-split records, 10,389 shorter than 50 characters are dropped, and the 1,469 ShreyasP123 passages are rejected at indexing because their identifiers are integers rather than strings, so 10,572 pre-split passages are indexed. The index therefore holds 112,555 chunks from four of the five datasets."
- **Table I note, replace** with: "Indexed chunks: 112,555 (judgment 85,979, constitution 22,576, statute 4,000). The 1,469 ShreyasP123 passages are not indexed (Section IV-A)."
- **Abstract, contribution 1 and Conclusion:** write "from four of the five public Hugging Face datasets" or "from public Hugging Face datasets" instead of "from the 5 ... datasets".
- (Alternative) Fix the code (`str(...)` around the id in `prepare_hf_data.py`) and rebuild. The index then has 114,024 chunks, and **every** retrieval and end-to-end number must be measured again. This is not advised before submission.

### 3.2 `nisaar/Constitution_of_India` contains Q&A pairs, not Constitution articles (serious)

- Table I describes it as "Constitution articles", and III-B says "...a small legal RAG collection and the articles of the Constitution". But the data has the fields `question` and `answer`, for example "What is the definition of 'Panchayat' in the Constitution of India?". The code also calls it "QA pairs about Constitution".
- In the default index, **every** chunk typed "constitution" (22,576) or "statute" (4,000) is a question–answer pair. Of the "constitution" chunks, 20,711 come from `viber1` and 1,865 from `nisaar`. All 4,000 "statute" chunks come from `viber1`. `prepare_hf_data.py` assigns these types by keywords: "article", "constitution" or "fundamental" give "constitution", and "section", "ipc" and similar words give "statute". So the default index contains no bare text of the Constitution or of any Act. The article-wise and section-wise chunkers of III-C only see such text when the optional `download_statutes.py` is run.
- Because A(c) = 1 for every chunk labelled statute or constitution, these Q&A passages get the same authority as binding text. Section IV-C ("It defines rules and Constitution passages, which genrally scores 1.0...") should say this.
- Fix: in Table I write "933 | Q&A pairs on the Constitution". In III-B write "question–answer pairs on the Constitution". In IV-A, replace "and default build does not contain just text of every IPC and BNS section" with: "The default build contains no bare text of the Constitution or of any Act. The passages typed 'constitution' or 'statute' are question–answer pairs that the conversion script labels by keywords, and they still receive authority 1.0." Also note that `viber1` is general legal Q&A (for example, "the difference between a petition and a plaint"), not only "Statute Q&A pairs".

### 3.3 The percentages in IV-A use the wrong total

"judgments ... 76.4% of the total chunks, Constitution passages the 19.9% and other passages only just 3.7%". These were computed on 114,024 chunks (87,099 / 22,654 / 4,271), which include the 1,469 rejected chunks. For the 112,555 indexed chunks, the figures are judgment 85,979 (**76.4%**), constitution 22,576 (**20.1%**) and statute 4,000 (**3.6%**). Change 19.9 to 20.1 and 3.7 to 3.6, and update the counts in the Table I note.

### 3.4 Table VIII text: the adaptive weights do not match the best fixed setting on P_kw

The text says "it presets matches the best of the fixed setting". Table VIII (right) shows adaptive P_kw = **0.811**, but fixed (0.0, 1.0) = **0.827** and fixed (0.2, 0.8) = **0.821**. Only Hit is equal (0.878). Write instead: "The adaptive preset reaches the same Hit as the best fixed setting (0.878), with a P_kw 1.6 points lower (0.811 against 0.827), while it keeps a high semantic weight for conceptual queries, which have no exact matches."

### 3.5 Table VIII "fixed (0.5, 0.5)" disagrees with Table III row (c)

Table III row (c), "Hybrid, fixed RRF (0.5/0.5)", gives P_kw@4 = 0.776 and Hit@4 = 0.878. Table VIII (right), "fixed (0.5, 0.5)", gives 0.719 and 0.837. The labels look like the same setting. The numbers differ because the Table VIII right half also applies the deployed score weights (0.5, 0.25, 0.25), including authority, while Table III (c) ranks by RRF only. Say this in the caption of Table VIII: "Right: fusion weights varied, with the final score (6) and the deployed weights (0.5, 0.25, 0.25)."

### 3.6 The rows that "isolate" each part are named wrongly

IV-C says "Rows (e) and (f) isolate all the contribution of queries adaptive weights and a authority prior". In fact, row (d) against row (c) isolates the adaptive weights, and row (f) against row (e) isolates the authority prior. Also, rows (d) and (e) are identical, and the text does not say which weights row (e) uses. Table VIII (left) suggests (w_s, w_r, w_a) = (0.5, 0.5, 0), which also gives 0.842 / 0.878. Check this in the experiment script and state it.

### 3.7 The Abstract makes a comparison that Section IV-C says cannot be made

The Abstract says the hallucination rate of 0.04 "is also good compared to others". IV-C says "Commercial legal tools cannot be compared using numbers". Remove "compared to others" from the Abstract (see rewrite 4 in Section 6).

### 3.8 Conclusion: a number is missing

"...the exact same match scan takes the  seconds per query..." has no number. Insert the measured time, or write "takes several seconds per query".

### 3.9 Table IV latency is averaged over 49 queries, but the caption says 50

From the result files, mean latency over all 50 queries is 24.53 s (Run A), 15.06 s (Run B) and 17.20 s (Run C). Over the 49 non-empty queries it is 25.03 s, 15.37 s and 17.55 s (17.549), which round to the values in Table IV. Add to the caption "(latency: mean over the 49 non-empty queries)".

### 3.10 The reasons given for K = 4 and pool = 20 do not follow from Tables VI and VII

- "K = 4 was selected as the smallest value ... that allows multi sources answers while keeping all the keywords precisions above 0.8". But K = 2 also allows a two-source answer and has a higher P_kw (0.867). The real reason given earlier is the source-agreement signal G, which needs 3 cited sources for full credit. Write: "the smallest tested value that allows three cited sources, which G needs for full credit, while P_kw stays above 0.8."
- Table VII shows pool 5 with P_kw 0.815 and MRR 0.878, slightly better than pool 20 (0.811, 0.878). The text says pools above 20 lower MRR, but it gives no reason for choosing 20 over 5. Add the reason, for example "20 leaves more candidates for the later stages at no cost in MRR".

### 3.11 Test count

Section III-A says 250 tests. `pytest --collect-only` collects **251** today. Write 251, or "about 250".

### 3.12 Numbers I checked and found correct (no change needed)

- 15.5 points (0.811 − 0.656).
- 6.6 and 4.1 points (rows d and c).
- 3.1 points (row f against row e, and in Table VIII).
- Hit@4 0.776 → 0.796.
- "more than 2 times the response rate (0.30 to 0.68)".
- "35 of its 50 answers".
- Every value in Tables IV and V (checked against `eval_results.json`, `eval_results_nemotron.json` and `eval_results_deepseek.json`).
- 50 queries = 15 + 10 + 10 + 10 + 5.
- 49 non-empty queries.
- The record counts in Table I.
- 63 IPC/BNS rows.
- About 31,000 documents chunked.
- About 3,600 lines of Python (3,557 in `src/` and `config/`).
- Dahl et al. 69–88%, Magesh et al. 17–33%.
- IPC 302 → BNS 103, IPC 376 → BNS 64.

---

## 4. Technical statements that disagree with the code or the README

### 4.1 Which LLM produced Run C?

Table IV and the README say Runs B and C used `nvidia/nemotron-3-ultra-550b-a55b`. Run B's file is `eval_results_nemotron.json`. But the file whose numbers match Run C and Table V exactly is **`eval_results_deepseek.json`**: validity 0.70, response rate 0.68, hallucination rate 0.04, retrieval precision 0.915, confidence 0.607. The README calls this file `eval_results_latest.json`, and the default model in `config/settings.py` is DeepSeek-V3.2. The files do not record the model name. Confirm which model was used. If it was DeepSeek, correct Table IV, the Abstract ("with a Nemotron model") and Section IV-C. If it was Nemotron, rename the file.

### 4.2 Embedding does not run "on the CPU"

Section IV says "Embedding and retrieval runned locally on the CPU". `EmbeddingService` sets no device, so Sentence-Transformers picks the Apple GPU. In the project's environment, `sentence_transformers.util.get_device_name()` returns `mps`. Write: "Embedding runs locally on the M2 GPU through PyTorch's MPS back end, retrieval runs on the CPU, and generation runs remotely on NVIDIA NIM."

### 4.3 Indian Kanoon judgments are HTML pages, not scanned PDFs

III-B says "landmark judgments from Indian Kanoon scanned PDFs ... parsed with PyMuPDF and an OCR fallback". `download_indiankanoon.py` saves **HTML** pages. PyMuPDF with the Tesseract OCR fallback is used for PDFs, such as the India Code and Gazette files from `download_statutes.py` and any PDF placed in `data/raw/`. Correct the sentence.

### 4.4 Missing fact: MiniLM reads only the first 256 word pieces

`all-MiniLM-L6-v2` has `max_seq_length = 256`. Chunks aim for 450 estimated tokens, and can have up to 800. So for most chunks only the beginning affects s(q,c). Add this sentence to III-D: "The encoder reads at most 256 word pieces, so for longer chunks only the beginning is embedded."

### 4.5 The hallucination rate ignores 16 well-known cases

`detect_hallucinations` does not flag 16 well-known landmark cases, such as Kesavananda Bharati, Maneka Gandhi and Vishaka, when the model names them without a source. IV-B does not mention this. Add: "Sixteen well-known landmark cases are not counted when they are named without a source, so the rate is a lower bound."

### 4.6 IPC/BNS examples

III-E says "for example IP 376 to BNS 64 and IPC 420 to BNS 318". The mapping file has IPC 420 → **BNS 318(4)**, which is the sub-section example the sentence promises. Also, "IP 376" should be "IPC 376".

### 4.7 Chunk identifier format

III-C prints "(doc_idindex)". The code builds `f"{doc_id}::{index}"`, so write `doc_id::index`.

### 4.8 Statements about the literature that are not accurate

- **Paul et al. [13]:** "semantic segmentation of Indian legislation" should be "semantic segmentation (rhetorical-role labelling) of Indian court judgments". The three tasks were legal statute identification, semantic segmentation of judgments and court judgment prediction.
- **Lewis et al. [4]:** "trained the two of them together fully" should be "fine-tuned the retriever's query encoder and the generator together (the document index stays fixed)".
- **Karpukhin et al. [7]:** "they also reported that all of the writing techniques have become increasingly good at dealing with different kinds of queries, rather than on rare exact tokens" does not make sense. Write: "they also noted that dense retrievers match meaning well but are weaker than BM25 on highly selective keywords, which is exactly the case for section and article numbers."
- **Gao et al. [17]:** ALCE is a benchmark "for LLMs that answer with citations", not "for agents".
- **Magesh et al. [3]:** "the retrieved document was almost always wrong" overstates the paper. Write "often because the retrieved documents were not relevant".
- **Wang et al. [10]:** MiniLM was distilled from large pre-trained transformers, not from Sentence-BERT models. Write: "Wang et al. [10] proposed MiniLM, a compact transformer distilled from a larger one; the all-MiniLM-L6-v2 sentence encoder used here is built on it."

---

## 5. Presentation problems a teacher will see

- **Hyphens and symbols are missing everywhere**, probably lost when the text was edited. Restore them, especially in names, which must be exact:
  - all-MiniLM-L6-v2, Sentence-BERT, LEGAL-BERT, DeepSeek-V3.2, nemotron-3-ultra-550b-a55b, doc_id::index
  - the Table I dataset ids: viber1/indian-law-dataset, ninadn/indian-legal, vihaannnn/Indian-Supreme-Court-Judgements-Chunked, ShreyasP123/Legal-Dataset-for-india, nisaar/Constitution_of_India
  - compound words: re-ranking, pre-trained, structure-aware, query-adaptive, exact-match, end-to-end, cross-reference, 50-query, learning-to-rank, question–answer
- Use one name for each query category everywhere. Table II has "Sectional lookup" but Table V has "Section lookup". The text mixes "conceptual" with "theoretical", and "cross reference" with "referenced queries".
- Fix the captions:
  - Table III: "(49 queries,)" → "(49 queries)".
  - Table IV: "End to end results on a 50 query tests set" → "End-to-end results on the 50-query test set".
  - Table V: "Per categories results" → "Per-category results".
  - Table VI: "number of all the passages" → "number of passages".
- Table II: write the two edge-case examples with a separator, as "Section 99999 of IPC; empty query". The other three edge cases (a very long padded query, a Hinglish query and a special-character query) are not mentioned anywhere. Add one sentence about them.
- Write "Fig. 1", not "Fig 1" (III-J). Write "et al." with a full stop and a space before the bracket ("Lewis et al. [4]", "Gao et al. [17]", "Es et al. [19]").
- In Table IV, Run A's empty cells should show "–", with a note: "not recorded in Run A".
- IEEE style: the abstract starts with "*Abstract*—", and an "*Index Terms*—" line follows it (for example: retrieval-augmented generation, legal information retrieval, Indian law, citation verification, hybrid search). An e-mail line under the authors is also usual.

---

## 6. English

The English needs a careful sentence-by-sentence proofread before submission. Many sentences have words that add no meaning ("thus", "now", "just", "all of the", "much more", "usually"), wrong verb forms ("runned", "founded" for "found", "do not changes") and spelling mistakes. Below are the 15 worst sentences, rewritten without changing their meaning, except where the original was factually wrong (marked).

1. **Abstract.** "The large language models are currently primarily used for the legal researches, although mostly they make up a new case name, the citations mentioned in them and section numbers which are not existing."
   → "Large language models are now widely used for legal research, but they often invent case names, citations and section numbers that do not exist."

2. **Abstract.** "Firstly, the documents are divided into legal (the sections, the articles, numbered paragraphs) and the rest, with embedding of MiniLM sentence encoder."
   → "Documents are first split along their legal structure (sections, articles and numbered paragraphs) and embedded with a MiniLM sentence encoder."

3. **Abstract.** "Upon asking a question, the rule-based classifier assigns weights to the deep searches path and a exact match over all other legal identifiers, after which two of the ranked lists are fused with Reciprocal Rank Fusion, and the authority is assigned before to the re-ranking of all the candidates."
   → "For each question, a rule-based classifier sets the weights of a dense search path and an exact-match path over legal identifiers; the two ranked lists are fused with Reciprocal Rank Fusion, and the candidates are re-ranked with an authority score."

4. **Abstract.** "Thus every end, with a Nemotron model served by a NVIDIA NIM, our system reach a citation validity of 0.70 for now, and thus the response rate of just 0.68 which is decent one and the pattern detected hallucination rate of 0.04 only now now which is also good compared to others."
   → "End to end, with a Nemotron model served through NVIDIA NIM, the system reaches a citation validity of 0.70, a response rate of 0.68 and a pattern-detected hallucination rate of 0.04."
   (The phrase "compared to others" is dropped; see Section 3.7.)

5. **Section I.** "Legal research is mainly locating a , constitutional article or one that answers a question and reading it much more accurately. In majority India of this is the majority work."
   → "Legal research is mainly the work of finding the statute section, constitutional article or judgment that answers a question, and then reading it carefully. In India, this is a large part of legal work."

6. **Section I.** "LLMs ussually appear to solve this problem because they answers like they all know answer, but they are'nt mcuh trustable sources of law."
   → "LLMs seem to solve this problem because they answer fluently and confidently, but they are not reliable sources of law."

7. **Section I.** "Whereas RAG addresses part of this problem by running a model on multiple paragraohs that it got from a mcuh more trusted dataset. Though much more standard of RAG pipelines, however, rely on a singler much more dense retriever."
   → "Retrieval-augmented generation (RAG) addresses part of this problem: the model writes its answer from passages retrieved from a trusted collection. Most standard RAG pipelines, however, rely on a single dense retriever."

8. **Section I.** "Though dense retrievers capture meaning well but handle exact once poorly a query about "Section 498A" or "Article 21" must match all identifiers fully, and the embeddings model trained on a general text just does not take away Article 21 from Article 210 or Section 302 to Section 304."
   → "Dense retrievers capture meaning well but handle exact identifiers poorly: a query about "Section 498A" or "Article 21" must match the identifier exactly, and an embedding model trained on general text does not reliably separate Article 21 from Article 210, or Section 302 from Section 304."

9. **Section I.** "Just these Existing systems also rarely check the other citations in generated answers pointed to all the passages that were given to it, and they do not handle the IPC to BNS transition that is mainly used in Indian criminal law questions today also produces corre ct answers."
   → "Existing systems also rarely check whether the citations in a generated answer point to the passages the model was given, and they do not handle the IPC-to-BNS transition, which affects most Indian criminal-law questions today."

10. **Section II.** "According to Dahl et al. [4], the legal hallucinations of GPT type model like the gpt or the llama model is based on the federal court cases, and found that models many times actually accepted the bad things and make lie holdings."
    → "Dahl et al. [2] profiled the legal hallucinations of models such as GPT-3.5 and Llama 2 on questions about US federal court cases, and found that the models often accept false premises in the question and invent holdings."

11. **Section II.** "Also, according to Magesh et al. [3], they tested it on commercial RAG legal tools on more than 200 questions and found that the retrieval actually reduced but does not remove hallucinations fully, many times because the retrieved document was almost always wrong."
    → "Magesh et al. [3] tested commercial RAG-based legal research tools on more than 200 questions and found that retrieval reduces hallucinations but does not remove them, often because the retrieved documents are not relevant to the question." (corrected: "almost always wrong" overstated the paper)

12. **Section III-C.** "Judgments are then split at the numbered paragraphs and short paragraphs are merged this is done so that ther meanings doesnt change when chunked, segments are then packed to a target length, the case name is then linked to every chunk so that every passage is known by self only."
    → "Judgments are split at their numbered paragraphs, and very short paragraphs are merged so that their meaning is not lost; the segments are then packed to a target length, and the case name is added to every chunk so that each passage can be understood on its own."

13. **Section IV-C.** "However the authority prior costs only 3.1 points of the total number of keyword precisions (row f against row e) and do not changes the Hit@4 or MRR. It defines rules and Constitution passages, which genrally scores 1.0, above judgments whose metadata are usually missing and cannot be found and which thus receives defaulted authority values for the case links and theoretical questions this sometimes replaces the relevant judgments paragraphs with the lesser particular constitutional passages."
    → "The authority prior costs 3.1 points of keyword precision (row (f) against row (e)) and does not change Hit@4 or MRR. It ranks statute and Constitution passages, which score 1.0, above judgments, whose metadata are often missing and which therefore receive default authority values; for case lookups and conceptual questions, this sometimes replaces a relevant judgment paragraph with a less specific constitutional passage."

14. **Section IV-C.** "In this case lookups are weakest category retrieval precision and accuracy is a perfect under the lighter metrics, but corpus often usually holds only tye fragments of a named judgment without its case name, and the model then declines it to an attribute."
    → "Case lookups are the weakest category: retrieval precision is perfect under the lighter metric, but the corpus often holds only fragments of a named judgment without its case name, and the model then declines to attribute them to the case."

15. **Section V.** "Though structure aware chunking of the 5 available public datasets produced the 112,555 chunked indexes i.e query adaptives hybrid retrieval with weighted RRF raises keywords grounded precision at upto 4 passages from the 0.656 for the much more denser retrieval to 0.811 and Hit@4 from 0.796 to 0.878, and an IPC/BNS table done as a citable source lets our system answer questions."
    → "Structure-aware chunking of public datasets produced a 112,555-chunk index; query-adaptive hybrid retrieval with weighted RRF raises keyword-grounded precision at 4 passages from 0.656 for dense retrieval to 0.811 and Hit@4 from 0.796 to 0.878; and an IPC/BNS table, given to the model as a citable source, lets the system answer questions about both codes." ("5 available" is dropped; see Section 3.1.)

Other passages that need the same treatment:
- the whole of III-B
- the last sentence of III-E
- item (iii) of the Research gap
- the first two sentences of IV-A
- "because of the type thing alone is easy to satisfy" (IV-B)
- "More precised matches retrieval has the opposite profile..." (IV-C)
- "Run A's high validity gives mostly the refusals..." (IV-C)
- the first sentence of IV-D

Spelling mistakes to search for: mcuh, paragraohs, halucinated, founded (→ found), anfd, togeter, duplicartes, whic, represets, eachtime, confiidence, runned, jugments, lsited, genrally, theoretcial, consistenly, pragraphes, beacuse, becauseof, missesall, Howver, was'nt, are'nt, ussually / Ussually / usally, tye, unsual, releated, awared, singler, Sanhita(BNS) (missing space), "corre ct", "ge set" (→ retrieved set).
