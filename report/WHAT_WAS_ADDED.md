# What was added to the report (Nyaya-Gen)

Final report: `report/Final_Report_with_Nyaya-Gen.pdf` (9 pages; the original was 7).

**Nothing in the original report was changed or corrected.**
- Pages 1–6 are byte-for-byte the original pages (checked: their text is identical).
- On the original page 7, the end of Table VIII and its paragraph, the whole "V. Conclusion" paragraph, and references [1]–[14] are copied exactly (as clipped parts of the original page, not retyped). They only move further down because the new section comes before them.

## Everything that is new, in page order

| # | Where (final PDF) | What was added |
|---|---|---|
| 1 | Page 7, left column, after the paragraph "Table VIII (left) thus confirms…" | New subsection heading **E. Local Answer Writer: Nyaya-Gen** (the last subsection of Section IV, after "D. Hyperparameter Tuning") |
| 2 | Page 7, left column | Opening paragraph: why a local writer (Runs A–C need the API, network and key); Section III has no trainable parameters, so one component is trained; only the writer changes |
| 3 | Page 7, left column | **1) Model and objective:** SmolLM2-135M-Instruct [15], 134.5M parameters, all weights trained, loss only on answer tokens |
| 4 | Page 7, left column | **Equation (9)**: J_G(θ) = −(1/\|A\|) Σ_{t∈A} log p_θ(y_t \| y_<t, x). The report's last equation was (8). |
| 5 | Page 7, left → right column | **2) Training data:** how the examples were built (own record hidden among distractors, mapping, refusal), the leakage filter (572 removed), the 94/3/3 split, the 90-word passages and the 1,024-token limit |
| 6 | Page 7, right column | **TABLE IX.** Training data of the Nyaya-Gen (3,590 Q&A, 141 mapping, 269 refusal, 4,000 built, 2,585 used, 24 dev, 572 removed) |
| 7 | Page 7, right column | **3) Training:** M2 8 GB, MPS, fp32, gradient checkpointing [17], AdamW [16]; 323 updates planned, about 80 done in 65 min, then out of memory; update 50 kept; dev loss 0.833 → 0.033 |
| 8 | Page 7, right column | **TABLE X.** Training settings of the Nyaya-Gen |
| 9 | Page 8, left column, top | **Fig. 2.** Training loss and dev loss for each update (drawn from `data/generator/training_log.json`) |
| 10 | Page 8, left column | **4) End to end results:** Run D vs Run C: validity 0.74 vs 0.70, coverage 0.66 vs 0.42, support 0.56 vs 0.50, hallucination 0.00 vs 0.04, response 0.72 vs 0.68, confidence 0.45 vs 0.61 (and why it is lower) |
| 11 | Page 8, left column | **TABLE XI.** End-to-end results of Run C (API) and Run D (Nyaya-Gen), also listing writer size and whether a network is needed |
| 12 | Page 8, left column | Latency paragraph: 45.0 s vs 17.5 s, most of it the exact-match scan (~62 s on identifier queries, with memory swapping); writing takes 11–15 s |
| 13 | Page 8, left → right column | **5) Standard metrics:** token F1, ROUGE-L F1 and BERTScore F1 on the 200 held-out test examples, for Nyaya-Gen and the same model before training |
| 14 | Page 8, right column | **TABLE XII.** Standard LLM metrics: F1 0.42 → 0.93, ROUGE-L 0.40 → 0.93, BERTScore 0.83 → 0.98 |
| 15 | Page 8, right column | Limitations paragraph: short plain answers; 50 updates with one seed; the base model was compared only on the held-out examples |
| 16 | Page 8, right column | (original "V. Conclusion" paragraph, unchanged) |
| 17 | Page 8 right column → page 9 | New closing paragraph of the Conclusion: Nyaya-Gen runs the whole system offline, its key results, and future work |
| 18 | Page 9 | (original References [1]–[14], unchanged) |
| 19 | Page 9, after reference [14] | New references **[15]** SmolLM2 (Allal et al., 2025), **[16]** AdamW (Loshchilov and Hutter, 2019), **[17]** gradient checkpointing (Chen et al., 2016), **[18]** PyTorch (Paszke et al., 2019) |

New numbering used: Equation (9); Tables IX, X, XI, XII; Fig. 2; References [15]–[18]. All continue from the report's existing numbers.

## Things you may want to know
- The new text is written in the same style as the rest of the report.
- The original Table IV (Runs A–C) was **not** given a new column, because nothing original was to be changed. Run D is instead compared with Run C in the new Table XI.
- Section III-J still says "Nyaya has no trainable parameters" (original text, unchanged). The new subsection E answers this in its first paragraph: the retrieval system has none, and Nyaya-Gen is the one trained part added on top.
- To rebuild the PDF after editing the new text, change `report/nyaya_gen_section.py` and run:
  `python report/build_final_report.py ~/Downloads/report.pdf report/Final_Report_with_Nyaya-Gen.pdf`
  It needs PyMuPDF (`pip install pymupdf`).

## Where every number comes from
- Training: `data/generator/training_log.json`, `data/generator/data_stats.json`
- Run D results: `data/evaluation/run_D_nyaya_gen.json` (2 Nov 2026)
- Run C numbers: the original Table IV of the report
- Table XII (F1, ROUGE-L, BERTScore): `data/evaluation/generator_metrics.json`, made by `scripts/evaluation/evaluate_generator_metrics.py`
