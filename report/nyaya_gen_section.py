# Content added to the report for Nyaya-Gen. Read by build_final_report.py,
# which defines para, html, table_block, figure_block, refs_block, piece_block.

T_IX = table_block(
    "<b>TABLE IX.</b> Training data of the Nyaya-Gen.",
    ["Kind", "Examples", "What the target is"],
    [
        ["Q&amp;A, cited", "3,590", "answer of own record, each sentence cited [SOURCE_<i>k</i>]"],
        ["IPC/BNS mapping", "141", "BNS equivalent, cited to the table chunk"],
        ["Refusal", "269", "refusal sentence (no relevant source)"],
        ["Total built", "4,000", "cap on training set"],
        ["Used (&#8804; 1,024 tokens)", "2,585", "median length 931 tokens"],
        ["Dev (checkpoint choice)", "24", "dev loss only"],
        ["Removed by leakage filter", "572", "overlap with the test queries"],
    ],
    [0.36, 0.16, 0.48],
)

T_X = table_block(
    "<b>TABLE X.</b> Training settings of the Nyaya-Gen.",
    ["Setting", "Value"],
    [
        ["Base model", "SmolLM2-135M-Instruct [15], 134.5M parameters"],
        ["Hardware", "Apple M2, 8 GB, MPS back end, fp32"],
        ["Memory saving", "gradient checkpointing [17], 90 word passages"],
        ["Optimiser", "AdamW [16], lr 1&#215;10<sup>&#8722;4</sup>, &#946; = (0.9, 0.95)"],
        ["Schedule", "20 warm up updates, then cosine decay"],
        ["Batch", "1 example &#215; 8 accumulation = 8 per update"],
        ["Max length", "1,024 tokens"],
        ["Planned / done", "323 updates (1 epoch) / about 80 in 65 min"],
        ["Kept checkpoint", "update 50 (lowest dev loss)"],
        ["Dev loss", "0.833 &#8594; 0.033"],
        ["Decoding", "greedy, repetition penalty 1.1, &#8804; 256 new tokens"],
    ],
    [0.32, 0.68],
)

T_XI = table_block(
    "<b>TABLE XI.</b> End to end results of the API writer (Run C) and the Nyaya-Gen (Run D) on a 50 query tests set.",
    ["Metric", "Run C (Nemotron, API)", "Run D (Nyaya-Gen, local)"],
    [
        ["Citation validity", "0.70", "0.74"],
        ["Cited claim coverage", "0.42", "0.66"],
        ["Citation support", "0.50", "0.56"],
        ["Hallucination rate &#8595;", "0.04", "0.00"],
        ["Response rate", "0.68", "0.72"],
        ["Retrieval precision", "0.92", "0.92"],
        ["Average confidence", "0.61", "0.45"],
        ["Mean latency (s)", "17.5", "45.0"],
        ["Writer parameters", "about 550B", "134.5M"],
        ["Network needed", "yes", "no"],
    ],
    [0.40, 0.30, 0.30],
    bold={(0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (6, 1), (7, 1)},
)

T_XII = table_block(
    "<b>TABLE XII.</b> Standard LLM metrics on the 200 held out test examples of the Nyaya-Gen data.",
    ["Metric", "SmolLM2-135M (not fine tuned)", "Nyaya-Gen"],
    [
        ["Token F1", "0.42", "0.93"],
        ["ROUGE-L F1", "0.40", "0.93"],
        ["BERTScore F1", "0.83", "0.98"],
    ],
    [0.34, 0.38, 0.28],
    bold={(0, 2), (1, 2), (2, 2)},
)

EQ = ('<p class="eq"><i>J</i><sub>G</sub>(<i>&#952;</i>) = &#8722;&#160;<sup>1</sup>&#8260;<sub>|<i>A</i>|</sub>'
      '&#160;&#8721;<sub><i>t</i>&#8712;<i>A</i></sub>&#160;log&#160;<i>p</i><sub><i>&#952;</i></sub>'
      '(<i>y<sub>t</sub></i>&#160;|&#160;<i>y</i><sub>&lt;<i>t</i></sub>,&#160;<i>x</i>)'
      '&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;&#160;(9)</p>')

BLOCKS = [
    html('<p class="h">E. Local Answer Writer: Nyaya-Gen</p>', gap_before=2, gap_after=1),
    para("All of the Runs A to C that are above depends on the API model for writing of the answer, so every query "
         "and its all retrieved passages leaves the laptop, it needs a network and an API key, and each of the request "
         "has a cost in it. The system described in Section III has no trainable parameters, thus as a last step we "
         "trained only one component, a small answer writer which is called Nyaya-Gen, that replaces the API model while "
         "the retrieval, prompt, citation validation and confidence scoring are kept exactly same. Nothing else in the "
         "pipeline is changed, so any difference between the runs usually comes from the writer only."),
    para("<i>1)</i> <i>Model and objective:</i> We used the SmolLM2-135M-Instruct [15], a decoder only model with 134.5M "
         "parameters which is about 4,000 times much more smaller than the 550B API model, and fine tuned all of its "
         "weights. Each training example is the exact same prompt <i>x</i> that the pipeline sends to the LLM (system "
         "rules, numbered sources and the question) and a target answer <i>y</i> that cites the right source as "
         "[SOURCE_<i>k</i>]. The loss is the next token cross entropy computed only over the answer tokens <i>A</i>, "
         "so model learns to write the answer and does not learn to repeat the prompt,"),
    html(EQ, gap_before=3, gap_after=3),
    para("where <i>&#952;</i> are all of the model weights and the prompt tokens are masked out of the loss.", indent=False),
    para("<i>2)</i> <i>Training data:</i> No labelled legal answers were available, so the data was built from the same "
         "public Q&amp;A datasets that are indexed (statute and Constitution question answer records). For a Q&amp;A "
         "example the own record of the question is hidden at a random position <i>k</i> among 2 to 4 distractor passages "
         "that are taken randomly from whole corpus, and the target restates the answer of that record in at most four "
         "sentences with each sentence cited as [SOURCE_<i>k</i>]. Mapping examples ask for the BNS equivalent of an IPC "
         "section and use the IPC/BNS table chunk as a source, and refusal examples leave out the own record, so the target "
         "is the refusal sentence of the system prompt. To keep the test set unseen, 572 questions which have a word "
         "Jaccard overlap of 0.5 or more with any of the 50 test queries were removed, and the mapping rows of IPC sections "
         "that are in test queries were skipped. The records are split by record into train, dev and test (94%, 3% and "
         "3%), so no record appears in two of the splits. Each passage is cut to 90 words so that examples fits in the "
         "8 GB memory, and the examples longer than 1,024 tokens are dropped. Table IX gives all of the counts."),
    T_IX,
    para("<i>3)</i> <i>Training:</i> Training is done on the same MacBook Air M2 with 8 GB memory, on the MPS GPU back end "
         "in fp32, with the gradient checkpointing [17] for fitting it in the memory; Table X lists the settings. AdamW "
         "[16] with the learning rate 1&#215;10<sup>&#8722;4</sup> is used, with 20 warm up updates and then a cosine "
         "decay, and a gradient accumulation of 8 so each of the update sees 8 examples. One epoch would be 323 updates, "
         "but each update took about 0.8 minute and after about 80 updates in 65 minutes the laptop ran out of memory "
         "(the swap was full) and training stalled, so it was stopped. The checkpoint with the lowest dev loss, at update "
         "50 (about 400 examples, 15% of one epoch), is kept as Nyaya-Gen. The dev loss falls from 0.833 before the "
         "training to 0.033 at that update and the training loss from 0.98 to about 0.01 (Fig. 2), because the format of "
         "answer is fixed and it is learned very fast. A low loss does not prove that answers are correct, thus the end to "
         "end test below is the real test of it."),
    T_X,
    figure_block(),
    para("<i>4)</i> <i>End to end results:</i> Table XI compares Run C with Run D, in which Nyaya-Gen writes the answer "
         "fully on the laptop for the same 50 test queries, with the same retrieval, prompt and the scoring. Nyaya-Gen gets "
         "a citation validity of 0.74 against 0.70, a cited claim coverage of 0.66 against 0.42, a citation support of 0.56 "
         "against 0.50 and a response rate of 0.72 against 0.68, and the hallucination detector flags none of its answers "
         "(0.00 against 0.04). Retrieval precision is the same 0.92 because retrieval is not changed. The average "
         "confidence is lower (0.45 against 0.61) since Nyaya-Gen usually cites only one source in an answer, so the "
         "source agreement part of the confidence score stays low."),
    T_XI,
    para("The mean latency is 45.0 s against 17.5 s, but most of it is the exact match scan, which takes about 62 s on "
         "the queries that have an identifier when the generator is also loaded and 8 GB memory swaps, and the writing "
         "of the answer itself takes only about 11 to 15 s. Run D needs no network, no API key and it has no cost per "
         "request."),
    para("<i>5)</i> <i>Standard metrics:</i> To compare with the usual way an LLM is evaluated in the industry, both "
         "the Nyaya-Gen and the same SmolLM2-135M before fine tuning answered the 200 held out test examples of its data "
         "(181 cited answers and 19 refusals, never seen in the training), and their answers are compared with the gold "
         "answer by three common metrics with the [SOURCE_<i>k</i>] tags removed: token F1 (the word overlap used in the "
         "SQuAD), ROUGE-L F1 (the longest common word sequence) and BERTScore F1 (meaning similarity from BERT "
         "embeddings, here with DistilBERT). Table XII shows that fine tuning raises the token F1 from 0.42 to 0.93, the "
         "ROUGE-L from 0.40 to 0.93 and the BERTScore from 0.83 to 0.98. These scores are high because the target "
         "usually restates the right source closely, so they show mostly that the model learned to find and restate the "
         "right source in the required format and not a deeper legal reasoning."),
    T_XII,
    para("These results should be read much more carefully. Answers of the Nyaya-Gen are much shorter and plainer, "
         "because its training targets restate one source in at most four sentences, so it was never taught to combine "
         "the several sources or to compare them by their authority, and the six metrics do not measure these qualities. "
         "It was trained for only 50 updates with one seed, and the same small model before the fine tuning was "
         "compared only on the held out examples of Table XII and not on the 50 end to end test queries."),
    ("gap", 6),
    piece_block(fitz.Rect(40, 257, 305, 521)),          # original "V. Conclusion"
    para("In addition, a 135M parameter answer writer, Nyaya-Gen, that was fine tuned on the same 8 GB laptop in about "
         "one hour lets the whole system run offline, and on the 50 test queries it reached a citation validity of 0.74 "
         "and a response rate of 0.72 with no flagged hallucinations, against 0.70, 0.68 and 0.04 for the 550B API model. "
         "Its training was cut short by the memory of the laptop, so the longer training, answers which combine several "
         "sources and a comparison with the model before fine tuning are left for the future work."),
    ("gap", 14),
    piece_block(fitz.Rect(40, 544, 305, 699)),          # original "References" [1]-[4]
    piece_block(fitz.Rect(310, 50, 575, 333)),          # original references [5]-[14]
    ("gap", -8),
    refs_block([
        (15, "L. B. Allal A. Lozhkov E. Bakouch G. M. Blázquez G. Penedo L. Tunstall <i>et al.</i>, “SmolLM2 When smol "
             "goes big, data centric training of a small language model,” arXiv preprint arXiv:2502.02737, 2025."),
        (16, "I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in <i>Proc. Int. Conf. Learning "
             "Representations</i>, 2019."),
        (17, "T. Chen B. Xu C. Zhang and C. Guestrin, “Training deep nets with sublinear memory cost,” arXiv preprint "
             "arXiv:1604.06174, 2016."),
        (18, "A. Paszke S. Gross F. Massa <i>et al.</i>, “PyTorch An imperative style, high performance deep learning "
             "library,” in <i>Advances in Neural Information Processing Systems</i>, 2019."),
    ]),
]
