# PA-LawSearch Evaluation Results

Fine-tuning a domain-adapted dual-encoder on citation-derived PA case pairs yields a statistically significant lift across all retrieval methods, with the trained LegalBERT encoder improving nDCG@10 by +0.051 over its frozen baseline on hybrid RRF (paired bootstrap, p < 0.0001, N = 12,035), and the improvement generalizes cross-jurisdiction on an independent public benchmark.

---

## 1. Headline retrieval results

All figures use hybrid RRF (BM25 + dense, k = 60) on citation-grounded qrels, N = 12,035 evaluation queries. Significance is a one-tailed paired bootstrap over per-query nDCG@10 (10,000 resamples, seed 42). BM25-only is included as an unconditional anchor.

**Metric summary (hybrid RRF)**

| System | nDCG@10 | R@10 | R@20 |
|---|---|---|---|
| BM25 (anchor) | 0.0692 | 0.0646 | 0.0850 |
| LegalBERT — frozen | 0.0399 | 0.0419 | 0.0640 |
| LegalBERT — trained | **0.0910** | 0.0895 | 0.1251 |
| MiniLM — frozen | 0.0562 | 0.0534 | 0.0751 |
| MiniLM — trained | 0.0838 | 0.0822 | 0.1147 |

**Frozen → trained significance (hybrid RRF, per-query nDCG@10)**

| Encoder | Frozen mean | Trained mean | Delta | 95% CI | p (one-tailed) | N |
|---|---|---|---|---|---|---|
| LegalBERT | 0.0399 | 0.0910 | +0.0511 | [+0.0487, +0.0535] | < 0.0001 | 12,035 |
| MiniLM | 0.0562 | 0.0838 | +0.0276 | [+0.0255, +0.0298] | < 0.0001 | 12,035 |

The trained LegalBERT hybrid RRF is the best-performing system. For reference, the dense-only (no BM25 fusion) comparisons show larger encoder-isolated effects, because BM25 partially compensates for frozen encoder weakness and compresses the gap:

| Encoder | Frozen dense nDCG@10 | Trained dense nDCG@10 | Δ (dense) | 95% CI | p |
|---|---|---|---|---|---|
| LegalBERT | 0.0148 | 0.0860 | +0.0712 | [+0.0684, +0.0742] | < 0.0001 |
| MiniLM | 0.0315 | 0.0720 | +0.0404 | [+0.0380, +0.0429] | < 0.0001 |

All four lifts (hybrid RRF and dense, both bases) are significant at p < 0.0001, N = 12,035.

---

## 2. Encoder ablation: LegalBERT vs MiniLM (trained)

Both encoders were fine-tuned on the same 859,002 citation-derived triplets under identical training conditions (same data, same seed, same eval set). The comparison isolates the effect of the pre-training domain: a legal-domain base (LegalBERT, 768d) vs a general-purpose base (MiniLM, 384d).

| Method | Trained MiniLM nDCG@10 | Trained LegalBERT nDCG@10 | Delta | 95% CI | p | N |
|---|---|---|---|---|---|---|
| hybrid RRF | 0.0838 | 0.0910 | +0.0071 | [+0.0056, +0.0087] | < 0.0001 | 12,035 |

LegalBERT's domain pre-training provides a statistically significant additional lift beyond what fine-tuning alone achieves on MiniLM. The effect (+0.0071 nDCG@10, hybrid RRF) is smaller than the frozen→trained lift for either base (LegalBERT +0.0511, MiniLM +0.0276, both hybrid RRF), as expected: task-specific fine-tuning dominates domain pre-training once both models see the same citation-retrieval signal.

---

## 3. Reranker

A fine-tuned cross-encoder reranker (BGE-reranker-base, trained on the same citation pairs) was evaluated on top of the trained LegalBERT dense retrieval.

| System | Method | nDCG@10 | R@10 | Delta vs dense | 95% CI | p |
|---|---|---|---|---|---|---|
| trained LegalBERT | dense | 0.0860 | 0.0836 | — | — | — |
| trained LegalBERT + reranker | dense_rerank | 0.0845 | 0.0806 | -0.0015 | [-0.0044, +0.0014] | 0.855 |

The reranker showed **no significant improvement** on citation-grounded evaluation (p = 0.855). The model's in-loop training thermometer reached nDCG@10 = 0.979 on a held-out slice of training triplets — a sign that it fit the in-distribution training pairs well, not that it learned a generalizable reranking signal. The citation-grounded eval harness caught this precisely because it uses independent qrels from the test split, not derivatives of training data. This is the eval harness doing its job.

---

## 4. Judge validation

The LLM judge (claude-haiku-4-5-20251001, rubric v2) was validated against owner-provided binary labels on a stratified 80-pair sample (clearly relevant / borderline / clearly irrelevant). Four pairs with no judgeable legal content were excluded from both the judge run and the kappa computation, leaving N = 76 pairs.

**Cohen's kappa: 0.773** ("substantial agreement"). Raw (percent) agreement: 0.908.

**Confusion matrix (rows = human, cols = judge)**

| | Judge: RELEVANT | Judge: NOT RELEVANT |
|---|---|---|
| Human: RELEVANT | 18 | 2 |
| Human: NOT RELEVANT | 5 | 51 |

**Per-band breakdown**

| Band | n | Agreement | Kappa |
|---|---|---|---|
| Relevant (clearly relevant) | 25 | 0.920 | 0.833 |
| Borderline | 27 | 0.852 | 0.585 |
| Irrelevant (clearly irrelevant) | 24 | 0.958 | 0.000 |

The per-band kappa degeneration on the clearly-irrelevant band (kappa = 0.000 despite 95.8% agreement) is a well-known artifact of Cohen's kappa when one class dominates a band — the marginal probability correction inflates p_e toward 1, making kappa undefined or zero even at high agreement. This is the kappa paradox; the overall kappa of 0.773 is the meaningful figure.

The rubric was frozen at v2 after validation. All judge deployments in this project used claude-haiku-4-5-20251001 with rubric v2.

---

## 5. Recall-gap estimation

Citation qrels are precision-oriented: a case cites what it judged authoritative, not everything topically relevant. This systematically understates recall. To quantify how much, the validated judge (rubric v2, claude-haiku-4-5-20251001) was applied to the non-cited top-10 documents retrieved by the best system (trained LegalBERT hybrid RRF) across an 80-query sample (seed 42, drawn from the 12,035 eval queries).

**67 documents** were excluded before judging as content-free (per-curiam orders, pointer opinions, one-line affirmances — the same content-based detector used during judge validation). No PARSE_ERRORs.

| Metric | Value |
|---|---|
| Queries sampled | 80 |
| Non-cited top-10 pairs judged | 672 |
| Pairs judged RELEVANT | 346 |
| Pairs judged NOT RELEVANT | 326 |
| **Recall-gap rate** | **0.515** |
| 95% CI (bootstrap, N = 10,000, seed 42) | [0.478, 0.552] |

Roughly half of the top-10 documents that citation qrels count as misses are judged topically relevant by the validated judge. This does not invalidate the citation-grounded eval — it contextualizes it. The nDCG@10 numbers in Section 1 are a lower bound on true retrieval quality: the citation graph records only what authors chose to cite, not everything the retrieved case could have cited. Absolute scores understate real performance; the frozen→trained deltas remain valid comparisons because the lower-bound applies equally to all systems.

---

## 6. External benchmark: CLERC

To verify that the trained encoder's lift reflects a transferable legal retrieval signal rather than PA-specific overfitting, it was evaluated on CLERC (jhu-clsp/CLERC, NAACL 2025), a public US federal case citation-retrieval benchmark with independent qrels. The task structure is identical to this project's: given a judicial opinion excerpt with a citation removed, retrieve the cited prior case.

**Corpus:** 150,000 documents — all 2,723 qrel-relevant documents plus 147,277 randomly sampled distractors (reservoir sampling, seed 42). Brute-force cosine retrieval, no BM25 fusion.

| Model | nDCG@10 | R@10 | R@100 |
|---|---|---|---|
| Frozen LegalBERT | 0.0344 | 0.0572 | 0.1357 |
| Trained PA-LegalBERT | 0.0714 | 0.1284 | 0.3290 |

**Frozen → trained (CLERC, per-query nDCG@10):** delta = +0.0370, 95% CI [+0.0299, +0.0444], p < 0.0001, N = 2,851 queries (10,000 bootstrap resamples, seed 42).

The trained PA encoder more than doubles the frozen baseline on every metric on a dataset it was never trained on, in a different jurisdiction. The citation-retrieval signal transferred. The cross-jurisdiction lift (+0.037 nDCG@10) is smaller than the in-domain lift (+0.051, Section 1), consistent with expected domain fine-tuning behavior.

**Subsampling caveat.** The 150k-doc corpus makes absolute scores incomparable to the CLERC paper's full-corpus numbers (paper BM25 nDCG@10 = 0.054 on 1.84M docs). The valid finding here is the frozen→trained delta, not the absolute position on the CLERC leaderboard.

---

## 7. Methodology notes

**Lower-bound caveat.** Citation links are precision-oriented relevance signals: authors cite what they judged controlling authority, not an exhaustive topical survey. The citation-grounded qrels are a reliable ground truth for precision-oriented retrieval but a known undercount of true topical recall (Section 5 quantifies this as roughly 50% of top-10 non-cited docs being topically relevant). All nDCG and Recall figures are lower bounds on true performance. Comparisons between systems are valid because the undercount applies uniformly across all systems evaluated on the same qrels.

**Judge role-separation.** The LLM judge (claude-haiku-4-5-20251001) is used only for recall-gap estimation and rubric validation, not for computing the headline retrieval metrics. The generator (qwen3:8b via Ollama) handles RAG answer generation. These are distinct models with distinct roles; the judge never evaluates its own generator's output.

**Which judge produced which numbers.** All judge labels in this project — both the validation run (76 pairs, κ = 0.773) and the recall-gap deployment (672 pairs) — were produced by claude-haiku-4-5-20251001 with rubric v2. The rubric was frozen after validation and is committed to `eval/judge.py`.

**Leakage controls.** The citation graph is built on the full corpus and then split temporally: training triplets are drawn exclusively from the train split; all evaluation qrels are drawn from the eval split (cases decided after the training split cutoff). The split uses `graph/split.py` with seed 42. Two independent leakage checks were run and committed; zero overlap between training pair IDs and eval query IDs was confirmed.

**Seeds.** All randomized operations use seed 42: train/eval split, pair mining sampler, bootstrap resampler, CLERC corpus sampling, recall-gap query sampling.

**Metric cross-check.** The hand-written metrics in `eval/metrics.py` (nDCG@10, Recall@k, MRR, MAP) are cross-checked against pytrec_eval in `tests/test_metrics.py` on the same qrels + run, with assertion to floating-point tolerance. All results reported here use the hand-written implementation.

---

## 8. Hardware and reproducibility

**Hardware**

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 5060 Laptop GPU (Blackwell, sm_120, 8 GB VRAM) |
| RAM | 32 GB |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |

**Training cost (from manifests)**

| Model | Triplets | Epochs | Peak VRAM | Wall-clock |
|---|---|---|---|---|
| LegalBERT encoder | 859,002 | 1 | 5.0 GB | ~10.8 h |
| MiniLM encoder | 859,002 | 2 | 0.9 GB | ~3.8 h |
| Cross-encoder reranker | 859,002 | 1 (early-stopped) | 5.5 GB | ~5.1 h |

All training ran within the 8 GB VRAM budget with bf16 mixed precision and gradient checkpointing. The MiniLM encoder's small footprint (0.9 GB peak) reflects its 384-dimensional architecture; the reranker's 5.5 GB peak reflects the cross-encoder's longer input sequences (query + document concatenated).

**Reproducing the eval**

```
just eval          # regenerates results.csv from committed qrels + encoder checkpoints
just kappa         # recomputes Cohen's kappa from committed judge + human labels (no API calls)
just recall-gap    # re-runs LLM-judge recall-gap estimation (~672 API calls)
just clerc-bench   # re-runs CLERC external benchmark (~60 min, 7.6 GB collection cached)
```

All eval recipes are deterministic given the committed checkpoints and qrels. `just recall-gap` requires an Anthropic API key in `.env`.
