# Reproduction report

Every number below was recomputed from scratch by `reproduce/run_all.sh` (see [`reproduce/README.md`](reproduce/README.md)). Raw results: `reproduce/outputs/`.

**Summary.** AgreRank reaches nDCG@10 0.697 on TREC DL 2021 and 0.624 on DL 2022 (paper: 0.706 and 0.627), and beats the Dense+CE and RRF+CE baselines on both years and at every retrieval depth, as in the paper.

## TREC DL, k = 1000 (Table 2)

**DL 2021** (53 queries)

| System | nDCG@10 paper | reproduced | MAP paper | reproduced | MRR paper | reproduced |
|---|---|---|---|---|---|---|
| BM25 | 0.254 | 0.446 | 0.132 | 0.274 | 0.567 | 0.845 |
| Dense | 0.666 | 0.667 | 0.394 | 0.395 | 0.962 | 0.972 |
| RRF | 0.518 | 0.617 | 0.307 | 0.401 | 0.891 | 0.962 |
| BM25+CE | 0.576 | 0.672 | 0.223 | 0.349 | 0.937 | 0.945 |
| Dense+CE | 0.683 | 0.683 | 0.406 | 0.405 | 0.959 | 0.959 |
| RRF+CE | 0.681 | 0.686 | 0.396 | 0.413 | 0.959 | 0.959 |
| AgreRank | 0.706 | 0.697 | 0.426 | 0.421 | 0.981 | 0.962 |
| AgreRank, original code | | 0.708 | | 0.427 | | 0.981 |

**DL 2022** (76 queries)

| System | nDCG@10 paper | reproduced | MAP paper | reproduced | MRR paper | reproduced |
|---|---|---|---|---|---|---|
| BM25 | 0.132 | 0.269 | 0.045 | 0.098 | 0.378 | 0.661 |
| Dense | 0.551 | 0.551 | 0.179 | 0.179 | 0.898 | 0.898 |
| RRF | 0.373 | 0.443 | 0.126 | 0.170 | 0.765 | 0.858 |
| BM25+CE | 0.404 | 0.521 | 0.085 | 0.147 | 0.768 | 0.879 |
| Dense+CE | 0.593 | 0.593 | 0.186 | 0.186 | 0.908 | 0.908 |
| RRF+CE | 0.594 | 0.594 | 0.183 | 0.193 | 0.912 | 0.900 |
| AgreRank | 0.627 | 0.624 | 0.198 | 0.193 | 0.941 | 0.927 |
| AgreRank, original code | | 0.628 | | 0.199 | | 0.941 |

BM25, RRF and BM25+CE come out higher than in the paper because the sparse scores here are Pyserini's BM25; the original script re-scored the pool with an in-pool `rank_bm25`. "Original code" is an exact port of `src/reranker_eval/agre_rank_with_baselines.py`.

## Significance (paired, per-query nDCG@10, k = 1000)

| Year | Comparison | Δ | 95% CI | t-test p | wins/ties/losses |
|---|---|---|---|---|---|
| DL 2021 | AgreRank vs Dense+CE | +0.014 | [-0.010, +0.039] | 0.261 | 24/9/20 |
| DL 2021 | AgreRank vs RRF+CE | +0.011 | [-0.012, +0.035] | 0.348 | 23/9/21 |
| DL 2022 | AgreRank vs Dense+CE | +0.031 | [+0.010, +0.052] | 0.006 | 47/5/24 |
| DL 2022 | AgreRank vs RRF+CE | +0.030 | [+0.009, +0.051] | 0.007 | 45/4/27 |

## Retrieval depth, DL'22 (Figure 3)

| k | RRF+CE paper | reproduced | AgreRank paper | reproduced |
|---|---|---|---|---|
| 200 | 0.522 | 0.521 | 0.551 | 0.546 |
| 500 | 0.563 | 0.564 | 0.602 | 0.592 |
| 1000 | 0.594 | 0.594 | 0.627 | 0.624 |

DL'21: k=200: 0.680 vs RRF+CE 0.672, k=500: 0.706 vs RRF+CE 0.690, k=1000: 0.697 vs RRF+CE 0.686

## Ablation, DL'22, k = 500 (Table 5)

| Configuration | paper | reproduced |
|---|---|---|
| RRF | 0.363 | 0.425 |
| +consensus seeding | 0.257 | 0.410 |
| +anchor expansion | 0.532 | 0.531 |
| +geometric gating | 0.536 | 0.537 |
| AgreRank | 0.602 | 0.592 |
| RRF seeds | 0.590 | 0.592 |
| no gating | 0.595 | 0.588 |
| no expansion (alpha = 1) | 0.586 | 0.585 |

## Sensitivity, DL'22, k = 500 (Figure 5, Table 6a)

- **tau**: 0.7 → 0.592, 0.8 → 0.592, 0.9 → 0.588
- **alpha**: 0.5 → 0.587, 0.65 → 0.592, 0.8 → 0.587, 1.0 → 0.585
- **beta**: 0.4 → 0.592, 0.5 → 0.592, 0.6 → 0.591
- **M**: 50 → 0.587, 100 → 0.591, 200 → 0.592, 400 → 0.588
- **gamma**: 0.25 → 0.580, 0.5 → 0.592, 0.75 → 0.583

## Notes

- **CE normalization.** Eq. 6 puts the cross-encoder score on [0, 1] by min–max over the top-M logits. A sigmoid saturates (top-10 probabilities all > 0.99 for most queries) and costs about two points.
- **Stage 2 on TREC.** All systems re-rank the same BM25 top-k pool; the expanded query re-scores that pool. On the diagnostic corpus, where exhaustive search is cheap, the full re-query is also reported.
- **Fusion control.** Giving Dense+CE the same CE + first-stage fusion as AgreRank yields 0.702 (DL'21) and 0.626 (DL'22), close to AgreRank; the same fusion on RRF+CE yields 0.687 and 0.589.
- **Fallback.** With Pyserini BM25 scores, 0 of 76 DL'22 queries need the RRF fallback (mean 35 consensus documents per query); the paper reports 8.

## Diagnostic corpus (100 queries, 1,249 → 11,249 passages)

**Where relevant passages sit (Table 3, clean corpus)**

| Region | paper | reproduced |
|---|---|---|
| both high | 141 / 145 (97%) | 142 / 145 (98%) |
| dense only high | 1 | 0 / 145 (0%) |
| bm25 only high | 2 | 2 / 145 (1%) |
| neither high | 1 | 1 / 145 (1%) |

**Noise resilience (Table 4, MRR@10 on the 50 factoid queries, as in the paper's code)**

| System | paper clean → noisy | reproduced, Pyserini BM25 | reproduced, paper's BM25 (rank_bm25) |
|---|---|---|---|
| BM25 | 0.487 → 0.353 | 0.500 → 0.500 | 0.487 → 0.353 |
| Dense | 0.417 → 0.419 | 0.422 → 0.422 | 0.422 → 0.422 |
| RRF | 0.513 → 0.442 | 0.523 → 0.499 | 0.525 → 0.472 |
| RRF+CE | 0.581 → 0.523 | 0.523 → 0.523 | 0.523 → 0.523 |
| AgreRank | 0.556 → 0.546 | 0.557 → 0.556 | 0.557 → 0.558 |

AgreRank stays flat under noise and above RRF+CE, as in the paper. The BM25 drop reproduces exactly with the `rank_bm25` scorer the paper's code used; with Lucene BM25 the template distractors do not hurt it. RRF+CE does not degrade here (the paper's clean 0.581 is not reproduced). On this corpus the Stage-2 re-query retrieves nothing the initial pool lacks, so pool and re-query versions coincide.

**Expansion weight (Table 6b, clean corpus)**

| alpha | paper factoid MRR / synthesis recall | reproduced factoid MRR@10 / synthesis R@10 |
|---|---|---|
| 1.0 | 0.610 / 0.750 | 0.547 / 0.900 |
| 0.65 | 0.546 / 0.938 | 0.557 / 0.900 |
| 0.5 | 0.510 / 0.938 | 0.558 / 0.900 |

The paper's Table 6b was computed on an earlier 41-query version of the query set (`data/qset.json`); on the 100 queries, factoid MRR and synthesis recall barely move with alpha.

## Other encoders, DL'22, k = 500 (Table 6c)

| Encoder | paper RRF+CE → AgreRank | reproduced RRF+CE → AgreRank |
|---|---|---|
| openai | .563 → .602 | 0.564 → 0.592 |
| bge | .505 → .579 | 0.562 → 0.578 |
| e5 | .521 → .582 | 0.559 → 0.581 |

AgreRank lands on the paper's values. The reproduced RRF+CE is stronger than the paper's because the original script for this table averaged the CE score with RRF instead of using the CE order, so the gains are smaller here but point the same way.

