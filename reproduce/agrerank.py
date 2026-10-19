"""AgreRank scoring functions.

`agrerank` implements the method as written in the SIGIR 2026 paper (Eqs. 1-6),
with switches for the ablations and controls. `agrerank_legacy` is a line-by-line
port of the code that produced the paper's TREC numbers
(src/reranker_eval/agre_rank_with_baselines.py), kept so the two can be compared.

All functions work on one query at a time:
    q        (d,)   L2-normalized query embedding
    D        (n,d)  L2-normalized passage embeddings for the candidate pool
    sparse   (n,)   sparse (BM25) scores for the same passages
    ce       callable(indices) -> raw cross-encoder logits for those passages
Each returns a score per pool position (higher = better); positions never sent to
the cross-encoder are placed below every position that was.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.stats import rankdata

CEFn = Callable[[np.ndarray], np.ndarray]


def percentile(x: np.ndarray) -> np.ndarray:
    """Eq. 1: (rank - 1) / (|S| - 1), ascending rank, ties get the average rank."""
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 2:
        return np.zeros_like(x)
    return (rankdata(x, method="average") - 1.0) / (len(x) - 1.0)


def l2n(v: np.ndarray) -> np.ndarray:
    return v / max(np.linalg.norm(v), 1e-12)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def ranks_desc(s: np.ndarray) -> np.ndarray:
    """1-based ranks, best first; ties broken by pool position (deterministic)."""
    order = np.lexsort((np.arange(len(s)), -s))
    r = np.empty(len(s), dtype=np.int64)
    r[order] = np.arange(1, len(s) + 1)
    return r


def rrf(*scores: np.ndarray, k: int = 60) -> np.ndarray:
    return sum(1.0 / (k + ranks_desc(s)) for s in scores)


def top_m(s: np.ndarray, m: int) -> np.ndarray:
    return np.lexsort((np.arange(len(s)), -s))[:m]


def minmax(x: np.ndarray) -> np.ndarray:
    return (x - x.min()) / max(x.max() - x.min(), 1e-12)


def rerank_top(first_stage: np.ndarray, ce: CEFn, m: int, gamma: float = 1.0,
               ce_norm: str = "minmax") -> np.ndarray:
    """Cross-encode the top-m by `first_stage`.

    gamma = 1.0 gives the standard pipeline (pure CE order inside the top-m).
    gamma < 1 interpolates the CE score with the percentile of the first-stage
    score inside the top-m, i.e. the same fusion AgreRank uses in Eq. 6.

    The CE score is put on [0, 1] by min-max over the top-m logits. A sigmoid also
    maps to [0, 1] but saturates: for 85-96% of TREC DL queries the top-10 CE
    probabilities all exceed 0.99, which erases the CE's ordering at the top.
    """
    idx = top_m(first_stage, m)
    logits = ce(idx)
    c = sigmoid(logits) if ce_norm == "sigmoid" else minmax(logits)
    fused = gamma * c + (1.0 - gamma) * percentile(first_stage[idx]) if gamma < 1.0 else c
    out = np.full(len(first_stage), -np.inf)
    # Everything outside the CE pool keeps its first-stage order, below the CE pool.
    rest = np.setdiff1d(np.arange(len(first_stage)), idx)
    out[rest] = -1e6 + percentile(first_stage)[rest]
    out[idx] = fused
    return out


@dataclass(frozen=True)
class Config:
    tau: float = 0.8          # consensus threshold (Eq. 2)
    max_seeds: int = 15       # anchor cap
    min_seeds: int = 5        # below this, fall back to RRF seeds
    fallback_k: int = 10      # RRF top-k used by the fallback
    alpha: float = 0.65       # weight on the original query (Eq. 3)
    beta: float = 0.5         # gate floor (Eq. 4)
    gamma: float = 0.5        # weight on the CE in the final score (Eq. 6)
    M: int = 200              # cross-encoder depth
    ce_norm: str = "minmax"   # how the CE logit is mapped to [0, 1] in Eq. 6: "minmax" | "sigmoid"
    seeds: str = "consensus"  # "consensus" | "rrf" (leave-one-out: RRF top-max_seeds)
    weights: str = "product"  # "product" (w_d = Φsp·Φde) | "uniform"
    gate: bool = True
    stop_after: Optional[str] = None  # "seeds" | "expand" | "gate" (stepwise ablation)


@dataclass
class Trace:
    seeds: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    n_consensus: int = 0
    fallback: bool = False
    expanded_query: Optional[np.ndarray] = None


def select_seeds(ps: np.ndarray, pd: np.ndarray, rrf_s: np.ndarray, cfg: Config, tr: Trace):
    consensus = np.flatnonzero((ps > cfg.tau) & (pd > cfg.tau))
    tr.n_consensus = len(consensus)
    if cfg.seeds == "rrf":
        idx = top_m(rrf_s, cfg.max_seeds)
    elif len(consensus) >= cfg.min_seeds:
        w = ps[consensus] * pd[consensus]
        idx = consensus[np.lexsort((consensus, -w))][: cfg.max_seeds]
    else:
        tr.fallback = True
        idx = top_m(rrf_s, cfg.fallback_k)
        return idx, np.ones(len(idx))
    w = ps[idx] * pd[idx] if cfg.weights == "product" else np.ones(len(idx))
    return idx, w


def agrerank(q: np.ndarray, D: np.ndarray, sparse: np.ndarray, ce: Optional[CEFn],
             cfg: Config = Config(), requery: Optional[Callable[[np.ndarray], None]] = None,
             trace: Optional[Trace] = None) -> np.ndarray:
    """Paper-faithful AgreRank over one candidate pool.

    `requery`, if given, is called with the expanded query so the caller can add
    newly retrieved passages to the pool before scoring (Sec. 3.3). Without an
    index over the full corpus, expansion re-scores the existing pool.
    """
    tr = trace if trace is not None else Trace()
    dense = D @ q
    ps, pd = percentile(sparse), percentile(dense)
    rrf_s = rrf(sparse, dense)

    seeds, w = select_seeds(ps, pd, rrf_s, cfg, tr)
    tr.seeds = seeds
    if cfg.stop_after == "seeds":
        # Consensus documents first (by Φsp·Φde), then everything else by RRF.
        is_c = (ps > cfg.tau) & (pd > cfg.tau)
        return np.where(is_c, 2.0 + ps * pd, percentile(rrf_s))

    centroid = (w[:, None] * D[seeds]).sum(0) / w.sum()
    qp = l2n(cfg.alpha * q + (1.0 - cfg.alpha) * centroid)
    tr.expanded_query = qp
    if requery is not None:
        requery(qp)
    s_exp = D @ qp
    if cfg.stop_after == "expand":
        return s_exp

    gate = cfg.beta + (1.0 - cfg.beta) * np.maximum(0.0, dense) if cfg.gate else 1.0
    s3 = s_exp * gate
    if cfg.stop_after == "gate" or ce is None:
        return s3
    return rerank_top(s3, ce, cfg.M, gamma=cfg.gamma, ce_norm=cfg.ce_norm)


def agrerank_requery(q: np.ndarray, E: np.ndarray, sparse_all: np.ndarray, pool: np.ndarray,
                     ce: Optional[CEFn], cfg: Config = Config(), k_requery: int = 500,
                     trace: Optional[Trace] = None):
    """AgreRank with the Stage-2 re-query of Sec. 3.3, for corpora small enough to
    search exhaustively. Seeds come from the initial pool; the expanded query then
    retrieves its own top-`k_requery` from the whole corpus, and Stages 3-4 run on
    the union. `E` and `sparse_all` cover the whole corpus; `ce` takes corpus indices.
    Returns (corpus indices of the final pool, scores).
    """
    tr = trace if trace is not None else Trace()
    D = E[pool]
    dense = D @ q
    ps, pd = percentile(sparse_all[pool]), percentile(dense)
    seeds, w = select_seeds(ps, pd, rrf(sparse_all[pool], dense), cfg, tr)
    tr.seeds = pool[seeds]
    centroid = (w[:, None] * D[seeds]).sum(0) / w.sum()
    qp = l2n(cfg.alpha * q + (1.0 - cfg.alpha) * centroid)
    tr.expanded_query = qp
    new = top_m(E @ qp, k_requery)
    union = np.union1d(pool, new)
    Du = E[union]
    gate = cfg.beta + (1.0 - cfg.beta) * np.maximum(0.0, Du @ q) if cfg.gate else 1.0
    s3 = (Du @ qp) * gate
    if ce is None:
        return union, s3
    return union, rerank_top(s3, lambda idx: ce(union[idx]), cfg.M, gamma=cfg.gamma, ce_norm=cfg.ce_norm)


# ---------------------------------------------------------------------------
# Port of the code behind the paper's TREC tables (agre_rank_with_baselines.py).
# Differences from the paper: Φ = rank/n; no explicit seed set (RRF + 0.5 boost,
# z-scored and clipped at ±3, top-15); unweighted centroid; Φ applied before the
# gate; CE depth = 2 * cross_encoder_k; CE logits min-max mapped onto the range
# of the stage-3 scores. Documents outside the CE pool keep their stage-3 score.
# ---------------------------------------------------------------------------

def _legacy_pct(x):
    return rankdata(x, method="average") / len(x)


def _legacy_z(x):
    sd = np.std(x)
    return np.zeros_like(x) if sd == 0 else np.clip((x - x.mean()) / sd, -3, 3)


def agrerank_legacy(q, D, sparse, ce: CEFn, cross_encoder_k: int = 200) -> np.ndarray:
    dense = D @ q
    pd, ps = _legacy_pct(dense), _legacy_pct(sparse)
    dr = np.argsort(np.argsort(dense)[::-1])
    sr = np.argsort(np.argsort(sparse)[::-1])
    rrf_s = 1.0 / (60 + dr + 1) + 1.0 / (60 + sr + 1)
    boost = ((pd > 0.8) & (ps > 0.8)).astype(float) * 0.5
    init = _legacy_z(rrf_s + boost)
    seeds = np.argsort(init)[::-1][:15]
    qp = l2n(0.65 * q + 0.35 * D[seeds].mean(0))
    s3 = _legacy_pct(D @ qp) * (0.5 + 0.5 * np.maximum(dense, 0.0))
    pool = np.argsort(s3)[-min(2 * cross_encoder_k, len(s3)):]
    logits = ce(pool)
    lo, hi = logits.min(), logits.max()
    cn = (logits - lo) / (hi - lo if hi - lo > 1e-9 else 1.0)
    calib = cn * (s3.max() - s3.min()) + s3.min()
    out = s3.copy()
    out[pool] = 0.5 * calib + 0.5 * s3[pool]
    return out


def legacy_ce_baseline(first_stage: np.ndarray, ce: CEFn, m: int) -> np.ndarray:
    """Legacy X+CE baselines: CE order overrides the top-m, rest keep first-stage order."""
    idx = np.argsort(first_stage)[::-1][:m]
    logits = ce(idx)
    out = first_stage.astype(float).copy()
    out[idx] = first_stage.max() + 1.0 + (logits - logits.min())
    return out
