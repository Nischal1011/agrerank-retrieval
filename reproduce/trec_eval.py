"""Evaluate every system, control, ablation and sensitivity setting on TREC DL 2021/2022.

Run trec_build.py first. Everything here reads cached pools, embeddings and CE
logits, so it is deterministic and takes a few minutes.

All systems re-rank the same Pyserini BM25 top-k pool and use the same
cross-encoder depth (200). Writes outputs/trec_results.json and outputs/trec_per_query.json.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np

import agrerank as ag
from infra import ROOT, WORK, CrossEncoderCache, Embedder, dump, mean_metrics, paired_tests, trec_metrics, write_run

OUT = ROOT / "outputs"
DEPTHS = [200, 500, 1000]

# Legacy in-pool sparse scorer (rank_bm25 with library defaults, custom tokenizer).
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)*")
_STOP = set("the a an and or for to of in on at by from with as is are was were be be be being this that these those which who whose where when".split())


def legacy_tokenize(text):
    return [t for t in (x.lower() for x in _TOKEN_RE.findall(text or "")) if t not in _STOP and len(t) > 1]


def legacy_sparse(query, passages):
    from rank_bm25 import BM25Okapi
    return np.asarray(BM25Okapi([legacy_tokenize(t) for t in passages]).get_scores(legacy_tokenize(query)), dtype=float)


def systems_for(q, D, bm25, lsp, ce):
    """All systems for one pool. Returns {name: scores} plus seed statistics."""
    P = ag.Config()
    dense = D @ q
    rrf = ag.rrf(bm25, dense)
    S = {}
    # Baselines as defined in the paper: first stage, then pure CE order in the top-200
    S["BM25"] = bm25
    S["Dense"] = dense
    S["RRF"] = rrf
    S["BM25+CE"] = ag.rerank_top(bm25, ce, 200)
    S["Dense+CE"] = ag.rerank_top(dense, ce, 200)
    S["RRF+CE"] = ag.rerank_top(rrf, ce, 200)
    # AgreRank (Eqs. 1-6)
    tr = ag.Trace()
    S["AgreRank"] = ag.agrerank(q, D, bm25, ce, P, trace=tr)
    # Code behind the paper's Table 2, for reference
    S["AgreRank (original code)"] = ag.agrerank_legacy(q, D, lsp, ce, cross_encoder_k=200)
    # Control: baselines with the same CE + first-stage fusion AgreRank uses
    S["Dense+CE fused"] = ag.rerank_top(dense, ce, 200, gamma=0.5)
    S["RRF+CE fused"] = ag.rerank_top(rrf, ce, 200, gamma=0.5)
    # Ablation (Table 5)
    S["abl: +consensus seeding"] = ag.agrerank(q, D, bm25, None, replace(P, stop_after="seeds"))
    S["abl: +anchor expansion"] = ag.agrerank(q, D, bm25, None, replace(P, stop_after="expand"))
    S["abl: +geometric gating"] = ag.agrerank(q, D, bm25, None, replace(P, stop_after="gate"))
    S["abl: RRF seeds"] = ag.agrerank(q, D, bm25, ce, replace(P, seeds="rrf"))
    S["abl: no gating"] = ag.agrerank(q, D, bm25, ce, replace(P, gate=False))
    S["abl: no expansion"] = ag.agrerank(q, D, bm25, ce, replace(P, alpha=1.0))
    # Sensitivity (one parameter at a time around the defaults)
    for t in (0.7, 0.8, 0.9):
        S[f"sens tau={t}"] = ag.agrerank(q, D, bm25, ce, replace(P, tau=t))
    for a in (0.5, 0.65, 0.8, 1.0):
        S[f"sens alpha={a}"] = ag.agrerank(q, D, bm25, ce, replace(P, alpha=a))
    for b in (0.4, 0.5, 0.6):
        S[f"sens beta={b}"] = ag.agrerank(q, D, bm25, ce, replace(P, beta=b))
    for M in (50, 100, 200, 400):
        S[f"sens M={M}"] = ag.agrerank(q, D, bm25, ce, replace(P, M=M))
    for g in (0.25, 0.5, 0.75):
        S[f"sens gamma={g}"] = ag.agrerank(q, D, bm25, ce, replace(P, gamma=g))
    stats = {"n_consensus": tr.n_consensus, "fallback": tr.fallback}
    return S, stats


COMPARISONS = [
    ("AgreRank", "Dense+CE"),
    ("AgreRank", "RRF+CE"),
    ("AgreRank", "Dense+CE fused"),
    ("AgreRank", "RRF+CE fused"),
    ("AgreRank (original code)", "RRF+CE"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", default=["2021", "2022"])
    ap.add_argument("--save-runs", action="store_true", help="write TREC run files to work/runs/")
    ap.add_argument("--encoder", default="openai", choices=["openai", "bge", "e5"])
    a = ap.parse_args()
    import ir_datasets
    from trec_build import load_pool

    emb, cache = Embedder(a.encoder), CrossEncoderCache()
    results, per_query = {}, {}
    for year in a.years:
        ds = ir_datasets.load(f"msmarco-passage-v2/trec-dl-{year}/judged")
        qrels = {}
        for r in ds.qrels_iter():
            qrels.setdefault(r.query_id, {})[r.doc_id] = int(r.relevance)
        pools, texts = load_pool(year)
        runs = {k: {} for k in DEPTHS}
        seedstats = {k: {} for k in DEPTHS}
        for i, p in enumerate(pools):
            passages = [texts[d] for d in p["docids"]]
            q = emb.encode([p["query"]], role="query")[0].astype(np.float64)
            Dfull = emb.encode(passages, role="doc").astype(np.float64)
            logits = cache.score(p["query"], passages)
            for k in DEPTHS:
                D, ids = Dfull[:k], p["docids"][:k]
                bm25 = np.asarray(p["bm25"][:k], dtype=float)
                lsp = legacy_sparse(p["query"], passages[:k])
                ce = lambda idx, _l=logits: _l[idx]
                S, st = systems_for(q, D, bm25, lsp, ce)
                seedstats[k][p["qid"]] = st
                for name, s in S.items():
                    runs[k].setdefault(name, {})[p["qid"]] = {d: float(v) for d, v in zip(ids, s)}
            print(f"dl{year}: {i + 1}/{len(pools)}", end="\r", flush=True)
        print()
        for k in DEPTHS:
            key = f"dl{year}_k{k}" + ("" if a.encoder == "openai" else f"_{a.encoder}")
            pq = {name: trec_metrics(qrels, run) for name, run in runs[k].items()}
            per_query[key] = {name: {q: m["ndcg10"] for q, m in v.items()} for name, v in pq.items()}
            ss = seedstats[k]
            results[key] = {
                "n_queries": len(pools),
                "systems": {name: mean_metrics(v) for name, v in pq.items()},
                "tests_ndcg10": {f"{x} vs {y}": paired_tests(per_query[key][x], per_query[key][y]) for x, y in COMPARISONS},
                "seeds": {"mean_consensus_docs": float(np.mean([s["n_consensus"] for s in ss.values()])),
                          "fallback_queries": int(sum(s["fallback"] for s in ss.values()))},
            }
            if a.save_runs:
                for name, run in runs[k].items():
                    write_run(WORK / "runs" / key / f"{name}.run", run, name)
            print(f"{key}: AgreRank {results[key]['systems']['AgreRank']['ndcg10']:.4f}  "
                  f"RRF+CE {results[key]['systems']['RRF+CE']['ndcg10']:.4f}")
    sfx = "" if a.encoder == "openai" else f"_{a.encoder}"
    dump(OUT / f"trec_results{sfx}.json", results)
    dump(OUT / f"trec_per_query{sfx}.json", per_query)


if __name__ == "__main__":
    main()
