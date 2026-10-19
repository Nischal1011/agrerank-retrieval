"""Diagnostic study (paper Sec. 4.1, Tables 3, 4, 6b) rebuilt from the files in data/diagnostic/.

  python diag.py build --encoder openai --legacy-data /path/to/rag-lab/data   # import paper vectors
  python diag.py build --encoder bge                                          # or encode locally
  python diag.py eval  --encoder openai

Corpus: 1,249 chunks of 19 papers (data/diagnostic/chunks.jsonl).
Queries: 100 (50 single-chunk "factoid", 50 multi-chunk "synthesis"), 145 relevant chunks
(data/diagnostic/queries.json).
Noise: 10,000 template distractors from src/noisy_corpus.py (seed 42) -> 11,249 passages.
BM25: Pyserini/Lucene, k1=0.9, b=0.4 (as stated in the paper). The legacy rank_bm25 scorer
(library defaults) is reported as well, because the paper's diagnostic numbers used it.
Pool: dense top-500 union BM25 top-500 (score > 0), as in the paper's code.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

import agrerank as ag
from infra import ROOT, WORK, CrossEncoderCache, Embedder, dump
from trec_eval import legacy_tokenize

REPO = ROOT.parent
DATA = REPO / "data" / "diagnostic"
POOL_K = 500


def load_corpus(noisy: bool):
    chunks = [json.loads(l) for l in open(DATA / "chunks.jsonl")]
    ids = [c["chunk_id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    if noisy:
        sys.path.insert(0, str(REPO / "src"))
        from noisy_corpus import generate_noise_text
        noise = generate_noise_text(10000, seed=42)
        ids += [f"noise_{i:06d}" for i in range(len(noise))]
        texts += noise
    return ids, texts


def load_queries():
    return json.loads((DATA / "queries.json").read_text())


# ------------------------------- build --------------------------------------

def import_legacy(legacy: Path, emb: Embedder):
    """Import the paper's text-embedding-3-large vectors, with integrity checks."""
    rep = {}
    ids, texts = load_corpus(noisy=True)
    old_ids = [x.decode() if isinstance(x, bytes) else str(x)
               for x in np.load(legacy / "embeddings/fixed250/corpus_ids.npy", allow_pickle=True)]
    old_noisy = [json.loads(l) for l in open(legacy / "noisy_corpus_10k.jsonl")]
    E = np.load(legacy / "noisy_embeddings_10k.npy").astype(np.float32)
    rep["chunk_ids_match"] = old_ids == ids[:1249]
    rep["noisy_corpus_ids_match"] = [r["chunk_id"] for r in old_noisy] == ids
    rep["noisy_corpus_texts_match"] = [r["text"] for r in old_noisy] == texts
    norms = np.linalg.norm(E, axis=1)
    rep["bad_vectors"] = int((~np.isfinite(norms) | (np.abs(norms - 1) > 1e-3)).sum())
    if not (rep["chunk_ids_match"] and rep["noisy_corpus_ids_match"] and rep["noisy_corpus_texts_match"]):
        raise SystemExit(f"legacy corpus does not match data/diagnostic: {rep}")
    emb.store.put_many((emb.key(t, "doc"), v.tobytes()) for t, v in zip(texts, E))
    qs = load_queries()
    qids = [x.decode() if isinstance(x, bytes) else str(x)
            for x in np.load(legacy / "qset_embeddings/openai-large/qset_ids.npy", allow_pickle=True)]
    Q = np.load(legacy / "qset_embeddings/openai-large/qset_embeddings.npy").astype(np.float32)
    rep["query_ids_match"] = qids == [q["query_id"] for q in qs]
    if not rep["query_ids_match"]:
        raise SystemExit(f"legacy query embeddings do not match queries.json: {rep}")
    emb.store.put_many((emb.key(q["query_text"], "query"), v.tobytes()) for q, v in zip(qs, Q))
    print("imported legacy vectors:", rep)
    dump(ROOT / "outputs" / "legacy_diag_embedding_import.json", rep)


def build_index(noisy: bool) -> Path:
    name = "noisy" if noisy else "clean"
    out = WORK / "diag" / f"lucene_{name}"
    if out.exists():
        return out
    src = WORK / "diag" / f"json_{name}"
    src.mkdir(parents=True, exist_ok=True)
    ids, texts = load_corpus(noisy)
    with open(src / "docs.jsonl", "w") as f:
        for i, t in zip(ids, texts):
            f.write(json.dumps({"id": i, "contents": t}) + "\n")
    subprocess.run([sys.executable, "-m", "pyserini.index.lucene", "--collection", "JsonCollection",
                    "--input", str(src), "--index", str(out), "--generator", "DefaultLuceneDocumentGenerator",
                    "--threads", "4", "--storeRaw"], check=True, stdout=subprocess.DEVNULL)
    shutil.rmtree(src)
    return out


def build(encoder: str, legacy: Path | None):
    emb = Embedder(encoder)
    if encoder == "openai" and legacy is not None:
        import_legacy(legacy, emb)
    ids, texts = load_corpus(noisy=True)
    emb.encode(texts, role="doc", progress=f"diag {encoder}")
    emb.encode([q["query_text"] for q in load_queries()], role="query")
    for noisy in (False, True):
        build_index(noisy)
    print("diagnostic build done")


# -------------------------------- eval --------------------------------------

def bm25_all(index: Path, ids, queries):
    from pyserini.search.lucene import LuceneSearcher
    s = LuceneSearcher(str(index))
    s.set_bm25(k1=0.9, b=0.4)
    pos = {d: i for i, d in enumerate(ids)}
    out = np.zeros((len(queries), len(ids)))
    for qi, q in enumerate(queries):
        for h in s.search(q["query_text"], k=len(ids)):
            out[qi, pos[h.docid]] = h.score
    return out


def legacy_bm25_all(texts, queries):
    from rank_bm25 import BM25Okapi
    bm = BM25Okapi([legacy_tokenize(t) for t in texts])
    return np.stack([bm.get_scores(legacy_tokenize(q["query_text"])) for q in queries])


def ranks_of(rel_idx, order):
    pos = {d: r for r, d in enumerate(order, 1)}
    return [pos.get(i, 10 ** 9) for i in rel_idx]


def query_metrics(r):
    best = min(r)
    return {"mrr10": 1.0 / best if best <= 10 else 0.0, "r10": float(best <= 10),
            "cov10": float(np.mean([x <= 10 for x in r]))}


def evaluate_corpus(noisy: bool, encoder: str, alphas=(1.0, 0.65, 0.5)):
    ids, texts = load_corpus(noisy)
    qs = load_queries()
    emb, cecache = Embedder(encoder), CrossEncoderCache()
    E = emb.encode(texts, role="doc").astype(np.float64)
    Q = emb.encode([q["query_text"] for q in qs], role="query").astype(np.float64)
    B = bm25_all(build_index(noisy), ids, qs)
    LB = legacy_bm25_all(texts, qs)
    pos = {d: i for i, d in enumerate(ids)}
    P = ag.Config()
    per_sys, consensus, seeds = {}, [], []
    for qi, q in enumerate(qs):
        qv, bm, lb = Q[qi], B[qi], LB[qi]
        dense_all = E @ qv
        rel = [pos[c] for c in q["relevant_chunks"]]

        def ce_corpus(idx, _q=q["query_text"]):
            return cecache.score(_q, [texts[i] for i in idx])

        def pool_for(sp):
            d_top = ag.top_m(dense_all, POOL_K)
            s_top = [i for i in ag.top_m(sp, POOL_K) if sp[i] > 0]
            return np.union1d(d_top, s_top).astype(int)

        out = {}

        def add(name, pool, scores):
            order = pool[np.lexsort((pool, -scores))]
            out[name] = ranks_of(rel, order)

        allidx = np.arange(len(ids))
        add("BM25", allidx, bm)
        add("legacy-BM25", allidx, lb)
        add("Dense", allidx, dense_all)
        for tag, sp in (("", bm), ("legacy-", lb)):
            pool = pool_for(sp)
            D, s, d = E[pool], sp[pool], dense_all[pool]
            ce = lambda idx, _p=pool: ce_corpus(_p[idx])
            rrf = ag.rrf(s, d)
            add(f"{tag}RRF", pool, rrf)
            add(f"{tag}RRF+CE", pool, ag.rerank_top(rrf, ce, 200))
            if tag == "legacy-":  # paper's BM25 scorer: only the systems shown in Table 4
                add("legacy-AgreRank-pool", pool, ag.agrerank(qv, D, s, ce, P))
                continue
            add(f"{tag}Dense+CE", pool, ag.rerank_top(d, ce, 200))
            add(f"{tag}control-Dense+CE-interp", pool, ag.rerank_top(d, ce, 200, gamma=0.5))
            for a in alphas:
                cfg = replace(P, alpha=a)
                sfx = "" if a == 0.65 else f"-alpha={a}"
                tr = ag.Trace()
                if a == 0.65:
                    add(f"{tag}AgreRank-pool", pool, ag.agrerank(qv, D, s, ce, cfg, trace=tr))
                u, sc = ag.agrerank_requery(qv, E, sp, pool, ce_corpus, cfg, k_requery=POOL_K)
                add(f"{tag}AgreRank-requery{sfx}", u, sc)
                if tag == "" and a == 0.65:
                    seeds.append({"qid": q["query_id"], "n_consensus": tr.n_consensus, "fallback": tr.fallback})
            if tag == "":
                ps, pd = ag.percentile(s), ag.percentile(d)
                region = np.where(ps > P.tau, np.where(pd > P.tau, "both", "bm25_only"),
                                  np.where(pd > P.tau, "dense_only", "neither"))
                relset = set(rel)
                consensus.append({"pool": len(pool),
                                  "sizes": {r: int((region == r).sum()) for r in ("both", "dense_only", "bm25_only", "neither")},
                                  "relevant": {r: int(sum(1 for j, i in enumerate(pool) if i in relset and region[j] == r))
                                               for r in ("both", "dense_only", "bm25_only", "neither")},
                                  "relevant_outside_pool": int(len(relset - set(pool.tolist())))})
        for name, r in out.items():
            per_sys.setdefault(name, []).append({"qid": q["query_id"], "type": q["query_type"], **query_metrics(r)})
        print(f"{'noisy' if noisy else 'clean'} {encoder}: {qi + 1}/{len(qs)}", end="\r", flush=True)
    print()
    return per_sys, consensus, seeds


def summarize(per_sys):
    out = {}
    for name, rows in per_sys.items():
        s = {}
        for subset, keep in (("all", lambda r: True), ("factoid", lambda r: "single" in r["type"]),
                             ("synthesis", lambda r: "single" not in r["type"])):
            rr = [r for r in rows if keep(r)]
            s[subset] = {m: float(np.mean([r[m] for r in rr])) for m in ("mrr10", "r10", "cov10")}
        out[name] = s
    return out


def boot_diff(rows_a, rows_b, metric, subset):
    keep = (lambda r: True) if subset == "all" else (lambda r: ("single" in r["type"]) == (subset == "factoid"))
    a = np.array([r[metric] for r in rows_a if keep(r)])
    b = np.array([r[metric] for r in rows_b if keep(r)])
    d = a - b
    rng = np.random.default_rng(0)
    bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
    return {"diff": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def consensus_table(rows):
    tot_rel = {r: sum(x["relevant"][r] for x in rows) for r in rows[0]["relevant"]}
    tot_size = {r: sum(x["sizes"][r] for x in rows) for r in rows[0]["sizes"]}
    prec = {r: (tot_rel[r] / tot_size[r] if tot_size[r] else 0.0) for r in tot_rel}
    return {"relevant": tot_rel, "relevant_outside_pool": sum(x["relevant_outside_pool"] for x in rows),
            "region_sizes": tot_size, "mean_pool": float(np.mean([x["pool"] for x in rows])),
            "precision": prec,
            "lift_both_vs_dense_only": (prec["both"] / prec["dense_only"]) if prec["dense_only"] else None,
            "lift_both_vs_pool": prec["both"] / (sum(tot_rel.values()) / sum(tot_size.values()))}


def evaluate(encoder: str):
    res = {}
    for noisy in (False, True):
        key = "noisy" if noisy else "clean"
        per_sys, cons, seeds = evaluate_corpus(noisy, encoder)
        res[key] = {"systems": summarize(per_sys),
                    "consensus_regions": consensus_table(cons),
                    "seeds": {"mean_consensus_docs": float(np.mean([s["n_consensus"] for s in seeds])),
                              "fallback_queries": int(sum(s["fallback"] for s in seeds))},
                    "bootstrap_vs_RRF+CE": {f"{sysname} {m} {sub}": boot_diff(per_sys[sysname], per_sys["RRF+CE"], m, sub)
                                            for sysname in ("AgreRank-pool", "AgreRank-requery", "control-Dense+CE-interp")
                                            for m in ("mrr10", "r10") for sub in ("all", "factoid")},
                    "per_query": per_sys}
    sfx = "" if encoder == "openai" else f"_{encoder}"
    dump(ROOT / "outputs" / f"diag_results{sfx}.json", res)
    for key in ("clean", "noisy"):
        s = res[key]["systems"]
        print(key, {n: round(s[n]["factoid"]["mrr10"], 3) for n in ("BM25", "legacy-BM25", "Dense", "RRF", "RRF+CE",
                                                                      "AgreRank-pool", "AgreRank-requery")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "eval"])
    ap.add_argument("--encoder", default="openai", choices=["openai", "bge", "e5"])
    ap.add_argument("--legacy-data", type=Path, default=None)
    a = ap.parse_args()
    build(a.encoder, a.legacy_data) if a.cmd == "build" else evaluate(a.encoder)
