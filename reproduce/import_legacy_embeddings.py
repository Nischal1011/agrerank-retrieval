"""Import the text-embedding-3-large vectors cached when the paper's runs were made.

Use this when re-embedding through the OpenAI API is not possible. Every vector is
checked before it is accepted:
  * the cached candidate list equals the freshly built Pyserini pool (same ids, same order)
  * the cached query text equals the TREC query text
  * the cached passage text equals the ir_datasets passage text (after strip())
  * the vector is finite, non-zero and unit-norm (the legacy code silently zero-filled
    failed API batches)
Writes outputs/legacy_embedding_import.json with the counts.

    python import_legacy_embeddings.py --legacy-cache /path/to/rag-lab/cache
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from infra import ROOT, Embedder, dump
from trec_build import load_pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy-cache", required=True, type=Path)
    ap.add_argument("--years", nargs="+", default=["2021", "2022"])
    a = ap.parse_args()
    emb = Embedder("openai")
    report = {}
    for year in a.years:
        pools, texts = load_pool(year)
        c = {"queries": 0, "pool_mismatch": 0, "query_text_mismatch": 0, "passage_text_mismatch": 0,
             "bad_vectors": 0, "imported_passages": 0}
        rows = {}
        for p in pools:
            f = a.legacy_cache / f"msmarco-passage-v2_trec-dl-{year}_judged_k1000" / f"{p['qid']}.npz"
            d = np.load(f, allow_pickle=True)
            c["queries"] += 1
            if list(d["cand_ids"]) != p["docids"]:
                c["pool_mismatch"] += 1
                continue
            if str(d["query_text"]).strip() != p["query"].strip():
                c["query_text_mismatch"] += 1
                continue
            old_docs = d["candidate_docs"].item()
            V = np.asarray(d["doc_vecs"], dtype=np.float32)
            qv = np.asarray(d["query_vec"], dtype=np.float32).reshape(-1)
            ok = lambda v: np.all(np.isfinite(v)) and abs(np.linalg.norm(v) - 1) < 1e-3
            if not ok(qv):
                c["bad_vectors"] += 1
                continue
            rows[emb.key(p["query"], "query")] = qv
            for j, docid in enumerate(p["docids"]):
                if old_docs[docid].strip() != texts[docid].strip():
                    c["passage_text_mismatch"] += 1
                    continue
                if not ok(V[j]):
                    c["bad_vectors"] += 1
                    continue
                rows[emb.key(texts[docid], "doc")] = V[j]
        emb.store.put_many((k, v.tobytes()) for k, v in rows.items())
        c["imported_passages"] = len(rows) - c["queries"]
        c["unique_passages_in_pool"] = len(texts)
        report[f"dl{year}"] = c
        print(year, c)
    dump(ROOT / "outputs" / "legacy_embedding_import.json", report)


if __name__ == "__main__":
    main()
