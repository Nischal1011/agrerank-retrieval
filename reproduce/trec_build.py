"""Build the TREC DL candidate pools from scratch.

For each judged query of TREC DL 2021 / 2022 (MS MARCO v2 passages):
  1. Pyserini BM25 (k1=0.9, b=0.4) top-1000 from the prebuilt msmarco-v2-passage index.
     Pools for k=200/500 are prefixes of this list.
  2. Passage text from ir_datasets.
  3. Query and passage embeddings (--encoder openai | bge | e5). OpenAI vectors can
     instead be imported from the paper's cache with import_legacy_embeddings.py.
  4. Cross-encoder logits for every (query, pool passage) pair.

Outputs (under $AGRERANK_WORK, default reproduce/work/):
  trec/dl{year}.pool.jsonl   qid, query, docids, bm25 (Pyserini scores)
  trec/dl{year}.texts.jsonl  docid -> passage text
  cache/*.sqlite             embeddings and CE logits (keyed by text hash)

Requires Java 21 for Pyserini (set JAVA_HOME).
"""
from __future__ import annotations

import argparse
import json
import time

from infra import WORK, CrossEncoderCache, Embedder

DEPTH = 1000


def build_pool(year: str):
    import ir_datasets
    from pyserini.search.lucene import LuceneSearcher

    ds = ir_datasets.load(f"msmarco-passage-v2/trec-dl-{year}/judged")
    searcher = LuceneSearcher.from_prebuilt_index("msmarco-v2-passage")
    searcher.set_bm25(k1=0.9, b=0.4)
    store = ds.docs_store()
    out_pool = WORK / "trec" / f"dl{year}.pool.jsonl"
    out_text = WORK / "trec" / f"dl{year}.texts.jsonl"
    out_pool.parent.mkdir(parents=True, exist_ok=True)
    texts = {}
    with open(out_pool, "w") as f:
        for q in ds.queries_iter():
            hits = searcher.search(q.text, k=DEPTH)
            ids = [h.docid for h in hits]
            docs = store.get_many(ids)
            for d in ids:
                texts[d] = docs[d].text
            f.write(json.dumps({"qid": q.query_id, "query": q.text, "docids": ids,
                                "bm25": [float(h.score) for h in hits]}) + "\n")
    with open(out_text, "w") as f:
        for d, t in texts.items():
            f.write(json.dumps({"docid": d, "text": t}) + "\n")
    print(f"dl{year}: pools written ({len(texts)} unique passages)")


def load_pool(year: str):
    pools = [json.loads(l) for l in open(WORK / "trec" / f"dl{year}.pool.jsonl")]
    texts = {}
    for l in open(WORK / "trec" / f"dl{year}.texts.jsonl"):
        r = json.loads(l)
        texts[r["docid"]] = r["text"]
    return pools, texts


def embed(year: str, encoder: str):
    pools, texts = load_pool(year)
    emb = Embedder(encoder)
    emb.encode([p["query"] for p in pools], role="query", progress=f"dl{year} {encoder} queries")
    ids = list(texts)
    emb.encode([texts[d] for d in ids], role="doc", progress=f"dl{year} {encoder} passages")
    print(f"dl{year}: {encoder} embeddings cached")


def cross_encode(year: str):
    pools, texts = load_pool(year)
    ce = CrossEncoderCache()
    t0 = time.time()
    for i, p in enumerate(pools):
        ce.score(p["query"], [texts[d] for d in p["docids"]])
        if i % 10 == 0:
            print(f"  dl{year} CE {i + 1}/{len(pools)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"dl{year}: CE logits cached")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", default=["2021", "2022"])
    ap.add_argument("--steps", nargs="+", default=["pool", "embed", "ce"])
    ap.add_argument("--encoder", default="openai", choices=["openai", "bge", "e5"])
    a = ap.parse_args()
    for y in a.years:
        if "pool" in a.steps:
            build_pool(y)
        if "embed" in a.steps:
            embed(y, a.encoder)
        if "ce" in a.steps:
            cross_encode(y)
