"""Shared infrastructure: embedding and cross-encoder caches, metrics, statistics.

Caches are SQLite files keyed by a hash of the exact input text, so every run is
resumable and two scripts never disagree about a vector or a score. API errors
raise instead of silently substituting zero vectors.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent
WORK = Path(os.environ.get("AGRERANK_WORK", ROOT / "work"))
EMBED_MODEL = "text-embedding-3-large"
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"


def _key(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


class _Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), timeout=300)
        self.db.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v BLOB)")

    def get_many(self, keys: Sequence[str]) -> Dict[str, bytes]:
        out = {}
        for i in range(0, len(keys), 900):
            chunk = keys[i:i + 900]
            q = "SELECT k, v FROM kv WHERE k IN (%s)" % ",".join("?" * len(chunk))
            out.update(dict(self.db.execute(q, chunk).fetchall()))
        return out

    def put_many(self, items: Iterable[tuple]):
        self.db.executemany("INSERT OR REPLACE INTO kv VALUES (?, ?)", items)
        self.db.commit()


def load_dotenv(path: Path):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$", line)
        if m and m.group(1) not in os.environ:
            os.environ[m.group(1)] = m.group(2).strip().strip('"').strip("'")


# Encoders. "openai" needs OPENAI_API_KEY; the others run locally.
ENCODERS = {
    "openai": {"model": EMBED_MODEL},
    "bge": {"model": "BAAI/bge-large-en-v1.5",
            "query_prefix": "Represent this sentence for searching relevant passages: ", "doc_prefix": ""},
    "e5": {"model": "intfloat/e5-large-v2", "query_prefix": "query: ", "doc_prefix": "passage: "},
}


class Embedder:
    """Query/passage encoder with a persistent cache. Returns L2-normalized float32.

    encoder="openai" calls text-embedding-3-large (vectors can also be imported
    from the paper's cache, see import_legacy_embeddings.py); "bge" and "e5" run
    locally with sentence-transformers.
    """

    def __init__(self, encoder: str = "openai", batch_size: int = 256):
        self.encoder = encoder
        self.spec = ENCODERS[encoder]
        name = "embeddings.sqlite" if encoder == "openai" else f"embeddings_{encoder}.sqlite"
        self.store = _Store(WORK / "cache" / name)
        self.bs = batch_size
        self._client = None

    def key(self, text: str, role: str) -> str:
        if self.encoder == "openai":  # OpenAI embeddings are symmetric: no role in the key
            return _key(EMBED_MODEL, text)
        return _key(self.spec["model"], role, text)

    def _api(self, texts: List[str], role: str) -> np.ndarray:
        if self.encoder != "openai":
            if self._client is None:
                import torch
                from sentence_transformers import SentenceTransformer
                dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
                self._client = SentenceTransformer(self.spec["model"], device=dev)
                self._client.max_seq_length = 512
            pre = self.spec["query_prefix" if role == "query" else "doc_prefix"]
            return self._client.encode([pre + t for t in texts], batch_size=64, convert_to_numpy=True,
                                       normalize_embeddings=True).astype(np.float32)
        if self._client is None:
            load_dotenv(Path(os.environ.get("AGRERANK_DOTENV", ROOT.parent / ".env")))
            from openai import OpenAI
            self._client = OpenAI()
        for attempt in range(8):
            try:
                resp = self._client.embeddings.create(model=EMBED_MODEL, input=texts)
                return np.array([d.embedding for d in resp.data], dtype=np.float32)
            except Exception as e:  # rate limits, transient network errors
                if attempt == 7 or "insufficient_quota" in str(e) or "no credits" in str(e):
                    raise
                wait = min(60, 2 ** attempt)
                print(f"  embedding retry in {wait}s: {type(e).__name__}: {str(e)[:120]}", flush=True)
                time.sleep(wait)

    def encode(self, texts: Sequence[str], role: str = "doc", progress: str = "") -> np.ndarray:
        texts = [t if t and t.strip() else " " for t in texts]
        keys = [self.key(t, role) for t in texts]
        have = self.store.get_many(list(dict.fromkeys(keys)))
        missing = list(dict.fromkeys(k for k in keys if k not in have))
        if missing:
            by_key = {k: t for k, t in zip(keys, texts)}
            for i in range(0, len(missing), self.bs):
                ks = missing[i:i + self.bs]
                vecs = self._api([by_key[k] for k in ks], role)
                self.store.put_many((k, v.tobytes()) for k, v in zip(ks, vecs))
                have.update({k: v.tobytes() for k, v in zip(ks, vecs)})
                if progress and (i // self.bs) % 20 == 0:
                    print(f"  {progress}: embedded {min(i + self.bs, len(missing))}/{len(missing)}", flush=True)
        V = np.stack([np.frombuffer(have[k], dtype=np.float32) for k in keys])
        return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)


class CrossEncoderCache:
    """Raw logits of the MS MARCO MiniLM-L-12 cross-encoder, cached per (query, passage)."""

    def __init__(self, model_name: str = CE_MODEL, batch_size: int = 64):
        self.model_name = model_name
        self.store = _Store(WORK / "cache" / ("ce_" + model_name.replace("/", "_") + ".sqlite"))
        self.bs = batch_size
        self._model = None

    def _load(self):
        if self._model is None:
            import torch
            from sentence_transformers import CrossEncoder
            dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = CrossEncoder(self.model_name, max_length=512, device=dev)
        return self._model

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        keys = [_key(self.model_name, query, p) for p in passages]
        have = self.store.get_many(list(dict.fromkeys(keys)))
        missing = [i for i, k in enumerate(keys) if k not in have]
        if missing:
            import torch
            m = self._load()
            pairs = [[query, passages[i]] for i in missing]
            logits = m.predict(pairs, batch_size=self.bs, activation_fn=torch.nn.Identity(),
                               show_progress_bar=False, convert_to_numpy=True).astype(np.float64)
            self.store.put_many((keys[i], float(s)) for i, s in zip(missing, logits))
            for i, s in zip(missing, logits):
                have[keys[i]] = float(s)
        return np.array([have[k] for k in keys], dtype=np.float64)


# ----------------------------- evaluation -----------------------------------

def write_run(path: Path, run: Dict[str, Dict[str, float]], tag: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for qid, docs in run.items():
            ranked = sorted(docs.items(), key=lambda x: (-x[1], x[0]))
            for r, (d, s) in enumerate(ranked, 1):
                f.write(f"{qid} Q0 {d} {r} {s:.8f} {tag}\n")


def to_rank_scores(run: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Replace scores by reciprocal positions so pytrec_eval's tie handling cannot matter."""
    out = {}
    for qid, docs in run.items():
        ranked = sorted(docs.items(), key=lambda x: (-x[1], x[0]))
        out[qid] = {d: float(len(ranked) - i) for i, (d, _) in enumerate(ranked)}
    return out


def trec_metrics(qrels, run) -> Dict[str, Dict[str, float]]:
    """Per-query metrics. nDCG@10 uses graded labels; MAP/MRR are reported with
    label>=1 (pytrec_eval default, as in the paper's code) and label>=2 (TREC DL
    passage convention)."""
    import pytrec_eval
    e1 = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10", "map", "recip_rank", "recall_1000"})
    e2 = pytrec_eval.RelevanceEvaluator(qrels, {"map", "recip_rank"}, relevance_level=2)
    r1, r2 = e1.evaluate(run), e2.evaluate(run)
    return {q: {"ndcg10": r1[q]["ndcg_cut_10"], "map": r1[q]["map"], "mrr": r1[q]["recip_rank"],
                "recall1000": r1[q]["recall_1000"], "map_rel2": r2[q]["map"], "mrr_rel2": r2[q]["recip_rank"]}
            for q in r1}


def mean_metrics(per_q: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    keys = next(iter(per_q.values())).keys()
    return {k: float(np.mean([v[k] for v in per_q.values()])) for k in keys}


def paired_tests(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    from scipy import stats
    qs = sorted(set(a) & set(b))
    x, y = np.array([a[q] for q in qs]), np.array([b[q] for q in qs])
    d = x - y
    t = stats.ttest_rel(x, y)
    try:
        w = stats.wilcoxon(x, y).pvalue
    except ValueError:
        w = 1.0
    rng = np.random.default_rng(0)
    boots = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(1)
    return {"n": len(qs), "mean_a": float(x.mean()), "mean_b": float(y.mean()), "diff": float(d.mean()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "t_p": float(t.pvalue), "wilcoxon_p": float(w),
            "wins": int((d > 1e-9).sum()), "ties": int((np.abs(d) <= 1e-9).sum()), "losses": int((d < -1e-9).sum())}


def dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
