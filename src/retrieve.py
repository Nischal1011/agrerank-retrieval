# Retrieve top-k relevant texts for a query using FAISS and OpenAI embeddings

import os
import json
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

# ---------- config ----------
MODEL = "text-embedding-3-large"
CORPUS_PATH = "data/chunks/1-s2.0-S0020025524008260-am.chunks.json"
IDS_PATH = "data/embeddings/corpus_ids.npy"
EMB_PATH = "data/embeddings/corpus_emb.npy"
INDEX_PATH = "data/FAISS_index/index.faiss"
USE_COSINE = True 
# ---------- setup ----------
load_dotenv()
client = OpenAI()

def _load_id_to_text(corpus_path: str) -> dict:
    """Map ids (line_0, line_1, ...) -> text lines. Skips blank lines."""
    if corpus_path.lower().endswith(".txt"):
        with open(corpus_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        return {f"line_{i}": t for i, t in enumerate(lines)}
    # Optional JSON fallback: [{"id": "...", "text": "..."}]
    with open(corpus_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {d["id"]: d["text"] for d in data}

# Load assets
IDS = np.load(IDS_PATH, allow_pickle=True)                 # (N,)
EMB = np.load(EMB_PATH).astype("float32")                  # (N, d)
INDEX = faiss.read_index(INDEX_PATH)
ID_TO_TEXT = _load_id_to_text(CORPUS_PATH)

# Sanity checks
if getattr(INDEX, "ntotal", None) is not None and INDEX.ntotal != EMB.shape[0]:
    raise SystemExit(
        f"Index size ({INDEX.ntotal}) != embeddings count ({EMB.shape[0]}). "
        "Rebuild the index from the same embeddings."
    )

def embed_query(query: str) -> np.ndarray:
    """Return a (1, d) float32 query vector, normalized if using cosine."""
    e = client.embeddings.create(model=MODEL, input=[query]).data[0].embedding
    v = np.array(e, dtype="float32").reshape(1, -1)
    if USE_COSINE:
        faiss.normalize_L2(v)
    return v

def search(query: str, k: int = 5, min_score: float | None = None):
    """
    Return top-k results as [{id, score, text}], optionally filtering by min_score.
    For cosine setup, score is cosine similarity in [-1, 1].
    """
    q = embed_query(query)
    D, I = INDEX.search(q, k)
    results = []
    for rank, idx in enumerate(I[0]):
        if idx < 0:  # FAISS pads with -1 when fewer than k results
            continue
        pid = str(IDS[idx])
        score = float(D[0][rank])
        if (min_score is not None) and (score < min_score):
            continue
        text = ID_TO_TEXT.get(pid, f"[missing text for id {pid}]")
        results.append({"id": pid, "score": score, "text": text})
    return results

def pretty_print(results):
    for r in results:
        preview = (r["text"][:100] + "...") if len(r["text"]) > 100 else r["text"]
        print(f"[{r['id']}] (score={r['score']:.3f}) {preview}")

if __name__ == "__main__":
    # Example usage
    q = "Explain graph fusion embedding"
    hits = search(q, k=5)
    print(f"Query: {q}\n")
    pretty_print(hits)
