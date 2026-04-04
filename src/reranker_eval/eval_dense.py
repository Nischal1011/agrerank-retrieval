#!/usr/bin/env python3
# eval_scripts/eval_baseline.py
from __future__ import annotations
import time
import argparse, json, sys, re
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

try:
    import faiss
except Exception:
    print("pip install faiss-cpu", file=sys.stderr); raise
try:
    from openai import OpenAI
except Exception:
    print("pip install openai  # >=1.0", file=sys.stderr); raise
try:
    import tiktoken
except Exception:
    tiktoken = None
MODEL = "text-embedding-3-large"

# ----------------- load helpers -----------------
def load_master(path: Path):
    if path.suffix.lower()==".jsonl":
        ids, txt = [], {}
        with path.open(encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line: continue
                o=json.loads(line)
                cid = o.get("chunk_id") or o.get("id")
                ids.append(cid); txt[cid]=o["text"]
        return ids, txt
    else:
        arr=json.loads(path.read_text(encoding="utf-8"))
        ids, txt = [], {}
        for o in arr:
            cid = o.get("chunk_id") or o.get("id")
            ids.append(cid); txt[cid]=o["text"]
        return ids, txt

def load_qset(path: Path)->List[dict]:
    return json.loads(path.read_text(encoding="utf-8"))

# ----------------- utils -----------------
def l2n(V: np.ndarray, eps: float=1e-12)->np.ndarray:
    n=np.linalg.norm(V,axis=1,keepdims=True); n=np.maximum(n,eps); return V/n

class OAI:
    def __init__(self, bs:int=128, max_tokens:int=8192):
        self.c=OpenAI(); self.bs=bs; self.max_tokens=max_tokens
    def encode(self, texts: List[str])->np.ndarray:
        if not texts: return np.zeros((0,0),dtype=np.float32)
        out=[]
        for i in range(0,len(texts),self.bs):
            batch=texts[i:i+self.bs]
            resp=self.c.embeddings.create(model=MODEL, input=batch)
            out.append(np.array([d.embedding for d in resp.data], dtype=np.float32))
        V=np.vstack(out); return l2n(V).astype(np.float32)

# ----------------- metrics -----------------
def recall_at_k(ranks: List[int], k: int) -> float: return float(any(r <= k for r in ranks))
def mrr_at_k(ranks: List[int], k: int) -> float:
    v = [r for r in ranks if r <= k]; return 1.0 / v[0] if v else 0.0
def coverage_at_k(ranks: List[int], total: int, k: int) -> float:
    return min(len([r for r in ranks if r <= k]), total) / max(total, 1)
def ranks_for(toplist: List[str], relevant: List[str]) -> List[int]:
    pos = {cid: i + 1 for i, cid in enumerate(toplist)}; BIG = 10 ** 9
    return [pos.get(cid, BIG) for cid in relevant]

def eval_system(search_fn, qset: List[dict], top_k: int) -> Dict[str, float]:
    single, multi = [], []
    all_f1_scores = []

    # --- Timing Start ---
    start_time = time.perf_counter()

    for i, q in enumerate(qset):
        print(f"Processing query {i+1}/{len(qset)}...", file=sys.stderr)
        res = search_fn(q["query_text"])
        toplist = [cid for cid, _, _ in res][:top_k]
        
        relevant_chunks_set = set(q["relevant_chunks"])
        if not relevant_chunks_set:
            continue

        # --- Per-Query F1 Score Calculation ---
        toplist_set = set(toplist)
        true_positives = len(relevant_chunks_set.intersection(toplist_set))
        
        precision_at_k = true_positives / len(toplist) if toplist else 0.0
        recall_at_k_for_f1 = true_positives / len(relevant_chunks_set)
        
        if (precision_at_k + recall_at_k_for_f1) == 0:
            f1_at_k = 0.0
        else:
            f1_at_k = 2 * (precision_at_k * recall_at_k_for_f1) / (precision_at_k + recall_at_k_for_f1)
        
        all_f1_scores.append(f1_at_k)
        # --- End of F1 Calculation ---

        r = ranks_for(toplist, q["relevant_chunks"])
        if q.get("query_type", "single_chunk").startswith("single"):
            single.append(r)
        else:
            multi.append((r, len(q["relevant_chunks"])))

    # --- Timing End ---
    end_time = time.perf_counter()
    total_time_seconds = end_time - start_time
    num_queries = len(qset)
    avg_time_ms = (total_time_seconds / num_queries) * 1000 if num_queries > 0 else 0.0
            
    out = {}
    if single:
        out["single_R@1"] = float(np.mean([recall_at_k(r,1) for r in single])) if single else 0.0
        out["single_R@5"] = float(np.mean([recall_at_k(r,5) for r in single])) if single else 0.0
        out["single_R@10"] = float(np.mean([recall_at_k(r,10) for r in single])) if single else 0.0
        out["single_MRR@10"] = float(np.mean([mrr_at_k(r,10) for r in single])) if single else 0.0
    if multi:
        out["multi_MaxR@5"] = float(np.mean([recall_at_k(r,5) for r,_ in multi])) if multi else 0.0
        out["multi_MaxR@10"] = float(np.mean([recall_at_k(r,10) for r,_ in multi])) if multi else 0.0
        out["multi_Cov@5"] = float(np.mean([coverage_at_k(r,n,5) for r,n in multi])) if multi else 0.0
        out["multi_Cov@10"] = float(np.mean([coverage_at_k(r,n,10) for r,n in multi])) if multi else 0.0
        
    out["overall_F1@10"] = float(np.mean(all_f1_scores)) if all_f1_scores else 0.0
    
    # Add new timing metrics to the output dictionary
    out["total_time_s"] = total_time_seconds
    out["avg_time_ms"] = avg_time_ms
    
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=Path, required=True)
    ap.add_argument("--embeddings", type=Path, required=True)
    ap.add_argument("--qset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="Output JSON file for results")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    # --- Global Setup ---
    E = np.load(args.embeddings).astype(np.float32)
    if not np.allclose(np.linalg.norm(E, axis=1).mean(), 1.0, atol=1e-2): E = l2n(E)
    d = E.shape[1]
    index = faiss.IndexFlatIP(d); index.add(E)
    
    chunk_ids, _ = load_master(args.chunks)
    id2chunk = {i: cid for i, cid in enumerate(chunk_ids)}
    enc = OAI()

    def baseline_search(qtext: str):
        qv = enc.encode([qtext]).astype(np.float32)
        scores, ids = index.search(qv, args.topk)
        pairs = sorted([(id2chunk[i], s) for i, s in zip(ids[0], scores[0]) if i != -1], key=lambda x:x[1], reverse=True)
        return [(cid, s, 0.0) for cid, s in pairs]

    # --- Evaluation ---
    print("--- Evaluating method: baseline ---")
    metrics = eval_system(baseline_search, load_qset(args.qset), args.topk)
    
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Baseline results saved to {args.out}")

if __name__ == "__main__":
    main()



# python vectorrag/eval/retrievers/eval_baseline.py \
#   --chunks data/master_in_embedding_order.jsonl \
#   --embeddings data/embeddings/fixed250/corpus_emb.npy \
#   --qset data/qqset.json \
#   --out data/results_with_timing/results_baseline.json
