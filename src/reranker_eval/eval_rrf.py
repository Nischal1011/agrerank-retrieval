from __future__ import annotations
import argparse, json, sys, re
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import time

# --- Dependencies ---
try:
    import faiss
except Exception:
    print("pip install faiss-cpu", file=sys.stderr); raise
try:
    from openai import OpenAI
except Exception:
    print("pip install openai  # >=1.0", file=sys.stderr); raise
## NEW: Added rank_bm25 dependency
try:
    from rank_bm25 import BM25Okapi
except Exception:
    print("pip install rank_bm25", file=sys.stderr); raise

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

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)*")
STOP = set("the a an and or for to of in on at by from with as is are was were be been being this that these those which who whose where when".split())
def tokenize(text: str)->List[str]:
    toks = [t.lower() for t in _TOKEN_RE.findall(text or "")]
    return [t for t in toks if t not in STOP and len(t) > 1]

class OAI:
    def __init__(self, bs:int=128):
        self.c=OpenAI(); self.bs=bs
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
    ap.add_argument("--candidates", type=int, default=100)
    ap.add_argument("--rrf_k", type=int, default=60)
    args = ap.parse_args()

    # --- Global Setup ---
    chunk_ids, chunk_text = load_master(args.chunks)
    E = np.load(args.embeddings).astype(np.float32)
    if not np.allclose(np.linalg.norm(E, axis=1).mean(), 1.0, atol=1e-2): E = l2n(E)
    
    # Setup for Dense Search
    d = E.shape[1]
    index = faiss.IndexFlatIP(d); index.add(E)
    id2chunk = {i: cid for i, cid in enumerate(chunk_ids)}
    
    ## Setup for BM25 Sparse Search ##
    print("Building BM25 index...")
    # Tokenize the entire corpus to build the index
    tokenized_corpus = [tokenize(chunk_text[cid]) for cid in chunk_ids]
    bm25 = BM25Okapi(tokenized_corpus)
    # Create a mapping from the BM25's internal index (0, 1, 2...) back to our chunk_ids
    bm25_idx_to_cid = {i: cid for i, cid in enumerate(chunk_ids)}
    
    enc = OAI()

    ## The search function now uses Dense + BM25 ##
    def rrf_search_bm25(qtext: str):
        # --- 1. Get Top Candidates from Dense Search ---
        qv_doc = enc.encode([qtext]).astype(np.float32)
        d_scores, d_ids = index.search(qv_doc, args.candidates)
        dense_cids_ranked = [id2chunk[i] for i in d_ids[0] if i != -1]
        dense_ranks = {cid: r + 1 for r, cid in enumerate(dense_cids_ranked)}
        
        # --- 2. Get Top Candidates from BM25 Search ---
        tokenized_query = tokenize(qtext)
        # Get scores for all documents, which is efficient
        bm25_scores_all = bm25.get_scores(tokenized_query)
        # Get the indices of the top N candidates
        top_sparse_indices = np.argsort(bm25_scores_all)[::-1][:args.candidates]
        sparse_cids_ranked = [bm25_idx_to_cid[i] for i in top_sparse_indices if bm25_scores_all[i] > 0]
        sparse_ranks = {cid: r + 1 for r, cid in enumerate(sparse_cids_ranked)}
        
        # --- 3. Fuse the results using RRF ---
        # Get the union of all candidates from both lists
        all_candidate_cids = set(dense_cids_ranked) | set(sparse_cids_ranked)
        if not all_candidate_cids: return []

        k = float(args.rrf_k)
        final_scores = []
        for cid in all_candidate_cids:
            # Get the rank from each list, providing a default large number if not present
            r_dense = dense_ranks.get(cid, args.candidates * 2)
            r_sparse = sparse_ranks.get(cid, args.candidates * 2)
            # Calculate the RRF score
            score = (1 / (k + r_dense)) + (1 / (k + r_sparse))
            final_scores.append((cid, score, 0.0))
            
        final_scores.sort(key=lambda x: x[1], reverse=True)
        return final_scores[:args.topk]

    # --- Evaluation ---
    print("--- Evaluating method: RRF (Dense + BM25) ---")
    metrics = eval_system(rrf_search_bm25, load_qset(args.qset), args.topk)
    
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ RRF (Dense + BM25) results saved to {args.out}")

if __name__ == "__main__":
    main()



# python vectorrag/eval/retrievers/eval_rrf.py \
#   --chunks data/master_in_embedding_order.jsonl \
#   --embeddings data/embeddings/fixed250/corpus_emb.npy \
#   --qset data/qset.json \
#   --out data/results/results_rrf.json \
#   --candidates 100 \
#   --rrf_k 60


# python vectorrag/eval/retrievers/eval_rrf.py \
#     --chunks data/master_in_embedding_order.jsonl \
#    --embeddings data/embeddings/fixed250/corpus_emb.npy \
#     --qset data/qqset.json \
#      --out data/results_with_timing/results_rrf.json \
#     --candidates 100 \
#     --topk 10
