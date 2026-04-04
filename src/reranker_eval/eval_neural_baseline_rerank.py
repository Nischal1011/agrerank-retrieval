import argparse
import json
import time
from pathlib import Path
import numpy as np
import sys
import re
from typing import List, Dict, Tuple, Any

# --- Dependencies ---
try: from rank_bm25 import BM25Okapi
except ImportError: sys.exit("pip install rank_bm25")
try: from sentence_transformers import CrossEncoder
except ImportError: sys.exit("pip install sentence-transformers")

# --- Configuration ---
# MUST match AgreRank's model to be "fair"
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2" 

# --- Utilities ---
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)*")
STOP = set("the a an and or for to of in on at by from with as is are was were be been being this that these those which who whose where when".split())

def toks(s: str) -> List[str]:
    if not s: return []
    out = [w.lower() for w in _TOKEN_RE.findall(s)]
    return [w for w in out if w not in STOP and len(w) > 1]

def safe_text(text: Any) -> str:
    try: return str(text or "").strip()
    except: return ""

def load_master(path: Path) -> Tuple[List[str], Dict[str, str]]:
    ids, txt = [], {}
    print(f"Loading corpus from {path}...")
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            o = json.loads(line)
            cid = o.get("chunk_id") or o.get("id")
            ids.append(cid)
            txt[cid] = o["text"]
    return ids, txt

def load_qset(path: Path) -> List[dict]:
    return json.loads(path.read_text(encoding="utf-8"))

class SparseNeuralModel:
    def __init__(self, args):
        self.args = args
        self.chunk_ids, self.chunk_text = load_master(args.chunks)
        
        print("Building BM25 index...")
        tokenized_corpus = [toks(text) for text in self.chunk_text.values()]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        print(f"Loading Cross-Encoder: {CE_MODEL}...")
        self.reranker = CrossEncoder(CE_MODEL)

    def search(self, query_text: str) -> List[Tuple[str, float]]:
        # 1. Sparse Search Only (BM25)
        s_scores_all = self.bm25.get_scores(toks(query_text))
        
        # Get Top-K candidates
        top_indices = np.argsort(s_scores_all)[::-1][:self.args.candidates]
        
        # Filter 0 scores
        candidates = [self.chunk_ids[i] for i in top_indices if s_scores_all[i] > 0]
        bm25_scores = [s_scores_all[i] for i in top_indices if s_scores_all[i] > 0]
        
        if not candidates: return []

        # 2. Neural Reranking
        rerank_depth = min(len(candidates), self.args.cross_encoder_k)
        
        pairs = []
        valid_indices = []
        q_clean = safe_text(query_text)
        
        for i in range(rerank_depth):
            cid = candidates[i]
            txt = safe_text(self.chunk_text.get(cid, ""))
            if q_clean and txt:
                pairs.append([q_clean, txt])
                valid_indices.append(i)
        
        final_scores = np.array(bm25_scores, dtype=float)
        
        if pairs:
            ce_scores = self.reranker.predict(pairs, show_progress_bar=False)
            
            # Normalize to force CE results to the top
            ce_min = ce_scores.min()
            # If min == max, denom = 1
            denom = (ce_scores.max() - ce_min) + 1e-9
            ce_norm = (ce_scores - ce_min) / denom
            
            for i, list_idx in enumerate(valid_indices):
                # Overwrite BM25 score with boosted CE score (Base 1000 + 0-1 score)
                final_scores[list_idx] = 1000.0 + ce_norm[i]
                
        return sorted(zip(candidates, final_scores), key=lambda x: x[1], reverse=True)

def eval_system(search_fn, qset: List[dict], top_k: int) -> Dict[str, float]:
    single, multi, all_f1 = [], [], []
    start = time.perf_counter()
    
    for i, q in enumerate(qset):
        print(f"Processing {i+1}/{len(qset)}...", file=sys.stderr, end='\r')
        res = search_fn(q["query_text"])
        toplist = [cid for cid, _ in res]
        rel_set = set(q["relevant_chunks"])
        if not rel_set: continue
        
        # F1 Calculation
        top_k_list = toplist[:top_k]
        tp = len(rel_set.intersection(set(top_k_list)))
        prec = tp / len(top_k_list) if top_k_list else 0.0
        rec = tp / len(rel_set)
        # Avoid div/0
        if (prec + rec) > 0:
            f1 = 2 * (prec * rec) / (prec + rec)
        else:
            f1 = 0.0
        all_f1.append(f1)
        
        # Ranks Calculation
        ranks = {cid: idx + 1 for idx, cid in enumerate(toplist)}
        # If chunk not found, assign rank 1,000,000
        r_list = [ranks.get(cid, 1000000) for cid in rel_set]
        
        if q.get("query_type", "single_chunk").startswith("single"):
            single.append(r_list)
        else:
            # For multi, we need the total count of relevant chunks
            multi.append((r_list, len(rel_set)))
            
    total_time = time.perf_counter() - start
    out = {}
    
    # --- Single Answer Metrics ---
    if single:
        # For MRR, we only care if the answer is in top 10
        # Flatten all ranks found for single queries
        v_ranks = [r for rl in single for r in rl if r <= 10]
        if v_ranks:
            out["single_MRR@10"] = float(np.mean([1.0/r for r in v_ranks]))
        else:
            out["single_MRR@10"] = 0.0
            
        out["single_R@1"] = float(np.mean([any(r<=1 for r in rl) for rl in single]))
        out["single_R@5"] = float(np.mean([any(r<=5 for r in rl) for rl in single]))
        out["single_R@10"] = float(np.mean([any(r<=10 for r in rl) for rl in single]))
    
    # --- Multi Answer Metrics ---
    if multi:
        # Coverage: What % of relevant docs did we find in top K?
        out["multi_Cov@5"] = float(np.mean([sum(1 for r in rl if r<=5)/n for rl, n in multi]))
        out["multi_Cov@10"] = float(np.mean([sum(1 for r in rl if r<=10)/n for rl, n in multi]))
        # Max Recall: Did we find ANY relevant doc in top K?
        out["multi_MaxR@5"] = float(np.mean([any(r<=5 for r in rl) for rl, n in multi]))
        out["multi_MaxR@10"] = float(np.mean([any(r<=10 for r in rl) for rl, n in multi]))
        
    out["overall_F1@10"] = float(np.mean(all_f1)) if all_f1 else 0.0
    out["total_time_s"] = total_time
    out["avg_time_ms"] = (total_time / len(qset)) * 1000 if qset else 0.0
    
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True) # Unused but kept for compatibility
    parser.add_argument("--qset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=200)
    parser.add_argument("--cross_encoder_k", type=int, default=200) 
    parser.add_argument("--topk", type=int, default=10) # For metric calculation
    args = parser.parse_args()
    
    print("--- Running Sparse-Neural Baseline (BM25 -> CE) ---")
    print(f"Candidates: {args.candidates}, Rerank Depth: {args.cross_encoder_k}")
    
    model = SparseNeuralModel(args)
    metrics = eval_system(model.search, load_qset(args.qset), args.topk)
    
    # Save flat JSON as requested
    with args.out.open("w") as f:
        json.dump(metrics, f, indent=2)
    
    print("\nResults:")
    for k,v in metrics.items(): print(f"{k}: {v}")
    print(f"\n✅ Saved to {args.out}")

if __name__ == "__main__":
    main()

    
# python vectorrag/eval/retrievers/eval_neural_baseline_rerank.py \
#     --chunks data/master_in_embedding_order.jsonl \
#     --embeddings data/embeddings/fixed250/corpus_emb.npy \
#     --qset data/qqset.json \
#     --out data/results_with_timing/results_neural_rank.json \
#     --candidates 200 \
#     --cross_encoder_k 200