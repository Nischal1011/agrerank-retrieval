import argparse
import json
import time
from pathlib import Path
import ir_datasets
import pytrec_eval
import numpy as np
import os
import re
import sys
from typing import Dict, List, Tuple

from pyserini.search.lucene import LuceneSearcher
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from scipy import stats
import torch

MODEL = "text-embedding-3-large"

# --- UTILS ---
def l2n(V: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if V.ndim == 1: return V / max(np.linalg.norm(V), eps)
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), eps)

def percentile_normalize(scores: np.ndarray) -> np.ndarray:
    return stats.rankdata(scores, 'average') / len(scores)

def standard_normalize(scores: np.ndarray) -> np.ndarray:
    z_scores = (scores - np.mean(scores)) / np.std(scores)
    return np.clip(z_scores, -3, 3)

def safe_text(text: any) -> str:
    # Strict string conversion for CrossEncoder
    if text is None: return ""
    return str(text).strip()

class OAI:
    def __init__(self):
        self.c = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    def encode(self, texts: List[str]) -> np.ndarray:
        return np.zeros((len(texts), 3072), dtype=np.float32) # Dummy for cache hit

class AgreRankLOOSystem:
    def __init__(self, args):
        self.args = args
        self.encoder = OAI()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.backends.mps.is_available(): device = "mps" # Mac support
        
        print(f"INFO: Loading CrossEncoder on {device}...", flush=True)
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', device=device)
        self.NEURAL_DEPTH = 400 

    def compute_rrf_score(self, rank1, rank2, k=60):
        return 1.0/(k+rank1) + 1.0/(k+rank2)

    def get_prf_vector(self, q_vec, doc_vecs, ranking, top_k=15):
        limit = min(top_k, len(ranking))
        if limit == 0: return q_vec
        seed_vecs = doc_vecs[ranking[:limit]]
        centroid = np.mean(seed_vecs, axis=0)
        return l2n((0.65 * q_vec + 0.35 * centroid).reshape(1, -1))[0]

    def run_neural_cascade(self, cand_ids, scores, q_text, c_docs):
        idx_sorted = np.argsort(scores)[-self.NEURAL_DEPTH:] 
        
        pairs, valid_indices = [], []
        q_text_str = safe_text(q_text) # Enforce str
        
        for i in idx_sorted:
            did = cand_ids[i]
            dt = safe_text(c_docs.get(did, ""))
            if q_text_str and dt:
                pairs.append([q_text_str, dt])
                valid_indices.append(i)
        
        final_scores = np.copy(scores)
        
        if pairs:
            # NO TRY-EXCEPT BLOCK. LET IT CRASH IF IT FAILS.
            preds = self.cross_encoder.predict(pairs, show_progress_bar=False, batch_size=32)
            
            p_min, p_max = preds.min(), preds.max()
            p_norm = (preds - p_min) / (p_max - p_min + 1e-9)
            s_min, s_max = scores.min(), scores.max()
            p_calib = p_norm * (s_max - s_min) + s_min
            
            for j, original_idx in enumerate(valid_indices):
                final_scores[original_idx] = 0.5 * p_calib[j] + 0.5 * final_scores[original_idx]
        
        return final_scores

    def run_variants(self, precomputed: dict):
        q_vec = precomputed['query_vec'][0]
        d_vecs = precomputed['doc_vecs']
        cand_ids = precomputed['cand_ids']
        
        d_raw = (precomputed['query_vec'] @ d_vecs.T)[0]
        s_raw = precomputed['sparse_scores']
        d_norm = percentile_normalize(d_raw)
        s_norm = percentile_normalize(s_raw)
        d_rank = np.argsort(np.argsort(d_raw)[::-1])
        s_rank = np.argsort(np.argsort(s_raw)[::-1])
        
        rrf = np.array([self.compute_rrf_score(d_rank[i]+1, s_rank[i]+1) for i in range(len(cand_ids))])
        
        # Variant A
        rrf_sorted_idx = np.argsort(rrf)[::-1]
        q_dirty = self.get_prf_vector(q_vec, d_vecs, rrf_sorted_idx)
        scores_dirty = percentile_normalize((q_dirty.reshape(1,-1) @ d_vecs.T)[0])
        final_A = self.run_neural_cascade(cand_ids, scores_dirty, precomputed['query_text'], precomputed['candidate_docs'])
        
        # Consensus Seeds
        agreement_boost = ((d_norm > 0.8) & (s_norm > 0.8)).astype(float) * 0.5
        consensus_signal = standard_normalize(rrf + agreement_boost)
        consensus_sorted_idx = np.argsort(consensus_signal)[::-1]
        q_clean = self.get_prf_vector(q_vec, d_vecs, consensus_sorted_idx)
        scores_clean = percentile_normalize((q_clean.reshape(1,-1) @ d_vecs.T)[0])
        
        # Variant B
        final_B = self.run_neural_cascade(cand_ids, scores_clean, precomputed['query_text'], precomputed['candidate_docs'])
        
        # Variant C
        orig_align = np.maximum((precomputed['query_vec'] @ d_vecs.T)[0], 0.0)
        scores_gated = scores_clean * (0.5 + 0.5 * orig_align)
        final_C = self.run_neural_cascade(cand_ids, scores_gated, precomputed['query_text'], precomputed['candidate_docs'])
        
        def pack(s): return sorted(zip(cand_ids, s), key=lambda x: x[1], reverse=True)
        return {"A_No_Consensus": pack(final_A), "B_No_Gating": pack(final_B), "C_Full_AgreRank": pack(final_C)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="./cache")
    args = parser.parse_args()

    K_CANDIDATES = 500
    
    print(f"--- LOO ABLATION DEBUG ---")
    ds_safe = args.dataset.replace('/', '_')
    CACHE_DIR = Path(args.cache_dir) / f"{ds_safe}_k{K_CANDIDATES}"
    
    dataset = ir_datasets.load(args.dataset)
    qrels = {}
    for q in dataset.qrels_iter(): qrels.setdefault(q.query_id, {})[q.doc_id] = q.relevance
    
    queries = list(dataset.queries_iter())
    system = AgreRankLOOSystem(args)
    runs = {"A_No_Consensus": {}, "B_No_Gating": {}, "C_Full_AgreRank": {}}
    
    print(f"Processing {len(queries)} queries...")
    
    for i, query in enumerate(queries):
        print(f"\rQuery {i+1}", end='', flush=True)
        cache_file = CACHE_DIR / f"{query.query_id}.npz"
        
        if cache_file.exists():
            try:
                data = np.load(cache_file, allow_pickle=True)
                precomputed = {k: data[k] for k in data}
                precomputed['cand_ids'] = precomputed['cand_ids'].tolist()
                precomputed['candidate_docs'] = precomputed['candidate_docs'].item()
                
                results = system.run_variants(precomputed)
                for name, ranking in results.items():
                    runs[name][query.query_id] = {d: float(s) for d, s in ranking}
            except Exception as e:
                print(f"\nERROR on QID {query.query_id}: {e}")
                sys.exit(1)
                
    print(f"\nComputing Metrics...")
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg_cut_10'})
    scores = {}
    for name, run in runs.items():
        res = evaluator.evaluate(run)
        scores[name] = np.mean([v['ndcg_cut_10'] for v in res.values()])
        
    base_score = scores['C_Full_AgreRank']
    print(f"{'VARIANT':<25} | {'nDCG':<8} | {'DROP':<8}")
    print("-" * 45)
    for name in ['C_Full_AgreRank', 'B_No_Gating', 'A_No_Consensus']:
        s = scores[name]
        diff = (s - base_score) / base_score * 100
        print(f"{name:<25} | {s:.4f}   | {diff:+.2f}%")

if __name__ == "__main__":
    main()

# python vectorrag/eval/leave_one_out_abalation.py\
#   --dataset "msmarco-passage-v2/trec-dl-2022/judged"