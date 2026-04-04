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

# Core Dependencies
from pyserini.search.lucene import LuceneSearcher
from openai import OpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from scipy import stats

# =======================================================================================
# UTILITY FUNCTIONS
# =======================================================================================
MODEL = "text-embedding-3-large"

def l2n(V: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if V.ndim == 1:
        norm = np.linalg.norm(V)
        return V / max(norm, eps)
    norm = np.linalg.norm(V, axis=1, keepdims=True)
    norm = np.maximum(norm, eps)
    return V / norm

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)*")
STOP = set("the a an and or for to of in on at by from with as is are was were be being this that these those which who whose where when".split())

def tokenize(text: str) -> List[str]:
    try:
        toks = [t.lower() for t in _TOKEN_RE.findall(text or "")]
        return [t for t in toks if t not in STOP and len(t) > 1]
    except:
        return []

def percentile_normalize(scores: np.ndarray) -> np.ndarray:
    if len(scores) < 2: return np.zeros_like(scores, dtype=np.float32)
    return stats.rankdata(scores, 'average') / len(scores)

def standard_normalize(scores: np.ndarray) -> np.ndarray:
    if len(scores) < 2: return np.zeros_like(scores, dtype=float)
    mean, std = np.mean(scores), np.std(scores)
    if std == 0: return np.zeros_like(scores)
    z_scores = (scores - mean) / std
    return np.clip(z_scores, -3, 3)

def safe_text(text: any) -> str:
    try: return str(text or "").strip()
    except: return ""

class RobustOAIEncoder:
    def __init__(self, bs: int = 64):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key: raise ValueError("OPENAI_API_KEY environment variable not set")
        self.c = OpenAI(api_key=api_key); self.bs = bs
    
    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts: return np.zeros((0, 3072), dtype=np.float32)
        out = []
        for i in range(0, len(texts), self.bs):
            batch = [safe_text(t) if t else " " for t in texts[i:i + self.bs]]
            try:
                resp = self.c.embeddings.create(model=MODEL, input=batch)
                out.append(np.array([d.embedding for d in resp.data], dtype=np.float32))
            except Exception as e:
                print(f"\n⚠️  OpenAI API error (batch {i//self.bs}): {e}", file=sys.stderr)
                out.append(np.zeros((len(batch), 3072), dtype=np.float32))
        return l2n(np.vstack(out)).astype(np.float32)

# =======================================================================================
# ABLATION SYSTEM
# =======================================================================================
class AgreRankAblationSystem:
    def __init__(self, args):
        self.args = args; self.encoder = RobustOAIEncoder()
        if args.ablation_mode == 'agrerank_full':
            print("  Loading CrossEncoder model (L-12)...", flush=True)
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
    
    def compute_rrf_score(self, rank1: int, rank2: int, k: float = 60.0) -> float:
        return 1.0 / (k + rank1) + 1.0 / (k + rank2)

    def rerank(self, precomputed: dict) -> List[Tuple[str, float]]:
        query_vec = precomputed['query_vec'][0]; doc_vecs = precomputed['doc_vecs']; cand_ids = precomputed['cand_ids']
        
        # --- Base Retrieval Scores ---
        # Ensure shapes for broadcasting
        q_vec_flat = query_vec.reshape(-1)
        q_vec_2d = q_vec_flat.reshape(1, -1)
        
        dense_scores_raw = (q_vec_2d @ doc_vecs.T)[0]
        sparse_scores_raw = precomputed['sparse_scores']
        
        dense_scores = percentile_normalize(dense_scores_raw)
        sparse_scores = percentile_normalize(sparse_scores_raw)
        
        dense_ranks = np.argsort(np.argsort(dense_scores_raw)[::-1])
        sparse_ranks = np.argsort(np.argsort(sparse_scores_raw)[::-1])
        
        # --- Stage 0: RRF (Baseline) ---
        rrf_scores = np.array([
            self.compute_rrf_score(dense_ranks[i] + 1, sparse_ranks[i] + 1) for i in range(len(cand_ids))
        ])
        if self.args.ablation_mode == 'rrf_baseline': 
            return sorted(zip(cand_ids, rrf_scores), key=lambda x: x[1], reverse=True)
        
        # --- Stage 1: Agreement Boost ---
        agreement_boost = ((dense_scores > self.args.agreement_threshold) & (sparse_scores > self.args.agreement_threshold)).astype(float) * 0.5
        stage1_scores = standard_normalize(rrf_scores + agreement_boost)
        if self.args.ablation_mode == 'stage1_boost': 
            return sorted(zip(cand_ids, stage1_scores), key=lambda x: x[1], reverse=True)
        
        # --- Stage 2: PRF Expansion ---
        initial_ranking = np.argsort(stage1_scores)[::-1]
        top_k_prf = min(15, len(cand_ids))
        top_doc_vecs = doc_vecs[initial_ranking[:top_k_prf]]
        
        expanded_query_vec = l2n((self.args.prf_alpha * query_vec + (1 - self.args.prf_alpha) * np.mean(top_doc_vecs, axis=0)))
        expanded_dense_scores_raw = (expanded_query_vec.reshape(1, -1) @ doc_vecs.T)[0]
        expanded_dense_scores = percentile_normalize(expanded_dense_scores_raw)
        
        if self.args.ablation_mode == 'stage2_prf': 
            return sorted(zip(cand_ids, expanded_dense_scores), key=lambda x: x[1], reverse=True)

        # --- Stage 3: Multiplicative Drift Gating (Strategy P) ---
        # This is the 0.6020 SOTA Logic.
        # Formula: PRF * (0.5 + 0.5 * Original_Alignment)
        
        # 1. Calculate Alignment with ORIGINAL Query (Geometric Consistency)
        dot_orig = (q_vec_2d @ doc_vecs.T)[0]
        
        # 2. Calculate Alignment Factor (0.0 = Orthogonal, 1.0 = Parallel)
        orig_alignment = np.maximum(dot_orig, 0.0)
        
        # 3. Apply Soft Gate (Scale invariant)
        intermediate_scores = expanded_dense_scores * (0.5 + 0.5 * orig_alignment)
        
        if self.args.ablation_mode == 'stage3_features': 
            return sorted(zip(cand_ids, intermediate_scores), key=lambda x: x[1], reverse=True)
        
        # --- Stage 4: Cross-Encoder (Full) ---
        final_scores = np.copy(intermediate_scores)
        
        if self.args.ablation_mode == 'agrerank_full':
            ce_pool_k = min(self.args.cross_encoder_k * 2, len(cand_ids)) 
            
            ce_candidate_indices = np.argsort(final_scores)[-ce_pool_k:]
            query_text = safe_text(precomputed['query_text'])
            
            ce_pairs, id_to_idx, valid_indices = [], {}, []
            for idx in ce_candidate_indices:
                doc_id = cand_ids[idx]
                doc_text = safe_text(precomputed['candidate_docs'].get(doc_id, ""))
                if query_text and doc_text:
                    ce_pairs.append([query_text, doc_text])
                    id_to_idx[doc_id] = idx
                    valid_indices.append(idx)
            
            if ce_pairs:
                try:
                    ce_scores_raw = self.cross_encoder.predict(ce_pairs, show_progress_bar=False)
                    
                    ce_min, ce_max = ce_scores_raw.min(), ce_scores_raw.max()
                    target_min, target_max = final_scores.min(), final_scores.max()
                    
                    denom = (ce_max - ce_min) if (ce_max - ce_min) > 1e-9 else 1.0
                    ce_scores_norm = (ce_scores_raw - ce_min) / denom
                    ce_scores_calibrated = ce_scores_norm * (target_max - target_min) + target_min
                    
                    # 0.5 / 0.5 Fusion
                    ce_weight = 0.5
                    
                    ce_doc_ids = [cand_ids[idx] for idx in valid_indices]
                    for i, doc_id in enumerate(ce_doc_ids):
                        orig_idx = id_to_idx[doc_id]
                        final_scores[orig_idx] = ce_weight * ce_scores_calibrated[i] + (1 - ce_weight) * final_scores[orig_idx]
                        
                except Exception as e: print(f"\n⚠️  CrossEncoder failed: {e}. Skipping.", file=sys.stderr)
                
        return sorted(zip(cand_ids, final_scores), key=lambda x: x[1], reverse=True)

def main():
    parser = argparse.ArgumentParser(description="AgreRank Ablation")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ablation_mode", type=str, required=True, 
                        choices=["rrf_baseline", "stage1_boost", "stage2_prf", "stage3_features", "agrerank_full"])
    parser.add_argument("--candidates_k", type=int, default=500)
    parser.add_argument("--cross_encoder_k", type=int, default=100)
    
    # Hyperparameters
    parser.add_argument("--prf_alpha", type=float, default=0.65)
    parser.add_argument("--agreement_threshold", type=float, default=0.8)
    
    parser.add_argument("--limit_queries", type=int, default=None)
    parser.add_argument("--cache_dir", type=str, default="./cache")
    args = parser.parse_args()
    
    dataset_name_safe = args.dataset.replace('/', '_')
    CACHE_DIR = Path(args.cache_dir) / f"{dataset_name_safe}_k{args.candidates_k}"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Running Mode: {args.ablation_mode} on {args.dataset}")
    
    try: dataset = ir_datasets.load(args.dataset)
    except: sys.exit(f"❌ Failed to load {args.dataset}")
    
    qrels = {}
    for qrel in dataset.qrels_iter():
        qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance
        
    pyserini_index = 'msmarco-v2-passage' if 'v2' in args.dataset else 'msmarco-v1-passage'
    try: bm25_searcher = LuceneSearcher.from_prebuilt_index(pyserini_index); docs_store = dataset.docs_store()
    except: sys.exit(f"❌ Pyserini index load failed: {pyserini_index}")
    
    queries = list(dataset.queries_iter())
    if args.limit_queries: queries = queries[:args.limit_queries]
    
    system = AgreRankAblationSystem(args)
    run_results = {}
    
    start = time.perf_counter()
    for i, query in enumerate(queries):
        print(f"\rQuery {i+1}/{len(queries)}", end='', flush=True)
        cache_file = CACHE_DIR / f"{query.query_id}.npz"
        
        try:
            if cache_file.exists():
                data = np.load(cache_file, allow_pickle=True)
                precomputed = {k: data[k] for k in data}
                precomputed['cand_ids'] = precomputed['cand_ids'].tolist()
                precomputed['candidate_docs'] = precomputed['candidate_docs'].item()
            else:
                hits = bm25_searcher.search(query.text, k=args.candidates_k)
                cand_ids = [h.docid for h in hits]
                if not cand_ids: continue
                docs_dict = docs_store.get_many(cand_ids)
                cand_texts = [safe_text(getattr(docs_dict.get(cid), 'text', '')) for cid in cand_ids]
                bm25 = BM25Okapi([tokenize(t) for t in cand_texts])
                sparse_scores = bm25.get_scores(tokenize(query.text))
                q_vec = system.encoder.encode([query.text])
                d_vecs = system.encoder.encode(cand_texts)
                precomputed = {
                    'cand_ids': cand_ids,
                    'candidate_docs': dict(zip(cand_ids, cand_texts)),
                    'query_text': query.text,
                    'query_vec': q_vec,
                    'doc_vecs': d_vecs,
                    'sparse_scores': sparse_scores
                }
                np.savez_compressed(cache_file, **precomputed)
            
            ranked = system.rerank(precomputed)
            run_results[query.query_id] = {d: float(s) for d, s in ranked}
            
        except Exception as e:
            print(f"\nError qid={query.query_id}: {e}", file=sys.stderr)
            continue
            
    print(f"\nTime: {(time.perf_counter()-start)/60:.1f}m")
    
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg_cut_10', 'recall_1000'})
    scores = evaluator.evaluate(run_results)
    
    agg = {m: np.mean([v[m] for v in scores.values()]) for m in list(scores.values())[0]}
    print(json.dumps(agg, indent=2))
    
    RESULTS = Path("./ablation_results"); RESULTS.mkdir(exist_ok=True)
    fname = f"res_{dataset_name_safe}_{args.ablation_mode}_k{args.candidates_k}.json"
    with open(RESULTS / fname, 'w') as f:
        json.dump({'args': vars(args), 'metrics': agg}, f, indent=2)

if __name__ == "__main__":
    main()


# python vectorrag/eval/eval_agrerank_abalation_trec.py \
#   --dataset "msmarco-passage-v2/trec-dl-2022/judged" \
#   --ablation_mode agrerank_full \
#   --candidates_k 500 \
#   --cross_encoder_k 200
