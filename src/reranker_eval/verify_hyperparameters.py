# Purpose: Verify Tau and Beta sensitivity to justify the "Robustness" claim.
import argparse
import json
import numpy as np
from pathlib import Path
import ir_datasets
import pytrec_eval
from scipy import stats

# --- CONFIG ---
DEFAULT_TAU = 0.8
DEFAULT_BETA = 0.5
DEFAULT_ALPHA = 0.65
DATASET = "msmarco-passage-v2/trec-dl-2022/judged"
CACHE_DIR = Path("cache/msmarco-passage-v2_trec-dl-2022_judged_k500")

def percentile_normalize(scores):
    if len(scores) < 2: return np.zeros_like(scores, dtype=float)
    return stats.rankdata(scores, method='average') / len(scores)

def standard_normalize(scores):
    if len(scores) < 2: return np.zeros_like(scores, dtype=float)
    std = np.std(scores)
    if std == 0: return np.zeros_like(scores)
    return np.clip((scores - np.mean(scores)) / std, -3, 3)

def run_pipeline(precomputed, tau, beta, alpha=0.65):
    # 1. Unpack
    q_vec = precomputed['query_vec'].reshape(1, -1)
    doc_vecs = precomputed['doc_vecs']
    cand_ids = precomputed['cand_ids']
    sparse_scores = precomputed['sparse_scores']
    
    # 2. Scores
    dense_scores_raw = (q_vec @ doc_vecs.T)[0]
    d_norm = percentile_normalize(dense_scores_raw)
    s_norm = percentile_normalize(sparse_scores)
    
    d_rnk = np.argsort(np.argsort(dense_scores_raw)[::-1])
    s_rnk = np.argsort(np.argsort(sparse_scores)[::-1])
    
    # 3. Stage 1: Consensus (TAU SENSITIVITY)
    rrf = np.array([(1/(60+d_rnk[i]+1)) + (1/(60+s_rnk[i]+1)) for i in range(len(cand_ids))])
    
    # --- VARIED PARAMETER: TAU ---
    boost = ((d_norm > tau) & (s_norm > tau)).astype(float) * 0.5
    init_scores = standard_normalize(rrf + boost)
    
    # 4. Stage 2: PRF
    rank1 = np.argsort(init_scores)[::-1]
    top_k = 15
    top_vecs = doc_vecs[rank1[:top_k]]
    
    # Handle edge case where top_vecs is empty
    if len(top_vecs) > 0:
        exp_q = alpha * q_vec + (1 - alpha) * np.mean(top_vecs, axis=0)
        exp_q = exp_q / np.maximum(np.linalg.norm(exp_q), 1e-12)
    else:
        exp_q = q_vec

    prf_raw = (exp_q @ doc_vecs.T)[0]
    prf_norm = percentile_normalize(prf_raw)
    
    # 5. Stage 3: Geometric Gating (BETA SENSITIVITY)
    dot_orig = (q_vec @ doc_vecs.T)[0]
    align = np.maximum(dot_orig, 0.0)
    
    # --- VARIED PARAMETER: BETA ---
    # Gate = Beta + (1-Beta) * Alignment
    gate = beta + (1.0 - beta) * align
    
    s_inter = prf_scores = prf_norm * gate
    
    return {cid: float(s_inter[i]) for i, cid in enumerate(cand_ids)}

def main():
    print(f"Loading Data from {CACHE_DIR}...")
    try:
        dataset = ir_datasets.load(DATASET)
    except:
        print("Error loading ir_datasets. Ensure credentials/path are correct.")
        return

    # Fixed qrels loading
    qrels = {}
    for qrel in dataset.qrels_iter():
        if qrel.query_id not in qrels:
            qrels[qrel.query_id] = {}
        qrels[qrel.query_id][qrel.doc_id] = int(qrel.relevance)
        
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg_cut_10'})
    
    # Load Cache
    cache_files = list(CACHE_DIR.glob("*.npz"))
    data_cache = {}
    print(f"Loaded {len(cache_files)} queries into RAM...")
    for cf in cache_files:
        qid = cf.stem
        try:
            d = np.load(cf, allow_pickle=True)
            data_cache[qid] = {
                'query_vec': d['query_vec'], 'doc_vecs': d['doc_vecs'],
                'cand_ids': d['cand_ids'], 'sparse_scores': d['sparse_scores']
            }
        except: pass

    if not data_cache:
        print("No cache files found. Run the main evaluation script first.")
        return

    # --- TEST TAU ---
    print("\n--- Testing Consensus Threshold (Tau) ---")
    taus = [0.7, 0.8, 0.9]
    results_tau = {}
    for t in taus:
        run = {}
        for qid, pc in data_cache.items():
            run[qid] = run_pipeline(pc, tau=t, beta=DEFAULT_BETA)
        
        # Handle missing queries in eval
        valid_run = {k:v for k,v in run.items() if k in qrels}
        if valid_run:
            res = evaluator.evaluate(valid_run)
            score = np.mean([v['ndcg_cut_10'] for v in res.values()])
            results_tau[t] = score
            print(f"Tau={t}: {score:.4f}")
    
    # --- TEST BETA ---
    print("\n--- Testing Drift Floor (Beta) ---")
    betas = [0.4, 0.5, 0.6]
    results_beta = {}
    for b in betas:
        run = {}
        for qid, pc in data_cache.items():
            run[qid] = run_pipeline(pc, tau=DEFAULT_TAU, beta=b)
            
        valid_run = {k:v for k,v in run.items() if k in qrels}
        if valid_run:
            res = evaluator.evaluate(valid_run)
            score = np.mean([v['ndcg_cut_10'] for v in res.values()])
            results_beta[b] = score
            print(f"Beta={b}: {score:.4f}")

    # --- CALC VARIANCE ---
    def get_var(res_dict):
        if not res_dict: return 0.0
        vals = list(res_dict.values())
        return (max(vals) - min(vals)) / min(vals) * 100

    var_tau = get_var(results_tau)
    var_beta = get_var(results_beta)

    print("\n=== CONCLUSION FOR PAPER ===")
    print(f"Tau Variance: {var_tau:.2f}%")
    print(f"Beta Variance: {var_beta:.2f}%")
    
    if var_tau < 2.0 and var_beta < 2.0:
        print("✅ CLAIM VERIFIED: You can include the paragraph.")
    else:
        print("❌ CLAIM FAILED: Variance is high. Remove the 'validated post-hoc' sentence.")

if __name__ == "__main__":
    main()