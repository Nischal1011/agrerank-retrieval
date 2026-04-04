"""
Verify specific queries - check what each method actually returns.
"""

import argparse
from pathlib import Path
import ir_datasets
import numpy as np
from sentence_transformers import CrossEncoder
from scipy import stats
import re
from typing import List, Any

MODEL_CROSS_ENCODER = 'cross-encoder/ms-marco-MiniLM-L-12-v2'

def l2n(V, eps=1e-12):
    if V.ndim == 1: return V / max(np.linalg.norm(V), eps)
    norm = np.linalg.norm(V, axis=1, keepdims=True)
    return V / np.maximum(norm, eps)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)*")
STOP = set("the a an and or for to of in on at by from with as is are was were be been being this that these those which who whose where when".split())

def tokenize(text: str) -> List[str]:
    toks = [t.lower() for t in _TOKEN_RE.findall(text or "")]
    return [t for t in toks if t not in STOP and len(t) > 1]

def percentile_normalize(scores):
    if len(scores) < 2: return np.zeros_like(scores, dtype=float)
    return stats.rankdata(scores, method='average') / len(scores)

def standard_normalize(scores):
    if len(scores) < 2: return np.zeros_like(scores, dtype=float)
    mean, std = np.mean(scores), np.std(scores)
    if std == 0: return np.zeros_like(scores)
    return np.clip((scores - mean) / std, -3, 3)

def safe_text(text): 
    try: return str(text or "").strip()
    except: return ""

def compute_rrf_score(rank1, rank2, k=60.0):
    return 1.0 / (k + rank1) + 1.0 / (k + rank2)

def pseudo_relevance_feedback(query_vec, doc_vecs, initial_ranking, top_k=15):
    top_k = min(top_k, len(initial_ranking))
    if top_k == 0: return query_vec
    alpha = 0.65
    expanded_query = alpha * query_vec + (1 - alpha) * np.mean(doc_vecs[initial_ranking[:top_k]], axis=0)
    return l2n(expanded_query.reshape(1, -1))[0]

def run_method(precomputed, cross_encoder, method, cross_encoder_k=150):
    cand_ids = precomputed['cand_ids']
    d_scores = (precomputed['query_vec'] @ precomputed['doc_vecs'].T)[0]
    s_scores = precomputed['sparse_scores']
    d_rnk = np.argsort(np.argsort(d_scores)[::-1])
    s_rnk = np.argsort(np.argsort(s_scores)[::-1])
    rrf_scores = np.array([compute_rrf_score(d_rnk[x]+1, s_rnk[x]+1) for x in range(len(cand_ids))])
    
    query_text = safe_text(precomputed['query_text'])
    
    if method == "bm25":
        return {cand_ids[i]: float(s_scores[i]) for i in range(len(cand_ids))}
    
    elif method == "dense":
        return {cand_ids[i]: float(d_scores[i]) for i in range(len(cand_ids))}
    
    elif method == "bm25_ce":
        top_k = min(cross_encoder_k, len(cand_ids))
        top_indices = np.argsort(s_scores)[-top_k:]
        ce_pairs, valid_indices = [], []
        for idx in top_indices:
            doc_text = safe_text(precomputed['candidate_docs'].get(cand_ids[idx], ""))
            if query_text and doc_text:
                ce_pairs.append([query_text, doc_text])
                valid_indices.append(idx)
        if not ce_pairs:
            return {cand_ids[i]: float(s_scores[i]) for i in range(len(cand_ids))}
        ce_scores = cross_encoder.predict(ce_pairs, show_progress_bar=False)
        final_scores = s_scores.copy()
        for i, idx in enumerate(valid_indices):
            final_scores[idx] = s_scores.max() + 1.0 + (ce_scores[i] - ce_scores.min())
        return {cand_ids[i]: float(final_scores[i]) for i in range(len(cand_ids))}
    
    elif method == "dense_ce":
        top_k = min(cross_encoder_k, len(cand_ids))
        top_indices = np.argsort(d_scores)[-top_k:]
        ce_pairs, valid_indices = [], []
        for idx in top_indices:
            doc_text = safe_text(precomputed['candidate_docs'].get(cand_ids[idx], ""))
            if query_text and doc_text:
                ce_pairs.append([query_text, doc_text])
                valid_indices.append(idx)
        if not ce_pairs:
            return {cand_ids[i]: float(d_scores[i]) for i in range(len(cand_ids))}
        ce_scores = cross_encoder.predict(ce_pairs, show_progress_bar=False)
        final_scores = d_scores.copy()
        for i, idx in enumerate(valid_indices):
            final_scores[idx] = d_scores.max() + 1.0 + (ce_scores[i] - ce_scores.min())
        return {cand_ids[i]: float(final_scores[i]) for i in range(len(cand_ids))}
    
    elif method == "rrf_ce":
        top_k = min(cross_encoder_k, len(cand_ids))
        top_indices = np.argsort(rrf_scores)[-top_k:]
        ce_pairs, valid_indices = [], []
        for idx in top_indices:
            doc_text = safe_text(precomputed['candidate_docs'].get(cand_ids[idx], ""))
            if query_text and doc_text:
                ce_pairs.append([query_text, doc_text])
                valid_indices.append(idx)
        if not ce_pairs:
            return {cand_ids[i]: float(rrf_scores[i]) for i in range(len(cand_ids))}
        ce_scores = cross_encoder.predict(ce_pairs, show_progress_bar=False)
        final_scores = rrf_scores.copy()
        for i, idx in enumerate(valid_indices):
            final_scores[idx] = rrf_scores.max() + 1.0 + (ce_scores[i] - ce_scores.min())
        return {cand_ids[i]: float(final_scores[i]) for i in range(len(cand_ids))}
    
    elif method == "agrerank":
        q_vec_flat = precomputed['query_vec'].reshape(-1)
        q_vec_2d = q_vec_flat.reshape(1, -1)
        doc_vecs = precomputed['doc_vecs']
        
        dense_scores = percentile_normalize(d_scores)
        sparse_scores = percentile_normalize(s_scores)
        
        agreement_boost = ((dense_scores > 0.8) & (sparse_scores > 0.8)).astype(float) * 0.5
        initial_scores = standard_normalize(rrf_scores + agreement_boost)
        initial_ranking = np.argsort(initial_scores)[::-1]
        
        expanded_query_vec = pseudo_relevance_feedback(q_vec_flat, doc_vecs, initial_ranking)
        expanded_dense_scores = percentile_normalize((expanded_query_vec.reshape(1,-1) @ doc_vecs.T)[0])
        
        orig_alignment = np.maximum((q_vec_2d @ doc_vecs.T)[0], 0.0)
        all_scores = expanded_dense_scores * (0.5 + 0.5 * orig_alignment)
        
        ce_pool_k = min(cross_encoder_k * 2, len(cand_ids))
        ce_candidate_indices = np.argsort(all_scores)[-ce_pool_k:]
        
        ce_pairs, id_to_idx, valid_ce_indices = [], {}, []
        for idx in ce_candidate_indices:
            doc_text = safe_text(precomputed['candidate_docs'].get(cand_ids[idx], ""))
            if query_text and doc_text:
                ce_pairs.append([query_text, doc_text])
                id_to_idx[cand_ids[idx]] = idx
                valid_ce_indices.append(idx)
        
        if ce_pairs:
            try:
                ce_scores_raw = cross_encoder.predict(ce_pairs, show_progress_bar=False)
                if ce_scores_raw.size > 0:
                    ce_min, ce_max = ce_scores_raw.min(), ce_scores_raw.max()
                    t_min, t_max = all_scores.min(), all_scores.max()
                    denom = (ce_max - ce_min) if (ce_max - ce_min) > 1e-9 else 1.0
                    ce_scores_calib = ((ce_scores_raw - ce_min) / denom) * (t_max - t_min) + t_min
                    for i, idx in enumerate(valid_ce_indices):
                        all_scores[idx] = 0.5 * ce_scores_calib[i] + 0.5 * all_scores[idx]
            except: pass
        
        return {cand_ids[i]: float(all_scores[i]) for i in range(len(cand_ids))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="msmarco-passage-v2/trec-dl-2022/judged")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--candidates_k", type=int, default=1000)
    parser.add_argument("--query_text", type=str, help="Search for query containing this text")
    args = parser.parse_args()

    dataset_name_safe = args.dataset.replace('/', '_')
    CACHE_DIR = Path(args.cache_dir) / f"{dataset_name_safe}_k{args.candidates_k}"

    print(f"Loading dataset: {args.dataset}...")
    dataset = ir_datasets.load(args.dataset)
    
    qrels = {}
    for qrel in dataset.qrels_iter():
        qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = int(qrel.relevance)
    
    queries = {q.query_id: q.text for q in dataset.queries_iter()}
    docs_store = dataset.docs_store()
    
    # Find matching queries
    matching_qids = []
    for qid, qtext in queries.items():
        if args.query_text.lower() in qtext.lower():
            matching_qids.append(qid)
    
    if not matching_qids:
        print(f"No queries found containing: '{args.query_text}'")
        print("\nAvailable queries:")
        for qid, qtext in list(queries.items())[:20]:
            print(f"  {qid}: {qtext}")
        return
    
    print(f"Found {len(matching_qids)} matching queries")
    
    cross_encoder = CrossEncoder(MODEL_CROSS_ENCODER)
    
    for qid in matching_qids:
        cache_file = CACHE_DIR / f"{qid}.npz"
        if not cache_file.exists():
            print(f"Cache not found for {qid}")
            continue
        
        d = np.load(cache_file, allow_pickle=True)
        precomputed = {k: d[k] for k in d}
        precomputed['cand_ids'] = precomputed['cand_ids'].tolist()
        precomputed['candidate_docs'] = precomputed['candidate_docs'].item()
        
        print(f"\n{'='*70}")
        print(f"Query: '{queries[qid]}'")
        print(f"QID: {qid}")
        print(f"{'='*70}")
        
        methods = ["bm25", "dense", "bm25_ce", "dense_ce", "rrf_ce", "agrerank"]
        
        for method in methods:
            results = run_method(precomputed, cross_encoder, method)
            top3 = sorted(results.items(), key=lambda x: x[1], reverse=True)[:3]
            
            print(f"\n--- {method.upper()} Top 3 ---")
            for rank, (doc_id, score) in enumerate(top3, 1):
                rel = qrels.get(qid, {}).get(doc_id, 0)
                try:
                    text = docs_store.get(doc_id).text[:300]
                except:
                    text = "[Not found]"
                print(f"  Rank {rank} [rel={rel}]: {text}...")


if __name__ == "__main__":
    main()


# # Check Q1
# python vectorrag/rerank/find_examples_of_hard_query.py --query_text "london breed" --dataset "msmarco-passage-v2/trec-dl-2022/judged"

# # Check Q2  
# python vectorrag/rerank/find_examples_of_hard_query.py --query_text "baby nurse" --dataset "msmarco-passage-v2/trec-dl-2022/judged"