import argparse
import json
import time
from pathlib import Path
import ir_datasets
import pytrec_eval
import numpy as np
import re
import os
import sys
from typing import List, Tuple, Any, Dict
from sentence_transformers import CrossEncoder
from pyserini.search.lucene import LuceneSearcher
from openai import OpenAI
from rank_bm25 import BM25Okapi
from scipy import stats

# --- CONFIGURATION ---
MODEL_EMBEDDING = "text-embedding-3-large"
MODEL_CROSS_ENCODER = 'cross-encoder/ms-marco-MiniLM-L-12-v2'

# --- UTILITIES ---

def l2n(V: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if V.ndim == 1: return V / max(np.linalg.norm(V), eps)
    norm = np.linalg.norm(V, axis=1, keepdims=True)
    norm = np.maximum(norm, eps)
    return V / norm

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)*")
STOP = set("the a an and or for to of in on at by from with as is are was were be be be being this that these those which who whose where when".split())

def tokenize(text: str) -> List[str]:
    toks = [t.lower() for t in _TOKEN_RE.findall(text or "")]
    return [t for t in toks if t not in STOP and len(t) > 1]

def percentile_normalize(scores: np.ndarray) -> np.ndarray:
    if len(scores) < 2: return np.zeros_like(scores, dtype=float)
    return stats.rankdata(scores, method='average') / len(scores)

def standard_normalize(scores: np.ndarray) -> np.ndarray:
    if len(scores) < 2: return np.zeros_like(scores, dtype=float)
    mean, std = np.mean(scores), np.std(scores)
    if std == 0: return np.zeros_like(scores)
    return np.clip((scores - mean) / std, -3, 3)

def safe_text(text: Any) -> str:
    try: return str(text or "").strip()
    except: return ""

def save_trec_run_file(results_dict: Dict[str, Dict[str, float]], run_tag: str, output_path: Path):
    print(f"Saving TREC run file ({run_tag}) to: {output_path}...")
    with open(output_path, 'w') as f:
        for qid, doc_scores in results_dict.items():
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            for rank, (doc_id, score) in enumerate(sorted_docs, start=1):
                line = f"{qid} Q0 {doc_id} {rank} {score:.6f} {run_tag}\n"
                f.write(line)

class OAI:
    def __init__(self, bs: int = 128):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key: sys.exit("Error: OPENAI_API_KEY not set.")
        self.c = OpenAI(api_key=api_key); self.bs = bs
    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts: return np.zeros((0, 3072), dtype=np.float32)
        out = []
        for i in range(0, len(texts), self.bs):
            batch = [safe_text(t) if t else " " for t in texts[i:i + self.bs]]
            try:
                resp = self.c.embeddings.create(model=MODEL_EMBEDDING, input=batch)
                out.append(np.array([d.embedding for d in resp.data], dtype=np.float32))
            except: out.append(np.zeros((len(batch), 3072), dtype=np.float32))
        return l2n(np.vstack(out)).astype(np.float32)

class EnhancedRerankingSystem:
    def __init__(self, args):
        self.args = args
        self.encoder = OAI()
        print(f"Loading Cross-Encoder: {MODEL_CROSS_ENCODER}...")
        self.cross_encoder = CrossEncoder(MODEL_CROSS_ENCODER)

    def compute_rrf_score(self, rank1: int, rank2: int, k: float = 60.0) -> float:
        return 1.0 / (k + rank1) + 1.0 / (k + rank2)
    
    def pseudo_relevance_feedback(self, query_vec: np.ndarray, doc_vecs: np.ndarray, 
                                initial_ranking: List[int], top_k: int = 15) -> np.ndarray:
        top_k = min(top_k, len(initial_ranking))
        if top_k == 0: return query_vec
        top_doc_vecs = doc_vecs[initial_ranking[:top_k]]
        alpha = 0.65
        expanded_query = alpha * query_vec + (1 - alpha) * np.mean(top_doc_vecs, axis=0)
        return l2n(expanded_query.reshape(1, -1))[0]

    def single_modality_reranker(self, precomputed: dict, scores: np.ndarray, modality_name: str) -> List[Tuple[str, float]]:
        """
        Baseline: Take Top K from single modality (Dense or BM25) -> Run Cross-Encoder.
        """
        cand_ids = precomputed['cand_ids']
        
        # 1. Select Top K based on single modality score
        top_k = min(self.args.cross_encoder_k, len(cand_ids))
        top_indices = np.argsort(scores)[-top_k:]
        
        query_text = safe_text(precomputed['query_text'])
        ce_pairs, valid_indices = [], []

        for idx in top_indices:
            doc_id = cand_ids[idx]
            doc_text = safe_text(precomputed['candidate_docs'].get(doc_id, ""))
            if query_text and doc_text:
                ce_pairs.append([query_text, doc_text])
                valid_indices.append(idx)
        
        if not ce_pairs:
            return [(cand_ids[i], float(scores[i])) for i in range(len(cand_ids))]

        # 2. Predict
        ce_scores = self.cross_encoder.predict(ce_pairs, show_progress_bar=False)
        
        # 3. Merge back into list
        final_scores = scores.copy()
        max_score = scores.max()
        min_ce = ce_scores.min()
        
        for i, idx in enumerate(valid_indices):
            # Boost CE scores above the highest original score
            final_scores[idx] = max_score + 1.0 + (ce_scores[i] - min_ce)

        final_ranking = sorted([
            (cand_ids[i], float(final_scores[i])) for i in range(len(cand_ids))
        ], key=lambda x: x[1], reverse=True)
        
        return final_ranking

    def rrf_reranker(self, precomputed: dict, rrf_scores: np.ndarray) -> List[Tuple[str, float]]:
        """
        Baseline: Take Top K from RRF -> Run Cross-Encoder.
        """
        cand_ids = precomputed['cand_ids']
        
        top_k = min(self.args.cross_encoder_k, len(cand_ids))
        top_indices = np.argsort(rrf_scores)[-top_k:]
        
        query_text = safe_text(precomputed['query_text'])
        ce_pairs, valid_indices = [], []

        for idx in top_indices:
            doc_id = cand_ids[idx]
            doc_text = safe_text(precomputed['candidate_docs'].get(doc_id, ""))
            if query_text and doc_text:
                ce_pairs.append([query_text, doc_text])
                valid_indices.append(idx)
        
        if not ce_pairs:
            return [(cand_ids[i], float(rrf_scores[i])) for i in range(len(cand_ids))]

        ce_scores = self.cross_encoder.predict(ce_pairs, show_progress_bar=False)
        
        final_scores = rrf_scores.copy()
        min_ce = ce_scores.min()
        max_rrf = rrf_scores.max()
        
        for i, idx in enumerate(valid_indices):
            final_scores[idx] = max_rrf + 1.0 + (ce_scores[i] - min_ce)

        final_ranking = sorted([
            (cand_ids[i], float(final_scores[i])) for i in range(len(cand_ids))
        ], key=lambda x: x[1], reverse=True)
        
        return final_ranking

    def enhanced_reranker(self, precomputed: dict) -> List[Tuple[str, float]]:
        """AgreRank (Strategy P)"""
        q_vec_flat = precomputed['query_vec'].reshape(-1)
        q_vec_2d = q_vec_flat.reshape(1, -1)
        doc_vecs = precomputed['doc_vecs']
        cand_ids = precomputed['cand_ids']
        
        dense_scores_raw = (q_vec_2d @ doc_vecs.T)[0]
        sparse_scores_raw = precomputed['sparse_scores']
        
        dense_scores = percentile_normalize(dense_scores_raw)
        sparse_scores = percentile_normalize(sparse_scores_raw)
        
        dense_ranks = np.argsort(np.argsort(dense_scores_raw)[::-1])
        sparse_ranks = np.argsort(np.argsort(sparse_scores_raw)[::-1])
        
        rrf_scores = np.array([
            self.compute_rrf_score(dense_ranks[i] + 1, sparse_ranks[i] + 1) for i in range(len(cand_ids))
        ])
        
        agreement_boost = ((dense_scores > 0.8) & (sparse_scores > 0.8)).astype(float) * 0.5
        initial_scores = standard_normalize(rrf_scores + agreement_boost)
        
        initial_ranking = np.argsort(initial_scores)[::-1]
        expanded_query_vec = self.pseudo_relevance_feedback(q_vec_flat, doc_vecs, initial_ranking)
        expanded_dense_scores_raw = (expanded_query_vec.reshape(1, -1) @ doc_vecs.T)[0]
        expanded_dense_scores = percentile_normalize(expanded_dense_scores_raw)
        
        dot_orig = (q_vec_2d @ doc_vecs.T)[0]
        orig_alignment = np.maximum(dot_orig, 0.0)
        all_scores = expanded_dense_scores * (0.5 + 0.5 * orig_alignment)
        
        ce_pool_k = min(self.args.cross_encoder_k * 2, len(cand_ids))
        ce_candidate_indices = np.argsort(all_scores)[-ce_pool_k:]
        
        query_text = safe_text(precomputed['query_text'])
        ce_pairs, id_to_idx, valid_ce_indices = [], {}, []
        for idx in ce_candidate_indices:
            doc_id = cand_ids[idx]
            doc_text = safe_text(precomputed['candidate_docs'].get(doc_id, ""))
            if query_text and doc_text:
                ce_pairs.append([query_text, doc_text])
                id_to_idx[doc_id] = idx
                valid_ce_indices.append(idx)
        
        if not ce_pairs:
             return [(doc_id, float(score)) for doc_id, score in sorted(zip(cand_ids, all_scores), key=lambda x: x[1], reverse=True)]

        try:
            ce_scores_raw = self.cross_encoder.predict(ce_pairs, show_progress_bar=False)
            if ce_scores_raw.size > 0:
                ce_min, ce_max = ce_scores_raw.min(), ce_scores_raw.max()
                t_min, t_max = all_scores.min(), all_scores.max()
                denom = (ce_max - ce_min) if (ce_max - ce_min) > 1e-9 else 1.0
                ce_scores_norm = (ce_scores_raw - ce_min) / denom
                ce_scores_calib = ce_scores_norm * (t_max - t_min) + t_min
                
                ce_weight = 0.5
                ce_doc_ids = [cand_ids[idx] for idx in valid_ce_indices]
                for i, doc_id in enumerate(ce_doc_ids):
                    original_idx = id_to_idx[doc_id]
                    all_scores[original_idx] = ce_weight * ce_scores_calib[i] + (1 - ce_weight) * all_scores[original_idx]
        except: pass
        
        return sorted([(cand_ids[i], float(all_scores[i])) for i in range(len(cand_ids))], key=lambda x: x[1], reverse=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--candidates_k", type=int, default=1000)
    parser.add_argument("--cross_encoder_k", type=int, default=150)
    parser.add_argument("--limit_queries", type=int, default=None)
    parser.add_argument("--cache_dir", type=str, default="./cache")
    args = parser.parse_args()

    dataset_name_safe = args.dataset.replace('/', '_')
    CACHE_DIR = Path(args.cache_dir) / f"{dataset_name_safe}_k{args.candidates_k}"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    dataset = ir_datasets.load(args.dataset)
    qrels, docs_store = {}, dataset.docs_store()
    for qrel in dataset.qrels_iter():
        qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance
    
    pyserini_index = 'msmarco-v2-passage' if 'v2' in args.dataset else 'msmarco-v1-passage'
    try: 
        bm25_searcher = LuceneSearcher.from_prebuilt_index(pyserini_index)
        bm25_searcher.set_bm25(k1=0.9, b=0.4)
    except: sys.exit("Pyserini Load Error")
    
    queries = list(dataset.queries_iter())
    if args.limit_queries: queries = queries[:args.limit_queries]
    
    system = EnhancedRerankingSystem(args)
    
    # All runs
    run_bm25 = {}       # NEW: BM25 only (no CE)
    run_dense = {}      # NEW: Dense only (no CE)
    run_rrf = {}
    run_dense_ce = {}
    run_bm25_ce = {}
    run_rrf_ce = {}
    run_agrerank = {}
    
    print(f"Processing {len(queries)} queries...")
    for i, query in enumerate(queries):
        print(f"\rQuery {i+1}/{len(queries)}", end='', flush=True)
        query_cache_file = CACHE_DIR / f"{query.query_id}.npz"
        precomputed = {}
        
        if query_cache_file.exists():
            try:
                d = np.load(query_cache_file, allow_pickle=True)
                precomputed = {k: d[k] for k in d}
                precomputed['cand_ids'] = precomputed['cand_ids'].tolist()
                precomputed['candidate_docs'] = precomputed['candidate_docs'].item()
            except: precomputed = {}
        
        if not precomputed:
            hits = bm25_searcher.search(query.text, k=args.candidates_k)
            cand_ids = [hit.docid for hit in hits]
            if not cand_ids: continue
            docs_dict = docs_store.get_many(cand_ids)
            cdocs = {cid: safe_text(getattr(docs_dict.get(cid), 'text', "")) for cid in cand_ids}
            qvec = system.encoder.encode([query.text])
            dvecs = system.encoder.encode(list(cdocs.values()))
            bm25 = BM25Okapi([tokenize(t) for t in cdocs.values()])
            sscores = bm25.get_scores(tokenize(query.text))
            precomputed = {'cand_ids': cand_ids, 'candidate_docs': cdocs, 'query_text': query.text,
                           'query_vec': qvec, 'doc_vecs': dvecs, 'sparse_scores': sscores}
            np.savez_compressed(query_cache_file, **precomputed)

        # Compute base scores
        d_scores = (precomputed['query_vec'] @ precomputed['doc_vecs'].T)[0]
        s_scores = precomputed['sparse_scores']
        d_rnk = np.argsort(np.argsort(d_scores)[::-1])
        s_rnk = np.argsort(np.argsort(s_scores)[::-1])
        rrf_scores_raw = np.array([(1/(60+d_rnk[x]+1)) + (1/(60+s_rnk[x]+1)) for x in range(len(precomputed['cand_ids']))])
        
        # 0a. NEW: BM25 only (no reranking)
        run_bm25[query.query_id] = {precomputed['cand_ids'][x]: float(s_scores[x]) for x in range(len(precomputed['cand_ids']))}
        
        # 0b. NEW: Dense only (no reranking)
        run_dense[query.query_id] = {precomputed['cand_ids'][x]: float(d_scores[x]) for x in range(len(precomputed['cand_ids']))}
        
        # 1. Baseline: RRF (no reranking)
        run_rrf[query.query_id] = {precomputed['cand_ids'][x]: rrf_scores_raw[x] for x in range(len(precomputed['cand_ids']))}

        # 2. Dense + CE
        dense_ce_res = system.single_modality_reranker(precomputed, d_scores, "Dense")
        run_dense_ce[query.query_id] = {d: s for d, s in dense_ce_res}

        # 3. BM25 + CE
        bm25_ce_res = system.single_modality_reranker(precomputed, s_scores, "BM25")
        run_bm25_ce[query.query_id] = {d: s for d, s in bm25_ce_res}

        # 4. RRF + CE
        rrf_ce_res = system.rrf_reranker(precomputed, rrf_scores_raw)
        run_rrf_ce[query.query_id] = {d: s for d, s in rrf_ce_res}

        # 5. AgreRank (Strategy P)
        ag_res = system.enhanced_reranker(precomputed)
        run_agrerank[query.query_id] = {d: s for d, s in ag_res}

    # --- EVALUATION ---
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg_cut_10', 'recip_rank'})
    
    def evaluate_and_print(run, name):
        res = evaluator.evaluate(run)
        metrics = {
            'map': np.mean([r['map'] for r in res.values()]), 
            'ndcg_cut_10': np.mean([r['ndcg_cut_10'] for r in res.values()]),
            'mrr': np.mean([r['recip_rank'] for r in res.values()])
        }
        print(f"\n--- {name} ---")
        print(json.dumps(metrics, indent=2))
        return metrics

    print("\n\n=== FINAL RESULTS ===")
    m_bm25 = evaluate_and_print(run_bm25, "0a. BM25 (No Reranking)")
    m_dense = evaluate_and_print(run_dense, "0b. Dense (No Reranking)")
    m_rrf = evaluate_and_print(run_rrf, "1. RRF (No Reranking)")
    m_dense_ce = evaluate_and_print(run_dense_ce, "2. Dense + CE")
    m_bm25_ce = evaluate_and_print(run_bm25_ce, "3. BM25 + CE")
    m_rrf_ce = evaluate_and_print(run_rrf_ce, "4. RRF + CE")
    m_agrerank = evaluate_and_print(run_agrerank, "5. AgreRank")

    # Print comparison table
    print("\n=== COMPARISON TABLE ===")
    print(f"{'Method':<25} {'nDCG@10':>10} {'MAP':>10} {'MRR':>10}")
    print("-" * 60)
    for name, metrics in [
        ("BM25", m_bm25),
        ("Dense", m_dense),
        ("RRF", m_rrf),
        ("BM25 + CE", m_bm25_ce),
        ("Dense + CE", m_dense_ce),
        ("RRF + CE", m_rrf_ce),
        ("AgreRank", m_agrerank)
    ]:
        print(f"{name:<25} {metrics['ndcg_cut_10']:>10.4f} {metrics['map']:>10.4f} {metrics['mrr']:>10.4f}")

    # --- SAVE FILES ---
    run_prefix = f"run_{dataset_name_safe}_k{args.candidates_k}"
    
    save_trec_run_file(run_bm25, "BM25", CACHE_DIR.parent / f"{run_prefix}_BM25.txt")
    save_trec_run_file(run_dense, "Dense", CACHE_DIR.parent / f"{run_prefix}_Dense.txt")
    save_trec_run_file(run_rrf, "RRF", CACHE_DIR.parent / f"{run_prefix}_RRF.txt")
    save_trec_run_file(run_dense_ce, "DenseCE", CACHE_DIR.parent / f"{run_prefix}_DenseCE.txt")
    save_trec_run_file(run_bm25_ce, "BM25CE", CACHE_DIR.parent / f"{run_prefix}_BM25CE.txt")
    save_trec_run_file(run_rrf_ce, "RRFCE", CACHE_DIR.parent / f"{run_prefix}_RRFCE.txt")
    save_trec_run_file(run_agrerank, "AgreRank", CACHE_DIR.parent / f"{run_prefix}_AgreRank.txt")
    
    # Save Metrics JSON
    with open(CACHE_DIR.parent / f"metrics_{run_prefix}.json", "w") as f:
        json.dump({
            "bm25": m_bm25,
            "dense": m_dense,
            "rrf": m_rrf,
            "dense_ce": m_dense_ce,
            "bm25_ce": m_bm25_ce,
            "rrf_ce": m_rrf_ce,
            "agrerank": m_agrerank
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {CACHE_DIR.parent}")

if __name__ == "__main__":
    main()


# Usage:

# python vectorrag/eval/agre_rank_with_baselines.py \
#     --dataset "msmarco-passage-v2/trec-dl-2022/judged" \
#     --candidates_k 1000 \
#     --cross_encoder_k 200

