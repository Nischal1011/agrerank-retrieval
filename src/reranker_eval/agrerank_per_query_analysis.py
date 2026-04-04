import argparse
import json
import time
from pathlib import Path
import ir_datasets
import pytrec_eval
import numpy as np
import matplotlib.pyplot as plt
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
    try:
        toks = [t.lower() for t in _TOKEN_RE.findall(text or "")]
        return [t for t in toks if t not in STOP and len(t) > 1]
    except: return []

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
    print(f"  -> Saving TREC run file: {output_path}")
    with open(output_path, 'w') as f:
        for qid, doc_scores in results_dict.items():
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            for rank, (doc_id, score) in enumerate(sorted_docs, start=1):
                f.write(f"{qid} Q0 {doc_id} {rank} {score:.6f} {run_tag}\n")

# --- ANALYSIS FUNCTIONS ---

def save_per_query_deltas_json(
    base_res,
    model_res,
    output_path: Path,
    metric: str = "ndcg_cut_10"
):
    """
    Saves a clean JSON with only the per-query baseline score,
    model score, and relative % improvement.
    """
    print(f"\n💾 Saving raw per-query deltas to {output_path}...")

    common_qids = sorted(set(base_res.keys()) & set(model_res.keys()))
    if not common_qids:
        print("No overlapping qids between baseline and model results.")
        return

    per_query = []
    deltas = []

    for qid in common_qids:
        b = base_res[qid][metric]
        m = model_res[qid][metric]

        if b < 1e-9:
            gain = 100.0 if m > 1e-9 else 0.0
        else:
            gain = ((m - b) / b) * 100.0

        per_query.append({
            "qid": qid,
            "baseline": float(b),
            "agrerank": float(m),
            "relative_improvement_pct": float(gain)
        })

        deltas.append(gain)

    deltas = np.array(deltas)
    stats_block = {
        "num_queries": len(deltas),
        "mean_pct": float(np.mean(deltas)),
        "median_pct": float(np.median(deltas)),
        "num_positive": int(np.sum(deltas > 0)),
        "num_negative": int(np.sum(deltas < 0)),
        "num_bigwins": int(np.sum(deltas >= 50)),
    }

    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "w") as f:
        json.dump({
            "metric": metric,
            "stats": stats_block,
            "per_query": per_query
        }, f, indent=2)

    print("✅ Done.")


# --- CLASSES ---

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
    
    def enhanced_reranker(self, precomputed: dict) -> List[Tuple[str, float]]:
        q_vec_flat = precomputed['query_vec'].reshape(-1)
        q_vec_2d = q_vec_flat.reshape(1, -1)
        doc_vecs = precomputed['doc_vecs']
        cand_ids = precomputed['cand_ids']
        
        # 1. Raw Scores
        d_scores_raw = (q_vec_2d @ doc_vecs.T)[0]
        s_scores_raw = precomputed['sparse_scores']
        d_norm = percentile_normalize(d_scores_raw)
        s_norm = percentile_normalize(s_scores_raw)
        d_rnk = np.argsort(np.argsort(d_scores_raw)[::-1])
        s_rnk = np.argsort(np.argsort(s_scores_raw)[::-1])
        
        # 2. Consensus
        rrf = np.array([self.compute_rrf_score(d_rnk[i]+1, s_rnk[i]+1) for i in range(len(cand_ids))])
        boost = ((d_norm > 0.8) & (s_norm > 0.8)).astype(float) * 0.5
        init_scores = standard_normalize(rrf + boost)
        
        # 3. PRF
        rank1 = np.argsort(init_scores)[::-1]
        exp_q = self.pseudo_relevance_feedback(q_vec_flat, doc_vecs, rank1)
        prf_raw = (exp_q.reshape(1, -1) @ doc_vecs.T)[0]
        prf_norm = percentile_normalize(prf_raw)
        
        # 4. Geometric Gating (Strategy P)
        dot_orig = (q_vec_2d @ doc_vecs.T)[0]
        align = np.maximum(dot_orig, 0.0)
        all_scores = prf_norm * (0.5 + 0.5 * align)
        
        # 5. Neural
        ce_pool_k = min(self.args.cross_encoder_k * 2, len(cand_ids))
        ce_idx = np.argsort(all_scores)[-ce_pool_k:]
        q_txt = safe_text(precomputed['query_text'])
        ce_pairs, id_map, valid_idx = [], {}, []

        for idx in ce_idx:
            did = cand_ids[idx]
            txt = safe_text(precomputed['candidate_docs'].get(did, ""))
            if q_txt and txt:
                ce_pairs.append([q_txt, txt]); id_map[did]=idx; valid_idx.append(idx)
        
        if not ce_pairs:
            return sorted(zip(cand_ids, all_scores), key=lambda x: x[1], reverse=True)

        try:
            preds = self.cross_encoder.predict(ce_pairs, show_progress_bar=False)
            if preds.size > 0:
                mn, mx = preds.min(), preds.max()
                den = (mx - mn) if (mx - mn) > 1e-9 else 1.0
                preds_norm = (preds - mn) / den
                t_mn, t_mx = all_scores.min(), all_scores.max()
                preds_cal = preds_norm * (t_mx - t_mn) + t_mn
                w_ce = 0.5
                for i, idx in enumerate(valid_idx):
                    oid = id_map[cand_ids[idx]]
                    all_scores[oid] = w_ce * preds_cal[i] + (1-w_ce) * all_scores[oid]
        except: pass
        
        return sorted([(cand_ids[i], float(all_scores[i])) for i in range(len(cand_ids))], key=lambda x: x[1], reverse=True)

# --- MAIN ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--candidates_k", type=int, default=500)
    parser.add_argument("--cross_encoder_k", type=int, default=200)
    parser.add_argument("--limit_queries", type=int, default=None)
    parser.add_argument("--cache_dir", type=str, default="./cache")
    args = parser.parse_args()

    dataset_name_safe = args.dataset.replace('/', '_')
    CACHE_DIR = Path(args.cache_dir) / f"{dataset_name_safe}_k{args.candidates_k}"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Load Dataset: {args.dataset}")
    dataset = ir_datasets.load(args.dataset)
    
    # Map QID to Text for Qualitative Analysis
    queries_dict = {q.query_id: q.text for q in dataset.queries_iter()}
    
    qrels = {}; docs_store = dataset.docs_store()
    for qrel in dataset.qrels_iter(): qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance
    
    try: bm25_searcher = LuceneSearcher.from_prebuilt_index('msmarco-v2-passage'); bm25_searcher.set_bm25(k1=0.9, b=0.4)
    except: pass
    
    queries = list(dataset.queries_iter())
    if args.limit_queries: queries = queries[:args.limit_queries]
    
    system = EnhancedRerankingSystem(args)
    run_b, run_a = {}, {}
    
    print("Running Pipeline...")
    start_time = time.perf_counter()
    
    for i, query in enumerate(queries):
        print(f"\rQuery {i+1}/{len(queries)}", end='', flush=True)
        query_cache_file = CACHE_DIR / f"{query.query_id}.npz"
        precomputed = {}
        if query_cache_file.exists():
            try: d=np.load(query_cache_file, allow_pickle=True); pc={k:d[k] for k in d}; pc['cand_ids']=pc['cand_ids'].tolist(); pc['candidate_docs']=pc['candidate_docs'].item()
            except: pc={}
        
        if not pc:
            hits = bm25_searcher.search(query.text, k=args.candidates_k)
            cand_ids = [h.docid for h in hits] # FIXED: NameError was here
            if not cand_ids: continue
            docs_dict = docs_store.get_many(cand_ids)
            cdocs = {cid: safe_text(getattr(docs_dict.get(cid), 'text', "")) for cid in cand_ids}
            qv = system.encoder.encode([query.text])
            dv = system.encoder.encode(list(cdocs.values()))
            b = BM25Okapi([tokenize(t) for t in cdocs.values()])
            ss = b.get_scores(tokenize(query.text))
            pc = {'cand_ids':cand_ids, 'candidate_docs':cdocs, 'query_text':query.text, 'query_vec':qv, 'doc_vecs':dv, 'sparse_scores':ss}
            np.savez_compressed(query_cache_file, **pc)

        # Baseline
        d_scores = (pc['query_vec'] @ pc['doc_vecs'].T)[0]
        d_rnk = np.argsort(np.argsort(d_scores)[::-1])
        s_rnk = np.argsort(np.argsort(pc['sparse_scores'])[::-1])
        run_b[query.query_id] = {pc['cand_ids'][i]: (1/(60+d_rnk[i]+1)) + (1/(60+s_rnk[i]+1)) for i in range(len(pc['cand_ids']))}

        # AgreRank
        adv = system.enhanced_reranker(pc)
        run_a[query.query_id] = {d: s for d, s in adv}
        
    print(f"\nTotal Time: {(time.perf_counter() - start_time)/60:.1f}m")

    # --- EVALUATION ---
    ev = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg_cut_10'})
    res_b = ev.evaluate(run_b)
    res_a = ev.evaluate(run_a)

    def agg(res): return {m: np.mean([r[m] for r in res.values()]) for m in list(res.values())[0].keys()} if res else {}
    print("\n--- Results ---")
    print("Baseline:", json.dumps(agg(res_b), indent=2))
    print("AgreRank:", json.dumps(agg(res_a), indent=2))

    # --- 1. Save JSONs (for safety) ---
    path_b = CACHE_DIR.parent / f"results_{dataset_name_safe}_BASELINE_k{args.candidates_k}.json"
    with open(path_b, "w") as f: json.dump({"metrics": agg(res_b), "per_query": res_b}, f, indent=2)
    
    path_a = CACHE_DIR.parent / f"results_{dataset_name_safe}_AGRERANK_k{args.candidates_k}.json"
    with open(path_a, "w") as f: json.dump({"metrics": agg(res_a), "per_query": res_a}, f, indent=2)

    # --- 2. Save TREC Files ---
    save_trec_run_file(run_b, "RRF_Base", CACHE_DIR.parent / f"run_{dataset_name_safe}_baseline.txt")
    save_trec_run_file(run_a, "AgreRank", CACHE_DIR.parent / f"run_{dataset_name_safe}_agrerank.txt")
    print("\nSaved .txt run files.")

    # --- 3. SAVE the Delta  ---
    plot_path = CACHE_DIR.parent / f"per_query_deltas_{dataset_name_safe}.json"


if __name__ == "__main__":
    main()

# python vectorrag/eval/agrerank_per_query_analysis.py \
#     --dataset "msmarco-passage-v2/trec-dl-2022/judged" \
#     --candidates_k 500 \
#     --cross_encoder_k 200