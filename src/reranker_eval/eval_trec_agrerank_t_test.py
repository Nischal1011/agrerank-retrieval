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
from sentence_transformers import CrossEncoder, SentenceTransformer
from pyserini.search.lucene import LuceneSearcher
from scipy import stats

# --- CONFIGURATION ---
MODEL_CROSS_ENCODER = 'cross-encoder/ms-marco-MiniLM-L-12-v2'

# Encoder configurations
ENCODER_CONFIGS = {
    'openai': {
        'model': 'text-embedding-3-large',
        'dim': 3072,
    },
    'bge': {
        'model': 'BAAI/bge-large-en-v1.5',
        'dim': 1024,
    },
    'e5': {
        'model': 'intfloat/e5-large-v2',
        'dim': 1024,
    }
}

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

# --- ENCODER CLASSES ---

class OpenAIEncoder:
    """OpenAI text-embedding-3-large encoder"""
    def __init__(self, bs: int = 128):
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key: 
            sys.exit("Error: OPENAI_API_KEY not set.")
        self.client = OpenAI(api_key=api_key)
        self.bs = bs
        self.model = ENCODER_CONFIGS['openai']['model']
        self.dim = ENCODER_CONFIGS['openai']['dim']
        print(f"Initialized OpenAI encoder: {self.model} (dim={self.dim})")

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts: return np.zeros((0, self.dim), dtype=np.float32)
        out = []
        for i in range(0, len(texts), self.bs):
            batch = [safe_text(t) if t else " " for t in texts[i:i + self.bs]]
            try:
                resp = self.client.embeddings.create(model=self.model, input=batch)
                out.append(np.array([d.embedding for d in resp.data], dtype=np.float32))
            except Exception as e:
                print(f"Warning: OpenAI API error: {e}")
                out.append(np.zeros((len(batch), self.dim), dtype=np.float32))
        return l2n(np.vstack(out)).astype(np.float32)


class HuggingFaceEncoder:
    """HuggingFace SentenceTransformer encoder (BGE, E5, etc.)"""
    def __init__(self, encoder_name: str = 'bge', bs: int = 32):
        config = ENCODER_CONFIGS[encoder_name]
        self.model_name = config['model']
        self.dim = config['dim']
        self.bs = bs
        self.encoder_name = encoder_name
        
        print(f"Loading HuggingFace encoder: {self.model_name} (dim={self.dim})...")
        self.model = SentenceTransformer(self.model_name)
        print(f"Initialized {encoder_name.upper()} encoder")

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts: return np.zeros((0, self.dim), dtype=np.float32)
        
        processed_texts = []
        for t in texts:
            t = safe_text(t) if t else " "
            processed_texts.append(t)
        
        embeddings = self.model.encode(
            processed_texts, 
            batch_size=self.bs, 
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return embeddings.astype(np.float32)
    
    def encode_queries(self, texts: List[str]) -> np.ndarray:
        """Encode queries with appropriate prefixes"""
        if not texts: return np.zeros((0, self.dim), dtype=np.float32)
        
        processed_texts = []
        for t in texts:
            t = safe_text(t) if t else " "
            if self.encoder_name == 'e5':
                processed_texts.append(f"query: {t}")
            elif self.encoder_name == 'bge':
                processed_texts.append(f"Represent this sentence for searching relevant passages: {t}")
            else:
                processed_texts.append(t)
        
        embeddings = self.model.encode(
            processed_texts,
            batch_size=self.bs,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return embeddings.astype(np.float32)
    
    def encode_documents(self, texts: List[str]) -> np.ndarray:
        """Encode documents (no prefix needed for most models)"""
        if not texts: return np.zeros((0, self.dim), dtype=np.float32)
        
        processed_texts = []
        for t in texts:
            t = safe_text(t) if t else " "
            if self.encoder_name == 'e5':
                processed_texts.append(f"passage: {t}")
            else:
                processed_texts.append(t)
        
        embeddings = self.model.encode(
            processed_texts,
            batch_size=self.bs,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return embeddings.astype(np.float32)


def get_encoder(encoder_name: str):
    """Factory function to get the appropriate encoder"""
    if encoder_name == 'openai':
        return OpenAIEncoder()
    elif encoder_name in ['bge', 'e5']:
        return HuggingFaceEncoder(encoder_name)
    else:
        raise ValueError(f"Unknown encoder: {encoder_name}. Choose from: openai, bge, e5")


# --- MAIN SYSTEM CLASS ---

class EnhancedRerankingSystem:
    def __init__(self, args):
        self.args = args
        self.encoder_name = args.encoder
        self.encoder = get_encoder(args.encoder)
        self.use_hf_encoder = args.encoder in ['bge', 'e5']
        
        print(f"Loading Cross-Encoder: {MODEL_CROSS_ENCODER}...")
        self.cross_encoder = CrossEncoder(MODEL_CROSS_ENCODER)

    def encode_query(self, query_text: str) -> np.ndarray:
        """Encode a single query"""
        if self.use_hf_encoder:
            return self.encoder.encode_queries([query_text])
        else:
            return self.encoder.encode([query_text])
    
    def encode_documents(self, doc_texts: List[str]) -> np.ndarray:
        """Encode documents"""
        if self.use_hf_encoder:
            return self.encoder.encode_documents(doc_texts)
        else:
            return self.encoder.encode(doc_texts)

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
    
    def rrf_baseline(self, precomputed: dict) -> Dict[str, float]:
        """RRF baseline without cross-encoder"""
        q_vec = precomputed['query_vec']
        doc_vecs = precomputed['doc_vecs']
        cand_ids = precomputed['cand_ids']
        
        d_scores = (q_vec @ doc_vecs.T)[0]
        s_scores = precomputed['sparse_scores']
        
        d_rnk = np.argsort(np.argsort(d_scores)[::-1])
        s_rnk = np.argsort(np.argsort(s_scores)[::-1])
        
        return {
            cand_ids[i]: (1/(60 + d_rnk[i] + 1)) + (1/(60 + s_rnk[i] + 1)) 
            for i in range(len(cand_ids))
        }
    
    def rrf_ce_baseline(self, precomputed: dict) -> Dict[str, float]:
        """RRF + Cross-Encoder baseline (industry standard pipeline)"""
        q_vec = precomputed['query_vec']
        doc_vecs = precomputed['doc_vecs']
        cand_ids = precomputed['cand_ids']
        q_txt = safe_text(precomputed['query_text'])
        
        # Step 1: Compute RRF scores
        d_scores = (q_vec @ doc_vecs.T)[0]
        s_scores = precomputed['sparse_scores']
        
        d_rnk = np.argsort(np.argsort(d_scores)[::-1])
        s_rnk = np.argsort(np.argsort(s_scores)[::-1])
        
        rrf_scores = np.array([
            (1/(60 + d_rnk[i] + 1)) + (1/(60 + s_rnk[i] + 1)) 
            for i in range(len(cand_ids))
        ])
        
        # Step 2: Select top-M candidates for cross-encoder
        ce_k = min(self.args.cross_encoder_k, len(cand_ids))
        top_indices = np.argsort(rrf_scores)[-ce_k:]
        
        # Step 3: Apply cross-encoder to top candidates
        ce_pairs = []
        valid_indices = []
        for idx in top_indices:
            did = cand_ids[idx]
            txt = safe_text(precomputed['candidate_docs'].get(did, ""))
            if q_txt and txt:
                ce_pairs.append([q_txt, txt])
                valid_indices.append(idx)
        
        # Initialize final scores with RRF scores
        final_scores = rrf_scores.copy()
        
        if ce_pairs:
            try:
                ce_preds = self.cross_encoder.predict(ce_pairs, show_progress_bar=False)
                
                # Normalize CE scores to [0, 1]
                if ce_preds.size > 0:
                    mn, mx = ce_preds.min(), ce_preds.max()
                    if mx - mn > 1e-9:
                        ce_preds_norm = (ce_preds - mn) / (mx - mn)
                    else:
                        ce_preds_norm = np.ones_like(ce_preds) * 0.5
                    
                    # Scale to RRF score range for fusion
                    rrf_min, rrf_max = rrf_scores.min(), rrf_scores.max()
                    ce_preds_scaled = ce_preds_norm * (rrf_max - rrf_min) + rrf_min
                    
                    # Fuse: 50% CE + 50% RRF (same as AgreRank's Stage 4)
                    w_ce = 0.5
                    for i, idx in enumerate(valid_indices):
                        final_scores[idx] = w_ce * ce_preds_scaled[i] + (1 - w_ce) * rrf_scores[idx]
            except Exception as e:
                print(f"Warning: CE baseline error: {e}")
        
        return {cand_ids[i]: float(final_scores[i]) for i in range(len(cand_ids))}
    
    def agrerank_reranker(self, precomputed: dict) -> Dict[str, float]:
        """Full AgreRank pipeline"""
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
        
        # 2. Consensus Seeding
        rrf = np.array([self.compute_rrf_score(d_rnk[i]+1, s_rnk[i]+1) for i in range(len(cand_ids))])
        boost = ((d_norm > 0.8) & (s_norm > 0.8)).astype(float) * 0.5
        init_scores = standard_normalize(rrf + boost)
        
        # 3. Anchor-Guided Expansion (PRF)
        rank1 = np.argsort(init_scores)[::-1]
        exp_q = self.pseudo_relevance_feedback(q_vec_flat, doc_vecs, rank1)
        prf_raw = (exp_q.reshape(1, -1) @ doc_vecs.T)[0]
        prf_norm = percentile_normalize(prf_raw)
        
        # 4. Geometric Gating (Strategy P)
        dot_orig = (q_vec_2d @ doc_vecs.T)[0]
        align = np.maximum(dot_orig, 0.0)
        all_scores = prf_norm * (0.5 + 0.5 * align)
        
        # 5. Neural Refinement
        ce_pool_k = min(self.args.cross_encoder_k * 2, len(cand_ids))
        ce_idx = np.argsort(all_scores)[-ce_pool_k:]
        q_txt = safe_text(precomputed['query_text'])
        ce_pairs, id_map, valid_idx = [], {}, []

        for idx in ce_idx:
            did = cand_ids[idx]
            txt = safe_text(precomputed['candidate_docs'].get(did, ""))
            if q_txt and txt:
                ce_pairs.append([q_txt, txt])
                id_map[did] = idx
                valid_idx.append(idx)
        
        if not ce_pairs:
            return {cand_ids[i]: float(all_scores[i]) for i in range(len(cand_ids))}

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
                    all_scores[oid] = w_ce * preds_cal[i] + (1 - w_ce) * all_scores[oid]
        except Exception as e:
            print(f"Warning: AgreRank CE error: {e}")
        
        return {cand_ids[i]: float(all_scores[i]) for i in range(len(cand_ids))}


# --- MAIN ---

def main():
    parser = argparse.ArgumentParser(description="AgreRank evaluation with multiple encoder support")
    parser.add_argument("--dataset", type=str, required=True, 
                        help="IR dataset name (e.g., msmarco-passage-v2/trec-dl-2022/judged)")
    parser.add_argument("--encoder", type=str, default="openai", choices=["openai", "bge", "e5"],
                        help="Dense encoder to use: openai (text-embedding-3-large), bge (BGE-large-en-v1.5), e5 (E5-large-v2)")
    parser.add_argument("--candidates_k", type=int, default=500,
                        help="Number of candidates to retrieve")
    parser.add_argument("--cross_encoder_k", type=int, default=200,
                        help="Number of candidates to re-rank with cross-encoder")
    parser.add_argument("--limit_queries", type=int, default=None,
                        help="Limit number of queries for testing")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="Directory for caching embeddings")
    parser.add_argument("--no_cache", action="store_true",
                        help="Disable caching (recompute all embeddings)")
    args = parser.parse_args()

    # Include encoder in cache path to avoid mixing embeddings
    dataset_name_safe = args.dataset.replace('/', '_')
    CACHE_DIR = Path(args.cache_dir) / f"{dataset_name_safe}_k{args.candidates_k}_{args.encoder}"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=" * 70)
    print(f"AgreRank Evaluation (Multi-Encoder)")
    print(f"=" * 70)
    print(f"Dataset:         {args.dataset}")
    print(f"Encoder:         {args.encoder} ({ENCODER_CONFIGS[args.encoder]['model']})")
    print(f"Candidates K:    {args.candidates_k}")
    print(f"Cross-Encoder K: {args.cross_encoder_k}")
    print(f"Cache Dir:       {CACHE_DIR}")
    print(f"=" * 70)

    print(f"\nLoading dataset: {args.dataset}")
    dataset = ir_datasets.load(args.dataset)
    qrels, docs_store = {}, dataset.docs_store()
    for qrel in dataset.qrels_iter(): 
        qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance
    
    try: 
        bm25_searcher = LuceneSearcher.from_prebuilt_index('msmarco-v2-passage')
        bm25_searcher.set_bm25(k1=0.9, b=0.4)
        print("Loaded Pyserini BM25 index")
    except Exception as e:
        print(f"Warning: Could not load Pyserini index: {e}")
        bm25_searcher = None
    
    queries = list(dataset.queries_iter())
    if args.limit_queries: 
        queries = queries[:args.limit_queries]
    print(f"Processing {len(queries)} queries")
    
    system = EnhancedRerankingSystem(args)
    
    # Three runs: RRF, RRF+CE, AgreRank
    run_rrf = {}      # RRF only (no CE)
    run_rrf_ce = {}   # RRF + CE (baseline)
    run_agrerank = {} # AgreRank (ours)
    
    print("\nRunning Pipeline...")
    start_time = time.time()
    
    for i, query in enumerate(queries):
        print(f"\rQuery {i+1}/{len(queries)}: {query.query_id}", end='', flush=True)
        
        query_cache_file = CACHE_DIR / f"{query.query_id}.npz"
        precomputed = {}
        
        # Try to load from cache
        if query_cache_file.exists() and not args.no_cache:
            try:
                d = np.load(query_cache_file, allow_pickle=True)
                precomputed = {k: d[k] for k in d}
                precomputed['cand_ids'] = precomputed['cand_ids'].tolist()
                precomputed['candidate_docs'] = precomputed['candidate_docs'].item()
            except Exception as e:
                print(f"\nWarning: Cache load failed for {query.query_id}: {e}")
                precomputed = {}
        
        # Compute if not cached
        if not precomputed:
            if bm25_searcher is None:
                print(f"\nSkipping {query.query_id}: No BM25 searcher")
                continue
                
            hits = bm25_searcher.search(query.text, k=args.candidates_k)
            cids = [h.docid for h in hits]
            if not cids: 
                continue
            
            cdocs = {}
            for c in cids:
                doc = docs_store.get(c)
                cdocs[c] = safe_text(getattr(doc, 'text', "")) if doc else ""
            
            # Use appropriate encoding method
            qv = system.encode_query(query.text)
            dv = system.encode_documents(list(cdocs.values()))
            
            # BM25 scores
            from rank_bm25 import BM25Okapi
            b = BM25Okapi([tokenize(t) for t in cdocs.values()])
            ss = b.get_scores(tokenize(query.text))
            
            precomputed = {
                'cand_ids': cids,
                'candidate_docs': cdocs,
                'query_text': query.text,
                'query_vec': qv,
                'doc_vecs': dv,
                'sparse_scores': ss
            }
            
            # Save to cache
            if not args.no_cache:
                np.savez_compressed(query_cache_file, **precomputed)

        # Run all three methods
        run_rrf[query.query_id] = system.rrf_baseline(precomputed)
        run_rrf_ce[query.query_id] = system.rrf_ce_baseline(precomputed)
        run_agrerank[query.query_id] = system.agrerank_reranker(precomputed)
    
    elapsed = time.time() - start_time
    print(f"\n\nCompleted in {elapsed:.1f}s ({elapsed/len(queries):.2f}s per query)")
        
    # Evaluation
    print(f"\n{'=' * 70}")
    print("Evaluation Results")
    print(f"{'=' * 70}")
    
    ev = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg_cut_10', 'recall_1000', 'recip_rank'})
    res_rrf = ev.evaluate(run_rrf)
    res_rrf_ce = ev.evaluate(run_rrf_ce)
    res_agrerank = ev.evaluate(run_agrerank)

    def agg(res): 
        if not res: return {}
        return {m: np.mean([r[m] for r in res.values()]) for m in list(res.values())[0].keys()}
    
    agg_rrf = agg(res_rrf)
    agg_rrf_ce = agg(res_rrf_ce)
    agg_agrerank = agg(res_agrerank)
    
    print(f"\nEncoder: {args.encoder.upper()} ({ENCODER_CONFIGS[args.encoder]['model']})")
    
    print(f"\n{'Method':<20} {'nDCG@10':>10} {'MAP':>10} {'MRR':>10}")
    print("-" * 52)
    print(f"{'RRF':<20} {agg_rrf.get('ndcg_cut_10', 0):>10.4f} {agg_rrf.get('map', 0):>10.4f} {agg_rrf.get('recip_rank', 0):>10.4f}")
    print(f"{'RRF + CE':<20} {agg_rrf_ce.get('ndcg_cut_10', 0):>10.4f} {agg_rrf_ce.get('map', 0):>10.4f} {agg_rrf_ce.get('recip_rank', 0):>10.4f}")
    print(f"{'AgreRank (Ours)':<20} {agg_agrerank.get('ndcg_cut_10', 0):>10.4f} {agg_agrerank.get('map', 0):>10.4f} {agg_agrerank.get('recip_rank', 0):>10.4f}")
    
    # Compute improvements
    rrf_ce_ndcg = agg_rrf_ce.get('ndcg_cut_10', 0)
    agrerank_ndcg = agg_agrerank.get('ndcg_cut_10', 0)
    improvement = 0
    if rrf_ce_ndcg > 0:
        improvement = ((agrerank_ndcg - rrf_ce_ndcg) / rrf_ce_ndcg) * 100
        print(f"\n{'Gain vs RRF+CE:':<20} {improvement:>+10.1f}%")

    # --- SIGNIFICANCE TESTING (PAIRED T-TEST) ---
    print(f"\n{'=' * 70}")
    print("Significance Testing: AgreRank vs RRF+CE (Paired t-test on nDCG@10)")
    print(f"{'=' * 70}")
    
    common_ids = sorted(list(set(res_rrf_ce.keys()) & set(res_agrerank.keys())))
    p_val = None
    
    if len(common_ids) > 1:
        scores_baseline = [res_rrf_ce[qid]['ndcg_cut_10'] for qid in common_ids]
        scores_agrerank = [res_agrerank[qid]['ndcg_cut_10'] for qid in common_ids]
        
        t_stat, p_val = stats.ttest_rel(scores_agrerank, scores_baseline)
        
        print(f"Samples:         {len(common_ids)}")
        print(f"RRF+CE Mean:     {np.mean(scores_baseline):.4f}")
        print(f"AgreRank Mean:   {np.mean(scores_agrerank):.4f}")
        print(f"Improvement:     {((np.mean(scores_agrerank) - np.mean(scores_baseline)) / np.mean(scores_baseline)) * 100:+.2f}%")
        print(f"t-statistic:     {t_stat:.4f}")
        print(f"p-value:         {p_val:.6f}")
        
        if p_val < 0.01: 
            print("Result:          Highly Significant (p < 0.01) ‡")
        elif p_val < 0.05: 
            print("Result:          Significant (p < 0.05) †")
        else: 
            print("Result:          Not Significant")
    else:
        print("Not enough common queries for t-test.")

    # Save artifacts
    output_dir = CACHE_DIR.parent
    
    save_trec_run_file(run_rrf, f"RRF_{args.encoder}", 
                       output_dir / f"run_{dataset_name_safe}_{args.encoder}_rrf.txt")
    save_trec_run_file(run_rrf_ce, f"RRF_CE_{args.encoder}", 
                       output_dir / f"run_{dataset_name_safe}_{args.encoder}_rrf_ce.txt")
    save_trec_run_file(run_agrerank, f"AgreRank_{args.encoder}", 
                       output_dir / f"run_{dataset_name_safe}_{args.encoder}_agrerank.txt")
    
    # Save summary JSON
    summary = {
        'dataset': args.dataset,
        'encoder': args.encoder,
        'encoder_model': ENCODER_CONFIGS[args.encoder]['model'],
        'candidates_k': args.candidates_k,
        'cross_encoder_k': args.cross_encoder_k,
        'num_queries': len(common_ids),
        'results': {
            'rrf': agg_rrf,
            'rrf_ce': agg_rrf_ce,
            'agrerank': agg_agrerank,
        },
        'improvement_vs_rrf_ce': {
            'ndcg_cut_10': ((agg_agrerank.get('ndcg_cut_10', 0) - agg_rrf_ce.get('ndcg_cut_10', 0)) / agg_rrf_ce.get('ndcg_cut_10', 1)) * 100 if agg_rrf_ce.get('ndcg_cut_10', 0) > 0 else 0,
            'map': ((agg_agrerank.get('map', 0) - agg_rrf_ce.get('map', 0)) / agg_rrf_ce.get('map', 1)) * 100 if agg_rrf_ce.get('map', 0) > 0 else 0,
        },
        'p_value': p_val,
        'significant_01': p_val < 0.01 if p_val else None,
        'significant_05': p_val < 0.05 if p_val else None,
    }
    
    summary_file = output_dir / f"summary_{dataset_name_safe}_{args.encoder}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {summary_file}")
    


if __name__ == "__main__":
    main()


# Usage examples:

# OpenAI encoder (default):
# python eval_trec_agrerank_multi_encoder.py \
#   --dataset "msmarco-passage-v2/trec-dl-2022/judged" \
#   --encoder openai \
#   --candidates_k 500 \
#   --cross_encoder_k 200
#
# BGE encoder:
# python eval_trec_agrerank_multi_encoder.py \
#   --dataset "msmarco-passage-v2/trec-dl-2022/judged" \
#   --encoder bge \
#   --candidates_k 500 \
#   --cross_encoder_k 200
#
# E5 encoder:
# python eval_trec_agrerank_multi_encoder.py \
#   --dataset "msmarco-passage-v2/trec-dl-2022/judged" \
#   --encoder e5 \
#   --candidates_k 500 \
#   --cross_encoder_k 200
#
# Quick test with limited queries:
# python eval_trec_agrerank_multi_encoder.py \
#   --dataset "msmarco-passage-v2/trec-dl-2022/judged" \
#   --encoder bge \
#   --candidates_k 500 \
#   --limit_queries 10
# Usage examples:
#
# OpenAI encoder (default):
# python eval_trec_agrerank_multi_encoder.py \
#   --dataset "msmarco-passage-v2/trec-dl-2022/judged" \
#   --encoder openai \
#   --candidates_k 500 \
#   --cross_encoder_k 200
#
# BGE encoder:
# python vectorrag/eval/eval_trec_agrerank_t_test.py \
#   --dataset "msmarco-passage-v2/trec-dl-2022/judged" \
#   --encoder bge \
#   --candidates_k 500 \
#   --cross_encoder_k 200
#
# E5 encoder:
# python vectorrag/eval/eval_trec_agrerank_t_test.py \
#   --dataset "msmarco-passage-v2/trec-dl-2022/judged" \
#   --encoder e5 \
#   --candidates_k 500 \
#   --cross_encoder_k 200
#
# Quick test with limited queries:
# python eval_trec_agrerank_multi_encoder.py \
#   --dataset "msmarco-passage-v2/trec-dl-2022/judged" \
#   --encoder bge \
#   --candidates_k 500 \
#   --limit_queries 10