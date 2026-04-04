import argparse
import json
import time
from pathlib import Path
import numpy as np
import re
import os
import sys
from typing import Dict, List, Tuple, Any

# --- Dependencies ---
try: import faiss
except: pass
try: from openai import OpenAI
except: pass
try: from rank_bm25 import BM25Okapi
except: pass
try: from sentence_transformers import CrossEncoder
except: pass
try: from scipy import stats
except: pass

# --- Config ---
MODEL_EMBEDDING = "text-embedding-3-large"
MODEL_CROSS_ENCODER = 'cross-encoder/ms-marco-MiniLM-L-12-v2'

# --- Utils ---
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)*")
STOP = set("the a an and or for to of in on at by from with as is are was were be been being this that these those which who whose where when".split())

def toks(s: str) -> List[str]:
    if not s: return []
    out = [w.lower() for w in _TOKEN_RE.findall(s)]
    return [w for w in out if w not in STOP and len(w) > 1]

def safe_text(text: Any) -> str:
    try: return str(text or "").strip()
    except: return ""

def l2n(V: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if V.ndim == 1:
        n = np.linalg.norm(V)
        return V / max(n, eps)
    n = np.linalg.norm(V, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return V / n

def percentile_normalize(scores: np.ndarray) -> np.ndarray:
    """Equation (1): Rank-based percentile normalization to [0,1]."""
    if len(scores) < 2: 
        return np.zeros_like(scores, dtype=np.float32)
    # rankdata returns 1-based ranks, we convert to [0,1]
    ranks = stats.rankdata(scores, 'average')
    return ((ranks - 1) / (len(scores) - 1)).astype(np.float32)

class OAI:
    def __init__(self, bs: int = 128):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key: 
            sys.exit("Error: OPENAI_API_KEY not set.")
        self.c = OpenAI(api_key=api_key)
        self.bs = bs
        
    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts: 
            return np.zeros((0, 3072), dtype=np.float32)
        out = []
        for i in range(0, len(texts), self.bs):
            batch = [safe_text(t) if t.strip() else " " for t in texts[i:i + self.bs]]
            try:
                resp = self.c.embeddings.create(model=MODEL_EMBEDDING, input=batch)
                out.append(np.array([d.embedding for d in resp.data], dtype=np.float32))
            except: 
                out.append(np.zeros((len(batch), 3072), dtype=np.float32))
        return l2n(np.vstack(out)).astype(np.float32)

def load_master(path: Path) -> Tuple[List[str], Dict[str, str]]:
    ids, txt = [], {}
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

def eval_system(search_fn, qset: List[dict], top_k: int) -> Dict[str, float]:
    single, multi, all_f1 = [], [], []
    start_time = time.perf_counter()
    
    for i, q in enumerate(qset):
        print(f"Processing query {i+1}/{len(qset)}...", file=sys.stderr, end='\r')
        res = search_fn(q["query_text"])
        toplist = [cid for cid, _ in res]
        rel_set = set(q["relevant_chunks"])
        if not rel_set: continue
        
        # Calculation for F1@10
        toplist_at_k = toplist[:top_k]
        tp = len(rel_set.intersection(set(toplist_at_k)))
        prec = tp / len(toplist_at_k) if toplist_at_k else 0.0
        rec = tp / len(rel_set)
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        all_f1.append(f1)
        
        # Rank extraction for breakdown
        ranks = {cid: idx + 1 for idx, cid in enumerate(toplist)}
        r_list = [ranks.get(cid, 1000000) for cid in rel_set]
        
        # Categorize query
        q_type = q.get("query_type", "single_chunk")
        if "single" in q_type:
            single.append(r_list)
        else:
            multi.append((r_list, len(rel_set)))
            
    total_time = time.perf_counter() - start_time
    print("", file=sys.stderr)  # newline after progress
    out = {}
    
    # --- SINGLE CHUNK METRICS ---
    if single:
        # MRR@10
        mrr_values = []
        for rl in single:
            best_rank = min(rl)
            if best_rank <= 10:
                mrr_values.append(1.0 / best_rank)
            else:
                mrr_values.append(0.0)
        out["single_MRR@10"] = float(np.mean(mrr_values))
        
        # Success Rate @ K (Recall@K for single doc)
        out["single_R@1"] = float(np.mean([any(r <= 1 for r in rl) for rl in single]))
        out["single_R@5"] = float(np.mean([any(r <= 5 for r in rl) for rl in single]))
        out["single_R@10"] = float(np.mean([any(r <= 10 for r in rl) for rl in single]))
        
    # --- MULTI CHUNK METRICS ---
    if multi:
        # Coverage@K: Fraction of relevant docs retrieved in top K
        out["multi_Cov@5"] = float(np.mean([sum(1 for r in rl if r <= 5) / n for rl, n in multi]))
        out["multi_Cov@10"] = float(np.mean([sum(1 for r in rl if r <= 10) / n for rl, n in multi]))
        
        # MaxRecall@K: Did we get AT LEAST ONE relevant doc in top K?
        out["multi_MaxR@5"] = float(np.mean([any(r <= 5 for r in rl) for rl, n in multi]))
        out["multi_MaxR@10"] = float(np.mean([any(r <= 10 for r in rl) for rl, n in multi]))
        
    # --- OVERALL METRICS ---
    out["overall_F1@10"] = float(np.mean(all_f1)) if all_f1 else 0.0
    out["total_time_s"] = total_time
    out["num_single"] = len(single)
    out["num_multi"] = len(multi)
    return out


class AgrerankModel:
    """
    AgreRank implementation:
    - Stage 1: Consensus Seeding  - strict AND threshold
    - Stage 2: Query Expansion via Weighted Centroid 
    - Stage 3: Geometric Gating 
    - Stage 4: Neural Refinement 
    """
    
    def __init__(self, args):
        self.args = args
        
        # Hyperparameters 
        self.tau = args.tau              # Consensus threshold (default 0.8)
        self.alpha = args.alpha          # Expansion interpolation (default 0.65)
        self.beta = args.beta            # Drift floor (default 0.5)
        self.M = args.rerank_depth       # Re-rank depth (default 200)
        self.gamma = args.gamma          # CE fusion weight (default 0.5)
        self.min_seeds = args.min_seeds  # Minimum seeds before fallback (default 5)
        self.max_seeds = args.max_seeds  # Maximum anchor count (default 15)
        
        print("Loading Corpus...", file=sys.stderr)
        self.chunk_ids, self.chunk_text = load_master(args.chunks)
        print(f"  Loaded {len(self.chunk_ids)} chunks", file=sys.stderr)
        
        print("Loading Embeddings...", file=sys.stderr)
        self.E = np.load(args.embeddings).astype(np.float32)
        if not np.allclose(np.linalg.norm(self.E, axis=1).mean(), 1.0, atol=1e-2):
            self.E = l2n(self.E)
        
        d = self.E.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(self.E)
        
        self.id2chunk = {i: cid for i, cid in enumerate(self.chunk_ids)}
        self.chunk2id = {cid: i for i, cid in enumerate(self.chunk_ids)}
        
        print("Building BM25...", file=sys.stderr)
        tokenized_corpus = [toks(text) for text in self.chunk_text.values()]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        self.enc = OAI()
        
        print(f"Loading Cross-Encoder: {MODEL_CROSS_ENCODER}...", file=sys.stderr)
        self.reranker = CrossEncoder(MODEL_CROSS_ENCODER)
        
        print("Model initialized.", file=sys.stderr)

    def _compute_rrf_score(self, rank1: int, rank2: int, k: float = 60.0) -> float:
        """RRF score for fallback when consensus yields too few seeds."""
        return 1.0 / (k + rank1) + 1.0 / (k + rank2)

    def search(self, query_text: str) -> List[Tuple[str, float]]:
        """
        AgreRank search pipeline as described in paper Sections 3.1-3.5.
        """
        q_vec = self.enc.encode([safe_text(query_text)])[0]
        
        # =====================================================================
        # Initial Retrieval: Get candidates from both sparse and dense
        # =====================================================================
        
        # Dense retrieval
        _, d_idxs = self.index.search(q_vec[None, :], self.args.candidates)
        d_cids = {self.id2chunk[i] for i in d_idxs[0] if i != -1}

        # Sparse retrieval (BM25)
        s_scores_all = self.bm25.get_scores(toks(query_text))
        s_indices = np.argsort(s_scores_all)[::-1][:self.args.candidates]
        s_cids = {self.chunk_ids[i] for i in s_indices if s_scores_all[i] > 0}
        
        # Union of candidates
        cand_cids = list(d_cids | s_cids)
        if not cand_cids:
            return []
        
        cand_indices = np.array([self.chunk2id[cid] for cid in cand_cids])
        doc_vecs = self.E[cand_indices]

        # =====================================================================
        #Percentile Normalization
        # =====================================================================
        
        d_scores_raw = q_vec @ doc_vecs.T  # Dense cosine similarity
        s_scores_raw = s_scores_all[cand_indices]  # BM25 scores
        
        # Equation (1): Percentile normalization
        d_norm = percentile_normalize(d_scores_raw)
        s_norm = percentile_normalize(s_scores_raw)
        
        # =====================================================================
        # Equation (2): D_seed = {d : Φ(s_sp) > τ AND Φ(s_de) > τ}
        # =====================================================================
        
        consensus_mask = (d_norm > self.tau) & (s_norm > self.tau)
        consensus_indices = np.where(consensus_mask)[0]
        
        # Fallback: If fewer than min_seeds, use top-10 from RRF
        if len(consensus_indices) < self.min_seeds:
            d_ranks = stats.rankdata(-d_scores_raw, method='ordinal')
            s_ranks = stats.rankdata(-s_scores_raw, method='ordinal')
            rrf_scores = np.array([
                self._compute_rrf_score(d_ranks[i], s_ranks[i]) 
                for i in range(len(cand_cids))
            ])
            consensus_indices = np.argsort(rrf_scores)[-10:]
            # Use uniform weights for fallback
            seed_weights = np.ones(len(consensus_indices))
        else:
            # Limit to max_seeds (paper: ≤15)
            if len(consensus_indices) > self.max_seeds:
                # Select top seeds by consensus strength
                consensus_strengths = d_norm[consensus_indices] * s_norm[consensus_indices]
                top_seed_idx = np.argsort(consensus_strengths)[-self.max_seeds:]
                consensus_indices = consensus_indices[top_seed_idx]
            
            # Equation (3) weights: w_d = Φ(s_sp) × Φ(s_de)
            seed_weights = d_norm[consensus_indices] * s_norm[consensus_indices]
        
        # =====================================================================
        # Equation (3): v_q' = α*v_q + (1-α) * weighted_centroid
        # =====================================================================
        
        seed_vecs = doc_vecs[consensus_indices]
        
        # Normalize weights to sum to 1
        weight_sum = seed_weights.sum()
        if weight_sum > 0:
            norm_weights = seed_weights / weight_sum
        else:
            norm_weights = np.ones(len(seed_weights)) / len(seed_weights)
        
        # Weighted centroid
        weighted_centroid = np.sum(seed_vecs * norm_weights[:, None], axis=0)
        
        # Expanded query vector
        exp_q = self.alpha * q_vec + (1 - self.alpha) * weighted_centroid
        exp_q = l2n(exp_q.reshape(1, -1))[0]
        
        # Re-query ANN index with expanded query (paper mentions this)
        # For simplicity, we score existing candidates with expanded query
        # A full implementation would re-query the index
        
        # =====================================================================
        # Equation (4): G(d) = β + (1-β) * max(0, cos(v_q, v_d))
        # Equation (5): S_stage3 = cos(v_q', v_d) × G(d)
        # =====================================================================
        
        # Score with expanded query
        prf_scores_raw = exp_q @ doc_vecs.T
        
        # Geometric gate based on alignment with ORIGINAL query
        alignment = np.maximum(0, q_vec @ doc_vecs.T)
        gate = self.beta + (1 - self.beta) * alignment
        
        # Gated scores
        s_stage3 = prf_scores_raw * gate
        
        # Percentile normalize for fusion (end of Section 3.4)
        s_stage3_norm = percentile_normalize(s_stage3)
        
        # =====================================================================
        # Select top M candidates, apply cross-encoder
        # Equation (6): S_final = γ*CE(q,d) + (1-γ)*Φ(S_stage3)
        # =====================================================================
        
        # Select top M candidates for cross-encoder
        ce_pool_size = min(self.M, len(cand_cids))
        ce_indices = np.argsort(s_stage3)[-ce_pool_size:]
        
        # Prepare pairs for cross-encoder
        ce_pairs = []
        ce_idx_map = []  # Maps CE result index to candidate index
        q_clean = safe_text(query_text)
        
        for idx in ce_indices:
            doc_id = cand_cids[idx]
            txt = safe_text(self.chunk_text.get(doc_id, ""))
            if q_clean and txt:
                ce_pairs.append([q_clean, txt])
                ce_idx_map.append(idx)
        
        # Initialize final scores with stage 3 normalized scores
        final_scores = np.copy(s_stage3_norm)
        
        if ce_pairs:
            try:
                # Get cross-encoder scores
                ce_scores_raw = self.reranker.predict(ce_pairs, show_progress_bar=False)
                
                if ce_scores_raw.size > 0:
                    # Normalize CE scores to [0, 1]
                    ce_min, ce_max = ce_scores_raw.min(), ce_scores_raw.max()
                    if ce_max - ce_min > 1e-9:
                        ce_scores_norm = (ce_scores_raw - ce_min) / (ce_max - ce_min)
                    else:
                        ce_scores_norm = np.zeros_like(ce_scores_raw)
                    
                    # Equation (6): Fusion
                    for i, idx in enumerate(ce_idx_map):
                        final_scores[idx] = (
                            self.gamma * ce_scores_norm[i] + 
                            (1 - self.gamma) * s_stage3_norm[idx]
                        )
            except Exception as e:
                print(f"CE error: {e}", file=sys.stderr)
        
        # Return sorted results
        results = [(cand_cids[i], float(final_scores[i])) for i in range(len(cand_cids))]
        return sorted(results, key=lambda x: x[1], reverse=True)


def main():
    parser = argparse.ArgumentParser(description="AgreRank Evaluation (Paper-Accurate Implementation)")
    
    # Data paths
    parser.add_argument("--chunks", type=Path, required=True, help="Path to corpus JSONL")
    parser.add_argument("--embeddings", type=Path, required=True, help="Path to corpus embeddings .npy")
    parser.add_argument("--qset", type=Path, required=True, help="Path to query set JSON")
    parser.add_argument("--out", type=Path, required=True, help="Output path for results JSON")
    
    # Retrieval
    parser.add_argument("--candidates", type=int, default=500, help="Initial candidates per retriever")
    
    # AgreRank hyperparameters (Table 1 in paper)
    parser.add_argument("--tau", type=float, default=0.8, help="Consensus threshold τ (Stage 1)")
    parser.add_argument("--min_seeds", type=int, default=5, help="Minimum seeds before RRF fallback")
    parser.add_argument("--max_seeds", type=int, default=15, help="Maximum anchor count (Stage 2)")
    parser.add_argument("--alpha", type=float, default=0.65, help="Expansion interpolation α (Stage 2)")
    parser.add_argument("--beta", type=float, default=0.5, help="Drift floor β (Stage 3)")
    parser.add_argument("--rerank_depth", type=int, default=200, help="Re-rank depth M (Stage 4)")
    parser.add_argument("--gamma", type=float, default=0.5, help="CE fusion weight γ (Stage 4)")
    
    args = parser.parse_args()
    
    print("=" * 70, file=sys.stderr)
    print("AgreRank Evaluation:", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"Hyperparameters:", file=sys.stderr)
    print(f"  τ (consensus threshold) = {args.tau}", file=sys.stderr)
    print(f"  α (expansion weight)    = {args.alpha}", file=sys.stderr)
    print(f"  β (drift floor)         = {args.beta}", file=sys.stderr)
    print(f"  M (rerank depth)        = {args.rerank_depth}", file=sys.stderr)
    print(f"  γ (CE fusion weight)    = {args.gamma}", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    
    system = AgrerankModel(args)
    qset = load_qset(args.qset)
    
    print(f"\n--- Evaluating AgreRank on {len(qset)} queries ---", file=sys.stderr)
    metrics = eval_system(system.search, qset, top_k=10)
    
    print("\n" + "=" * 70, file=sys.stderr)
    print("RESULTS", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:<15}: {v:.4f}", file=sys.stderr)
        else:
            print(f"{k:<15}: {v}", file=sys.stderr)
    
    # Save results
    output = {
        "hyperparameters": {
            "tau": args.tau,
            "alpha": args.alpha,
            "beta": args.beta,
            "M": args.rerank_depth,
            "gamma": args.gamma,
            "candidates": args.candidates,
        },
        "metrics": metrics
    }
    
    with args.out.open("w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()


#  export JAVA_HOME=$(/usr/libexec/java_home -v 21) 
# python -i vectorrag/eval/retrievers/eval_agrerank.py  \
#     --chunks data/master_in_embedding_order.jsonl \
#     --embeddings data/embeddings/fixed250/corpus_emb.npy \
#     --qset data/qqset.json \
#     --out data/results_agrerank_corrected.json \
#     --candidates 500 \
#     --tau 0.8 \
#     --alpha 0.65 \
#     --beta 0.5 \
#     --rerank_depth 200 \
#     --gamma 0.5


#  export JAVA_HOME=$(/usr/libexec/java_home -v 21)    
# python -i vectorrag/eval/retrievers/eval_agrerank.py \
#     --chunks data/master_in_embedding_order.jsonl \
#     --embeddings data/embeddings/fixed250/corpus_emb.npy \
#     --qset data/qqset.json \
#     --out data/results_with_timing/results_agrerank_diagnostic.json \
#     --candidates 500 \
#     --cross_encoder_k 200

