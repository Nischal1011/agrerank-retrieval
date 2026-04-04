# Questions answered:
# 1. Where do relevant documents land in score space?
# 2. What is the precision lift of each bucket vs random selection?
# """
import argparse
import json
from pathlib import Path
import numpy as np
import re
import os
import sys
from typing import Dict, List, Tuple, Any
from collections import defaultdict

# --- Dependencies ---
try:
    import faiss
except ImportError:
    faiss = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

try:
    from scipy import stats
except ImportError:
    stats = None

# --- Config ---
MODEL_EMBEDDING = "text-embedding-3-large"

# --- Utils ---
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)*")
STOP = set("the a an and or for to of in on at by from with as is are was were be been being this that these those which who whose where when".split())


def toks(s: str) -> List[str]:
    if not s:
        return []
    out = [w.lower() for w in _TOKEN_RE.findall(s)]
    return [w for w in out if w not in STOP and len(w) > 1]


def safe_text(text: Any) -> str:
    try:
        return str(text or "").strip()
    except Exception:
        return ""


def l2n(V: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if V.ndim == 1:
        n = np.linalg.norm(V)
        return V / max(n, eps)
    n = np.linalg.norm(V, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return V / n


def percentile_normalize(scores: np.ndarray) -> np.ndarray:
    """Forces scores into Uniform[0,1] distribution based on rank."""
    if scores is None or len(scores) < 2 or stats is None:
        return np.zeros_like(scores, dtype=np.float32)
    return stats.rankdata(scores, 'average') / len(scores)


class OAI:
    def __init__(self, bs: int = 128):
        if OpenAI is None:
            sys.exit("Error: openai library not available.")
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
            batch = [safe_text(t) if safe_text(t) else " " for t in texts[i:i + self.bs]]
            try:
                resp = self.c.embeddings.create(model=MODEL_EMBEDDING, input=batch)
                out.append(np.array([d.embedding for d in resp.data], dtype=np.float32))
            except Exception as e:
                print(f"Embedding error: {e}", file=sys.stderr)
                out.append(np.zeros((len(batch), 3072), dtype=np.float32))
        return l2n(np.vstack(out)).astype(np.float32)


def load_master(path: Path) -> Tuple[List[str], Dict[str, str]]:
    ids, txt = [], {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            cid = o.get("chunk_id") or o.get("id")
            ids.append(cid)
            txt[cid] = o["text"]
    return ids, txt


def load_qset(path: Path) -> List[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


class RetrievalModel:
    """Simplified model for consensus analysis - no cross-encoder needed."""
    
    def __init__(self, chunks_path: Path, embeddings_path: Path, candidates: int = 500):
        self.candidates = candidates

        if faiss is None:
            sys.exit("Error: faiss is required but not available.")
        if BM25Okapi is None:
            sys.exit("Error: rank_bm25 is required but not available.")

        print("Loading Corpus...", file=sys.stderr)
        self.chunk_ids, self.chunk_text = load_master(chunks_path)
        print(f"  Loaded {len(self.chunk_ids)} chunks", file=sys.stderr)

        print("Loading Embeddings...", file=sys.stderr)
        self.E = np.load(embeddings_path).astype(np.float32)
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

        print("Initializing OpenAI encoder...", file=sys.stderr)
        self.enc = OAI()

    def get_scores_for_query(self, query_text: str) -> Dict[str, Tuple[float, float]]:
        """
        Returns dict mapping chunk_id -> (dense_norm, sparse_norm)
        for all candidates retrieved by either method.
        Scores are percentile-normalized within each query's candidate pool.
        """
        q_vec = self.enc.encode([safe_text(query_text)])[0]

        # Dense retrieval
        _, d_idxs = self.index.search(q_vec[None, :], self.candidates)
        d_cids = {self.id2chunk[i] for i in d_idxs[0] if i != -1}

        # Sparse retrieval (BM25)
        tokens = toks(query_text)
        s_scores_all = self.bm25.get_scores(tokens)
        s_indices = np.argsort(s_scores_all)[::-1][:self.candidates]
        s_cids = {self.chunk_ids[i] for i in s_indices if s_scores_all[i] > 0}

        # Union of candidates
        cand_cids = list(d_cids | s_cids)
        if not cand_cids:
            return {}

        cand_indices = np.array([self.chunk2id[cid] for cid in cand_cids])
        doc_vecs = self.E[cand_indices]

        # Raw scores
        d_scores_raw = (q_vec @ doc_vecs.T)
        s_scores_raw = s_scores_all[cand_indices]

        # Normalized scores (percentile -> ~Uniform[0,1])
        d_norm = percentile_normalize(d_scores_raw)
        s_norm = percentile_normalize(s_scores_raw)

        return {
            cid: (float(d_norm[i]), float(s_norm[i]))
            for i, cid in enumerate(cand_cids)
        }


def classify_bucket(d_score: float, s_score: float, thresh: float = 0.8) -> str:
    """Classify a document into one of four buckets based on scores."""
    d_high = d_score >= thresh
    s_high = s_score >= thresh
    
    if d_high and s_high:
        return "both_high"
    elif d_high and not s_high:
        return "dense_only"
    elif s_high and not d_high:
        return "sparse_only"
    else:
        return "both_low"


def analyze_relevant_document_distribution(
    model: RetrievalModel,
    qset: List[dict],
    thresh: float = 0.8
) -> Dict[str, Any]:
    """
    Analysis 1: Where do relevant documents land in score space?
    
    For each relevant document, determine which bucket it falls into.
    This answers: "Do relevant docs tend to have sparse-dense consensus?"
    """
    bucket_counts = defaultdict(int)
    total_relevant = 0
    
    relevant_doc_details = []
    
    for i, q in enumerate(qset):
        print(f"Processing query {i+1}/{len(qset)}: {q.get('query_id', i)}", file=sys.stderr)
        
        rel_set = set(q.get("relevant_chunks", []))
        if not rel_set:
            continue
        
        scores = model.get_scores_for_query(q["query_text"])
        
        for rel_cid in rel_set:
            total_relevant += 1
            
            if rel_cid not in scores:
                bucket_counts["not_retrieved"] += 1
                relevant_doc_details.append({
                    "query_id": q.get("query_id", i),
                    "chunk_id": rel_cid,
                    "bucket": "not_retrieved",
                    "dense_score": None,
                    "sparse_score": None
                })
                continue
            
            d_score, s_score = scores[rel_cid]
            bucket = classify_bucket(d_score, s_score, thresh)
            bucket_counts[bucket] += 1
            
            relevant_doc_details.append({
                "query_id": q.get("query_id", i),
                "chunk_id": rel_cid,
                "bucket": bucket,
                "dense_score": round(d_score, 3),
                "sparse_score": round(s_score, 3)
            })
    
    results = {
        "total_relevant_docs": total_relevant,
        "threshold": thresh,
        "bucket_counts": dict(bucket_counts),
        "bucket_percentages": {
            k: round(v / total_relevant * 100, 1) if total_relevant > 0 else 0
            for k, v in bucket_counts.items()
        },
        "relevant_doc_details": relevant_doc_details
    }
    
    return results


def analyze_bucket_precision(
    model: RetrievalModel,
    qset: List[dict],
    thresh: float = 0.8
) -> Dict[str, Any]:
    """
    Analysis 2: What is the precision of each bucket?
    
    Lift is computed relative to RANDOM BASELINE (total_rel / total_cand).
    This is more defensible than comparing to any single bucket.
    """
    bucket_relevant = defaultdict(int)
    bucket_total = defaultdict(int)
    
    for i, q in enumerate(qset):
        print(f"Processing query {i+1}/{len(qset)}: {q.get('query_id', i)}", file=sys.stderr)
        
        rel_set = set(q.get("relevant_chunks", []))
        scores = model.get_scores_for_query(q["query_text"])
        
        for cid, (d_score, s_score) in scores.items():
            bucket = classify_bucket(d_score, s_score, thresh)
            bucket_total[bucket] += 1
            
            if cid in rel_set:
                bucket_relevant[bucket] += 1
    
    # Random baseline: precision if you picked randomly from candidate pool
    total_relevant = sum(bucket_relevant.values())
    total_candidates = sum(bucket_total.values())
    random_precision = total_relevant / total_candidates if total_candidates > 0 else 0
    
    # Compute precision and lift for each bucket
    bucket_precision = {}
    for bucket in ["both_high", "dense_only", "sparse_only", "both_low"]:
        total = bucket_total[bucket]
        relevant = bucket_relevant[bucket]
        precision = relevant / total if total > 0 else 0
        
        # Lift vs random baseline
        lift = precision / random_precision if random_precision > 0 else 0
        
        bucket_precision[bucket] = {
            "relevant": relevant,
            "total": total,
            "precision": round(precision, 6),
            "precision_pct": round(precision * 100, 3),
            "lift_vs_random": round(lift, 1)
        }
    
    return {
        "threshold": thresh,
        "bucket_precision": bucket_precision,
        "total_candidates": total_candidates,
        "total_relevant": total_relevant,
        "random_baseline_precision": round(random_precision, 6),
        "random_baseline_precision_pct": round(random_precision * 100, 4)
    }


def analyze_score_distributions(
    model: RetrievalModel,
    qset: List[dict]
) -> Dict[str, Any]:
    """
    Analysis 3: Score distribution statistics for relevant vs non-relevant docs.
    """
    rel_dense_scores = []
    rel_sparse_scores = []
    nonrel_dense_scores = []
    nonrel_sparse_scores = []
    
    for i, q in enumerate(qset):
        print(f"Processing query {i+1}/{len(qset)}: {q.get('query_id', i)}", file=sys.stderr)
        
        rel_set = set(q.get("relevant_chunks", []))
        scores = model.get_scores_for_query(q["query_text"])
        
        for cid, (d_score, s_score) in scores.items():
            if cid in rel_set:
                rel_dense_scores.append(d_score)
                rel_sparse_scores.append(s_score)
            else:
                nonrel_dense_scores.append(d_score)
                nonrel_sparse_scores.append(s_score)
    
    return {
        "relevant": {
            "count": len(rel_dense_scores),
            "dense_mean": round(np.mean(rel_dense_scores), 3) if rel_dense_scores else 0,
            "dense_std": round(np.std(rel_dense_scores), 3) if rel_dense_scores else 0,
            "sparse_mean": round(np.mean(rel_sparse_scores), 3) if rel_sparse_scores else 0,
            "sparse_std": round(np.std(rel_sparse_scores), 3) if rel_sparse_scores else 0,
        },
        "non_relevant": {
            "count": len(nonrel_dense_scores),
            "dense_mean": round(np.mean(nonrel_dense_scores), 3) if nonrel_dense_scores else 0,
            "dense_std": round(np.std(nonrel_dense_scores), 3) if nonrel_dense_scores else 0,
            "sparse_mean": round(np.mean(nonrel_sparse_scores), 3) if nonrel_sparse_scores else 0,
            "sparse_std": round(np.std(nonrel_sparse_scores), 3) if nonrel_sparse_scores else 0,
        }
    }


def print_results(
    dist_results: Dict[str, Any],
    precision_results: Dict[str, Any],
    score_results: Dict[str, Any]
):
    """Print results"""
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    # Analysis 1: Where relevant docs land
    print("\n### Analysis 1: Distribution of Relevant Documents Across Buckets ###")
    print(f"Total relevant documents: {dist_results['total_relevant_docs']}")
    print(f"Threshold: {dist_results['threshold']}")
    print()
    print(f"{'Bucket':<15} {'Count':>8} {'Percentage':>12}")
    print("-" * 40)
    
    for bucket in ["both_high", "dense_only", "sparse_only", "both_low", "not_retrieved"]:
        count = dist_results['bucket_counts'].get(bucket, 0)
        pct = dist_results['bucket_percentages'].get(bucket, 0)
        print(f"{bucket:<15} {count:>8} {pct:>11.1f}%")
    
    # Analysis 2: Bucket precision with RANDOM BASELINE
    print("\n### Analysis 2: Precision by Bucket (Lift vs Random Baseline) ###")
    print(f"Total candidates evaluated: {precision_results['total_candidates']}")
    print(f"Total relevant in pool: {precision_results['total_relevant']}")
    print(f"Random baseline precision: {precision_results['random_baseline_precision_pct']:.4f}%")
    print()
    print(f"{'Bucket':<15} {'Relevant':>10} {'Total':>10} {'Precision':>12} {'Lift':>10}")
    print("-" * 65)
    
    for bucket in ["both_high", "dense_only", "sparse_only", "both_low"]:
        data = precision_results['bucket_precision'][bucket]
        print(f"{bucket:<15} {data['relevant']:>10} {data['total']:>10} {data['precision_pct']:>11.3f}% {data['lift_vs_random']:>9.1f}x")
    
    # Analysis 3: Score distributions
    print("\n### Analysis 3: Score Distributions ###")
    print(f"{'Category':<15} {'Count':>8} {'Dense μ':>10} {'Dense σ':>10} {'Sparse μ':>10} {'Sparse σ':>10}")
    print("-" * 70)
    
    for cat in ["relevant", "non_relevant"]:
        data = score_results[cat]
        print(f"{cat:<15} {data['count']:>8} {data['dense_mean']:>10.3f} {data['dense_std']:>10.3f} {data['sparse_mean']:>10.3f} {data['sparse_std']:>10.3f}")
    

    both_high = precision_results['bucket_precision']['both_high']
    print("\n" + "="*70)
    print("KEY FINDING")
    print("="*70)
    print(f"Both-high bucket contains {dist_results['bucket_counts'].get('both_high', 0)} of {dist_results['total_relevant_docs']} relevant docs ({dist_results['bucket_percentages'].get('both_high', 0):.1f}%)")
    print(f"Both-high precision: {both_high['precision_pct']:.2f}%")
    print(f"Lift vs random: {both_high['lift_vs_random']:.1f}x")
    
    if both_high['lift_vs_random'] >= 100:
        print("\n→ TWO ORDERS OF MAGNITUDE lift confirmed!")
    elif both_high['lift_vs_random'] >= 50:
        print("\n→ Nearly two orders of magnitude lift")
    elif both_high['lift_vs_random'] >= 10:
        print("\n→ One order of magnitude lift")
    
   


def main():
    parser = argparse.ArgumentParser(
        description="Consensus analysis with lift vs random baseline"
    )
    parser.add_argument("--chunks", type=Path, required=True,
                        help="Path to corpus JSONL file")
    parser.add_argument("--embeddings", type=Path, required=True,
                        help="Path to embeddings .npy file")
    parser.add_argument("--qset", type=Path, required=True,
                        help="Path to query set JSON file")
    parser.add_argument("--out", type=Path, default=None,
                        help="Path to write full results JSON")
    parser.add_argument("--candidates", type=int, default=500,
                        help="Number of candidates to retrieve per method")
    parser.add_argument("--thresh", type=float, default=0.8,
                        help="Threshold for 'high' score classification")
    
    args = parser.parse_args()
    
    # Load model and data
    model = RetrievalModel(args.chunks, args.embeddings, args.candidates)
    qset = load_qset(args.qset)
    
    # Print qset statistics
    n_single = sum(1 for q in qset if "single" in q.get("query_type", ""))
    n_multi = sum(1 for q in qset if "multi" in q.get("query_type", ""))
    n_relevant = sum(len(q.get("relevant_chunks", [])) for q in qset)
    
    print(f"\n=== Query Set Statistics ===", file=sys.stderr)
    print(f"Total queries: {len(qset)}", file=sys.stderr)
    print(f"  Single-chunk: {n_single}", file=sys.stderr)
    print(f"  Multi-chunk: {n_multi}", file=sys.stderr)
    print(f"Total relevant documents: {n_relevant}", file=sys.stderr)
    
    # Run analyses
    print("\n=== Running Analysis 1: Relevant Document Distribution ===", file=sys.stderr)
    dist_results = analyze_relevant_document_distribution(model, qset, args.thresh)
    
    print("\n=== Running Analysis 2: Bucket Precision (vs Random Baseline) ===", file=sys.stderr)
    precision_results = analyze_bucket_precision(model, qset, args.thresh)
    
    print("\n=== Running Analysis 3: Score Distributions ===", file=sys.stderr)
    score_results = analyze_score_distributions(model, qset)
    
    # Print formatted results
    print_results(dist_results, precision_results, score_results)
    
    # Save full results
    if args.out:
        full_results = {
            "qset_stats": {
                "total_queries": len(qset),
                "single_chunk_queries": n_single,
                "multi_chunk_queries": n_multi,
                "total_relevant_docs": n_relevant,
            },
            "distribution_analysis": dist_results,
            "precision_analysis": precision_results,
            "score_analysis": score_results
        }
        
        # Remove detailed doc list for cleaner output
        full_results["distribution_analysis"].pop("relevant_doc_details", None)
        
        with args.out.open("w") as f:
            json.dump(full_results, f, indent=2)
        print(f"\nFull results saved to: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()


# Usage:
# python vectorrag/eval/sparse_dense_fusion_analysis.py \
#     --chunks data/master_in_embedding_order.jsonl \
#     --embeddings data/embeddings/fixed250/corpus_emb.npy \
#     --qset data/qqset.json \
#     --out results/consensus_analysis_v3.json \
#     --candidates 500 \
#     --thresh 0.8