# Methods: Dense, BM25, RRF, RRF+CE
import argparse
import json
import time
from pathlib import Path
import numpy as np
import sys
import re
import os
from typing import List, Dict, Tuple, Any

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
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

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

def load_master(path: Path) -> Tuple[List[str], Dict[str, str]]:
    ids, txt = [], {}
    print(f"Loading corpus from {path}...", file=sys.stderr)
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            o = json.loads(line)
            cid = o.get("chunk_id") or o.get("id")
            ids.append(cid)
            txt[cid] = o["text"]
    print(f"  Loaded {len(ids)} chunks", file=sys.stderr)
    return ids, txt

def load_qset(path: Path) -> List[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


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
            except Exception as e:
                print(f"Embedding error: {e}", file=sys.stderr)
                out.append(np.zeros((len(batch), 3072), dtype=np.float32))
        return l2n(np.vstack(out)).astype(np.float32)


def eval_system(search_fn, qset: List[dict], top_k: int) -> Dict[str, float]:
    """Evaluate with precision metrics for noise resilience."""
    single, multi, all_f1 = [], [], []
    all_p5, all_p10 = [], []
    start = time.perf_counter()
    
    for i, q in enumerate(qset):
        print(f"Processing {i+1}/{len(qset)}...", file=sys.stderr, end='\r')
        res = search_fn(q["query_text"])
        toplist = [cid for cid, _ in res]
        rel_set = set(q["relevant_chunks"])
        if not rel_set: continue
        
        # Precision@K
        top5 = set(toplist[:5])
        top10 = set(toplist[:10])
        p5 = len(rel_set.intersection(top5)) / 5
        p10 = len(rel_set.intersection(top10)) / 10
        all_p5.append(p5)
        all_p10.append(p10)
        
        # F1@10
        tp = len(rel_set.intersection(top10))
        prec = tp / 10
        rec = tp / len(rel_set)
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        all_f1.append(f1)
        
        # Ranks
        ranks = {cid: idx + 1 for idx, cid in enumerate(toplist)}
        r_list = [ranks.get(cid, 1000000) for cid in rel_set]
        
        q_type = q.get("query_type", "single_chunk")
        if "single" in q_type:
            single.append(r_list)
        else:
            multi.append((r_list, len(rel_set)))
            
    total_time = time.perf_counter() - start
    print("", file=sys.stderr)
    out = {}
    
    # Precision metrics
    out["P@5"] = float(np.mean(all_p5))
    out["P@10"] = float(np.mean(all_p10))
    
    # Single metrics
    if single:
        mrr_values = []
        for rl in single:
            best_rank = min(rl)
            if best_rank <= 10:
                mrr_values.append(1.0 / best_rank)
            else:
                mrr_values.append(0.0)
        out["single_MRR@10"] = float(np.mean(mrr_values))
        out["single_R@1"] = float(np.mean([any(r <= 1 for r in rl) for rl in single]))
        out["single_R@5"] = float(np.mean([any(r <= 5 for r in rl) for rl in single]))
        out["single_R@10"] = float(np.mean([any(r <= 10 for r in rl) for rl in single]))
    
    # Multi metrics
    if multi:
        out["multi_Cov@5"] = float(np.mean([sum(1 for r in rl if r <= 5) / n for rl, n in multi]))
        out["multi_Cov@10"] = float(np.mean([sum(1 for r in rl if r <= 10) / n for rl, n in multi]))
        out["multi_MaxR@5"] = float(np.mean([any(r <= 5 for r in rl) for rl, n in multi]))
        out["multi_MaxR@10"] = float(np.mean([any(r <= 10 for r in rl) for rl, n in multi]))
        
    out["overall_F1@10"] = float(np.mean(all_f1)) if all_f1 else 0.0
    out["total_time_s"] = total_time
    out["num_single"] = len(single)
    out["num_multi"] = len(multi)
    
    return out


class BaselinesModel:
    """All baseline methods: Dense, BM25, RRF, RRF+CE."""
    
    def __init__(self, args):
        self.args = args
        self.chunk_ids, self.chunk_text = load_master(args.chunks)
        
        print("Loading embeddings...", file=sys.stderr)
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
        
        print(f"Loading Cross-Encoder: {CE_MODEL}...", file=sys.stderr)
        self.reranker = CrossEncoder(CE_MODEL)
        print("Model initialized.", file=sys.stderr)

    def search_dense(self, query_text: str) -> List[Tuple[str, float]]:
        """Dense retrieval only."""
        q_vec = self.enc.encode([safe_text(query_text)])[0]
        scores, indices = self.index.search(q_vec[None, :], self.args.candidates)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                results.append((self.id2chunk[idx], float(scores[0][i])))
        return sorted(results, key=lambda x: x[1], reverse=True)

    def search_bm25(self, query_text: str) -> List[Tuple[str, float]]:
        """BM25 retrieval only."""
        scores = self.bm25.get_scores(toks(query_text))
        top_indices = np.argsort(scores)[::-1][:self.args.candidates]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.chunk_ids[idx], float(scores[idx])))
        return results

    def search_rrf(self, query_text: str) -> List[Tuple[str, float]]:
        """RRF fusion of Dense + BM25."""
        q_vec = self.enc.encode([safe_text(query_text)])[0]
        
        # Dense
        _, d_indices = self.index.search(q_vec[None, :], self.args.candidates)
        d_cids = {self.id2chunk[i]: rank for rank, i in enumerate(d_indices[0]) if i != -1}
        
        # BM25
        s_scores = self.bm25.get_scores(toks(query_text))
        s_top_indices = np.argsort(s_scores)[::-1][:self.args.candidates]
        s_cids = {self.chunk_ids[i]: rank for rank, i in enumerate(s_top_indices) if s_scores[i] > 0}
        
        # RRF
        all_cids = set(d_cids.keys()) | set(s_cids.keys())
        k = 60.0
        rrf_scores = {}
        for cid in all_cids:
            d_rank = d_cids.get(cid, 1000) + 1
            s_rank = s_cids.get(cid, 1000) + 1
            rrf_scores[cid] = 1.0 / (k + d_rank) + 1.0 / (k + s_rank)
        
        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    def search_rrf_ce(self, query_text: str) -> List[Tuple[str, float]]:
        """RRF + Cross-Encoder."""
        q_vec = self.enc.encode([safe_text(query_text)])[0]
        
        # Dense
        _, d_indices = self.index.search(q_vec[None, :], self.args.candidates)
        d_cids = {self.id2chunk[i]: rank for rank, i in enumerate(d_indices[0]) if i != -1}
        
        # BM25
        s_scores = self.bm25.get_scores(toks(query_text))
        s_top_indices = np.argsort(s_scores)[::-1][:self.args.candidates]
        s_cids = {self.chunk_ids[i]: rank for rank, i in enumerate(s_top_indices) if s_scores[i] > 0}
        
        # RRF
        all_cids = list(set(d_cids.keys()) | set(s_cids.keys()))
        k = 60.0
        rrf_scores = np.zeros(len(all_cids))
        for i, cid in enumerate(all_cids):
            d_rank = d_cids.get(cid, 1000) + 1
            s_rank = s_cids.get(cid, 1000) + 1
            rrf_scores[i] = 1.0 / (k + d_rank) + 1.0 / (k + s_rank)
        
        # Cross-encoder on top candidates
        rerank_depth = min(len(all_cids), self.args.rerank_depth)
        top_indices = np.argsort(rrf_scores)[-rerank_depth:]
        
        pairs = []
        valid_indices = []
        q_clean = safe_text(query_text)
        
        for idx in top_indices:
            cid = all_cids[idx]
            txt = safe_text(self.chunk_text.get(cid, ""))
            if q_clean and txt:
                pairs.append([q_clean, txt])
                valid_indices.append(idx)
        
        final_scores = rrf_scores.copy()
        
        if pairs:
            ce_scores = self.reranker.predict(pairs, show_progress_bar=False)
            ce_min, ce_max = ce_scores.min(), ce_scores.max()
            if ce_max - ce_min > 1e-9:
                ce_norm = (ce_scores - ce_min) / (ce_max - ce_min)
            else:
                ce_norm = np.zeros_like(ce_scores)
            
            for i, idx in enumerate(valid_indices):
                final_scores[idx] = 1000.0 + ce_norm[i]
        
        results = [(all_cids[i], float(final_scores[i])) for i in range(len(all_cids))]
        return sorted(results, key=lambda x: x[1], reverse=True)


def main():
    parser = argparse.ArgumentParser(description="Baselines Evaluation on Noisy Corpus")
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--qset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--candidates", type=int, default=500)
    parser.add_argument("--rerank_depth", type=int, default=200)
    parser.add_argument("--method", type=str, default="all",
                       choices=["all", "dense", "bm25", "rrf", "rrf_ce"])
    args = parser.parse_args()
    
    args.out.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70, file=sys.stderr)
    print("Baselines Evaluation (Noisy Corpus)", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    
    model = BaselinesModel(args)
    qset = load_qset(args.qset)
    
    methods = {
        "dense": model.search_dense,
        "bm25": model.search_bm25,
        "rrf": model.search_rrf,
        "rrf_ce": model.search_rrf_ce,
    }
    
    if args.method == "all":
        methods_to_run = methods
    else:
        methods_to_run = {args.method: methods[args.method]}
    
    all_results = {}
    
    for name, search_fn in methods_to_run.items():
        print(f"\n{'='*70}", file=sys.stderr)
        print(f"Evaluating: {name.upper()}", file=sys.stderr)
        print(f"{'='*70}", file=sys.stderr)
        
        metrics = eval_system(search_fn, qset, top_k=10)
        all_results[name] = metrics
        
        out_path = args.out / f"results_{name}.json"
        with out_path.open("w") as f:
            json.dump({"method": name, "metrics": metrics}, f, indent=2)
        
        print(f"\nResults for {name}:", file=sys.stderr)
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k:<15}: {v:.4f}", file=sys.stderr)
            else:
                print(f"  {k:<15}: {v}", file=sys.stderr)
    
    # Save combined
    combined_path = args.out / "results_all.json"
    with combined_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print comparison
    print("\n" + "=" * 90, file=sys.stderr)
    print("COMPARISON TABLE", file=sys.stderr)
    print("=" * 90, file=sys.stderr)
    
    header = f"{'Method':<10} {'P@5':>8} {'P@10':>8} {'MRR@10':>10} {'R@10':>8} {'Cov@10':>10}"
    print(header, file=sys.stderr)
    print("-" * 90, file=sys.stderr)
    
    for name, metrics in all_results.items():
        row = f"{name:<10}"
        row += f" {metrics.get('P@5', 0):>8.4f}"
        row += f" {metrics.get('P@10', 0):>8.4f}"
        row += f" {metrics.get('single_MRR@10', 0):>10.4f}"
        row += f" {metrics.get('single_R@10', 0):>8.4f}"
        row += f" {metrics.get('multi_Cov@10', 0):>10.4f}"
        print(row, file=sys.stderr)
    
    print(f"\nResults saved to: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

# Usage:
# python vectorrag/eval_baseline_noisy.py\
#     --chunks data/master_in_embedding_order.jsonl \
#     --embeddings data/embeddings/fixed250/corpus_emb.npy \
#     --qset data/qset.json \
#     --out data/results_baseline_clean.json \
#     --method all


# python vectorrag/eval_baseline_noisy.py\
#     --chunks data/noisy_corpus_10k.jsonl \
#     --embeddings data/noisy_embeddings_10k.npy \
#     --qset data/qset.json \
#     --out data/results_baselines_noisy_10k/ \
#     --candidates 500 \
#     --rerank_depth 200 \
#     --method all


# python vectorrag/eval_agrerank_noisy.py \
#     --chunks data/master_in_embedding_order.jsonl \
#     --embeddings data/embeddings/fixed250/corpus_emb.npy \
#     --qset data/qset.json \
#     --out data/results_agrerank.json \
#     --candidates 500 \
    


    