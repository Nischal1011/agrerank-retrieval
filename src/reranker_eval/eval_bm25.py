"""
BM25 Baseline Retriever (no dense, no fusion)

Usage:
  pip install rank-bm25 numpy
  python bm25_baseline.py \
    --chunks data/master_in_embedding_order.jsonl \
    --qset data/qset.json \
    --out data/results_with_timing/results_bm25.json \
    --topk 10 \
    --k1 1.2 \
    --b 0.75 \
    --neighbor_window 0

Notes:
- If your chunk_ids follow the pattern "paper::section::...<orderNumber>", you can set --neighbor_window>0 to expand each chunk
  with its adjacent neighbors from the same section (e.g., window=1 adds prev/next). Default is 0 (no expansion).
- Metrics & timing match your existing evaluator, but avg_time_ms divides by #evaluated queries (with labels).
"""
from __future__ import annotations
import argparse, json, re, sys, time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

# ----------------- IO -----------------
def load_master(path: Path):
    if path.suffix.lower() == ".jsonl":
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
    else:
        arr = json.loads(path.read_text(encoding="utf-8"))
        ids, txt = [], {}
        for o in arr:
            cid = o.get("chunk_id") or o.get("id")
            ids.append(cid)
            txt[cid] = o["text"]
        return ids, txt


def load_qset(path: Path) -> List[dict]:
    return json.loads(path.read_text(encoding="utf-8"))

# ----------------- tokenization -----------------
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)*")
STOP = set(
    "the a an and or for to of in on at by from with as is are was were be been being this that these those which who whose where when".split()
)

def tokenize(text: str) -> List[str]:
    toks = [t.lower() for t in _TOKEN_RE.findall(text or "")]
    return [t for t in toks if t not in STOP and len(t) > 1]

# ----------------- neighbors (optional expansion) -----------------
_NUM = re.compile(r"(\d+)$")

def parse_sec_ord(cid: str) -> Tuple[str, int, str]:
    parts = cid.split("::")
    paper = parts[0] if len(parts) >= 3 else "paper"
    sec = parts[1] if len(parts) >= 3 else "unknown"
    m = _NUM.search(parts[-1])
    order = int(m.group(1)) if m else 0
    return sec, order, paper


def build_neighbors(chunk_ids: List[str]) -> Dict[str, List[str]]:
    parsed = {cid: parse_sec_ord(cid) for cid in chunk_ids}
    groups: Dict[Tuple[str, str], List[Tuple[int, str]]] = {}
    for cid, (sec, order, paper) in parsed.items():
        groups.setdefault((paper, sec), []).append((order, cid))

    neigh = {cid: [] for cid in chunk_ids}
    for _, lst in groups.items():
        lst.sort(key=lambda t: t[0])
        for i, (_, cid) in enumerate(lst):
            # fill sequential neighbors; final windowing happens later
            neigh[cid] = [x[1] for x in lst]
    return neigh

# ----------------- metrics -----------------
def recall_at_k(ranks: List[int], k: int) -> float:
    return float(any(r <= k for r in ranks))

def mrr_at_k(ranks: List[int], k: int) -> float:
    v = [r for r in ranks if r <= k]
    return 1.0 / v[0] if v else 0.0

def coverage_at_k(ranks: List[int], total: int, k: int) -> float:
    return min(len([r for r in ranks if r <= k]), total) / max(total, 1)

def ranks_for(toplist: List[str], relevant: List[str]) -> List[int]:
    pos = {cid: i + 1 for i, cid in enumerate(toplist)}
    BIG = 10 ** 9
    return [pos.get(cid, BIG) for cid in relevant]

# ----------------- evaluation -----------------
def eval_system(search_fn, qset: List[dict], top_k: int) -> Dict[str, float]:
    single, multi = [], []
    all_f1_scores: List[float] = []

    start_time = time.perf_counter()
    evaluated = 0

    for i, q in enumerate(qset):
        print(f"Processing query {i+1}/{len(qset)}...", file=sys.stderr)
        relevant = list(set(q.get("relevant_chunks", [])))
        if not relevant:
            continue

        res = search_fn(q["query_text"])
        toplist = [cid for cid, *_ in res][:top_k]

        tp = len(set(toplist) & set(relevant))
        prec = tp / len(toplist) if toplist else 0.0
        rec = tp / len(relevant)
        f1_at_k = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        all_f1_scores.append(f1_at_k)

        r = ranks_for(toplist, relevant)
        if q.get("query_type", "single_chunk").startswith("single"):
            single.append(r)
        else:
            multi.append((r, len(relevant)))
        evaluated += 1

    total_time_seconds = time.perf_counter() - start_time
    avg_time_ms = (total_time_seconds / evaluated) * 1000 if evaluated else 0.0

    out: Dict[str, float] = {}
    if single:
        out["single_R@1"] = float(np.mean([recall_at_k(r, 1) for r in single]))
        out["single_R@5"] = float(np.mean([recall_at_k(r, 5) for r in single]))
        out["single_R@10"] = float(np.mean([recall_at_k(r, 10) for r in single]))
        out["single_MRR@10"] = float(np.mean([mrr_at_k(r, 10) for r in single]))
    if multi:
        out["multi_MaxR@5"] = float(np.mean([recall_at_k(r, 5) for r, _ in multi]))
        out["multi_MaxR@10"] = float(np.mean([recall_at_k(r, 10) for r, _ in multi]))
        out["multi_Cov@5"] = float(np.mean([coverage_at_k(r, n, 5) for r, n in multi]))
        out["multi_Cov@10"] = float(np.mean([coverage_at_k(r, n, 10) for r, n in multi]))

    out["overall_F1@10"] = float(np.mean(all_f1_scores)) if all_f1_scores else 0.0
    out["total_time_s"] = total_time_seconds
    out["avg_time_ms"] = avg_time_ms
    out["evaluated_queries"] = evaluated

    return out

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=Path, required=True)
    ap.add_argument("--qset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--k1", type=float, default=1.2)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("--neighbor_window", type=int, default=0, help="0 = no expansion; N adds N neighbors on each side within the same (paper,section)")
    args = ap.parse_args()

    # Load data
    chunk_ids, chunk_text = load_master(args.chunks)

    # Optional neighbor expansion
    expanded_text: Dict[str, str] = {}
    if args.neighbor_window > 0:
        neigh_seq = build_neighbors(chunk_ids)  # sequences per section
        # build quick position maps
        pos_map: Dict[str, Dict[str, int]] = {}
        for seq in neigh_seq.values():
            for i, cid in enumerate(seq):
                pos_map[cid] = {"seq": tuple(seq), "idx": i}
        for cid in chunk_ids:
            info = pos_map.get(cid)
            if not info:
                expanded_text[cid] = chunk_text.get(cid, "")
                continue
            seq, idx = list(info["seq"]), info["idx"]
            L = max(0, idx - args.neighbor_window)
            R = min(len(seq), idx + args.neighbor_window + 1)
            parts = [chunk_text.get(x, "") for x in seq[L:R]]
            expanded_text[cid] = " ".join(parts)
    else:
        for cid in chunk_ids:
            expanded_text[cid] = chunk_text.get(cid, "")

    # Tokenize corpus and build BM25
    docs_tokens = [tokenize(expanded_text[cid]) for cid in chunk_ids]
    bm25 = BM25Okapi(docs_tokens, k1=args.k1, b=args.b)

    cid2pos = {cid: i for i, cid in enumerate(chunk_ids)}

    def bm25_search(qtext: str):
        q_toks = tokenize(qtext)
        scores = bm25.get_scores(q_toks)  # np.ndarray over all docs
        order = np.argsort(-scores)[:args.topk]
        return [(chunk_ids[i], float(scores[i]), 0.0) for i in order]

    print("--- Evaluating baseline: BM25 ---")
    metrics = eval_system(bm25_search, load_qset(args.qset), args.topk)

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"✅ BM25 results saved to {out_path}")

if __name__ == "__main__":
    main()



# python vectorrag/eval/retrievers/eval_bm25.py \
#   --chunks data/master_in_embedding_order.jsonl \
#   --qset data/qqset.json \
#   --out data/results_with_timing/results_bm25.json \
#   --topk 10 \
#   --k1 1.2 \
#   --b 0.75 \
#   --neighbor_window 0




