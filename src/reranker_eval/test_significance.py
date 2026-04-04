import argparse
import pytrec_eval
import numpy as np
from scipy import stats
import ir_datasets
from pathlib import Path
import sys

def load_run_file(path):
    """Parses a standard TREC run file into {qid: {docid: score}}"""
    run = {}
    print(f"Loading run file: {path}")
    try:
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6: continue
                qid, _, docid, rank, score, _ = parts
                run.setdefault(qid, {})[docid] = float(score)
    except FileNotFoundError:
        sys.exit(f"Error: File not found at {path}")
    return run

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="ir_datasets ID")
    parser.add_argument("--baseline", required=True, help="Path to Strong Baseline .txt file")
    parser.add_argument("--treatment", required=True, help="Path to AgreRank .txt file")
    args = parser.parse_args()

    # 1. Load Qrels correctly
    print(f"Loading Qrels for {args.dataset}...")
    try:
        dataset = ir_datasets.load(args.dataset)
        qrels = {}
        for qrel in dataset.qrels_iter():
            # Corrected Loop Logic
            qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance
    except Exception as e:
        sys.exit(f"Error loading dataset: {e}")

    # 2. Load Runs
    run_base = load_run_file(args.baseline)
    run_treat = load_run_file(args.treatment)

    # 3. Evaluate Per-Query nDCG@10
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg_cut_10'})
    
    res_base = evaluator.evaluate(run_base)
    res_treat = evaluator.evaluate(run_treat)

    # 4. Align Scores for T-Test
    # We only compare queries that exist in the qrels (judged queries)
    qids = sorted(list(res_base.keys()))
    
    scores_base = []
    scores_treat = []
    
    print(f"\nComparing {len(qids)} queries...")
    
    skipped = 0
    for qid in qids:
        # Ensure query exists in treatment
        if qid not in res_treat:
            skipped += 1
            continue
            
        b_score = res_base[qid]['ndcg_cut_10']
        t_score = res_treat[qid]['ndcg_cut_10']
        
        scores_base.append(b_score)
        scores_treat.append(t_score)
    
    if skipped > 0:
        print(f"Warning: {skipped} queries in baseline were missing from treatment run.")

    # 5. Perform Paired T-Test
    t_stat, p_val = stats.ttest_rel(scores_treat, scores_base)
    
    mean_base = np.mean(scores_base)
    mean_treat = np.mean(scores_treat)
    delta = mean_treat - mean_base
    pct_imp = (delta / mean_base) * 100 if mean_base > 0 else 0.0
    
    print("\n=== Significance Test Results ===")
    print(f"Dataset:               {args.dataset}")
    print(f"Baseline Mean nDCG@10: {mean_base:.4f}")
    print(f"AgreRank Mean nDCG@10: {mean_treat:.4f}")
    print(f"Improvement:           +{delta:.4f} ({pct_imp:.1f}%)")
    print(f"P-Value (Paired t-test): {p_val:.5f}")
    print("-" * 30)
    
    if p_val < 0.05:
        print("✅ RESULT IS STATISTICALLY SIGNIFICANT (p < 0.05)")
        if p_val < 0.01:
            print("   (Highly Significant, p < 0.01)")
    else:
        print("⚠️ RESULT IS NOT SIGNIFICANT (p >= 0.05)")

if __name__ == "__main__":
    main()

# python vectorrag/eval/test_significane.py \
#   --dataset "msmarco-passage-v2/trec-dl-2022/judged" \
#   --baseline "./cache/run_msmarco-passage-v2_trec-dl-2021_judged_k500_RRF.txt" \
#   --treatment "./cache/run_msmarco-passage-v2_trec-dl-2021_judged_k500_AgreRankP.txt"

# python vectorrag/eval/test_significance.py  \
#   --dataset "msmarco-passage-v2/trec-dl-2021/judged" \
#   --baseline "./cache/run_msmarco-passage-v2_trec-dl-2021_judged_k1000_StandardCE.txt" \
#   --treatment "./cache/run_msmarco-passage-v2_trec-dl-2021_judged_k1000_AgreRankP.txt"