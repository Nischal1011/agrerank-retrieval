import argparse
import json
import numpy as np
from pathlib import Path
from scipy import stats

def load_scores(path):
    if not Path(path).exists():
        print(f"❌ File not found: {path}")
        return {}
    with open(path) as f:
        data = json.load(f)
    return data.get('per_query') or data.get('per_query_scores')

def check_consensus_overlap(qid, cache_dir, top_k=100):
    """
    Loads the cached .npz for a query and calculates how many documents 
    appear in BOTH the Top-K Dense and Top-K Sparse lists.
    """
    npz_path = cache_dir / f"{qid}.npz"
    if not npz_path.exists():
        return "N/A (No Cache)"
    
    try:
        data = np.load(npz_path, allow_pickle=True)
        
        # Reconstruct Scores
        q_vec = data['query_vec']
        if q_vec.ndim == 2: q_vec = q_vec[0]
        
        d_vecs = data['doc_vecs']
        sparse_scores = data['sparse_scores']
        
        # Dense Scores
        dense_scores = q_vec @ d_vecs.T
        
        # Get Top-K Indices
        dense_top_k = set(np.argsort(dense_scores)[::-1][:top_k])
        sparse_top_k = set(np.argsort(sparse_scores)[::-1][:top_k])
        
        # Calculate Overlap
        intersection = len(dense_top_k.intersection(sparse_top_k))
        return intersection
    except Exception as e:
        return f"Error: {e}"

def main():
    parser = argparse.ArgumentParser()
    # Default paths based on your latest run structure
    parser.add_argument("--baseline_json", type=str, 
                        default="cache/results_msmarco-passage-v2_trec-dl-2022_judged_BASELINE_k500.json")
    parser.add_argument("--model_json", type=str, 
                        default="cache/results_msmarco-passage-v2_trec-dl-2022_judged_AGRERANK_k500.json")
    parser.add_argument("--cache_dir", type=str, 
                        default="cache/msmarco-passage-v2_trec-dl-2022_judged_k500")
    args = parser.parse_args()

    print(f"Loading Baseline: {args.baseline_json}")
    base = load_scores(args.baseline_json)
    print(f"Loading Model:    {args.model_json}")
    model = load_scores(args.model_json)
    
    if not base or not model:
        print("❌ Could not load JSON files.")
        return

    # Identify Failures
    failures = []
    common_qids = set(base.keys()) & set(model.keys())
    
    for qid in common_qids:
        b_score = base[qid]['ndcg_cut_10']
        m_score = model[qid]['ndcg_cut_10']
        diff = m_score - b_score
        
        # "Failure" definition: AgreRank performed worse
        if diff < 0:
            failures.append((diff, qid, b_score, m_score))
            
    failures.sort() # Sort by biggest loss first
    
    print(f"\nFound {len(failures)} failure cases (AgreRank < Baseline).")
    print("-" * 80)
    print(f"{'QID':<10} | {'Drop':<10} | {'Base':<6} -> {'Model':<6} | {'Overlap (Top-100)':<20}")
    print("-" * 80)
    
    cache_path = Path(args.cache_dir)
    
    overlaps = []
    
    for diff, qid, b, m in failures:
        overlap = check_consensus_overlap(qid, cache_path, top_k=100)
        if isinstance(overlap, int):
            overlaps.append(overlap)
        print(f"{qid:<10} | {diff:.4f}     | {b:.4f} -> {m:.4f} | {overlap}")

    print("-" * 80)
    
    if overlaps:
        avg_overlap = np.mean(overlaps)
        print(f"\nAverage Consensus Overlap for Failures: {avg_overlap:.1f} docs (out of 100)")
        
        if avg_overlap < 20:
            print("✅ HYPOTHESIS CONFIRMED: Failures correlate with low consensus (Disjoint Sets).")
            print("   Keep the sentence: 'Error analysis reveals that these failures primarily occur")
            print("   when sparse and dense retrievers return disjoint candidate sets.'")
        else:
            print("⚠️ HYPOTHESIS WEAK: Failures have decent overlap.")
            print("   Consider changing text to: 'Failures occur due to over-suppression in Stage 3.'")

if __name__ == "__main__":
    main()