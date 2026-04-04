# Purpose: Execute the full suite of experiments for the AgreRank.
# 1. Main Results (TREC 21 vs 22 across K) -> Tables 3 & 4
# 2. Ablation Waterfall -> Table 5 & Figure 3
# 3. Sensitivity Analysis -> Figure 4

set -e # Stop execution if any step fails
SCRIPT_PATH="vectorrag/eval/eval_agrerank_abalation_trec.py"

# --- Pre-flight Checks ---
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "❌ CRITICAL: Script not found at $SCRIPT_PATH"
    exit 1
fi
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "❌ CRITICAL: OPENAI_API_KEY environment variable not set"
    exit 1
fi

# --- SOTA Hyperparameters (Locked in from Optimization) ---
PRF_ALPHA=0.65
AGREEMENT_THRESH=0.8
CE_K=200  # Rerank Top 400 (200 * 2). This yields the 0.6025 result.

echo "================================================================="
echo "==      PART 1: MAIN RESULTS & ROBUSTNESS (Tables 3 & 4)       =="
echo "==      Comparing RRF vs AgreRank across Easy/Hard tasks       =="
echo "================================================================="

# We compare 2021 (Easy) vs 2022 (Hard) across varying retrieval depths
DATASETS=("msmarco-passage-v2/trec-dl-2021/judged" "msmarco-passage-v2/trec-dl-2022/judged")
K_VALUES=(200 500 1000)
MODES_MAIN=("rrf_baseline" "agrerank_full")

for dataset in "${DATASETS[@]}"; do
    for k in "${K_VALUES[@]}"; do
        for mode in "${MODES_MAIN[@]}"; do
            echo -e "\n>>> [Main Result] Dataset: $dataset | K: $k | Mode: $mode"
            python3 $SCRIPT_PATH \
                --dataset "$dataset" \
                --candidates_k "$k" \
                --ablation_mode "$mode" \
                --prf_alpha "$PRF_ALPHA" \
                --agreement_threshold "$AGREEMENT_THRESH" \
                --cross_encoder_k "$CE_K"
        done
    done
done

echo -e "\n\n"
echo "================================================================="
echo "==      PART 2: ABLATION WATERFALL (Table 5 & Fig 3)           =="
echo "==      Proving the Monotonic Improvement on TREC 22           =="
echo "================================================================="

DATASET_ABLATION="msmarco-passage-v2/trec-dl-2022/judged"
K_ABLATION=500
# The definitive sequence
MODES_ABLATION=("rrf_baseline" "stage1_boost" "stage2_prf" "stage3_features" "agrerank_full")

for mode in "${MODES_ABLATION[@]}"; do
    echo -e "\n>>> [Ablation Step] Mode: $mode"
    python3 $SCRIPT_PATH \
        --dataset "$DATASET_ABLATION" \
        --candidates_k "$K_ABLATION" \
        --ablation_mode "$mode" \
        --prf_alpha "$PRF_ALPHA" \
        --agreement_threshold "$AGREEMENT_THRESH" \
        --cross_encoder_k "$CE_K"
done

echo -e "\n\n"
echo "================================================================="
echo "==      PART 3: SENSITIVITY ANALYSIS (Figure 4)                =="
echo "==      Proving Stability of Alpha and Cascade Depth           =="
echo "================================================================="

# 1. Alpha Sensitivity (Standard Neural Baseline usually sits around 0.56)
# Testing extremes to show convexity
for alpha in 0.5 0.65 0.8 1.0; do
    echo -e "\n>>> [Sensitivity] Alpha=$alpha"
    python3 $SCRIPT_PATH \
        --dataset "$DATASET_ABLATION" \
        --candidates_k "$K_ABLATION" \
        --ablation_mode "agrerank_full" \
        --prf_alpha "$alpha" \
        --agreement_threshold "$AGREEMENT_THRESH" \
        --cross_encoder_k "$CE_K"
done

# 2. Cascade Depth Sensitivity (Efficiency Trade-off)
# Testing K=50 (Rank 100) up to K=200 (Rank 400)
for depth in 50 100 200; do
    echo -e "\n>>> [Sensitivity] CE_Depth=$depth (Total Ranked: $((depth * 2)))"
    python3 $SCRIPT_PATH \
        --dataset "$DATASET_ABLATION" \
        --candidates_k "$K_ABLATION" \
        --ablation_mode "agrerank_full" \
        --prf_alpha "$PRF_ALPHA" \
        --agreement_threshold "$AGREEMENT_THRESH" \
        --cross_encoder_k "$depth"
done

echo -e "\n✅ All experiments complete. Results saved in ./ablation_results/"