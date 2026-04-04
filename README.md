# AgreRank: Consensus-Anchored Expansion for Noise-Resilient Re-ranking

Official code repository for the SIGIR 2026 paper:

> **Consensus-Anchored Expansion for Noise-Resilient Re-ranking**
> Nischal Subedi, Cencheng Shen
> *Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '26), July 20–24, 2026, Melbourne, Australia.*
> [Paper](https://openreview.net/forum?id=Nk118oAX2L)

---

## Overview

Dense retrievers bridge vocabulary gaps but suffer from **semantic drift** — ranking topically similar passages that lack factual relevance. Sparse retrievers like BM25 avoid such false positives but miss paraphrases. These failure modes are complementary.

**AgreRank** exploits this complementarity. Rather than naively fusing scores (as in RRF), AgreRank identifies *anchor documents* where both sparse and dense retrievers agree, uses them to safely expand the query, and applies a geometric gating mechanism to suppress drift before passing a clean candidate pool to a cross-encoder re-ranker.

The pipeline operates in four stages:

1. **Consensus Seeding** — identify high-precision anchors where both BM25 and dense retrieval score above the 80th percentile
2. **Anchor-Guided Expansion** — expand the query embedding via a weighted centroid of anchor document vectors
3. **Geometric Gating** — penalize candidates that diverge from the original query's vector direction
4. **Neural Refinement** — re-rank the filtered top-200 candidates with a lightweight 33M-parameter cross-encoder

On TREC Deep Learning 2021 and 2022 (MS MARCO v2, 138M passages), AgreRank achieves **nDCG@10 of 0.706 and 0.627** respectively — improving over Dense+CE by 3.4–5.7% and over RRF+CE by 3.7–5.6%, with no additional parameters or training.

---

## Results

| Method | DL'21 nDCG@10 | DL'22 nDCG@10 |
|---|---|---|
| BM25 | 0.254 | 0.132 |
| Dense | 0.666 | 0.551 |
| RRF | 0.518 | 0.373 |
| Dense + CE | 0.683 | 0.593 |
| RRF + CE | 0.681 | 0.594 |
| **AgreRank (Ours)** | **0.706** | **0.627** |

All systems use identical components: Pyserini BM25, `text-embedding-3-large`, and `cross-encoder/ms-marco-MiniLM-L-12-v2`.

---

## Repository Structure

```
agrerank-retrieval/
├── data/
│   ├── pdf_files/                              # Source PDF papers (tracked via Git LFS)
│   └── preprocessed_to_json/                  # Preprocessed JSON files (one per paper, named by arXiv ID)
│
└── src/
    ├── reranker_eval/                          # Evaluation scripts for all systems
    │   ├── agre_rank_with_baselines.py         # AgreRank + all baseline comparisons (Tables 2 & 3)
    │   ├── agrerank_fallback.py                # Fallback mechanism evaluation
    │   ├── agrerank_per_query_analysis.py      # Per-query improvement analysis (Figure 4)
    │   ├── compare_results.py                  # Cross-system result comparison
    │   ├── error_analysis.py                   # Failure case analysis (Table 7)
    │   ├── eval_agrerank.py                    # Main AgreRank evaluation
    │   ├── eval_agrerank_abalation_trec.py     # Ablation waterfall on TREC DL (Table 5)
    │   ├── eval_agrerank_noisy.py              # Noise resilience evaluation (Table 4)
    │   ├── eval_baseline_noisy.py              # Baseline noise evaluation (Table 4)
    │   ├── eval_bm25.py                        # BM25 standalone evaluation
    │   ├── eval_dense.py                       # Dense retrieval standalone evaluation
    │   ├── eval_neural_baseline_rerank.py      # Cross-encoder baseline evaluation
    │   ├── eval_rrf.py                         # RRF baseline evaluation
    │   ├── eval_trec_agrerank_t_test.py        # Statistical significance testing (p<0.01)
    │   ├── examples_of_hard_query.py           # Hard query qualitative examples (Table 7)
    │   ├── leave_one_out_abalation.py          # Leave-one-out ablation (Table 5)
    │   ├── sparse_dense_fusion_analysis.py     # Consensus region analysis (Table 3)
    │   └── test_significance.py                # Paired t-test significance testing
    │
    ├── scripts/
    │   └── agre_abalation.sh                   # Shell script to run full experiment suite
    │
    ├── visualization/
    │   ├── viz_agrerank_abalation_statistical_consensus.py  # Ablation + consensus stats plots
    │   ├── viz_geometric_intuition_agrerank.py              # Geometric gating visualization
    │   └── viz_per_query_analysis.py                        # Per-query distribution plots (Figure 4)
    │
    ├── build_faiss_index.py                    # Build FAISS HNSW index (M=64, efConstruction=200)
    ├── chunk_json.py                           # Split JSON documents into chunked retrieval units
    ├── create_qset_embeddings.py               # Generate embeddings for query sets
    ├── embed_chunks.py                         # Generate vector embeddings for corpus chunks
    ├── export_embeddings.py                    # Export embedding artifacts for downstream use
    ├── noisy_corpus.py                         # Inject controlled distractor passages (9x expansion)
    ├── pdf_to_json_with_metadata.py            # Convert source PDFs to JSON with metadata
    ├── retrieve.py                             # Run AgreRank retrieval pipeline
    ├── retrieve_and_pass_to_llm.py             # Retrieval with downstream LLM handoff (RAG)
    └── validate_qset.py                        # Validate question-set quality and formatting
```

---

## Data

The `data/` directory contains the diagnostic corpus used in the controlled evaluation (Study 1 in the paper).

- **`data/pdf_files/`** — source PDF papers from the statistical machine learning literature. Tracked via [Git LFS](https://git-lfs.com/). Clone with LFS enabled to pull the actual files:
  ```bash
  git lfs install
  git clone https://github.com/Nischal1011/agrerank-retrieval.git
  ```
- **`data/preprocessed_to_json/`** — JSON files preprocessed from the PDFs (one file per paper, named by arXiv ID). These are the inputs to `chunk_json.py` and are ready to use without re-running `pdf_to_json_with_metadata.py`.

> Note: The large-scale TREC DL experiments use MS MARCO v2 (138M passages), which is not included in this repo. Follow the [MS MARCO v2 instructions](https://microsoft.github.io/msmarco/) to obtain it, then index with Pyserini.

---

## Pipeline (Diagnostic Corpus — Study 1)

This pipeline builds and runs AgreRank over the controlled diagnostic corpus of statistical machine learning papers (1,249 passages, 100 queries). The preprocessed JSON files in `data/preprocessed_to_json/` mean you can skip Step 1 and start directly from chunking.

### 1. Ingest: PDF → JSON
```bash
# Skip this if using data/preprocessed_to_json/ directly
python src/pdf_to_json_with_metadata.py
# Input:  data/pdf_files/*.pdf
# Output: data/preprocessed_to_json/*.json (one file per paper, named by arXiv ID)
```

### 2. Chunk JSON into Retrieval Units
```bash
python src/chunk_json.py
# Input:  data/preprocessed_to_json/*.json
# Output: chunked passage units ready for embedding
```

### 3. Generate Embeddings
```bash
python src/embed_chunks.py               # Embed corpus chunks (text-embedding-3-large)
python src/create_qset_embeddings.py     # Embed query set (text-embedding-3-large)
```

### 4. Build FAISS Index
```bash
python src/build_faiss_index.py
# Builds HNSW index: M=64, efConstruction=200
# Output: FAISS index file for ANN search
```

### 5. Run AgreRank Retrieval
```bash
python src/retrieve.py                   # Full AgreRank pipeline (consensus seed → expand → gate → rerank)
# or, for RAG downstream handoff:
python src/retrieve_and_pass_to_llm.py  # Retrieval + LLM response generation
```

### Noise Resilience Experiments
To reproduce the 9× corpus expansion results from Table 4 (Section 5.4):
```bash
python src/noisy_corpus.py
# Injects 10,000 distractor passages from orthogonal domains (sports, finance, weather, entertainment)
# Expands corpus: 1,249 → 11,249 passages
```

> For TREC DL 2021/2022 large-scale experiments (MS MARCO v2, 138M passages), see the [Reproducing Paper Experiments](#reproducing-paper-experiments) section below.

---

## Reproducing Paper Experiments

All TREC DL experiments can be reproduced via the ablation shell script. Set your OpenAI API key first:

```bash
export OPENAI_API_KEY=your_key_here
bash src/scripts/agre_abalation.sh
```

The script runs three experiment suites in sequence:

**Part 1 — Main Results** (Tables 2 & 3): AgreRank vs RRF across TREC DL 2021 and 2022 at k ∈ {200, 500, 1000}.

**Part 2 — Ablation Waterfall** (Table 5): Stepwise addition of each pipeline stage on TREC DL 2022 at k=500, using modes `rrf_baseline → stage1_boost → stage2_prf → stage3_features → agrerank_full`.

**Part 3 — Sensitivity Analysis** (Figure 5): Stability of expansion weight α ∈ {0.5, 0.65, 0.8, 1.0} and re-ranking depth M ∈ {50, 100, 200}.

Results are saved to `./ablation_results/`.

### Running Individual Evaluations
```bash
# Evaluate AgreRank on TREC DL 2022
python src/reranker_eval/eval_agrerank.py

# Noise resilience (9x corpus expansion)
python src/noisy_corpus.py
python src/reranker_eval/eval_agrerank_noisy.py
python src/reranker_eval/eval_baseline_noisy.py

# Statistical significance (paired t-test, p<0.01)
python src/reranker_eval/eval_trec_agrerank_t_test.py

# Leave-one-out ablation
python src/reranker_eval/leave_one_out_abalation.py

# Per-query analysis
python src/reranker_eval/agrerank_per_query_analysis.py
```

### Reproducing Figures
```bash
python src/visualization/viz_agrerank_abalation_statistical_consensus.py  # Ablation + Table 3
python src/visualization/viz_geometric_intuition_agrerank.py               # Geometric gating
python src/visualization/viz_per_query_analysis.py                         # Figure 4
```

---

## Key Hyperparameters

| Stage | Parameter | Value |
|---|---|---|
| Consensus Seeding | Threshold τ | 0.8 (80th percentile) |
| Anchor-Guided Expansion | Interpolation α | 0.65 |
| Anchor-Guided Expansion | Max anchors | ≤15 (fallback to RRF top-10 if <5) |
| Geometric Gating | Drift floor β | 0.5 |
| Neural Refinement | Re-rank depth M | 200 |
| Neural Refinement | Fusion weight γ | 0.5 |

Performance is stable across τ ∈ [0.7, 0.9] and α ∈ [0.5, 0.8] — no careful tuning required.

---

## Models Used

- **Sparse retrieval**: Pyserini BM25 (k1=0.9, b=0.4) via Lucene 8.11.0
- **Dense retrieval**: `text-embedding-3-large` (3072-dim) via OpenAI
- **FAISS index**: HNSW with M=64, efConstruction=200
- **Cross-encoder**: `cross-encoder/ms-marco-MiniLM-L-12-v2` (33M parameters)

AgreRank also generalizes to open-source encoders — tested with `BGE-large-en-v1.5` (+14.7%) and `E5-large-v2` (+11.6%) over RRF+CE.

---

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{subedi2026agrerank,
  title     = {Consensus-Anchored Expansion for Noise-Resilient Re-ranking},
  author    = {Subedi, Nischal and Shen, Cencheng},
  booktitle = {Proceedings of the 49th International ACM SIGIR Conference on
               Research and Development in Information Retrieval},
  series    = {SIGIR '26},
  year      = {2026},
  location  = {Melbourne, VIC, Australia},
  publisher = {ACM},
  doi       = {10.1145/XXXXXX.XXXXXX}
}
```
*(Update the DOI once assigned by ACM.)*

---

## License

This project is licensed under the MIT License.
