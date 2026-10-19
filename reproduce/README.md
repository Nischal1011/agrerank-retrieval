# Reproducing AgreRank from scratch

This directory rebuilds every number in the paper from public inputs, with one
canonical implementation, so anyone can verify them. Results are in
[`../REPRODUCTION.md`](../REPRODUCTION.md); raw numbers are in `outputs/`.

## What is here

| File | Purpose |
|---|---|
| `agrerank.py` | The method as written in the paper (Eqs. 1–6), its ablation switches, the Stage-2 re-query for small corpora, and an exact port of the code that produced the paper's TREC tables (`agrerank_legacy`) |
| `test_agrerank.py` | Unit tests: each equation, the fallback, the CE depth, the re-query |
| `infra.py` | Embedding and cross-encoder caches (SQLite, keyed by text hash), TREC metrics, paired tests |
| `trec_build.py` | TREC DL 2021/2022 pools: Pyserini BM25 top-1000, passage text, embeddings, CE logits |
| `import_legacy_embeddings.py` | Import the paper's cached `text-embedding-3-large` vectors, with integrity checks |
| `trec_eval.py` | Every system, control, ablation and sensitivity setting; writes `outputs/trec_results*.json` |
| `diag.py` | Diagnostic corpus (Tables 3, 4, 6b) from `data/diagnostic/` |
| `report.py` | Writes `REPRODUCTION.md` (paper value next to each reproduced value) |
| `run_all.sh` | Runs everything in order |

## Inputs

- **TREC DL 2021 / 2022**: queries and qrels from `ir_datasets`
  (`msmarco-passage-v2/trec-dl-{2021,2022}/judged`), the prebuilt Pyserini index
  `msmarco-v2-passage`, passage text from `ir_datasets`.
- **Diagnostic corpus**: `data/diagnostic/chunks.jsonl` (1,249 chunks of the 19 papers in
  `data/pdf_files`, 250-token windows with title/section prefix) and
  `data/diagnostic/queries.json` (the 100 queries, 145 relevant chunks, used for Tables 3 and 4).
  The 10,000 distractors are generated deterministically by `src/noisy_corpus.py` (seed 42).
  `data/qset.json` is an earlier 41-query version.
- **Models**: `text-embedding-3-large` (OpenAI), `cross-encoder/ms-marco-MiniLM-L-12-v2`,
  and for Table 6c `BAAI/bge-large-en-v1.5` and `intfloat/e5-large-v2`.

## Running

```bash
pip install -r reproduce/requirements.txt
export JAVA_HOME=/path/to/jdk-21          # Pyserini needs Java 21
export OPENAI_API_KEY=...                 # only for fresh OpenAI embeddings (about $2)
./reproduce/run_all.sh
```

Without an OpenAI key, `LEGACY=/path/to/original/workspace ./reproduce/run_all.sh` imports the
vectors that were cached when the paper's runs were made. The importer refuses any vector whose
pool, query text or passage text differs from the freshly built inputs, or that is not unit norm.
The BGE and E5 parts need no key at all.

Everything is cached under `reproduce/work/` (about 3 GB), so reruns are fast and resumable.
On an M1 Pro the cross-encoder pass over both TREC pools takes about 40 minutes; everything
else takes minutes.

## Systems

All TREC systems re-rank the same Pyserini BM25 top-k pool with the same cross-encoder and
CE depth (200), matching the paper's setup. `AgreRank` is Eqs. 1–6 of the paper with the CE score
min–max normalized over the top-M (see the note in `agrerank.rerank_top`).
`AgreRank (original code)` is an exact port of the script that produced the paper's Table 2,
kept for reference. `Dense+CE fused` / `RRF+CE fused` give the baselines the same CE + first-stage
fusion AgreRank uses. `abl:` and `sens` rows are the ablation and sensitivity settings.

## Known limits

- On TREC, every system re-ranks the same BM25 top-k pool. The expanded query re-scores that
  pool. Re-querying a dense index over all 138M passages (Sec. 3.3) would need embedding the
  whole collection. On the diagnostic corpus, where exhaustive search is cheap, both versions are
  reported (`AgreRank-pool` and `AgreRank-requery`).
- How the diagnostic queries and relevance labels were written is not recorded in code.
