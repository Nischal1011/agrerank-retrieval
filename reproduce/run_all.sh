#!/usr/bin/env bash
# Rebuild every number from scratch. Steps are cached and resumable.
#   LEGACY=/path/to/rag-lab  ./run_all.sh     # reuse the paper's text-embedding-3-large vectors
#   ./run_all.sh                              # embed through the OpenAI API (needs OPENAI_API_KEY, ~$2)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
: "${JAVA_HOME:?set JAVA_HOME to a Java 21 JDK (Pyserini requirement)}"

$PY -m pytest -q test_agrerank.py

# TREC DL 2021/2022: BM25 pools, embeddings, cross-encoder logits
$PY trec_build.py --steps pool ce
if [[ -n "${LEGACY:-}" ]]; then
  $PY import_legacy_embeddings.py --legacy-cache "$LEGACY/cache"
else
  $PY trec_build.py --steps embed --encoder openai
fi
$PY trec_eval.py

# Diagnostic corpus (Tables 3, 4, 6b)
if [[ -n "${LEGACY:-}" ]]; then
  $PY diag.py build --encoder openai --legacy-data "$LEGACY/data"
else
  $PY diag.py build --encoder openai
fi
$PY diag.py eval --encoder openai

# Open encoders (Table 6c, DL'22); no API key needed
for ENC in bge e5; do
  $PY trec_build.py --years 2022 --steps embed --encoder $ENC
  $PY trec_eval.py --years 2022 --encoder $ENC
done

$PY report.py
