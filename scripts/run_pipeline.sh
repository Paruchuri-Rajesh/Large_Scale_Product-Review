#!/usr/bin/env bash
# Run the full pipeline end-to-end: ingest -> ETL -> train -> stream once -> serve.
# Use ROWS=N to override sample size.
set -euo pipefail

cd "$(dirname "$0")/.."

ROWS="${ROWS:-30000}"
FRAUD_SHARE="${FRAUD_SHARE:-0.06}"
PORT="${PORT:-8000}"

echo "[1/5] generating sample (${ROWS} rows, fraud=${FRAUD_SHARE}) ..."
python3 -m src.ingest.generate_sample --rows "${ROWS}" --fraud-share "${FRAUD_SHARE}"

echo "[2/5] running Spark batch ETL ..."
python3 -m src.etl.batch_etl

echo "[3/5] training models + logging to MLflow ..."
python3 -m src.train.train

echo "[4/5] one-shot streaming pass on existing data/streaming_in/ ..."
mkdir -p data/streaming_in
if [ -z "$(ls -A data/streaming_in 2>/dev/null)" ]; then
  head -50 data/raw/reviews.jsonl > data/streaming_in/seed.jsonl
fi
rm -rf data/streaming_out/_checkpoints data/streaming_out/scored
python3 -m src.stream.score_stream --once || true

echo "[5/5] starting FastAPI on http://127.0.0.1:${PORT}/  (Ctrl-C to stop)"
exec python3 -m uvicorn src.serve.app:app --host 127.0.0.1 --port "${PORT}"
