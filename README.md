# Large-Scale Product Review Sentiment & Fraud Detection

Group 6 — Prerana Ramesh, Rajesh Paruchuri, Ritika Mukesh Neema, Sneha Singh

End-to-end distributed system that ingests Amazon-shaped product reviews,
runs **Apache Spark** for batch cleaning / feature engineering / aggregation,
trains **sentiment** and **fraud** models tracked in **MLflow**, scores new
reviews continuously through **Spark Structured Streaming**, and serves
predictions + a dashboard from a **FastAPI** app.

```
┌────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  ingest    │──>│ Spark batch  │──>│  train (ML-  │──>│  Spark       │──>│  FastAPI     │
│  (JSONL)   │   │  ETL +       │   │  flow runs)  │   │  Structured  │   │  +  Web UI   │
│            │   │  features +  │   │  joblib /    │   │  Streaming   │   │  /predict    │
│            │   │  aggregates  │   │  meta.json   │   │  scorer      │   │  /aggregates │
└────────────┘   └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
        │               │                   │                  │                  │
        ▼               ▼                   ▼                  ▼                  ▼
 data/raw       data/processed/        models/*.joblib   data/streaming_out/   http://…/
 reviews.jsonl  train,test,product,    mlruns/           scored/*.json          dashboard
                reviewer parquets
```

## Repo layout

```
src/
  ingest/generate_sample.py   # synthetic Amazon-shaped JSONL producer
  etl/batch_etl.py            # Spark batch: clean, features, aggregates, weak fraud labels
  train/train.py              # sklearn pipelines + MLflow tracking
  train/{features,registry,evaluate,...}.py  # shared feature defs + helpers
  stream/score_stream.py      # Spark Structured Streaming foreachBatch scorer
  serve/app.py                # FastAPI endpoints
  serve/templates/index.html  # dashboard
  serve/static/{app.js,style.css}
  common/{config,schema,spark,text}.py
scripts/
  feed_stream.py              # drip raw reviews into data/streaming_in/
  run_pipeline.sh             # end-to-end: ingest -> ETL -> train -> stream once -> serve
tests/test_pipeline.py
data/{raw,processed,streaming_in,streaming_out}/
models/   mlruns/
```

## Quickstart

```bash
make install     # pip install -r requirements.txt
make all         # ingest + Spark ETL + train + MLflow logging
make stream-once # one-shot scoring of whatever's in data/streaming_in/
make serve       # http://127.0.0.1:8000/  (dashboard + REST)
make mlflow-ui   # http://127.0.0.1:5000/  (experiment tracking)
```

Or all at once:

```bash
bash scripts/run_pipeline.sh   # ROWS=200000 FRAUD_SHARE=0.05 PORT=8000 to override
```

To watch the streaming dashboard refresh, open two terminals:

```bash
# terminal 1 — continuous streaming scorer
make stream

# terminal 2 — drip new reviews into data/streaming_in/
make feed
```

## Using a real Amazon Reviews dataset

The pipeline accepts any JSONL file matching the schema in
`src/common/schema.py`. To run on a real Amazon Customer Reviews TSV slice,
convert it once:

```bash
# example: snap a slice of reviews to JSONL with the expected fields
python -c "
import pandas as pd, json, sys
df = pd.read_csv(sys.argv[1], sep='\t')[
    ['review_id','product_id','customer_id','star_rating','helpful_votes',
     'total_votes','verified_purchase','review_headline','review_body','review_date',
     'product_category']
].rename(columns={'customer_id':'reviewer_id'})
df['verified_purchase'] = df['verified_purchase'].eq('Y')
df['event_ts'] = pd.to_datetime(df['review_date']).astype('int64')//10**9
df.to_json('data/raw/reviews.jsonl', orient='records', lines=True)
" /path/to/amazon_reviews.tsv

make etl train
```

Spark configuration in `src/common/spark.py` is defaulted for local laptops
(2 GB driver, 8 shuffle partitions). For a real cluster, override via
`spark-submit --master ...` and tune partitions.

## API

| Method | Path | What |
|---|---|---|
| GET | `/healthz` | liveness |
| GET | `/metadata` | training metrics + feature columns |
| POST | `/predict` | score one review |
| POST | `/predict/batch` | score up to 1000 reviews |
| GET | `/aggregates/products?limit=N&by=col` | top-N from product rollup |
| GET | `/aggregates/fraud-reviewers?limit=N` | suspicious reviewers |
| GET | `/stream/recent?limit=N` | latest streaming-scored rows |

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'content-type: application/json' \
  -d '{"review_body":"absolutely love this, exceeded my expectations","star_rating":5}'
# {"sentiment":"positive","fraud_proba":3.4e-07,"fraud_flag":0,...}
```

## Dashboard (Web UI)

The home page (`make serve` → `/`) is a single-page dashboard that polls the
same REST endpoints as the table above. Highlights:

- **Live sections** — recent streaming scores, product/reviewer aggregates,
  and model overview KPI cards (latency, class balance, basic health).
- **Threshold study** — precision/recall/F1 vs. fraud threshold from the
  sweep in `models/thresholds.json`. The chart uses an **adaptive Y-axis**:
  when metrics sit in a narrow band it zooms to that range (instead of always
  stretching 0–1) so small tradeoffs stay visible; flat series get a small
  padded band. Copy on the panel explains that harder synthetic evaluation is
  meant to show a modest holdout tradeoff, not a flat curve by default.
- **Model metadata** — a compact summary of `GET /metadata` (mirrors
  `models/meta.json`): sentiment/fraud run IDs and headline metrics, fraud
  ROC-AUC, numeric feature count, and selected model names when present —
  styled like the ML overview cards rather than a raw JSON dump.

## Models

- **Sentiment** — TF-IDF (uni+bi-gram) + multinomial logistic regression.
  Labels derived from star rating: 1–2 negative, 3 neutral, 4–5 positive.
- **Fraud** — TF-IDF over body + 17 numeric / behavioral features
  (per-row stats + reviewer/product aggregates + duplicate count) into a
  Gradient Boosted Trees classifier; weak labels from heuristic
  (duplicate-text bursts, high-velocity 5-star unverified reviewers).
  Probabilities are returned so you can choose a threshold per business need.

Both pipelines are logged to MLflow under experiment
`amazon_reviews_sentiment_fraud` (params, metrics, classification report,
the joblib artifact).

## Notes on the synthetic generator

The proposal points at a 10 GB+ public dataset. The generator
(`src/ingest/generate_sample.py`) produces statistically similar JSONL
locally so the full pipeline can be exercised without the multi-GB
download. Phrase pools are sentiment-stratified; a configurable share of
reviews are planted with fraud-like patterns (duplicate bursts, same-day
velocity with paraphrased text, slower reviewer rings, plus organic hard
negatives). Set `--rows 1000000` for large samples; Spark scales linearly.

**Difficulty (`--difficulty`, default `medium`):** `easy` stays closer to the
legacy generator (cleaner separation). `medium` adds cross-class phrase overlap,
typos/punctuation noise, mixed-sentiment wording, and a richer fraud mix.
`easy`/`medium`/`hard` step up overlap and subtle fraud so baseline gaps,
threshold curves, and error analysis are more informative — harder synthetic
data stresses models without changing the JSONL schema or downstream formats.

Examples:

```bash
python -m src.ingest.generate_sample --rows 30000 --fraud-share 0.06 --difficulty hard
make ingest   # Makefile uses medium by default
ROWS=50000 DIFFICULTY=easy bash scripts/run_pipeline.sh
```

## Tests

```bash
make test
```

Covers text cleaning, the per-row serving feature path (including fraud
numeric columns), and the FastAPI endpoints with the trained models loaded
off disk.
