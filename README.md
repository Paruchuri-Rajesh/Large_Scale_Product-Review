# Large-Scale Product Review Sentiment & Fraud Detection

Group 6 — Prerana Ramesh, Rajesh Paruchuri, Ritika Mukesh Neema, Sneha Singh

End-to-end distributed system that ingests Amazon-shaped product reviews,
runs **Apache Spark** for batch cleaning / feature engineering / aggregation,
trains **sentiment** and **fraud** models tracked in **MLflow**, scores new
reviews continuously through **Spark Structured Streaming**, and serves
predictions + a dashboard from a **FastAPI** app.

On top of that core loop, the repo includes **ML evaluation and reporting**:
baseline model comparison, fraud threshold tuning (CSV + optional `models/thresholds.json`),
error-analysis exports, drift monitoring, and an **upgraded dashboard** that surfaces
those artifacts alongside aggregates and streaming output. Single-review scoring adds an
optional, **rule-based `fraud_explanation`** field on `POST /predict` (plain-language summary,
risk band, and bullets from existing scores and serving-time features—no LLM and no external APIs).

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
reports/ml/                  # baseline / threshold / error / drift outputs (when scripts run)
```

## Commands

Typical workflow commands:

```bash
make install          # pip install -r requirements.txt
make ingest           # synthetic JSONL (see difficulty below; Makefile defaults to medium)
make etl              # Spark batch ETL → Parquet features + aggregates
make train            # train sentiment + fraud, MLflow logging, meta.json
make serve            # FastAPI + dashboard → http://127.0.0.1:8000/
make stream           # Spark Structured Streaming scorer (continuous)
make feed             # drip raw reviews into data/streaming_in/
make test             # pytest
```

Optional ML evaluation scripts (write under `reports/ml/` when successful):

```bash
python -m src.train.baselines          # baseline comparison CSV (needs train/test Parquet)
python -m src.train.threshold_tuning    # threshold sweep + thresholds.json (needs test Parquet + trained fraud model)
python -m src.train.error_analysis      # misclassified samples (needs Parquet + both trained models)
python -m src.train.drift_monitor       # drift summary JSON (needs train + test Parquet)
```

Other Makefile targets:

```bash
make stream-once      # score whatever is already in data/streaming_in/, then exit
make mlflow-ui        # MLflow UI on http://127.0.0.1:5000/
make all              # ingest + etl + train
```

One-shot full pipeline:

```bash
bash scripts/run_pipeline.sh   # ROWS=200000 FRAUD_SHARE=0.05 PORT=8000 etc.
```

## Live demo (three terminals)

For an end-to-end **live** demo with the dashboard and streaming updates:

```bash
# Terminal 1 — API + web UI
make serve

# Terminal 2 — streaming scorer (reads data/streaming_in/, writes scored JSON)
make stream

# Terminal 3 — synthetic drip producer into the streaming inbox
make feed
```

Open **http://127.0.0.1:8000/** in a browser: use **Score a review** (includes fraud explanation when scoring succeeds), watch **Streaming feed** refresh, and browse the ML sections fed by `/metadata`.

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
| POST | `/predict` | score one review (`fraud_explanation` optional object when generation succeeds) |
| POST | `/predict/batch` | score up to 1000 reviews (each item may include `fraud_explanation`) |
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

The home page (`make serve` → `/`) is a single-page dashboard backed by the REST
API above. It includes:

- **ML overview** — KPI-style cards from `/metadata` (train/test counts, best baseline
  picks from the comparison CSV when present, selected fraud threshold, fraud ROC-AUC).
- **Baseline comparison** — preview table and bar charts driven by baseline CSV
  data exposed through `/metadata`.
- **Threshold study** — precision/recall/F1 vs. fraud threshold using the sweep in
  `models/thresholds.json` when available. The chart uses an **adaptive Y-axis** so
  narrow metric bands remain readable (with a safe layout when curves are nearly flat).
- **Drift summary** — population cards and shift summaries when `reports/ml/drift_report.json`
  exists and is surfaced via `/metadata`.
- **Selected thresholds** — chosen fraud threshold payload from `/metadata`.
- **Model metadata** — compact summary of training runs and feature lists (no raw JSON dump).
- **Score a review** — calls `POST /predict` and shows the JSON response plus a **fraud
  explanation panel** (risk level, short summary, bullet reasons) when `fraud_explanation`
  is returned.
- **Operational views** — batch aggregates (products, suspicious reviewers) and a
  **streaming feed** of recent scored rows from the streaming job.

## Models and evaluation

**Training (production path)** — `make train` / `src/train/train.py`:

- **Sentiment** — TF-IDF (uni+bi-gram) + multinomial logistic regression; labels from
  star rating (1–2 negative, 3 neutral, 4–5 positive).
- **Fraud** — TF-IDF on cleaned review text plus **numeric behavioral features** passed
  into the classifier (exact column names are recorded in `models/meta.json` after a
  successful train). Gradient boosted trees; weak fraud labels come from batch heuristics
  in ETL. The API returns **probabilities**; optional threshold tuning writes
  `models/thresholds.json` for analysis and dashboard use.

Both runs log to MLflow under experiment `amazon_reviews_sentiment_fraud`.

**Extra evaluation scripts** (run after ETL has produced train/test Parquet — same
schema as production):

| Script | Role |
|--------|------|
| `python -m src.train.baselines` | Compare simpler/alternate models; writes `reports/ml/baseline_comparison.csv`. |
| `python -m src.train.threshold_tuning` | Sweep fraud thresholds; writes threshold study CSV + optional `models/thresholds.json`. |
| `python -m src.train.error_analysis` | Export misclassification samples for inspection. |
| `python -m src.train.drift_monitor` | Compare train vs holdout (or configured splits); writes drift JSON for the dashboard. |
| `python -m src.train.ablation_study` | Compare **full numeric + leakage-prone** fraud columns vs **reduced production** features on the same holdout; writes `reports/ml/fraud_ablation_*.csv/json` (offline study only). |

These were added so evaluation is **interpretable and realistic**: baselines show uplift
over naive choices, threshold tuning surfaces precision/recall tradeoffs, error exports
support qualitative review, and drift summarizes distribution shift—together with the
dashboard and the rule-based explanation layer on `/predict`.

`python -m src.train.error_analysis` expects **trained** `models/*.joblib` artifacts in addition
to Parquet splits (it loads the same pipelines as serving).

## Synthetic data generation

The proposal references a large public dataset. The generator (`src/ingest/generate_sample.py`)
produces Amazon-shaped **JSONL** locally so the full pipeline runs without a multi-GB download.
Phrase pools are sentiment-stratified; a configurable share of rows simulate fraud-like
patterns (e.g. duplicate bursts, velocity effects, reviewer rings, hard negatives). Large
runs: e.g. `--rows 1000000`; Spark scales with data size.

**Difficulty (`easy` \| `medium` \| `hard`):**

- **`medium` is the default** (`make ingest`, pipeline scripts unless overridden).
- **`easy`** — cleaner separation between classes; closer to a simplified legacy mix.
- **`medium`** — cross-class phrase overlap, typos/punctuation noise, mixed-sentiment wording,
  richer fraud mixture.
- **`hard`** — stronger overlap and subtler fraud-like patterns so models face a tougher task.

Harder modes increase realism for **sentiment confusion** and **fraud subtlety** while keeping
the **JSONL schema and downstream column layout unchanged**.

Examples:

```bash
python -m src.ingest.generate_sample --rows 30000 --fraud-share 0.06 --difficulty hard
make ingest   # uses medium by default
ROWS=50000 DIFFICULTY=easy bash scripts/run_pipeline.sh
```

## Results and interpretation

Metrics on **synthetic** data are for development and apples-to-apples comparisons—not a claim
of production accuracy. With **`medium`/`hard`** difficulty, holdout curves are **more
informative**: fraud threshold sweeps show clearer precision/recall tradeoffs than an
everywhere-easy generator, and sentiment/fraud error analysis becomes more meaningful.

The **dashboard + drift + explanation** pieces are meant for monitoring and transparency:
they do not replace rigorous evaluation on real Amazon data, but they make the current run
easier to interpret than headline numbers alone.

## Tests

```bash
make test
```

Covers text cleaning, the per-row serving feature path (including fraud
numeric columns), and the FastAPI endpoints with the trained models loaded
off disk.
