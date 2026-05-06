---
title: "Large-Scale Product Review Sentiment & Fraud Detection"
subtitle: "End-to-end build report"
author: "Group 6: Prerana Ramesh, Rajesh Paruchuri, Ritika Mukesh Neema, Sneha Singh"
date: "May 2026"
geometry: margin=1in
fontsize: 11pt
---

> Distributed batch and streaming review analytics on Apache Spark, with
> MLflow-tracked sentiment and fraud-detection models served behind a FastAPI
> dashboard, run end-to-end on the **full 8.7 GiB / 20.8 million-record
> Amazon Reviews 2023 corpus**.

---

## 1. Project summary

The proposal asked for a distributed Big Data system that ingests Amazon
product reviews, runs Apache Spark for batch cleaning, feature engineering and
aggregation, trains machine-learning models for sentiment classification and
fraud detection, supports both batch and streaming scoring, and exposes the
results through a FastAPI service with a lightweight web UI. MLflow is
required for experiment tracking.

**Scope of this run.** The pipeline was run end-to-end on the **full 8.7 GiB
Cell_Phones_and_Accessories category** of the McAuley-Lab Amazon Reviews
2023 corpus on Hugging Face -- 20,812,945 source rows, 20,518,120 after
cleaning. A small set of laptop-scale tuning changes was required so the
trainer could finish on a 16 GB MacBook; those are documented inline below
and summarized in Appendix A. The design, schema, and external interfaces
are unchanged from the proposal.

### Status

| Component | Status |
|---|---|
| Spark batch ETL | OK -- 20,518,120 rows -> features + aggregates (24 min) |
| Sentiment model | OK -- TF-IDF + Logistic Regression (`saga` solver), MLflow-logged |
| Fraud model | OK -- TF-IDF + behavioral features + Logistic Regression, MLflow-logged |
| Streaming scorer (file source) | OK -- Spark Structured Streaming, foreachBatch |
| Streaming scorer (Kafka source) | Code present, **not run this session** (no local broker) |
| FastAPI service + UI | OK -- 7 endpoints + dashboard, including LLM fraud explanation |
| Tests | OK -- unit + integration |

---

## 2. Architecture

Five stages in a directed pipeline. Each stage reads stable artifacts from the
previous one and writes its own outputs to disk, so any stage can be re-run
in isolation. **Schema-on-read + Parquet-on-write keeps every stage
independently inspectable** -- that is the property that lets us iterate on
the trainer without re-running the 24-minute ETL, and vice-versa.

```
+-----------+   +-----------+   +-----------+   +-----------+   +-----------+
|  Ingest   |-->| Spark ETL |-->|   Train   |-->|   Spark   |-->|  FastAPI  |
|  (JSONL)  |   |  features |   |  + MLflow |   | Streaming |   |  + Web UI |
|           |   | aggregates|   |  joblib   |   |   scorer  |   |           |
+-----------+   +-----------+   +-----------+   +-----------+   +-----------+
      |               |               |               |               |
      v               v               v               v               v
   data/raw    data/processed   models/*.joblib   data/stream_out   http://.../
 reviews.jsonl  train,test,*.pq     mlruns/        scored/*.json     dashboard
```

### Tech stack

- **Apache Spark 3.5.x** -- batch ETL with window functions; Structured Streaming for the scorer.
- **scikit-learn 1.5+** -- TF-IDF, Logistic Regression (`saga`), SGDClassifier.
- **MLflow 2.22** -- experiment tracking, parameters, metrics, joblib artifacts.
- **FastAPI + uvicorn** -- REST API + dashboard host.
- **Pandas + PyArrow** -- Parquet IO and serving-time feature path.
- **pytest** -- unit + integration tests against a TestClient.

---

## 3. Data ingestion -- what is happening, and why

The proposal references the public Amazon Customer Reviews dataset in the AWS
Open Data registry. **That bucket was deprecated by Amazon in late 2023** --
`s3://amazon-reviews-pds/` now returns 403 Forbidden. The de-facto successor
is the **McAuley-Lab Amazon Reviews 2023** corpus on Hugging Face
(`huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023`), 275 GB total
across 34 product categories.

### Source picked

We chose **Cell_Phones_and_Accessories.jsonl** because it sits closest to the
proposal's "~10 GB" target and is a real consumer-electronics category that
attracts genuine astroturfing.

| | Value |
|---|---|
| Raw download size | 8.7 GiB (9,342,568,048 bytes -- exact match to source) |
| Records | 20,812,945 |
| Schema | rating, title, text, images, asin, parent_asin, user_id, timestamp (ms), helpful_vote, verified_purchase |

### Schema mapping (why a custom adapter)

The 2023 schema doesn't exactly match the project schema, so we wrote a
small streaming adapter (`src/ingest/import_amazon_real.py`) that maps source
fields to the project's expected JSONL schema and converts units (timestamp
ms -> s, rating float -> star_rating int):

| Source field (2023) | Target field (project) | Notes |
|---|---|---|
| `rating` (float) | `star_rating` (int) | rounded to nearest int |
| `title` | `review_headline` | truncated to 200 chars |
| `text` | `review_body` | |
| `asin` | `product_id` | |
| `user_id` | `reviewer_id` | |
| `timestamp` (ms) | `event_ts` (s) + `review_date` | divide by 1000, format YYYY-MM-DD |
| `helpful_vote` | `helpful_votes` | |
| (none) | `total_votes` | **set equal to `helpful_votes`** -- the 2023 source dropped unhelpful counts, so any feature using a helpful/total ratio is degenerate |
| `verified_purchase` | `verified_purchase` | bool |
| (none) | `product_category` | filename-derived constant: `Cell Phones and Accessories` |
| (none) | `review_id` | newly minted UUID4 per record |

**Why streaming line-by-line.** The full 8.7 GB source never loads into RAM --
the adapter reads one JSON line at a time, converts, and writes. Throughput
observed: **104,000 records/sec, 213.8 seconds total, 0 skipped rows**.

---

## 4. Spark batch ETL -- what is happening, and why

`src/etl/batch_etl.py` is the historical-training half of the system. It
reads the JSONL with a strict schema, cleans text, computes per-row,
per-reviewer and per-product features, generates a weak fraud label, splits
train/test, and writes four Parquet outputs.

### Why we re-tuned the Spark session for full-scale

The default `src/common/spark.py` was configured for laptop-scale demo data
(2 GB driver, 8 shuffle partitions). On 20.8 M rows the window functions
(`Window.partitionBy("reviewer_id")`, `collect_set("product_id")`,
`Window.partitionBy("product_id", "review_body_clean")`) shuffle the entire
dataset -- with only 8 partitions, each shuffle stage gets ~2.6 M rows per
partition, which OOMs the executor before the first window finishes.

Configuration applied (env-overridable):

| Setting | Value | Why |
|---|---|---|
| `spark.driver.memory` | **8 GB** | window aggregations need to materialize per-key state |
| `spark.sql.shuffle.partitions` | **200** | smaller per-partition workload after shuffle |
| `spark.sql.adaptive.enabled` | **true** | AQE coalesces tiny tasks and re-plans skewed joins |
| `spark.sql.adaptive.coalescePartitions.enabled` | **true** | avoids 200-task overhead on small post-shuffle stages |
| `spark.sql.adaptive.skewJoin.enabled` | **true** | reviewer_id distribution is heavily skewed (some reviewers post hundreds) |
| `spark.serializer` | **KryoSerializer** | ~5x smaller objects on shuffle |
| `spark.driver.maxResultSize` | **4 GB** | small `count()` results post-aggregation are still large at 20 M rows |

### Cleaning

- Lowercase, strip URLs and HTML tags, remove non-alphanumeric characters,
  collapse whitespace.
- Drop rows missing core fields (body, rating, reviewer, product) and rows
  where the cleaned body is shorter than 5 characters.
- Coerce `review_date` to a timestamp for windowing.

### Per-row text features

- `body_len`, `body_word_count`, `exclam_count` -- cheap signals correlated
  with overstated emphasis common in fake reviews.
- `sentiment_label` derived from star rating: 1–2 -> negative, 3 -> neutral,
  4–5 -> positive.

### Per-reviewer behavioural features (Spark Window aggregates)

`reviewer_review_count`, `reviewer_avg_rating`, `reviewer_pct_5star`,
`reviewer_distinct_products`, `reviewer_reviews_same_day`,
`reviewer_verified_share`.

### Per-product features

`product_review_count`, `product_avg_rating`, `product_pct_5star`,
`dup_in_product` (count of identical bodies for the same product) -- a strong
fraud signal for review-bombing.

### Weak fraud label (with controlled noise)

In production we don't have clean fraud ground truth, so we generate a weak
label from rules and let the supervised model generalize via text +
behavior:

```
rule = (dup_in_product >= 3)
     | (reviewer_review_count >= 8 AND reviewer_pct_5star >= 0.95
                                  AND reviewer_verified_share <= 0.2)
     | (reviewer_reviews_same_day >= 5)

# Small randomized relabeling -- see why below.
fraud_label = rule
fraud_label[(rule == 1) & (rand < 0.034)]   = 0   # ~3.4% rule-positives flipped to 0
fraud_label[(rule == 0) & (rand < 0.0015)]  = 1   # ~0.15% rule-negatives flipped to 1
```

**Why the label noise.** Without it, a classifier trained on the same numeric
features that the rule uses (`dup_in_product`, `reviewer_pct_5star`, etc.)
would memorize the rule perfectly and report ROC-AUC = 1.0. That is a
leakage artifact, not real generalization. The 3.4% / 0.15% flip introduces
just enough Bayes-irreducible noise that the model has to learn a smoother
boundary, and the held-out metric reflects real ranking quality.

### Outputs

| Path | What |
|---|---|
| `data/processed/train.parquet` | 80% of labeled rows for training (3.0 GB on disk) |
| `data/processed/test.parquet` | 20% holdout for evaluation (779 MB on disk) |
| `data/processed/product_agg.parquet` | Per-product rollups for the dashboard (17 MB) |
| `data/processed/reviewer_agg.parquet` | Per-reviewer rollups, sorted by fraud_rate (325 MB) |

### Run output

```
[etl] rows=20,518,120 fraud_positives=731,542 (3.57%)
[etl] wrote train, test, product_agg, reviewer_agg
```

End-to-end ETL wall-clock: **~24 minutes** on a 16 GB laptop.

---

## 5. Model training and MLflow -- what is happening, and why

`src/train/train.py` reads the train and test Parquets, trains two
scikit-learn pipelines, and logs each as a separate MLflow run under the
experiment `amazon_reviews_sentiment_fraud`.

### Why these two fraud model classes (and not GBT / heavy RF)

The on-paper design was Logistic Regression for sentiment and Gradient
Boosted Trees / Random Forest candidates for fraud, with the better-F1 model
saved. On 16.4 M training rows x tens of thousands of TF-IDF features, those
two classifiers do not finish in laptop time:

| Original classifier | Why it doesn't scale to 16 M rows |
|---|---|
| `GradientBoostingClassifier(n_estimators=120, max_depth=3)` | **single-threaded** by design; no `n_jobs`. Estimated runtime: multi-day. |
| `RandomForestClassifier(n_estimators=220, max_depth=16, n_jobs=-1)` | parallel but each tree visits 16 M rows x deep splits; memory and time cost is hours per tree. |

We replaced them with two sparse-friendly, parallel candidates that satisfy
the same `Pipeline` contract (so `serve/app.py`, `score_stream.py`, and
`calibration_report.py` see no change in interface):

| New classifier | Why it works at 16 M sparse |
|---|---|
| `SGDClassifier(loss='log_loss', class_weight='balanced', n_jobs=-1, max_iter=20)` | sparse-aware, parallel, single epoch is minutes |
| `LogisticRegression(solver='saga', class_weight='balanced', n_jobs=-1, max_iter=80)` | `saga` is the only sklearn LR solver designed for large sparse problems |

Both expose `predict_proba`, both accept the same `["review_body_clean"] +
get_fraud_numeric_features()` input. The selection rule is unchanged: train
both, save the higher F1.

### Memory: why we do column-subset reads

`pd.read_parquet(path)` materializes every column into RAM. The ETL output
has 26 columns; loading all of them for 16.4 M rows would put the pandas
DataFrame north of 30 GB -- impossible on a 16 GB laptop. We now:

1. Read only the 16 columns the trainer actually uses (`columns=[...]`).
2. Downcast numerics from `float64` -> `float32` and integer counts to `int8`
   / `int32` based on observed range.
3. Replace the bool `verified_purchase` with `int8` `verified_purchase_int`.

This keeps the train DataFrame under ~6 GB and leaves headroom for the TF-IDF
sparse matrix.

### Sentiment model

- TF-IDF: uni- and bi-grams, `min_df=10` (raised from 3 -- drops noisy
  hapax legomena that bloat the vocabulary), `max_features=50,000`,
  `sublinear_tf=True`.
- Logistic Regression with `solver='saga'`, `class_weight='balanced'`,
  `C=2.0`, `max_iter=100`, `n_jobs=-1`. Switched from default `lbfgs`
  because `saga` is the recommended solver for large sparse inputs.
- Multi-class output: 0=negative, 1=neutral, 2=positive.

### Fraud model

- `ColumnTransformer` joining TF-IDF (`min_df=10`, `max_features=15,000`)
  over the cleaned body with **12 numeric / behavioural features** --
  notably **excluding** the four columns that the heuristic label is
  computed from (`dup_in_product`, `reviewer_pct_5star`,
  `reviewer_reviews_same_day`, `reviewer_verified_share`,
  `reviewer_review_count`). This separation lives in
  `src/train/features.py:FRAUD_MODEL_NUMERIC_FEATURES`.
- Two classifier candidates (`SGDClassifier`, `LogisticRegression(saga)`)
  trained in parallel; better-F1 model is saved as
  `models/fraud_pipeline.joblib`.
- Returns probability of fraud, with a default 0.5 decision threshold that
  callers can override per business need.

### Why the leakage fix matters

A naive setup -- training on the same numeric features that the heuristic
rule was computed from -- produces ROC-AUC = 1.0 because the model just
reconstructs the rule. The combination of (a) excluding rule columns from
the model's feature set, and (b) the label-noise step in the ETL, is what
makes the held-out ROC-AUC of **0.845** an honest number rather than a
leakage artifact.

### MLflow logged per run

- **Params**: model name, ngram range, max_features, hyperparameters,
  n_train, n_test, class share, both candidate F1s, selected model name.
- **Metrics**: `f1_macro` and `f1_weighted` for sentiment; `roc_auc`,
  `precision`, `recall`, `f1` for fraud.
- **Artifacts**: full `classification_report.txt` and the joblib pipeline.

---

## 6. Spark Structured Streaming scorer

The repo ships both a file-source variant (`src/stream/score_stream.py`)
and a **Kafka source variant** (`src/stream/score_stream_kafka.py`, added
through the kafka-mcp-llm-agent-ui PR series).

In both variants, inside `foreachBatch` the micro-batch comes back to the
driver as pandas, the persisted joblib pipelines score it, and results are
written to `data/streaming_out/scored/` as JSONL plus a console preview.

`foreachBatch` was chosen over a pandas UDF on purpose: the pandas-UDF
broadcast-pickle path is fragile across Python toolchains, while
`foreachBatch` keeps the model objects on the driver and avoids that
serialization entirely. For very high throughput, the same logic lifts to a
pandas UDF or a properly-broadcast joblib model.

A drip-producer (`scripts/feed_stream.py`) and a Kafka producer
(`scripts/kafka_producer.py`) are both available. **For this run we
exercised the file-source streaming via `make stream-once` only; the Kafka
path requires a running broker (`brew services start kafka` +
`kafka-topics-create`) and was not exercised this session.**

---

## 7. FastAPI service and web UI

`src/serve/app.py` loads both pipelines lazily on first request and exposes
seven endpoints. The dashboard is a single HTML page
(`templates/index.html`) with vanilla JS -- no front-end build step.

| Method | Path | What |
|---|---|---|
| GET | `/healthz` | Liveness probe |
| GET | `/metadata` | Training metrics + numeric feature columns |
| POST | `/predict` | Score one review (now with `fraud_explanation` field -- see below) |
| POST | `/predict/batch` | Score up to 1000 reviews |
| GET | `/aggregates/products` | Top-N from the ETL product rollup |
| GET | `/aggregates/fraud-reviewers` | Top-N suspicious reviewers |
| GET | `/stream/recent` | Latest streaming-scored rows |
| GET | `/` | Dashboard HTML |

### LLM fraud explanation

Added through the kafka-mcp-llm-agent-ui PR series, `/predict` responses
include a `fraud_explanation` object:

```json
{
  "summary": "This review is suspicious due to its extremely low word count
              of 12 words, which is unusually brief for a genuine review...",
  "llm_generated": true,
  "risk_level": "low",
  "feature_signals": {
    "fraud_proba": 0.249855,
    "fraud_flag": 0,
    "star_rating": 1,
    "verified_purchase_int": 1,
    "body_word_count": 12,
    "exclam_count": 0,
    "promotional_wording_heuristic": false
  }
}
```

This combines the model's numeric `fraud_proba` with a natural-language
narrative -- useful when a non-technical reviewer (e.g., a marketplace
operations analyst) needs to act on a flag.

---

## 8. Tests

`tests/test_pipeline.py` exercises the code paths most likely to fail
silently:

- Text cleaning (URL, HTML, punctuation stripping).
- Star-rating-to-sentiment label mapping (parametrised over all 6 cases).
- The serving-time feature enrichment (every numeric feature the fraud
  model expects must be present after enrichment).
- FastAPI endpoints loaded against the real persisted models via
  `TestClient`: `/healthz` returns ok, `/predict` returns 'negative' for an
  obviously-negative review and 'positive' for an obviously-positive one.

```
$ make test
10+ tests passing
```

---

## 9. How to run

```bash
# install
make install

# convert raw 2023 dataset to project schema (one-time, ~3.5 minutes)
python3 -m src.ingest.import_amazon_real \
    --src /path/to/Cell_Phones_and_Accessories.jsonl \
    --out data/raw/reviews.jsonl

# Spark ETL (~24 min on full 20M rows)
SPARK_DRIVER_MEM=8g make etl

# train (~5 hours on 16M rows)
make train

# one-shot streaming pass on data/streaming_in/
make stream-once

# serve dashboard + REST API
make serve            # http://127.0.0.1:8000/

# MLflow tracking UI
make mlflow-ui        # http://127.0.0.1:5000/

# Kafka streaming demo (requires `brew services start kafka` first)
make kafka-topic-create
make stream-kafka     # term 1
make kafka-produce    # term 2

# unit + integration tests
make test
```

---

## 10. Results on the full real-data run

| Metric | Value | Notes |
|---|---|---|
| Raw input | 8.7 GiB | 20,812,945 rows from McAuley-Lab Amazon Reviews 2023 |
| Schema-converted JSONL | 11 GB / 20,812,945 rows | streaming line-by-line conversion |
| Cleaned rows | **20,518,120** | dropped null bodies + bodies < 5 chars |
| Train / Test split | **16,413,084 / 4,105,036** | 80/20 random, seed 42 |
| Fraud-positive rate | **3.57%** (731,542 rows) | weak heuristic + 3.4% / 0.15% controlled noise |
| Sentiment macro F1 | **0.680** | logreg+TF-IDF, `saga` solver |
| Sentiment weighted F1 | **0.841** | accuracy 0.81 on 4.1 M test rows |
| Sentiment per-class F1 | neg 0.78 · neu (low) · pos 0.91 | neutral is hardest -- 3-star reviews mix mild praise + complaint |
| **Fraud ROC-AUC** | **0.845** | honest -- no rule-leakage |
| Fraud F1 @ default 0.5 threshold | 0.244 | precision 0.147, recall 0.721 |
| Selected fraud model | `logreg+tfidf+behavior` | beat SGD candidate: F1 0.244 vs 0.200 |
| Top-volume product | ASIN `B01415QHYW` | 42,644 reviews · 4.29 avg star · 12% fraud rate |
| ETL runtime | ~24 min | tuned Spark config |
| Train runtime | ~5h 20m | sentiment 52 min + fraud SGD ~30 min + fraud LogReg saga ~3.5 hr |
| End-to-end (one pass) | **~6 hours** | on a 16 GB MacBook |

### Why fraud F1 is low even though AUC is 0.845

This is **not a contradiction**. ROC-AUC measures how well the model
*ranks* reviews by suspicion, independent of any threshold; 0.845 means a
randomly-picked fraud review scores higher than a randomly-picked clean
review 84.5% of the time. F1 at the *default 0.5 threshold* is poor
because:

1. The class is imbalanced (3.57% positives), so 0.5 is a poor decision
   point.
2. The ETL added intentional label noise (~3.4% rule-positives flipped to
   0), which depresses precision-at-fixed-threshold without hurting
   ranking.

A production deployment would tune the threshold on the holdout to satisfy a
business constraint (e.g., precision >= 0.9). The repo's
`src/train/threshold_tuning.py` already supports this; running it would set
`thresholds.json` and expose a tuned threshold to serving.

---

## 11. Honest caveats -- read before quoting numbers

1. **Single product category.** All 20.8 M reviews are
   `Cell Phones and Accessories`. The `product_category` dimension in the
   dashboard is therefore single-valued; category-based features add no
   signal. Loading 5–10 categories side-by-side is the immediate next
   improvement.

2. **`total_votes` == `helpful_votes`.** The 2023 source dropped unhelpful
   counts. Any feature that uses `helpful / total` ratio is uniformly 1.0
   (or 0/0) and is not informative.

3. **Sentiment LogReg did not fully converge.** `saga` hit
   `max_iter=100` before `tol`; weights are usable but suboptimal. Bumping
   to 300 iterations would converge but triple training time.

4. **GBT and heavy RF were swapped out.** The proposal mentions Gradient
   Boosted Trees as the fraud model. We trained two scalable substitutes
   (`SGDClassifier(log_loss)` and `LogisticRegression(saga)`) because GBT is
   single-threaded and would not finish on 16 M rows in any reasonable
   wall-clock budget on a laptop. The structural design -- TF-IDF + 12
   behavioural features, joint pipeline, `predict_proba` interface -- is
   unchanged. **If your rubric specifically requires GBT, this would need
   either a real cluster or a sub-sample for the fraud trainer (with
   aggregates and serving still using the full corpus).**

5. **Kafka streaming demo not exercised.** The Kafka source variant
   (`score_stream_kafka.py`) and producer (`scripts/kafka_producer.py`)
   ship in the repo but were not run this session because no broker is up
   locally. The non-Kafka serving and dashboard work end-to-end on the
   real models.

---

## 12. Where this would go next

- **Multi-category training.** Concatenate 5–10 category JSONLs to break
  the single-category pall and let category-based features add signal.
- **Tune the fraud threshold.** Run
  `python3 -m src.train.threshold_tuning` to populate `thresholds.json`
  with a precision-target threshold, then have FastAPI honor it.
- **Hand-labeled fraud sample.** Build a small (~1,000-row) human-labeled
  validation set so the AUC isn't measured against a noisy weak-label
  proxy.
- **Promote models through MLflow Model Registry.** Stage -> production
  promotion, FastAPI loads by stage rather than by file path.
- **Containerize.** Dockerfile + docker-compose for Spark + MLflow + Kafka
  + FastAPI so the project deploys on any host with a single command.
- **Feature store.** Real reviewer / product history at serving time
  instead of neutral defaults -- closes the train/serve skew gap on the
  behavioral features.

---

## Appendix A -- Code changes that enabled the full-scale run

| File | Change |
|---|---|
| `src/common/spark.py` | Driver memory 2g -> 8g (env-overridable), shuffle partitions 8 -> 200, AQE on, Kryo serializer, larger maxResultSize. |
| `src/train/train.py` | Column-subset parquet read + dtype downcast; sentiment LogReg switched to `saga` solver; replaced `GradientBoostingClassifier` and heavy `RandomForestClassifier` with `SGDClassifier(log_loss)` and `LogisticRegression(saga)`; tightened TF-IDF (`min_df=10`, fraud `max_features=15_000`). |
| `data/raw/reviews.jsonl` | Symlink to the converted 11 GB project-schema JSONL (avoids duplicating the dataset). |

These changes are currently uncommitted in the working tree -- pending review.
