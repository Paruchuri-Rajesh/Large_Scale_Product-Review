---

## Dataset

| Property | Value |
|----------|-------|
| Source | McAuley-Lab Amazon Reviews 2023 (Hugging Face) |
| Category | Cell Phones & Accessories |
| Raw size | 11 GiB JSONL |
| Source records | 20,812,945 |
| Cleaned records | 20,518,120 |
| Train / Test split | 16,413,084 / 4,105,036 (80/20, seed 42) |
| Fraud-positive rate | 3.57% (731,542 rows) |

### Schema Mapping

During ingestion: `rating` → `star_rating` (int), `timestamp` (ms) → `event_ts` (sec), `asin` → `product_id`. Note: the dataset lacks unhelpful-votes, so `total_votes = helpful_votes` and helpfulness-ratio features are not used.

---

## Pipeline Stages

### 1. Ingest / ETL (AWS)

Raw JSONL (11 GB) is uploaded to **Amazon S3** (`cellphonesandaccessories` bucket). An **AWS Glue** job (`json-to-parquet`) converts the JSONL to compressed Parquet (about 3.3 GB) in roughly 2 minutes using 4 DPUs. **Amazon Athena** is connected on top of the Parquet for SQL-based validation (record counts, sample rows, schema checks).

### 2. Spark Batch ETL

The batch ETL module (`src/etl/batch_etl.py`) reads the raw JSONL via a strict schema, applies text cleaning, computes window-based features, generates a weak fraud label, performs an 80/20 random split, and writes four Parquet outputs:

- `train.parquet`
- `test.parquet`
- `product_agg.parquet`
- `reviewer_agg.parquet`

**Spark configuration:** 8 GB driver memory, 200 shuffle partitions, Adaptive Query Execution (AQE), Kryo serialization. Without these settings, window aggregations over 20.8M rows OOM on a 16 GB laptop. **Total ETL wall-clock: about 24 minutes.**

### 3. Feature Engineering

Features are grouped into three categories:

**Per-row (text-based):** `body_len`, `body_word_count`, `exclam_count`

**Reviewer-level (behavioral):** total reviews written, average rating given, percentage of 5-star reviews, number of distinct products reviewed, same-day post count, share of verified purchases

**Product-level:** total review count, average rating, percentage of 5-star reviews, duplicate review text count

### 4. Weak Fraud Label and Leakage Prevention

The dataset does not include true fraud labels, so weak labels are generated using rules:

- Repeated/duplicate review text for the same product
- A reviewer posting many reviews with mostly 5-star ratings and low verified purchases
- Multiple reviews from the same user on the same day

To prevent the model from simply memorizing these rules, two mitigations are applied:

1. **Remove rule-based features** from training input (any feature that algebraically mirrors the labeling logic is excluded)
2. **Add controlled label noise** by randomly flipping a small percentage of labels

This pushes the fraud model toward learning more generalizable patterns and yields an honest ROC-AUC of 0.845 rather than an artificially inflated score.

### 5. Model Training

Both models are trained via scikit-learn Pipelines and logged to MLflow under experiment `amazon_reviews_sentiment_fraud`.

**Sentiment Model**
- TF-IDF vectorization (unigrams + bigrams, vocab capped at 50,000)
- Logistic Regression (saga solver, class balancing)
- 3 classes: 0 = Negative (1–2★), 1 = Neutral (3★), 2 = Positive (4–5★)

**Fraud Model**
- ColumnTransformer concatenating sparse TF-IDF (15k features) + 12 numeric behavioral features
- Two candidates trained in parallel: SGDClassifier (log loss) and LogisticRegression (saga); the higher-F1 model is saved
- Originally planned GradientBoostingClassifier was dropped because it is single-threaded and cannot complete in reasonable time on 16.4M rows

Memory optimization: only required columns loaded, float64 → float32 downcast, compact integer formats. Everything tracked in MLflow (parameters, metrics, classification reports, joblib pipelines).

### 6. Streaming Scoring

Spark Structured Streaming (`src/stream/score_stream_kafka.py`) subscribes to the Kafka topic `reviews`, scores each micro-batch via `foreachBatch` using the persisted joblib pipelines, and writes JSON output. The `foreachBatch` pattern avoids the fragile pandas-UDF broadcast path.

**Throughput:** about 153 messages/second (2,000 messages scored in 13 s). End-to-end latency: about 5 seconds with a 5-second trigger interval.

---

## Model Diagnostics

### Threshold Tuning

Default 0.5 threshold yields high recall but poor precision because fraud is rare (3.57%). Sweeping thresholds from 0.05 to 0.95:

| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|-----|
| 0.50 | 0.147 | 0.721 | 0.244 |
| 0.60 | 0.193 | 0.635 | 0.296 |
| 0.70 | 0.242 | 0.549 | 0.336 |
| **0.80** | **0.292** | **0.457** | **0.356** |
| 0.90 | 0.358 | 0.265 | 0.305 |

Threshold 0.80 is saved and used in production.

### Probability Calibration

Raw model probabilities are not reliable as-is. Both Platt scaling and isotonic regression were tested on N=2,052,518:

| Method | Brier | ECE | ROC-AUC |
|--------|-------|-----|---------|
| Raw | 0.143 | 0.302 | 0.846 |
| Sigmoid (Platt) | 0.030 | 0.002 | 0.846 |
| **Isotonic (best)** | **0.029** | **0.000146** | **0.846** |

Isotonic calibration reduces ECE by three orders of magnitude without affecting ranking quality.

### Distribution Drift

Population Stability Index (PSI) was used to compare training and test distributions; drift is minimal as expected for a random 80/20 split. The same monitor can be reused in production to detect when retraining is needed.

---

## Serving Surfaces

### A. FastAPI REST Service

Models are loaded lazily via joblib for fast startup. Seven endpoints:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/healthz` | Liveness probe |
| GET | `/metadata` | Training metrics, feature columns, drift report |
| POST | `/predict` | Score single review; returns sentiment, fraud_proba, LLM explanation |
| POST | `/predict/batch` | Score up to 1,000 reviews |
| GET | `/aggregates/products` | Top-N products from ETL rollup |
| GET | `/aggregates/fraud-reviewers` | Top-N suspicious reviewers |
| GET | `/stream/recent` | Latest Kafka-scored rows |

### B. Streamlit Dashboard

Visual interface for overall metrics, manual review scoring, top products, suspicious reviewers, and streaming results.

### C. MCP Server (Claude Desktop)

The MCP server (`src/serve/mcp_server.py`) exposes three tools to Claude Desktop:

- `predict_review` — score a single review for sentiment and fraud
- `get_fraud_reviewers` — top suspicious reviewers from ETL aggregates
- `get_top_products` — top products by review count or fraud rate

Users can ask natural-language questions like *"What are the top 5 most fraudulent products?"* and Claude calls the appropriate tool. Also works with Cowork for non-developer access.

### D. LLM Fraud Explanation

After scoring, **Ollama with Llama 3.2** generates a human-readable explanation of why a review may be suspicious (duplicate wording, unusual reviewer behavior, extreme ratings, etc.). A rule-based fallback ensures the API still returns an explanation if the LLM is unavailable.

### E. Autonomous Review Auditor

Implemented in `src/agents/review_auditor.py`. An LLM-driven multi-step loop (up to 8 iterations) that calls tools — `get_product_aggregate`, `get_product_reviews`, `get_top_fraud_reviewers`, `score_review` — to investigate a product and produce a structured risk verdict. Tested on ASIN B01415QHYW (42,644 reviews, ~12% fraud rate), correctly flagged as high-risk after only a few tool calls.

### F. Kafka Fan-out

With `PUBLISH_PREDICT_TO_KAFKA=1`, the FastAPI service returns a synchronous prediction *and* simultaneously publishes the review to Kafka for streaming-based processing. Both paths produce consistent results.

---

## Database Schema (MySQL)

Eight core tables organized into two zones:

**Source & Processing**
- `RAW_REVIEW` — every ingested record, keyed by `review_id`
- `FEATURED_REVIEW` — ETL output with derived features, weak `fraud_label`, cleaned body
- `PRODUCT_AGG` — per-product rollup (review_count, avg_rating, pct_5star, fraud_rate)
- `REVIEWER_AGG` — per-reviewer rollup (reviewer_review_count, avg_rating, fraud_rate, verified_share)

**Modeling & Artifacts**
- `MLFLOW_RUN` — training experiments (run_id, hyperparameters, metrics, artifact paths)
- `MODEL_ARTIFACT` — serialized joblib pipeline bytes + `META_JSON` + `THRESHOLDS_JSON`
- `SCORED_REVIEW` — every inference (raw fraud_proba, fraud_flag, sentiment label, batch_id)
- `DIAGNOSTIC_REPORT` — calibration and drift reports keyed by `report_kind`

---

## Experimental Results

### Full-Scale Spark Performance

A Spark aggregation over all 20.5M reviews (about 3.8 GB processed) completes in **30.8 seconds** in four stages on a single machine. Key aggregates:

- Average product rating: 4.008
- Overall fraud rate: 3.57%
- Most-reviewed product: B01415QHYW (42,644 reviews, ~12% fraud)
- Highest-fraud product with sufficient volume: B00BI9AKJI (41.3% fraud rate, 363 reviews)

### Test Suite

11 unit and integration tests cover text cleaning, sentiment label mapping, feature presence at inference time, FastAPI endpoint responses, and fraud explanation format. All pass in ~13.4 seconds.

### Runtime Summary (16 GB laptop)

| Stage | Time |
|-------|------|
| Data ingestion + schema conversion | ~3.5 min |
| Spark ETL | ~24 min |
| Sentiment model training | ~52 min |
| Fraud model training (SGD) | ~30 min |
| Fraud model training (LogReg) | ~3.5 hr |
| Diagnostics (threshold, calibration, drift) | ~15 min |
| **Total end-to-end** | **~6 hours** |

---

## Tech Stack

- **Cloud / Storage:** AWS S3, AWS Glue, Amazon Athena
- **Big Data:** Apache Spark 3.5 (batch + Structured Streaming)
- **Streaming:** Apache Kafka (KRaft mode, no ZooKeeper)
- **ML:** scikit-learn (TF-IDF, Logistic Regression, SGDClassifier)
- **Experiment Tracking:** MLflow
- **Serving:** FastAPI, Streamlit
- **LLM:** Ollama + Llama 3.2
- **Agents / Tooling:** MCP server, Claude Desktop, Cowork
- **Database:** MySQL

---

## Limitations

- **Single category** — only Cell Phones & Accessories; category-level features add little value. Expanding to multiple categories would improve generalization.
- **No unhelpful-votes** in the dataset; helpfulness-ratio features are meaningless.
- **Convergence** — sentiment LogReg (saga) hits max iterations before fully converging.
- **No true fraud labels** — training and evaluation use weak heuristic labels, so reported performance may not fully reflect real-world accuracy.
- **No GBT** — Gradient Boosted Trees were originally planned but not practical at this scale on a single machine; linear models scale better but may miss complex patterns.

---

## Future Work

- Integrate the tuned threshold (0.80) and calibrated probabilities directly into the serving layer
- Expand to multiple product categories for more meaningful category features
- Build a small hand-labeled fraud benchmark (~1,000 reviews) for honest evaluation
- Package the full pipeline with Docker Compose for single-command deployment
- Use the MLflow Model Registry for versioning and controlled promotion to production
- Replace static behavioral features with a real-time feature store

---

## References

1. D. Kotzias, M. Denil, N. de Freitas, and P. Smyth, "From group to individual labels using deep features," in *Proc. ACM KDD*, 2015, pp. 597–606.
2. M. Pontiki et al., "SemEval-2014 Task 4: Aspect based sentiment analysis," in *Proc. SemEval*, 2014, pp. 27–35.
3. A. Mukherjee, B. Liu, and N. Glance, "Spotting fake reviewer groups in consumer reviews," in *Proc. WWW*, 2012, pp. 191–200.
4. G. Fei, A. Mukherjee, B. Liu, M. Hsu, M. Castellanos, and R. Ghosh, "Exploiting burstiness in reviews for review spammer detection," in *Proc. AAAI ICWSM*, 2013.
5. X. Meng et al., "MLlib: Machine learning in Apache Spark," *J. Mach. Learn. Res.*, vol. 17, pp. 1–7, 2016.
6. M. Zaharia et al., "Apache Spark: A unified engine for big data processing," *Commun. ACM*, vol. 59, no. 11, pp. 56–65, 2016.
7. J. Ni, J. Li, and J. McAuley, "Justifying recommendations using distantly-labeled reviews and fine-grained aspects," in *Proc. EMNLP-IJCNLP*, 2019, pp. 188–197.
8. S. J. Pan and Q. Yang, "A survey on transfer learning," *IEEE Trans. Knowl. Data Eng.*, vol. 22, no. 10, pp. 1345–1359, 2010.
9. J. Platt, "Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods," *Adv. Large Margin Classifiers*, vol. 10, no. 3, pp. 61–74, 1999.
