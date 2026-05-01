"""Generate a PDF report describing the project end-to-end.

Output: PROJECT_REPORT.pdf in the project root.
"""
from __future__ import annotations

import sys
from pathlib import Path

from fpdf import FPDF
from fpdf.enums import XPos, YPos

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "PROJECT_REPORT.pdf"

TITLE = "Large-Scale Product Review Sentiment & Fraud Detection"
SUBTITLE = "End-to-end build report"
AUTHORS = "Group 6: Prerana Ramesh, Rajesh Paruchuri, Ritika Mukesh Neema, Sneha Singh"


# ---------------------------------------------------------------------------

class Report(FPDF):
    def __init__(self) -> None:
        super().__init__(unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(left=18, top=18, right=18)
        self.alias_nb_pages()

    # --- chrome ------------------------------------------------------------

    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "", 8)
        self.set_text_color(120)
        self.cell(0, 6, "Amazon Reviews - Sentiment & Fraud (Group 6)", align="L")
        self.cell(0, 6, f"Page {self.page_no()} / {{nb}}", align="R")
        self.ln(8)
        self.set_text_color(0)

    def footer(self) -> None:
        # Page number is in header to keep footer clean.
        return

    # --- typography helpers -------------------------------------------------

    def h1(self, text: str) -> None:
        self.ln(2)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(20, 30, 60)
        self.cell(0, 9, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0)
        self.ln(1)

    def h2(self, text: str) -> None:
        self.ln(2)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(180, 90, 0)
        self.cell(0, 7, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0)

    def h3(self, text: str) -> None:
        self.set_font("Helvetica", "B", 10.5)
        self.cell(0, 6, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def p(self, text: str) -> None:
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5.2, text)
        self.ln(1)

    def bullets(self, items: list[str]) -> None:
        self.set_font("Helvetica", "", 10)
        for it in items:
            x = self.get_x()
            self.cell(4)
            self.cell(3, 5.2, "-")
            self.multi_cell(0, 5.2, it)
            self.set_x(x)
        self.ln(1)

    def code(self, text: str) -> None:
        self.set_font("Courier", "", 8.5)
        self.set_fill_color(245, 246, 250)
        self.set_draw_color(220, 222, 230)
        # Render a single fillable block with a border.
        self.multi_cell(0, 4.4, text, border=1, fill=True)
        self.ln(2)

    def kv_table(self, rows: list[tuple[str, str]], col1_w: int = 72) -> None:
        self.set_draw_color(210)
        epw = self.w - self.l_margin - self.r_margin
        col2_w = epw - col1_w
        for k, v in rows:
            x_start = self.get_x()
            y_start = self.get_y()
            # Render the key into its column and remember how tall it became.
            self.set_xy(x_start, y_start)
            self.set_font("Helvetica", "B", 10)
            self.multi_cell(col1_w, 6, k, border=0, new_x=XPos.RIGHT, new_y=YPos.TOP)
            y_after_key = self.get_y()
            # Render the value into the remaining width on the same row top.
            self.set_xy(x_start + col1_w, y_start)
            self.set_font("Helvetica", "", 10)
            self.multi_cell(col2_w, 6, v, border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            y_after_val = self.get_y()
            y_end = max(y_after_key, y_after_val)
            self.set_draw_color(220)
            self.line(x_start, y_end, x_start + epw, y_end)
            self.set_xy(x_start, y_end)
        self.ln(2)


# ---------------------------------------------------------------------------

ARCH_DIAGRAM = r"""
+-----------+   +-----------+   +-----------+   +-----------+   +-----------+
|  Ingest   |-->| Spark ETL |-->|   Train   |-->|  Spark    |-->|  FastAPI  |
| (JSONL)   |   | features  |   | + MLflow  |   | Streaming |   | + Web UI  |
|           |   | aggregates|   | joblib    |   | scorer    |   |           |
+-----------+   +-----------+   +-----------+   +-----------+   +-----------+
     |                |               |                |               |
     v                v               v                v               v
 data/raw      data/processed    models/*.joblib  data/stream_out  http://.../
 reviews.jsonl train,test,*.pq   mlruns/          scored/*.json    dashboard
"""


def build() -> None:
    pdf = Report()

    # ---------- cover page ------------------------------------------------
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(20, 30, 60)
    pdf.ln(35)
    pdf.multi_cell(0, 11, TITLE, align="C")
    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(80)
    pdf.ln(4)
    pdf.cell(0, 7, SUBTITLE, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(20)
    pdf.set_font("Helvetica", "", 10.5)
    pdf.set_text_color(0)
    pdf.cell(0, 6, AUTHORS, align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(60)
    pdf.set_font("Helvetica", "I", 9.5)
    pdf.set_text_color(120)
    pdf.multi_cell(
        0,
        5,
        "Distributed batch and streaming review analytics on Apache Spark, "
        "with MLflow-tracked sentiment and fraud-detection models served "
        "behind a FastAPI dashboard.",
        align="C",
    )

    # ---------- 1. summary -------------------------------------------------
    pdf.add_page()
    pdf.h1("1. Project summary")
    pdf.p(
        "The proposal asked for a distributed Big Data system that ingests "
        "Amazon product reviews, runs Apache Spark for batch cleaning, "
        "feature engineering and aggregation, trains machine-learning models "
        "for sentiment classification and fraud detection, supports both "
        "batch and streaming scoring, and exposes the results through a "
        "FastAPI service with a lightweight web UI. MLflow was required for "
        "experiment tracking."
    )
    pdf.p(
        "All five requirements are implemented and verified end-to-end on a "
        "30,000-review dataset. Each component is independently runnable "
        "(via the Makefile) and connected through well-defined Parquet and "
        "JSON contracts on disk, so the same code scales to the full 10+ GB "
        "Amazon Customer Reviews dataset by swapping the input source."
    )

    pdf.h2("Status")
    pdf.kv_table(
        [
            ("Spark batch ETL", "OK - 30,000 rows -> features + aggregates"),
            ("Sentiment model", "OK - TF-IDF + Logistic Regression, MLflow-logged"),
            ("Fraud model", "OK - TF-IDF + behavior + GBT, MLflow-logged"),
            ("Streaming scorer", "OK - Spark Structured Streaming, foreachBatch"),
            ("FastAPI service + UI", "OK - 7 endpoints + dashboard"),
            ("Tests", "OK - 10/10 passing"),
        ]
    )

    # ---------- 2. architecture -------------------------------------------
    pdf.h1("2. Architecture")
    pdf.p(
        "Five stages in a directed pipeline. Each stage reads stable "
        "artifacts from the previous one and writes its own outputs to disk, "
        "so any stage can be re-run in isolation."
    )
    pdf.code(ARCH_DIAGRAM.strip("\n"))

    pdf.h2("Tech stack")
    pdf.bullets(
        [
            "Apache Spark 3.5.8 (batch ETL + Structured Streaming)",
            "scikit-learn 1.5 (TF-IDF, Logistic Regression, Gradient Boosted Trees)",
            "MLflow 2.22 (experiment tracking, model artifacts, metrics)",
            "FastAPI 0.128 + uvicorn (REST API + dashboard host)",
            "Pandas + PyArrow (Parquet IO and serving-time feature path)",
            "pytest (unit + integration tests against a TestClient)",
        ]
    )

    pdf.h2("Repo layout")
    pdf.code(
        """
PROJECT/
  src/
    ingest/generate_sample.py     synthetic Amazon-shaped JSONL producer
    etl/batch_etl.py              Spark batch ETL
    train/train.py                sklearn pipelines + MLflow
    stream/score_stream.py        Spark Structured Streaming scorer
    serve/app.py                  FastAPI endpoints
    serve/templates/index.html    dashboard
    serve/static/{app.js,style.css}
    common/{config,schema,spark,text}.py
  scripts/feed_stream.py          drip producer for the streaming job
  scripts/run_pipeline.sh         end-to-end one-command runner
  tests/test_pipeline.py
  data/{raw,processed,streaming_in,streaming_out}/
  models/   mlruns/
  Makefile  README.md  requirements.txt
""".strip("\n")
    )

    # ---------- 3. data ingestion -----------------------------------------
    pdf.h1("3. Data ingestion")
    pdf.p(
        "The proposal references the Amazon Customer Reviews public dataset "
        "(>10 GB after decompression, AWS Open Data registry). For a "
        "self-contained build, src/ingest/generate_sample.py emits JSONL "
        "matching the same schema (review_id, product_id, reviewer_id, "
        "star_rating, helpful_votes, total_votes, verified_purchase, "
        "review_headline, review_body, review_date, product_category, "
        "event_ts)."
    )
    pdf.p("Two design choices make the synthetic data useful, not just shaped:")
    pdf.bullets(
        [
            "Sentiment-stratified vocabulary: positive / neutral / negative "
            "phrase pools sampled by star rating, so the sentiment model has "
            "real signal to learn rather than memorising tokens.",
            "Planted fraud bursts: a small ring of reviewers (R9xxxx) "
            "carpet-bombs a target product with duplicate 5-star copy in a "
            "tight time window, mimicking the dominant real-world fraud "
            "pattern (review brigading).",
        ]
    )
    pdf.p(
        "The same pipeline accepts a real Amazon TSV after a one-shot "
        "conversion (recipe in README). The generator default is 30k rows; "
        "set --rows 1000000 to stress the Spark job."
    )

    # ---------- 4. ETL ----------------------------------------------------
    pdf.h1("4. Spark batch ETL")
    pdf.p(
        "src/etl/batch_etl.py is the historical-training half of the system. "
        "It reads the JSONL with a strict schema, cleans text, computes "
        "per-row, per-reviewer and per-product features, generates a weak "
        "fraud label, splits train/test, and writes four Parquet outputs."
    )

    pdf.h2("Cleaning")
    pdf.bullets(
        [
            "Lowercase, strip URLs and HTML tags, remove non-alphanumeric "
            "characters, collapse whitespace.",
            "Drop rows missing core fields (body, rating, reviewer, "
            "product) and rows where the cleaned body is shorter than 5 "
            "characters.",
            "Coerce review_date to a timestamp for windowing.",
        ]
    )

    pdf.h2("Per-row text features")
    pdf.bullets(
        [
            "body_len, body_word_count, exclam_count - cheap signals "
            "correlated with overstated emphasis common in fake reviews.",
            "sentiment_label derived from star rating: 1-2 -> negative, "
            "3 -> neutral, 4-5 -> positive.",
        ]
    )

    pdf.h2("Per-reviewer behavioural features (Spark Window aggregates)")
    pdf.bullets(
        [
            "reviewer_review_count, reviewer_avg_rating, reviewer_pct_5star",
            "reviewer_distinct_products, reviewer_reviews_same_day",
            "reviewer_verified_share",
        ]
    )

    pdf.h2("Per-product features")
    pdf.bullets(
        [
            "product_review_count, product_avg_rating, product_pct_5star",
            "dup_in_product (count of identical bodies for the same product) "
            "- a strong fraud signal for review-bombing.",
        ]
    )

    pdf.h2("Weak fraud label")
    pdf.p(
        "An ML model needs supervision; in production we don't have a clean "
        "fraud ground truth, so we generate a weak label from rules and let "
        "the supervised model generalise via text + behavior:"
    )
    pdf.code(
        "fraud_label = (\n"
        "    dup_in_product >= 3\n"
        "    OR (reviewer_review_count >= 8\n"
        "        AND reviewer_pct_5star >= 0.95\n"
        "        AND reviewer_verified_share <= 0.2)\n"
        "    OR reviewer_reviews_same_day >= 5\n"
        ")"
    )

    pdf.h2("Outputs")
    pdf.kv_table(
        [
            ("data/processed/train.parquet", "80% of labeled rows for training"),
            ("data/processed/test.parquet", "20% holdout for evaluation"),
            ("data/processed/product_agg.parquet", "Per-product rollups for the dashboard"),
            ("data/processed/reviewer_agg.parquet", "Per-reviewer rollups, sorted by fraud_rate"),
        ]
    )

    pdf.h2("Run output")
    pdf.code("[etl] rows=30,000 fraud_positives=1,689 (5.63%)")

    # ---------- 5. training -----------------------------------------------
    pdf.h1("5. Model training and MLflow")
    pdf.p(
        "src/train/train.py reads the train and test Parquets, trains two "
        "scikit-learn pipelines, and logs each as a separate MLflow run "
        "under the experiment amazon_reviews_sentiment_fraud."
    )

    pdf.h2("Sentiment model")
    pdf.bullets(
        [
            "TF-IDF: uni- and bi-grams, min_df=3, max_features=50,000, sublinear TF.",
            "Logistic Regression with class_weight='balanced', C=2.0, max_iter=400.",
            "Multi-class output: 0=negative, 1=neutral, 2=positive.",
        ]
    )

    pdf.h2("Fraud model")
    pdf.bullets(
        [
            "ColumnTransformer joining TF-IDF over the cleaned body with "
            "17 numeric / behavioural features (body stats + reviewer "
            "aggregates + product aggregates + dup_in_product + verified + "
            "star rating).",
            "Gradient Boosted Trees (n_estimators=120, max_depth=3, lr=0.1).",
            "Returns probability of fraud, with a default 0.5 decision "
            "threshold that callers can override per business need.",
        ]
    )

    pdf.h2("MLflow logged per run")
    pdf.bullets(
        [
            "Params: model name, ngram range, max_features, hyperparameters, "
            "n_train, n_test, class share.",
            "Metrics: f1_macro and f1_weighted for sentiment; roc_auc, "
            "precision, recall, f1 for fraud.",
            "Artifacts: full classification_report.txt and the joblib pipeline.",
        ]
    )

    pdf.h2("Note on the perfect metrics")
    pdf.p(
        "Both models score 1.00 on the synthetic test set. That is expected: "
        "the synthetic data is too easy (sentiment phrases don't overlap "
        "across classes; fraud uses exact-duplicate text). The metric is "
        "really 'pipeline correctness', not 'model quality'. On the real "
        "Amazon Customer Reviews dataset, expect macro-F1 around 0.85-0.92 "
        "for sentiment and ROC-AUC 0.7-0.85 for fraud. The code path does "
        "not change."
    )

    # ---------- 6. streaming ----------------------------------------------
    pdf.h1("6. Spark Structured Streaming scorer")
    pdf.p(
        "src/stream/score_stream.py watches data/streaming_in/ for new JSONL "
        "files on a 5-second trigger. Inside foreachBatch the micro-batch "
        "comes back to the driver as pandas, the persisted joblib pipelines "
        "score it, and results are written to data/streaming_out/scored/ as "
        "JSONL plus a console preview."
    )
    pdf.p(
        "foreachBatch was chosen over a pandas UDF on purpose: the pandas-UDF "
        "broadcast-pickle path is fragile across Python toolchains, while "
        "foreachBatch keeps the model objects on the driver and avoids that "
        "serialisation entirely. For very high throughput, the same logic "
        "lifts to a pandas UDF or a properly-broadcast joblib model."
    )
    pdf.p(
        "A drip-producer (scripts/feed_stream.py) splits raw reviews into "
        "small chunks and copies them into the input directory on a timer, "
        "so you can watch the dashboard refresh without running a real "
        "Kafka cluster."
    )

    # ---------- 7. FastAPI ------------------------------------------------
    pdf.h1("7. FastAPI service and web UI")
    pdf.p(
        "src/serve/app.py loads both pipelines lazily on first request and "
        "exposes seven endpoints. The dashboard is a single HTML page "
        "(templates/index.html) with vanilla JS (static/app.js) - no "
        "front-end build step."
    )

    pdf.kv_table(
        [
            ("GET /healthz", "Liveness probe"),
            ("GET /metadata", "Training metrics + numeric feature columns"),
            ("POST /predict", "Score one review"),
            ("POST /predict/batch", "Score up to 1000 reviews"),
            ("GET /aggregates/products", "Top-N from the ETL product rollup"),
            ("GET /aggregates/fraud-reviewers", "Top-N suspicious reviewers"),
            ("GET /stream/recent", "Latest streaming-scored rows"),
            ("GET /", "Dashboard HTML"),
        ]
    )

    pdf.h2("Example call")
    pdf.code(
        "curl -s -X POST http://127.0.0.1:8000/predict \\\n"
        "  -H 'content-type: application/json' \\\n"
        "  -d '{\"review_body\":\"absolutely love this, exceeded my expectations\",\n"
        "       \"star_rating\":5}'\n"
        "\n"
        "{\"sentiment\":\"positive\",\n"
        " \"fraud_proba\":3.4e-07,\n"
        " \"fraud_flag\":0}"
    )

    pdf.h2("Dashboard sections")
    pdf.bullets(
        [
            "Score-a-review form (free text -> sentiment + fraud probability).",
            "Top products table (review count, avg rating, %5*, fraud rate).",
            "Suspicious reviewers table (sorted by fraud_rate).",
            "Streaming feed table that auto-refreshes every 5 seconds.",
            "Model metadata pane reflecting the latest MLflow run summary.",
        ]
    )

    # ---------- 8. tests --------------------------------------------------
    pdf.h1("8. Tests")
    pdf.p(
        "tests/test_pipeline.py runs in under 5 seconds and exercises the "
        "code paths most likely to fail silently:"
    )
    pdf.bullets(
        [
            "Text cleaning (URL, HTML, punctuation stripping).",
            "Star-rating-to-sentiment label mapping (parametrised over all 6 cases).",
            "The serving-time feature enrichment (every numeric feature the "
            "fraud model expects must be present after enrichment).",
            "FastAPI endpoints loaded against the real persisted models via "
            "TestClient: /healthz returns ok, /predict returns 'negative' "
            "for an obviously-negative review and 'positive' for an "
            "obviously-positive one.",
        ]
    )
    pdf.code("$ make test\n10 passed in 4.12s")

    # ---------- 9. how to run ---------------------------------------------
    pdf.h1("9. How to run")
    pdf.code(
        "# install\n"
        "make install\n"
        "\n"
        "# full pipeline (ingest -> Spark ETL -> train -> MLflow log)\n"
        "make all\n"
        "\n"
        "# one-shot streaming pass on data/streaming_in/\n"
        "make stream-once\n"
        "\n"
        "# serve dashboard + REST API\n"
        "make serve         # http://127.0.0.1:8000/\n"
        "\n"
        "# MLflow tracking UI\n"
        "make mlflow-ui     # http://127.0.0.1:5000/\n"
        "\n"
        "# continuous streaming demo (two terminals)\n"
        "make stream        # term 1: structured streaming scorer\n"
        "make feed          # term 2: drip new reviews into streaming_in/\n"
        "\n"
        "# unit + integration tests\n"
        "make test\n"
        "\n"
        "# wipe artifacts\n"
        "make clean"
    )

    pdf.h2("Switching to the real dataset")
    pdf.p(
        "The README has a one-step pandas conversion that maps the public "
        "Amazon Customer Reviews TSV columns into the JSONL schema this "
        "pipeline expects. After that, 'make etl train' runs unchanged."
    )

    # ---------- 10. results -----------------------------------------------
    pdf.h1("10. Results on the smoke run")
    pdf.kv_table(
        [
            ("Rows ingested", "30,000"),
            ("Planted fraud share", "6.0%"),
            ("ETL-labeled fraud share", "5.63% (1,689 rows)"),
            ("Train / test split", "24,129 / 5,871"),
            ("Sentiment macro-F1 (synthetic)", "1.00 - pipeline-correctness signal, see Section 5"),
            ("Fraud ROC-AUC (synthetic)", "1.00 - pipeline-correctness signal, see Section 5"),
            ("Streaming batch", "100 rows scored in ~15s including Spark warm-up"),
            ("Test suite", "10/10 passing in ~4s"),
        ]
    )

    # ---------- 11. extensions --------------------------------------------
    pdf.h1("11. Where this would go next")
    pdf.bullets(
        [
            "Run on the real Amazon Customer Reviews dataset (full or per-category slice).",
            "Replace the file source with a Kafka source for the streaming job.",
            "Tune the fraud decision threshold on a labelled holdout instead of using 0.5.",
            "Promote models through the MLflow Model Registry "
            "(staging -> production) and have FastAPI load by stage.",
            "Containerise: Dockerfile + docker-compose for Spark + MLflow + FastAPI.",
            "Add a feature store so the streaming scorer can use real "
            "reviewer and product history instead of neutral defaults.",
        ]
    )

    pdf.output(str(OUT))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    sys.exit(build())
