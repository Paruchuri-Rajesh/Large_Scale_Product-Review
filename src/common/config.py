"""Centralized paths and constants."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
STREAM_IN_DIR = DATA_DIR / "streaming_in"
STREAM_OUT_DIR = DATA_DIR / "streaming_out"
MODELS_DIR = PROJECT_ROOT / "models"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
REPORTS_DIR = PROJECT_ROOT / "reports"
ML_REPORTS_DIR = REPORTS_DIR / "ml"

RAW_REVIEWS = RAW_DIR / "reviews.jsonl"
TRAIN_PARQUET = PROCESSED_DIR / "train.parquet"
TEST_PARQUET = PROCESSED_DIR / "test.parquet"
PRODUCT_AGG_PARQUET = PROCESSED_DIR / "product_agg.parquet"
REVIEWER_AGG_PARQUET = PROCESSED_DIR / "reviewer_agg.parquet"

SENTIMENT_MODEL_PATH = MODELS_DIR / "sentiment_pipeline.joblib"
FRAUD_MODEL_PATH = MODELS_DIR / "fraud_pipeline.joblib"
META_PATH = MODELS_DIR / "meta.json"
THRESHOLDS_PATH = MODELS_DIR / "thresholds.json"
BASELINE_REPORT_PATH = ML_REPORTS_DIR / "baseline_comparison.csv"
THRESHOLD_REPORT_PATH = ML_REPORTS_DIR / "threshold_study.csv"
SENTIMENT_ERROR_REPORT_PATH = ML_REPORTS_DIR / "sentiment_error_samples.csv"
FRAUD_ERROR_REPORT_PATH = ML_REPORTS_DIR / "fraud_error_samples.csv"
DRIFT_REPORT_PATH = ML_REPORTS_DIR / "drift_report.json"
FRAUD_ABLATION_COMPARISON_PATH = ML_REPORTS_DIR / "fraud_ablation_comparison.csv"
FRAUD_ABLATION_THRESHOLDS_PATH = ML_REPORTS_DIR / "fraud_ablation_thresholds.csv"
FRAUD_ABLATION_SUMMARY_PATH = ML_REPORTS_DIR / "fraud_ablation_summary.json"

MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", f"file://{MLRUNS_DIR}"
)
MLFLOW_EXPERIMENT = os.environ.get(
    "MLFLOW_EXPERIMENT", "amazon_reviews_sentiment_fraud"
)

# Sentiment label mapping derived from star rating.
# 1-2 -> negative, 3 -> neutral, 4-5 -> positive
SENTIMENT_LABELS = {0: "negative", 1: "neutral", 2: "positive"}


def ensure_dirs() -> None:
    for d in (
        RAW_DIR,
        PROCESSED_DIR,
        STREAM_IN_DIR,
        STREAM_OUT_DIR,
        MODELS_DIR,
        MLRUNS_DIR,
        REPORTS_DIR,
        ML_REPORTS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
