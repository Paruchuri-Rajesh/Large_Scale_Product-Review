"""Helpers and CLI for exporting misclassified test rows to CSV (no API changes).

Writes optional reports under ``reports/ml/``; serving behavior unchanged.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.common.config import (  # noqa: E402
    FRAUD_ERROR_REPORT_PATH,
    FRAUD_MODEL_PATH,
    SENTIMENT_ERROR_REPORT_PATH,
    SENTIMENT_MODEL_PATH,
    TEST_PARQUET,
    ensure_dirs,
)
from src.train.features import (  # noqa: E402
    FRAUD_NUMERIC_FEATURES,
    FRAUD_TARGET,
    SENTIMENT_TARGET,
    TEXT_COLUMN,
    get_fraud_numeric_features,
)


# Same coercion as ``train._read_parquet`` so models receive expected dtypes.
def _load_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["verified_purchase_int"] = df["verified_purchase"].fillna(False).astype(int)
    df["review_body_clean"] = df["review_body_clean"].fillna("")
    for c in FRAUD_NUMERIC_FEATURES:
        if c not in df.columns:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


# Build a dataframe of sentiment mistakes.
def build_sentiment_error_table(
    df: pd.DataFrame,
    y_true,
    y_pred,
    max_rows: int = 25,
) -> pd.DataFrame:
    error_df = df.copy()
    error_df["true_label"] = y_true
    error_df["pred_label"] = y_pred
    error_df = error_df[error_df["true_label"] != error_df["pred_label"]]

    keep_cols: List[str] = [TEXT_COLUMN, "true_label", "pred_label"]
    keep_cols = [col for col in keep_cols if col in error_df.columns]
    return error_df[keep_cols].head(max_rows)


# Build a dataframe of fraud mistakes.
def build_fraud_error_table(
    df: pd.DataFrame,
    y_true,
    y_pred,
    y_proba=None,
    max_rows: int = 25,
) -> pd.DataFrame:
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    mask = yt != yp
    error_df = df.loc[mask].copy()
    error_df["true_label"] = yt[mask]
    error_df["pred_label"] = yp[mask]
    if y_proba is not None:
        error_df["fraud_proba"] = np.asarray(y_proba)[mask]

    keep_cols: List[str] = [
        TEXT_COLUMN,
        "star_rating",
        "helpful_votes",
        "total_votes",
        "true_label",
        "pred_label",
        "fraud_proba",
    ]
    keep_cols = [col for col in keep_cols if col in error_df.columns]
    return error_df[keep_cols].head(max_rows)


# Count total mistakes for a target column.
def count_prediction_errors(y_true, y_pred) -> int:
    return int((pd.Series(y_true) != pd.Series(y_pred)).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description="Export sentiment/fraud error samples from test Parquet.")
    parser.add_argument("--test", type=Path, default=TEST_PARQUET, help="Test Parquet (default: config).")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=10_000_000,
        help="Max rows per error CSV (default: effectively all errors).",
    )
    args = parser.parse_args()

    if not args.test.exists():
        raise SystemExit(f"test parquet not found: {args.test}")
    if not SENTIMENT_MODEL_PATH.exists():
        raise SystemExit(f"sentiment model not found: {SENTIMENT_MODEL_PATH}")
    if not FRAUD_MODEL_PATH.exists():
        raise SystemExit(f"fraud model not found: {FRAUD_MODEL_PATH}")

    ensure_dirs()
    df = _load_parquet(args.test)

    sentiment = joblib.load(SENTIMENT_MODEL_PATH)
    fraud = joblib.load(FRAUD_MODEL_PATH)

    y_sent_true = df[SENTIMENT_TARGET].to_numpy()
    y_sent_pred = sentiment.predict(df[TEXT_COLUMN])

    feat_cols = [TEXT_COLUMN] + get_fraud_numeric_features()
    y_fraud_true = df[FRAUD_TARGET].to_numpy()
    y_fraud_proba = fraud.predict_proba(df[feat_cols])[:, 1]
    y_fraud_pred = (y_fraud_proba >= 0.5).astype(int)

    n_sent_err = count_prediction_errors(y_sent_true, y_sent_pred)
    n_fraud_err = count_prediction_errors(y_fraud_true, y_fraud_pred)

    sent_tbl = build_sentiment_error_table(df, y_sent_true, y_sent_pred, max_rows=args.max_rows)
    fraud_tbl = build_fraud_error_table(df, y_fraud_true, y_fraud_pred, y_fraud_proba, max_rows=args.max_rows)

    ML_DIR = SENTIMENT_ERROR_REPORT_PATH.parent
    ML_DIR.mkdir(parents=True, exist_ok=True)

    sent_tbl.to_csv(SENTIMENT_ERROR_REPORT_PATH, index=False)
    fraud_tbl.to_csv(FRAUD_ERROR_REPORT_PATH, index=False)

    print(f"sentiment errors: {n_sent_err}")
    print(f"fraud errors: {n_fraud_err}")
    print(f"wrote {SENTIMENT_ERROR_REPORT_PATH}")
    print(f"wrote {FRAUD_ERROR_REPORT_PATH}")


if __name__ == "__main__":
    main()
