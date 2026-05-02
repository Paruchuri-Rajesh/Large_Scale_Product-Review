"""Helpers for choosing a fraud decision threshold + CLI to export reports.

The app still uses 0.5 until serving reads ``models/thresholds.json``. This module
writes ``reports/ml/threshold_study.csv`` and ``models/thresholds.json`` only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.common.config import (  # noqa: E402
    FRAUD_MODEL_PATH,
    TEST_PARQUET,
    THRESHOLD_REPORT_PATH,
    THRESHOLDS_PATH,
    ensure_dirs,
)
from src.train.evaluate import evaluate_fraud_thresholds  # noqa: E402
from src.train.features import (  # noqa: E402
    FRAUD_NUMERIC_FEATURES,
    get_fraud_numeric_features,
    get_fraud_target,
    get_text_column,
)


# Same coercion path as ``train._read_parquet`` so fraud pipeline sees expected dtypes.
def _load_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["verified_purchase_int"] = df["verified_purchase"].fillna(False).astype(int)
    df["review_body_clean"] = df["review_body_clean"].fillna("")
    for c in FRAUD_NUMERIC_FEATURES:
        if c not in df.columns:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


# Return a default set of thresholds to compare.
def get_default_thresholds() -> List[float]:
    return [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


# Run a threshold study for fraud probabilities.
def run_threshold_study(y_true, y_proba, thresholds: Optional[List[float]] = None) -> List[Dict[str, float]]:
    active_thresholds = thresholds or get_default_thresholds()
    return evaluate_fraud_thresholds(y_true, y_proba, active_thresholds)


# Pick the threshold row with the best F1 score.
def choose_best_f1_threshold(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        raise ValueError("Threshold study rows cannot be empty.")
    return max(rows, key=lambda row: row["f1"])


# First row meeting min_precision, or None if no threshold qualifies.
def choose_threshold_for_precision(
    rows: List[Dict[str, float]],
    min_precision: float,
) -> Optional[Dict[str, float]]:
    for row in rows:
        if row["precision"] >= min_precision:
            return row
    return None


# Load fraud model + test frame, return y_true, y_proba, feat_cols (for reuse/testing).
def load_fraud_scores(test_parquet: Path):
    df = _load_parquet(test_parquet)
    model = joblib.load(FRAUD_MODEL_PATH)
    text_col = get_text_column()
    feat_cols = [text_col] + get_fraud_numeric_features()
    y_true = df[get_fraud_target()].to_numpy()
    y_proba = model.predict_proba(df[feat_cols])[:, 1]
    return y_true, y_proba, feat_cols


def main() -> None:
    parser = argparse.ArgumentParser(description="Fraud threshold study on holdout Parquet (no API change).")
    parser.add_argument(
        "--test",
        type=Path,
        default=TEST_PARQUET,
        help="Test Parquet path (default: config TEST_PARQUET).",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default=None,
        help="Comma-separated thresholds, e.g. 0.3,0.5,0.7 (default: built-in grid).",
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=0.95,
        help="Target precision for precision_target_* fields (default 0.95).",
    )
    args = parser.parse_args()

    if not FRAUD_MODEL_PATH.exists():
        raise SystemExit(f"fraud model not found: {FRAUD_MODEL_PATH}")
    if not args.test.exists():
        raise SystemExit(f"test parquet not found: {args.test}")

    thresholds: Optional[List[float]] = None
    if args.thresholds:
        thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]

    ensure_dirs()
    y_true, y_proba, _feat_cols = load_fraud_scores(args.test)
    rows = run_threshold_study(y_true, y_proba, thresholds)
    df_out = pd.DataFrame(rows)

    best = choose_best_f1_threshold(rows)
    prec_row = choose_threshold_for_precision(rows, args.min_precision)

    THRESHOLD_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(THRESHOLD_REPORT_PATH, index=False)

    summary = {
        "best_f1_threshold": best["threshold"],
        "best_f1_metrics": {
            "precision": best["precision"],
            "recall": best["recall"],
            "f1": best["f1"],
        },
        "precision_target_threshold": prec_row["threshold"] if prec_row else None,
        "precision_target_metrics": (
            {
                "precision": prec_row["precision"],
                "recall": prec_row["recall"],
                "f1": prec_row["f1"],
            }
            if prec_row
            else None
        ),
    }
    THRESHOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    THRESHOLDS_PATH.write_text(json.dumps(summary, indent=2))

    print(df_out.to_string(index=False))
    print()
    print(f"best F1 threshold: {best['threshold']:.6f}  (precision={best['precision']:.4f} recall={best['recall']:.4f} f1={best['f1']:.4f})")
    if prec_row:
        print(
            f"precision>={args.min_precision} threshold: {prec_row['threshold']:.6f}  "
            f"(precision={prec_row['precision']:.4f} recall={prec_row['recall']:.4f} f1={prec_row['f1']:.4f})"
        )
    else:
        print(f"precision>={args.min_precision}: no threshold met (stored null in JSON)")
    print()
    print(f"wrote {THRESHOLD_REPORT_PATH}")
    print(f"wrote {THRESHOLDS_PATH}")


if __name__ == "__main__":
    main()
