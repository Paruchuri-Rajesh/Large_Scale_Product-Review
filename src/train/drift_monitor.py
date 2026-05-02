"""Lightweight drift checks for training vs recent data + CLI report writer.

Compares reference (train) vs current (test) Parquet; writes ``reports/ml/drift_report.json``.
Serving code unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.common.config import DRIFT_REPORT_PATH, TEST_PARQUET, TRAIN_PARQUET, ensure_dirs  # noqa: E402
from src.train.features import FRAUD_NUMERIC_FEATURES, TEXT_COLUMN  # noqa: E402


# Same coercion as ``train._read_parquet`` (skip unknown cols quietly).
def _load_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["verified_purchase_int"] = df["verified_purchase"].fillna(False).astype(int)
    df["review_body_clean"] = df["review_body_clean"].fillna("")
    for c in FRAUD_NUMERIC_FEATURES:
        if c not in df.columns:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


# Compare column means between reference and current data.
def compare_numeric_feature_means(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_columns: List[str] | None = None,
) -> List[Dict[str, float]]:
    columns = feature_columns or FRAUD_NUMERIC_FEATURES
    rows: List[Dict[str, float]] = []

    for col in columns:
        if col not in reference_df.columns or col not in current_df.columns:
            continue

        ref_mean = float(pd.to_numeric(reference_df[col], errors="coerce").fillna(0).mean())
        cur_mean = float(pd.to_numeric(current_df[col], errors="coerce").fillna(0).mean())
        rows.append(
            {
                "feature": col,
                "reference_mean": ref_mean,
                "current_mean": cur_mean,
                "mean_delta": cur_mean - ref_mean,
            }
        )

    return rows


# Compare review length behavior between reference and current data.
def compare_text_length_summary(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    text_column: str = TEXT_COLUMN,
) -> Dict[str, float]:
    if text_column not in reference_df.columns or text_column not in current_df.columns:
        return {}

    ref_lengths = reference_df[text_column].fillna("").astype(str).str.len()
    cur_lengths = current_df[text_column].fillna("").astype(str).str.len()

    return {
        "reference_avg_length": float(ref_lengths.mean()),
        "current_avg_length": float(cur_lengths.mean()),
        "avg_length_delta": float(cur_lengths.mean() - ref_lengths.mean()),
    }


# Build one simple drift summary object for later reporting.
def build_drift_summary(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> Dict[str, object]:
    return {
        "row_count_reference": int(len(reference_df)),
        "row_count_current": int(len(current_df)),
        "numeric_mean_comparison": compare_numeric_feature_means(reference_df, current_df),
        "text_length_summary": compare_text_length_summary(reference_df, current_df),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Drift summary: train Parquet vs test Parquet.")
    parser.add_argument("--reference", type=Path, default=TRAIN_PARQUET, help="Reference Parquet (default: train).")
    parser.add_argument("--current", type=Path, default=TEST_PARQUET, help="Current Parquet (default: test).")
    args = parser.parse_args()

    if not args.reference.exists():
        raise SystemExit(f"reference parquet not found: {args.reference}")
    if not args.current.exists():
        raise SystemExit(f"current parquet not found: {args.current}")

    ensure_dirs()
    ref_df = _load_parquet(args.reference)
    cur_df = _load_parquet(args.current)

    summary = build_drift_summary(ref_df, cur_df)
    n_numeric = len(summary["numeric_mean_comparison"])

    DRIFT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DRIFT_REPORT_PATH.write_text(json.dumps(summary, indent=2))

    print(f"reference rows: {summary['row_count_reference']}")
    print(f"current rows: {summary['row_count_current']}")
    print(f"numeric features compared: {n_numeric}")
    print(f"wrote {DRIFT_REPORT_PATH}")


if __name__ == "__main__":
    main()
