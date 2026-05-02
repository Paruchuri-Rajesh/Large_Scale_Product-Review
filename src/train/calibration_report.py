"""Fraud probability calibration report (raw vs sigmoid vs isotonic).

This is an offline evaluation artifact; it does not change serving behavior.
It keeps the current precision/recall/F1/ROC metrics and adds calibration
quality scores (Brier) + reliability-bin exports.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.common.config import (  # noqa: E402
    FRAUD_CALIBRATION_BINS_PATH,
    FRAUD_CALIBRATION_REPORT_PATH,
    FRAUD_CALIBRATION_SUMMARY_PATH,
    FRAUD_MODEL_PATH,
    TEST_PARQUET,
    ensure_dirs,
)
from src.train.evaluate import evaluate_fraud  # noqa: E402
from src.train.features import (  # noqa: E402
    FRAUD_NUMERIC_FEATURES,
    get_fraud_numeric_features,
    get_fraud_target,
    get_text_column,
)


def _load_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["verified_purchase_int"] = df["verified_purchase"].fillna(False).astype(int)
    df["review_body_clean"] = df["review_body_clean"].fillna("")
    for c in FRAUD_NUMERIC_FEATURES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def _reliability_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        cnt = int(mask.sum())
        if cnt == 0:
            rows.append(
                {
                    "bin_idx": i,
                    "bin_lo": float(lo),
                    "bin_hi": float(hi),
                    "count": 0,
                    "mean_pred": np.nan,
                    "empirical_pos_rate": np.nan,
                    "abs_gap": np.nan,
                }
            )
            continue
        p_hat = float(np.mean(y_prob[mask]))
        p_emp = float(np.mean(y_true[mask]))
        rows.append(
            {
                "bin_idx": i,
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "count": cnt,
                "mean_pred": p_hat,
                "empirical_pos_rate": p_emp,
                "abs_gap": abs(p_hat - p_emp),
            }
        )
    return pd.DataFrame(rows)


def _ece_from_bins(bins_df: pd.DataFrame) -> float:
    valid = bins_df[bins_df["count"] > 0]
    total = float(valid["count"].sum())
    if total <= 0:
        return float("nan")
    return float(np.sum((valid["count"] / total) * valid["abs_gap"]))


def _evaluate_variant(name: str, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = evaluate_fraud(y_true, y_pred, y_prob)
    bins_df = _reliability_bins(y_true, y_prob, n_bins=10)
    ece = _ece_from_bins(bins_df)
    brier = float(brier_score_loss(y_true, y_prob))
    return {
        "method": name,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "brier": brier,
        "ece": ece,
        "bins_df": bins_df,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fraud calibration report on holdout Parquet.")
    parser.add_argument("--test", type=Path, default=TEST_PARQUET)
    parser.add_argument("--calib-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not FRAUD_MODEL_PATH.exists():
        raise SystemExit(f"fraud model not found: {FRAUD_MODEL_PATH}")
    if not args.test.exists():
        raise SystemExit(f"test parquet not found: {args.test}")
    if args.calib_fraction <= 0 or args.calib_fraction >= 1:
        raise SystemExit("calib-fraction must be between 0 and 1 (exclusive)")

    ensure_dirs()
    df = _load_parquet(args.test)
    text_col = get_text_column()
    target = get_fraud_target()
    feat_cols = [text_col] + get_fraud_numeric_features()
    model = joblib.load(FRAUD_MODEL_PATH)

    y_true = df[target].to_numpy().astype(int)
    p_raw = model.predict_proba(df[feat_cols])[:, 1]

    # Split test into calibration/eval subsets to avoid calibrating + measuring on exact same rows.
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_calib = int(len(idx) * args.calib_fraction)
    calib_idx = idx[:n_calib]
    eval_idx = idx[n_calib:]
    if len(eval_idx) == 0:
        raise SystemExit("Not enough rows after split; adjust calib-fraction.")

    y_cal = y_true[calib_idx]
    p_cal = p_raw[calib_idx]
    y_eval = y_true[eval_idx]
    p_eval_raw = p_raw[eval_idx]

    # Platt / sigmoid calibration
    platt = LogisticRegression(solver="lbfgs")
    platt.fit(p_cal.reshape(-1, 1), y_cal)
    p_eval_sigmoid = platt.predict_proba(p_eval_raw.reshape(-1, 1))[:, 1]

    # Isotonic calibration
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_cal, y_cal)
    p_eval_iso = iso.predict(p_eval_raw)

    variants = [
        _evaluate_variant("raw", y_eval, p_eval_raw),
        _evaluate_variant("sigmoid_platt", y_eval, p_eval_sigmoid),
        _evaluate_variant("isotonic", y_eval, p_eval_iso),
    ]

    report_rows = [
        {
            "method": v["method"],
            "precision": v["precision"],
            "recall": v["recall"],
            "f1": v["f1"],
            "roc_auc": v["roc_auc"],
            "brier": v["brier"],
            "ece": v["ece"],
            "n_eval": int(len(y_eval)),
            "n_calib": int(len(y_cal)),
        }
        for v in variants
    ]
    report_df = pd.DataFrame(report_rows)
    FRAUD_CALIBRATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(FRAUD_CALIBRATION_REPORT_PATH, index=False)

    bins_frames: List[pd.DataFrame] = []
    for v in variants:
        b = v["bins_df"].copy()
        b.insert(0, "method", v["method"])
        bins_frames.append(b)
    bins_df = pd.concat(bins_frames, ignore_index=True)
    bins_df.to_csv(FRAUD_CALIBRATION_BINS_PATH, index=False)

    # Select "best calibrated" by brier; tie-break by ece
    sorted_rows = sorted(
        report_rows,
        key=lambda r: (r["brier"] if r["brier"] is not None else 1e9, r["ece"] if r["ece"] is not None else 1e9),
    )
    best = sorted_rows[0]
    summary = {
        "calibration_split": {"n_calib": int(len(y_cal)), "n_eval": int(len(y_eval)), "seed": args.seed},
        "best_calibrated_method_by_brier": best["method"],
        "report_rows": report_rows,
        "note": "This report is offline analysis only. Serving behavior is unchanged unless explicitly updated.",
    }
    FRAUD_CALIBRATION_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    print(report_df.to_string(index=False))
    print()
    print(f"best calibrated method (by brier): {best['method']}")
    print(f"wrote {FRAUD_CALIBRATION_REPORT_PATH}")
    print(f"wrote {FRAUD_CALIBRATION_BINS_PATH}")
    print(f"wrote {FRAUD_CALIBRATION_SUMMARY_PATH}")


if __name__ == "__main__":
    main()

