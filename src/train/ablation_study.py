"""Offline fraud feature ablation: leakage-prone full numerics vs reduced model set.

Does not change production training, serving, or saved joblibs. Writes CSV/JSON
under reports/ml/ for writeups.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.common.config import (  # noqa: E402
    FRAUD_ABLATION_COMPARISON_PATH,
    FRAUD_ABLATION_SUMMARY_PATH,
    FRAUD_ABLATION_THRESHOLDS_PATH,
    TEST_PARQUET,
    TRAIN_PARQUET,
    ensure_dirs,
)
from src.train.evaluate import apply_threshold, evaluate_fraud_thresholds  # noqa: E402
from src.train.features import (  # noqa: E402
    FRAUD_NUMERIC_FEATURES,
    get_fraud_numeric_features,
    get_fraud_target,
)

TEXT_COL = "review_body_clean"
TARGET = get_fraud_target()

# Same fraud pipeline family / hyperparameters as src/train/train.py (train_fraud).
_TFIDF_KW = dict(
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.95,
    max_features=20_000,
    sublinear_tf=True,
)
_GBT_KW = dict(
    n_estimators=120,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
)

# Full ETL numeric set (includes weak-label-adjacent columns). Reduced = production model input.
SETTING_LEAKAGE = "leakage_prone_full"
SETTING_REDUCED = "reduced_current"


def _read_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["verified_purchase_int"] = df["verified_purchase"].fillna(False).astype(int)
    df[TEXT_COL] = df[TEXT_COL].fillna("")
    for c in FRAUD_NUMERIC_FEATURES:
        if c not in df.columns:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def build_fraud_pipeline(numeric_features: List[str]) -> Pipeline:
    text_vec = TfidfVectorizer(**_TFIDF_KW)
    pre = ColumnTransformer(
        [
            ("text", text_vec, TEXT_COL),
            ("num", StandardScaler(with_mean=False), numeric_features),
        ]
    )
    return Pipeline(
        [
            ("pre", pre),
            ("clf", GradientBoostingClassifier(**_GBT_KW)),
        ]
    )


def _metrics_at_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float
) -> Tuple[float, float, float]:
    y_pred = apply_threshold(y_proba, threshold)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return float(p), float(r), float(f1)


def _threshold_grid() -> List[float]:
    # Wide enough to show tradeoffs; aligned with threshold_tuning-style grid + finer mid range.
    base = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    extra = [round(x, 2) for x in np.linspace(0.05, 0.95, 19)]
    merged = sorted(set(base + extra))
    return merged


def run_one_setting(
    name: str,
    numeric_features: List[str],
    train: pd.DataFrame,
    test: pd.DataFrame,
    thresholds: List[float],
) -> Tuple[Dict[str, Any], List[Dict[str, float]], np.ndarray]:
    feat_cols = [TEXT_COL] + numeric_features
    pipe = build_fraud_pipeline(numeric_features)
    pipe.fit(train[feat_cols], train[TARGET])
    y_proba = pipe.predict_proba(test[feat_cols])[:, 1]
    y_true = test[TARGET].to_numpy()

    try:
        auc = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        auc = float("nan")

    p05, r05, f05 = _metrics_at_threshold(y_true, y_proba, 0.5)
    sweep = evaluate_fraud_thresholds(y_true, y_proba, thresholds)
    best_row = max(sweep, key=lambda row: row["f1"])
    best_thr = float(best_row["threshold"])
    y_hat_best = apply_threshold(y_proba, best_thr)
    holdout_errors = int(np.sum(y_true != y_hat_best))

    summary_row = {
        "setting": name,
        "n_numeric_features": len(numeric_features),
        "numeric_features": ";".join(numeric_features),
        "precision_0.5": p05,
        "recall_0.5": r05,
        "f1_0.5": f05,
        "roc_auc": auc,
        "best_f1_threshold": best_thr,
        "best_f1": float(best_row["f1"]),
        "precision_at_best_f1": float(best_row["precision"]),
        "recall_at_best_f1": float(best_row["recall"]),
        "holdout_errors_at_best_f1": holdout_errors,
        "n_test": len(test),
    }

    thresh_rows = [{**r, "setting": name} for r in sweep]
    return summary_row, thresh_rows, y_proba


def main() -> None:
    parser = argparse.ArgumentParser(description="Fraud numeric feature ablation (leakage vs reduced).")
    parser.add_argument("--train", type=Path, default=TRAIN_PARQUET)
    parser.add_argument("--test", type=Path, default=TEST_PARQUET)
    args = parser.parse_args()

    if not args.train.exists():
        raise SystemExit(f"train parquet not found: {args.train}")
    if not args.test.exists():
        raise SystemExit(f"test parquet not found: {args.test}")

    ensure_dirs()
    train = _read_parquet(args.train)
    test = _read_parquet(args.test)

    reduced = get_fraud_numeric_features()
    # Full ETL numerics = reduced + columns excluded from production model (weak-label mirrors).
    leakage_full = list(FRAUD_NUMERIC_FEATURES)

    thresholds = _threshold_grid()

    print(f"[ablation] train={len(train):,} test={len(test):,}")
    print(f"[ablation] reduced_current: {len(reduced)} numeric columns")
    print(f"[ablation] leakage_prone_full: {len(leakage_full)} numeric columns")

    row_l, thr_l, _ = run_one_setting(SETTING_LEAKAGE, leakage_full, train, test, thresholds)
    row_r, thr_r, _ = run_one_setting(SETTING_REDUCED, reduced, train, test, thresholds)

    comp = pd.DataFrame([row_l, row_r])
    comp.to_csv(FRAUD_ABLATION_COMPARISON_PATH, index=False)

    thr_df = pd.DataFrame(thr_l + thr_r)
    thr_df.to_csv(FRAUD_ABLATION_THRESHOLDS_PATH, index=False)

    # Interpretation for JSON + console
    auc_l = row_l["roc_auc"]
    auc_r = row_r["roc_auc"]
    f1_l = row_l["best_f1"]
    f1_r = row_r["best_f1"]
    gap_auc = auc_l - auc_r if not (np.isnan(auc_l) or np.isnan(auc_r)) else float("nan")
    gap_f1 = f1_l - f1_r

    suspicious = False
    if not np.isnan(gap_auc) and gap_auc > 0.03 and gap_f1 > 0.03:
        suspicious = True
    if row_l["holdout_errors_at_best_f1"] < row_r["holdout_errors_at_best_f1"] * 0.5 and len(test) > 100:
        suspicious = True

    summary = {
        "settings": {
            SETTING_LEAKAGE: {
                "n_numeric_features": row_l["n_numeric_features"],
                "roc_auc": auc_l,
                "best_f1": f1_l,
                "best_f1_threshold": row_l["best_f1_threshold"],
                "holdout_errors_at_best_f1": row_l["holdout_errors_at_best_f1"],
            },
            SETTING_REDUCED: {
                "n_numeric_features": row_r["n_numeric_features"],
                "roc_auc": auc_r,
                "best_f1": f1_r,
                "best_f1_threshold": row_r["best_f1_threshold"],
                "holdout_errors_at_best_f1": row_r["holdout_errors_at_best_f1"],
            },
        },
        "delta_roc_auc_leakage_minus_reduced": gap_auc,
        "delta_best_f1_leakage_minus_reduced": gap_f1,
        "leakage_setting_looks_artificially_stronger": suspicious,
        "which_setting_has_higher_best_f1_on_holdout": (
            SETTING_LEAKAGE if f1_l >= f1_r else SETTING_REDUCED
        ),
        "trust_note": (
            "The reduced feature set omits columns that mirror Spark weak-label construction; "
            "metrics on the full set can look unrealistically strong and flatten threshold tradeoffs. "
            "Production training uses the reduced set for more trustworthy evaluation."
        ),
    }
    FRAUD_ABLATION_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    print()
    print("=== Fraud feature ablation (same GBT+TF-IDF recipe as train.py) ===")
    print(f"Wrote: {FRAUD_ABLATION_COMPARISON_PATH}")
    print(f"Wrote: {FRAUD_ABLATION_THRESHOLDS_PATH}")
    print(f"Wrote: {FRAUD_ABLATION_SUMMARY_PATH}")
    print()
    print(f"Higher best F1 on this holdout: {summary['which_setting_has_higher_best_f1_on_holdout']}")
    print(f"Δ ROC-AUC (leakage − reduced): {gap_auc:.4f}" if not np.isnan(gap_auc) else "Δ ROC-AUC: n/a")
    print(f"Δ best F1 (leakage − reduced): {gap_f1:+.4f}")
    if suspicious:
        print(
            "Leakage-prone full numerics look artificially stronger here "
            "(large AUC/F1 lift vs reduced—likely weak-label leakage through numeric mirrors)."
        )
    else:
        print(
            "Gap between settings is modest on this run; still prefer reduced features for "
            "deployment trust (no rule-mirroring columns)."
        )
    print()
    print(
        "Why reduced is more trustworthy: dup/same-day/reviewer-count style columns let the "
        "GBDT approximate batch heuristics; removing them forces the model to rely on text + "
        "non-leaky behavior, so precision/recall tradeoffs reflect scoring you can defend."
    )


if __name__ == "__main__":
    main()
