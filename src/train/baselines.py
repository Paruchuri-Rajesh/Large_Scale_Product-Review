"""Baseline model experiments for sentiment and fraud tasks.

This file adds simple comparison models so we can show why the current final
models were chosen. It does not change the main training pipeline yet.
"""
from __future__ import annotations

import argparse
from typing import Dict, List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.common.config import BASELINE_REPORT_PATH, TEST_PARQUET, TRAIN_PARQUET, ensure_dirs
from src.train.evaluate import apply_threshold, evaluate_fraud, evaluate_sentiment
from src.train.features import (
    FRAUD_NUMERIC_FEATURES,
    FRAUD_TARGET,
    SENTIMENT_TARGET,
    TEXT_COLUMN,
    get_fraud_numeric_features,
)


# Read parquet and keep feature columns compatible with current training flow.
def _read_parquet(path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["verified_purchase_int"] = df["verified_purchase"].fillna(False).astype(int)
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("")
    for col in FRAUD_NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


# Train and evaluate a few simple sentiment baselines.
def run_sentiment_baselines(train_df: pd.DataFrame, test_df: pd.DataFrame) -> List[Dict[str, object]]:
    models = {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "tfidf_naive_bayes": Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9)),
                ("clf", MultinomialNB()),
            ]
        ),
        "tfidf_logistic_regression": Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9)),
                ("clf", LogisticRegression(max_iter=400, class_weight="balanced")),
            ]
        ),
        "tfidf_linear_svm": Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9)),
                ("clf", LinearSVC()),
            ]
        ),
    }

    results: List[Dict[str, object]] = []
    X_train = train_df[TEXT_COLUMN]
    y_train = train_df[SENTIMENT_TARGET]
    X_test = test_df[TEXT_COLUMN]
    y_test = test_df[SENTIMENT_TARGET]

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = evaluate_sentiment(y_test, preds)
        results.append(
            {
                "task": "sentiment",
                "model_name": model_name,
                "f1_macro": metrics["f1_macro"],
                "f1_weighted": metrics["f1_weighted"],
            }
        )

    return results


# Train and evaluate a few simple fraud baselines.
def run_fraud_baselines(train_df: pd.DataFrame, test_df: pd.DataFrame) -> List[Dict[str, object]]:
    model_nums = get_fraud_numeric_features()
    preprocessor = ColumnTransformer(
        [
            ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95), TEXT_COLUMN),
            ("num", StandardScaler(with_mean=False), model_nums),
        ]
    )

    models = {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "numeric_logistic_regression": Pipeline(
            [
                (
                    "num_only",
                    ColumnTransformer(
                        [("num", StandardScaler(with_mean=False), model_nums)]
                    ),
                ),
                ("clf", LogisticRegression(max_iter=400, class_weight="balanced")),
            ]
        ),
        "hybrid_logistic_regression": Pipeline(
            [
                ("pre", preprocessor),
                ("clf", LogisticRegression(max_iter=400, class_weight="balanced")),
            ]
        ),
        "hybrid_random_forest": Pipeline(
            [
                ("pre", preprocessor),
                ("clf", RandomForestClassifier(n_estimators=120, random_state=42)),
            ]
        ),
    }

    results: List[Dict[str, object]] = []
    feat_cols = [TEXT_COLUMN] + model_nums
    X_train = train_df[feat_cols]
    y_train = train_df[FRAUD_TARGET]
    X_test = test_df[feat_cols]
    y_test = test_df[FRAUD_TARGET]

    for model_name, model in models.items():
        model.fit(X_train, y_train)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:, 1]
            preds = apply_threshold(proba, 0.5)
        else:
            preds = model.predict(X_test)
            proba = None

        metrics = evaluate_fraud(y_test, preds, proba)
        results.append(
            {
                "task": "fraud",
                "model_name": model_name,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
            }
        )

    return results


# Run all baseline experiments and save one comparison CSV.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=str(TRAIN_PARQUET))
    parser.add_argument("--test", default=str(TEST_PARQUET))
    args = parser.parse_args()

    ensure_dirs()
    train_df = _read_parquet(args.train)
    test_df = _read_parquet(args.test)

    rows = []
    rows.extend(run_sentiment_baselines(train_df, test_df))
    rows.extend(run_fraud_baselines(train_df, test_df))

    out_df = pd.DataFrame(rows)
    out_df.to_csv(BASELINE_REPORT_PATH, index=False)

    print(out_df.to_string(index=False))
    print(f"[baselines] wrote report -> {BASELINE_REPORT_PATH}")


if __name__ == "__main__":
    main()
