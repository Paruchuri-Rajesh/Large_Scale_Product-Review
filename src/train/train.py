"""Train sentiment + fraud models on the Spark ETL output, log to MLflow.

We materialize the Parquet via Spark -> Pandas (small, post-aggregation) and
train scikit-learn pipelines for both tasks. Pipelines are persisted so the
FastAPI service can score single reviews with no Spark dependency, and the
Spark streaming scorer can broadcast them to executors.

Two tasks:
  * sentiment   — multinomial logistic regression on TF-IDF text features.
  * fraud       — gradient-boosted trees on TF-IDF + numeric behavioral
                  features, calibrated for probability output.

Both runs log params, metrics, and artifacts to MLflow.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.common.config import (  # noqa: E402
    BASELINE_REPORT_PATH,
    FRAUD_MODEL_PATH,
    META_PATH,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    SENTIMENT_MODEL_PATH,
    TEST_PARQUET,
    THRESHOLD_REPORT_PATH,
    THRESHOLDS_PATH,
    TRAIN_PARQUET,
    ensure_dirs,
)
from src.train.features import (  # noqa: E402
    FRAUD_NUMERIC_FEATURES,
    get_fraud_numeric_features,
)
from src.train.registry import load_json_optional  # noqa: E402


# Stable names aligned with MLflow params / proposal wording (for meta.json only).
_SENTIMENT_MODEL_NAME = "logreg+tfidf"
_FRAUD_MODEL_NAME = "gbt+tfidf+behavior"


def _artifact_path_or_null(path: Path) -> str | None:
    """Resolved path string if the file exists, else None (optional reports)."""
    return str(path.resolve()) if path.exists() else None


def _read_parquet(path: Path) -> pd.DataFrame:
    # Spark writes a directory of part-*.parquet files; pandas/pyarrow handles that.
    df = pd.read_parquet(path)
    df["verified_purchase_int"] = df["verified_purchase"].fillna(False).astype(int)
    df["review_body_clean"] = df["review_body_clean"].fillna("")
    for c in FRAUD_NUMERIC_FEATURES:
        if c not in df.columns:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


def train_sentiment(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=3,
                    max_df=0.9,
                    max_features=50_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=400,
                    class_weight="balanced",
                    C=2.0,
                ),
            ),
        ]
    )

    with mlflow.start_run(run_name="sentiment-logreg-tfidf") as run:
        mlflow.log_params(
            {
                "model": "logreg+tfidf",
                "ngram_range": "1,2",
                "max_features": 50000,
                "C": 2.0,
                "class_weight": "balanced",
                "n_train": len(train),
                "n_test": len(test),
            }
        )
        pipe.fit(train["review_body_clean"], train["sentiment_label"])
        preds = pipe.predict(test["review_body_clean"])
        f1_macro = f1_score(test["sentiment_label"], preds, average="macro")
        f1_weighted = f1_score(test["sentiment_label"], preds, average="weighted")
        report = classification_report(
            test["sentiment_label"], preds, target_names=["neg", "neu", "pos"]
        )
        print("[sentiment]\n" + report)
        mlflow.log_metric("f1_macro", f1_macro)
        mlflow.log_metric("f1_weighted", f1_weighted)
        mlflow.log_text(report, "classification_report.txt")
        mlflow.sklearn.log_model(pipe, artifact_path="sentiment_model")

        joblib.dump(pipe, SENTIMENT_MODEL_PATH)
        mlflow.log_artifact(str(SENTIMENT_MODEL_PATH))
        return {
            "run_id": run.info.run_id,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
        }


def train_fraud(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    text_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
        max_features=20_000,
        sublinear_tf=True,
    )
    pre = ColumnTransformer(
        [
            ("text", text_vec, "review_body_clean"),
            ("num", StandardScaler(with_mean=False), get_fraud_numeric_features()),
        ]
    )
    pipe = Pipeline(
        [
            ("pre", pre),
            (
                "clf",
                GradientBoostingClassifier(
                    n_estimators=120,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=42,
                ),
            ),
        ]
    )

    with mlflow.start_run(run_name="fraud-gbt-tfidf+behavior") as run:
        mlflow.log_params(
            {
                "model": "gbt+tfidf+behavior",
                "ngram_range": "1,2",
                "max_features_text": 20000,
                "n_estimators": 120,
                "max_depth": 3,
                "n_train": len(train),
                "n_test": len(test),
                "fraud_share_train": float(train["fraud_label"].mean()),
            }
        )
        feat_cols = ["review_body_clean"] + get_fraud_numeric_features()
        pipe.fit(train[feat_cols], train["fraud_label"])
        proba = pipe.predict_proba(test[feat_cols])[:, 1]
        preds = (proba >= 0.5).astype(int)
        try:
            auc = roc_auc_score(test["fraud_label"], proba)
        except ValueError:
            auc = float("nan")
        prec, rec, f1, _ = precision_recall_fscore_support(
            test["fraud_label"], preds, average="binary", zero_division=0
        )
        report = classification_report(
            test["fraud_label"], preds, target_names=["clean", "fraud"], zero_division=0
        )
        print("[fraud]\n" + report)
        print(f"[fraud] roc_auc={auc:.4f}")
        mlflow.log_metric("roc_auc", float(auc) if not np.isnan(auc) else 0.0)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)
        mlflow.log_text(report, "classification_report.txt")
        mlflow.sklearn.log_model(pipe, artifact_path="fraud_model")

        joblib.dump(pipe, FRAUD_MODEL_PATH)
        mlflow.log_artifact(str(FRAUD_MODEL_PATH))
        return {
            "run_id": run.info.run_id,
            "roc_auc": float(auc) if not np.isnan(auc) else None,
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=TRAIN_PARQUET)
    parser.add_argument("--test", type=Path, default=TEST_PARQUET)
    args = parser.parse_args()

    ensure_dirs()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    train = _read_parquet(args.train)
    test = _read_parquet(args.test)
    print(f"[train] loaded train={len(train):,} test={len(test):,}")

    sentiment_metrics = train_sentiment(train, test)
    fraud_metrics = train_fraud(train, test)

    meta = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "sentiment": sentiment_metrics,
        "fraud": fraud_metrics,
        "numeric_fraud_features": get_fraud_numeric_features(),
        "numeric_fraud_features_excluded_from_model": sorted(
            set(FRAUD_NUMERIC_FEATURES) - set(get_fraud_numeric_features())
        ),
        "artifacts": {
            "baseline_report_path": _artifact_path_or_null(BASELINE_REPORT_PATH),
            "threshold_report_path": _artifact_path_or_null(THRESHOLD_REPORT_PATH),
            "thresholds_path": _artifact_path_or_null(THRESHOLDS_PATH),
        },
        "selection": {
            "sentiment_model_name": _SENTIMENT_MODEL_NAME,
            "fraud_model_name": _FRAUD_MODEL_NAME,
        },
        "thresholds": load_json_optional(THRESHOLDS_PATH),
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"[train] wrote meta -> {META_PATH}")
    print(
        "[train] meta.json includes artifacts + selection + thresholds embed "
        f"(baseline_csv={'yes' if BASELINE_REPORT_PATH.exists() else 'no'}, "
        f"threshold_study_csv={'yes' if THRESHOLD_REPORT_PATH.exists() else 'no'}, "
        f"thresholds_json={'yes' if THRESHOLDS_PATH.exists() else 'no'})"
    )


if __name__ == "__main__":
    main()
