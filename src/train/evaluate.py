"""Reusable evaluation helpers for sentiment and fraud models.

These helpers keep metric logic in one place so baseline experiments, final
training, and later analysis files can reuse the same evaluation code.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


# Build summary metrics for multiclass sentiment predictions.
def evaluate_sentiment(y_true, y_pred) -> Dict[str, object]:
    labels = ["neg", "neu", "pos"]
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=labels,
            zero_division=0,
        ),
    }


# Build summary metrics for binary fraud predictions.
def evaluate_fraud(y_true, y_pred, y_proba=None) -> Dict[str, object]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=["clean", "fraud"],
            zero_division=0,
        ),
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return metrics


# Convert fraud probabilities into binary predictions using a threshold.
def apply_threshold(y_proba, threshold: float = 0.5) -> np.ndarray:
    return (np.asarray(y_proba) >= threshold).astype(int)


# Build a simple threshold study table for binary fraud scoring.
def evaluate_fraud_thresholds(
    y_true,
    y_proba,
    thresholds: List[float],
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []

    for threshold in thresholds:
        y_pred = apply_threshold(y_proba, threshold)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )

    return rows
