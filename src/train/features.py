"""Shared ML feature names and small validation helpers.

This module centralizes the existing training/serving column names so new ML
files can reuse them without changing current behavior.
"""
from __future__ import annotations

from typing import Iterable, List

TEXT_COLUMN = "review_body_clean"
SENTIMENT_TARGET = "sentiment_label"
FRAUD_TARGET = "fraud_label"

# All numeric columns produced by Spark ETL (Parquet / drift / coercion).
FRAUD_NUMERIC_FEATURES = [
    "star_rating",
    "helpful_votes",
    "total_votes",
    "verified_purchase_int",
    "body_len",
    "body_word_count",
    "exclam_count",
    "reviewer_review_count",
    "reviewer_avg_rating",
    "reviewer_pct_5star",
    "reviewer_distinct_products",
    "reviewer_reviews_same_day",
    "reviewer_verified_share",
    "product_review_count",
    "product_avg_rating",
    "product_pct_5star",
    "dup_in_product",
]

# Subset passed into the fraud classifier only. Excludes columns that are direct
# algebraic mirrors of the Spark weak-label rules (dup count, same-day count,
# reviewer monoculture thresholds): training on them lets GBDT reconstruct
# fraud_label with near-perfect accuracy and flat threshold curves.
FRAUD_MODEL_NUMERIC_FEATURES = [
    "star_rating",
    "helpful_votes",
    "total_votes",
    "verified_purchase_int",
    "body_len",
    "body_word_count",
    "exclam_count",
    "reviewer_avg_rating",
    "reviewer_distinct_products",
    "product_review_count",
    "product_avg_rating",
    "product_pct_5star",
]


# Return the shared text feature column name.
def get_text_column() -> str:
    return TEXT_COLUMN


# Return the sentiment target column name.
def get_sentiment_target() -> str:
    return SENTIMENT_TARGET


# Return the fraud target column name.
def get_fraud_target() -> str:
    return FRAUD_TARGET


# Return the fraud numeric feature columns used by the model.
def get_fraud_numeric_features() -> List[str]:
    """Columns the persisted fraud pipeline expects (text + these numerics)."""
    return list(FRAUD_MODEL_NUMERIC_FEATURES)


def get_all_etl_numeric_columns() -> List[str]:
    """Full numeric feature list from batch ETL Parquet (includes leaky signals)."""
    return list(FRAUD_NUMERIC_FEATURES)


# Check whether a dataframe-like object has all required columns.
def validate_required_columns(df, required_columns: Iterable[str]) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
