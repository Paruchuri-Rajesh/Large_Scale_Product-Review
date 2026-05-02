"""Spark batch ETL: clean, feature engineer, aggregate Amazon reviews.

Inputs : data/raw/reviews.jsonl (or --input <path>)
Outputs:
  data/processed/train.parquet         per-review features + labels (train split)
  data/processed/test.parquet          per-review features + labels (test split)
  data/processed/product_agg.parquet   per-product rollups
  data/processed/reviewer_agg.parquet  per-reviewer rollups (used as fraud signal)

This is the "batch processing for historical training" half of the system.
The same feature columns are reused at serving time so train/serve skew stays
minimal.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from pyspark.sql import DataFrame, functions as F, Window  # noqa: E402

from src.common.config import (  # noqa: E402
    PRODUCT_AGG_PARQUET,
    RAW_REVIEWS,
    REVIEWER_AGG_PARQUET,
    TEST_PARQUET,
    TRAIN_PARQUET,
    ensure_dirs,
)
from src.common.schema import REVIEW_SCHEMA  # noqa: E402
from src.common.spark import get_spark  # noqa: E402


# ---- text cleaning ----------------------------------------------------------

def _clean_text(col: F.Column) -> F.Column:
    c = F.lower(F.coalesce(col, F.lit("")))
    c = F.regexp_replace(c, r"https?://\S+|www\.\S+", " ")
    c = F.regexp_replace(c, r"<[^>]+>", " ")
    c = F.regexp_replace(c, r"[^a-z0-9\s']", " ")
    c = F.regexp_replace(c, r"\s+", " ")
    return F.trim(c)


def _sentiment_label(col: F.Column) -> F.Column:
    return (
        F.when(col <= 2, F.lit(0))
        .when(col == 3, F.lit(1))
        .otherwise(F.lit(2))
    )


# ---- pipeline steps ---------------------------------------------------------

def load_raw(spark, path: Path) -> DataFrame:
    return spark.read.schema(REVIEW_SCHEMA).json(str(path))


def clean(df: DataFrame) -> DataFrame:
    return (
        df.dropna(subset=["review_body", "star_rating", "reviewer_id", "product_id"])
        .filter(F.col("star_rating").between(1, 5))
        .withColumn("review_body_clean", _clean_text(F.col("review_body")))
        .withColumn("review_headline_clean", _clean_text(F.col("review_headline")))
        .filter(F.length("review_body_clean") >= 5)
        .withColumn("body_len", F.length("review_body_clean"))
        .withColumn("body_word_count", F.size(F.split("review_body_clean", " ")))
        .withColumn(
            "exclam_count",
            F.size(F.split(F.coalesce("review_body", F.lit("")), r"!")) - 1,
        )
        .withColumn(
            "review_ts",
            F.to_timestamp("review_date", "yyyy-MM-dd"),
        )
        .withColumn("sentiment_label", _sentiment_label(F.col("star_rating")))
    )


def reviewer_features(df: DataFrame) -> DataFrame:
    """Per-reviewer aggregates that double as fraud signals."""
    w_all = Window.partitionBy("reviewer_id")
    w_day = Window.partitionBy("reviewer_id", F.to_date("review_ts"))
    enriched = (
        df.withColumn("reviewer_review_count", F.count("*").over(w_all))
        .withColumn("reviewer_avg_rating", F.avg("star_rating").over(w_all))
        .withColumn("reviewer_pct_5star", F.avg((F.col("star_rating") == 5).cast("double")).over(w_all))
        .withColumn("reviewer_distinct_products", F.size(F.collect_set("product_id").over(w_all)))
        .withColumn("reviewer_reviews_same_day", F.count("*").over(w_day))
        .withColumn(
            "reviewer_verified_share",
            F.avg(F.col("verified_purchase").cast("double")).over(w_all),
        )
    )
    return enriched


def product_features(df: DataFrame) -> DataFrame:
    w = Window.partitionBy("product_id")
    return (
        df.withColumn("product_review_count", F.count("*").over(w))
        .withColumn("product_avg_rating", F.avg("star_rating").over(w))
        .withColumn("product_pct_5star", F.avg((F.col("star_rating") == 5).cast("double")).over(w))
    )


def fraud_label(df: DataFrame) -> DataFrame:
    """Heuristic weak label for fraud — exact-duplicate body within a product, or
    a high-velocity reviewer with a 5-star monoculture, or same-day spam.

    A small random relabeling step weakens perfect alignment between the rule
    outputs and the binary target (the full rule-derived columns still exist
    in Parquet, but the label is not 100% deterministic in Y), so the model
    cannot simply memorize predicate outcomes when leaky numerics are withheld.
    """
    dup_w = Window.partitionBy("product_id", "review_body_clean")
    r = F.rand(42)
    rule = (
        (F.col("dup_in_product") >= 3)
        | (
            (F.col("reviewer_review_count") >= 8)
            & (F.col("reviewer_pct_5star") >= 0.95)
            & (F.col("reviewer_verified_share") <= 0.2)
        )
        | (F.col("reviewer_reviews_same_day") >= 5)
    ).cast("int")
    return (
        df.withColumn("dup_in_product", F.count("*").over(dup_w))
        .withColumn("_rule_fraud", rule)
        .withColumn(
            "fraud_label",
            F.when((F.col("_rule_fraud") == 1) & (r < F.lit(0.034)), F.lit(0))
            .when((F.col("_rule_fraud") == 0) & (r < F.lit(0.0015)), F.lit(1))
            .otherwise(F.col("_rule_fraud"))
            .cast("int"),
        )
        .drop("_rule_fraud")
    )


FEATURE_COLS = [
    "review_id",
    "product_id",
    "reviewer_id",
    "star_rating",
    "verified_purchase",
    "helpful_votes",
    "total_votes",
    "review_headline_clean",
    "review_body_clean",
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
    "sentiment_label",
    "fraud_label",
    "review_ts",
    "product_category",
]


def split_train_test(df: DataFrame, test_frac: float, seed: int) -> tuple[DataFrame, DataFrame]:
    return df.randomSplit([1 - test_frac, test_frac], seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=RAW_REVIEWS)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dirs()
    spark = get_spark("amazon-batch-etl")
    spark.sparkContext.setLogLevel("WARN")

    raw = load_raw(spark, args.input)
    cleaned = clean(raw)
    enriched = product_features(reviewer_features(cleaned))
    labeled = fraud_label(enriched).select(*FEATURE_COLS)

    # cache because we read it 4x below
    labeled = labeled.cache()
    total = labeled.count()
    n_fraud = labeled.filter(F.col("fraud_label") == 1).count()
    print(f"[etl] rows={total:,} fraud_positives={n_fraud:,} ({n_fraud / max(total,1):.2%})")

    train, test = split_train_test(labeled, args.test_frac, args.seed)
    train.write.mode("overwrite").parquet(str(TRAIN_PARQUET))
    test.write.mode("overwrite").parquet(str(TEST_PARQUET))

    # Aggregates for dashboards.
    product_agg = (
        labeled.groupBy("product_id", "product_category")
        .agg(
            F.count("*").alias("review_count"),
            F.avg("star_rating").alias("avg_rating"),
            F.avg((F.col("star_rating") == 5).cast("double")).alias("pct_5star"),
            F.sum("fraud_label").alias("fraud_review_count"),
            F.avg("fraud_label").alias("fraud_rate"),
        )
        .orderBy(F.desc("review_count"))
    )
    product_agg.write.mode("overwrite").parquet(str(PRODUCT_AGG_PARQUET))

    reviewer_agg = (
        labeled.groupBy("reviewer_id")
        .agg(
            F.count("*").alias("review_count"),
            F.avg("star_rating").alias("avg_rating"),
            F.avg("fraud_label").alias("fraud_rate"),
            F.avg(F.col("verified_purchase").cast("double")).alias("verified_share"),
        )
        .orderBy(F.desc("fraud_rate"), F.desc("review_count"))
    )
    reviewer_agg.write.mode("overwrite").parquet(str(REVIEWER_AGG_PARQUET))

    print(f"[etl] wrote train={TRAIN_PARQUET}, test={TEST_PARQUET}")
    print(f"[etl] wrote product_agg={PRODUCT_AGG_PARQUET}")
    print(f"[etl] wrote reviewer_agg={REVIEWER_AGG_PARQUET}")
    spark.stop()


if __name__ == "__main__":
    main()
