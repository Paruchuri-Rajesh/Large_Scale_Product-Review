"""Spark Structured Streaming scorer — Kafka source."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from pyspark.sql import DataFrame, functions as F
from pyspark.sql.types import StringType

from src.common.config import (
    FRAUD_MODEL_PATH, SENTIMENT_LABELS, SENTIMENT_MODEL_PATH,
    STREAM_OUT_DIR, ensure_dirs,
)
from src.common.spark import get_spark
from src.common.text import clean_text
from src.train.features import get_fraud_numeric_features

NUMERIC_FRAUD_FEATURES = get_fraud_numeric_features()


def _enrich(pdf: pd.DataFrame) -> pd.DataFrame:
    out = pdf.copy()
    body = out.get("review_body", pd.Series([], dtype=str)).fillna("").astype(str)
    out["review_body_clean"] = body.apply(clean_text)
    out["body_len"] = out["review_body_clean"].str.len().astype(int)
    out["body_word_count"] = out["review_body_clean"].str.split().map(len).astype(int)
    out["exclam_count"] = body.str.count("!").astype(int)
    out["verified_purchase_int"] = out.get("verified_purchase", pd.Series([False]*len(out))).fillna(False).astype(int)
    out["star_rating"] = pd.to_numeric(out.get("star_rating", pd.Series([3]*len(out))), errors="coerce").fillna(3).astype(int)
    out["helpful_votes"] = pd.to_numeric(out.get("helpful_votes", pd.Series([0]*len(out))), errors="coerce").fillna(0).astype(int)
    out["total_votes"] = pd.to_numeric(out.get("total_votes", pd.Series([0]*len(out))), errors="coerce").fillna(0).astype(int)
    out["reviewer_review_count"] = 1
    out["reviewer_avg_rating"] = out["star_rating"].astype(float)
    out["reviewer_pct_5star"] = (out["star_rating"] == 5).astype(float)
    out["reviewer_distinct_products"] = 1
    out["reviewer_reviews_same_day"] = 1
    out["reviewer_verified_share"] = out["verified_purchase_int"].astype(float)
    out["product_review_count"] = 1
    out["product_avg_rating"] = out["star_rating"].astype(float)
    out["product_pct_5star"] = (out["star_rating"] == 5).astype(float)
    out["dup_in_product"] = 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--broker", default="localhost:9092")
    parser.add_argument("--topic", default="reviews")
    parser.add_argument("--trigger-seconds", type=int, default=5)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    out_scored = STREAM_OUT_DIR / "scored"
    out_scored.mkdir(parents=True, exist_ok=True)
    checkpoint = STREAM_OUT_DIR / "_checkpoints_kafka"
    checkpoint.mkdir(parents=True, exist_ok=True)

    spark = get_spark("amazon-kafka-scorer", shuffle_partitions=4)
    spark.sparkContext.setLogLevel("WARN")

    sentiment = joblib.load(SENTIMENT_MODEL_PATH)
    fraud = joblib.load(FRAUD_MODEL_PATH)

    kafka_df = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", args.broker)
        .option("subscribe", args.topic)
        .option("startingOffsets", "latest")
        .load()
    )

    reviews_df = kafka_df.select(F.col("value").cast(StringType()).alias("raw_json"))

    def score_and_write(batch_df: DataFrame, batch_id: int) -> None:
        if batch_df.rdd.isEmpty():
            return
        raw_rows = [row["raw_json"] for row in batch_df.collect()]
        records = []
        for r in raw_rows:
            try:
                records.append(json.loads(r))
            except Exception:
                continue
        if not records:
            return

        pdf = pd.DataFrame(records)
        feats = _enrich(pdf)
        sent_label = sentiment.predict(feats["review_body_clean"]).astype(int)
        feat_cols = ["review_body_clean"] + NUMERIC_FRAUD_FEATURES
        fraud_proba = fraud.predict_proba(feats[feat_cols])[:, 1]

        out = pd.DataFrame({
            "review_id": pdf.get("review_id"),
            "product_id": pdf.get("product_id"),
            "reviewer_id": pdf.get("reviewer_id"),
            "star_rating": pdf.get("star_rating"),
            "review_body": pdf.get("review_body"),
            "sentiment_label": sent_label,
            "sentiment": [SENTIMENT_LABELS[int(x)] for x in sent_label],
            "fraud_proba": fraud_proba.astype(float),
            "fraud_flag": (fraud_proba >= 0.5).astype(int),
            "scored_at": pd.Timestamp.utcnow().isoformat(),
            "batch_id": batch_id,
            "source": "kafka",
        })

        print(f"\n[kafka-stream batch {batch_id}] {len(out)} rows scored")
        print(out[["review_id", "star_rating", "sentiment", "fraud_proba", "fraud_flag"]].head(5).to_string(index=False))

        out_path = out_scored / f"kafka-batch-{batch_id:08d}.json"
        out.to_json(out_path, orient="records", lines=True)

    query = (
        reviews_df.writeStream
        .foreachBatch(score_and_write)
        .option("checkpointLocation", str(checkpoint))
        .trigger(processingTime=f"{args.trigger_seconds} seconds")
        .start()
    )

    if args.once:
        query.processAllAvailable()
        query.stop()
    else:
        query.awaitTermination()


if __name__ == "__main__":
    main()
