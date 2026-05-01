"""Spark Structured Streaming review scorer (foreachBatch flavor).

Reads newline-delimited JSON files dropped into ``data/streaming_in/`` and
writes scored predictions to ``data/streaming_out/scored/`` (also prints to
console). Scoring runs in ``foreachBatch`` against pandas + the persisted
sklearn pipelines — that keeps Python<->JVM serialization simple and avoids
the pandas-UDF broadcast pickle path, which is finicky on some Python
toolchains.

Run:
    python -m src.stream.score_stream
Then drop JSONL files into data/streaming_in/. See scripts/feed_stream.py for
a small producer.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from pyspark.sql import DataFrame, functions as F  # noqa: E402

from src.common.config import (  # noqa: E402
    FRAUD_MODEL_PATH,
    SENTIMENT_LABELS,
    SENTIMENT_MODEL_PATH,
    STREAM_IN_DIR,
    STREAM_OUT_DIR,
    ensure_dirs,
)
from src.common.schema import REVIEW_SCHEMA  # noqa: E402
from src.common.spark import get_spark  # noqa: E402
from src.train.train import NUMERIC_FRAUD_FEATURES  # noqa: E402


def _enrich_for_serving(pdf: pd.DataFrame) -> pd.DataFrame:
    """Compute the per-row features the fraud model expects.

    Behavioral aggregates (reviewer/product history) are unknown for a
    never-seen review, so they fall back to neutral defaults; the model still
    leans on TF-IDF + per-row signals.
    """
    out = pdf.copy()
    body = out.get("review_body", pd.Series([], dtype=str)).fillna("").astype(str).str.lower()
    body = body.str.replace(r"https?://\S+|www\.\S+", " ", regex=True)
    body = body.str.replace(r"<[^>]+>", " ", regex=True)
    body = body.str.replace(r"[^a-z0-9\s']", " ", regex=True)
    body = body.str.replace(r"\s+", " ", regex=True).str.strip()
    out["review_body_clean"] = body

    out["body_len"] = body.str.len().astype(int)
    out["body_word_count"] = body.str.split(" ").map(lambda xs: len([x for x in xs if x])).astype(int)
    out["exclam_count"] = (
        out.get("review_body", pd.Series([""] * len(out))).fillna("").astype(str).str.count("!").astype(int)
    )
    out["verified_purchase_int"] = out.get("verified_purchase", pd.Series([False] * len(out))).fillna(False).astype(int)
    out["star_rating"] = pd.to_numeric(out.get("star_rating", pd.Series([3] * len(out))), errors="coerce").fillna(3).astype(int)
    out["helpful_votes"] = pd.to_numeric(out.get("helpful_votes", pd.Series([0] * len(out))), errors="coerce").fillna(0).astype(int)
    out["total_votes"] = pd.to_numeric(out.get("total_votes", pd.Series([0] * len(out))), errors="coerce").fillna(0).astype(int)

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
    parser.add_argument("--input-dir", type=Path, default=STREAM_IN_DIR)
    parser.add_argument("--output-dir", type=Path, default=STREAM_OUT_DIR)
    parser.add_argument("--trigger-seconds", type=int, default=5)
    parser.add_argument(
        "--once", action="store_true", help="Process whatever is there once and exit (for tests)."
    )
    args = parser.parse_args()

    ensure_dirs()
    args.input_dir.mkdir(parents=True, exist_ok=True)
    out_scored = args.output_dir / "scored"
    out_scored.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "_checkpoints"
    checkpoint.mkdir(parents=True, exist_ok=True)

    spark = get_spark("amazon-stream-scorer", shuffle_partitions=4)
    spark.sparkContext.setLogLevel("WARN")

    sentiment = joblib.load(SENTIMENT_MODEL_PATH)
    fraud = joblib.load(FRAUD_MODEL_PATH)

    def score_and_write(batch_df: DataFrame, batch_id: int) -> None:
        if batch_df.rdd.isEmpty():
            return
        pdf = batch_df.toPandas()
        if pdf.empty:
            return
        feats = _enrich_for_serving(pdf)
        sent_label = sentiment.predict(feats["review_body_clean"]).astype(int)
        feat_cols = ["review_body_clean"] + NUMERIC_FRAUD_FEATURES
        fraud_proba = fraud.predict_proba(feats[feat_cols])[:, 1]

        out = pd.DataFrame(
            {
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
            }
        )
        # Console preview.
        print(f"\n[stream batch {batch_id}] {len(out)} rows")
        print(
            out[
                [
                    "review_id",
                    "product_id",
                    "star_rating",
                    "sentiment",
                    "fraud_proba",
                    "fraud_flag",
                ]
            ]
            .head(5)
            .to_string(index=False)
        )
        out_path = out_scored / f"batch-{batch_id:08d}.json"
        out.to_json(out_path, orient="records", lines=True)

    stream = (
        spark.readStream.schema(REVIEW_SCHEMA)
        .option("maxFilesPerTrigger", 5)
        .json(str(args.input_dir))
    )

    query = (
        stream.writeStream.foreachBatch(score_and_write)
        .option("checkpointLocation", str(checkpoint / "fb"))
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
