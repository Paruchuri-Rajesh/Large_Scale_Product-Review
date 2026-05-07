"""Spark big-data demo: 20.5M rows, multi-stage aggregations with shuffles.

Run:
    python3 -m scripts.spark_bigdata_demo

While it runs, open http://localhost:4040 to watch jobs, stages, tasks,
shuffle read/write, executor memory, and the SQL physical plan in real time.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from pyspark.sql import functions as F  # noqa: E402

from src.common.spark import get_spark  # noqa: E402

TRAIN_PARQUET = "data/processed/train.parquet"
TEST_PARQUET = "data/processed/test.parquet"


def main() -> None:
    spark = get_spark("spark-bigdata-demo")
    sc = spark.sparkContext
    print(f"Spark version          : {spark.version}")
    print(f"Spark UI               : {sc.uiWebUrl}")
    print(f"defaultParallelism     : {sc.defaultParallelism}")
    print(f"shuffle.partitions     : {spark.conf.get('spark.sql.shuffle.partitions')}")
    print(f"AQE enabled            : {spark.conf.get('spark.sql.adaptive.enabled')}")
    print(f"driver memory          : {spark.conf.get('spark.driver.memory')}")

    print("\n--- Loading 20.5M-row corpus (train+test parquet) ---")
    t0 = time.time()
    df = (
        spark.read.parquet(TRAIN_PARQUET)
        .unionByName(spark.read.parquet(TEST_PARQUET), allowMissingColumns=True)
    )
    n_partitions = df.rdd.getNumPartitions()
    print(f"input partitions       : {n_partitions}")

    n_rows = df.count()
    print(f"total rows             : {n_rows:,}  (load+count: {time.time()-t0:.1f}s)")

    # Stage 1 — narrow aggregation, no shuffle (each partition reduces locally)
    print("\n--- Stage 1: avg star_rating + count [no shuffle] ---")
    t1 = time.time()
    r = df.agg(F.avg("star_rating").alias("avg"),
               F.count(F.lit(1)).alias("n"),
               F.avg("fraud_label").alias("fraud_rate")).collect()[0]
    print(f"  avg star_rating={r['avg']:.3f}  n={r['n']:,}  global fraud_rate={r['fraud_rate']:.4f}  ({time.time()-t1:.1f}s)")

    # Stage 2 — shuffle: top reviewers by review count
    print("\n--- Stage 2: top-10 reviewers by review_count [shuffle by reviewer_id] ---")
    t2 = time.time()
    top_reviewers = (
        df.groupBy("reviewer_id")
        .agg(F.count("*").alias("n"),
             F.avg("star_rating").alias("avg_rating"),
             F.avg("fraud_label").alias("fraud_rate"))
        .filter("n >= 10")
        .orderBy(F.col("n").desc())
        .limit(10)
        .collect()
    )
    print(f"  ({time.time()-t2:.1f}s)")
    for row in top_reviewers:
        print(f"    {row['reviewer_id']:30s}  n={row['n']:>5d}  avg={row['avg_rating']:.2f}  fraud_rate={row['fraud_rate']:.3f}")

    # Stage 3 — heavier shuffle: per-product agg, filter by min count, sort
    print("\n--- Stage 3: highest fraud-rate products with >=200 reviews [shuffle by product_id] ---")
    t3 = time.time()
    suspicious = (
        df.groupBy("product_id")
        .agg(F.count("*").alias("review_count"),
             F.avg("star_rating").alias("avg_rating"),
             F.avg("fraud_label").alias("fraud_rate"))
        .filter("review_count >= 200")
        .orderBy(F.col("fraud_rate").desc())
        .limit(5)
        .collect()
    )
    print(f"  ({time.time()-t3:.1f}s)")
    for row in suspicious:
        print(f"    {row['product_id']:12s}  n={row['review_count']:>5d}  avg={row['avg_rating']:.2f}  fraud_rate={row['fraud_rate']:.3f}")

    # Stage 4 — join: per-product agg back to per-review level, count above-threshold reviews
    print("\n--- Stage 4: shuffle-hash join, per-product fraud_rate joined back to reviews ---")
    t4 = time.time()
    prod_agg = df.groupBy("product_id").agg(F.avg("fraud_label").alias("p_fraud_rate"))
    joined = df.join(prod_agg, on="product_id", how="inner")
    n_high_risk = joined.filter("p_fraud_rate >= 0.20").count()
    print(f"  rows on products with fraud_rate >= 0.20 : {n_high_risk:,}  ({time.time()-t4:.1f}s)")

    print(f"\nTotal job time: {time.time()-t0:.1f}s")
    print(f"\nSpark UI still up at: {sc.uiWebUrl}")
    print("Sleeping 90s before exit so you can browse the UI ...")
    time.sleep(90)
    spark.stop()


if __name__ == "__main__":
    main()
