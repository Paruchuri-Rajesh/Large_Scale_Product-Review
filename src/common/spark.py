"""Spark session factory."""
from __future__ import annotations

import os

from pyspark.sql import SparkSession


def get_spark(app_name: str = "amazon-reviews", shuffle_partitions: int = 200) -> SparkSession:
    driver_mem = os.environ.get("SPARK_DRIVER_MEM", "8g")
    max_result = os.environ.get("SPARK_MAX_RESULT_SIZE", "4g")
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.memory", driver_mem)
        .config("spark.driver.maxResultSize", max_result)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
