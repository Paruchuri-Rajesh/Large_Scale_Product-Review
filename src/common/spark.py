"""Spark session factory."""
from __future__ import annotations

from pyspark.sql import SparkSession


def get_spark(app_name: str = "amazon-reviews", shuffle_partitions: int = 8) -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
