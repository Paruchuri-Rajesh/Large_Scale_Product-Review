"""Spark schema for raw Amazon reviews JSONL records."""
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    LongType,
    DoubleType,
    BooleanType,
)

REVIEW_SCHEMA = StructType(
    [
        StructField("review_id", StringType(), True),
        StructField("product_id", StringType(), True),
        StructField("reviewer_id", StringType(), True),
        StructField("star_rating", IntegerType(), True),
        StructField("helpful_votes", IntegerType(), True),
        StructField("total_votes", IntegerType(), True),
        StructField("verified_purchase", BooleanType(), True),
        StructField("review_headline", StringType(), True),
        StructField("review_body", StringType(), True),
        StructField("review_date", StringType(), True),  # YYYY-MM-DD
        StructField("product_category", StringType(), True),
        StructField("event_ts", LongType(), True),  # epoch seconds (streaming)
    ]
)
