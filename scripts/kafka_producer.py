"""Kafka producer: reads reviews.jsonl and publishes to a Kafka topic.
Replaces the file-drop drip producer (scripts/feed_stream.py).

Run (with Kafka broker on localhost:9092):
    python -m scripts.kafka_producer
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import uuid
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from kafka import KafkaProducer  # type: ignore
from src.common.config import RAW_REVIEWS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=RAW_REVIEWS)
    parser.add_argument("--broker", default="localhost:9092")
    parser.add_argument("--topic", default="reviews")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--n-batches", type=int, default=10)
    parser.add_argument("--sleep", type=float, default=2.0)
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(f"source {args.source} does not exist; run ingest first")

    producer = KafkaProducer(
        bootstrap_servers=args.broker,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    with args.source.open() as fh:
        lines = fh.readlines()

    # Cycle if not enough lines
    if len(lines) < args.batch_size * args.n_batches:
        lines = lines * ((args.batch_size * args.n_batches) // len(lines) + 1)

    cursor = 0
    for i in range(args.n_batches):
        chunk = lines[cursor: cursor + args.batch_size]
        cursor += args.batch_size
        for line in chunk:
            record = json.loads(line)
            producer.send(args.topic, value=record)
        producer.flush()
        print(f"[kafka-producer] batch {i+1}/{args.n_batches} → topic={args.topic} ({len(chunk)} msgs)")
        time.sleep(args.sleep)

    producer.close()
    print("[kafka-producer] done")


if __name__ == "__main__":
    main()