"""Publish a single user-supplied review to the Kafka 'reviews' topic.

Useful for demo: type your own review and watch it flow through Kafka → Spark
→ FastAPI streaming feed in real time.

Example:
    python3 -m scripts.publish_one_review \
        --body "Hi, I have used the Chanel perfume and it is awesome" \
        --rating 5 --product B0CHANEL01 --reviewer R_DEMO_USER

By default it makes up a product_id and reviewer_id if you don't pass any.
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from datetime import date

from kafka import KafkaProducer  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body", required=True, help="review text")
    parser.add_argument("--rating", type=int, default=5, help="star rating 1-5")
    parser.add_argument("--headline", default="user-submitted review")
    parser.add_argument("--product", default=None, help="product_id (random if omitted)")
    parser.add_argument("--reviewer", default=None, help="reviewer_id (random if omitted)")
    parser.add_argument("--verified", type=lambda s: s.lower() in ("1", "true", "yes", "y"), default=True)
    parser.add_argument("--broker", default="localhost:9092")
    parser.add_argument("--topic", default="reviews")
    args = parser.parse_args()

    record = {
        "review_id": str(uuid.uuid4()),
        "product_id": args.product or f"BUSER{uuid.uuid4().hex[:6].upper()}",
        "reviewer_id": args.reviewer or f"R{uuid.uuid4().hex[:24].upper()}",
        "star_rating": args.rating,
        "helpful_votes": 0,
        "total_votes": 0,
        "verified_purchase": bool(args.verified),
        "review_headline": args.headline,
        "review_body": args.body,
        "review_date": date.today().isoformat(),
        "product_category": "User Submitted",
        "event_ts": int(time.time()),
    }

    producer = KafkaProducer(
        bootstrap_servers=args.broker,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    future = producer.send(args.topic, value=record)
    metadata = future.get(timeout=10)
    producer.flush()
    producer.close()

    print(f"[publish] review_id={record['review_id']}")
    print(f"[publish] topic={metadata.topic}  partition={metadata.partition}  offset={metadata.offset}")
    print(f"[publish] sent at: {time.strftime('%H:%M:%S')}")
    print()
    print("Now watch:")
    print(f"  - Kafka UI       http://localhost:8090   → topic 'reviews' → message at offset {metadata.offset}")
    print(f"  - Spark UI       http://localhost:4040   → next micro-batch in <5s")
    print(f"  - FastAPI feed   http://127.0.0.1:8000   → 'Streaming feed' panel after Spark scores it")


if __name__ == "__main__":
    main()
