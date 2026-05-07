"""Live tap on the Kafka 'reviews' topic — prints each incoming message.

Use as a watcher to see what's flowing through the broker in real time.
Reads from the latest offset (only new messages produced after start).
"""
from __future__ import annotations

import json
import sys

from kafka import KafkaConsumer  # type: ignore


def main() -> None:
    consumer = KafkaConsumer(
        "reviews",
        bootstrap_servers="localhost:9092",
        auto_offset_reset="latest",
        enable_auto_commit=False,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        group_id=None,
    )
    print("[kafka-tap] subscribed to topic 'reviews', reading from latest", flush=True)
    for msg in consumer:
        d = msg.value
        body = (d.get("review_body") or "").replace("\n", " ")[:70]
        print(
            f"[in→kafka offset={msg.offset:>5}] product={(d.get('product_id') or '')[:10]:<10}  "
            f"rating={d.get('star_rating')}  body=\"{body}...\"",
            flush=True,
        )


if __name__ == "__main__":
    main()
