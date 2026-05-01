"""Generate a synthetic Amazon-reviews-shaped JSONL dataset.

The schema mirrors the public Amazon Customer Reviews dataset
(`registry.opendata.aws/amazon-reviews/`). Sentiment-loaded vocabulary is
sampled per star rating, and a configurable share of reviews is planted with
fraud patterns (duplicate text bursts from a small pool of reviewers, all
5-star, unverified, in a tight time window). Use this when the real dataset
is not available locally.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Make `src.common` importable when running as a script.
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.common.config import RAW_REVIEWS, ensure_dirs  # noqa: E402

POSITIVE_PHRASES = [
    "absolutely love this product",
    "exceeded my expectations",
    "high quality and well made",
    "works perfectly every time",
    "great value for the money",
    "would buy again without hesitation",
    "fast shipping and great packaging",
    "fantastic build quality",
    "exactly as described",
    "highly recommend to everyone",
]

NEUTRAL_PHRASES = [
    "it is okay for the price",
    "does what it says",
    "average product, nothing special",
    "fine but not amazing",
    "works as expected mostly",
    "decent quality but could be better",
    "neither great nor terrible",
    "useable but unremarkable",
]

NEGATIVE_PHRASES = [
    "completely useless and broke fast",
    "terrible quality, do not buy",
    "stopped working after a week",
    "waste of money and time",
    "poorly made and arrived damaged",
    "extremely disappointed with this purchase",
    "did not match the description at all",
    "would never recommend this to anyone",
    "worst product I have ever bought",
]

FILLERS = [
    "I purchased this last month.",
    "The packaging was simple.",
    "Delivery was on time.",
    "I have used several similar products before.",
    "I tested it for two weeks before reviewing.",
    "Customer service was responsive.",
    "It looks just like the picture.",
    "The instructions were easy to follow.",
]

CATEGORIES = [
    "Books",
    "Electronics",
    "Home",
    "Toys",
    "Beauty",
    "Apparel",
    "Kitchen",
    "Sports",
]


def _phrases_for_rating(rating: int) -> list[str]:
    if rating <= 2:
        return NEGATIVE_PHRASES
    if rating == 3:
        return NEUTRAL_PHRASES
    return POSITIVE_PHRASES


def _build_text(rating: int, rng: random.Random) -> tuple[str, str]:
    base = _phrases_for_rating(rating)
    n_main = rng.randint(2, 4)
    n_filler = rng.randint(0, 2)
    parts = rng.sample(base, k=min(n_main, len(base)))
    parts.extend(rng.sample(FILLERS, k=n_filler))
    rng.shuffle(parts)
    body = " ".join(parts)
    headline = parts[0].split(",")[0][:60].capitalize()
    return headline, body + "."


def _rating_distribution(rng: random.Random) -> int:
    """Skew toward 5 stars to mirror real Amazon distributions."""
    r = rng.random()
    if r < 0.55:
        return 5
    if r < 0.75:
        return 4
    if r < 0.85:
        return 3
    if r < 0.93:
        return 2
    return 1


def _emit_organic(rng: random.Random, start: datetime, span_days: int) -> dict:
    rating = _rating_distribution(rng)
    headline, body = _build_text(rating, rng)
    review_date = start + timedelta(days=rng.randint(0, span_days))
    helpful = rng.randint(0, 50)
    total = helpful + rng.randint(0, 20)
    return {
        "review_id": str(uuid.uuid4()),
        "product_id": f"P{rng.randint(1000, 1999):04d}",
        "reviewer_id": f"R{rng.randint(10000, 19999):05d}",
        "star_rating": rating,
        "helpful_votes": helpful,
        "total_votes": total,
        "verified_purchase": rng.random() < 0.85,
        "review_headline": headline,
        "review_body": body,
        "review_date": review_date.strftime("%Y-%m-%d"),
        "product_category": rng.choice(CATEGORIES),
        "event_ts": int(review_date.replace(tzinfo=timezone.utc).timestamp()),
    }


def _emit_fraud_burst(
    rng: random.Random, start: datetime, span_days: int, burst_size: int
) -> list[dict]:
    """A small reviewer ring carpet-bombing a target product with 5-star copy."""
    target_product = f"P{rng.randint(2000, 2099):04d}"
    template_headline, template_body = _build_text(5, rng)
    burst_day = start + timedelta(days=rng.randint(0, span_days))
    reviewers = [f"R{rng.randint(90000, 99999):05d}" for _ in range(max(2, burst_size // 4))]
    out = []
    for i in range(burst_size):
        ts = burst_day + timedelta(minutes=rng.randint(0, 240))
        # Slight mutation to make it look semi-natural.
        body = template_body
        if rng.random() < 0.3:
            body += " " + rng.choice(FILLERS)
        out.append(
            {
                "review_id": str(uuid.uuid4()),
                "product_id": target_product,
                "reviewer_id": rng.choice(reviewers),
                "star_rating": 5,
                "helpful_votes": 0,
                "total_votes": 0,
                "verified_purchase": False,
                "review_headline": template_headline,
                "review_body": body,
                "review_date": ts.strftime("%Y-%m-%d"),
                "product_category": rng.choice(CATEGORIES),
                "event_ts": int(ts.replace(tzinfo=timezone.utc).timestamp()),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--fraud-share", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=RAW_REVIEWS)
    args = parser.parse_args()

    ensure_dirs()
    rng = random.Random(args.seed)
    start = datetime(2023, 1, 1)
    span = 730  # ~2 years

    n_fraud = int(args.rows * args.fraud_share)
    n_organic = args.rows - n_fraud

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    written = 0
    with args.out.open("w") as f:
        for _ in range(n_organic):
            f.write(json.dumps(_emit_organic(rng, start, span)) + "\n")
            written += 1
        # Emit fraud as bursts of varying size.
        remaining = n_fraud
        while remaining > 0:
            burst = min(remaining, rng.randint(15, 60))
            for rec in _emit_fraud_burst(rng, start, span, burst):
                f.write(json.dumps(rec) + "\n")
                written += 1
            remaining -= burst

    print(
        f"Wrote {written:,} reviews to {args.out} "
        f"({n_fraud:,} planted as fraud) in {time.time() - t0:.1f}s"
    )


if __name__ == "__main__":
    main()
