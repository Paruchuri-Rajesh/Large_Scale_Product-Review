"""Generate a synthetic Amazon-reviews-shaped JSONL dataset.

The schema mirrors the public Amazon Customer Reviews dataset
(`registry.opendata.aws/amazon-reviews/`). Sentiment-loaded vocabulary is
sampled per star rating, with cross-class phrase overlap and configurable
noise so text classification is not trivially separable. Fraud-like rows
mix exact-duplicate bursts (high dup-in-product signal), same-day velocity
with paraphrased bodies (strong temporal signal without identical text), and
slow rings that trigger reviewer-level heuristics — plus organic "hard
negatives" that look superficially suspicious but stay below `batch_etl`
fraud thresholds so the model sees borderline negatives.

Weak fraud labels in `batch_etl.fraud_label` require duplicate bodies on a
product and/or reviewer velocity patterns; the generator is shaped so those
rules still fire on many planted rows while reducing trivial copy-paste
dominance at higher difficulty.
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

# ---------------------------------------------------------------------------
# Phrase pools — overlap across sentiment buckets makes linear separation
# harder (shared n-grams between classes).
# ---------------------------------------------------------------------------
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
    "average product nothing special",
    "fine but not amazing",
    "works as expected mostly",
    "decent quality but could be better",
    "neither great nor terrible",
    "useable but unremarkable",
]

NEGATIVE_PHRASES = [
    "completely useless and broke fast",
    "terrible quality do not buy",
    "stopped working after a week",
    "waste of money and time",
    "poorly made and arrived damaged",
    "extremely disappointed with this purchase",
    "did not match the description at all",
    "would never recommend this to anyone",
    "worst product I have ever bought",
]

# Mild poles — used for mixed-sentiment and edge ratings (overlap by design).
MILD_POSITIVE = [
    "pretty happy overall",
    "mostly satisfied",
    "good enough for daily use",
    "better than I feared",
    "pleasantly surprised",
]

MILD_NEGATIVE = [
    "a bit disappointed honestly",
    "not great but usable",
    "mixed feelings about it",
    "could have been better",
    "sort of underwhelming",
]

# Ambiguous strings sampled into multiple star buckets so neutrals share
# tokens with strong positives/negatives — reduces trivial TF-IDF separation.
SHARED_AMBIGUOUS = [
    "quality seems okay",
    "shipping was fine",
    "worth considering maybe",
    "not bad overall really",
    "does the job I guess",
    "okay product for casual use",
    "mixed experience honestly",
    "fine for the price point",
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

CASUAL_SHORT = [
    "meh.",
    "Its fine.",
    "ok.",
    "whatever works.",
    "nah.",
    "yeah good.",
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

# Busy SKUs — many *distinct* organic bodies raise product counts without fraud-level dup clusters.
POPULAR_PRODUCT_IDS = [f"P{1600 + i:04d}" for i in range(55)]

# Paraphrased fraud-ish praise — same manipulative intent, different surface
# text so dup_in_product stays low while same-day velocity still triggers ETL.
FRAUD_PARAPHRASES = [
    "Absolutely love this purchase five stars all the way.",
    "Really impressed with this item shipping was quick too.",
    "Cannot recommend enough fantastic quality for the money.",
    "Exceeded every expectation would buy again in a heartbeat.",
    "Top notch product exactly what I needed right now.",
    "Super happy with this order arrived ahead of schedule.",
    "Blown away by the value here truly outstanding stuff.",
    "Perfect fit for my needs five stars without hesitation.",
    "Amazing deal quality far exceeds the listing photos.",
    "So pleased with this purchase everything works beautifully.",
    "Five stars easy this is exactly what the reviews promised.",
    "Incredible product wish I had bought sooner honestly.",
    "Rock solid quality shipping packaging both were great.",
    "Love it love it love it no complaints whatsoever.",
    "Prime example of great manufacturing highly satisfied customer.",
    "Wonderful experience from click to delivery to first use.",
    "Best purchase I made this month hands down recommend.",
    "Flawless transaction item matches description perfectly.",
    "Stellar performance could not ask for more really.",
    "Very satisfied customer posting this right after unboxing.",
]

# ---------------------------------------------------------------------------
# Difficulty presets: control overlap, noise, fraud mix, hard negatives.
# ---------------------------------------------------------------------------
def _preset(difficulty: str) -> dict[str, float | int]:
    d = difficulty.strip().lower()
    if d == "easy":
        return {
            "phrase_overlap_prob": 0.08,
            "mixed_sentiment_prob": 0.06,
            "noise_typo_prob": 0.05,
            "noise_punct_prob": 0.06,
            "casual_short_prob": 0.02,
            "fraud_weight_dup": 0.72,
            "fraud_weight_same_day": 0.18,
            "fraud_weight_slow_ring": 0.10,
            "dup_burst_min": 12,
            "dup_burst_max": 48,
            "same_day_burst_min": 6,
            "same_day_burst_max": 14,
            "same_day_verified_fraud_prob": 0.12,
            "same_day_non_five_star_prob": 0.04,
            "slow_ring_reviews": 10,
            "hard_negative_clusters": 12,
            "reviews_per_hard_neg_cluster": 7,
        }
    if d == "hard":
        return {
            "phrase_overlap_prob": 0.42,
            "mixed_sentiment_prob": 0.28,
            "noise_typo_prob": 0.22,
            "noise_punct_prob": 0.24,
            "casual_short_prob": 0.12,
            "fraud_weight_dup": 0.12,
            "fraud_weight_same_day": 0.54,
            "fraud_weight_slow_ring": 0.34,
            "dup_burst_min": 8,
            "dup_burst_max": 28,
            "same_day_burst_min": 7,
            "same_day_burst_max": 18,
            "same_day_verified_fraud_prob": 0.35,
            "same_day_non_five_star_prob": 0.14,
            "slow_ring_reviews": 11,
            "hard_negative_clusters": 48,
            "reviews_per_hard_neg_cluster": 7,
        }
    # medium (default)
    return {
        "phrase_overlap_prob": 0.22,
        "mixed_sentiment_prob": 0.14,
        "noise_typo_prob": 0.11,
        "noise_punct_prob": 0.14,
        "casual_short_prob": 0.06,
        "fraud_weight_dup": 0.26,
        "fraud_weight_same_day": 0.46,
        "fraud_weight_slow_ring": 0.28,
        "dup_burst_min": 8,
        "dup_burst_max": 34,
        "same_day_burst_min": 6,
        "same_day_burst_max": 16,
        "same_day_verified_fraud_prob": 0.22,
        "same_day_non_five_star_prob": 0.08,
        "slow_ring_reviews": 10,
        "hard_negative_clusters": 28,
        "reviews_per_hard_neg_cluster": 7,
    }


def _phrases_for_rating(rating: int) -> list[str]:
    if rating <= 2:
        return list(NEGATIVE_PHRASES)
    if rating == 3:
        return list(NEUTRAL_PHRASES)
    if rating == 4:
        return list(POSITIVE_PHRASES + MILD_POSITIVE)
    return list(POSITIVE_PHRASES)


def _inject_typo(text: str, rng: random.Random, intensity: float) -> str:
    """Light character noise — intensity scales typo rate (controlled realism)."""
    if intensity <= 0 or len(text) < 4:
        return text
    chars = list(text)
    for i in range(len(chars)):
        if rng.random() > intensity:
            continue
        c = chars[i]
        if c.isalpha() and rng.random() < 0.45:
            chars[i] = rng.choice("abcdefghijklmnopqrstuvwxyz")
        elif c == " " and rng.random() < 0.25:
            chars[i] = ""
    return "".join(chars)


def _inject_punctuation_noise(text: str, rng: random.Random, prob: float) -> str:
    if rng.random() > prob:
        return text
    suffix = rng.choice(["...", "!!", "?!", " — okay.", " haha.", " idk."])
    return text.rstrip(".") + suffix


def _build_text(rating: int, rng: random.Random, preset: dict[str, float | int]) -> tuple[str, str]:
    base = _phrases_for_rating(rating)
    n_main = rng.randint(2, 4)
    n_filler = rng.randint(0, 2)
    parts = rng.sample(base, k=min(n_main, len(base)))
    parts.extend(rng.sample(FILLERS, k=min(n_filler, len(FILLERS))))

    # Overlap: ambiguous phrases appear across ratings — harder classification.
    if rng.random() < float(preset["phrase_overlap_prob"]):
        parts.append(rng.choice(SHARED_AMBIGUOUS))

    # Mixed sentiment: mild opposite polarity cue (esp. borderline ratings).
    if rng.random() < float(preset["mixed_sentiment_prob"]):
        if rating >= 4:
            parts.append(rng.choice(MILD_NEGATIVE))
        elif rating <= 2:
            parts.append(rng.choice(MILD_POSITIVE))
        elif rating == 3:
            parts.append(rng.choice(rng.choice([POSITIVE_PHRASES[:4], NEGATIVE_PHRASES[:4]])))

    rng.shuffle(parts)
    body = " ".join(parts)

    if rng.random() < float(preset["casual_short_prob"]) and len(body) > 80:
        body = rng.choice(CASUAL_SHORT) + " " + body

    body = _inject_typo(body, rng, float(preset["noise_typo_prob"]))
    body = _inject_punctuation_noise(body, rng, float(preset["noise_punct_prob"]))
    body = body.strip()
    if len(body) < 12:
        body = body + " " + rng.choice(FILLERS)

    headline = parts[0].split(",")[0][:58].strip().capitalize() if parts else "Review"
    return headline, body + "."


def _rating_distribution(rng: random.Random) -> int:
    r = rng.random()
    if r < 0.52:
        return 5
    if r < 0.74:
        return 4
    if r < 0.84:
        return 3
    if r < 0.93:
        return 2
    return 1


def _base_record(
    *,
    rng: random.Random,
    product_id: str,
    reviewer_id: str,
    star_rating: int,
    verified_purchase: bool,
    headline: str,
    body: str,
    review_ts: datetime,
) -> dict:
    helpful = rng.randint(0, 50)
    total = helpful + rng.randint(0, 20)
    return {
        "review_id": str(uuid.uuid4()),
        "product_id": product_id,
        "reviewer_id": reviewer_id,
        "star_rating": star_rating,
        "helpful_votes": helpful,
        "total_votes": total,
        "verified_purchase": verified_purchase,
        "review_headline": headline,
        "review_body": body,
        "review_date": review_ts.strftime("%Y-%m-%d"),
        "product_category": rng.choice(CATEGORIES),
        "event_ts": int(review_ts.replace(tzinfo=timezone.utc).timestamp()),
    }


def _emit_organic(
    rng: random.Random,
    start: datetime,
    span_days: int,
    preset: dict[str, float | int],
) -> dict:
    rating = _rating_distribution(rng)
    headline, body = _build_text(rating, rng, preset)
    review_date = start + timedelta(days=rng.randint(0, span_days))
    return _base_record(
        rng=rng,
        product_id=f"P{rng.randint(1000, 1999):04d}",
        reviewer_id=f"R{rng.randint(10000, 19999):05d}",
        star_rating=rating,
        verified_purchase=rng.random() < 0.84,
        headline=headline,
        body=body,
        review_ts=review_date,
    )


def _emit_organic_fixed_product(
    rng: random.Random,
    start: datetime,
    span_days: int,
    preset: dict[str, float | int],
    product_id: str,
) -> dict:
    """Organic traffic concentrated on popular SKUs — overlapping vocabulary, not duplicate bodies."""
    rating = _rating_distribution(rng)
    headline, body = _build_text(rating, rng, preset)
    review_date = start + timedelta(days=rng.randint(0, span_days))
    return _base_record(
        rng=rng,
        product_id=product_id,
        reviewer_id=f"R{rng.randint(10000, 19999):05d}",
        star_rating=rating,
        verified_purchase=rng.random() < 0.86,
        headline=headline,
        body=body,
        review_ts=review_date,
    )


def _emit_clean_same_day_burst(
    rng: random.Random,
    start: datetime,
    span_days: int,
    preset: dict[str, float | int],
) -> list[dict]:
    """Four organic posts same calendar day — realistic burst without hitting same-day>=5."""
    burst_day = start + timedelta(days=rng.randint(0, span_days))
    reviewer_id = f"R{rng.randint(63000, 63999):05d}"
    out = []
    for _ in range(4):
        rating = rng.choices([4, 5], weights=[0.28, 0.72])[0]
        headline, body = _build_text(rating, rng, preset)
        ts = burst_day + timedelta(minutes=rng.randint(0, 200))
        out.append(
            _base_record(
                rng=rng,
                product_id=f"P{rng.randint(1700, 1799):04d}",
                reviewer_id=reviewer_id,
                star_rating=rating,
                verified_purchase=rng.random() < 0.88,
                headline=headline,
                body=body,
                review_ts=ts,
            )
        )
    return out


def _emit_hard_negative_organic(
    rng: random.Random,
    start: datetime,
    span_days: int,
    preset: dict[str, float | int],
    reviewer_id: str,
    product_id: str,
) -> dict:
    """Superficially suspicious volume later but verified mix avoids fraud heuristic.

    Hard negatives matter: without them the model equates 'many 5-stars +
    low verified' with fraud only and does not learn nuance at the margin.
    """
    rating = rng.choices([4, 5], weights=[0.35, 0.65])[0]
    headline, body = _build_text(rating, rng, preset)
    review_date = start + timedelta(days=rng.randint(0, span_days))
    verified = rng.random() < 0.62
    return _base_record(
        rng=rng,
        product_id=product_id,
        reviewer_id=reviewer_id,
        star_rating=rating,
        verified_purchase=verified,
        headline=headline,
        body=body,
        review_ts=review_date,
    )


def _emit_fraud_duplicate_burst(
    rng: random.Random,
    start: datetime,
    span_days: int,
    burst_size: int,
    preset: dict[str, float | int],
) -> list[dict]:
    """Classic coordinated duplicate carpet-bomb — keeps dup_in_product signal."""
    target_product = f"P{rng.randint(2000, 2099):04d}"
    template_headline, template_body = _build_text(5, rng, preset)
    burst_day = start + timedelta(days=rng.randint(0, span_days))
    reviewers = [
        f"R{rng.randint(90000, 99999):05d}" for _ in range(max(2, burst_size // 5))
    ]
    out = []
    for _ in range(burst_size):
        ts = burst_day + timedelta(minutes=rng.randint(0, 240))
        body = template_body
        if rng.random() < 0.28:
            body = body + " " + rng.choice(FILLERS)
        out.append(
            _base_record(
                rng=rng,
                product_id=target_product,
                reviewer_id=rng.choice(reviewers),
                star_rating=5,
                verified_purchase=False,
                headline=template_headline,
                body=body,
                review_ts=ts,
            )
        )
    return out


def _emit_fraud_same_day_paraphrase(
    rng: random.Random,
    start: datetime,
    span_days: int,
    burst_size: int,
    preset: dict[str, float | int],
) -> list[dict]:
    """Velocity without identical bodies — paraphrases reduce trivial duplicate counts."""
    target_product = f"P{rng.randint(2100, 2199):04d}"
    burst_day = start + timedelta(days=rng.randint(0, span_days))
    reviewer_id = f"R{rng.randint(92000, 98999):05d}"
    templates = FRAUD_PARAPHRASES.copy()
    rng.shuffle(templates)
    out = []
    for i in range(burst_size):
        ts = burst_day + timedelta(minutes=rng.randint(0, 360))
        raw_body = templates[i % len(templates)]
        body = _inject_typo(raw_body, rng, float(preset["noise_typo_prob"]) * 0.5)
        if rng.random() < 0.15:
            body = body + " " + rng.choice(FILLERS[:3])
        headline = raw_body.split(".")[0][:58].strip().capitalize()
        vf = rng.random() < float(preset["same_day_verified_fraud_prob"])
        star = 5
        if rng.random() < float(preset["same_day_non_five_star_prob"]):
            star = rng.choice([3, 4])
        out.append(
            _base_record(
                rng=rng,
                product_id=target_product,
                reviewer_id=reviewer_id,
                star_rating=star,
                verified_purchase=vf,
                headline=headline,
                body=body + ".",
                review_ts=ts,
            )
        )
    return out


def _emit_fraud_slow_ring(
    rng: random.Random,
    start: datetime,
    span_days: int,
    preset: dict[str, float | int],
) -> list[dict]:
    """Many posts same reviewer unverified 5-star — triggers reviewer heuristic without same-day spam."""
    n = int(preset["slow_ring_reviews"])
    reviewer_id = f"R{rng.randint(91000, 91999):05d}"
    products = [f"P{rng.randint(2200, 2299):04d}" for _ in range(3)]
    templates = FRAUD_PARAPHRASES.copy()
    rng.shuffle(templates)
    out = []
    for i in range(n):
        day_off = rng.randint(0, max(1, span_days // 2))
        ts = start + timedelta(days=day_off, hours=rng.randint(0, 20))
        raw_body = templates[i % len(templates)]
        body = _inject_typo(raw_body, rng, float(preset["noise_typo_prob"]) * 0.45)
        headline = raw_body.split(".")[0][:58].strip().capitalize()
        out.append(
            _base_record(
                rng=rng,
                product_id=rng.choice(products),
                reviewer_id=reviewer_id,
                star_rating=5,
                verified_purchase=False,
                headline=headline,
                body=body + ".",
                review_ts=ts,
            )
        )
    return out


def _pick_fraud_kind(rng: random.Random, preset: dict[str, float | int]) -> str:
    w_dup = float(preset["fraud_weight_dup"])
    w_sd = float(preset["fraud_weight_same_day"])
    w_slow = float(preset["fraud_weight_slow_ring"])
    s = w_dup + w_sd + w_slow
    r = rng.random() * s
    if r < w_dup:
        return "dup"
    if r < w_dup + w_sd:
        return "same_day"
    return "slow"


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic Amazon-shaped JSONL.")
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--fraud-share", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--difficulty",
        choices=("easy", "medium", "hard"),
        default="medium",
        help="easy≈legacy separation; medium/hard add overlap, noise, subtler fraud.",
    )
    parser.add_argument("--out", type=Path, default=RAW_REVIEWS)
    args = parser.parse_args()

    preset = _preset(args.difficulty)
    ensure_dirs()
    rng = random.Random(args.seed)
    start = datetime(2023, 1, 1)
    span = 730

    n_fraud = int(args.rows * args.fraud_share)
    hn_clusters = int(preset["hard_negative_clusters"])
    hn_each = int(preset["reviews_per_hard_neg_cluster"])
    budget_non_fraud = max(0, args.rows - n_fraud)
    hn_budget = min(hn_clusters * hn_each, max(0, budget_non_fraud - 120))
    tail = max(0, budget_non_fraud - hn_budget)
    n_echo = min(400, tail // 11)
    n_burst_groups = min(26, max(0, (tail - n_echo) // 36))
    n_burst_reviews = n_burst_groups * 4
    n_organic_plain = max(0, tail - n_echo - n_burst_reviews)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    written = 0
    hn_written = 0
    with args.out.open("w") as f:
        for _ in range(n_organic_plain):
            f.write(json.dumps(_emit_organic(rng, start, span, preset)) + "\n")
            written += 1

        for _ in range(n_echo):
            pid = rng.choice(POPULAR_PRODUCT_IDS)
            f.write(
                json.dumps(
                    _emit_organic_fixed_product(rng, start, span, preset, pid)
                )
                + "\n"
            )
            written += 1

        for _ in range(n_burst_groups):
            for rec in _emit_clean_same_day_burst(rng, start, span, preset):
                f.write(json.dumps(rec) + "\n")
                written += 1

        # Hard-negative clusters: borderline benign reviewers (subtle fraud realism).
        for c in range(hn_clusters):
            if hn_written >= hn_budget:
                break
            rid = f"R{71000 + c:05d}"
            pid = f"P{31500 + (c % 40):04d}"
            for _ in range(hn_each):
                if hn_written >= hn_budget:
                    break
                f.write(
                    json.dumps(
                        _emit_hard_negative_organic(rng, start, span, preset, rid, pid)
                    )
                    + "\n"
                )
                written += 1
                hn_written += 1

        remaining = n_fraud
        while remaining > 0:
            kind = _pick_fraud_kind(rng, preset)
            if kind == "dup":
                burst = min(
                    remaining,
                    rng.randint(int(preset["dup_burst_min"]), int(preset["dup_burst_max"])),
                )
                for rec in _emit_fraud_duplicate_burst(rng, start, span, burst, preset):
                    f.write(json.dumps(rec) + "\n")
                    written += 1
                remaining -= burst
            elif kind == "same_day":
                burst = min(
                    remaining,
                    rng.randint(int(preset["same_day_burst_min"]), int(preset["same_day_burst_max"])),
                )
                for rec in _emit_fraud_same_day_paraphrase(rng, start, span, burst, preset):
                    f.write(json.dumps(rec) + "\n")
                    written += 1
                remaining -= burst
            else:
                chunk = _emit_fraud_slow_ring(rng, start, span, preset)
                take = min(remaining, len(chunk))
                for rec in chunk[:take]:
                    f.write(json.dumps(rec) + "\n")
                    written += 1
                remaining -= take

    print(
        f"Wrote {written:,} reviews to {args.out} "
        f"(fraud_quota={n_fraud:,}, hn_injected={hn_written}, "
        f"difficulty={args.difficulty}) in {time.time() - t0:.1f}s"
    )


if __name__ == "__main__":
    main()
