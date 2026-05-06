"""Convert McAuley-Lab Amazon-Reviews-2023 JSONL into the project schema.

Source schema (one record per line):
    rating (float), title (str), text (str), images (list), asin (str),
    parent_asin (str), user_id (str), timestamp (int, ms),
    helpful_vote (int), verified_purchase (bool)

Target schema (see src/common/schema.py): review_id, product_id, reviewer_id,
star_rating, helpful_votes, total_votes, verified_purchase, review_headline,
review_body, review_date (YYYY-MM-DD), product_category, event_ts (epoch s).

Streams line-by-line so the ~9 GB Cell_Phones_and_Accessories.jsonl doesn't
need to fit in memory.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.common.config import RAW_DIR, ensure_dirs  # noqa: E402


def _category_from_filename(path: Path) -> str:
    return path.stem.replace("_", " ")


def _convert(record: dict, category: str) -> dict | None:
    rating = record.get("rating")
    text = record.get("text") or ""
    title = record.get("title") or ""
    asin = record.get("asin")
    user_id = record.get("user_id")
    ts_ms = record.get("timestamp")

    if rating is None or asin is None or user_id is None or ts_ms is None:
        return None
    if not text and not title:
        return None

    ts_s = int(ts_ms) // 1000
    review_date = datetime.fromtimestamp(ts_s, tz=timezone.utc).strftime("%Y-%m-%d")
    helpful = int(record.get("helpful_vote") or 0)

    return {
        "review_id": str(uuid.uuid4()),
        "product_id": asin,
        "reviewer_id": user_id,
        "star_rating": int(round(float(rating))),
        "helpful_votes": helpful,
        "total_votes": helpful,
        "verified_purchase": bool(record.get("verified_purchase", False)),
        "review_headline": title.strip()[:200],
        "review_body": text.strip(),
        "review_date": review_date,
        "product_category": category,
        "event_ts": ts_s,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Path to a McAuley-Lab Amazon-Reviews-2023 *.jsonl file.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=RAW_DIR / "reviews_real.jsonl",
        help="Where to write project-schema JSONL.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If > 0, stop after this many converted rows (for sampling).",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Override product_category (defaults to the source filename stem).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500_000,
        help="Print a progress line every N input lines.",
    )
    args = parser.parse_args()

    if not args.src.exists():
        sys.exit(f"source file not found: {args.src}")

    ensure_dirs()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    category = args.category or _category_from_filename(args.src)
    t0 = time.time()
    read = written = skipped = 0

    with args.src.open("r", encoding="utf-8") as fin, args.out.open("w", encoding="utf-8") as fout:
        for line in fin:
            read += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            converted = _convert(record, category)
            if converted is None:
                skipped += 1
            else:
                fout.write(json.dumps(converted) + "\n")
                written += 1
                if args.limit and written >= args.limit:
                    break

            if args.progress_every and read % args.progress_every == 0:
                elapsed = time.time() - t0
                print(
                    f"  read={read:,} written={written:,} skipped={skipped:,} "
                    f"({read / elapsed:,.0f} rec/s)",
                    flush=True,
                )

    elapsed = time.time() - t0
    print(
        f"Done. read={read:,} written={written:,} skipped={skipped:,} "
        f"in {elapsed:.1f}s -> {args.out}"
    )


if __name__ == "__main__":
    main()
