"""Drip a JSONL file into the streaming input directory, one chunk at a time.

Useful as a tiny producer: split an existing reviews.jsonl into N chunked
files and drop them every few seconds. Spark Structured Streaming picks them
up via its file source.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.common.config import RAW_REVIEWS, STREAM_IN_DIR, ensure_dirs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=RAW_REVIEWS)
    parser.add_argument("--out-dir", type=Path, default=STREAM_IN_DIR)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--n-batches", type=int, default=10)
    parser.add_argument("--sleep", type=float, default=2.0)
    args = parser.parse_args()

    ensure_dirs()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.source.exists():
        raise SystemExit(f"source {args.source} does not exist; run ingest first")

    with args.source.open() as fh:
        lines = fh.readlines()

    if len(lines) < args.batch_size * args.n_batches:
        # Cycle if we run out.
        lines = (lines * ((args.batch_size * args.n_batches) // len(lines) + 1))

    cursor = 0
    for i in range(args.n_batches):
        chunk = lines[cursor : cursor + args.batch_size]
        cursor += args.batch_size
        out = args.out_dir / f"batch-{int(time.time())}-{uuid.uuid4().hex[:8]}.jsonl"
        out.write_text("".join(chunk))
        print(f"[feed] wrote {out.name} ({len(chunk)} lines)")
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
