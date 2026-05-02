"""MCP server exposing sentiment & fraud detection as Claude-callable tools.

Tools:
  - predict_review      : score a single review for sentiment + fraud
  - get_fraud_reviewers : return top suspicious reviewers from ETL aggregates
  - get_top_products    : return top products by review count

Run:
    python -m src.serve.mcp_server

Then add to Claude Desktop config (~/Library/Application Support/Claude/claude_desktop_config.json):
{
  "mcpServers": {
    "amazon-reviews": {
      "command": "/Users/ritika.mac/Large_Scale_Product-Review/venv/bin/python",
      "args": ["-m", "src.serve.mcp_server"],
      "cwd": "/Users/ritika.mac/Large_Scale_Product-Review"
    }
  }
}
"""
from __future__ import annotations
import sys
from pathlib import Path

import joblib
import pandas as pd
from fastmcp import FastMCP

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.common.config import (  # noqa: E402
    FRAUD_MODEL_PATH,
    REVIEWER_AGG_PARQUET,
    PRODUCT_AGG_PARQUET,
    SENTIMENT_LABELS,
    SENTIMENT_MODEL_PATH,
)
from src.common.text import clean_text  # noqa: E402
from src.train.features import get_fraud_numeric_features  # noqa: E402

NUMERIC_FRAUD_FEATURES = get_fraud_numeric_features()

mcp = FastMCP("Amazon Reviews — Sentiment & Fraud Detector")

_models: dict = {"sentiment": None, "fraud": None}


def _load() -> None:
    if _models["sentiment"] is None:
        _models["sentiment"] = joblib.load(SENTIMENT_MODEL_PATH)
    if _models["fraud"] is None:
        _models["fraud"] = joblib.load(FRAUD_MODEL_PATH)


def _enrich(review_body: str, star_rating: int, verified_purchase: bool,
            helpful_votes: int, total_votes: int) -> pd.DataFrame:
    """Build the feature DataFrame the fraud model expects."""
    body_clean = clean_text(review_body)
    row = {
        "review_body_clean": body_clean,
        "body_len": len(body_clean),
        "body_word_count": len([w for w in body_clean.split() if w]),
        "exclam_count": review_body.count("!"),
        "verified_purchase_int": int(verified_purchase),
        "star_rating": star_rating,
        "helpful_votes": helpful_votes,
        "total_votes": total_votes,
        # Behavioral aggregates — neutral defaults for a single unseen review
        "reviewer_review_count": 1,
        "reviewer_avg_rating": float(star_rating),
        "reviewer_pct_5star": float(star_rating == 5),
        "reviewer_distinct_products": 1,
        "reviewer_reviews_same_day": 1,
        "reviewer_verified_share": float(verified_purchase),
        "product_review_count": 1,
        "product_avg_rating": float(star_rating),
        "product_pct_5star": float(star_rating == 5),
        "dup_in_product": 1,
    }
    return pd.DataFrame([row])


@mcp.tool()
def predict_review(
    review_body: str,
    star_rating: int = 3,
    verified_purchase: bool = True,
    helpful_votes: int = 0,
    total_votes: int = 0,
) -> dict:
    """Score a product review for sentiment (positive/neutral/negative) and fraud probability.

    Args:
        review_body: The text of the review.
        star_rating: Star rating 1-5 (default 3).
        verified_purchase: Whether this is a verified purchase (default True).
        helpful_votes: Number of helpful votes (default 0).
        total_votes: Total votes (default 0).
    """
    _load()
    feats = _enrich(review_body, star_rating, verified_purchase, helpful_votes, total_votes)
    sent_label = int(_models["sentiment"].predict(feats["review_body_clean"])[0])
    feat_cols = ["review_body_clean"] + NUMERIC_FRAUD_FEATURES
    fraud_proba = float(_models["fraud"].predict_proba(feats[feat_cols])[0, 1])
    fraud_flag = int(fraud_proba >= 0.5)

    return {
        "sentiment": SENTIMENT_LABELS[sent_label],
        "fraud_proba": round(fraud_proba, 4),
        "fraud_flag": fraud_flag,
        "explanation": (
            f"Review is {SENTIMENT_LABELS[sent_label]}. "
            f"Fraud probability is {fraud_proba:.1%} — "
            f"{'FLAGGED as suspicious.' if fraud_flag else 'looks clean.'}"
        ),
    }


@mcp.tool()
def get_fraud_reviewers(limit: int = 10) -> list[dict]:
    """Return the top suspicious reviewers ranked by fraud rate from the ETL aggregates.

    Args:
        limit: Number of reviewers to return (default 10, max 50).
    """
    if not REVIEWER_AGG_PARQUET.exists():
        return [{"error": "Run batch ETL first (make etl)"}]
    limit = min(limit, 50)
    df = pd.read_parquet(REVIEWER_AGG_PARQUET)
    df = df.sort_values(["fraud_rate", "review_count"], ascending=[False, False]).head(limit)
    cols = [c for c in ["reviewer_id", "fraud_rate", "review_count", "reviewer_avg_rating", "reviewer_pct_5star"] if c in df.columns]
    return df[cols].to_dict(orient="records")


@mcp.tool()
def get_top_products(limit: int = 10, sort_by: str = "review_count") -> list[dict]:
    """Return top products from the ETL product aggregates.

    Args:
        limit: Number of products to return (default 10, max 50).
        sort_by: Column to sort by — 'review_count', 'fraud_rate', or 'product_avg_rating'.
    """
    if not PRODUCT_AGG_PARQUET.exists():
        return [{"error": "Run batch ETL first (make etl)"}]
    valid_sorts = {"review_count", "fraud_rate", "product_avg_rating"}
    if sort_by not in valid_sorts:
        sort_by = "review_count"
    limit = min(limit, 50)
    df = pd.read_parquet(PRODUCT_AGG_PARQUET)
    df = df.sort_values(sort_by, ascending=False).head(limit)
    cols = [c for c in ["product_id", "review_count", "product_avg_rating", "product_pct_5star", "fraud_rate"] if c in df.columns]
    return df[cols].to_dict(orient="records")


if __name__ == "__main__":
    mcp.run()
