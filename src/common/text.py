"""Lightweight text utilities shared by training and serving paths."""
from __future__ import annotations

import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<[^>]+>")
_NON_ALPHA_RE = re.compile(r"[^a-z0-9\s']")
_WS_RE = re.compile(r"\s+")


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    t = text.lower()
    t = _URL_RE.sub(" ", t)
    t = _HTML_RE.sub(" ", t)
    t = _NON_ALPHA_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


def sentiment_label_from_rating(rating: int | float | None) -> int:
    """0=negative, 1=neutral, 2=positive."""
    if rating is None:
        return 1
    r = int(rating)
    if r <= 2:
        return 0
    if r == 3:
        return 1
    return 2

import pandas as pd

def enrich_for_serving(pdf: pd.DataFrame) -> pd.DataFrame:
    """Compute per-row features the fraud model expects. Used by app.py and score_stream.py."""
    out = pdf.copy()
    body = out.get("review_body", pd.Series([], dtype=str)).fillna("").astype(str)
    out["review_body_clean"] = body.apply(clean_text)
    out["body_len"] = out["review_body_clean"].str.len().astype(int)
    out["body_word_count"] = out["review_body_clean"].str.split().map(lambda x: len([w for w in x if w])).astype(int)
    out["exclam_count"] = body.str.count("!").astype(int)
    out["verified_purchase_int"] = out.get("verified_purchase", pd.Series([False]*len(out))).fillna(False).astype(int)
    out["star_rating"] = pd.to_numeric(out.get("star_rating", pd.Series([3]*len(out))), errors="coerce").fillna(3).astype(int)
    out["helpful_votes"] = pd.to_numeric(out.get("helpful_votes", pd.Series([0]*len(out))), errors="coerce").fillna(0).astype(int)
    out["total_votes"] = pd.to_numeric(out.get("total_votes", pd.Series([0]*len(out))), errors="coerce").fillna(0).astype(int)
    out["reviewer_review_count"] = 1
    out["reviewer_avg_rating"] = out["star_rating"].astype(float)
    out["reviewer_pct_5star"] = (out["star_rating"] == 5).astype(float)
    out["reviewer_distinct_products"] = 1
    out["reviewer_reviews_same_day"] = 1
    out["reviewer_verified_share"] = out["verified_purchase_int"].astype(float)
    out["product_review_count"] = 1
    out["product_avg_rating"] = out["star_rating"].astype(float)
    out["product_pct_5star"] = (out["star_rating"] == 5).astype(float)
    out["dup_in_product"] = 1
    return out
