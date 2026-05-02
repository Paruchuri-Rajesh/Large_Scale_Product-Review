"""Rule-based fraud explanations for POST /predict (no LLM, no external APIs).

Uses only fields available at serving time: model outputs, request fields, and
numeric features from _enrich_for_serving. Wording avoids certainty claims.
"""
from __future__ import annotations

import math
from typing import Any, Literal, Optional

RiskLevel = Literal["low", "medium", "high"]

# Short phrases that often co-occur with promotional / spam-like reviews (heuristic only).
_PROMO_SUBSTRINGS = (
    "must buy",
    "best ever",
    "changed my life",
    "highly recommend",
    "amazing product",
    "five star",
    "5 star",
    "100 ",
    "love this product",
    "exceeded my expectations",
    "buy now",
    "don't miss",
    "dont miss",
    "life changing",
)


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def _safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def build_fraud_explanation(
    *,
    fraud_proba: float,
    fraud_flag: int,
    star_rating: Optional[int],
    verified_purchase: bool,
    review_body: str,
    row_features: dict[str, Any],
) -> dict[str, Any]:
    """Return structured fraud_explanation for JSON responses."""
    reasons: list[str] = []
    text_lower = (review_body or "").lower()

    fp = _safe_float(fraud_proba)
    if fp is None:
        fp = 0.0

    star = _safe_int(star_rating)
    if star is None:
        star = _safe_int(row_features.get("star_rating"))
    if star is None:
        star = 3

    vp_int = _safe_int(row_features.get("verified_purchase_int"))
    if vp_int is None:
        vp_int = 1 if verified_purchase else 0

    body_len = _safe_int(row_features.get("body_len")) or 0
    body_wc = _safe_int(row_features.get("body_word_count")) or 0
    exclam = _safe_int(row_features.get("exclam_count")) or 0
    helpful = _safe_int(row_features.get("helpful_votes")) or 0
    total_v = _safe_int(row_features.get("total_votes")) or 0

    promo_hit = any(s in text_lower for s in _PROMO_SUBSTRINGS)
    # Extra light cue: many exclamation marks already counted; all-caps ratio optional
    shout_ratio = 0.0
    if review_body and len(review_body) > 0:
        letters = sum(1 for c in review_body if c.isalpha())
        if letters > 0:
            shout_ratio = sum(1 for c in review_body if c.isupper() and c.isalpha()) / letters

    if fp >= 0.65:
        reasons.append(
            f"The fraud probability is relatively high (about {fp:.2f}), which pulls the risk up."
        )
    elif fp >= 0.45:
        reasons.append(
            f"The fraud probability is moderate (about {fp:.2f}), so this sits in a borderline band."
        )

    if fraud_flag and fp >= 0.5:
        reasons.append(
            "The score is elevated above the default 0.5 decision threshold used for the fraud flag."
        )

    if star >= 5 and promo_hit:
        reasons.append(
            "A top star rating appears together with emphatic or promotional wording in the text."
        )
    elif star >= 5 and exclam >= 2:
        reasons.append(
            "A five-star rating appears with multiple exclamation marks, adding emphasis."
        )

    if vp_int == 0:
        reasons.append(
            "Verified purchase is false in the request; when present, that is a weaker trust signal than verified buys."
        )

    if exclam >= 3:
        reasons.append("Several exclamation marks may indicate exaggerated tone (a weak cue only).")

    if body_wc > 0 and body_wc < 10:
        reasons.append(
            "The review text is very short; very brief posts sometimes overlap with low-effort patterns."
        )
    if body_wc > 120:
        reasons.append(
            "The review is long relative to a typical one-liner; length alone is not proof of fraud."
        )

    if total_v > 0 and helpful / max(total_v, 1) < 0.15:
        reasons.append(
            "Helpful votes are low compared to total votes on this row, which can reflect weak community traction."
        )

    if shout_ratio > 0.35 and len(review_body) > 20:
        reasons.append(
            "A large share of letters are uppercase, which can co-occur with shouty or spam-like text."
        )

    # Honest note about cold-start defaults (always true for single-review API path).
    reasons.append(
        "Reviewer and product aggregates use neutral defaults for cold-start scoring; history is not known for one-off requests."
    )

    # Derive risk_level
    signal_weight = 0
    if fp >= 0.7:
        signal_weight += 3
    elif fp >= 0.45:
        signal_weight += 1
    if fraud_flag:
        signal_weight += 2
    if promo_hit and star >= 5:
        signal_weight += 2
    if vp_int == 0:
        signal_weight += 1
    if exclam >= 3:
        signal_weight += 1

    if fp < 0.32 and not fraud_flag and signal_weight <= 2:
        risk: RiskLevel = "low"
    elif fp >= 0.72 or (fraud_flag and fp >= 0.55) or signal_weight >= 5:
        risk = "high"
    else:
        risk = "medium"

    # Summary (one cautious sentence)
    if risk == "high":
        summary = (
            "The model flagged this as suspicious mainly because the fraud score is elevated "
            "and several available cues align with patterns that often raise the probability."
        )
    elif risk == "medium":
        summary = (
            "This looks borderline because the fraud probability is in a middling range "
            "with mixed text and rating signals in the inputs we have."
        )
    else:
        summary = (
            "The fraud probability stays relatively low for this review given the available "
            "text, rating, and trust fields."
        )

    feature_signals = {
        "fraud_proba": round(fp, 6),
        "fraud_flag": int(fraud_flag),
        "star_rating": star,
        "verified_purchase_int": vp_int,
        "body_word_count": body_wc,
        "body_len": body_len,
        "exclam_count": exclam,
        "promotional_wording_heuristic": bool(promo_hit),
        "neutral_history_defaults": True,
    }

    return {
        "summary": summary,
        "reasons": reasons[:10],
        "risk_level": risk,
        "feature_signals": feature_signals,
    }


def safe_fraud_explanation(**kwargs: Any) -> Optional[dict[str, Any]]:
    """Never raises; returns None on failure so /predict stays robust."""
    try:
        return build_fraud_explanation(**kwargs)
    except Exception:
        return None
