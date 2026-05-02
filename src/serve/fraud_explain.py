"""Fraud explanations: rule-based fallback + optional LLM explanation via Ollama."""
from __future__ import annotations
import math
from typing import Any, Literal, Optional

RiskLevel = Literal["low", "medium", "high"]

_PROMO_SUBSTRINGS = (
    "must buy", "best ever", "changed my life", "highly recommend",
    "amazing product", "five star", "5 star", "love this product",
    "exceeded my expectations", "buy now", "don't miss", "life changing",
)

def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        return None if math.isnan(v) else v
    except (TypeError, ValueError):
        return None

def _safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None

def _llm_explanation(
    fraud_proba: float,
    fraud_flag: int,
    star_rating: int,
    verified_purchase: bool,
    review_body: str,
    row_features: dict,
) -> Optional[str]:
    """Call Ollama llama3.2 for a human-readable fraud explanation. Returns None on failure."""
    try:
        from ollama import Client
        client = Client()

        body_wc = _safe_int(row_features.get("body_word_count")) or 0
        exclam  = _safe_int(row_features.get("exclam_count")) or 0
        dup     = _safe_int(row_features.get("dup_in_product")) or 1
        rev_cnt = _safe_int(row_features.get("reviewer_review_count")) or 1
        pct5    = _safe_float(row_features.get("reviewer_pct_5star")) or 0.0
        same_day= _safe_int(row_features.get("reviewer_reviews_same_day")) or 1

        prompt = f"""You are a fraud analyst reviewing an Amazon product review.
Given the signals below, write ONE concise paragraph (3-4 sentences) explaining
why this review is or is not suspicious. Be specific about which signals matter most.
Do not use bullet points. Do not repeat the numbers back verbatim — interpret them.

Signals:
- Review text: "{review_body[:300]}"
- Star rating: {star_rating}/5
- Verified purchase: {verified_purchase}
- Fraud probability from ML model: {fraud_proba:.2%}
- Fraud flagged (threshold 0.5): {"YES" if fraud_flag else "NO"}
- Word count: {body_wc}
- Exclamation marks: {exclam}
- Duplicate reviews for same product: {dup}
- Reviewer total reviews: {rev_cnt}
- Reviewer % five-star reviews: {pct5:.0%}
- Reviewer reviews same day: {same_day}

Write the explanation now:"""

        response = client.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3, "num_predict": 200},
        )
        return response.message.content.strip()
    except Exception:
        return None


def build_fraud_explanation(
    *,
    fraud_proba: float,
    fraud_flag: int,
    star_rating: Optional[int],
    verified_purchase: bool,
    review_body: str,
    row_features: dict[str, Any],
    use_llm: bool = True,
) -> dict[str, Any]:
    fp   = _safe_float(fraud_proba) or 0.0
    star = _safe_int(star_rating) or _safe_int(row_features.get("star_rating")) or 3
    vp   = _safe_int(row_features.get("verified_purchase_int"))
    vp   = vp if vp is not None else (1 if verified_purchase else 0)
    body_wc = _safe_int(row_features.get("body_word_count")) or 0
    exclam  = _safe_int(row_features.get("exclam_count")) or 0
    helpful = _safe_int(row_features.get("helpful_votes")) or 0
    total_v = _safe_int(row_features.get("total_votes")) or 0
    text_lower = (review_body or "").lower()
    promo_hit  = any(s in text_lower for s in _PROMO_SUBSTRINGS)

    # Risk level
    weight = 0
    if fp >= 0.7:  weight += 3
    elif fp >= 0.45: weight += 1
    if fraud_flag: weight += 2
    if promo_hit and star >= 5: weight += 2
    if vp == 0: weight += 1
    if exclam >= 3: weight += 1

    if fp < 0.32 and not fraud_flag and weight <= 2:
        risk: RiskLevel = "low"
    elif fp >= 0.72 or (fraud_flag and fp >= 0.55) or weight >= 5:
        risk = "high"
    else:
        risk = "medium"

    # Rule-based summary fallback
    if risk == "high":
        summary = "The model flagged this as suspicious — fraud score is elevated with several corroborating signals."
    elif risk == "medium":
        summary = "Borderline case — fraud probability is in a middling range with mixed signals."
    else:
        summary = "Fraud probability is low given the available text, rating, and trust fields."

    # LLM explanation (best effort)
    llm_text = None
    if use_llm:
        llm_text = _llm_explanation(fp, fraud_flag, star, verified_purchase, review_body, row_features)

    return {
        "summary": llm_text if llm_text else summary,
        "llm_generated": llm_text is not None,
        "risk_level": risk,
        "feature_signals": {
            "fraud_proba": round(fp, 6),
            "fraud_flag": int(fraud_flag),
            "star_rating": star,
            "verified_purchase_int": vp,
            "body_word_count": body_wc,
            "exclam_count": exclam,
            "promotional_wording_heuristic": bool(promo_hit),
        },
    }


def safe_fraud_explanation(**kwargs: Any) -> Optional[dict[str, Any]]:
    try:
        return build_fraud_explanation(**kwargs)
    except Exception:
        return None
