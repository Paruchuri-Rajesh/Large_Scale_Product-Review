"""Smoke tests that exercise the same code paths used in production.

These do not start Spark — Spark is covered by the actual ETL/streaming
runs. We focus on the bits that fail silently without tests: text cleaning,
the per-row feature enrichment used for online scoring, and the FastAPI
endpoints (loaded models off disk, in-process via TestClient).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.common.text import clean_text, sentiment_label_from_rating
from src.stream.score_stream import _enrich_for_serving
# Assert enrichment produces every column the persisted fraud model expects.
from src.train.features import get_fraud_numeric_features


def test_fraud_explanation_structure_no_models() -> None:
    """Rule-based explanation uses only serving-time fields; must always shape."""
    from src.serve.fraud_explain import build_fraud_explanation

    row = {
        "star_rating": 5,
        "verified_purchase_int": 1,
        "body_len": 80,
        "body_word_count": 14,
        "exclam_count": 0,
        "helpful_votes": 0,
        "total_votes": 0,
    }
    ex = build_fraud_explanation(
        fraud_proba=0.22,
        fraud_flag=0,
        star_rating=5,
        verified_purchase=True,
        review_body="Great product, works as expected.",
        row_features=row,
    )
    assert ex["risk_level"] in ("low", "medium", "high")
    assert isinstance(ex["summary"], str) and ex["summary"]
    assert isinstance(ex["reasons"], list)
    assert "feature_signals" in ex and isinstance(ex["feature_signals"], dict)


def test_clean_text_strips_html_and_urls() -> None:
    assert (
        clean_text("Check <b>this</b> https://example.com !!!").strip()
        == "check this"
    )


@pytest.mark.parametrize(
    "rating,expected",
    [(1, 0), (2, 0), (3, 1), (4, 2), (5, 2), (None, 1)],
)
def test_sentiment_label_from_rating(rating, expected) -> None:
    assert sentiment_label_from_rating(rating) == expected


def test_enrich_for_serving_has_all_numeric_features() -> None:
    pdf = pd.DataFrame(
        [
            {
                "review_body": "amazing product, love it!",
                "review_headline": "great",
                "star_rating": 5,
                "helpful_votes": 1,
                "total_votes": 2,
                "verified_purchase": True,
            }
        ]
    )
    out = _enrich_for_serving(pdf)
    for col in get_fraud_numeric_features():
        assert col in out.columns, f"missing serving feature: {col}"
    assert out.loc[0, "review_body_clean"].startswith("amazing product")


def test_predict_endpoint_classifies_signal() -> None:
    from fastapi.testclient import TestClient

    from src.common.config import FRAUD_MODEL_PATH, SENTIMENT_MODEL_PATH

    if not SENTIMENT_MODEL_PATH.exists() or not FRAUD_MODEL_PATH.exists():
        pytest.skip("models not trained yet; run `make all`")

    from src.serve.app import app

    client = TestClient(app)
    r = client.post(
        "/predict",
        json={
            "review_body": "absolutely terrible quality, broke after one day, do not buy",
            "star_rating": 1,
            "verified_purchase": True,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["sentiment"] == "negative"
    assert "fraud_explanation" in body
    fe = body["fraud_explanation"]
    assert fe is None or (
        isinstance(fe, dict)
        and "summary" in fe
        and "reasons" in fe
        and fe.get("risk_level") in ("low", "medium", "high")
    )

    r = client.post(
        "/predict",
        json={
            "review_body": "exceeded my expectations, fantastic build quality, highly recommend",
            "star_rating": 5,
            "verified_purchase": True,
        },
    )
    assert r.status_code == 200
    pos = r.json()
    assert pos["sentiment"] == "positive"
    assert "fraud_explanation" in pos


def test_healthz() -> None:
    from fastapi.testclient import TestClient

    from src.serve.app import app

    client = TestClient(app)
    assert client.get("/healthz").json() == {"status": "ok"}
