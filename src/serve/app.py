"""FastAPI service: predict sentiment + fraud, expose lightweight dashboard.

Endpoints:
  GET  /                       -> dashboard (HTML)
  GET  /healthz                -> {"status": "ok"}
  GET  /metadata               -> training metrics + feature columns
  POST /predict                -> single-review prediction
  POST /predict/batch          -> list-of-reviews prediction
  GET  /aggregates/products    -> top-N products by review_count (from ETL)
  GET  /aggregates/fraud-reviewers -> top-N suspicious reviewers (from ETL)
  GET  /stream/recent          -> latest scored stream rows on disk
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
import time
import uuid
from datetime import date
from pathlib import Path
from typing import Any, Optional  # Optional keeps Pydantic happy on Python 3.9 (no `str | None` runtime eval).

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.requests import Request

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.common.config import (  # noqa: E402
    BASELINE_REPORT_PATH,
    DRIFT_REPORT_PATH,
    FRAUD_MODEL_PATH,
    META_PATH,
    PRODUCT_AGG_PARQUET,
    REVIEWER_AGG_PARQUET,
    SENTIMENT_LABELS,
    SENTIMENT_MODEL_PATH,
    STREAM_OUT_DIR,
    THRESHOLD_REPORT_PATH,
    THRESHOLDS_PATH,
)
from src.serve.fraud_explain import safe_fraud_explanation  # noqa: E402
from src.common.text import enrich_for_serving as _enrich_for_serving  # noqa: E402
# Fraud joblib pipeline columns: cleaned text + numeric behavioral features from training.
from src.train.features import get_fraud_numeric_features  # noqa: E402

NUMERIC_FRAUD_FEATURES = get_fraud_numeric_features()

# Optional Kafka mirror: when PUBLISH_PREDICT_TO_KAFKA=1, every /predict call
# also fire-and-forget publishes its review to the 'reviews' topic so it flows
# through the Kafka → Spark Streaming → /stream/recent path. Failures here
# never break the sync /predict response.
_kafka_producer: Any = None


def _maybe_get_kafka_producer() -> Any:
    global _kafka_producer
    if os.environ.get("PUBLISH_PREDICT_TO_KAFKA") != "1":
        return None
    if _kafka_producer is not None:
        return _kafka_producer
    try:
        from kafka import KafkaProducer  # type: ignore
        _kafka_producer = KafkaProducer(
            bootstrap_servers=os.environ.get("KAFKA_BROKER", "localhost:9092"),
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks=1,
            linger_ms=0,
            request_timeout_ms=2000,
        )
    except Exception:
        _kafka_producer = None
    return _kafka_producer


def _publish_review_to_kafka(review: "Review") -> None:
    producer = _maybe_get_kafka_producer()
    if producer is None:
        return
    record = {
        "review_id": str(uuid.uuid4()),
        "product_id": review.product_id or f"BFORM{uuid.uuid4().hex[:6].upper()}",
        "reviewer_id": review.reviewer_id or f"R{uuid.uuid4().hex[:24].upper()}",
        "star_rating": review.star_rating or 5,
        "helpful_votes": review.helpful_votes or 0,
        "total_votes": review.total_votes or 0,
        "verified_purchase": bool(review.verified_purchase),
        "review_headline": review.review_headline or "form-submitted review",
        "review_body": review.review_body,
        "review_date": date.today().isoformat(),
        "product_category": "User Submitted",
        "event_ts": int(time.time()),
    }
    try:
        producer.send(os.environ.get("KAFKA_TOPIC", "reviews"), value=record)
    except Exception:
        pass


app = FastAPI(title="Amazon Reviews — Sentiment & Fraud", version="1.0.0")

_HERE = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=_HERE / "static"), name="static")
templates = Jinja2Templates(directory=_HERE / "templates")


class Review(BaseModel):
    review_body: str = Field(..., min_length=1)
    review_headline: Optional[str] = ""
    star_rating: Optional[int] = Field(default=None, ge=1, le=5)
    helpful_votes: Optional[int] = 0
    total_votes: Optional[int] = 0
    verified_purchase: Optional[bool] = True
    product_id: Optional[str] = None
    reviewer_id: Optional[str] = None


class BatchRequest(BaseModel):
    reviews: list[Review]


# ---- model load (lazy, once) -----------------------------------------------

_state: dict[str, Any] = {"sentiment": None, "fraud": None, "meta": None}


# First N rows of a CSV as JSON-serializable records (optional ML reports).
def _csv_preview_records(path: Path, max_rows: int = 5) -> Any | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        return df.head(max_rows).to_dict(orient="records")
    except Exception:
        return None


# Parse JSON artifact if present; never raise for optional dashboard fields.
def _json_optional(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _float_or_none(x: Any) -> float | None:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        if pd.isna(x):
            return None
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


# Full baseline CSV: best rows per task + series for dashboard charts (preview stays separate).
def _baseline_extras(path: Path) -> dict[str, Any]:
    empty: dict[str, Any] = {
        "best_sentiment_baseline": None,
        "best_fraud_baseline": None,
        "baseline_chart_series": {"sentiment": [], "fraud": []},
    }
    if not path.exists():
        return empty
    try:
        df = pd.read_csv(path)
    except Exception:
        return empty
    if df.empty or "task" not in df.columns:
        return empty
    tcol = df["task"].astype(str).str.strip().str.lower()

    def sentiment_block(sdf: pd.DataFrame) -> None:
        if sdf.empty or "f1_macro" not in sdf.columns:
            return
        fm = pd.to_numeric(sdf["f1_macro"], errors="coerce")
        rows_chart = []
        for _, r in sdf.iterrows():
            rows_chart.append(
                {
                    "model_name": str(r.get("model_name", "") or ""),
                    "f1_macro": _float_or_none(r.get("f1_macro")),
                }
            )
        empty["baseline_chart_series"]["sentiment"] = rows_chart
        if not fm.notna().any():
            return
        row = sdf.loc[fm.idxmax()]
        empty["best_sentiment_baseline"] = {
            "model_name": str(row.get("model_name", "") or ""),
            "f1_macro": _float_or_none(row.get("f1_macro")),
            "f1_weighted": _float_or_none(row.get("f1_weighted")),
        }

    def fraud_block(fdf: pd.DataFrame) -> None:
        if fdf.empty or "f1" not in fdf.columns:
            return
        f1 = pd.to_numeric(fdf["f1"], errors="coerce")
        rows_chart = []
        for _, r in fdf.iterrows():
            rows_chart.append(
                {
                    "model_name": str(r.get("model_name", "") or ""),
                    "f1": _float_or_none(r.get("f1")),
                    "roc_auc": _float_or_none(r.get("roc_auc")),
                }
            )
        empty["baseline_chart_series"]["fraud"] = rows_chart
        if not f1.notna().any():
            return
        row = fdf.loc[f1.idxmax()]
        empty["best_fraud_baseline"] = {
            "model_name": str(row.get("model_name", "") or ""),
            "f1": _float_or_none(row.get("f1")),
            "precision": _float_or_none(row.get("precision")),
            "recall": _float_or_none(row.get("recall")),
            "roc_auc": _float_or_none(row.get("roc_auc")),
        }

    sentiment_block(df[tcol == "sentiment"])
    fraud_block(df[tcol == "fraud"])
    return empty


def _load_models() -> None:
    if _state["sentiment"] is None:
        if not SENTIMENT_MODEL_PATH.exists():
            raise HTTPException(503, f"sentiment model not found at {SENTIMENT_MODEL_PATH}")
        _state["sentiment"] = joblib.load(SENTIMENT_MODEL_PATH)
    if _state["fraud"] is None:
        if not FRAUD_MODEL_PATH.exists():
            raise HTTPException(503, f"fraud model not found at {FRAUD_MODEL_PATH}")
        _state["fraud"] = joblib.load(FRAUD_MODEL_PATH)
    if _state["meta"] is None and META_PATH.exists():
        _state["meta"] = json.loads(META_PATH.read_text())


def _score(reviews: list[Review]) -> list[dict]:
    _load_models()
    pdf = pd.DataFrame([r.model_dump() for r in reviews])
    feats = _enrich_for_serving(pdf)
    sent_label = _state["sentiment"].predict(feats["review_body_clean"]).astype(int)
    feat_cols = ["review_body_clean"] + NUMERIC_FRAUD_FEATURES
    fraud_proba = _state["fraud"].predict_proba(feats[feat_cols])[:, 1]
    out = []
    for i, r in enumerate(reviews):
        fp = float(fraud_proba[i])
        ff = int(fp >= 0.5)
        row_dict = feats.iloc[i].to_dict()
        explain = safe_fraud_explanation(
            fraud_proba=fp,
            fraud_flag=ff,
            star_rating=r.star_rating,
            verified_purchase=bool(r.verified_purchase),
            review_body=r.review_body,
            row_features=row_dict,
        )
        row_out: dict[str, Any] = {
            "review_body": r.review_body,
            "sentiment_label": int(sent_label[i]),
            "sentiment": SENTIMENT_LABELS[int(sent_label[i])],
            "fraud_proba": fp,
            "fraud_flag": ff,
            "product_id": r.product_id,
            "reviewer_id": r.reviewer_id,
        }
        if explain is not None:
            row_out["fraud_explanation"] = explain
        else:
            row_out["fraud_explanation"] = None
        out.append(row_out)
    return out


# ---- routes ---------------------------------------------------------------

@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/metadata")
def metadata() -> dict:
    _load_models()
    if _state["meta"]:
        out: dict[str, Any] = dict(_state["meta"])
    else:
        out = {"warning": "no meta.json on disk"}
    out["baseline_report_preview"] = _csv_preview_records(BASELINE_REPORT_PATH)
    bl_ex = _baseline_extras(BASELINE_REPORT_PATH)
    out["best_sentiment_baseline"] = bl_ex["best_sentiment_baseline"]
    out["best_fraud_baseline"] = bl_ex["best_fraud_baseline"]
    out["baseline_chart_series"] = bl_ex["baseline_chart_series"]
    out["threshold_report_preview"] = _csv_preview_records(THRESHOLD_REPORT_PATH)
    out["drift_summary"] = _json_optional(DRIFT_REPORT_PATH)
    out["selected_thresholds"] = _json_optional(THRESHOLDS_PATH)
    return out


@app.post("/predict")
def predict(review: Review) -> dict:
    _publish_review_to_kafka(review)
    return _score([review])[0]


@app.post("/predict/batch")
def predict_batch(req: BatchRequest) -> dict:
    if not req.reviews:
        raise HTTPException(400, "reviews must not be empty")
    if len(req.reviews) > 1000:
        raise HTTPException(400, "max 1000 reviews per batch")
    for r in req.reviews:
        _publish_review_to_kafka(r)
    return {"predictions": _score(req.reviews)}


def _read_parquet_safe(path: Path, limit: int) -> list[dict]:
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    return df.head(limit).to_dict(orient="records")


@app.get("/aggregates/products")
def aggregates_products(limit: int = 25, by: str = "review_count") -> JSONResponse:
    if not PRODUCT_AGG_PARQUET.exists():
        raise HTTPException(404, "run the batch ETL first")
    df = pd.read_parquet(PRODUCT_AGG_PARQUET)
    if by not in df.columns:
        raise HTTPException(400, f"unknown sort column {by}")
    df = df.sort_values(by, ascending=False).head(limit)
    return JSONResponse(df.to_dict(orient="records"))


@app.get("/aggregates/fraud-reviewers")
def aggregates_reviewers(limit: int = 25) -> JSONResponse:
    if not REVIEWER_AGG_PARQUET.exists():
        raise HTTPException(404, "run the batch ETL first")
    df = pd.read_parquet(REVIEWER_AGG_PARQUET)
    df = df.sort_values(["fraud_rate", "review_count"], ascending=[False, False]).head(limit)
    return JSONResponse(df.to_dict(orient="records"))


@app.get("/stream/recent")
def stream_recent(limit: int = 50) -> JSONResponse:
    pattern = str(STREAM_OUT_DIR / "scored" / "*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return JSONResponse([])
    rows: list[dict] = []
    for f in files[-5:]:  # last 5 batches
        with open(f) as fh:
            for line in fh:
                rows.append(json.loads(line))
    rows = rows[-limit:]
    return JSONResponse(rows)


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/audit/{product_id}")
async def audit_product(product_id: str) -> JSONResponse:
    """Run the agentic auditor for a product_id and return the report."""
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    try:
        from src.agents.review_auditor import run_auditor
        report = run_auditor(product_id)
        return JSONResponse({"product_id": product_id, "report": report})
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/audit/{product_id}")
async def audit_product(product_id: str) -> JSONResponse:
    """Run the agentic auditor for a product_id and return the report."""
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    try:
        from src.agents.review_auditor import run_auditor
        report = run_auditor(product_id)
        return JSONResponse({"product_id": product_id, "report": report})
    except Exception as e:
        raise HTTPException(500, str(e))
