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
import sys
from pathlib import Path
from typing import Any

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
    FRAUD_MODEL_PATH,
    META_PATH,
    PRODUCT_AGG_PARQUET,
    REVIEWER_AGG_PARQUET,
    SENTIMENT_LABELS,
    SENTIMENT_MODEL_PATH,
    STREAM_OUT_DIR,
)
from src.stream.score_stream import _enrich_for_serving  # noqa: E402
from src.train.train import NUMERIC_FRAUD_FEATURES  # noqa: E402

app = FastAPI(title="Amazon Reviews — Sentiment & Fraud", version="1.0.0")

_HERE = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=_HERE / "static"), name="static")
templates = Jinja2Templates(directory=_HERE / "templates")


class Review(BaseModel):
    review_body: str = Field(..., min_length=1)
    review_headline: str | None = ""
    star_rating: int | None = Field(default=None, ge=1, le=5)
    helpful_votes: int | None = 0
    total_votes: int | None = 0
    verified_purchase: bool | None = True
    product_id: str | None = None
    reviewer_id: str | None = None


class BatchRequest(BaseModel):
    reviews: list[Review]


# ---- model load (lazy, once) -----------------------------------------------

_state: dict[str, Any] = {"sentiment": None, "fraud": None, "meta": None}


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
        out.append(
            {
                "review_body": r.review_body,
                "sentiment_label": int(sent_label[i]),
                "sentiment": SENTIMENT_LABELS[int(sent_label[i])],
                "fraud_proba": float(fraud_proba[i]),
                "fraud_flag": int(fraud_proba[i] >= 0.5),
                "product_id": r.product_id,
                "reviewer_id": r.reviewer_id,
            }
        )
    return out


# ---- routes ---------------------------------------------------------------

@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/metadata")
def metadata() -> dict:
    _load_models()
    return _state["meta"] or {"warning": "no meta.json on disk"}


@app.post("/predict")
def predict(review: Review) -> dict:
    return _score([review])[0]


@app.post("/predict/batch")
def predict_batch(req: BatchRequest) -> dict:
    if not req.reviews:
        raise HTTPException(400, "reviews must not be empty")
    if len(req.reviews) > 1000:
        raise HTTPException(400, "max 1000 reviews per batch")
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
