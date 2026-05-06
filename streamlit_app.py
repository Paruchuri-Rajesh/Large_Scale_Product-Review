"""Streamlit dashboard for the Amazon Reviews sentiment + fraud pipeline.

Loads the trained joblib pipelines and the Spark-ETL parquet aggregates
directly (no FastAPI required). Run with:

    streamlit run streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from src.common.config import (  # noqa: E402
    FRAUD_MODEL_PATH,
    META_PATH,
    PRODUCT_AGG_PARQUET,
    RAW_REVIEWS,
    REVIEWER_AGG_PARQUET,
    SENTIMENT_LABELS,
    SENTIMENT_MODEL_PATH,
    TEST_PARQUET,
)
from src.common.text import enrich_for_serving  # noqa: E402
from src.train.features import get_fraud_numeric_features  # noqa: E402

st.set_page_config(
    page_title="Amazon Reviews — Sentiment & Fraud",
    page_icon=":mag:",
    layout="wide",
)

NUMERIC_FRAUD_FEATURES = get_fraud_numeric_features()


@st.cache_resource
def load_models():
    sentiment = joblib.load(SENTIMENT_MODEL_PATH)
    fraud = joblib.load(FRAUD_MODEL_PATH)
    meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    return sentiment, fraud, meta


@st.cache_data
def load_product_agg(top_n: int = 1000) -> pd.DataFrame:
    return pd.read_parquet(PRODUCT_AGG_PARQUET).head(top_n)


@st.cache_data
def load_reviewer_agg(top_n: int = 1000) -> pd.DataFrame:
    return pd.read_parquet(REVIEWER_AGG_PARQUET).head(top_n)


@st.cache_data
def load_test_sample(n: int = 50_000) -> pd.DataFrame:
    return pd.read_parquet(
        TEST_PARQUET,
        columns=[
            "star_rating",
            "body_len",
            "body_word_count",
            "sentiment_label",
            "fraud_label",
        ],
    ).sample(min(n, 1_000_000), random_state=42)


@st.cache_data
def load_raw_sample(n: int = 100, skip: int = 0) -> list[dict]:
    out: list[dict] = []
    if not RAW_REVIEWS.exists():
        return out
    with RAW_REVIEWS.open() as f:
        for i, line in enumerate(f):
            if i < skip:
                continue
            if len(out) >= n:
                break
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def score_review(sentiment, fraud, body: str, star: int, verified: bool) -> dict:
    pdf = pd.DataFrame(
        [
            {
                "review_body": body,
                "star_rating": star,
                "verified_purchase": verified,
                "helpful_votes": 0,
                "total_votes": 0,
            }
        ]
    )
    feats = enrich_for_serving(pdf)
    sent_idx = int(sentiment.predict(feats["review_body_clean"])[0])
    fcols = ["review_body_clean"] + NUMERIC_FRAUD_FEATURES
    fproba = float(fraud.predict_proba(feats[fcols])[0, 1])
    return {
        "sentiment_label": sent_idx,
        "sentiment": SENTIMENT_LABELS[sent_idx],
        "fraud_proba": fproba,
        "fraud_flag": int(fproba >= 0.5),
    }


# ---- App ----

sentiment_model, fraud_model, meta = load_models()

st.title("Amazon Reviews — Sentiment & Fraud")
st.caption(
    "Interactive dashboard backed by the Spark-ETL Parquet outputs and "
    "the persisted scikit-learn pipelines."
)

with st.sidebar:
    st.header("Pipeline status")
    if meta:
        n_train = meta.get("n_train", 0)
        n_test = meta.get("n_test", 0)
        st.metric("Train rows", f"{n_train:,}")
        st.metric("Test rows", f"{n_test:,}")
        st.divider()
        sm = meta.get("sentiment", {}) or {}
        fm = meta.get("fraud", {}) or {}
        st.subheader("Sentiment")
        st.metric("Macro F1", f"{sm.get('f1_macro', 0):.3f}")
        st.metric("Weighted F1", f"{sm.get('f1_weighted', 0):.3f}")
        st.subheader("Fraud")
        st.metric("ROC AUC", f"{fm.get('roc_auc') or 0:.3f}")
        st.metric("F1 @ 0.5", f"{fm.get('f1', 0):.3f}")
        st.metric("Precision / Recall", f"{fm.get('precision', 0):.2f} / {fm.get('recall', 0):.2f}")
        st.caption(f"Selected fraud model: `{fm.get('model_name', '?')}`")
    else:
        st.warning("No `models/meta.json` found — train the models first.")

t1, t2, t3, t4, t5 = st.tabs(
    ["Score a review", "Top products", "Suspicious reviewers", "Browse raw reviews", "Distributions"]
)

# Tab 1 — interactive predict
with t1:
    st.subheader("Score a single review")
    c1, c2 = st.columns([3, 1])
    with c1:
        body = st.text_area(
            "Review body",
            value="absolutely love this case, fits perfectly and looks great",
            height=140,
        )
    with c2:
        star = st.slider("Star rating", 1, 5, 5)
        verified = st.checkbox("Verified purchase", value=True)
        run = st.button("Score", type="primary", use_container_width=True)

    if run and body.strip():
        res = score_review(sentiment_model, fraud_model, body, star, verified)
        c3, c4, c5 = st.columns(3)
        c3.metric("Sentiment", res["sentiment"].upper())
        c4.metric("Fraud probability", f"{res['fraud_proba']:.3f}")
        c5.metric("Fraud flag (≥0.5)", "YES" if res["fraud_flag"] else "no")
        if res["fraud_flag"]:
            st.error("Fraud flag triggered.")
        elif res["fraud_proba"] >= 0.3:
            st.warning("Borderline — review the behavioral signal mix.")
        else:
            st.success("Looks clean by the trained model.")

# Tab 2 — top products
with t2:
    st.subheader("Top products by review volume")
    n_prod = st.slider("Top N", 5, 200, 25, key="np")
    df = load_product_agg(n_prod)
    st.caption(f"Loaded from `{PRODUCT_AGG_PARQUET.name}` ({len(df):,} rows shown)")
    df_show = df.copy()
    df_show["avg_rating"] = df_show["avg_rating"].round(3)
    df_show["pct_5star"] = (df_show["pct_5star"] * 100).round(1)
    df_show["fraud_rate"] = (df_show["fraud_rate"] * 100).round(2)
    df_show.rename(
        columns={
            "review_count": "reviews",
            "pct_5star": "% 5-star",
            "fraud_rate": "fraud %",
            "fraud_review_count": "fraud reviews",
        },
        inplace=True,
    )
    st.dataframe(df_show, use_container_width=True, hide_index=True)

# Tab 3 — suspicious reviewers
with t3:
    st.subheader("Suspicious reviewers (by fraud rate)")
    n_rev = st.slider("Top N", 5, 200, 25, key="nr")
    df = load_reviewer_agg(n_rev)
    st.caption(f"Loaded from `{REVIEWER_AGG_PARQUET.name}` ({len(df):,} rows shown)")
    df_show = df.copy()
    df_show["avg_rating"] = df_show["avg_rating"].round(3)
    df_show["fraud_rate"] = (df_show["fraud_rate"] * 100).round(2)
    df_show["verified_share"] = (df_show["verified_share"] * 100).round(1)
    df_show.rename(
        columns={
            "review_count": "reviews",
            "fraud_rate": "fraud %",
            "verified_share": "% verified",
        },
        inplace=True,
    )
    st.dataframe(df_show, use_container_width=True, hide_index=True)

# Tab 4 — raw browser
with t4:
    st.subheader("Browse raw reviews")
    c1, c2, c3 = st.columns([1, 1, 2])
    skip = c1.number_input("Skip rows", min_value=0, value=0, step=10)
    n = c2.number_input("Show", min_value=1, max_value=200, value=10, step=10)
    do_score = c3.checkbox("Score with the trained models", value=True)
    rows = load_raw_sample(int(n), int(skip))
    if not rows:
        st.warning(f"No raw data found at `{RAW_REVIEWS}`.")
    else:
        df = pd.DataFrame(rows)
        if do_score and not df.empty:
            feats = enrich_for_serving(df)
            sent = sentiment_model.predict(feats["review_body_clean"]).astype(int)
            fcols = ["review_body_clean"] + NUMERIC_FRAUD_FEATURES
            fproba = fraud_model.predict_proba(feats[fcols])[:, 1]
            df["sentiment"] = [SENTIMENT_LABELS[int(x)] for x in sent]
            df["fraud_proba"] = fproba.round(3)
            df["fraud_flag"] = (fproba >= 0.5).astype(int)
        cols = [
            c
            for c in [
                "star_rating",
                "review_date",
                "verified_purchase",
                "review_headline",
                "review_body",
                "sentiment",
                "fraud_proba",
                "fraud_flag",
            ]
            if c in df.columns
        ]
        st.dataframe(df[cols], use_container_width=True, hide_index=True)

# Tab 5 — distributions
with t5:
    st.subheader("Held-out test set distributions")
    df = load_test_sample(50_000)
    st.caption(f"Sample of {len(df):,} rows from `{TEST_PARQUET.name}`")

    c1, c2 = st.columns(2)
    with c1:
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("star_rating:O", title="Star rating"),
                y=alt.Y("count()", title="Reviews"),
                tooltip=["star_rating", "count()"],
            )
            .properties(height=280, title="Star rating distribution")
        )
        st.altair_chart(chart, use_container_width=True)
    with c2:
        df["sentiment_str"] = df["sentiment_label"].map(SENTIMENT_LABELS)
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("sentiment_str:N", title="Sentiment", sort=["negative", "neutral", "positive"]),
                y=alt.Y("count()", title="Reviews"),
                color=alt.Color("sentiment_str:N", legend=None),
                tooltip=["sentiment_str", "count()"],
            )
            .properties(height=280, title="Sentiment label distribution")
        )
        st.altair_chart(chart, use_container_width=True)

    st.markdown("**Body word count distribution** (clipped at 200 for readability)")
    df_clip = df.copy()
    df_clip["body_word_count_clip"] = df_clip["body_word_count"].clip(upper=200)
    chart = (
        alt.Chart(df_clip)
        .mark_bar()
        .encode(
            x=alt.X("body_word_count_clip:Q", bin=alt.Bin(maxbins=40), title="Word count"),
            y=alt.Y("count()", title="Reviews"),
        )
        .properties(height=240)
    )
    st.altair_chart(chart, use_container_width=True)

    st.caption(
        "Star rating is heavily skewed to 5★ in real Amazon data, "
        "which is why the sentiment macro-F1 (treating classes equally) "
        "is meaningfully lower than the weighted-F1."
    )
