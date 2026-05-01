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
