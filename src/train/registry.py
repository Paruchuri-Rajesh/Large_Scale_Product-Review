"""Small helpers for saving ML selection metadata.

This keeps model-choice metadata separate from the main training code while
remaining compatible with the current saved-artifact flow.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


# Save a JSON object to disk with readable formatting.
def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


# Load a JSON object if it exists, otherwise return a default value.
def load_json(path: Path, default: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if not path.exists():
        return default or {}
    return json.loads(path.read_text())


# Parse JSON from disk if present; otherwise None (for optional meta embeds).
def load_json_optional(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    return json.loads(path.read_text())


# Build a compact summary of selected model information.
def build_model_selection_summary(
    sentiment_summary: Dict[str, Any],
    fraud_summary: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "sentiment": sentiment_summary,
        "fraud": fraud_summary,
    }
