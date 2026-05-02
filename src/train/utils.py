"""Small shared helpers for ML-side scripts."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable


# Make sure a directory exists before writing outputs.
def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# Turn any iterable into a plain list.
def to_list(values: Iterable) -> list:
    return list(values)
