"""Shared, provider-neutral configuration helpers for optional examples."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping


def required_confidence_threshold(
    name: str, *, environ: Mapping[str, str] | None = None
) -> float:
    """Read an explicitly configured confidence threshold from the environment."""
    values = os.environ if environ is None else environ
    raw = values.get(name)
    if raw is None or not raw.strip() or raw.strip().casefold().startswith("insert-"):
        raise RuntimeError(f"set {name} to a number within (0, 1]")
    try:
        threshold = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number within (0, 1]") from exc
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise RuntimeError(f"{name} must be finite and within (0, 1]")
    return threshold
