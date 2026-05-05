"""Shared helpers for offline forecasting metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from forecasting_evaluation.config import ForecastingModelConfig


def get_model_name(config: ForecastingModelConfig) -> str:
    """Get deterministic model name used in output paths."""
    return config.name if config.name else config.type


def sanitize_name(name: str) -> str:
    """Convert name into filesystem-safe string."""
    sanitized = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)
    return sanitized or "unknown"


def coerce_non_negative_int(value: Any) -> int | None:
    """Parse a non-negative integer value."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def coerce_1d_float_array(value: Any) -> np.ndarray | None:
    """Coerce value to 1D float array."""
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    return arr if arr.ndim == 1 else None


def coerce_2d_float_array(value: Any) -> np.ndarray | None:
    """Coerce value to 2D float array."""
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    return arr if arr.ndim == 2 else None


def coerce_3d_float_array(value: Any) -> np.ndarray | None:
    """Coerce value to 3D float array."""
    if value is None:
        return None
    arr = np.asarray(value, dtype=float)
    return arr if arr.ndim == 3 else None


def resolve_quantile_levels(levels: np.ndarray | None, n_quantiles: int) -> np.ndarray:
    """Resolve quantile levels with evenly spaced fallback."""
    if levels is not None and levels.shape[0] == n_quantiles:
        return levels
    return np.linspace(1.0 / (n_quantiles + 1), n_quantiles / (n_quantiles + 1), n_quantiles)
