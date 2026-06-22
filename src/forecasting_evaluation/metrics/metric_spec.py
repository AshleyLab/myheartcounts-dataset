"""Single source of truth for forecasting metric semantics.

Centralizes the channel groupings, metric directionality, channel-name lookup,
and the small parquet/error helpers that the Layer-2 summary scripts
(``skill_score_summary``, ``grouped_metric_rank_summary``,
``fairness_skill_score_summary``) used to each redeclare. Importing from here
keeps the scoring definition consistent across all of them.

Channel groups (fixed by the benchmark):
    * continuous channels 0-6 (point-forecast targets),
    * sleep channels 7-8,
    * workout channels 9-18.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# --- Channel metadata (names), loaded once from visualizations/constants.py ---
_CHANNEL_CONSTANTS_PATH = SRC_ROOT / "visualizations" / "constants.py"
_CHANNEL_SPEC = (
    importlib.util.spec_from_file_location(
        "_forecasting_metric_spec_channel_constants",
        _CHANNEL_CONSTANTS_PATH,
    )
    if _CHANNEL_CONSTANTS_PATH.exists()
    else None
)
if _CHANNEL_SPEC is None or _CHANNEL_SPEC.loader is None:
    CHANNEL_INFO: dict[int, dict[str, Any]] = {}
else:
    _channel_module = importlib.util.module_from_spec(_CHANNEL_SPEC)
    _CHANNEL_SPEC.loader.exec_module(_channel_module)
    CHANNEL_INFO = getattr(_channel_module, "CHANNEL_INFO", {})

# --- Metric directionality ---
LOWER_IS_BETTER_METRICS = {"mae", "mse", "mase", "mase_all", "ql", "sql"}
HIGHER_IS_BETTER_METRICS = {"f1", "auprc", "auroc"}

# ε-floor on the higher-is-better error e = 1 − metric. A perfect score (e = 0)
# would make the paired skill ratio e_method/e_baseline either 0 or divide-by-zero;
# flooring keeps every defined-AUC user in the paired set symmetrically (a perfect
# user contributes a small finite error instead of being dropped). Treats a perfect
# AUROC as indistinguishable from 0.995 — a finite-sample resolution floor. Applies
# to binary/classification metrics only; continuous errors are left unfloored
# (an absolute floor is meaningless in their scale-dependent units).
BINARY_ERROR_FLOOR = 0.005

# --- Channel groups (benchmark-fixed) ---
CONTINUOUS_CHANNELS: tuple[int, ...] = tuple(range(0, 7))
SLEEP_CHANNELS: tuple[int, ...] = (7, 8)
WORKOUT_CHANNELS: tuple[int, ...] = tuple(range(9, 19))
BINARY_CHANNELS: tuple[int, ...] = tuple(range(7, 19))

# --- Sensor-category scopes (the category-balanced "overall") ---
# Partition of all 19 channels into the 4 semantic scopes, each weighted ONCE in
# the category-balanced overall so the 10 workout channels can't dominate the
# headline. This is the single source of truth — the per-track reporting group
# lists below derive from it. (Forecasting keeps its names, "workout" singular,
# vs the imputation track's "workouts".)
CATEGORY_SCOPES: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("activity", (0, 1, 2, 3, 4)),  # steps + distance + flights (iphone + watch)
    ("physiology", (5, 6)),  # heart rate + active energy (watch)
    ("sleep", SLEEP_CHANNELS),  # asleep / inbed (binary)
    ("workout", WORKOUT_CHANNELS),  # 10 workout-type channels (binary)
)
CHANNEL_TO_CATEGORY_SCOPE: dict[int, str] = {
    idx: name for name, idxs in CATEGORY_SCOPES for idx in idxs
}

# Per-track reporting groups, derived from CATEGORY_SCOPES by channel kind. Each
# channel keeps its own skill ratio and the group combines them with a geometric
# mean (per-task scoring), like the imputation track's per-channel categories. The
# earlier steps/distance device-pair scopes were dropped — redundant with the
# `activity` scope plus the per-channel rows.
CONTINUOUS_GROUPS: tuple[tuple[str, tuple[int, ...]], ...] = tuple(
    (name, idxs) for name, idxs in CATEGORY_SCOPES if set(idxs) <= set(CONTINUOUS_CHANNELS)
)
BINARY_GROUPS: tuple[tuple[str, tuple[int, ...]], ...] = tuple(
    (name, idxs) for name, idxs in CATEGORY_SCOPES if set(idxs) <= set(BINARY_CHANNELS)
)

# --- Paper-default scoring config -------------------------------------------
# The baked defaults so a user running the public API (``openmhc.evaluate_forecasting``)
# gets results identical to the paper CLI for the same method. These MIRROR
# ``configs/paper/sweep_forecasting.yaml`` (baseline=seasonal_naive, scored
# metrics mae/auroc, ratio clip [0.01, 100], micro/user aggregation) and are the
# single source of truth for the public-API skill computation.
PAPER_BASELINE = "seasonal_naive"
PAPER_CONTINUOUS_METRICS: tuple[str, ...] = ("mae",)
PAPER_BINARY_METRICS: tuple[str, ...] = ("auroc",)
PAPER_CLIP_LOWER: float = 1e-2
PAPER_CLIP_UPPER: float = 100.0
PAPER_MIN_PAIRS: int = 1


def category_scope_for_channel(channel_idx: int) -> str | None:
    """Return the sensor-category scope for a channel index, or None if unmapped."""
    return CHANNEL_TO_CATEGORY_SCOPE.get(int(channel_idx))


_METRIC_DISPLAY = {
    "mae": "MAE",
    "mse": "MSE",
    "mase": "MASE",
    "mase_all": "MASE_all",
    "ql": "QL",
    "sql": "sQL",
    "f1": "F1",
    "auprc": "AUPRC",
    "auroc": "AUROC",
}


def metric_display_name(metric_name: str) -> str:
    """Human-readable label for a metric key."""
    return _METRIC_DISPLAY.get(metric_name.strip().lower(), metric_name)


def metric_lower_is_better(metric_name: str) -> bool:
    """True if smaller is better; raises for metrics in neither direction set."""
    metric_key = metric_name.strip().lower()
    if metric_key in LOWER_IS_BETTER_METRICS:
        return True
    if metric_key in HIGHER_IS_BETTER_METRICS:
        return False
    raise ValueError(f"Unknown metric '{metric_name}'. Add it to lower- or higher-is-better sets.")


def metric_to_error(metric_name: str, metric_value: float) -> float:
    """Convert a metric value to an error (lower=better) for ratio/skill math."""
    metric_key = metric_name.strip().lower()
    if not np.isfinite(metric_value):
        return float("nan")
    if metric_key in LOWER_IS_BETTER_METRICS:
        return float(metric_value) if metric_value >= 0.0 else float("nan")
    if metric_key in HIGHER_IS_BETTER_METRICS:
        if metric_value < 0.0 or metric_value > 1.0:
            return float("nan")
        return float(max(1.0 - metric_value, BINARY_ERROR_FLOOR))
    raise ValueError(f"Unknown metric '{metric_name}'. Add it to lower- or higher-is-better sets.")


def channel_label(channel_idx: int) -> str:
    """Return the human-readable channel name, or ``Channel <idx>`` fallback."""
    metadata = CHANNEL_INFO.get(int(channel_idx))
    if metadata is None:
        return f"Channel {channel_idx}"
    return str(metadata["name"])


def parse_channel_indices(raw_value: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    """Parse a comma-separated channel-index string, or return ``default``."""
    if raw_value is None or not raw_value.strip():
        return default
    indices: list[int] = []
    for part in raw_value.split(","):
        token = part.strip()
        if token:
            indices.append(int(token))
    if not indices:
        raise ValueError("At least one channel index must be provided.")
    return tuple(indices)


def safe_read_parquet(file_path: str | Path, **kwargs: Any) -> pd.DataFrame | None:
    """Read a parquet file, returning None on missing/empty/unreadable input."""
    path = Path(file_path)
    try:
        if not path.exists() or path.stat().st_size == 0:
            return None
        return pd.read_parquet(path, **kwargs)
    except Exception:
        return None


def list_parquet_files(path: str | Path) -> list[Path]:
    """Sorted list of all ``*.parquet`` under ``path`` (recursive)."""
    root = Path(path)
    if not root.exists():
        return []
    return sorted(root.rglob("*.parquet"))


def safe_to_metric_array(value: Any) -> np.ndarray | None:
    """Coerce a nested metric value to a 1D/2D float array (min-length rows)."""
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float)
        if arr.ndim in {1, 2}:
            return arr
    except Exception:
        pass

    try:
        obj = np.asarray(value, dtype=object)
    except Exception:
        return None
    if obj.ndim != 1:
        return None

    rows: list[np.ndarray] = []
    for item in obj.tolist():
        try:
            row = np.asarray(item, dtype=float).reshape(-1)
        except Exception:
            return None
        if row.size == 0:
            return None
        rows.append(row)
    if not rows:
        return None
    min_len = min(row.shape[0] for row in rows)
    if min_len <= 0:
        return None
    return np.vstack([row[:min_len] for row in rows])


def metric_channel_sum_count(metric: np.ndarray, channel_idx: int) -> tuple[float, int] | None:
    """Sum and finite-cell count over the horizon for one channel.

    Returns ``(sum_of_finite_cells, count_of_finite_cells)``, or ``None`` if the
    channel index is out of range or the channel has no finite cells. The finite
    mask (``np.isfinite``) is the per-cell validity used to micro-average within a
    user's prediction windows.
    """
    if metric.ndim == 1:
        if channel_idx >= metric.shape[0]:
            return None
        value = float(metric[channel_idx])
        if not np.isfinite(value):
            return None
        return value, 1

    if channel_idx >= metric.shape[0]:
        return None
    values = metric[channel_idx]
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return None
    return float(np.sum(finite_values)), int(finite_values.size)


def metric_channel_value(metric: np.ndarray, channel_idx: int) -> float:
    """Collapse a (channel[,horizon]) metric array to one finite-mean for a channel."""
    sum_count = metric_channel_sum_count(metric, channel_idx)
    if sum_count is None:
        return float("nan")
    total, count = sum_count
    return total / count


def load_models_dict(args: argparse.Namespace) -> dict[str, dict[str, str]]:
    """Load a ``{model: {path, display_name}}`` map from --models-json/--config."""
    if args.models_json:
        parsed = json.loads(args.models_json)
    elif args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            raise ValueError(f"Config file not found: {config_path}")
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except ImportError as exc:
                raise ImportError("PyYAML is required for yaml config input") from exc
            with config_path.open("r", encoding="utf-8") as file:
                parsed = yaml.safe_load(file)
        else:
            with config_path.open("r", encoding="utf-8") as file:
                parsed = json.load(file)
    else:
        raise ValueError("Please provide --models-json or --config")

    if isinstance(parsed, dict) and "models" in parsed:
        parsed = parsed["models"]

    models: dict[str, dict[str, str]] = {}
    if isinstance(parsed, dict):
        for key, value in parsed.items():
            model_name = str(key).strip()
            if isinstance(value, dict):
                model_path = str(value.get("path", "")).strip()
                display_name = str(value.get("display_name", model_name)).strip()
            else:
                model_path = str(value).strip()
                display_name = model_name
            if not model_name or not model_path:
                raise ValueError("Model configuration must use non-empty model names and paths")
            models[model_name] = {
                "path": model_path,
                "display_name": display_name or model_name,
            }
    elif isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                raise ValueError("Model configuration list entries must be dictionaries")
            model_name = str(item.get("name", "")).strip()
            model_path = str(item.get("path", "")).strip()
            display_name = str(item.get("display_name", model_name)).strip()
            if not model_name or not model_path:
                raise ValueError("Each model entry must contain non-empty name and path")
            models[model_name] = {
                "path": model_path,
                "display_name": display_name or model_name,
            }
    else:
        raise ValueError("Model configuration must be a dict or list")

    if not models:
        raise ValueError("No model mappings found in configuration")
    return models
