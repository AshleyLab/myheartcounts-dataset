"""Canonical per-method forecasting per-user metric substrate (Track 3).

Single importable producer + serializer + reconstruction adapters for the
forecasting per-user metric table — the one artifact every downstream reducer
(skill point/bootstrap, rank point/bootstrap, fairness) consumes. Mirrors the
imputation track's :mod:`imputation_evaluation.evaluation.per_user_errors`,
adapted to forecasting's on-disk **metric trees**
(``<metrics_dir>/<metric>/<user>.parquet`` — forecasting has no ``pairs/``
layer) and stored at **float64** so the reconstructed skill/rank aggregates are
byte-exact against the legacy from-trees path.

The substrate stores one row per ``(model, group, metric, channel, user)`` with
the RAW micro-pooled ``metric_value`` (= Σcell / Σcount over the user's finite
horizon cells) and the finite-cell count ``n_values``. Storing RAW (not the
error-converted value) is what lets a single table serve all three reducers:

  * skill / fairness apply :func:`metric_spec.metric_to_error` on load (incl. the
    ``BINARY_ERROR_FLOOR`` floor) — see :func:`to_error_df`;
  * rank uses ``metric_value`` directly and rebuilds the activity/physiology
    (continuous) and sleep/workout (binary) group scopes — see
    :func:`to_rank_user_df`.

Only ``within_user_aggregation="micro"`` + ``aggregation_unit="user"`` are
supported (and recorded in the ``.meta.json`` sidecar): under micro pooling both
``metric_value`` and ``error = metric_to_error(metric_value)`` are exact
functions of the stored ``(Σcell, Σcount)``. ``macro`` is *not* reconstructable
for binary metrics (``mean_w(1−AUC_w) ≠ 1−mean_w(AUC_w)``) and stays on the
legacy from-trees path only.
"""

from __future__ import annotations

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

from forecasting_evaluation.metrics import metric_spec as _spec  # noqa: E402

PER_USER_METRICS_PARQUET_COLUMNS = [
    "model",
    "group",
    "metric",
    "channel_idx",
    "channel_name",
    "user_id",
    "metric_value",
    "n_values",
]

# group -> rank scope_type for the per-channel and the group-fold rows.
_CHANNEL_SCOPE_TYPE = {"continuous": "continuous_channel", "binary": "binary_channel"}
_GROUP_SCOPE_TYPE = {"continuous": "continuous_group", "binary": "binary_group"}


# ---------------------------------------------------------------------------
# Producer (reads metric trees)
# ---------------------------------------------------------------------------


def _load_model_group_rows(
    *,
    model_name: str,
    model_root: str | Path,
    group_name: str,
    metric_name: str,
    channel_indices: tuple[int, ...],
) -> list[dict[str, Any]]:
    """RAW micro per-(user, channel) rows for one model/group/metric.

    Mirrors ``grouped_metric_rank_summary._load_channel_user_metrics`` (micro
    branch): pool each user's per-window ``(sum, count)`` over a channel's finite
    horizon cells, then ``metric_value = Σsum / Σcount``, ``n_values = Σcount``.
    """
    metric_dir = Path(model_root) / metric_name
    per_user_pairs: dict[tuple[str, int], list[tuple[float, int]]] = {}
    for parquet_file in _spec.list_parquet_files(metric_dir):
        df = _spec.safe_read_parquet(parquet_file, columns=["user_id", metric_name])
        if df is None or "user_id" not in df.columns or metric_name not in df.columns:
            continue
        for _, row in df.iterrows():
            user_id = str(row.get("user_id"))
            metric = _spec.safe_to_metric_array(row.get(metric_name))
            if metric is None:
                continue
            for channel_idx in channel_indices:
                sum_count = _spec.metric_channel_sum_count(metric=metric, channel_idx=channel_idx)
                if sum_count is None:
                    continue
                per_user_pairs.setdefault((user_id, int(channel_idx)), []).append(sum_count)

    rows: list[dict[str, Any]] = []
    for (user_id, channel_idx), pairs in per_user_pairs.items():
        total_count = int(sum(count for _, count in pairs))
        if total_count == 0:
            continue
        metric_value = float(sum(value for value, _ in pairs)) / total_count
        rows.append(
            {
                "model": model_name,
                "group": group_name,
                "metric": metric_name,
                "channel_idx": int(channel_idx),
                "channel_name": _spec.channel_label(channel_idx),
                "user_id": user_id,
                "metric_value": metric_value,
                "n_values": total_count,
            }
        )
    return rows


def build_per_user_metrics(
    *,
    models: dict[str, dict[str, str]],
    continuous_metrics: list[str],
    binary_metrics: list[str],
    continuous_channel_indices: tuple[int, ...] = _spec.CONTINUOUS_CHANNELS,
    binary_channel_indices: tuple[int, ...] = _spec.BINARY_CHANNELS,
    within_user_aggregation: str = "micro",
    aggregation_unit: str = "user",
) -> pd.DataFrame:
    """Build the canonical per-(model, user, channel, metric) substrate from trees.

    Reads each model's metric trees (``<model_root>/<metric>/<user>.parquet``) and
    emits :data:`PER_USER_METRICS_PARQUET_COLUMNS`. Continuous metrics are scored
    on ``continuous_channel_indices`` (default 0-6), binary metrics on
    ``binary_channel_indices`` (default 7-18). Concatenate across methods before
    consuming; the ``model`` column distinguishes them.

    Args:
        models: ``{name: {"path": metrics_dir, "display_name": ...}}``.
        continuous_metrics: e.g. ``["mae", "mse", ...]`` — scored on continuous
            channels, stored under ``group="continuous"``.
        binary_metrics: e.g. ``["auroc", "auprc", "f1"]`` — scored on binary
            channels, stored under ``group="binary"``.
        continuous_channel_indices: continuous channel ids.
        binary_channel_indices: binary channel ids.
        within_user_aggregation: must be ``"micro"`` (only supported mode).
        aggregation_unit: must be ``"user"`` (only supported mode).

    Returns:
        DataFrame with :data:`PER_USER_METRICS_PARQUET_COLUMNS`; ``metric_value``
        is the RAW micro-pooled per-user value, ``n_values`` the finite-cell count.

    Raises:
        ValueError: if a non-``micro`` / non-``user`` mode is requested (those are
            not reconstructable from the single per-user value).
    """
    if within_user_aggregation != "micro":
        raise ValueError(
            "build_per_user_metrics only supports within_user_aggregation='micro' "
            "(macro is not reconstructable for binary metrics); got "
            f"{within_user_aggregation!r}."
        )
    if aggregation_unit != "user":
        raise ValueError(
            f"build_per_user_metrics only supports aggregation_unit='user'; got "
            f"{aggregation_unit!r}."
        )

    metric_groups = {
        "continuous": {
            "metrics": [m.strip().lower() for m in continuous_metrics if m.strip()],
            "channel_indices": tuple(continuous_channel_indices),
        },
        "binary": {
            "metrics": [m.strip().lower() for m in binary_metrics if m.strip()],
            "channel_indices": tuple(binary_channel_indices),
        },
    }

    rows: list[dict[str, Any]] = []
    for group_name, group_spec in metric_groups.items():
        for model_name, model_spec in models.items():
            for metric_name in group_spec["metrics"]:
                rows.extend(
                    _load_model_group_rows(
                        model_name=model_name,
                        model_root=model_spec["path"],
                        group_name=group_name,
                        metric_name=metric_name,
                        channel_indices=group_spec["channel_indices"],
                    )
                )

    if not rows:
        return pd.DataFrame(columns=PER_USER_METRICS_PARQUET_COLUMNS)
    return pd.DataFrame(rows, columns=PER_USER_METRICS_PARQUET_COLUMNS)


# ---------------------------------------------------------------------------
# Serialization (float64 — deliberately NOT float32, unlike the imputation track)
# ---------------------------------------------------------------------------


def write_per_user_metrics_parquet(
    df: pd.DataFrame,
    path: str | Path,
    meta: dict | None = None,
) -> None:
    """Write the per-user metrics substrate + optional sidecar metadata JSON.

    Categorical dtypes for the low-cardinality string keys, **float64**
    ``metric_value`` (deliberately not float32 — needed for byte-exact skill/rank
    parity), int dtypes for ``channel_idx`` / ``n_values``, zstd compression.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df[PER_USER_METRICS_PARQUET_COLUMNS].copy()
    df["metric_value"] = df["metric_value"].astype("float64")
    df["channel_idx"] = df["channel_idx"].astype("int16")
    df["n_values"] = df["n_values"].astype("int64")
    for col in ("model", "group", "metric", "channel_name", "user_id"):
        df[col] = df[col].astype("category")
    df.to_parquet(path, compression="zstd")
    if meta is not None:
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2, default=str))


def read_per_user_metrics_parquet(path: str | Path) -> tuple[pd.DataFrame, dict | None]:
    """Read the per-user metrics substrate and its sidecar metadata if present."""
    path = Path(path)
    df = pd.read_parquet(path)
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    meta: dict | None = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    return df, meta


# ---------------------------------------------------------------------------
# Reconstruction adapters (the single contract feeding every reducer)
# ---------------------------------------------------------------------------


def to_error_df(per_user_df: pd.DataFrame, *, user_col: str = "unit_id") -> pd.DataFrame:
    """Reconstruct the skill/fairness per-user error table from the substrate.

    Applies :func:`metric_spec.metric_to_error` per row (incl. the binary floor),
    drops non-finite errors (matching ``skill_score_summary._load_metric_values``),
    and renames ``user_id`` → ``user_col`` (``"unit_id"`` for skill/bootstrap,
    ``"user_id"`` for fairness). Keys are returned as plain ``str`` (not category)
    so the bootstrap replica/resample machinery works unchanged.
    """
    columns = [
        "model",
        "group",
        "metric",
        "channel_idx",
        "channel_name",
        user_col,
        "error",
        "n_values",
    ]
    if per_user_df.empty:
        return pd.DataFrame(columns=columns)

    df = per_user_df.copy()
    metrics = df["metric"].astype(str).to_numpy()
    values = df["metric_value"].to_numpy(dtype=float)
    df["error"] = [
        _spec.metric_to_error(metric, float(value)) for metric, value in zip(metrics, values)
    ]
    df = df.loc[np.isfinite(df["error"].to_numpy(dtype=float))].copy()
    if df.empty:
        return pd.DataFrame(columns=columns)
    df = df.rename(columns={"user_id": user_col})
    for col in ("model", "group", "metric", "channel_name", user_col):
        df[col] = df[col].astype(str)
    df["channel_idx"] = df["channel_idx"].astype(int)
    df["n_values"] = df["n_values"].astype(int)
    return df[columns].reset_index(drop=True)


def to_rank_user_df(
    per_user_df: pd.DataFrame,
    *,
    binary_groups: list[tuple[str, tuple[int, ...]]],
    continuous_groups: list[tuple[str, tuple[int, ...]]] | None = None,
) -> pd.DataFrame:
    """Reconstruct the rank per-user table (channel + group scopes) from the substrate.

    Channel rows map ``group`` → ``scope_type`` (continuous→continuous_channel,
    binary→binary_channel) and keep the RAW ``metric_value``. Group rows rebuild
    the activity/physiology (continuous) and sleep/workout (binary) scopes by
    averaging ``metric_value`` and summing ``n_values`` within
    ``(model, user_id, metric, metric_display)`` — matching
    ``grouped_metric_rank_summary._build_{continuous,binary}_user_rows``. Keys are
    returned as plain ``str`` for the bootstrap replica/resample machinery.
    """
    columns = [
        "model",
        "scope_type",
        "scope",
        "scope_label",
        "metric",
        "metric_display",
        "channel_idx",
        "user_id",
        "metric_value",
        "n_values",
    ]
    if per_user_df.empty:
        return pd.DataFrame(columns=columns)
    if continuous_groups is None:
        continuous_groups = [(name, tuple(idx)) for name, idx in _spec.CONTINUOUS_GROUPS]
    group_specs: dict[str, list[tuple[str, tuple[int, ...]]]] = {
        "continuous": [(name, tuple(idx)) for name, idx in continuous_groups],
        "binary": [(name, tuple(idx)) for name, idx in binary_groups],
    }

    frames: list[pd.DataFrame] = []
    group_values = per_user_df["group"].astype(str)
    for group_name, channel_scope_type in _CHANNEL_SCOPE_TYPE.items():
        slice_df = per_user_df.loc[group_values == group_name]
        if slice_df.empty:
            continue
        channel_idx = slice_df["channel_idx"].astype(int).to_numpy()
        metric_str = slice_df["metric"].astype(str).to_numpy()
        channel_rows = pd.DataFrame(
            {
                "model": slice_df["model"].astype(str).to_numpy(),
                "scope_type": channel_scope_type,
                "scope": [f"channel_{idx}" for idx in channel_idx],
                "scope_label": [_spec.channel_label(idx) for idx in channel_idx],
                "metric": metric_str,
                "metric_display": [_spec.metric_display_name(m) for m in metric_str],
                "channel_idx": channel_idx,
                "user_id": slice_df["user_id"].astype(str).to_numpy(),
                "metric_value": slice_df["metric_value"].astype(float).to_numpy(),
                "n_values": slice_df["n_values"].astype(int).to_numpy(),
            }
        )
        frames.append(channel_rows[columns])

        present = set(int(idx) for idx in channel_idx)
        for grp_name, grp_channels in group_specs[group_name]:
            if not set(grp_channels).issubset(present):
                continue
            group_slice = channel_rows.loc[channel_rows["channel_idx"].isin(grp_channels)]
            if group_slice.empty:
                continue
            grouped = group_slice.groupby(
                ["model", "user_id", "metric", "metric_display"], as_index=False
            ).agg(metric_value=("metric_value", "mean"), n_values=("n_values", "sum"))
            grouped["scope_type"] = _GROUP_SCOPE_TYPE[group_name]
            grouped["scope"] = grp_name
            grouped["scope_label"] = grp_name
            grouped["channel_idx"] = -1
            frames.append(grouped[columns])

    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


__all__ = [
    "PER_USER_METRICS_PARQUET_COLUMNS",
    "build_per_user_metrics",
    "write_per_user_metrics_parquet",
    "read_per_user_metrics_parquet",
    "to_error_df",
    "to_rank_user_df",
]
