"""Grouped forecasting metric/rank summary for continuous and binary scopes.

This script combines the reporting shape used by the continuous channel summary
and binary group summary:

* continuous channels are reported one channel at a time;
* binary channels are reported as configured groups, defaulting to sleep
  channels 7-8 and workout channels 9-18.

The minimal unit before model-level aggregation is ``(metric, channel_idx,
user_id)``. For binary groups, channel-level user metrics are averaged within
each group/user before model means and ranks are computed.
"""

from __future__ import annotations

import argparse
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

CHANNEL_INFO = _spec.CHANNEL_INFO
LOWER_IS_BETTER_METRICS = _spec.LOWER_IS_BETTER_METRICS
HIGHER_IS_BETTER_METRICS = _spec.HIGHER_IS_BETTER_METRICS
DEFAULT_CONTINUOUS_CHANNELS = _spec.CONTINUOUS_CHANNELS
DEFAULT_BINARY_GROUPS = _spec.BINARY_GROUPS


_safe_read_parquet = _spec.safe_read_parquet


_list_parquet_files = _spec.list_parquet_files


_channel_label = _spec.channel_label


_metric_display_name = _spec.metric_display_name


_metric_lower_is_better = _spec.metric_lower_is_better


_parse_channel_indices = _spec.parse_channel_indices


def _parse_group_arg(group_arg: str) -> tuple[str, tuple[int, ...]]:
    if "=" not in group_arg:
        raise argparse.ArgumentTypeError(
            f"Invalid --binary-group value: {group_arg}. Expected GROUP=7,8"
        )
    group_name, raw_indices = group_arg.split("=", 1)
    group_name = group_name.strip()
    if not group_name:
        raise argparse.ArgumentTypeError("Binary group name must be non-empty.")
    indices = [int(token.strip()) for token in raw_indices.split(",") if token.strip()]
    if not indices:
        raise argparse.ArgumentTypeError("Binary group must include at least one channel.")
    return group_name, tuple(indices)


def _resolve_binary_groups(group_args: list[str] | None) -> list[tuple[str, tuple[int, ...]]]:
    if not group_args:
        return [(name, tuple(indices)) for name, indices in DEFAULT_BINARY_GROUPS]
    groups: list[tuple[str, tuple[int, ...]]] = []
    seen: set[str] = set()
    for item in group_args:
        name, indices = _parse_group_arg(item)
        if name in seen:
            raise ValueError(f"Duplicate binary group name: {name}")
        seen.add(name)
        groups.append((name, indices))
    return groups


def _load_models_dict(args: argparse.Namespace) -> dict[str, dict[str, str]]:
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


def _safe_to_metric_array(value: Any) -> np.ndarray | None:
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


def _metric_channel_value(metric: np.ndarray, channel_idx: int) -> tuple[float, int] | None:
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
    return float(np.mean(finite_values)), int(finite_values.size)


def _load_channel_user_metrics(
    *,
    model_name: str,
    model_root: str | Path,
    metric_name: str,
    channel_indices: tuple[int, ...],
    scope_type: str,
) -> pd.DataFrame:
    metric_dir = Path(model_root) / metric_name
    rows: list[dict[str, Any]] = []
    per_user_values: dict[tuple[str, int], list[float]] = {}
    per_user_counts: dict[tuple[str, int], int] = {}

    for parquet_file in _list_parquet_files(metric_dir):
        df = _safe_read_parquet(parquet_file, columns=["user_id", metric_name])
        if df is None or "user_id" not in df.columns or metric_name not in df.columns:
            continue
        for _, row in df.iterrows():
            user_id = str(row.get("user_id"))
            metric = _safe_to_metric_array(row.get(metric_name))
            if metric is None:
                continue
            for channel_idx in channel_indices:
                collapsed = _metric_channel_value(metric=metric, channel_idx=channel_idx)
                if collapsed is None:
                    continue
                metric_value, n_values = collapsed
                per_user_values.setdefault((user_id, int(channel_idx)), []).append(metric_value)
                per_user_counts[(user_id, int(channel_idx))] = per_user_counts.get(
                    (user_id, int(channel_idx)), 0
                ) + int(n_values)

    for (user_id, channel_idx), values in per_user_values.items():
        finite_values = np.asarray(values, dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        if finite_values.size == 0:
            continue
        rows.append(
            {
                "model": model_name,
                "scope_type": scope_type,
                "scope": f"channel_{channel_idx}",
                "scope_label": _channel_label(channel_idx),
                "metric": metric_name,
                "metric_display": _metric_display_name(metric_name),
                "channel_idx": int(channel_idx),
                "user_id": user_id,
                "metric_value": float(np.mean(finite_values)),
                "n_values": int(per_user_counts.get((user_id, channel_idx), finite_values.size)),
            }
        )

    return pd.DataFrame(rows)


def _build_continuous_user_rows(
    *,
    models: dict[str, dict[str, str]],
    metrics: list[str],
    channel_indices: tuple[int, ...],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for model_name, model_spec in models.items():
        for metric_name in metrics:
            frame = _load_channel_user_metrics(
                model_name=model_name,
                model_root=model_spec["path"],
                metric_name=metric_name,
                channel_indices=channel_indices,
                scope_type="continuous_channel",
            )
            if not frame.empty:
                frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _build_binary_user_rows(
    *,
    models: dict[str, dict[str, str]],
    metrics: list[str],
    groups: list[tuple[str, tuple[int, ...]]],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    all_binary_channels = tuple(sorted({idx for _, indices in groups for idx in indices}))
    for model_name, model_spec in models.items():
        for metric_name in metrics:
            channel_rows = _load_channel_user_metrics(
                model_name=model_name,
                model_root=model_spec["path"],
                metric_name=metric_name,
                channel_indices=all_binary_channels,
                scope_type="binary_channel",
            )
            if channel_rows.empty:
                continue
            for group_name, channel_indices in groups:
                group_slice = channel_rows.loc[
                    channel_rows["channel_idx"].isin(channel_indices)
                ].copy()
                if group_slice.empty:
                    continue
                grouped = group_slice.groupby(
                    ["model", "user_id", "metric", "metric_display"],
                    as_index=False,
                ).agg(
                    metric_value=("metric_value", "mean"),
                    n_values=("n_values", "sum"),
                )
                grouped["scope_type"] = "binary_group"
                grouped["scope"] = group_name
                grouped["scope_label"] = group_name
                grouped["channel_idx"] = -1
                frames.append(
                    grouped[
                        [
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
                    ]
                )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _compute_mean_ranks(user_metric_df: pd.DataFrame) -> pd.DataFrame:
    if user_metric_df.empty:
        return pd.DataFrame(columns=["scope", "metric", "model", "rank", "rank_n_users"])

    finite_df = user_metric_df.loc[np.isfinite(user_metric_df["metric_value"])].copy()
    if finite_df.empty:
        return pd.DataFrame(columns=["scope", "metric", "model", "rank", "rank_n_users"])

    rank_rows: list[pd.DataFrame] = []
    for (scope, metric_name), group_slice in finite_df.groupby(["scope", "metric"], sort=True):
        pivot = group_slice.pivot(index="user_id", columns="model", values="metric_value")
        if pivot.empty:
            continue
        rank_df = pivot.rank(
            axis=1,
            method="average",
            ascending=_metric_lower_is_better(metric_name),
        )
        long_rank = rank_df.stack(future_stack=True).reset_index()
        long_rank.columns = ["user_id", "model", "rank"]
        long_rank["scope"] = scope
        long_rank["metric"] = metric_name
        rank_rows.append(long_rank)

    if not rank_rows:
        return pd.DataFrame(columns=["scope", "metric", "model", "rank", "rank_n_users"])
    rank_all = pd.concat(rank_rows, ignore_index=True)
    return rank_all.groupby(["scope", "metric", "model"], as_index=False).agg(
        rank=("rank", "mean"), rank_n_users=("user_id", "nunique")
    )


def _build_summary_tables(
    *,
    user_metric_df: pd.DataFrame,
    models: dict[str, dict[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_columns = [
        "scope_type",
        "scope",
        "scope_label",
        "metric",
        "metric_display",
        "model",
        "metric_mean",
        "rank",
        "n_users",
        "n_values",
        "rank_n_users",
    ]
    if user_metric_df.empty:
        return pd.DataFrame(columns=long_columns), pd.DataFrame(columns=["scope_type", "scope"])

    finite_df = user_metric_df.loc[np.isfinite(user_metric_df["metric_value"])].copy()
    metric_means = finite_df.groupby(
        ["scope_type", "scope", "scope_label", "metric", "metric_display", "model"],
        as_index=False,
    ).agg(
        metric_mean=("metric_value", "mean"),
        n_users=("user_id", "nunique"),
        n_values=("n_values", "sum"),
    )
    rank_df = _compute_mean_ranks(user_metric_df=user_metric_df)
    long_df = metric_means.merge(rank_df, on=["scope", "metric", "model"], how="left")
    model_order = list(models.keys())
    long_df["model"] = pd.Categorical(long_df["model"], categories=model_order, ordered=True)
    long_df = long_df.sort_values(["scope_type", "scope", "metric", "model"]).reset_index(drop=True)

    wide_df = (
        long_df[["scope_type", "scope", "scope_label", "metric", "metric_display"]]
        .drop_duplicates()
        .sort_values(["scope_type", "scope", "metric"])
        .reset_index(drop=True)
    )
    for model_name, model_spec in models.items():
        display_name = model_spec["display_name"]
        model_slice = long_df.loc[long_df["model"].astype(str) == model_name].copy()
        model_slice = model_slice[
            ["scope_type", "scope", "metric", "metric_mean", "rank", "n_users", "rank_n_users"]
        ].rename(
            columns={
                "metric_mean": f"{display_name}_metric",
                "rank": f"{display_name}_rank",
                "n_users": f"{display_name}_n_users",
                "rank_n_users": f"{display_name}_rank_n_users",
            }
        )
        wide_df = wide_df.merge(model_slice, on=["scope_type", "scope", "metric"], how="left")
    return long_df[long_columns], wide_df


def build_grouped_metric_rank_tables(
    *,
    models: dict[str, dict[str, str]],
    continuous_metrics: list[str],
    binary_metrics: list[str],
    continuous_channel_indices: tuple[int, ...],
    binary_groups: list[tuple[str, tuple[int, ...]]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build user-level, long, and wide grouped metric rank tables."""
    continuous_user = _build_continuous_user_rows(
        models=models,
        metrics=[metric.strip().lower() for metric in continuous_metrics if metric.strip()],
        channel_indices=continuous_channel_indices,
    )
    binary_user = _build_binary_user_rows(
        models=models,
        metrics=[metric.strip().lower() for metric in binary_metrics if metric.strip()],
        groups=binary_groups,
    )
    frames = [frame for frame in [continuous_user, binary_user] if not frame.empty]
    if frames:
        user_metric_df = pd.concat(frames, ignore_index=True)
    else:
        user_metric_df = pd.DataFrame()
    long_df, wide_df = _build_summary_tables(user_metric_df=user_metric_df, models=models)
    return user_metric_df, long_df, wide_df


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for grouped metric rank summaries."""
    parser = argparse.ArgumentParser(
        description=(
            "Summarize forecasting metrics/ranks with per-continuous-channel rows "
            "and grouped binary rows."
        )
    )
    parser.add_argument("--config", default=None, help="JSON/YAML config with model mappings.")
    parser.add_argument(
        "--models-json",
        default=None,
        help='Inline JSON dict, e.g. {"models":{"modelA":"/path/a"}}',
    )
    parser.add_argument(
        "--continuous-channel-indices",
        default="0,1,2,3,4,5,6",
        help="Comma-separated continuous channel indices. Defaults to 0-6.",
    )
    parser.add_argument(
        "--continuous-metrics",
        nargs="+",
        default=["mae"],
        help="Continuous metrics to report. Defaults to mae.",
    )
    parser.add_argument(
        "--binary-metrics",
        nargs="+",
        default=["auprc"],
        help="Binary-group metrics to report. Defaults to auprc.",
    )
    parser.add_argument(
        "--binary-group",
        action="append",
        default=None,
        help="Binary group mapping GROUP=idx,idx. Defaults to sleep=7,8 and workout=9-18.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/metrics_summary",
        help="Directory for generated CSV files.",
    )
    parser.add_argument(
        "--output-prefix",
        default="forecasting_grouped_metric_rank_summary",
        help="Filename prefix for generated CSV files.",
    )
    return parser


def main() -> None:
    """Generate grouped metric rank summary CSV outputs."""
    args = build_parser().parse_args()
    models = _load_models_dict(args)
    continuous_channel_indices = _parse_channel_indices(
        args.continuous_channel_indices,
        default=DEFAULT_CONTINUOUS_CHANNELS,
    )
    binary_groups = _resolve_binary_groups(args.binary_group)

    user_df, long_df, wide_df = build_grouped_metric_rank_tables(
        models=models,
        continuous_metrics=list(args.continuous_metrics),
        binary_metrics=list(args.binary_metrics),
        continuous_channel_indices=continuous_channel_indices,
        binary_groups=binary_groups,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    user_path = output_dir / f"{args.output_prefix}_user_level_long.csv"
    long_path = output_dir / f"{args.output_prefix}_long.csv"
    wide_path = output_dir / f"{args.output_prefix}_wide.csv"
    user_df.to_csv(user_path, index=False)
    long_df.to_csv(long_path, index=False)
    wide_df.to_csv(wide_path, index=False)

    print("=== Forecasting grouped metric/rank summary ===")
    if long_df.empty:
        print("(empty)")
    else:
        print(long_df.to_string(index=False))
    print(f"\nSaved user-level table: {user_path}")
    print(f"Saved long table: {long_path}")
    print(f"Saved wide table: {wide_path}")
    print(f"Continuous channels: {continuous_channel_indices}")
    print(f"Continuous metrics: {args.continuous_metrics}")
    print(f"Binary groups: {binary_groups}")
    print(f"Binary metrics: {args.binary_metrics}")


if __name__ == "__main__":
    main()
