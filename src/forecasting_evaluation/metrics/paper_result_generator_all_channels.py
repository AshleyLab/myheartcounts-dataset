"""Generate paper-ready grouped forecasting results across channel collections.

This module reads offline metrics parquet outputs (e.g. under
``results/metrics/<model_name>/<metric_name>``) and exports one combined table
that:

1. Reports aggregate results for three channel collections:
   - channels 0-6 using ``mase`` and ``sql``
   - channels 7-8 using ``mse`` and ``f1``
   - channels 9-18 using ``mse`` and ``f1``
2. Then reports each individual channel using the metric family assigned to its
   collection. For channels 0-6, the individual-channel rows use ``mae`` and
   ``mase`` rather than ``sql``.

Rows keep the same 3-hour grouped layout as the one-channel paper export while
adding a leading ``Channel`` column to identify the aggregate collection or
individual channel.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from forecasting_evaluation.metrics.paper_result_generator_one_channel import (
    aggregate_metric_3hour as _aggregate_metric_3hour_one_channel,
)
from forecasting_evaluation.metrics.paper_result_generator_one_channel import (
    rank_metric_by_user_3hour as _rank_metric_by_user_3hour_one_channel,
)

CHANNEL_NAME_MAP = {
    "first": ["hk_iphone:HKQuantityTypeIdentifierStepCount"],
    "iPhone": [
        "hk_iphone:HKQuantityTypeIdentifierStepCount",
        "hk_iphone:HKQuantityTypeIdentifierDistanceWalkingRunning",
        "hk_iphone:HKQuantityTypeIdentifierFlightsClimbed",
    ],
    "watch": [
        "hk_watch:HKQuantityTypeIdentifierStepCount",
        "hk_watch:HKQuantityTypeIdentifierDistanceWalkingRunning",
        "hk_watch:HKQuantityTypeIdentifierHeartRate",
        "hk_watch:HKQuantityTypeIdentifierActiveEnergyBurned",
    ],
    "all": [
        "hk_iphone:HKQuantityTypeIdentifierStepCount",
        "hk_iphone:HKQuantityTypeIdentifierDistanceWalkingRunning",
        "hk_iphone:HKQuantityTypeIdentifierFlightsClimbed",
        "hk_watch:HKQuantityTypeIdentifierStepCount",
        "hk_watch:HKQuantityTypeIdentifierDistanceWalkingRunning",
        "hk_watch:HKQuantityTypeIdentifierHeartRate",
        "hk_watch:HKQuantityTypeIdentifierActiveEnergyBurned",
        "sleep:asleep",
        "sleep:inbed",
        "workout:HKWorkoutActivityTypeWalking",
        "workout:HKWorkoutActivityTypeCycling",
        "workout:HKWorkoutActivityTypeRunning",
        "workout:HKWorkoutActivityTypeOther",
        "workout:HKWorkoutActivityTypeMixedMetabolicCardioTraining",
        "workout:HKWorkoutActivityTypeTraditionalStrengthTraining",
        "workout:HKWorkoutActivityTypeElliptical",
        "workout:HKWorkoutActivityTypeHighIntensityIntervalTraining",
        "workout:HKWorkoutActivityTypeFunctionalStrengthTraining",
        "workout:HKWorkoutActivityTypeYoga",
    ],
}


@dataclass(frozen=True)
class ScopeSpec:
    """Definition for one aggregate or per-channel reporting scope."""

    label: str
    channel_indices: tuple[int, ...]
    metric_columns: tuple[str, ...]


def _safe_read_parquet(file_path: str | Path, **kwargs: Any) -> pd.DataFrame | None:
    path = Path(file_path)
    try:
        if not path.exists() or path.stat().st_size == 0:
            return None
        return pd.read_parquet(path, **kwargs)
    except Exception:
        return None


def _list_parquet_files(model_dir: str | Path) -> list[Path]:
    path = Path(model_dir)
    if not path.exists():
        return []
    return sorted(path.rglob("*.parquet"))


def _resolve_metric_dir(model_dir: str | Path, metric_name: str) -> Path:
    """Resolve one model root to the corresponding metric directory."""
    path = Path(model_dir)
    metric_dir = path / metric_name
    return metric_dir


def _safe_to_2d_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None

    try:
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 2:
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


def _collect_users_for_model(model_dir: str | Path, metric_column: str) -> set[str]:
    resolved_dir = _resolve_metric_dir(model_dir, metric_column)
    if not resolved_dir.is_dir():
        return set()
    users: set[str] = set()
    for parquet_file in _list_parquet_files(resolved_dir):
        df = _safe_read_parquet(parquet_file, columns=["user_id"])
        if df is None or "user_id" not in df.columns:
            continue
        users.update(df["user_id"].astype(str).dropna().unique().tolist())
    return users


def _project_hour_index(hour: int, horizon: int) -> int:
    if horizon >= 48:
        return int(hour % 24)
    return int(hour)


def _group_start_from_hour(hour: int) -> int:
    return int((hour // 3) * 3)


def _group_label(group_start: int) -> str:
    return f"{group_start:02d}-{group_start + 2:02d}"


def _format_group_label_for_paper(group_label: str) -> str:
    start_str, end_str = group_label.split("-")
    return f"{int(start_str)}-{int(end_str)}"


def _read_config_channel(model_dir: str | Path) -> str | None:
    root = Path(model_dir)
    candidates = [root / "config.yaml", root.parent / "config.yaml"]
    config_path = next((p for p in candidates if p.exists()), None)
    if config_path is None:
        return None

    try:
        import yaml
    except ImportError:
        return None

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    return (config.get("features") or {}).get("channel")


def _infer_channel_names(model_dir: str | Path, n_features: int) -> list[str]:
    channel_key = _read_config_channel(model_dir)
    if channel_key is None:
        return [f"feature_{i}" for i in range(n_features)]

    mapped = CHANNEL_NAME_MAP.get(channel_key)
    if not mapped:
        return [f"feature_{i}" for i in range(n_features)]

    if len(mapped) >= n_features:
        return mapped[:n_features]
    return mapped + [f"feature_{i}" for i in range(len(mapped), n_features)]


def _mean_matrices_ignore_nan(mats: list[np.ndarray]) -> np.ndarray | None:
    if not mats:
        return None

    min_features = min(m.shape[0] for m in mats)
    min_horizon = min(m.shape[1] for m in mats)
    if min_features <= 0 or min_horizon <= 0:
        return None

    trimmed = [m[:min_features, :min_horizon] for m in mats]
    stack = np.stack(trimmed, axis=0)
    finite_mask = np.isfinite(stack)
    counts = finite_mask.sum(axis=0)
    sums = np.where(finite_mask, stack, 0.0).sum(axis=0)
    mean = np.full((min_features, min_horizon), np.nan, dtype=float)
    valid = counts > 0
    mean[valid] = sums[valid] / counts[valid]
    return mean


def _load_per_model_user_metric(
    model_dir: str | Path,
    metric_column: str,
    allowed_users: set[str] | None = None,
) -> dict[str, np.ndarray]:
    resolved_dir = _resolve_metric_dir(model_dir, metric_column)
    if not resolved_dir.is_dir():
        return {}
    per_user_rows: dict[str, list[np.ndarray]] = {}

    for parquet_file in _list_parquet_files(resolved_dir):
        df = _safe_read_parquet(parquet_file, columns=["user_id", metric_column])
        if df is None or "user_id" not in df.columns or metric_column not in df.columns:
            continue

        for _, row in df.iterrows():
            user = str(row.get("user_id"))
            if allowed_users is not None and user not in allowed_users:
                continue
            metric = _safe_to_2d_array(row.get(metric_column))
            if metric is None:
                continue
            per_user_rows.setdefault(user, [])
            per_user_rows[user].append(metric)

    per_user_avg: dict[str, np.ndarray] = {}
    for user, rows in per_user_rows.items():
        mean_metric = _mean_matrices_ignore_nan(rows)
        if mean_metric is not None:
            per_user_avg[user] = mean_metric
    return per_user_avg


def _metric_label_from_column(metric_column: str) -> str:
    normalized = metric_column.strip().lower()
    if normalized == "mae":
        return "MAE"
    if normalized == "mse":
        return "MSE"
    if normalized == "f1":
        return "F1"
    if normalized == "mase":
        return "MASE"
    if normalized == "ql":
        return "QL"
    if normalized == "sql":
        return "sQL"
    return metric_column


def _rank_label_from_metric_label(metric_label: str) -> str:
    return f"Rank_{metric_label}"


def _format_output_value(value: Any) -> str:
    if isinstance(value, str):
        return value

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "/"

    if not np.isfinite(numeric):
        return "/"
    return f"{numeric:.2f}"


def _render_plaintext_table(df: pd.DataFrame) -> str:
    columns = df.columns.tolist()
    str_df = df.copy()
    for column in columns[3:]:
        str_df[column] = str_df[column].map(_format_output_value)

    widths: dict[str, int] = {}
    for column in columns:
        widths[column] = max(len(str(column)), str_df[column].astype(str).map(len).max())

    header = " | ".join(str(column).ljust(widths[column]) for column in columns)
    divider = "-+-".join("-" * widths[column] for column in columns)
    body = [
        " | ".join(str(row[column]).ljust(widths[column]) for column in columns)
        for _, row in str_df.iterrows()
    ]
    return "\n".join([header, divider, *body])


def _render_latex_rows(df: pd.DataFrame) -> str:
    value_columns = df.columns.tolist()[3:]
    lines: list[str] = []
    for _, row in df.iterrows():
        values = " & ".join(_format_output_value(row[column]) for column in value_columns)
        lines.append(f"{row['Channel']} & {row['Model']} & {row['Metric']} & {values} \\\\")
    return "\n".join(lines)


def _load_models_dict(args: argparse.Namespace) -> dict[str, str]:
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
    if not isinstance(parsed, dict):
        raise ValueError("Model configuration must be a dict of {model_name: metrics_dir}")

    models = {str(k): str(v) for k, v in parsed.items()}
    if not models:
        raise ValueError("No model mappings found in configuration")
    return models


def _build_scope_specs(channel_names: list[str]) -> list[ScopeSpec]:
    specs: list[ScopeSpec] = [
        ScopeSpec(label="AVG[0-6]", channel_indices=tuple(range(0, 7)), metric_columns=("mase", "sql")),
        ScopeSpec(label="AVG[7-8]", channel_indices=(7, 8), metric_columns=("mse", "f1")),
        ScopeSpec(label="AVG[9-18]", channel_indices=tuple(range(9, 19)), metric_columns=("mse", "f1")),
    ]

    for channel_idx in range(len(channel_names)):
        if channel_idx <= 6:
            metric_columns = ("mae", "mase")
        else:
            metric_columns = ("mse", "f1")
        specs.append(
            ScopeSpec(
                label=channel_names[channel_idx],
                channel_indices=(channel_idx,),
                metric_columns=metric_columns,
            )
        )
    return specs


def _aggregate_scope_results(
    scope_specs: list[ScopeSpec],
    channel_names: list[str],
    base_metric_results: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[pd.DataFrame] = []
    rank_rows: list[pd.DataFrame] = []

    for scope in scope_specs:
        selected_channel_names = [
            channel_names[idx]
            for idx in scope.channel_indices
            if 0 <= idx < len(channel_names)
        ]
        if not selected_channel_names:
            continue

        for metric_column in scope.metric_columns:
            metric_label = _metric_label_from_column(metric_column)
            metric_df, rank_df = base_metric_results.get(
                metric_column,
                (pd.DataFrame(), pd.DataFrame()),
            )

            metric_slice = pd.DataFrame()
            if "channel" in metric_df.columns:
                metric_slice = metric_df.loc[metric_df["channel"].isin(selected_channel_names)].copy()
            if not metric_slice.empty:
                grouped_metric = (
                    metric_slice.groupby(["model", "group_start", "group_label"], as_index=False)["metric_mean"]
                    .mean()
                )
                grouped_metric["channel"] = scope.label
                grouped_metric["metric"] = metric_label
                metric_rows.append(
                    grouped_metric[["channel", "model", "metric", "group_start", "group_label", "metric_mean"]]
                )

            rank_slice = pd.DataFrame()
            if "channel" in rank_df.columns:
                rank_slice = rank_df.loc[rank_df["channel"].isin(selected_channel_names)].copy()
            if not rank_slice.empty:
                grouped_rank = (
                    rank_slice.groupby(["model", "group_start", "group_label"], as_index=False)["mean_rank"]
                    .mean()
                )
                grouped_rank["channel"] = scope.label
                grouped_rank["metric"] = metric_label
                rank_rows.append(
                    grouped_rank[["channel", "model", "metric", "group_start", "group_label", "mean_rank"]]
                )

    if metric_rows:
        metric_means = pd.concat(metric_rows, ignore_index=True)
    else:
        metric_means = pd.DataFrame(
            columns=["channel", "model", "metric", "group_start", "group_label", "metric_mean"]
        )

    if rank_rows:
        rank_means = pd.concat(rank_rows, ignore_index=True)
    else:
        rank_means = pd.DataFrame(
            columns=["channel", "model", "metric", "group_start", "group_label", "mean_rank"]
        )

    return metric_means, rank_means


def _compute_scope_user_metric_rows(
    models: dict[str, str],
    scope_specs: list[ScopeSpec],
    metric_data: dict[str, dict[str, dict[str, np.ndarray]]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for scope in scope_specs:
        for metric_column in scope.metric_columns:
            per_model_user_metric = metric_data.get(metric_column, {})
            display_metric = _metric_label_from_column(metric_column)

            for model_name in models:
                user_map = per_model_user_metric.get(model_name, {})
                for user_id, arr in user_map.items():
                    selected_indices = [idx for idx in scope.channel_indices if idx < arr.shape[0]]
                    if not selected_indices:
                        continue

                    _, horizon = arr.shape
                    group_hours: dict[int, list[int]] = {}
                    for hour in range(horizon):
                        summary_hour = _project_hour_index(hour=hour, horizon=horizon)
                        group_start = _group_start_from_hour(summary_hour)
                        group_hours.setdefault(group_start, []).append(hour)

                    for group_start, hours in sorted(group_hours.items()):
                        group_values = arr[np.asarray(selected_indices, dtype=int)][:, hours]
                        finite_values = group_values[np.isfinite(group_values)]
                        if finite_values.size == 0:
                            continue

                        rows.append(
                            {
                                "channel": scope.label,
                                "model": model_name,
                                "user_id": user_id,
                                "metric": display_metric,
                                "group_start": int(group_start),
                                "group_label": _group_label(int(group_start)),
                                "metric_value": float(np.mean(finite_values)),
                            }
                        )

    if not rows:
        return pd.DataFrame(
            columns=["channel", "model", "user_id", "metric", "group_start", "group_label", "metric_value"]
        )
    return pd.DataFrame(rows)


def _compute_model_group_means(user_metric_df: pd.DataFrame) -> pd.DataFrame:
    if user_metric_df.empty:
        return pd.DataFrame(
            columns=["channel", "model", "metric", "group_start", "group_label", "metric_mean"]
        )

    out = (
        user_metric_df.groupby(["channel", "model", "metric", "group_start", "group_label"], as_index=False)["metric_value"]
        .mean()
        .rename(columns={"metric_value": "metric_mean"})
    )
    return out


def _compute_model_group_ranks(user_metric_df: pd.DataFrame) -> pd.DataFrame:
    if user_metric_df.empty:
        return pd.DataFrame(
            columns=["channel", "model", "metric", "group_start", "group_label", "mean_rank"]
        )

    rank_rows: list[pd.DataFrame] = []

    for (channel_label, metric_name, group_start, group_label), group_slice in user_metric_df.groupby(
        ["channel", "metric", "group_start", "group_label"],
        sort=True,
    ):
        pivot = group_slice.pivot(index="user_id", columns="model", values="metric_value")
        if pivot.empty:
            continue

        rank_df = pivot.rank(axis=1, method="average", ascending=True)
        long_rank = rank_df.stack(dropna=True).reset_index()
        long_rank.columns = ["user_id", "model", "rank"]
        long_rank["channel"] = channel_label
        long_rank["metric"] = metric_name
        long_rank["group_start"] = int(group_start)
        long_rank["group_label"] = group_label
        rank_rows.append(long_rank)

    if not rank_rows:
        return pd.DataFrame(
            columns=["channel", "model", "metric", "group_start", "group_label", "mean_rank"]
        )

    rank_all = pd.concat(rank_rows, ignore_index=True)
    out = (
        rank_all.groupby(["channel", "model", "metric", "group_start", "group_label"], as_index=False)["rank"]
        .mean()
        .rename(columns={"rank": "mean_rank"})
    )
    return out


def _build_combined_table(
    metric_means: pd.DataFrame,
    rank_means: pd.DataFrame,
    scope_order: list[str],
    model_order: list[str],
    scope_metric_order: dict[str, list[str]],
) -> pd.DataFrame:
    all_group_starts = sorted(
        {
            int(group_start)
            for df in (metric_means, rank_means)
            for group_start in df["group_start"].tolist()
        }
    )
    if not all_group_starts:
        raise ValueError("No grouped metric rows were produced.")

    group_labels: dict[int, str] = {}
    for group_start in all_group_starts:
        matched = metric_means.loc[metric_means["group_start"] == group_start, "group_label"]
        if matched.empty:
            matched = rank_means.loc[rank_means["group_start"] == group_start, "group_label"]
        if matched.empty:
            group_labels[group_start] = _format_group_label_for_paper(_group_label(group_start))
        else:
            group_labels[group_start] = _format_group_label_for_paper(str(matched.iloc[0]))

    rows: list[dict[str, Any]] = []
    for scope_label in scope_order:
        metric_order = scope_metric_order.get(scope_label, [])
        for model_name in model_order:
            metric_slice = metric_means.loc[
                (metric_means["channel"] == scope_label) & (metric_means["model"] == model_name)
            ]
            rank_slice = rank_means.loc[
                (rank_means["channel"] == scope_label) & (rank_means["model"] == model_name)
            ]

            for metric_name in metric_order:
                metric_row = {"Channel": scope_label, "Model": model_name, "Metric": metric_name}
                for group_start in all_group_starts:
                    column_name = group_labels[group_start]
                    matched = metric_slice.loc[
                        (metric_slice["metric"] == metric_name) & (metric_slice["group_start"] == group_start),
                        "metric_mean",
                    ]
                    if matched.empty:
                        metric_row[column_name] = "/"
                    else:
                        value = float(matched.iloc[0])
                        metric_row[column_name] = round(value, 2) if np.isfinite(value) else "/"
                rows.append(metric_row)

            for metric_name in metric_order:
                rank_metric_name = _rank_label_from_metric_label(metric_name)
                rank_row = {"Channel": scope_label, "Model": model_name, "Metric": rank_metric_name}
                for group_start in all_group_starts:
                    column_name = group_labels[group_start]
                    matched = rank_slice.loc[
                        (rank_slice["metric"] == metric_name) & (rank_slice["group_start"] == group_start),
                        "mean_rank",
                    ]
                    if matched.empty:
                        rank_row[column_name] = "/"
                    else:
                        value = float(matched.iloc[0])
                        rank_row[column_name] = round(value, 2) if np.isfinite(value) else "/"
                rows.append(rank_row)

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for all-channel paper summaries."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate paper-ready grouped metric aggregation and user-level ranking "
            "across aggregate channel collections plus individual channels."
        )
    )
    parser.add_argument(
        "--models-json",
        default=None,
        help='Inline JSON dict, e.g. {"modelA":"/path/a","modelB":"/path/b"}',
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to JSON/YAML config with model dict (top-level dict or {models: {...}}).",
    )
    parser.add_argument(
        "--output-dir",
        default="results/metrics_summary",
        help="Output directory for generated paper CSV files.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help=(
            "Optional output file path. Defaults to "
            "<output-dir>/paper_result_table_all_channels_grouped.csv."
        ),
    )
    parser.add_argument(
        "--max-user",
        type=int,
        default=None,
        help="Optional cap for users after computing model user intersection.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used when --max-user is provided.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate all-channel paper result summaries."""
    args = parse_args()
    models = _load_models_dict(args)
    user_probe_metric = "mase"

    users_per_model = {
        name: _collect_users_for_model(path, metric_column=user_probe_metric)
        for name, path in models.items()
    }
    all_users: set[str] = set()
    for users in users_per_model.values():
        all_users.update(users)

    if not all_users:
        raise ValueError("No users found across the provided models.")

    selected_users = all_users
    if args.max_user is not None and args.max_user > 0 and len(all_users) > args.max_user:
        rng = random.Random(args.random_seed)
        selected_users = set(rng.sample(sorted(all_users), k=args.max_user))

    first_model_dir = next(iter(models.values()))
    channel_names = _infer_channel_names(first_model_dir, n_features=19)
    scope_specs = _build_scope_specs(channel_names)
    scope_order = [scope.label for scope in scope_specs]
    scope_metric_order = {
        scope.label: [_metric_label_from_column(metric_column) for metric_column in scope.metric_columns]
        for scope in scope_specs
    }

    metric_columns_needed = sorted({metric for scope in scope_specs for metric in scope.metric_columns})
    metric_data: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for metric_column in metric_columns_needed:
        metric_data[metric_column] = {
            model_name: _load_per_model_user_metric(
                model_dir=model_dir,
                metric_column=metric_column,
                allowed_users=selected_users,
            )
            for model_name, model_dir in models.items()
        }

    base_metric_results: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for metric_column, per_model_user_metric in metric_data.items():
        base_metric_results[metric_column] = (
            _aggregate_metric_3hour_one_channel(
                models=models,
                per_model_user_metric=per_model_user_metric,
            ),
            _rank_metric_by_user_3hour_one_channel(
                models=models,
                per_model_user_metric=per_model_user_metric,
            ),
        )

    metric_means, rank_means = _aggregate_scope_results(
        scope_specs=scope_specs,
        channel_names=channel_names,
        base_metric_results=base_metric_results,
    )
    paper_table_df = _build_combined_table(
        metric_means=metric_means,
        rank_means=rank_means,
        scope_order=scope_order,
        model_order=list(models.keys()),
        scope_metric_order=scope_metric_order,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = output_dir / "paper_result_table_all_channels_grouped.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    paper_table_df.to_csv(output_path, index=False)

    print("=== Paper result table (all channels grouped) ===")
    print(f"output: {output_path}")
    print(_render_plaintext_table(paper_table_df))
    print("=== LaTeX rows ===")
    print(_render_latex_rows(paper_table_df))


if __name__ == "__main__":
    main()
