"""Generate paper-ready forecasting results from offline metrics parquet files.

This module builds one combined table from offline metrics produced by
`src/forecasting_evaluation/metrics/offline_calculate.py`. For each requested
metric column, it computes:
1. 3-hour grouped metric aggregation per model/channel.
2. User-level average rank per model/channel/3-hour group.

The final output contains all requested metrics together so that cross-metric
comparison is possible in a single table.

Model inputs are provided as a dict mapping:
    {"model_name": "/path/to/model_metrics_root", ...}
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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


def _parse_metric_matrix(raw_value: Any) -> np.ndarray | None:
    return _safe_to_2d_array(raw_value)


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
    # For horizons >= 48, collapse to hour-of-day to align with prior summaries.
    if horizon >= 48:
        return int(hour % 24)
    return int(hour)


def _group_start_from_hour(hour: int) -> int:
    return int((hour // 3) * 3)


def _group_label(group_start: int) -> str:
    return f"{group_start:02d}-{group_start + 2:02d}"


def _normalize_model_display_name(model_name: str) -> str:
    return model_name


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

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
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


def _mean_matrices(mats: list[np.ndarray]) -> np.ndarray | None:
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
            metric = _parse_metric_matrix(row.get(metric_column))
            if metric is None:
                continue
            per_user_rows.setdefault(user, [])
            per_user_rows[user].append(metric)

    per_user_avg: dict[str, np.ndarray] = {}
    for user, rows in per_user_rows.items():
        mean_metric = _mean_matrices(rows)
        if mean_metric is not None:
            per_user_avg[user] = mean_metric
    return per_user_avg


def aggregate_metric_3hour(
    models: dict[str, str],
    per_model_user_metric: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    """Aggregate per-user metric matrices into 3-hour model/channel summaries."""
    acc: dict[tuple[str, str, int], dict[str, float | int]] = {}

    for model_name, user_map in per_model_user_metric.items():
        if not user_map:
            continue

        min_features = min(arr.shape[0] for arr in user_map.values())
        channels = _infer_channel_names(models[model_name], min_features)

        for arr in user_map.values():
            arr = arr[:min_features, :]
            n_features, horizon = arr.shape

            for feature_idx in range(n_features):
                channel = channels[feature_idx] if feature_idx < len(channels) else f"feature_{feature_idx}"
                for hour in range(horizon):
                    value = float(arr[feature_idx, hour])
                    if not np.isfinite(value):
                        continue
                    summary_hour = _project_hour_index(hour=hour, horizon=horizon)
                    group_start = _group_start_from_hour(summary_hour)
                    key = (model_name, channel, group_start)
                    if key not in acc:
                        acc[key] = {"sum": 0.0, "sum_sq": 0.0, "n": 0}
                    acc[key]["sum"] = float(acc[key]["sum"]) + value
                    acc[key]["sum_sq"] = float(acc[key]["sum_sq"]) + value * value
                    acc[key]["n"] = int(acc[key]["n"]) + 1

    rows: list[dict[str, Any]] = []
    for (model_name, channel, group_start), stats in acc.items():
        n = int(stats["n"])
        if n <= 0:
            continue
        mean = float(stats["sum"]) / n
        var = max(0.0, float(stats["sum_sq"]) / n - mean * mean)
        rows.append(
            {
                "model": model_name,
                "channel": channel,
                "group_start": int(group_start),
                "group_label": _group_label(int(group_start)),
                "metric_mean": float(mean),
                "metric_std": float(np.sqrt(var)),
                "n": n,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["model", "channel", "group_start", "group_label", "metric_mean", "metric_std", "n"]
        )
    return pd.DataFrame(rows).sort_values(["model", "channel", "group_start"]).reset_index(
        drop=True
    )


def rank_metric_by_user_3hour(
    models: dict[str, str],
    per_model_user_metric: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    """Rank models per user within each 3-hour group and average the ranks."""
    if not per_model_user_metric:
        return pd.DataFrame(
            columns=["model", "channel", "group_start", "group_label", "mean_rank", "n_users"]
        )

    model_names = [model_name for model_name in models if per_model_user_metric.get(model_name)]
    if not model_names:
        return pd.DataFrame(
            columns=["model", "channel", "group_start", "group_label", "mean_rank", "n_users"]
        )

    all_users: set[str] = set()
    for model_name in model_names:
        all_users.update(per_model_user_metric.get(model_name, {}).keys())

    if not all_users:
        return pd.DataFrame(
            columns=["model", "channel", "group_start", "group_label", "mean_rank", "n_users"]
        )

    # Use first available model for channel naming, then trim by the feature count among
    # models that have at least one user.
    first_model_name = model_names[0]
    n_features_common = min(
        min(arr.shape[0] for arr in per_model_user_metric[m].values())
        for m in model_names
        if per_model_user_metric.get(m)
    )
    channels = _infer_channel_names(models[first_model_name], n_features_common)

    rank_acc: dict[tuple[str, str, int], list[float]] = {}

    for user in sorted(all_users):
        model_arrs: dict[str, np.ndarray] = {
            m: per_model_user_metric[m][user]
            for m in model_names
            if user in per_model_user_metric.get(m, {})
        }
        if len(model_arrs) < 2:
            continue

        group_hours_per_model: dict[str, dict[int, list[int]]] = {}
        for model_name, arr in model_arrs.items():
            arr = arr[:n_features_common, :]
            model_arrs[model_name] = arr
            _, horizon = arr.shape
            group_hours: dict[int, list[int]] = {}
            for hour in range(horizon):
                summary_hour = _project_hour_index(hour=hour, horizon=horizon)
                group_start = _group_start_from_hour(summary_hour)
                group_hours.setdefault(group_start, []).append(hour)
            group_hours_per_model[model_name] = group_hours
        all_groups = sorted({group for groups in group_hours_per_model.values() for group in groups})

        for feature_idx in range(n_features_common):
            channel = channels[feature_idx] if feature_idx < len(channels) else f"feature_{feature_idx}"
            for group_start in all_groups:
                model_scores: list[float] = []
                model_order: list[str] = []
                for model_name in model_names:
                    if model_name not in model_arrs:
                        continue
                    arr = model_arrs[model_name]
                    hours = group_hours_per_model[model_name].get(group_start, [])
                    if not hours:
                        continue
                    group_values = arr[feature_idx, hours]
                    if group_values.size == 0:
                        continue
                    if not np.isfinite(group_values).any():
                        continue
                    score = float(np.nanmean(group_values))
                    if not np.isfinite(score):
                        continue
                    model_scores.append(score)
                    model_order.append(model_name)

                if len(model_scores) < 2:
                    continue

                # Lower metric is better: rank 1 is best.
                ranks = pd.Series(model_scores, index=model_order).rank(
                    method="average", ascending=True
                )
                for model_name, rank_value in ranks.items():
                    key = (model_name, channel, group_start)
                    rank_acc.setdefault(key, []).append(float(rank_value))

    rows: list[dict[str, Any]] = []
    for (model_name, channel, group_start), values in rank_acc.items():
        if not values:
            continue
        rows.append(
            {
                "model": model_name,
                "channel": channel,
                "group_start": int(group_start),
                "group_label": _group_label(int(group_start)),
                "mean_rank": float(np.mean(values)),
                "n_users": int(len(values)),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["model", "channel", "group_start", "group_label", "mean_rank", "n_users"]
        )
    return pd.DataFrame(rows).sort_values(["model", "channel", "group_start"]).reset_index(
        drop=True
    )


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
            with config_path.open("r", encoding="utf-8") as f:
                parsed = yaml.safe_load(f)
        else:
            with config_path.open("r", encoding="utf-8") as f:
                parsed = json.load(f)
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


def _build_combined_paper_ready_table(
    metric_results: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    channel: str,
    model_order: list[str],
) -> pd.DataFrame:
    """Build one paper table that contains all requested metrics and their ranks."""
    if not metric_results:
        raise ValueError("No metric results were provided for paper table construction")

    group_order = sorted(
        {
            int(group_start)
            for metric_3h_df, rank_df in metric_results.values()
            for df in (metric_3h_df, rank_df)
            if not df.empty
            for group_start in df.loc[df["channel"] == channel, "group_start"].tolist()
        }
    )
    if not group_order:
        raise ValueError(f"Channel not found in any metric results: {channel}")

    group_labels: dict[int, str] = {}
    for group_start in group_order:
        for metric_3h_df, rank_df in metric_results.values():
            for df in (metric_3h_df, rank_df):
                matched = df.loc[
                    (df["channel"] == channel) & (df["group_start"] == group_start),
                    "group_label",
                ]
                if not matched.empty:
                    group_labels[group_start] = _format_group_label_for_paper(matched.iloc[0])
                    break
            if group_start in group_labels:
                break

    rows: list[dict[str, Any]] = []
    for model_name in model_order:
        display_model_name = _normalize_model_display_name(model_name)

        for metric_column, (metric_3h_df, _) in metric_results.items():
            metric_label = _metric_label_from_column(metric_column)
            metric_filtered = metric_3h_df.loc[metric_3h_df["channel"] == channel].copy()
            metric_pivot = (
                metric_filtered.assign(model=lambda df: df["model"].map(_normalize_model_display_name))
                .pivot(index="model", columns="group_start", values="metric_mean")
                .reindex(index=[display_model_name], columns=group_order)
            )

            metric_row = {"Model": display_model_name, "Metric": metric_label}
            for group_start in group_order:
                column_name = group_labels[group_start]
                value = metric_pivot.loc[display_model_name, group_start]
                metric_row[column_name] = (
                    round(float(value), 2) if pd.notna(value) and np.isfinite(float(value)) else "/"
                )
            rows.append(metric_row)

        for metric_column, (_, rank_df) in metric_results.items():
            metric_label = _metric_label_from_column(metric_column)
            rank_filtered = rank_df.loc[rank_df["channel"] == channel].copy()
            rank_pivot = (
                rank_filtered.assign(model=lambda df: df["model"].map(_normalize_model_display_name))
                .pivot(index="model", columns="group_start", values="mean_rank")
                .reindex(index=[display_model_name], columns=group_order)
            )

            rank_row = {
                "Model": display_model_name,
                "Metric": _rank_label_from_metric_label(metric_label),
            }
            for group_start in group_order:
                column_name = group_labels[group_start]
                value = rank_pivot.loc[display_model_name, group_start]
                rank_row[column_name] = (
                    round(float(value), 2) if pd.notna(value) and np.isfinite(float(value)) else "/"
                )
            rows.append(rank_row)

    return pd.DataFrame(rows)


def _format_output_value(value: Any) -> str:
    """Format exported table values, using '/' for missing entries."""
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
    for column in columns[2:]:
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
    value_columns = df.columns.tolist()[2:]
    lines: list[str] = []
    for _, row in df.iterrows():
        values = " & ".join(_format_output_value(row[column]) for column in value_columns)
        lines.append(f"{row['Model']} & {row['Metric']} & {values} \\\\")
    return "\n".join(lines)


def _metric_label_from_column(metric_column: str) -> str:
    """Map parquet metric column names to paper-facing display labels."""
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
    """Build the row label used for metric-specific ranking rows."""
    return f"Rank_{metric_label}"


def _parse_metric_columns_arg(raw_metric_columns: list[list[str]] | None) -> list[str]:
    """Flatten and deduplicate requested metric columns while preserving order."""
    if not raw_metric_columns:
        return ["mae"]

    parsed: list[str] = []
    seen: set[str] = set()
    for group in raw_metric_columns:
        for metric in group:
            normalized = metric.strip()
            if not normalized or normalized in seen:
                continue
            parsed.append(normalized)
            seen.add(normalized)
    return parsed or ["mae"]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for paper-ready 3-hour metric aggregation."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate paper-ready 3-hour metric aggregation and user-level model ranking."
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
        "--metric-column",
        action="append",
        nargs="+",
        default=None,
        help=(
            "One or more metric column names in parquet rows. "
            "Example: --metric-column mae mase ql"
        ),
    )
    parser.add_argument(
        "--channel",
        default="hk_iphone:HKQuantityTypeIdentifierStepCount",
        help="Channel name to export for the paper table.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help=(
            "Optional output file path. Defaults to "
            "<output-dir>/paper_result_table_<sanitized_channel>.csv."
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
    """Load model mappings, compute summaries, and write one paper-ready table."""
    args = parse_args()
    models = _load_models_dict(args)
    metric_columns = _parse_metric_columns_arg(args.metric_column)

    primary_metric = metric_columns[0]
    users_per_model = {
        name: _collect_users_for_model(path, metric_column=primary_metric)
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

    metric_results: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for metric_column in metric_columns:
        per_model_user_metric = {
            model_name: _load_per_model_user_metric(
                model_dir=model_dir,
                metric_column=metric_column,
                allowed_users=selected_users,
            )
            for model_name, model_dir in models.items()
        }

        metric_3h_df = aggregate_metric_3hour(
            models=models,
            per_model_user_metric=per_model_user_metric,
        )
        rank_df = rank_metric_by_user_3hour(
            models=models,
            per_model_user_metric=per_model_user_metric,
        )
        metric_results[metric_column] = (metric_3h_df, rank_df)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_table_df = _build_combined_paper_ready_table(
        metric_results=metric_results,
        channel=args.channel,
        model_order=list(models.keys()),
    )

    if args.output_file:
        output_path = Path(args.output_file)
    else:
        safe_channel = args.channel.replace(":", "_").replace("/", "_")
        metric_suffix = "_".join(metric_columns)
        output_path = output_dir / f"paper_result_table_{metric_suffix}_{safe_channel}.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    paper_table_df.to_csv(output_path, index=False)

    print("=== Paper result table ===")
    print(f"channel: {args.channel}")
    print(f"metrics: {metric_columns}")
    print(f"output: {output_path}")
    print(_render_plaintext_table(paper_table_df))
    print("=== LaTeX rows ===")
    print(_render_latex_rows(paper_table_df))


if __name__ == "__main__":
    main()
