"""Aggregate binary forecasting metrics into grouped sample-based summaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

_METRIC_NAMES = ("auprc", "auroc", "f1")
_DEFAULT_GROUPS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("sleep", (7, 8)),
    ("workout", (9, 10, 11, 12, 13, 14, 15, 16, 17, 18)),
)


def _metric_display_name(metric_name: str) -> str:
    normalized = metric_name.strip().lower()
    if normalized == "auprc":
        return "AUPRC"
    if normalized == "auroc":
        return "AUROC"
    if normalized == "f1":
        return "F1"
    return metric_name


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


def _safe_to_1d_float_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    return arr if arr.ndim == 1 else None


def _safe_to_1d_int_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    return arr.astype(int) if arr.ndim == 1 else None


def _parse_model_arg(model_arg: str) -> tuple[str, str]:
    if "=" not in model_arg:
        raise argparse.ArgumentTypeError(
            f"Invalid --model value: {model_arg}. Expected format: MODEL_NAME=/path/to/model_root"
        )
    model_name, model_dir = model_arg.split("=", 1)
    model_name = model_name.strip()
    model_dir = model_dir.strip()
    if not model_name or not model_dir:
        raise argparse.ArgumentTypeError(
            f"Invalid --model value: {model_arg}. Model name and path must be non-empty."
        )
    return model_name, model_dir


def _parse_group_arg(group_arg: str) -> tuple[str, tuple[int, ...]]:
    if "=" not in group_arg:
        raise argparse.ArgumentTypeError(
            f"Invalid --group value: {group_arg}. Expected format: GROUP_NAME=7,8,9"
        )
    group_name, raw_indices = group_arg.split("=", 1)
    group_name = group_name.strip()
    if not group_name:
        raise argparse.ArgumentTypeError(
            f"Invalid --group value: {group_arg}. Group name must be non-empty."
        )

    indices: list[int] = []
    for part in raw_indices.split(","):
        token = part.strip()
        if not token:
            continue
        indices.append(int(token))
    if not indices:
        raise argparse.ArgumentTypeError(
            f"Invalid --group value: {group_arg}. At least one channel index is required."
        )
    return group_name, tuple(indices)


def _resolve_groups(group_args: list[str] | None) -> list[tuple[str, tuple[int, ...]]]:
    if not group_args:
        return [(name, tuple(indices)) for name, indices in _DEFAULT_GROUPS]

    groups: list[tuple[str, tuple[int, ...]]] = []
    seen_names: set[str] = set()
    for item in group_args:
        group_name, indices = _parse_group_arg(item)
        if group_name in seen_names:
            raise ValueError(f"Duplicate group name: {group_name}")
        seen_names.add(group_name)
        groups.append((group_name, indices))
    return groups


def _sample_occurrences(df: pd.DataFrame) -> pd.Series:
    return df.groupby(["user_id", "history_length"]).cumcount()


def _load_sample_metric_rows(
    *,
    model_name: str,
    model_root: str | Path,
    groups: list[tuple[str, tuple[int, ...]]],
) -> pd.DataFrame:
    root = Path(model_root)
    f1_dir = root / "f1"
    if not _list_parquet_files(f1_dir):
        return pd.DataFrame()

    sample_values: dict[tuple[str, int, int, str, str], list[float]] = {}
    coverage: dict[tuple[str, int, int, str], dict[str, int]] = {}

    for metric_name in _METRIC_NAMES:
        metric_dir = root / metric_name
        for parquet_file in _list_parquet_files(metric_dir):
            df = _safe_read_parquet(
                parquet_file,
                columns=[
                    "user_id",
                    "history_length",
                    metric_name,
                    "binary_valid_count",
                    "binary_positive_count",
                    "binary_negative_count",
                ],
            )
            if (
                df is None
                or "user_id" not in df.columns
                or "history_length" not in df.columns
                or metric_name not in df.columns
            ):
                continue

            df = df.copy()
            df["sample_occurrence"] = _sample_occurrences(df)
            for row in df.itertuples(index=False):
                user_id = str(getattr(row, "user_id"))
                history_length = int(getattr(row, "history_length"))
                sample_index = int(getattr(row, "sample_occurrence"))
                metric_arr = _safe_to_1d_float_array(getattr(row, metric_name))
                valid_arr = _safe_to_1d_int_array(getattr(row, "binary_valid_count"))
                positive_arr = _safe_to_1d_int_array(getattr(row, "binary_positive_count"))
                negative_arr = _safe_to_1d_int_array(getattr(row, "binary_negative_count"))
                if (
                    metric_arr is None
                    or valid_arr is None
                    or positive_arr is None
                    or negative_arr is None
                ):
                    continue

                n_features = min(
                    metric_arr.shape[0],
                    valid_arr.shape[0],
                    positive_arr.shape[0],
                    negative_arr.shape[0],
                )
                for group_name, channel_indices in groups:
                    group_metric_values: list[float] = []
                    group_valid_count = 0
                    group_positive_count = 0
                    group_negative_count = 0
                    for channel_idx in channel_indices:
                        if channel_idx < 0 or channel_idx >= n_features:
                            continue
                        group_valid_count += int(valid_arr[channel_idx])
                        group_positive_count += int(positive_arr[channel_idx])
                        group_negative_count += int(negative_arr[channel_idx])
                        metric_value = float(metric_arr[channel_idx])
                        if np.isfinite(metric_value):
                            group_metric_values.append(metric_value)

                    cov_key = (user_id, history_length, sample_index, group_name)
                    if metric_name == "f1":
                        coverage[cov_key] = {
                            "valid_count": int(group_valid_count),
                            "positive_count": int(group_positive_count),
                            "negative_count": int(group_negative_count),
                        }

                    if group_metric_values:
                        sample_values[(user_id, history_length, sample_index, group_name, metric_name)] = (
                            group_metric_values
                        )

    rows: list[dict[str, Any]] = []
    for (user_id, history_length, sample_index, group_name), cov in coverage.items():
        if int(cov["positive_count"]) <= 0:
            continue
        for metric_name in _METRIC_NAMES:
            values = sample_values.get((user_id, history_length, sample_index, group_name, metric_name), [])
            metric_value = float(np.mean(values)) if values else float("nan")
            rows.append(
                {
                    "model": model_name,
                    "group_name": group_name,
                    "user_id": user_id,
                    "history_length": int(history_length),
                    "sample_index": int(sample_index),
                    "metric": metric_name,
                    "metric_display": _metric_display_name(metric_name),
                    "metric_value": metric_value,
                    "valid_count": int(cov["valid_count"]),
                    "positive_count": int(cov["positive_count"]),
                    "negative_count": int(cov["negative_count"]),
                }
            )

    return pd.DataFrame(rows)


def _build_user_metric_rows(sample_metric_df: pd.DataFrame) -> pd.DataFrame:
    if sample_metric_df.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "group_name",
                "user_id",
                "metric",
                "metric_display",
                "metric_value",
                "valid_count",
                "positive_count",
                "negative_count",
            ]
        )

    key_columns = ["model", "group_name", "user_id", "metric", "metric_display"]
    coverage_df = (
        sample_metric_df.groupby(key_columns, as_index=False)
        .agg(
            valid_count=("valid_count", "sum"),
            positive_count=("positive_count", "sum"),
            negative_count=("negative_count", "sum"),
        )
    )

    finite_df = sample_metric_df.loc[np.isfinite(sample_metric_df["metric_value"])].copy()
    if finite_df.empty:
        user_metric_df = coverage_df.copy()
        user_metric_df["metric_value"] = np.nan
        return user_metric_df[
            [
                "model",
                "group_name",
                "user_id",
                "metric",
                "metric_display",
                "metric_value",
                "valid_count",
                "positive_count",
                "negative_count",
            ]
        ]

    metric_df = (
        finite_df.groupby(key_columns, as_index=False)
        .agg(metric_value=("metric_value", "mean"))
    )
    user_metric_df = coverage_df.merge(metric_df, on=key_columns, how="left")
    return user_metric_df[
        [
            "model",
            "group_name",
            "user_id",
            "metric",
            "metric_display",
            "metric_value",
            "valid_count",
            "positive_count",
            "negative_count",
        ]
    ]


def _compute_model_mean_ranks(sample_metric_df: pd.DataFrame) -> pd.DataFrame:
    if sample_metric_df.empty:
        return pd.DataFrame(
            columns=["metric", "group_name", "model", "rank", "rank_n_samples"]
        )

    finite_df = sample_metric_df.loc[np.isfinite(sample_metric_df["metric_value"])].copy()
    if finite_df.empty:
        return pd.DataFrame(
            columns=["metric", "group_name", "model", "rank", "rank_n_samples"]
        )

    rank_rows: list[pd.DataFrame] = []
    for (metric_name, group_name), group_slice in finite_df.groupby(
        ["metric", "group_name"],
        sort=True,
    ):
        pivot = group_slice.pivot(
            index=["user_id", "history_length", "sample_index"],
            columns="model",
            values="metric_value",
        )
        if pivot.empty:
            continue

        rank_df = pivot.rank(axis=1, method="average", ascending=False)
        long_rank = rank_df.stack(future_stack=True).reset_index()
        long_rank.columns = ["user_id", "history_length", "sample_index", "model", "rank"]
        long_rank["metric"] = metric_name
        long_rank["group_name"] = group_name
        long_rank["sample_key"] = (
            long_rank["user_id"].astype(str)
            + "::"
            + long_rank["history_length"].astype(str)
            + "::"
            + long_rank["sample_index"].astype(str)
        )
        rank_rows.append(long_rank)

    if not rank_rows:
        return pd.DataFrame(
            columns=["metric", "group_name", "model", "rank", "rank_n_samples"]
        )

    rank_all = pd.concat(rank_rows, ignore_index=True)
    return (
        rank_all.groupby(["metric", "group_name", "model"], as_index=False)
        .agg(rank=("rank", "mean"), rank_n_samples=("sample_key", "nunique"))
    )


def _build_summary(
    user_metric_df: pd.DataFrame,
    sample_metric_df: pd.DataFrame,
    model_order: list[str],
    group_order: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if user_metric_df.empty:
        empty_long = pd.DataFrame(
            columns=[
                "metric",
                "metric_display",
                "group_name",
                "model",
                "metric_mean",
                "rank",
                "n_users",
                "rank_n_samples",
                "total_valid_count",
                "total_positive_count",
                "total_negative_count",
            ]
        )
        return empty_long, pd.DataFrame(columns=["group_name", "metric", "metric_display"])

    finite_df = user_metric_df.loc[np.isfinite(user_metric_df["metric_value"])].copy()
    metric_means_df = (
        finite_df.groupby(
            ["metric", "metric_display", "group_name", "model"],
            as_index=False,
        )
        .agg(
            metric_mean=("metric_value", "mean"),
            n_users=("user_id", "nunique"),
            total_valid_count=("valid_count", "sum"),
            total_positive_count=("positive_count", "sum"),
            total_negative_count=("negative_count", "sum"),
        )
    )
    rank_df = _compute_model_mean_ranks(sample_metric_df)
    long_df = metric_means_df.merge(
        rank_df,
        on=["metric", "group_name", "model"],
        how="left",
    )
    long_df["model"] = pd.Categorical(long_df["model"], categories=model_order, ordered=True)
    long_df["group_name"] = pd.Categorical(
        long_df["group_name"], categories=group_order, ordered=True
    )
    long_df = long_df.sort_values(["group_name", "metric", "model"]).reset_index(drop=True)

    base_df = (
        long_df[["group_name", "metric", "metric_display"]]
        .drop_duplicates()
        .sort_values(["group_name", "metric"])
        .reset_index(drop=True)
    )
    wide_df = base_df.copy()
    for model_name in model_order:
        model_slice = long_df.loc[long_df["model"] == model_name].copy()
        if model_slice.empty:
            continue
        model_slice = model_slice[
            [
                "group_name",
                "metric",
                "metric_mean",
                "rank",
                "n_users",
                "rank_n_samples",
            ]
        ].rename(
            columns={
                "metric_mean": f"{model_name}_metric",
                "rank": f"{model_name}_rank",
                "n_users": f"{model_name}_n_users",
                "rank_n_samples": f"{model_name}_rank_n_samples",
            }
        )
        wide_df = wide_df.merge(model_slice, on=["group_name", "metric"], how="left")

    return long_df, wide_df


def _render_preview_table(wide_df: pd.DataFrame, model_order: list[str]) -> str:
    if wide_df.empty:
        return "(empty)"

    preview_columns = ["group_name", "metric_display"]
    for model_name in model_order:
        if f"{model_name}_metric" not in wide_df.columns:
            wide_df[f"{model_name}_metric"] = np.nan
        if f"{model_name}_rank" not in wide_df.columns:
            wide_df[f"{model_name}_rank"] = np.nan
        preview_columns.extend([f"{model_name}_metric", f"{model_name}_rank"])

    preview_df = wide_df[preview_columns].copy().rename(columns={"metric_display": "metric"})
    for column in preview_df.columns:
        if column.endswith("_metric") or column.endswith("_rank"):
            preview_df[column] = preview_df[column].map(
                lambda value: "/"
                if pd.isna(value) or not np.isfinite(float(value))
                else f"{float(value):.3f}"
            )

    widths = {
        column: max(len(str(column)), preview_df[column].astype(str).map(len).max())
        for column in preview_df.columns
    }
    header = " | ".join(str(column).ljust(widths[column]) for column in preview_df.columns)
    divider = "-+-".join("-" * widths[column] for column in preview_df.columns)
    body = [
        " | ".join(str(row[column]).ljust(widths[column]) for column in preview_df.columns)
        for _, row in preview_df.iterrows()
    ]
    return "\n".join([header, divider, *body])


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for grouped binary metric summaries."""
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate binary forecasting metrics from local per-user parquet files "
            "into grouped sample-based model summaries."
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="MODEL_NAME=/path/to/results/metrics/<model_root>",
    )
    parser.add_argument(
        "--group",
        action="append",
        default=None,
        help="GROUP_NAME=7,8,9. May be passed multiple times. Defaults to sleep/workout groups.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/metrics_summary",
        help="Directory for generated summary CSV files.",
    )
    parser.add_argument(
        "--output-prefix",
        default="binary_group_metric_rank_summary",
        help="Filename prefix for generated CSV files.",
    )
    return parser


def main() -> None:
    """Generate grouped binary metric summary outputs."""
    args = build_parser().parse_args()
    models = dict(_parse_model_arg(item) for item in args.model)
    model_order = list(models.keys())
    groups = _resolve_groups(args.group)
    group_order = [group_name for group_name, _ in groups]

    sample_frames: list[pd.DataFrame] = []
    for model_name, model_root in models.items():
        df = _load_sample_metric_rows(
            model_name=model_name,
            model_root=model_root,
            groups=groups,
        )
        if not df.empty:
            sample_frames.append(df)

    if sample_frames:
        sample_metric_df = pd.concat(sample_frames, ignore_index=True)
    else:
        sample_metric_df = pd.DataFrame()

    user_metric_df = _build_user_metric_rows(sample_metric_df)
    long_df, wide_df = _build_summary(
        user_metric_df,
        sample_metric_df,
        model_order=model_order,
        group_order=group_order,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    user_output_path = output_dir / f"{args.output_prefix}_user_level_long.csv"
    long_output_path = output_dir / f"{args.output_prefix}_long.csv"
    wide_output_path = output_dir / f"{args.output_prefix}_wide.csv"

    user_metric_df.to_csv(user_output_path, index=False)
    long_df.to_csv(long_output_path, index=False)
    wide_df.to_csv(wide_output_path, index=False)

    print("=== Binary grouped metric/rank summary preview ===")
    print(_render_preview_table(wide_df, model_order=model_order))
    print(f"\nSaved user-level table: {user_output_path}")
    print(f"Saved long table: {long_output_path}")
    print(f"Saved wide table: {wide_output_path}")
    print("Group rule: per-sample group metric is the mean of finite channel metrics within the group.")
    print("Skip rule: if total positive_count for one sample/group is zero, skip that sample/group.")
    print("Rank rule: rank models per (group, metric, sample), then average ranks over samples.")


if __name__ == "__main__":
    main()
