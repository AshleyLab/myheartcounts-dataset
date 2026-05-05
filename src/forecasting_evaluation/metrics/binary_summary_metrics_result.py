"""Aggregate binary forecasting metrics from local per-user parquet files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from visualizations.constants import CHANNEL_INFO  # noqa: E402

_METRIC_NAMES = ("auprc", "auroc", "f1")


def _metric_display_name(metric_name: str) -> str:
    normalized = metric_name.strip().lower()
    if normalized == "auprc":
        return "AUPRC"
    if normalized == "auroc":
        return "AUROC"
    if normalized == "f1":
        return "F1"
    return metric_name


def _channel_label(channel_idx: int) -> str:
    metadata = CHANNEL_INFO.get(channel_idx)
    if metadata is None:
        return f"Channel {channel_idx}"
    return str(metadata["name"])


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


def _parse_channel_indices(raw_value: str | None) -> tuple[int, ...]:
    if raw_value is None or not raw_value.strip():
        return tuple(range(7, 19))

    indices: list[int] = []
    for part in raw_value.split(","):
        token = part.strip()
        if not token:
            continue
        indices.append(int(token))

    if not indices:
        raise ValueError("At least one channel index must be provided.")
    return tuple(indices)


def _load_user_metric_rows(
    *,
    model_name: str,
    model_root: str | Path,
    channel_indices: tuple[int, ...],
) -> pd.DataFrame:
    root = Path(model_root)
    f1_dir = root / "f1"
    files = _list_parquet_files(f1_dir)
    if not files:
        return pd.DataFrame()

    sample_values: dict[tuple[str, int, str], list[float]] = {}
    coverage: dict[tuple[str, int], dict[str, int]] = {}

    for metric_name in _METRIC_NAMES:
        metric_dir = root / metric_name
        for parquet_file in _list_parquet_files(metric_dir):
            df = _safe_read_parquet(
                parquet_file,
                columns=[
                    "user_id",
                    metric_name,
                    "binary_valid_count",
                    "binary_positive_count",
                    "binary_negative_count",
                ],
            )
            if (
                df is None
                or "user_id" not in df.columns
                or metric_name not in df.columns
            ):
                continue

            for _, row in df.iterrows():
                user_id = str(row.get("user_id"))
                metric_arr = _safe_to_1d_float_array(row.get(metric_name))
                valid_arr = _safe_to_1d_int_array(row.get("binary_valid_count"))
                positive_arr = _safe_to_1d_int_array(row.get("binary_positive_count"))
                negative_arr = _safe_to_1d_int_array(row.get("binary_negative_count"))
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
                for channel_idx in channel_indices:
                    if channel_idx < 0 or channel_idx >= n_features:
                        continue

                    cov_key = (user_id, int(channel_idx))
                    coverage_entry = coverage.setdefault(
                        cov_key,
                        {"valid_count": 0, "positive_count": 0, "negative_count": 0},
                    )
                    if metric_name == "f1":
                        coverage_entry["valid_count"] += int(valid_arr[channel_idx])
                        coverage_entry["positive_count"] += int(positive_arr[channel_idx])
                        coverage_entry["negative_count"] += int(negative_arr[channel_idx])

                    metric_value = float(metric_arr[channel_idx])
                    if np.isfinite(metric_value):
                        sample_values.setdefault(
                            (user_id, int(channel_idx), metric_name),
                            [],
                        ).append(metric_value)

    rows: list[dict[str, Any]] = []
    for (user_id, channel_idx), cov in coverage.items():
        if int(cov["positive_count"]) <= 0:
            continue
        for metric_name in _METRIC_NAMES:
            values = sample_values.get((user_id, channel_idx, metric_name), [])
            metric_value = float(np.mean(values)) if values else float("nan")
            rows.append(
                {
                    "model": model_name,
                    "channel_idx": int(channel_idx),
                    "channel_name": _channel_label(channel_idx),
                    "user_id": user_id,
                    "metric": metric_name,
                    "metric_display": _metric_display_name(metric_name),
                    "metric_value": metric_value,
                    "valid_count": int(cov["valid_count"]),
                    "positive_count": int(cov["positive_count"]),
                    "negative_count": int(cov["negative_count"]),
                }
            )

    return pd.DataFrame(rows)


def _compute_model_mean_ranks(user_metric_df: pd.DataFrame) -> pd.DataFrame:
    if user_metric_df.empty:
        return pd.DataFrame(
            columns=["metric", "channel_idx", "model", "rank", "rank_n_users"]
        )

    finite_df = user_metric_df.loc[np.isfinite(user_metric_df["metric_value"])].copy()
    if finite_df.empty:
        return pd.DataFrame(
            columns=["metric", "channel_idx", "model", "rank", "rank_n_users"]
        )

    rank_rows: list[pd.DataFrame] = []
    for (metric_name, channel_idx), group_slice in finite_df.groupby(
        ["metric", "channel_idx"],
        sort=True,
    ):
        pivot = group_slice.pivot(index="user_id", columns="model", values="metric_value")
        if pivot.empty:
            continue

        rank_df = pivot.rank(axis=1, method="average", ascending=False)
        long_rank = rank_df.stack(future_stack=True).reset_index()
        long_rank.columns = ["user_id", "model", "rank"]
        long_rank["metric"] = metric_name
        long_rank["channel_idx"] = int(channel_idx)
        rank_rows.append(long_rank)

    if not rank_rows:
        return pd.DataFrame(
            columns=["metric", "channel_idx", "model", "rank", "rank_n_users"]
        )

    rank_all = pd.concat(rank_rows, ignore_index=True)
    return (
        rank_all.groupby(["metric", "channel_idx", "model"], as_index=False)
        .agg(rank=("rank", "mean"), rank_n_users=("user_id", "nunique"))
    )


def _build_summary(user_metric_df: pd.DataFrame, model_order: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if user_metric_df.empty:
        empty_long = pd.DataFrame(
            columns=[
                "metric",
                "metric_display",
                "channel_idx",
                "channel_name",
                "model",
                "metric_mean",
                "rank",
                "n_users",
                "rank_n_users",
                "total_valid_count",
                "total_positive_count",
                "total_negative_count",
            ]
        )
        return empty_long, pd.DataFrame(columns=["channel_idx", "channel_name", "metric"])

    finite_df = user_metric_df.loc[np.isfinite(user_metric_df["metric_value"])].copy()
    metric_means_df = (
        finite_df.groupby(
            ["metric", "metric_display", "channel_idx", "channel_name", "model"],
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
    rank_df = _compute_model_mean_ranks(user_metric_df)
    long_df = metric_means_df.merge(
        rank_df,
        on=["metric", "channel_idx", "model"],
        how="left",
    )
    long_df["model"] = pd.Categorical(long_df["model"], categories=model_order, ordered=True)
    long_df = long_df.sort_values(["channel_idx", "metric", "model"]).reset_index(drop=True)

    base_df = (
        long_df[["channel_idx", "channel_name", "metric", "metric_display"]]
        .drop_duplicates()
        .sort_values(["channel_idx", "metric"])
        .reset_index(drop=True)
    )
    wide_df = base_df.copy()
    for model_name in model_order:
        model_slice = long_df.loc[long_df["model"] == model_name].copy()
        if model_slice.empty:
            continue
        model_slice = model_slice[
            [
                "channel_idx",
                "metric",
                "metric_mean",
                "rank",
                "n_users",
                "rank_n_users",
            ]
        ].rename(
            columns={
                "metric_mean": f"{model_name}_metric",
                "rank": f"{model_name}_rank",
                "n_users": f"{model_name}_n_users",
                "rank_n_users": f"{model_name}_rank_n_users",
            }
        )
        wide_df = wide_df.merge(model_slice, on=["channel_idx", "metric"], how="left")

    return long_df, wide_df


def _render_preview_table(wide_df: pd.DataFrame, model_order: list[str]) -> str:
    if wide_df.empty:
        return "(empty)"

    preview_columns = ["channel_idx", "channel_name", "metric_display"]
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
    """Build the CLI parser for binary channel metric summaries."""
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate binary forecasting metrics from local per-user parquet files "
            "into channel-level model summaries."
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="MODEL_NAME=/path/to/results/metrics/<model_root>",
    )
    parser.add_argument(
        "--channel-indices",
        default="7,8,9,10,11,12,13,14,15,16,17,18",
        help="Comma-separated binary channel indices. Defaults to 7-18.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/metrics_summary",
        help="Directory for generated summary CSV files.",
    )
    parser.add_argument(
        "--output-prefix",
        default="binary_metric_rank_summary_channels_7_18",
        help="Filename prefix for generated CSV files.",
    )
    return parser


def main() -> None:
    """Generate binary channel metric summary outputs."""
    args = build_parser().parse_args()
    models = dict(_parse_model_arg(item) for item in args.model)
    model_order = list(models.keys())
    channel_indices = _parse_channel_indices(args.channel_indices)

    frames: list[pd.DataFrame] = []
    for model_name, model_root in models.items():
        df = _load_user_metric_rows(
            model_name=model_name,
            model_root=model_root,
            channel_indices=channel_indices,
        )
        if not df.empty:
            frames.append(df)

    if frames:
        user_metric_df = pd.concat(frames, ignore_index=True)
    else:
        user_metric_df = pd.DataFrame()

    long_df, wide_df = _build_summary(user_metric_df, model_order=model_order)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    user_output_path = output_dir / f"{args.output_prefix}_user_level_long.csv"
    long_output_path = output_dir / f"{args.output_prefix}_long.csv"
    wide_output_path = output_dir / f"{args.output_prefix}_wide.csv"

    user_metric_df.to_csv(user_output_path, index=False)
    long_df.to_csv(long_output_path, index=False)
    wide_df.to_csv(wide_output_path, index=False)

    print("=== Binary metric/rank summary preview ===")
    print(_render_preview_table(wide_df, model_order=model_order))
    print(f"\nSaved user-level table: {user_output_path}")
    print(f"Saved long table: {long_output_path}")
    print(f"Saved wide table: {wide_output_path}")
    print("User-level rule: mean over finite sample-level metrics within one (model, user, channel).")
    print("Skip rule: if total positive_count for a user/channel is zero, skip that user/channel.")
    print("Model-level rule: mean over finite user-level metric values only.")


if __name__ == "__main__":
    main()
