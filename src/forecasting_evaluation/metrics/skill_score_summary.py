"""Compute forecasting skill scores against a configurable baseline model.

The input layout matches the offline metrics tree:

``results/metrics/<model_name>/<metric_name>/<user_id>.parquet``

For each metric/channel, metric rows are first collapsed to either user-level
or sample-level values. Skill scores are then computed from paired model and
baseline errors using:

``skill = 1 - geometric_mean(clip(E_model / E_baseline, lower, upper))``.
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

_CHANNEL_CONSTANTS_PATH = SRC_ROOT / "visualizations" / "constants.py"
_CHANNEL_SPEC = (
    importlib.util.spec_from_file_location(
        "_forecasting_skill_score_channel_constants",
        _CHANNEL_CONSTANTS_PATH,
    )
    if _CHANNEL_CONSTANTS_PATH.exists()
    else None
)
if _CHANNEL_SPEC is None or _CHANNEL_SPEC.loader is None:
    CHANNEL_INFO = {}
else:
    _channel_module = importlib.util.module_from_spec(_CHANNEL_SPEC)
    _CHANNEL_SPEC.loader.exec_module(_channel_module)
    CHANNEL_INFO = getattr(_channel_module, "CHANNEL_INFO", {})

LOWER_IS_BETTER_METRICS = {"mae", "mse", "mase", "mase_all", "ql", "sql"}
HIGHER_IS_BETTER_METRICS = {"f1", "auprc", "auroc"}


def _safe_read_parquet(file_path: str | Path, **kwargs: Any) -> pd.DataFrame | None:
    path = Path(file_path)
    try:
        if not path.exists() or path.stat().st_size == 0:
            return None
        return pd.read_parquet(path, **kwargs)
    except Exception:
        return None


def _list_parquet_files(metric_dir: str | Path) -> list[Path]:
    path = Path(metric_dir)
    if not path.exists():
        return []
    return sorted(path.rglob("*.parquet"))


def _channel_label(channel_idx: int) -> str:
    metadata = CHANNEL_INFO.get(channel_idx)
    if metadata is None:
        return f"Channel {channel_idx}"
    return str(metadata["name"])


def _parse_channel_indices(raw_value: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
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


def _metric_to_error(metric_name: str, metric_value: float) -> float:
    metric_key = metric_name.strip().lower()
    if not np.isfinite(metric_value):
        return float("nan")
    if metric_key in LOWER_IS_BETTER_METRICS:
        return float(metric_value) if metric_value >= 0.0 else float("nan")
    if metric_key in HIGHER_IS_BETTER_METRICS:
        if metric_value < 0.0 or metric_value > 1.0:
            return float("nan")
        return float(1.0 - metric_value)
    raise ValueError(
        f"Unknown metric '{metric_name}'. Add it to lower- or higher-is-better sets."
    )


def compute_skill_from_errors(
    model_errors: np.ndarray,
    baseline_errors: np.ndarray,
    *,
    clip_lower: float = 0.01,
    clip_upper: float = 100.0,
    min_pairs: int = 1,
) -> tuple[float, float, int]:
    """Compute skill score from paired model and baseline errors."""
    model_arr = np.asarray(model_errors, dtype=float).reshape(-1)
    baseline_arr = np.asarray(baseline_errors, dtype=float).reshape(-1)
    n = min(model_arr.shape[0], baseline_arr.shape[0])
    if n == 0:
        return float("nan"), float("nan"), 0

    model_arr = model_arr[:n]
    baseline_arr = baseline_arr[:n]
    valid = np.isfinite(model_arr) & np.isfinite(baseline_arr) & (baseline_arr > 0.0)
    if int(valid.sum()) < int(min_pairs):
        return float("nan"), float("nan"), int(valid.sum())

    ratios = model_arr[valid] / baseline_arr[valid]
    ratios = np.clip(ratios, float(clip_lower), float(clip_upper))
    valid_ratios = ratios[np.isfinite(ratios) & (ratios > 0.0)]
    if valid_ratios.shape[0] < int(min_pairs):
        return float("nan"), float("nan"), int(valid_ratios.shape[0])

    geometric_mean_ratio = float(np.exp(np.mean(np.log(valid_ratios))))
    return float(1.0 - geometric_mean_ratio), geometric_mean_ratio, int(valid_ratios.shape[0])


def _metric_channel_value(metric: np.ndarray, channel_idx: int) -> float:
    if metric.ndim == 1:
        if channel_idx >= metric.shape[0]:
            return float("nan")
        value = float(metric[channel_idx])
        return value if np.isfinite(value) else float("nan")

    if channel_idx >= metric.shape[0]:
        return float("nan")
    values = metric[channel_idx]
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return float("nan")
    return float(np.mean(finite_values))


def _row_sample_key(row: pd.Series, occurrence: int) -> str:
    history_length = row.get("history_length")
    forecasting_length = row.get("forecasting_length")
    if pd.notna(history_length):
        return f"h={int(history_length)}|f={int(forecasting_length) if pd.notna(forecasting_length) else ''}|i={occurrence}"
    return f"row={occurrence}"


def _load_metric_values(
    *,
    model_name: str,
    model_root: str | Path,
    metric_name: str,
    channel_indices: tuple[int, ...],
    group_name: str,
    aggregation_unit: str,
) -> pd.DataFrame:
    metric_dir = Path(model_root) / metric_name
    rows: list[dict[str, Any]] = []
    per_unit_values: dict[tuple[str, int, str], list[float]] = {}

    for parquet_file in _list_parquet_files(metric_dir):
        df = _safe_read_parquet(
            parquet_file,
            columns=["user_id", "history_length", "forecasting_length", metric_name],
        )
        if df is None or "user_id" not in df.columns or metric_name not in df.columns:
            continue

        occurrence_by_user_history: dict[tuple[str, str], int] = {}
        for _, row in df.iterrows():
            user_id = str(row.get("user_id"))
            metric = _safe_to_metric_array(row.get(metric_name))
            if metric is None:
                continue

            sample_seed = _row_sample_key(row, occurrence=0)
            occurrence_key = (user_id, sample_seed)
            occurrence = occurrence_by_user_history.get(occurrence_key, 0)
            occurrence_by_user_history[occurrence_key] = occurrence + 1
            sample_id = _row_sample_key(row, occurrence=occurrence)
            unit_id = user_id if aggregation_unit == "user" else f"{user_id}|{sample_id}"

            for channel_idx in channel_indices:
                value = _metric_channel_value(metric=metric, channel_idx=channel_idx)
                if not np.isfinite(value):
                    continue
                error = _metric_to_error(metric_name=metric_name, metric_value=value)
                if not np.isfinite(error):
                    continue
                per_unit_values.setdefault((unit_id, int(channel_idx), metric_name), []).append(error)

    for (unit_id, channel_idx, metric), values in per_unit_values.items():
        finite_values = np.asarray(values, dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        if finite_values.size == 0:
            continue
        rows.append(
            {
                "model": model_name,
                "group": group_name,
                "metric": metric,
                "channel_idx": int(channel_idx),
                "channel_name": _channel_label(channel_idx),
                "unit_id": unit_id,
                "error": float(np.mean(finite_values)),
                "n_values": int(finite_values.size),
            }
        )

    return pd.DataFrame(rows)


def _build_error_table(
    *,
    models: dict[str, dict[str, str]],
    metric_groups: dict[str, dict[str, Any]],
    aggregation_unit: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for group_name, group_spec in metric_groups.items():
        metric_names = group_spec["metrics"]
        channel_indices = group_spec["channel_indices"]
        for model_name, model_spec in models.items():
            for metric_name in metric_names:
                frame = _load_metric_values(
                    model_name=model_name,
                    model_root=model_spec["path"],
                    metric_name=metric_name,
                    channel_indices=channel_indices,
                    group_name=group_name,
                    aggregation_unit=aggregation_unit,
                )
                if not frame.empty:
                    frames.append(frame)

    if not frames:
        return pd.DataFrame(
            columns=[
                "model",
                "group",
                "metric",
                "channel_idx",
                "channel_name",
                "unit_id",
                "error",
                "n_values",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def _compute_long_skill_scores(
    *,
    error_df: pd.DataFrame,
    models: dict[str, dict[str, str]],
    baseline_model: str,
    clip_lower: float,
    clip_upper: float,
    min_pairs: int,
) -> pd.DataFrame:
    columns = [
        "model",
        "baseline_model",
        "group",
        "metric",
        "channel_idx",
        "channel_name",
        "skill_score",
        "geometric_mean_ratio",
        "n_users",
        "n_pairs",
        "model_error_mean",
        "baseline_error_mean",
    ]
    if error_df.empty:
        return pd.DataFrame(columns=columns)

    baseline_df = error_df.loc[error_df["model"] == baseline_model].copy()
    if baseline_df.empty:
        raise ValueError(f"Baseline model '{baseline_model}' has no readable metric rows.")

    rows: list[dict[str, Any]] = []
    group_cols = ["group", "metric", "channel_idx", "channel_name"]
    baseline_groups = {
        key: group.set_index("unit_id")["error"]
        for key, group in baseline_df.groupby(group_cols, sort=True)
    }

    for model_name in models:
        model_df = error_df.loc[error_df["model"] == model_name].copy()
        for key, model_group in model_df.groupby(group_cols, sort=True):
            baseline_errors = baseline_groups.get(key)
            group_name, metric_name, channel_idx, channel_name = key
            if baseline_errors is None or baseline_errors.empty:
                rows.append(
                    {
                        "model": model_name,
                        "baseline_model": baseline_model,
                        "group": group_name,
                        "metric": metric_name,
                        "channel_idx": int(channel_idx),
                        "channel_name": channel_name,
                        "skill_score": float("nan"),
                        "geometric_mean_ratio": float("nan"),
                        "n_users": 0,
                        "n_pairs": 0,
                        "model_error_mean": float("nan"),
                        "baseline_error_mean": float("nan"),
                    }
                )
                continue

            model_errors = model_group.set_index("unit_id")["error"]
            paired = pd.concat(
                [model_errors.rename("model_error"), baseline_errors.rename("baseline_error")],
                axis=1,
                join="inner",
            ).dropna()

            skill, gm_ratio, n_pairs = compute_skill_from_errors(
                paired["model_error"].to_numpy(dtype=float),
                paired["baseline_error"].to_numpy(dtype=float),
                clip_lower=clip_lower,
                clip_upper=clip_upper,
                min_pairs=min_pairs,
            )
            rows.append(
                {
                    "model": model_name,
                    "baseline_model": baseline_model,
                    "group": group_name,
                    "metric": metric_name,
                    "channel_idx": int(channel_idx),
                    "channel_name": channel_name,
                    "skill_score": skill,
                    "geometric_mean_ratio": gm_ratio,
                    "n_users": _count_users_from_unit_index(paired.index),
                    "n_pairs": int(n_pairs),
                    "model_error_mean": float(paired["model_error"].mean()) if not paired.empty else float("nan"),
                    "baseline_error_mean": float(paired["baseline_error"].mean()) if not paired.empty else float("nan"),
                }
            )

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["group", "channel_idx", "metric", "model"]
    ).reset_index(drop=True)


def _count_users_from_unit_index(index: pd.Index) -> int:
    users = {str(value).split("|", 1)[0] for value in index.tolist()}
    users.discard("")
    return len(users)


def _aggregate_rows_score(rows_df: pd.DataFrame) -> tuple[float, int]:
    if rows_df.empty:
        return float("nan"), 0
    ratios = rows_df["geometric_mean_ratio"].to_numpy(dtype=float)
    ratios = ratios[np.isfinite(ratios) & (ratios > 0.0)]
    if ratios.size == 0:
        return float("nan"), 0
    return float(1.0 - np.exp(np.mean(np.log(ratios)))), int(ratios.size)


def _aggregate_group_score(long_df: pd.DataFrame, model_name: str, group_name: str) -> tuple[float, int]:
    group_rows = long_df.loc[
        (long_df["model"] == model_name)
        & (long_df["group"] == group_name)
        & np.isfinite(long_df["geometric_mean_ratio"])
    ]
    return _aggregate_rows_score(group_rows)


def _aggregate_channel_score(
    long_df: pd.DataFrame,
    model_name: str,
    channel_idx: int,
) -> tuple[float, int]:
    channel_rows = long_df.loc[
        (long_df["model"] == model_name)
        & (long_df["group"] == "continuous")
        & (long_df["channel_idx"] == int(channel_idx))
        & np.isfinite(long_df["geometric_mean_ratio"])
    ]
    return _aggregate_rows_score(channel_rows)


def _aggregate_binary_collection_score(
    long_df: pd.DataFrame,
    model_name: str,
    channel_indices: tuple[int, ...],
) -> tuple[float, int]:
    collection_rows = long_df.loc[
        (long_df["model"] == model_name)
        & (long_df["group"] == "binary")
        & (long_df["channel_idx"].isin(channel_indices))
        & np.isfinite(long_df["geometric_mean_ratio"])
    ]
    return _aggregate_rows_score(collection_rows)


def _build_model_summary(
    *,
    long_df: pd.DataFrame,
    models: dict[str, dict[str, str]],
    baseline_model: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name in models:
        row: dict[str, Any] = {
            "model": model_name,
            "baseline_model": baseline_model,
        }
        continuous_channels = sorted(
            int(idx)
            for idx in long_df.loc[long_df["group"] == "continuous", "channel_idx"].dropna().unique()
        )
        for channel_idx in continuous_channels:
            score, n_units = _aggregate_channel_score(long_df, model_name, channel_idx)
            row[f"channel_{channel_idx}_score"] = score
            row[f"channel_{channel_idx}_n_units"] = n_units

        sleep_score, n_sleep = _aggregate_binary_collection_score(long_df, model_name, (7, 8))
        workout_score, n_workout = _aggregate_binary_collection_score(
            long_df,
            model_name,
            tuple(range(9, 19)),
        )
        row.update(
            {
                "sleep_score": sleep_score,
                "workout_score": workout_score,
                "sleep_n_units": n_sleep,
                "workout_n_units": n_workout,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _build_wide_summary(long_df: pd.DataFrame, models: dict[str, dict[str, str]]) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame(columns=["group", "metric", "channel_idx", "channel_name"])

    wide_df = (
        long_df[["group", "metric", "channel_idx", "channel_name"]]
        .drop_duplicates()
        .sort_values(["group", "channel_idx", "metric"])
        .reset_index(drop=True)
    )
    for model_name, model_spec in models.items():
        display_name = model_spec["display_name"]
        model_slice = long_df.loc[long_df["model"] == model_name].copy()
        model_slice = model_slice[
            [
                "group",
                "metric",
                "channel_idx",
                "skill_score",
                "geometric_mean_ratio",
                "n_pairs",
            ]
        ].rename(
            columns={
                "skill_score": f"{display_name}_skill_score",
                "geometric_mean_ratio": f"{display_name}_geometric_mean_ratio",
                "n_pairs": f"{display_name}_n_pairs",
            }
        )
        wide_df = wide_df.merge(model_slice, on=["group", "metric", "channel_idx"], how="left")
    return wide_df


def compute_skill_score_tables(
    *,
    models: dict[str, dict[str, str]],
    baseline_model: str,
    continuous_metrics: list[str],
    binary_metrics: list[str],
    continuous_channel_indices: tuple[int, ...],
    binary_channel_indices: tuple[int, ...],
    clip_lower: float,
    clip_upper: float,
    min_pairs: int,
    aggregation_unit: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute long, model summary, and wide skill score tables."""
    if baseline_model not in models:
        raise ValueError(
            f"Baseline model '{baseline_model}' is not in model config. "
            f"Available models: {', '.join(models)}"
        )
    if clip_lower <= 0 or clip_upper <= 0 or clip_lower > clip_upper:
        raise ValueError("--clip-lower and --clip-upper must be positive with lower <= upper")
    if aggregation_unit not in {"user", "sample"}:
        raise ValueError("--aggregation-unit must be either 'user' or 'sample'")

    metric_groups = {
        "continuous": {
            "metrics": [m.strip().lower() for m in continuous_metrics if m.strip()],
            "channel_indices": continuous_channel_indices,
        },
        "binary": {
            "metrics": [m.strip().lower() for m in binary_metrics if m.strip()],
            "channel_indices": binary_channel_indices,
        },
    }
    error_df = _build_error_table(
        models=models,
        metric_groups=metric_groups,
        aggregation_unit=aggregation_unit,
    )
    long_df = _compute_long_skill_scores(
        error_df=error_df,
        models=models,
        baseline_model=baseline_model,
        clip_lower=clip_lower,
        clip_upper=clip_upper,
        min_pairs=min_pairs,
    )
    summary_df = _build_model_summary(
        long_df=long_df,
        models=models,
        baseline_model=baseline_model,
    )
    wide_df = _build_wide_summary(long_df=long_df, models=models)
    return long_df, summary_df, wide_df


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for forecasting skill score summaries."""
    parser = argparse.ArgumentParser(
        description="Compute forecasting skill scores from offline metric parquet files."
    )
    parser.add_argument("--config", default=None, help="JSON/YAML config with model mappings.")
    parser.add_argument(
        "--models-json",
        default=None,
        help='Inline JSON dict, e.g. {"models":{"modelA":"/path/a"}}',
    )
    parser.add_argument("--baseline", required=True, help="Baseline model key from config.")
    parser.add_argument(
        "--continuous-channel-indices",
        default="0,1,2,3,4,5,6",
        help="Comma-separated continuous channel indices. Defaults to 0-6.",
    )
    parser.add_argument(
        "--continuous-metrics",
        nargs="+",
        default=["mase", "sql"],
        help="Continuous metrics to include. Defaults to mase sql.",
    )
    parser.add_argument(
        "--binary-channel-indices",
        default="7,8,9,10,11,12,13,14,15,16,17,18",
        help="Comma-separated binary channel indices. Defaults to 7-18.",
    )
    parser.add_argument(
        "--binary-metrics",
        nargs="+",
        default=["mse", "f1"],
        help="Binary-channel metrics to include. Defaults to mse f1.",
    )
    parser.add_argument("--clip-lower", type=float, default=0.01)
    parser.add_argument("--clip-upper", type=float, default=100.0)
    parser.add_argument("--min-pairs", type=int, default=1)
    parser.add_argument(
        "--aggregation-unit",
        choices=["user", "sample"],
        default="user",
        help="Collapse metrics to user-level or sample-level before paired ratios.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/metrics_summary",
        help="Directory for generated CSV files.",
    )
    parser.add_argument(
        "--output-prefix",
        default="forecasting_skill_score",
        help="Filename prefix for generated CSV files.",
    )
    return parser


def main() -> None:
    """Generate forecasting skill score summary CSV outputs."""
    args = build_parser().parse_args()
    models = _load_models_dict(args)
    continuous_channel_indices = _parse_channel_indices(
        args.continuous_channel_indices,
        default=tuple(range(0, 7)),
    )
    binary_channel_indices = _parse_channel_indices(
        args.binary_channel_indices,
        default=tuple(range(7, 19)),
    )

    long_df, summary_df, wide_df = compute_skill_score_tables(
        models=models,
        baseline_model=args.baseline,
        continuous_metrics=list(args.continuous_metrics),
        binary_metrics=list(args.binary_metrics),
        continuous_channel_indices=continuous_channel_indices,
        binary_channel_indices=binary_channel_indices,
        clip_lower=float(args.clip_lower),
        clip_upper=float(args.clip_upper),
        min_pairs=int(args.min_pairs),
        aggregation_unit=str(args.aggregation_unit),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    long_path = output_dir / f"{args.output_prefix}_long.csv"
    summary_path = output_dir / f"{args.output_prefix}_model_summary.csv"
    wide_path = output_dir / f"{args.output_prefix}_wide.csv"

    long_df.to_csv(long_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    wide_df.to_csv(wide_path, index=False)

    print("=== Forecasting skill score summary ===")
    if summary_df.empty:
        print("(empty)")
    else:
        print(summary_df.to_string(index=False))
    print(f"\nSaved long table: {long_path}")
    print(f"Saved model summary: {summary_path}")
    print(f"Saved wide table: {wide_path}")
    print(f"Baseline: {args.baseline}")
    print(f"Aggregation unit: {args.aggregation_unit}")
    print(f"Ratio clip: [{args.clip_lower}, {args.clip_upper}]")


if __name__ == "__main__":
    main()
