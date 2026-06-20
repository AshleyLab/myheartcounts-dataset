"""Stream aggregation from forecasting metrics parquet files.

This script exports two CSVs:
- ``mae_by_model_channel_hour.csv``
- ``statistical_result.csv``

It reads files incrementally and never materializes a full MAE long table,
which keeps memory usage bounded for large runs.
"""

from __future__ import annotations

import argparse
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
    """Read parquet with guard rails; return None on read failures."""
    path = Path(file_path)
    try:
        if not path.exists():
            print(f"[skip] parquet not found: {path}")
            return None
        if path.stat().st_size == 0:
            print(f"[skip] parquet file is empty: {path}")
            return None
        return pd.read_parquet(path, **kwargs)
    except Exception as exc:  # pragma: no cover - defensive branch
        print(f"[skip] failed to read parquet: {path} | {type(exc).__name__}: {exc}")
        return None


def _safe_read_with_optional_columns(  # noqa: D206
    file_path: str | Path,
    required_columns: list[str],
    optional_columns: list[str],
) -> pd.DataFrame | None:
    """Read parquet with required columns and best-effort optional columns."""  # noqa: D206
    """If reading required+optional fails due schema mismatch, fallback to required-only."""
    columns_full = required_columns + optional_columns
    path = Path(file_path)
    try:
        if not path.exists():
            print(f"[skip] parquet not found: {path}")
            return None
        if path.stat().st_size == 0:
            print(f"[skip] parquet file is empty: {path}")
            return None
        return pd.read_parquet(path, columns=columns_full)
    except Exception:
        return _safe_read_parquet(path, columns=required_columns)


def _safe_to_2d_array(value: Any) -> np.ndarray | None:
    """Convert nested values to 2D float array, trimming rows to shared min horizon."""
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

    min_len = min(len(row) for row in rows)
    if min_len <= 0:
        return None

    return np.vstack([row[:min_len] for row in rows])


def _read_config_channel(model_dir: str | Path) -> str | None:
    """Read channel key from sibling run config, if present."""
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


def _list_parquet_files(model_dir: str | Path) -> list[Path]:
    path = Path(model_dir)
    if not path.exists():
        return []
    return sorted(path.rglob("*.parquet"))


def _collect_users_for_dir(model_dir: str | Path) -> set[str]:
    """Collect all available user ids for one input directory from parquet files."""
    users: set[str] = set()
    for parquet_file in _list_parquet_files(model_dir):
        df = _safe_read_parquet(parquet_file, columns=["user_id"])
        if df is None or "user_id" not in df.columns:
            continue
        users.update(df["user_id"].astype(str).dropna().unique().tolist())
    return users


def _select_users(users: set[str], max_user: int | None, rng: random.Random) -> set[str]:
    """Randomly select up to max_user users; return all users when max_user is None."""
    if max_user is None or max_user <= 0 or len(users) <= max_user:
        return users
    user_list = sorted(users)
    return set(rng.sample(user_list, k=max_user))


def _infer_channel_names(model_dir: str | Path, n_features: int) -> list[str]:
    """Infer channel names from config key; fallback to feature indices."""
    channel_key = _read_config_channel(model_dir)
    if channel_key is None:
        return [f"feature_{i}" for i in range(n_features)]

    mapped = CHANNEL_NAME_MAP.get(channel_key)
    if not mapped:
        return [f"feature_{i}" for i in range(n_features)]

    if len(mapped) >= n_features:
        return mapped[:n_features]
    return mapped + [f"feature_{i}" for i in range(len(mapped), n_features)]


def _parse_mae_value(raw_value: Any) -> np.ndarray | None:
    """Parse MAE matrix from list-like values stored in `mae` column."""
    if raw_value is None:
        return None
    return _safe_to_2d_array(raw_value)


def _project_hour_index(hour: int, horizon: int) -> int:
    """Project raw horizon index to summary hour index.

    For multi-day forecasts (horizon >= 48), aggregate by hour-of-day so
    hours like 0 and 24 contribute to the same summary bucket (0).
    """
    if horizon >= 48:
        return int(hour % 24)
    return int(hour)


def aggregate_mae_by_channel_hour_streaming(  # noqa: D206
    metrics_paths: dict[str, str],
    max_user: int | None,
    random_seed: int,
) -> pd.DataFrame:
    """Stream MAE rows and aggregate stats per model/channel/hour."""  # noqa: D206
    """The aggregator keeps only running stats in memory: sum, sum of squares, and count."""
    rng = random.Random(random_seed)
    acc: dict[tuple[str, str, int, int], dict[str, float | int]] = {}

    for input_name, model_dir in metrics_paths.items():
        files = _list_parquet_files(model_dir)
        if not files:
            print(f"[skip] no parquet found for input: {input_name} @ {model_dir}")
            continue

        users = _collect_users_for_dir(model_dir)
        selected_users = _select_users(users, max_user=max_user, rng=rng)
        print(f"input={input_name} users_total={len(users)} users_selected={len(selected_users)}")

        inferred_n_features = None
        for parquet_file in files:
            df = _safe_read_parquet(parquet_file, columns=["mae"])
            if df is None:
                continue
            if "mae" not in df.columns:
                continue
            for raw_mae in df["mae"]:
                mae = _parse_mae_value(raw_mae)
                if mae is not None:
                    inferred_n_features = mae.shape[0]
                    break
            if inferred_n_features is not None:
                break

        if inferred_n_features is None:
            print(f"[skip] no valid mae found for input: {input_name}")
            continue

        channel_names = _infer_channel_names(model_dir, inferred_n_features)

        for parquet_file in files:
            df = _safe_read_parquet(parquet_file, columns=["user_id", "model", "mae"])
            if df is None:
                continue
            if "user_id" not in df.columns:
                continue
            if "mae" not in df.columns:
                continue

            if selected_users:
                df = df[df["user_id"].astype(str).isin(selected_users)]
                if df.empty:
                    continue

            for _, row in df.iterrows():
                mae = _parse_mae_value(row.get("mae"))
                if mae is None:
                    continue

                row_model_name = input_name

                n_features, horizon = mae.shape
                channels = channel_names[:n_features]
                if len(channels) < n_features:
                    channels += [f"feature_{i}" for i in range(len(channels), n_features)]

                for channel_idx in range(n_features):
                    for hour in range(horizon):
                        value = mae[channel_idx, hour]
                        if not np.isfinite(value):
                            continue

                        summary_hour = _project_hour_index(hour=hour, horizon=horizon)
                        key = (row_model_name, channels[channel_idx], channel_idx, summary_hour)
                        if key not in acc:
                            acc[key] = {"sum": 0.0, "sum_sq": 0.0, "n": 0}
                        acc[key]["sum"] = float(acc[key]["sum"]) + float(value)
                        acc[key]["sum_sq"] = float(acc[key]["sum_sq"]) + float(value) * float(value)
                        acc[key]["n"] = int(acc[key]["n"]) + 1

    rows: list[dict[str, Any]] = []
    for (model, channel, channel_idx, hour), stats in acc.items():
        n = int(stats["n"])
        if n <= 0:
            continue
        sum_value = float(stats["sum"])
        sum_sq = float(stats["sum_sq"])
        mean = sum_value / n
        var = max(0.0, (sum_sq / n) - (mean * mean))
        std = float(np.sqrt(var))

        rows.append(
            {
                "model": model,
                "channel": channel,
                "channel_idx": channel_idx,
                "hour": hour,
                "mae_mean": mean,
                "mae_std": std,
                "n": n,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["model", "channel", "channel_idx", "hour", "mae_mean", "mae_std", "n"]
        )

    return pd.DataFrame(rows).sort_values(["model", "channel_idx", "hour"]).reset_index(drop=True)


def aggregate_model_statistics_streaming(  # noqa: D206
    metrics_paths: dict[str, str],
    max_user: int | None,
    random_seed: int,
) -> pd.DataFrame:
    """Aggregate model-level user/sample counts and average prediction time."""  # noqa: D206
    """Output columns: model, user_count, sample_count, avg_prediction_time_seconds."""
    rng = random.Random(random_seed)
    acc: dict[str, dict[str, Any]] = {}

    for input_name, model_dir in metrics_paths.items():
        files = _list_parquet_files(model_dir)
        if not files:
            print(f"[skip] no parquet found for input: {input_name} @ {model_dir}")
            continue

        users = _collect_users_for_dir(model_dir)
        selected_users = _select_users(users, max_user=max_user, rng=rng)

        for parquet_file in files:
            df = _safe_read_with_optional_columns(
                parquet_file,
                required_columns=["user_id", "model"],
                optional_columns=["perf_prediction_time_seconds"],
            )
            if df is None or "user_id" not in df.columns:
                continue

            df = df[df["user_id"].astype(str).isin(selected_users)]
            if df.empty:
                continue

            entry = acc.setdefault(
                input_name,
                {
                    "users": set(),
                    "sample_count": 0,
                    "prediction_time_sum": 0.0,
                    "prediction_time_count": 0,
                },
            )
            entry["users"].update(df["user_id"].astype(str).dropna().tolist())
            entry["sample_count"] = int(entry["sample_count"]) + int(len(df))

            if "perf_prediction_time_seconds" in df.columns:
                times = pd.to_numeric(df["perf_prediction_time_seconds"], errors="coerce")
                finite_mask = np.isfinite(times.to_numpy(dtype=float, na_value=np.nan))
                if finite_mask.any():
                    finite_times = times[finite_mask]
                    entry["prediction_time_sum"] = float(entry["prediction_time_sum"]) + float(
                        finite_times.sum()
                    )
                    entry["prediction_time_count"] = int(entry["prediction_time_count"]) + int(
                        finite_times.shape[0]
                    )

    if not acc:
        return pd.DataFrame(
            columns=["model", "user_count", "sample_count", "avg_prediction_time_seconds"]
        )

    rows: list[dict[str, Any]] = []
    for model_name, entry in acc.items():
        prediction_time_count = int(entry["prediction_time_count"])
        avg_prediction_time = (
            float(entry["prediction_time_sum"]) / prediction_time_count
            if prediction_time_count > 0
            else np.nan
        )
        rows.append(
            {
                "model": model_name,
                "user_count": int(len(entry["users"])),
                "sample_count": int(entry["sample_count"]),
                "avg_prediction_time_seconds": float(avg_prediction_time)
                if np.isfinite(avg_prediction_time)
                else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values(["model"]).reset_index(drop=True)


def _parse_model_arg(model_arg: str) -> tuple[str, str]:
    if "=" not in model_arg:
        raise argparse.ArgumentTypeError(
            f"Invalid --model value: {model_arg}. Expected format: MODEL_NAME=/path/to/model_dir"
        )
    model_name, model_dir = model_arg.split("=", 1)
    model_name = model_name.strip()
    model_dir = model_dir.strip()
    if not model_name or not model_dir:
        raise argparse.ArgumentTypeError(
            f"Invalid --model value: {model_arg}. Model name and path must be non-empty."
        )
    return model_name, model_dir


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for metrics summary generation."""
    parser = argparse.ArgumentParser(
        description=("Summarize forecasting metrics and export MAE plus model-level statistics.")
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model mapping in format MODEL_NAME=/path/to/model_metrics_dir. Can be repeated.",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/lp925/code/MHC-benchmark/results/metrics_summary",
        help="Output directory for mae_by_model_channel_hour.csv and statistical_result.csv.",
    )
    parser.add_argument(
        "--max-user",
        type=int,
        default=None,
        help="Max number of random users per model for small-scale testing.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used by --max-user sampling.",
    )
    return parser.parse_args()


def main() -> None:
    """Run streaming metric summaries and write CSV outputs."""
    args = parse_args()
    if not args.model:
        raise ValueError("Please provide at least one --model MODEL_NAME=/path/to/metrics_dir")
    metrics_paths = dict(_parse_model_arg(item) for item in args.model)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mae_agg_df = aggregate_mae_by_channel_hour_streaming(
        metrics_paths,
        max_user=args.max_user,
        random_seed=args.random_seed,
    )
    mae_agg_path = output_dir / "mae_by_model_channel_hour.csv"
    mae_agg_df.to_csv(mae_agg_path, index=False)

    stat_df = aggregate_model_statistics_streaming(
        metrics_paths,
        max_user=args.max_user,
        random_seed=args.random_seed,
    )
    stat_path = output_dir / "statistical_result.csv"
    stat_df.to_csv(stat_path, index=False)

    print("\n=== Output file ===")
    print(f"mae agg: {mae_agg_path}")
    print(f"stats: {stat_path}")


if __name__ == "__main__":
    main()
