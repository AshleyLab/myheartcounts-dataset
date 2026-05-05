"""Summarize valid MASE coverage by model and channel from metrics parquet files."""

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


def aggregate_mase_validity_streaming(
    metrics_paths: dict[str, str],
    max_user: int | None,
    random_seed: int,
) -> pd.DataFrame:
    """Aggregate total and valid MASE timestamps by model and channel."""
    rng = random.Random(random_seed)
    acc: dict[tuple[str, str, int], dict[str, int]] = {}

    for input_name, model_dir in metrics_paths.items():
        files = _list_parquet_files(model_dir)
        if not files:
            print(f"[skip] no parquet found for input: {input_name} @ {model_dir}")
            continue

        users = _collect_users_for_dir(model_dir)
        selected_users = _select_users(users, max_user=max_user, rng=rng)
        print(
            f"input={input_name} users_total={len(users)} users_selected={len(selected_users)}"
        )

        inferred_n_features = None
        for parquet_file in files:
            df = _safe_read_parquet(parquet_file, columns=["mase"])
            if df is None or "mase" not in df.columns:
                continue
            for raw_mase in df["mase"]:
                mase = _safe_to_2d_array(raw_mase)
                if mase is not None:
                    inferred_n_features = mase.shape[0]
                    break
            if inferred_n_features is not None:
                break

        if inferred_n_features is None:
            print(f"[skip] no valid mase found for input: {input_name}")
            continue

        channel_names = _infer_channel_names(model_dir, inferred_n_features)

        for parquet_file in files:
            df = _safe_read_parquet(parquet_file, columns=["user_id", "mase"])
            if df is None or "user_id" not in df.columns or "mase" not in df.columns:
                continue

            if selected_users:
                df = df[df["user_id"].astype(str).isin(selected_users)]
                if df.empty:
                    continue

            for _, row in df.iterrows():
                mase = _safe_to_2d_array(row.get("mase"))
                if mase is None:
                    continue

                n_features, horizon = mase.shape
                channels = channel_names[:n_features]
                if len(channels) < n_features:
                    channels += [f"feature_{i}" for i in range(len(channels), n_features)]

                for channel_idx in range(n_features):
                    channel_values = mase[channel_idx]
                    total_timestamps = int(channel_values.shape[0])
                    valid_timestamps = int(np.isfinite(channel_values).sum())
                    key = (input_name, channels[channel_idx], channel_idx)
                    if key not in acc:
                        acc[key] = {
                            "total_timestamps": 0,
                            "valid_mase_timestamps": 0,
                        }
                    acc[key]["total_timestamps"] += total_timestamps
                    acc[key]["valid_mase_timestamps"] += valid_timestamps

    rows: list[dict[str, Any]] = []
    for (model, channel, channel_idx), stats in acc.items():
        total_timestamps = int(stats["total_timestamps"])
        valid_mase_timestamps = int(stats["valid_mase_timestamps"])
        invalid_mase_timestamps = total_timestamps - valid_mase_timestamps
        valid_ratio = (
            float(valid_mase_timestamps) / float(total_timestamps)
            if total_timestamps > 0
            else np.nan
        )
        rows.append(
            {
                "model": model,
                "channel": channel,
                "channel_idx": channel_idx,
                "total_timestamps": total_timestamps,
                "valid_mase_timestamps": valid_mase_timestamps,
                "invalid_mase_timestamps": invalid_mase_timestamps,
                "valid_mase_ratio": valid_ratio,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "model",
                "channel",
                "channel_idx",
                "total_timestamps",
                "valid_mase_timestamps",
                "invalid_mase_timestamps",
                "valid_mase_ratio",
            ]
        )

    return pd.DataFrame(rows).sort_values(["model", "channel_idx"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for MASE validity summary generation."""
    parser = argparse.ArgumentParser(
        description=(
            "Summarize how many timestamps have valid MASE values by model and channel."
        )
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
        help="Output directory for mase_validity_by_model_channel.csv.",
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
    """Run streaming MASE validity summary and write CSV output."""
    args = parse_args()
    if not args.model:
        raise ValueError("Please provide at least one --model MODEL_NAME=/path/to/metrics_dir")
    metrics_paths = dict(_parse_model_arg(item) for item in args.model)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    validity_df = aggregate_mase_validity_streaming(
        metrics_paths,
        max_user=args.max_user,
        random_seed=args.random_seed,
    )
    output_path = output_dir / "mase_validity_by_model_channel.csv"
    validity_df.to_csv(output_path, index=False)

    print("\n=== Output file ===")
    print(f"mase validity: {output_path}")


if __name__ == "__main__":
    main()
