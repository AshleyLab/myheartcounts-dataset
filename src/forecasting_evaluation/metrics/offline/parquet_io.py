"""Parquet read/write helpers for offline metrics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from forecasting_evaluation.metrics.offline.common import sanitize_name

logger = logging.getLogger(__name__)

_METRIC_COLUMNS = ("mae", "mse", "mase", "mase_all", "ql", "sql")


def append_metrics_to_parquet(
    output_dir: Path,
    model_key: str,
    user_id: str,
    records: list[dict[str, Any]],
) -> None:
    """Append metrics records to output_dir/user_id.parquet."""
    if not records:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    user_key = sanitize_name(user_id)
    output_file = output_dir / f"{user_key}.parquet"
    df_new = pd.DataFrame(records)

    if output_file.exists():
        df_existing = pd.read_parquet(output_file)
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_parquet(output_file, engine="pyarrow", compression="snappy")
    logger.info("Saved metrics for %s/%s: %s (rows=%d)", model_key, user_id, output_file, len(records))


def save_metrics_result_dict(
    output_dir: Path,
    model_key: str,
    records_by_user: dict[str, list[dict[str, Any]]],
) -> dict[str, str]:
    """Save a metrics-result dict produced by the structured offline pipeline.

    Args:
        output_dir: Directory where user-level parquet files should be written.
        model_key: Filesystem-safe model identifier used only for logging.
        records_by_user: Mapping ``user_id -> list[metrics row dict]``.

    Returns:
        Mapping ``user_id -> saved parquet path`` for all users that had records.
    """
    saved_files: dict[str, str] = {}
    for user_id, records in records_by_user.items():
        append_metrics_to_parquet(
            output_dir=output_dir,
            model_key=model_key,
            user_id=user_id,
            records=records,
        )
        saved_files[user_id] = str(output_dir / f"{sanitize_name(user_id)}.parquet")
    return saved_files


def save_metrics_result_by_metric(
    output_root: Path,
    model_key: str,
    records_by_user: dict[str, list[dict[str, Any]]],
    metric_columns: tuple[str, ...] = _METRIC_COLUMNS,
) -> dict[str, dict[str, str]]:
    """Save offline metrics into per-metric directories.

    Output layout:
        ``output_root/<metric_name>/<user_id>.parquet``

    Each parquet keeps only the requested metric column plus row-identifying
    metadata columns needed by downstream readers.
    """
    saved_files_by_metric: dict[str, dict[str, str]] = {}

    for metric_name in metric_columns:
        metric_output_dir = output_root / metric_name
        metric_records_by_user: dict[str, list[dict[str, Any]]] = {}

        for user_id, records in records_by_user.items():
            metric_rows: list[dict[str, Any]] = []
            for record in records:
                if metric_name not in record:
                    continue
                metric_row = {
                    "user_id": record.get("user_id"),
                    "model": record.get("model"),
                    "history_length": record.get("history_length"),
                    "forecasting_length": record.get("forecasting_length"),
                    metric_name: record.get(metric_name),
                }
                metric_rows.append(metric_row)

            if metric_rows:
                metric_records_by_user[user_id] = metric_rows

        if metric_records_by_user:
            saved_files_by_metric[metric_name] = save_metrics_result_dict(
                output_dir=metric_output_dir,
                model_key=model_key,
                records_by_user=metric_records_by_user,
            )

    return saved_files_by_metric


def infer_user_id_from_filename(path: Path) -> str | None:
    """Infer user id from filename format: {user_id}.parquet."""
    return path.stem or None


def read_user_id_from_parquet(path: Path) -> str | None:
    """Fallback reader to fetch user_id from parquet content."""
    table = pq.read_table(path, columns=["user_id"])
    if table.num_rows == 0:
        return None
    return str(table.column("user_id")[0].as_py())


def read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    """Read all parquet rows as python dicts."""
    try:
        table = pq.read_table(path)
    except Exception as exc:
        logger.warning("Failed to read parquet file %s: %s. Skipping.", path, exc)
        return []
    return table.to_pylist()


def index_prediction_files(
    model_dir: Path,
    model_name: str,
) -> dict[str, list[Path]]:
    """Index prediction parquet files by user id for a single model directory."""
    if not model_dir.exists():
        logger.warning("Model output folder not found for %s under %s", model_name, model_dir)
        return {}

    user_to_files: dict[str, list[Path]] = {}
    for parquet_path in sorted(model_dir.glob("*.parquet")):
        user_id = infer_user_id_from_filename(parquet_path)
        if user_id is None:
            user_id = read_user_id_from_parquet(parquet_path)
        if user_id is None:
            logger.warning("Unable to infer user_id from %s, skipping", parquet_path)
            continue
        user_to_files.setdefault(user_id, []).append(parquet_path)

    return user_to_files
