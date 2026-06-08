"""Pipeline Signal Processing: Features from timeseries daily checkpoints.

Reads the daily-level Parquet checkpoints produced by the timeseries pipeline's
chunked processing, treats each user's sequence of daily values as a time series,
and extracts statistical, ARIMA, and cross-correlation features.

Two-phase approach:
  Phase 1 (stat) — Pure Polars expressions, fully parallel (140 features)
  Phase 2 (ARIMA/CC) — Python via map_groups (126 features)

Output: one row per user, ~266 signal-processing feature columns + user_id.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from .signal_features import (
    SIGNAL_PROCESSING_METRICS,
    build_stat_expressions,
    extract_arima_cc_for_user,
)

logger = logging.getLogger(__name__)


def build_signal_processing_features(
    checkpoint_dir: Path,
    output_path: Path | None = None,
    cutoff_dates: dict[str, str] | None = None,
) -> pl.DataFrame:
    """Build signal-processing user-level features from timeseries daily checkpoints.

    Args:
        checkpoint_dir: Directory containing timeseries per-file checkpoint
                        Parquet files (e.g., FEATURES_DIR / "timeseries_daily_chunks").
        output_path: Optional path to write the output Parquet file.
        cutoff_dates: Optional ``{user_id: "YYYY-MM-DD"}`` per-user data cutoff.
                      Rows with ``date > cutoff_dates[user_id]`` are excluded
                      before aggregation.

    Returns:
        DataFrame with one row per user and ~266 signal-processing feature columns.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    chunk_files = sorted(checkpoint_dir.glob("*.parquet"))
    if not chunk_files:
        raise ValueError(f"No Parquet checkpoint files in {checkpoint_dir}")

    logger.info("Loading %d daily checkpoint files...", len(chunk_files))
    daily_df = pl.concat([pl.read_parquet(f) for f in chunk_files])
    logger.info("Loaded daily data: %d rows, %d columns", daily_df.shape[0], daily_df.shape[1])

    # Future-data cutoff: drop rows after each user's cutoff date
    if cutoff_dates is not None:
        import datetime as dt

        if daily_df["date"].dtype == pl.Utf8:
            daily_df = daily_df.with_columns(pl.col("date").str.to_date())
        elif daily_df["date"].dtype != pl.Date:
            daily_df = daily_df.with_columns(pl.col("date").cast(pl.Date))
        cutoff_rows = [
            {"user_id": uid, "_cutoff": dt.date.fromisoformat(d)}
            for uid, d in cutoff_dates.items()
        ]
        cutoff_df = pl.DataFrame(cutoff_rows).with_columns(pl.col("_cutoff").cast(pl.Date))
        before = daily_df.shape[0]
        daily_df = daily_df.join(cutoff_df, on="user_id", how="left").filter(
            pl.col("_cutoff").is_null() | (pl.col("date") <= pl.col("_cutoff"))
        ).drop("_cutoff")
        after = daily_df.shape[0]
        if after < before:
            logger.info("  Cutoff filter: %d -> %d rows (-%d)", before, after, before - after)

    # Verify required metrics exist
    available = set(daily_df.columns)
    missing = [m for m in SIGNAL_PROCESSING_METRICS if m not in available]
    if missing:
        logger.warning("%d metrics missing from checkpoints: %s", len(missing), missing)
    metrics_to_use = [m for m in SIGNAL_PROCESSING_METRICS if m in available]
    logger.info("Using %d metrics for signal processing", len(metrics_to_use))

    # ── Phase 1: Stat features (pure Polars) ─────────────────────────────────
    logger.info("Phase 1: Statistical features (pure Polars)...")
    stat_exprs = build_stat_expressions(metrics_to_use)
    stat_df = daily_df.lazy().group_by("user_id").agg(stat_exprs).collect()
    logger.info("  Stat features: %d columns for %d users", stat_df.shape[1] - 1, stat_df.shape[0])

    # ── Phase 2: ARIMA + CC (Python via map_groups) ─────────────────────────
    logger.info("Phase 2: ARIMA + CC features (Python)...")
    # Select only needed columns to minimize data passed to map_groups
    cols_for_phase2 = ["user_id", "date"] + metrics_to_use
    cols_for_phase2 = [c for c in cols_for_phase2 if c in daily_df.columns]
    phase2_input = daily_df.select(cols_for_phase2).sort("user_id", "date")

    arima_cc_df = phase2_input.group_by("user_id", maintain_order=True).map_groups(
        extract_arima_cc_for_user
    )
    logger.info(
        "  ARIMA/CC features: %d columns for %d users",
        arima_cc_df.shape[1] - 1, arima_cc_df.shape[0],
    )

    # ── Join phases ──────────────────────────────────────────────────────────
    logger.info("Joining Phase 1 and Phase 2...")
    result = stat_df.join(arima_cc_df, on="user_id", how="left")
    logger.info("Combined: %d users x %d columns", result.shape[0], result.shape[1])

    # ── Write output ─────────────────────────────────────────────────────────
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.write_parquet(output_path)
        logger.info("Wrote signal processing features to %s", output_path)

    return result
