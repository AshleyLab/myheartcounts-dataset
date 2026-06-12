"""Feature extraction pipeline for MHC wearable data.

This module orchestrates the feature extraction process:
1. Load Arrow files lazily
2. Apply day-level feature extraction
3. Aggregate to user-level features
4. Output to Parquet

Usage:
    >>> from downstream_evaluation.models.xgboost.pipeline_timeseries import build_user_features
    >>> from pathlib import Path
    >>> result = build_user_features(
    ...     Path("data/processed/daily_hf"),
    ...     Path("data/features/xgboost/user_features.parquet")
    ... )
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl
import pyarrow as pa

from .extractors import get_all_daily_extractors, get_all_user_aggregators
from .preprocessing import apply_variance_filter, apply_zero_to_nan

logger = logging.getLogger(__name__)


def _detect_and_normalize(df: pl.DataFrame) -> pl.DataFrame:
    """Auto-detect Arrow schema format and normalize to the canonical schema.

    Two known formats:
    - **XGB (current)**: columns include ``data``, ``nonwear_vector``, ``timestamp``
    - **MHC-B daily_hf**: columns include ``values`` (instead of ``data``),
      no ``nonwear_vector``, no ``timestamp``

    Normalization:
    - Rename ``values`` → ``data`` if present.
    - If ``nonwear_vector`` is absent, synthesize it from ``total_nonwear_minutes``
      (uniform non-wear assumption — used only for the wake-sedentary extractor).
    """
    cols = set(df.columns)

    # Rename values → data (MHC-B format)
    if "values" in cols and "data" not in cols:
        df = df.rename({"values": "data"})

    # Cast List(List(Float*)) → Array(List(Float32), 19) so .arr.get() works
    # MHC-B daily_hf stores values as variable-size List; XGB uses fixed-size Array
    data_dtype = df["data"].dtype
    if data_dtype != pl.Array and str(data_dtype).startswith("List"):
        df = df.with_columns(pl.col("data").cast(pl.Array(pl.List(pl.Float32), 19)))

    # Synthesize nonwear_vector if missing
    if "nonwear_vector" not in df.columns:
        # Create a dummy 1440-element zero vector (assumes all minutes are wear).
        # The only extractor that reads nonwear_vector is _extract_sedentary_bouts_daily,
        # which subtracts wake-hour non-wear.  With an all-zero vector the subtraction
        # is a no-op — equivalent to "no non-wear correction".
        df = df.with_columns(pl.lit([0] * 1440).alias("nonwear_vector"))

    # Convert sensor zeros to NaN before feature extraction.
    # HR=0 → NaN; all-zero activity channels → all-NaN.
    df = apply_zero_to_nan(df)

    return df


def load_arrow_files(arrow_dir: Path, splits: list[str] | None = None) -> pl.LazyFrame:
    """Load Arrow stream files into a lazy DataFrame.

    Scans all .arrow files in the specified directory and its subdirectories
    (train/, test/, val/). Files are loaded lazily for memory efficiency.

    Automatically detects both the original XGB schema (``data`` column) and
    the MHC-B daily_hf schema (``values`` column) and normalizes to a
    canonical format with columns: user_id, date, data, nonwear_vector,
    total_nonwear_minutes.

    Args:
        arrow_dir: Path to directory containing Arrow files (with train/test/val subdirs)
        splits: Optional list of splits to load (e.g., ["train", "val"]).
                If None, loads all splits.

    Returns:
        Polars LazyFrame with columns: user_id, date, data,
        nonwear_vector, total_nonwear_minutes (plus any extras present)

    Raises:
        FileNotFoundError: If arrow_dir does not exist
        ValueError: If no Arrow files found

    Example:
        >>> lf = load_arrow_files(Path("data/processed/daily_hf"))
        >>> lf = load_arrow_files(Path("data/processed/daily_hf"), splits=["train"])
    """
    arrow_dir = Path(arrow_dir)
    if not arrow_dir.exists():
        raise FileNotFoundError(f"Directory not found: {arrow_dir}")

    # Determine which splits to load
    if splits is None:
        splits = ["train", "test", "val"]

    # Collect all arrow files (try subdirs first, then flat directory)
    arrow_files = []
    for split in splits:
        split_dir = arrow_dir / split
        if split_dir.exists():
            arrow_files.extend(split_dir.glob("*.arrow"))

    # Flat directory fallback (MHC-B daily_hf has no subdirs)
    if not arrow_files:
        arrow_files = sorted(arrow_dir.glob("data-*.arrow"))

    if not arrow_files:
        raise ValueError(f"No Arrow files found in {arrow_dir}")

    # Load each file and concatenate
    # Note: Arrow stream files need special handling - read with pyarrow first
    dfs = []
    for arrow_file in arrow_files:
        with pa.ipc.open_stream(arrow_file) as reader:
            table = reader.read_all()
            df = pl.from_arrow(table)
            df = _detect_and_normalize(df)
            dfs.append(df)

    # Concatenate all dataframes and convert to lazy
    combined = pl.concat(dfs)
    return combined.lazy()


def _apply_cutoff_filter(df: pl.DataFrame, cutoff_dates: dict[str, str]) -> pl.DataFrame:
    """Drop rows where date exceeds the user's cutoff date.

    Args:
        df: DataFrame with ``user_id`` and ``date`` columns.
        cutoff_dates: ``{user_id: "YYYY-MM-DD"}`` per-user cutoff dates.

    Returns:
        Filtered DataFrame.
    """
    import datetime as dt

    # Ensure date column is Date type for comparison
    date_col = df["date"]
    if date_col.dtype == pl.Utf8:
        df = df.with_columns(pl.col("date").str.to_date().alias("date"))
    elif date_col.dtype != pl.Date:
        df = df.with_columns(pl.col("date").cast(pl.Date))

    # Build cutoff lookup DataFrame (only for users present in this chunk)
    chunk_users = set(df["user_id"].unique().to_list())
    cutoff_rows = [
        {"user_id": uid, "_cutoff": dt.date.fromisoformat(cutoff_dates[uid])}
        for uid in chunk_users
        if uid in cutoff_dates
    ]
    if not cutoff_rows:
        return df

    cutoff_df = pl.DataFrame(cutoff_rows).with_columns(pl.col("_cutoff").cast(pl.Date))
    df = df.join(cutoff_df, on="user_id", how="left")
    df = df.filter(pl.col("_cutoff").is_null() | (pl.col("date") <= pl.col("_cutoff")))
    return df.drop("_cutoff")


def build_user_features_chunked(
    arrow_dir: Path,
    output_path: Path | None = None,
    splits: list[str] | None = None,
    checkpoint_dir: Path | None = None,
    max_nonwear_minutes: int | None = None,
    variance_filter: bool = True,
    cutoff_dates: dict[str, str] | None = None,
) -> pl.DataFrame:
    """Build user-level features from Arrow files using chunked loading.

    Processes one Arrow file at a time to reduce peak RAM from ~158GB to ~1-2GB.
    For each file: load → extract daily features → drop raw data → save checkpoint.
    Then aggregate all daily features to user level.

    Polars parallelizes feature extraction internally (uses all available cores).
    Checkpointing enables resume on restart — already-processed files are skipped.

    Args:
        arrow_dir: Path to directory containing Arrow files (with train/test/val subdirs)
        output_path: Optional path to write output Parquet file
        splits: Optional list of splits to process (default: all)
        checkpoint_dir: Directory for per-file checkpoints (enables resume).
                        If None, defaults to output_path's parent / "timeseries_daily_chunks".
        max_nonwear_minutes: If set, drop rows where total_nonwear_minutes exceeds
                             this value before feature extraction. E.g. 720 for ≤50% non-wear.
        variance_filter: If True (default), drop rows where a monitored channel
                         has near-zero variance (flat signal = sensor malfunction).
        cutoff_dates: Optional ``{user_id: "YYYY-MM-DD"}`` per-user data cutoff.
                      Rows with ``date > cutoff_dates[user_id]`` are dropped before
                      feature extraction.  Used to cap future data relative to
                      label measurement dates.

    Returns:
        DataFrame with one row per user and feature columns
    """
    import time

    arrow_dir = Path(arrow_dir)
    if not arrow_dir.exists():
        raise FileNotFoundError(f"Directory not found: {arrow_dir}")

    if splits is None:
        splits = ["train", "test", "val"]

    arrow_files = []
    for split in splits:
        split_dir = arrow_dir / split
        if split_dir.exists():
            arrow_files.extend(sorted(split_dir.glob("*.arrow")))

    # Flat directory fallback (MHC-B daily_hf has no subdirs)
    if not arrow_files:
        arrow_files = sorted(arrow_dir.glob("data-*.arrow"))

    if not arrow_files:
        raise ValueError(f"No Arrow files found in {arrow_dir}")

    # Checkpoint directory for per-file daily-feature chunks.
    if checkpoint_dir is None:
        parent = Path(output_path).parent if output_path is not None else arrow_dir.parent
        checkpoint_dir = parent / "timeseries_daily_chunks"
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def chunk_path(arrow_file: Path) -> Path:
        return checkpoint_dir / f"{arrow_file.parent.name}_{arrow_file.stem}.parquet"

    todo = [(i, f) for i, f in enumerate(arrow_files) if not chunk_path(f).exists()]
    done = len(arrow_files) - len(todo)

    logger.info("Polars thread pool: %d threads", pl.thread_pool_size())

    if done > 0:
        logger.info(
            "Resuming: %d/%d chunks already done, %d remaining",
            done,
            len(arrow_files),
            len(todo),
        )
    else:
        logger.info("Processing %d Arrow files (chunked)...", len(arrow_files))

    t0 = time.time()
    daily_extractors = get_all_daily_extractors()

    for idx, arrow_file in todo:
        out_path = chunk_path(arrow_file)
        with pa.ipc.open_stream(arrow_file) as reader:
            table = reader.read_all()
            df = pl.from_arrow(table)
            del table  # free PyArrow memory before cast
            # Drop columns not needed for feature extraction
            keep = {
                "user_id",
                "date",
                "data",
                "values",
                "nonwear_vector",
                "total_nonwear_minutes",
                "timestamp",
                "channel_variance",
            }
            df = df.select([c for c in df.columns if c in keep])
            df = _detect_and_normalize(df)
        before = len(df)
        if max_nonwear_minutes is not None and "total_nonwear_minutes" in df.columns:
            df = df.filter(pl.col("total_nonwear_minutes") <= max_nonwear_minutes)
        after_nonwear = len(df)
        if variance_filter:
            df = apply_variance_filter(df)
        after_all = len(df)
        if after_all == 0:
            logger.warning(
                "  SKIP %s: all %d rows filtered (nonwear: -%d, variance: -%d)",
                arrow_file.name,
                before,
                before - after_nonwear,
                after_nonwear - after_all,
            )
            continue
        if after_all < before:
            logger.info(
                "  %s: filtered %d/%d rows (nonwear: -%d, variance: -%d)",
                arrow_file.name,
                before - after_all,
                before,
                before - after_nonwear,
                after_nonwear - after_all,
            )
        # Drop channel_variance before feature extraction (not needed downstream)
        if "channel_variance" in df.columns:
            df = df.drop("channel_variance")
        n_rows = len(df)
        df_daily = df.lazy().with_columns(daily_extractors).collect()
        cols_to_keep = [
            c for c in df_daily.columns if c not in ("data", "nonwear_vector", "timestamp")
        ]
        df_daily = df_daily.select(cols_to_keep)
        df_daily.write_parquet(out_path)
        del df, df_daily

        elapsed = time.time() - t0
        done_now = done + sum(1 for j, _ in todo if j <= idx)
        remaining = len(arrow_files) - done_now
        rate = done_now / elapsed if elapsed > 0 else 0
        eta = remaining / rate if rate > 0 else 0
        logger.info(
            "  [%d/%d] %s: %d rows (%.0fs elapsed, ~%.0fs remaining)",
            done_now,
            len(arrow_files),
            arrow_file.name,
            n_rows,
            elapsed,
            eta,
        )

    # Concatenate all checkpoint files
    logger.info("Concatenating all daily feature chunks...")
    chunk_files = sorted(checkpoint_dir.glob("*.parquet"))
    all_daily = pl.concat([pl.read_parquet(f) for f in chunk_files])

    # Future-data cutoff: drop rows after each user's cutoff date.
    # Applied at aggregation time so checkpoints remain reusable.
    if cutoff_dates is not None:
        before_cutoff = all_daily.shape[0]
        all_daily = _apply_cutoff_filter(all_daily, cutoff_dates)
        after_cutoff = all_daily.shape[0]
        logger.info(
            "Cutoff filter: %d -> %d rows (-%d, %.1f%% removed)",
            before_cutoff,
            after_cutoff,
            before_cutoff - after_cutoff,
            (before_cutoff - after_cutoff) / before_cutoff * 100 if before_cutoff > 0 else 0,
        )

    # Aggregate to user level
    logger.info("Aggregating to user level...")
    user_features = (
        all_daily.lazy()
        .group_by("user_id")
        .agg(
            pl.len().alias("total_days"),
            *get_all_user_aggregators(),
        )
        .collect()
    )
    del all_daily

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        user_features.write_parquet(output_path)
        logger.info(
            "Wrote %d users x %d features to %s",
            user_features.shape[0],
            user_features.shape[1],
            output_path,
        )

    return user_features


def build_user_features(
    arrow_dir: Path,
    output_path: Path | None = None,
    splits: list[str] | None = None,
) -> pl.DataFrame:
    """Build user-level features from Arrow files.

    Main pipeline function that:
    1. Loads Arrow files lazily
    2. Applies all day-level feature extractors
    3. Aggregates to user-level (median, IQR, etc.)
    4. Optionally writes to Parquet

    Args:
        arrow_dir: Path to directory containing Arrow files
        output_path: Optional path to write output Parquet file
        splits: Optional list of splits to process (default: all)

    Returns:
        DataFrame with one row per user and feature columns

    Example:
        >>> result = build_user_features(
        ...     Path("data/processed/daily_hf"),
        ...     Path("data/features/xgboost/user_features.parquet")
        ... )
        >>> print(result.shape)
    """
    # Load data lazily
    lf = load_arrow_files(arrow_dir, splits=splits)

    # Apply day-level feature extraction
    lf_with_daily = lf.with_columns(get_all_daily_extractors())

    # Aggregate to user level
    user_features = lf_with_daily.group_by("user_id").agg(
        pl.len().alias("total_days"),
        *get_all_user_aggregators(),
    )

    # Collect results (streaming for large datasets)
    result = user_features.collect(streaming=True)

    # Write to Parquet if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.write_parquet(output_path)
        logger.info(
            "Wrote %d users x %d features to %s",
            result.shape[0],
            result.shape[1],
            output_path,
        )

    return result


def extract_daily_features(
    arrow_dir: Path,
    output_path: Path | None = None,
    splits: list[str] | None = None,
) -> pl.DataFrame:
    """Extract day-level features without user aggregation.

    Useful for:
    - Validating day-level extraction
    - Building day-level models
    - Debugging feature values

    Args:
        arrow_dir: Path to directory containing Arrow files
        output_path: Optional path to write output Parquet file
        splits: Optional list of splits to process (default: all)

    Returns:
        DataFrame with one row per user-day and feature columns

    Example:
        >>> daily = extract_daily_features(
        ...     Path("data/processed/daily_hf"),
        ...     splits=["train"]
        ... )
        >>> print(daily.head())
    """
    lf = load_arrow_files(arrow_dir, splits=splits)

    # Apply day-level extraction, keep user_id and date
    daily_features = lf.select(
        pl.col("user_id"),
        pl.col("date"),
        pl.col("total_nonwear_minutes"),
        *get_all_daily_extractors(),
    )

    result = daily_features.collect(streaming=True)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.write_parquet(output_path)
        logger.info(
            "Wrote %d day-records x %d features to %s",
            result.shape[0],
            result.shape[1],
            output_path,
        )

    return result
