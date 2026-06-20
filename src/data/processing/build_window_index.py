r"""Build lightweight window indices from daily_hourly_hf for on-the-fly weekly construction.

Instead of materialising a full weekly_hf dataset, this module scans the
``daily_hourly_hf`` Arrow dataset (which stores one (24, 19) row per user-day)
and emits a compact Parquet index of valid weekly windows.  Each window is a
tuple of HF row indices that can be concatenated at training time to form the
standard (168, 38) tensor expected by ``PairWeekDatasetHFRaw``.

The index supports configurable:
- **window_size**: number of consecutive calendar days per window (default 7)
- **stride**: advance in calendar days between window starts (default 7, non-overlapping)
- **min_data_days**: minimum number of data-present days required per window (default 5)

Usage (library)::

    from data.processing.build_window_index import build_window_index
    df = build_window_index(
        daily_hourly_hf_dir="data/processed/daily_hourly_hf",
        window_size=7,
        stride=7,
        min_data_days=5,
    )
    df.to_parquet("data/processed/window_index_w7_s7_d5.parquet")

Usage (CLI)::

    python -m data.processing.build_window_index \
        --daily-hourly-hf-dir data/processed/daily_hourly_hf \
        --output data/processed/window_index_w7_s7_d5.parquet \
        --window-size 7 --stride 7 --min-data-days 5
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, timedelta
from pathlib import Path

import datasets as hf_ds
import pandas as pd

logger = logging.getLogger(__name__)


def _scan_user_dates(
    ds: hf_ds.Dataset,
) -> dict[str, dict[date, int]]:
    """Build a mapping {user_id: {date: hf_row_index}} from the dataset.

    This only reads the lightweight string columns (user_id, date) via
    memory-mapped Arrow, so it is fast even for large datasets.

    Args:
        ds: HuggingFace Dataset with ``user_id`` (str) and ``date`` (str) columns.

    Returns:
        Nested dict mapping each user to their available dates and the
        corresponding HF row index.
    """
    # Batch access is much faster than row-by-row iteration
    user_ids = ds["user_id"]  # list[str]
    dates_str = ds["date"]  # list[str]  (ISO format YYYY-MM-DD)

    user_date_map: dict[str, dict[date, int]] = {}
    for idx, (uid, d_str) in enumerate(zip(user_ids, dates_str)):
        if uid not in user_date_map:
            user_date_map[uid] = {}
        user_date_map[uid][date.fromisoformat(d_str)] = idx

    return user_date_map


def build_window_index(
    daily_hourly_hf_dir: str | Path,
    window_size: int = 7,
    stride: int = 7,
    min_data_days: int = 5,
    split_file: str | Path | None = None,
    splits: list[str] | None = None,
) -> pd.DataFrame:
    """Build a window index from a daily_hourly_hf dataset.

    For each user, slides a ``window_size``-day window with ``stride``-day
    advance over their available date range.  Windows with fewer than
    ``min_data_days`` days of data are skipped.  Each valid window is
    recorded as one row in the output DataFrame.

    Args:
        daily_hourly_hf_dir: Path to the HuggingFace Arrow dataset on disk.
            Expected columns: ``user_id`` (str), ``date`` (str), ``values``
            (24, 19), plus optional metadata.
        window_size: Number of calendar days per window.
        stride: Calendar-day advance between consecutive window starts for
            the same user.
        min_data_days: Minimum number of data-present days required for a
            window to be considered valid.
        split_file: Optional path to a JSON user-split file.  If provided,
            a ``split`` column is added to the output (train/validation/test).
            Users not in the split file are dropped.
        splits: Optional list of split names to include (e.g. ["train", "validation"]).
            If None, all splits in the file are included.

    Returns:
        DataFrame with columns:
            - ``user_id`` (str): User identifier.
            - ``window_start`` (str): ISO date of the first calendar day.
            - ``n_data_days`` (int): Number of days with data in this window.
            - ``row_indices`` (list[int | None]): Length-``window_size`` list of
              HF dataset row indices.  ``None`` for missing days.
            - ``split`` (str, optional): User split if ``split_file`` given.
    """
    ds_path = Path(daily_hourly_hf_dir)
    logger.info("Loading daily_hourly_hf from %s ...", ds_path)
    ds = hf_ds.load_from_disk(str(ds_path))
    if isinstance(ds, hf_ds.DatasetDict):
        ds = hf_ds.concatenate_datasets(list(ds.values()))
    logger.info("Loaded %d rows.", len(ds))

    # Optional: load user splits for filtering and labelling
    user_to_split: dict[str, str] | None = None
    if split_file is not None:
        split_path = Path(split_file)
        raw = json.loads(split_path.read_text())
        user_to_split = {}
        for split_name, users in raw.items():
            if splits is not None and split_name not in splits:
                continue
            for uid in users:
                user_to_split[uid] = split_name
        logger.info(
            "Loaded split file: %d users across %s.",
            len(user_to_split),
            list(raw.keys()) if splits is None else splits,
        )

    # Scan user→date→row_index mapping
    user_date_map = _scan_user_dates(ds)
    logger.info("Scanned %d users.", len(user_date_map))

    # Build windows
    records: list[dict] = []
    one_day = timedelta(days=1)
    window_td = timedelta(days=window_size)
    stride_td = timedelta(days=stride)

    users_skipped_split = 0
    users_processed = 0
    windows_emitted = 0
    windows_skipped = 0

    for uid, date_to_idx in sorted(user_date_map.items()):
        # Filter by split if requested
        if user_to_split is not None and uid not in user_to_split:
            users_skipped_split += 1
            continue

        users_processed += 1
        sorted_dates = sorted(date_to_idx.keys())
        if not sorted_dates:
            continue

        first_date = sorted_dates[0]
        last_date = sorted_dates[-1]

        # Slide a window_size-day window with the given stride
        pointer = first_date
        while pointer + window_td - one_day <= last_date:
            # Collect row indices for each day in the window
            row_indices: list[int | None] = []
            n_data = 0
            for offset in range(window_size):
                d = pointer + timedelta(days=offset)
                idx = date_to_idx.get(d)
                row_indices.append(idx)
                if idx is not None:
                    n_data += 1

            if n_data >= min_data_days:
                rec = {
                    "user_id": uid,
                    "window_start": pointer.isoformat(),
                    "n_data_days": n_data,
                    "row_indices": row_indices,
                }
                if user_to_split is not None:
                    rec["split"] = user_to_split[uid]
                records.append(rec)
                windows_emitted += 1
            else:
                windows_skipped += 1

            pointer += stride_td

    df = pd.DataFrame(records)

    logger.info(
        "Window index built: %d windows from %d users "
        "(skipped %d windows < %d data days, %d users not in split).",
        windows_emitted,
        users_processed,
        windows_skipped,
        min_data_days,
        users_skipped_split,
    )

    return df


def load_window_index(path: str | Path) -> pd.DataFrame:
    """Load a previously built window index from Parquet.

    The ``row_indices`` column is stored as a JSON-encoded list in Parquet
    (since Parquet doesn't natively support list-of-nullable-int).  This
    function decodes it back to Python lists.

    If the parquet was saved without JSON encoding (e.g. plain ``to_parquet``),
    ``row_indices`` may be numpy arrays with ``NaN`` for missing days.  This
    function normalises those to Python lists with ``None``.

    Args:
        path: Path to the ``.parquet`` file.

    Returns:
        DataFrame with the same schema as ``build_window_index`` output.
    """
    import math

    df = pd.read_parquet(path)
    if "row_indices" in df.columns and len(df) > 0:
        first = df["row_indices"].iloc[0]
        if isinstance(first, str):
            # Stored as JSON strings — decode
            import json as _json

            df["row_indices"] = df["row_indices"].apply(_json.loads)
        elif hasattr(first, "__iter__") and not isinstance(first, (list, str)):
            # Numpy array or similar — convert NaN → None, floats → int
            def _normalise_row_indices(arr):
                return [None if (isinstance(v, float) and math.isnan(v)) else int(v) for v in arr]

            df["row_indices"] = df["row_indices"].apply(_normalise_row_indices)
    return df


def save_window_index(df: pd.DataFrame, path: str | Path) -> None:
    """Save a window index to Parquet.

    Encodes the ``row_indices`` list column as JSON strings for Parquet
    compatibility.

    Args:
        df: Window index DataFrame from ``build_window_index``.
        path: Output ``.parquet`` path.
    """
    import json as _json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df_out = df.copy()
    if "row_indices" in df_out.columns:
        df_out["row_indices"] = df_out["row_indices"].apply(_json.dumps)

    df_out.to_parquet(path, index=False)
    logger.info("Saved window index (%d rows) to %s", len(df_out), path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for building window indices."""
    parser = argparse.ArgumentParser(description="Build window index from daily_hourly_hf dataset.")
    parser.add_argument(
        "--daily-hourly-hf-dir",
        required=True,
        help="Path to the daily_hourly_hf HuggingFace Arrow dataset.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output Parquet path for the window index.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=7,
        help="Number of calendar days per window (default: 7).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=7,
        help="Calendar-day advance between window starts (default: 7).",
    )
    parser.add_argument(
        "--min-data-days",
        type=int,
        default=5,
        help="Minimum data-present days per window (default: 5).",
    )
    parser.add_argument(
        "--split-file",
        default=None,
        help="Optional JSON user-split file (adds 'split' column, filters users).",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Splits to include from the split file (default: all).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    df = build_window_index(
        daily_hourly_hf_dir=args.daily_hourly_hf_dir,
        window_size=args.window_size,
        stride=args.stride,
        min_data_days=args.min_data_days,
        split_file=args.split_file,
        splits=args.splits,
    )

    save_window_index(df, args.output)
    print(f"Done. {len(df)} windows saved to {args.output}")


if __name__ == "__main__":
    main()
