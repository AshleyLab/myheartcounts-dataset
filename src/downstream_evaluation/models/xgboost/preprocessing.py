"""Polars/numpy equivalents of upstream NaN-masking and variance-filtering transforms.

The upstream SSL pipeline applies two quality steps before feeding daily data to
the Transformer encoder:

1. ``ZeroToNaNTransform`` — converts physiologically impossible zeros to NaN so
   normalization ignores them.
2. ``LowChannelVarianceFilter`` — drops daily samples where a monitored channel has
   near-zero variance (flat signal = sensor malfunction or device not worn).

This module provides Polars/numpy equivalents of those two steps for the XGBoost
feature extraction pipeline.

Variance thresholds are imported from the canonical source
(``data.processing.hf_config.DEFAULT_VARIANCE_THRESHOLDS``) so configuration
stays in one place.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pyarrow as pa

from data.processing.hf_config import DEFAULT_VARIANCE_THRESHOLDS, N_CHANNELS

from .constants import (
    IPHONE_DISTANCE,
    IPHONE_STEPS,
    SLEEP_ASLEEP,
    SLEEP_INBED,
    WATCH_DISTANCE,
    WATCH_ENERGY,
    WATCH_HR,
    WATCH_STEPS,
)

# Heart rate: per-value zero -> NaN (0 bpm is physiologically impossible).
_HR_CHANNEL: int = WATCH_HR  # 5

# Sleep channels: zeros -> NaN when total detected sleep is > 0 but < 3 hours.
# Short sessions are mostly sleep-detection errors where the non-sleep (zero)
# portions are unreliable.  Matches ZeroToNaNTransform (nan_transforms.py).
_SHORT_SLEEP_NAN_CHANNELS: tuple[int, ...] = (SLEEP_ASLEEP, SLEEP_INBED)  # 7, 8
_SHORT_SLEEP_THRESHOLD: float = 180.0  # minutes (3 hours)

# Channels where an all-zero channel -> all-NaN (device not worn/carried).
# Matches ZeroToNaNTransform.__init__ defaults. Flights (ch 2) excluded
# because ~55% of samples are legitimately all-zero.
_ALL_ZERO_NAN_CHANNELS: tuple[int, ...] = (
    IPHONE_STEPS,  # 0
    IPHONE_DISTANCE,  # 1
    WATCH_STEPS,  # 3
    WATCH_DISTANCE,  # 4
    WATCH_ENERGY,  # 6
)

_MINUTES_PER_DAY = 1440


def apply_zero_to_nan(df: pl.DataFrame) -> pl.DataFrame:
    """Convert sensor zeros to NaN in the ``data`` column.

    Polars equivalent of the dataset's ``ZeroToNaNTransform``.

    Applied per-row to the ``data`` column (Array(List(Float32), 19)):

    1. **Heart rate (ch 5)**: every 0 -> NaN (0 bpm is impossible).
    2. **Steps/distance/energy (ch 0, 1, 3, 4, 6)**: if ALL values in the
       channel are 0 (or NaN), replace the entire channel with NaN.
       Individual zeros are valid (e.g. sitting still).
    3. **Sleep (ch 7, 8)**: if total detected sleep is > 0 but < 180 min
       (3 hours), set the zero portions to NaN. Short sessions are mostly
       detection errors where the "not asleep"/"not in bed" signal is
       unreliable. (Matches ZeroToNaNTransform.)
    4. **Flights (ch 2), workout channels (ch 9-18)**: untouched.

    Args:
        df: DataFrame with a ``data`` column of type Array(List(Float32), 19).

    Returns:
        DataFrame with the ``data`` column modified in-place (new column object,
        same name).
    """
    n = len(df)
    if n == 0:
        return df

    T = _MINUTES_PER_DAY

    # Extract all 19 channels as 2D numpy arrays (n x T) via Polars explode.
    arrays: list[np.ndarray] = []
    for i in range(N_CHANNELS):
        flat = df["data"].arr.get(i).explode().to_numpy(allow_copy=False)
        arrays.append(flat.reshape(n, T).copy())  # writable copy

    # 1. HR (ch 5): per-value 0 -> NaN
    hr = arrays[_HR_CHANNEL]
    hr[hr == 0] = np.nan

    # 2. Activity channels: rows where ALL values are 0 or NaN -> entire row NaN
    for ch in _ALL_ZERO_NAN_CHANNELS:
        arr = arrays[ch]
        dead_rows = np.all((arr == 0) | np.isnan(arr), axis=1)
        arr[dead_rows] = np.nan

    # 3. Sleep channels: zeros -> NaN when 0 < total_sleep < threshold
    for ch in _SHORT_SLEEP_NAN_CHANNELS:
        arr = arrays[ch]
        total_sleep = np.nansum(arr, axis=1)  # sum ignoring NaN, per row
        short = (total_sleep > 0) & (total_sleep < _SHORT_SLEEP_THRESHOLD)
        arr[np.ix_(short, np.arange(T))] = np.where(arr[short] == 0, np.nan, arr[short])

    # Reconstruct Array(List(Float32), 19) via PyArrow.
    stacked = np.empty((n, N_CHANNELS, T), dtype=np.float32)
    for i, arr in enumerate(arrays):
        stacked[:, i, :] = arr
    flat_values = stacked.ravel()
    offsets = np.arange(0, n * N_CHANNELS + 1, dtype=np.int64) * T
    inner = pa.ListArray.from_arrays(offsets, flat_values)
    outer = pa.FixedSizeListArray.from_arrays(inner, N_CHANNELS)
    new_data = pl.from_arrow(outer)

    return df.with_columns(new_data.alias("data"))


def apply_variance_filter(
    df: pl.DataFrame,
    thresholds: dict[int, float] | None = None,
) -> pl.DataFrame:
    """Drop daily samples where a monitored channel has near-zero variance.

    Polars equivalent of the dataset's ``LowChannelVarianceFilter``.

    Reads the pre-computed ``channel_variance`` column (list of per-channel
    variance values) and rejects rows where any monitored channel's variance
    falls below its threshold. Channels with NaN variance (insufficient data)
    are skipped, not penalized.

    Silent no-op if the ``channel_variance`` column is absent (e.g. older
    Arrow files produced before that column was added).

    Args:
        df: DataFrame that may contain a ``channel_variance`` column.
        thresholds: Mapping of channel index -> minimum variance. Defaults to
            ``DEFAULT_VARIANCE_THRESHOLDS`` from ``data.processing.hf_config``.

    Returns:
        Filtered DataFrame (rows with low-variance channels removed).
    """
    if "channel_variance" not in df.columns:
        return df

    if thresholds is None:
        thresholds = DEFAULT_VARIANCE_THRESHOLDS

    var_col = df["channel_variance"].to_list()
    keep_mask: list[bool] = []
    for variances in var_col:
        ok = True
        for ch_idx, min_var in thresholds.items():
            if ch_idx < len(variances):
                v = variances[ch_idx]
                # NaN/None means insufficient data (<2 valid values) — skip.
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    continue
                if v < min_var:
                    ok = False
                    break
        keep_mask.append(ok)

    return df.filter(pl.Series(keep_mask))


def build_cutoff_dates(max_future_days: int = 365) -> dict[str, str]:
    """Compute per-user data cutoff dates from label measurement dates.

    For each user, finds the latest label measurement date across all target
    labels, then adds *max_future_days* to get the cutoff date.  Daily data
    after this cutoff is excluded from feature extraction to prevent models
    from using wearable data far in the future of any label measurement.

    Args:
        max_future_days: Maximum days of data to allow after the user's
            latest label date.  Default 365 (1 year).

    Returns:
        ``{user_id: "YYYY-MM-DD"}`` mapping of per-user cutoff dates.
    """
    import datetime as dt
    import logging

    import pandas as pd

    from labels.api import STORE, TARGET_NAMES

    _log = logging.getLogger(__name__)

    raw_index = STORE.labels_index._index  # noqa: SLF001
    user_latest: dict[str, dt.date] = {}

    for label in TARGET_NAMES:
        per_label = raw_index.get(label, {})
        for uid, series in per_label.items():
            if not series.timestamps_ns:
                continue
            for ts_ns in series.timestamps_ns:
                label_date = pd.Timestamp(ts_ns).date()
                if uid not in user_latest or label_date > user_latest[uid]:
                    user_latest[uid] = label_date

    delta = dt.timedelta(days=max_future_days)
    cutoffs = {uid: (d + delta).isoformat() for uid, d in user_latest.items()}
    _log.info("Built cutoff dates: %d users, max_future_days=%d", len(cutoffs), max_future_days)
    return cutoffs
