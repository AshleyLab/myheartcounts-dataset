"""Calendar-gap-aware series windowing — the raw public form of one continuous window.

Mirrors the algorithm in ``models.tsfm`` (``build_user_timeline`` + ``build_window``):
lay each eligible day at its calendar position, NaN-fill gaps, crop to the last
observed hour, then take the last ``window_hours`` hours (left-padding when the
history is shorter). Unlike the TSFM internal path this emits the **raw public
contract** — values with NaN at missing positions plus a ``1 = missing`` mask — so a
model does its own normalization/cleaning. Pure numpy + stdlib (no heavy deps), so it
is unit-testable on synthetic days without the dataset.
"""

from __future__ import annotations

from datetime import date as _date

import numpy as np

_HOURS_PER_DAY = 24


def _to_date(d) -> _date:
    """Parse a ``YYYY-MM-DD`` (or longer) value to a ``date``."""
    s = d if isinstance(d, str) else str(d)
    return _date.fromisoformat(s[:10])


def _with_mask(window: np.ndarray) -> np.ndarray:
    """Concatenate ``(T, C)`` values (NaN at missing) with a ``1 = missing`` mask -> ``(T, 2C)``."""
    mask = np.isnan(window).astype(np.float32)
    return np.concatenate([window, mask], axis=-1)


def series_window(
    values, dates, window_hours: int, n_channels: int = 19, hours_per_day: int = _HOURS_PER_DAY
) -> np.ndarray:
    """One participant's continuous series window as ``(window_hours, 2*n_channels)``.

    Args:
        values: ``(n_days, hours_per_day, n_channels)`` day segments, NaN at missing,
            **date-ascending** and aligned with ``dates``.
        dates: per-day ``YYYY-MM-DD`` strings, ascending and aligned with ``values``.
        window_hours: window length in hours (e.g. 2048).
        n_channels: sensor channels per timestep (19); output has ``2*n_channels`` (values + mask).
        hours_per_day: timesteps per day in ``values`` (24 for the hourly store).

    Returns:
        ``(window_hours, 2*n_channels)`` float32: values (channels ``0..n_channels-1``,
        NaN at missing) concatenated with a ``1 = missing`` mask. Left-padded (values
        NaN, mask 1) when the observed history is shorter than ``window_hours``; the
        calendar gaps between non-consecutive eligible days are filled the same way.
    """
    values = np.asarray(values, dtype=np.float32)
    if len(dates) == 0:
        return _with_mask(np.full((window_hours, n_channels), np.nan, dtype=np.float32))

    first = _to_date(dates[0])
    last = _to_date(dates[-1])
    n_days = (last - first).days + 1
    timeline = np.full((n_days * hours_per_day, n_channels), np.nan, dtype=np.float32)
    for day_values, d in zip(values, dates):
        off = (_to_date(d) - first).days * hours_per_day
        timeline[off : off + hours_per_day] = day_values

    # Crop trailing all-missing hours so the window ends at the last observed hour.
    observed = ~np.isnan(timeline).all(axis=1)
    pos = np.flatnonzero(observed)
    history = timeline[: int(pos[-1]) + 1] if pos.size else timeline[:0]

    if history.shape[0] >= window_hours:
        window = history[-window_hours:]
    else:
        window = np.full((window_hours, n_channels), np.nan, dtype=np.float32)
        window[window_hours - history.shape[0] :] = history
    return _with_mask(window)
