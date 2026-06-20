"""API for accessing per-user device timelines.

Device records describe the iPhone and Apple Watch hardware a participant
used over time.  The underlying data file (``user_device_info.json``) is
produced upstream by the MHC-benchmark build pipeline and ships separately
from ``last_labels.json``.  This module loads it lazily so importing
:mod:`devices` works even when the file is not present in the bundle --
the first query is what raises :class:`FileNotFoundError`.

Timestamps must be tz-naive (the underlying interval bounds are date-only,
read as midnight UTC-naive); passing a tz-aware ``pd.Timestamp`` will raise
``TypeError`` on comparison, mirroring :mod:`labels.api`.

Quick start::

    >>> from devices import get_devices, get_device_timeline
    >>> import pandas as pd
    >>> snap = get_devices("user-123", pd.Timestamp("2020-06-01"))
    >>> if snap.phone is not None:
    ...     print(snap.phone.name)        # 'iPhone 8 Plus'
    >>> if snap.watch is not None:
    ...     print(snap.watch.name)        # 'Apple Watch Series 5 44mm GPS'

Both :func:`get_devices` and :func:`get_device_timeline` (and any direct
``STORE.timelines`` access) trigger the lazy load; whichever runs first
will raise :class:`FileNotFoundError` if the JSON is missing.

The JSON schema is::

    {
      "<healthCode>": {
        "phone": [["2017-12-18", "2023-05-18", "iPhone 8 Plus"], ...],
        "watch": [["2019-04-05", "2023-12-19", "Apple Watch Series 5 44mm GPS"], ...]
      }
    }

Intervals are contiguous-run, non-overlapping, sorted, and inclusive on
both ends.  Calendar gaps inside a same-model run are absorbed by the
upstream build; an unrecognised string falls through to
``iPhone Unknown Model`` / ``Apple Watch Unknown Model``.
"""

from __future__ import annotations

import json
import os
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "labels"
USER_DEVICE_INFO_PATH = Path(os.getenv("USER_DEVICE_INFO_PATH", DATA_DIR / "user_device_info.json"))


@dataclass(frozen=True)
class DeviceInterval:
    """One device a user wore over a contiguous date range.

    Attributes:
        start: First date the device was observed (inclusive).
        end: Last date the device was observed (inclusive).
        name: Full marketing name (e.g. ``"iPhone 8 Plus"``) or
            ``"iPhone Unknown Model"`` / ``"Apple Watch Unknown Model"``.
    """

    start: pd.Timestamp
    end: pd.Timestamp
    name: str

    def contains(self, timestamp: pd.Timestamp) -> bool:
        """Return True iff *timestamp* falls within ``[start, end]`` inclusive."""
        return self.start <= timestamp <= self.end


@dataclass(frozen=True)
class DeviceTimeline:
    """Full per-user, per-class device history.

    The ``_phone_starts`` / ``_watch_starts`` caches are populated in
    ``__post_init__`` so repeated lookups stay O(log N) instead of
    rebuilding the start-keys list per call.
    """

    health_code: str
    phone: tuple[DeviceInterval, ...] = field(default_factory=tuple)
    watch: tuple[DeviceInterval, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Precompute parallel start-key lists for O(log N) bisect lookups."""
        object.__setattr__(self, "_phone_starts", [iv.start for iv in self.phone])
        object.__setattr__(self, "_watch_starts", [iv.start for iv in self.watch])

    def phone_at(self, timestamp: pd.Timestamp) -> DeviceInterval | None:
        """Return the phone interval covering *timestamp*, or None."""
        return _interval_at(self.phone, self._phone_starts, timestamp)

    def watch_at(self, timestamp: pd.Timestamp) -> DeviceInterval | None:
        """Return the watch interval covering *timestamp*, or None."""
        return _interval_at(self.watch, self._watch_starts, timestamp)

    def snapshot_at(self, timestamp: pd.Timestamp) -> DeviceSnapshot:
        """Return both phone and watch active at *timestamp*."""
        return DeviceSnapshot(
            health_code=self.health_code,
            timestamp=timestamp,
            phone=self.phone_at(timestamp),
            watch=self.watch_at(timestamp),
        )


@dataclass(frozen=True)
class DeviceSnapshot:
    """The phone and watch a user was wearing at a single timestamp.

    Either field may be ``None`` if no interval covers the query time
    (e.g. before the user's first record or in a gap between watches).
    """

    health_code: str
    timestamp: pd.Timestamp
    phone: DeviceInterval | None
    watch: DeviceInterval | None


def _interval_at(
    intervals: tuple[DeviceInterval, ...],
    starts: list[pd.Timestamp],
    timestamp: pd.Timestamp,
) -> DeviceInterval | None:
    """Binary-search the sorted interval list for one containing *timestamp*.

    Intervals are non-overlapping and sorted by start, so the candidate is
    the rightmost interval whose start is ``<= timestamp``. ``starts`` is
    the precomputed parallel list of interval starts (kept on the timeline
    so repeated queries don't re-materialize it).
    """
    if not intervals:
        return None
    idx = bisect_right(starts, timestamp) - 1
    if idx < 0:
        return None
    candidate = intervals[idx]
    return candidate if candidate.contains(timestamp) else None


class _DeviceStore:
    """Lazily loads ``user_device_info.json`` and serves per-user lookups."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._timelines: dict[str, DeviceTimeline] | None = None

    @property
    def timelines(self) -> dict[str, DeviceTimeline]:
        """All per-user timelines, loaded on first access."""
        if self._timelines is None:
            self._timelines = _load_device_info(self.path)
        return self._timelines

    def get_timeline(self, health_code: str) -> DeviceTimeline:
        """Return the timeline for *health_code* or raise ``KeyError``."""
        timeline = self.timelines.get(health_code)
        if timeline is None:
            raise KeyError(f"Unknown healthCode in device info: {health_code}")
        return timeline


def _load_device_info(path: Path) -> dict[str, DeviceTimeline]:
    """Read the JSON file and build typed timelines.

    Raises:
        FileNotFoundError: With a guidance message if the file is missing.
            The accessor is intentionally importable without the data file
            present, so this error surfaces only on first lookup.
        ValueError: If a user has overlapping intervals within phone or
            watch (the upstream contract is non-overlapping; failing loudly
            beats silently returning the wrong device).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"user_device_info.json not found at {path}. "
            "This file ships separately from the labels bundle; "
            "set USER_DEVICE_INFO_PATH or place it under data/labels/."
        )
    with path.open("r") as handle:
        raw = json.load(handle)
    out: dict[str, DeviceTimeline] = {}
    for health_code, classes in raw.items():
        out[health_code] = DeviceTimeline(
            health_code=health_code,
            phone=_parse_intervals(classes.get("phone", []), health_code, "phone"),
            watch=_parse_intervals(classes.get("watch", []), health_code, "watch"),
        )
    return out


def _parse_intervals(
    rows: list[tuple[str, str, str]],
    health_code: str,
    device_class: str,
) -> tuple[DeviceInterval, ...]:
    """Turn ``[[start, end, name], ...]`` into typed, sorted, non-overlapping intervals."""
    parsed = [
        DeviceInterval(
            start=pd.Timestamp(start),
            end=pd.Timestamp(end),
            name=name,
        )
        for start, end, name in rows
    ]
    parsed.sort(key=lambda iv: iv.start)
    for prev, curr in zip(parsed, parsed[1:]):
        if curr.start <= prev.end:
            raise ValueError(
                f"Overlapping {device_class} intervals for healthCode "
                f"{health_code!r}: [{prev.start.date()}..{prev.end.date()}] "
                f"and [{curr.start.date()}..{curr.end.date()}]."
            )
    return tuple(parsed)


STORE = _DeviceStore(USER_DEVICE_INFO_PATH)


def get_devices(health_code: str, timestamp: pd.Timestamp) -> DeviceSnapshot:
    """Return the phone and watch active for *health_code* at *timestamp*.

    Args:
        health_code: The participant's health code.
        timestamp: Reference timestamp for the lookup. Must be tz-naive,
            consistent with the date-only interval bounds (mirrors the
            :mod:`labels.api` convention).

    Returns:
        A :class:`DeviceSnapshot` with ``phone`` and ``watch`` fields.
        Either may be ``None`` if no interval covers *timestamp*.

    Raises:
        KeyError: If *health_code* has no device records.
        FileNotFoundError: If ``user_device_info.json`` is not on disk.
    """
    return STORE.get_timeline(health_code).snapshot_at(timestamp)


def get_device_timeline(health_code: str) -> DeviceTimeline:
    """Return the full :class:`DeviceTimeline` for *health_code*.

    Args:
        health_code: The participant's health code.

    Raises:
        KeyError: If *health_code* has no device records.
        FileNotFoundError: If ``user_device_info.json`` is not on disk.
    """
    return STORE.get_timeline(health_code)
