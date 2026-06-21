"""DataLoader — the single per-participant segment materializer.

Loads the row-aligned segment source **once**, indexes it by ``(user_id, date)``, and
materializes each participant's eligible segments on demand — per-(task, split) cohort
binding (``bind``) and whole-history access (``user_segments`` / ``segment_store``)
share the one read of ``daily_hourly_hf``.

Keyed by ``(user_id, date)`` (not lookup row positions), so the same selection works
regardless of which segment source / resolution backs it — no lookup↔segment
row-alignment is assumed.

The data handed to a model is **raw** — values with NaN at missing positions plus
the missingness mask. Normalization is the model's concern: encoders z-score with
train-split stats; the Linear baseline pools raw values directly.

See the "Data layer" section of ``src/downstream_evaluation/README.md``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from downstream_evaluation.data.provider import TaskData

logger = logging.getLogger(__name__)

# Channels and hours per day in daily_hourly_hf segments.
_N_CHANNELS = 19
_HOURS_PER_DAY = 24
_MINUTES_PER_DAY = 1440  # daily_hf minute resolution


@dataclass
class ParticipantSegments:
    """One participant's eligible segments (raw; ``values`` carry NaN at missing).

    Attributes:
        values: ``(n_segments, T, 19)`` sensor values, NaN where missing.
        mask: ``(n_segments, T, 19)`` missingness mask (1 = missing, 0 = observed).
    """

    values: np.ndarray
    mask: np.ndarray

    def as_array(self) -> np.ndarray:
        """The public ``(n_segments, T, 38)`` form handed to a :class:`~openmhc.Method`:
        values (channels 0-18, NaN at missing) concatenated with the missingness mask
        (channels 19-37). Both halves are already float32, so this is lossless."""
        return np.concatenate(
            [
                np.asarray(self.values, dtype=np.float32),
                np.asarray(self.mask, dtype=np.float32),
            ],
            axis=-1,
        )


def _day(d) -> str:
    """Normalize a date value to a ``YYYY-MM-DD`` key."""
    s = d if isinstance(d, str) else str(d)
    return s[:10]


def _to_public_minute(values_nan_cf: np.ndarray) -> np.ndarray:
    """``(n, 19, 1440)`` NaN-at-missing channel-first -> public ``(n, 1440, 38)``.

    Transposes to time-first and appends a ``1 = missing`` mask (``isnan``), mirroring
    the hourly :meth:`ParticipantSegments.as_array` contract. Pure numpy (no torch / no
    dataset), so the layout is unit-testable in isolation.
    """
    v = np.ascontiguousarray(np.asarray(values_nan_cf, dtype=np.float32).transpose(0, 2, 1))
    mask = np.isnan(v).astype(np.float32)
    return np.concatenate([v, mask], axis=-1)


class _RawDailyRows:
    """Row-position view of the store in the raw on-disk form.

    Each row is ``{"values": (24, 19) zero-filled, "mask": (24, 19)}`` — the
    ``daily_hourly_ds`` shape window-index consumers expect
    (:class:`~data.datasets.indexed_week_dataset.IndexedWeekDataset`). Row order
    matches the source dataset, so window-index row positions apply directly.
    """

    def __init__(self, values: np.ndarray, mask: np.ndarray) -> None:
        self._values = values
        self._mask = mask

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, idx: int) -> dict:
        return {
            # NaN-at-masked → the stored zero-filled form (masked hours are 0.0 on disk).
            "values": np.nan_to_num(self._values[int(idx)], nan=0.0),
            "mask": self._mask[int(idx)],
        }


def prepare_daily_hourly_hf(ds) -> "hf_ds.Dataset":
    """Convert a daily_hourly_hf dataset to the downstream-pipeline format.

    daily_hourly_hf stores values/mask as (19, 24) channels-first, zero-filled
    (no NaN in values, separate mask column).  The downstream pipeline expects
    (24, 19) time-first with NaN where mask==1.

    This function applies:
      1. Transpose values/mask from (C, H) = (19, 24) → (H, C) = (24, 19)
      2. Restore NaN in values where mask == 1 (so nanmean/nanstd work correctly)

    Note: HF Dataset.map() cannot change the Array2D shape declared in the
    schema, so we rebuild the dataset from a generator with the correct
    (24, 19) features schema.

    Args:
        ds: HuggingFace Dataset from daily_hourly_hf (each row has
            ``values`` (19, 24) and ``mask`` (19, 24)).

    Returns:
        Transformed HuggingFace Dataset with ``values`` (24, 19) containing
        NaN for missing positions and ``mask`` (24, 19).
    """
    import datasets as hf_ds

    logger.info(
        "Preparing daily_hourly_hf: transposing (19,24)->(24,19) and restoring NaN "
        "(%d samples)", len(ds),
    )

    # Bulk-read values and mask, transpose, restore NaN
    all_vals = np.asarray(ds["values"], dtype=np.float32)  # (N, 19, 24)
    all_mask = np.asarray(ds["mask"], dtype=np.float32)  # (N, 19, 24)

    # Transpose: (N, 19, 24) → (N, 24, 19)
    all_vals = np.ascontiguousarray(all_vals.transpose(0, 2, 1))
    all_mask = np.ascontiguousarray(all_mask.transpose(0, 2, 1))

    # Restore NaN where mask == 1 (missing) so nanmean/nanstd work
    all_vals[all_mask > 0.5] = np.nan

    # Read metadata columns
    user_ids = ds["user_id"]
    dates = ds["date"]
    n_valid_hours = ds["n_valid_hours"]

    # Build new dataset with corrected schema (24, 19) instead of (19, 24)
    new_features = hf_ds.Features({
        "values": hf_ds.Array2D(shape=(_HOURS_PER_DAY, _N_CHANNELS), dtype="float32"),
        "mask": hf_ds.Array2D(shape=(_HOURS_PER_DAY, _N_CHANNELS), dtype="float32"),
        "user_id": hf_ds.Value("string"),
        "date": hf_ds.Value("string"),
        "n_valid_hours": hf_ds.Value("int32"),
    })

    new_ds = hf_ds.Dataset.from_dict(
        {
            "values": all_vals,
            "mask": all_mask,
            "user_id": user_ids,
            "date": dates,
            "n_valid_hours": n_valid_hours,
        },
        features=new_features,
    )

    logger.info(
        "Prepared: %d samples, values shape=%s",
        len(new_ds),
        np.asarray(new_ds[0]["values"]).shape,
    )
    return new_ds


class DataLoader:
    """Load the segment source once (lazily); materialize per-participant segments by date."""

    def __init__(
        self, data_dir: str | None, granularity: str = "daily", resolution: str = "hourly"
    ) -> None:
        """Args:
        data_dir: dataset root (``MHC_DATA_DIR`` / openmhc cache if ``None``).
        granularity: only ``"daily"`` is supported (hourly daily segments).
        resolution: which segment store backs the loader.
            ``"hourly"`` materializes the small ``daily_hourly_hf`` store in RAM and serves
            it via the array methods (``participant`` / ``user_days`` / ``segment_store`` /
            ``as_daily_rows``). ``"minute"`` indexes the large ``daily_hf`` minute store
            lazily (mmap; ``values`` fetched one row at a time) and serves it via
            ``participant_minute`` — that store is far too big to hold in RAM.

        Construction is cheap; the segment source is read + indexed on first access
        (so a cache-based model that never touches raw data pays nothing).
        """
        if granularity != "daily":
            raise NotImplementedError(
                f"DataLoader supports 'daily' granularity for now, got {granularity!r}"
            )
        if resolution not in ("hourly", "minute"):
            raise ValueError(f"resolution must be 'hourly' or 'minute', got {resolution!r}")
        self.granularity = granularity
        self.resolution = resolution
        self._data_dir = data_dir
        self._values: np.ndarray | None = None  # (N, 24, 19) f32, NaN at missing [hourly]
        self._mask: np.ndarray | None = None  # (N, 24, 19) f32 [hourly]
        self._minute_vals = None  # lazy mmap'd (19, 1440) values column [minute]
        self._users: np.ndarray | None = None  # per-row user_id
        self._dates: np.ndarray | None = None  # per-row YYYY-MM-DD
        self._row_by_key: dict[tuple[str, str], int] | None = None
        self._rows_by_user: dict[str, list[int]] | None = None

    def _ensure_loaded(self) -> None:
        """Read + index the segment source on first access (one read per run)."""
        if self._row_by_key is not None:
            return
        import datasets as hf_ds

        from openmhc._evaluate import _DatasetPaths

        paths = _DatasetPaths.from_root(self._data_dir)
        if self.resolution == "minute":
            self._load_minute_index(hf_ds, paths)
            return
        ds = hf_ds.load_from_disk(str(paths.daily_hourly_hf))
        if isinstance(ds, hf_ds.DatasetDict):
            ds = hf_ds.concatenate_datasets(list(ds.values()))
        ds = prepare_daily_hourly_hf(ds)  # (24,19) time-first, NaN at missing

        # Keep as numpy for fast select-by-index.
        self._values = np.asarray(ds["values"], dtype=np.float32)  # (N, 24, 19)
        self._mask = np.asarray(ds["mask"], dtype=np.float32)  # (N, 24, 19)
        self._users = np.asarray(ds["user_id"], dtype=object).astype(str)  # per-row user_id
        self._dates = np.asarray([_day(d) for d in ds["date"]], dtype=object)

        # (user_id, date) -> row, and user_id -> row list in store order. Built once;
        # reused across all tasks/splits.
        self._row_by_key = {}
        self._rows_by_user = {}
        for i, (u, d) in enumerate(zip(self._users, self._dates)):
            self._row_by_key[(u, d)] = i
            self._rows_by_user.setdefault(u, []).append(i)

    def _load_minute_index(self, hf_ds, paths) -> None:
        """Index the large minute store (``daily_hf``) without materializing it: build the
        ``(user_id, date) -> row`` map from the metadata columns and keep ``values`` mmap'd,
        fetched one row at a time by :meth:`participant_minute`. ``daily_hf`` is the
        *unfiltered* minute store, so eligibility is the caller's concern — supplied as the
        ``dates`` argument, exactly as the hourly :meth:`participant` takes ``td.dates``."""
        ds = hf_ds.load_from_disk(str(paths.daily_hf))
        if isinstance(ds, hf_ds.DatasetDict):
            ds = hf_ds.concatenate_datasets(list(ds.values()))
        self._users = np.asarray(ds["user_id"], dtype=object).astype(str)
        self._dates = np.asarray([_day(d) for d in ds["date"]], dtype=object)
        self._minute_vals = ds.select_columns(["values"])  # mmap'd; one row read on demand
        self._row_by_key = {}
        self._rows_by_user = {}
        for i, (u, d) in enumerate(zip(self._users, self._dates)):
            self._row_by_key[(u, d)] = i
            self._rows_by_user.setdefault(u, []).append(i)

    def participant(self, user_id, dates) -> ParticipantSegments:
        """One participant's eligible segments, selected by ``dates`` (date-ascending)."""
        self._ensure_loaded()
        u = str(user_id)
        rows = sorted(
            self._row_by_key[(u, k)] for d in dates if (u, k := _day(d)) in self._row_by_key
        )
        idx = np.asarray(rows, dtype=np.int64)
        return ParticipantSegments(values=self._values[idx], mask=self._mask[idx])

    def participant_minute(self, user_id, dates) -> tuple[np.ndarray, list[str]]:
        """One participant's minute segments for ``dates`` (lazy fetch, minute resolution).

        The minute twin of :meth:`participant`: the provider supplies the eligible
        ``dates`` (``td.dates``); the loader fetches just those rows' raw ``(19, 1440)``
        values from the mmap'd ``daily_hf`` store. Returns ``(values, dates_used)`` —
        ``values`` is ``(n, 19, 1440)`` float32 in store order, ``dates_used`` the matched
        dates. Requires ``resolution='minute'``."""
        if self.resolution != "minute":
            raise RuntimeError(
                f"participant_minute() requires resolution='minute', got '{self.resolution}'"
            )
        self._ensure_loaded()
        u = str(user_id)
        pairs = sorted(
            (self._row_by_key[(u, k)], k) for d in dates if (u, k := _day(d)) in self._row_by_key
        )
        if not pairs:
            return np.empty((0, _N_CHANNELS, _MINUTES_PER_DAY), dtype=np.float32), []
        vals = np.stack(
            [np.asarray(self._minute_vals[i]["values"], dtype=np.float32) for i, _ in pairs]
        )
        return vals, [k for _, k in pairs]

    def participant_minute_public(self, user_id, dates) -> np.ndarray:
        """One participant's minute days in the public ``(n, 1440, 38)`` form.

        The minute twin of :meth:`participant`'s ``as_array``: channels 0-18 are sensor
        values with NaN at missing positions, 19-37 the ``1 = missing`` mask. ``daily_hf``
        stores zero-filled ``(19, 1440)`` values with no mask column, so missingness is
        recovered with ``ZeroToNaNTransform`` (the per-channel heuristic — a naive ``== 0``
        is wrong), exactly as the minute imputation / MAE paths do. Requires
        ``resolution='minute'``.
        """
        import torch

        from data.transforms.nan_transforms import ZeroToNaNTransform

        values, _ = self.participant_minute(user_id, dates)  # (n, 19, 1440) zero-filled
        if len(values) == 0:
            return np.empty((0, _MINUTES_PER_DAY, 2 * _N_CHANNELS), dtype=np.float32)
        zero_to_nan = ZeroToNaNTransform()
        nan_cf = np.stack([
            zero_to_nan(torch.from_numpy(np.ascontiguousarray(day, dtype=np.float32))).numpy()
            for day in values
        ])  # (n, 19, 1440) NaN at missing
        return _to_public_minute(nan_cf)

    def bind(self, td: TaskData) -> TaskData:
        """Fill ``td.inputs`` with one :class:`ParticipantSegments` per cohort user,
        selecting each user's eligible-date segments (``td.dates``)."""
        td.inputs = [self.participant(u, d) for u, d in zip(td.user_ids, td.dates)]
        return td

    # ----- whole-history access (for global-fit consumers: gru_d, multirocket) -----
    def user_segments(self, user_id) -> np.ndarray:
        """All of a participant's segments ``(n, 24, 19)`` (raw, NaN at missing)."""
        self._ensure_loaded()
        idx = np.asarray(sorted(self._rows_by_user.get(str(user_id), [])), dtype=np.int64)
        return self._values[idx]

    def user_days(self, user_id) -> tuple[np.ndarray, list[str]]:
        """All of a participant's days, date-ascending: ``(values (n, 24, 19), dates)``.

        ``values`` carry NaN at masked positions; ``dates`` are ``YYYY-MM-DD`` strings.
        For consumers that assemble their own per-user timeline from dated days
        (the TSFM encoders' gap-aware hourly series).
        """
        self._ensure_loaded()
        rows = self._rows_by_user.get(str(user_id), [])
        order = sorted(rows, key=lambda i: self._dates[i])
        idx = np.asarray(order, dtype=np.int64)
        return self._values[idx], [self._dates[i] for i in order]

    def segment_store(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The raw row-aligned store: ``(values, mask, users)`` — ``values`` / ``mask``
        are ``(N, 24, 19)`` (NaN at missing), ``users`` the per-row ``user_id`` array.

        For consumers that fit on the global train split or transform every segment
        (GRU-D, MultiRocket) and need the whole store, not just a cohort.
        """
        self._ensure_loaded()
        return self._values, self._mask, self._users

    def as_daily_rows(self) -> _RawDailyRows:
        """Row-position view in the raw stored form (zero-filled values + mask) for
        window-index consumers (WBM weekly extraction)."""
        self._ensure_loaded()
        return _RawDailyRows(self._values, self._mask)
