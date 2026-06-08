"""RawBuilder — each cohort user's IC/TC-bounded raw days, at hourly or minute resolution.

The universal escape hatch (the :class:`~.spec.Raw` input): the framework still gates the
cohort (IC) and the in-window days (TC) from the daily lookup, then hands over the raw
per-day arrays at the requested resolution; the *model* windows / featurizes / encodes them
itself. One primitive covers minute feature-builders, custom-window TSFMs, anything — because
the shaping is the model's, while leakage-safety (which days are allowed) stays with the framework.

  - **hourly** — ``daily_hourly_hf`` (24 bins/day), row-aligned with the lookup → select-by-index.
  - **minute** — ``daily_hf`` (1440 bins/day), joined by ``(user_id, date)`` from ``td.dates``
    via a one-time index; rows are read lazily, so the full minute table is never materialized.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from downstream_evaluation.data.provider import TaskData

from .base import InputBuilder, ParticipantData


class RawBuilder(InputBuilder):
    """Deliver the cohort's eligible raw days at ``spec.resolution`` (model shapes them)."""

    def __init__(self, data_dir: str | None, spec) -> None:
        """Set up the builder from the dataset root and a :class:`~.spec.Raw` spec.

        Args:
            data_dir: dataset root (``MHC_DATA_DIR`` / openmhc cache if ``None``).
            spec: the :class:`~.spec.Raw` input spec; ``spec.resolution`` selects
                ``"hourly"`` or ``"minute"`` and must be one of those two.
        """
        self.spec = spec
        self._data_dir = data_dir
        self._res = getattr(spec, "resolution", "hourly")
        if self._res not in ("hourly", "minute"):
            raise ValueError(f"Raw resolution must be 'hourly' or 'minute', got {self._res!r}")
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self._res == "hourly":
            self._values, self._mask = _load_daily_hourly(self._data_dir)
        else:
            self._load_minute_index()
        self._loaded = True

    def _load_minute_index(self) -> None:
        """Open ``daily_hf`` lazily and index it by ``(user_id, date)``.

        Only the small id/date columns are materialized; the 1440-bin values are
        read per row on demand.
        """
        import datasets as hf_ds

        from data.transforms.nan_transforms import ZeroToNaNTransform
        from openmhc._evaluate import _DatasetPaths

        paths = _DatasetPaths.resolve(self._data_dir)
        ds = hf_ds.load_from_disk(str(paths.daily_hf))
        if isinstance(ds, hf_ds.DatasetDict):
            ds = hf_ds.concatenate_datasets(list(ds.values()))
        self._ds = ds
        users = np.asarray(ds["user_id"], dtype=object).astype(str)
        dates = np.asarray(ds["date"], dtype=object).astype(str)
        self._row = {(u, d[:10]): i for i, (u, d) in enumerate(zip(users, dates))}
        self._z2n = ZeroToNaNTransform()

    def iter_inputs(self, td: TaskData) -> Iterator[ParticipantData]:
        """Yield one :class:`ParticipantData` of eligible raw days per cohort user.

        Hourly days are selected by row index from the row-aligned table; minute
        days are gathered per user by ``(user_id, date)`` from ``td.dates``.
        """
        self._ensure_loaded()
        if self._res == "hourly":
            for rows in td.eligible_indices:
                idx = np.sort(np.asarray(rows, dtype=np.int64))
                yield ParticipantData(values=self._values[idx], mask=self._mask[idx])
        else:
            dates_by_user = td.dates if td.dates is not None else [None] * len(td.user_ids)
            for uid, dates in zip(td.user_ids, dates_by_user):
                yield self._minute_participant(str(uid), dates)

    def _minute_participant(self, uid: str, dates) -> ParticipantData:
        """Gather a user's eligible minute days → ``(n_days, 1440, 19)`` NaN-at-missing + mask."""
        import torch

        T, C = 1440, 19
        days = []
        if dates is not None:
            for d in dates:
                i = self._row.get((uid, str(d)[:10]))
                if i is None:
                    continue
                vals = np.asarray(self._ds[i]["values"], dtype=np.float32)  # (19, 1440)
                vals = self._z2n(torch.from_numpy(vals)).numpy()  # NaN at missing
                days.append(vals.T)  # (1440, 19) time-first
        if not days:
            return ParticipantData(
                values=np.empty((0, T, C), dtype=np.float32),
                mask=np.empty((0, T, C), dtype=np.float32),
            )
        values = np.stack(days).astype(np.float32)  # (n_days, 1440, 19)
        mask = np.isnan(values).astype(np.float32)  # 1 = missing
        return ParticipantData(values=values, mask=mask)


def _load_daily_hourly(data_dir: str | None):
    """Load ``daily_hourly_hf`` as ``(N, 24, 19)`` values + mask (row order == lookup order)."""
    import datasets as hf_ds

    from downstream_evaluation.data.data_loader import prepare_daily_hourly_hf
    from openmhc._evaluate import _DatasetPaths

    paths = _DatasetPaths.resolve(data_dir)
    ds = hf_ds.load_from_disk(str(paths.daily_hourly_hf))
    if isinstance(ds, hf_ds.DatasetDict):
        ds = hf_ds.concatenate_datasets(list(ds.values()))
    ds = prepare_daily_hourly_hf(ds)  # (24, 19) time-first, NaN at missing
    return (
        np.asarray(ds["values"], dtype=np.float32),
        np.asarray(ds["mask"], dtype=np.float32),
    )
