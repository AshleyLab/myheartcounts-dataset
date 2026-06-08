"""``SegmentBuilder`` — eligible daily/weekly segments via select-by-index.

Lifts :class:`~downstream_evaluation.data.binder.SegmentBinder`'s daily path: load the
row-aligned segment source once, then per user select that user's eligible rows
(``td.eligible_indices``) — no join. Daily output is byte-identical to the old binder.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from downstream_evaluation.data.provider import TaskData

from .base import InputBuilder, ParticipantData


class SegmentBuilder(InputBuilder):
    """Eligible per-participant segments selected by lookup row index."""

    def __init__(self, data_dir: str | None, spec) -> None:
        """Load the row-aligned segment source for ``spec``'s cohort granularity."""
        self.spec = spec
        if getattr(spec, "resolution", "hourly") != "hourly":
            raise NotImplementedError(
                "SegmentBuilder: minute resolution not yet supported (use hourly)"
            )
        if spec.cohort == "daily":
            self._values, self._mask = _load_daily_hourly(data_dir)
        elif spec.cohort == "weekly":
            raise NotImplementedError("SegmentBuilder: weekly segment source not yet wired")
        else:
            raise ValueError(f"unknown cohort granularity {spec.cohort!r}")

    def iter_inputs(self, td: TaskData) -> Iterator[ParticipantData]:
        """One :class:`ParticipantData` per cohort user (select-by-sorted-index)."""
        for rows in td.eligible_indices:
            idx = np.sort(np.asarray(rows, dtype=np.int64))
            yield ParticipantData(values=self._values[idx], mask=self._mask[idx])


def _load_daily_hourly(data_dir: str | None):
    """Load ``daily_hourly_hf`` as ``(N, 24, 19)`` values + mask (row order == lookup order)."""
    import datasets as hf_ds

    from openmhc._evaluate import _DatasetPaths

    from downstream_evaluation.data.data_loader import prepare_daily_hourly_hf

    paths = _DatasetPaths.resolve(data_dir)
    ds = hf_ds.load_from_disk(str(paths.daily_hourly_hf))
    if isinstance(ds, hf_ds.DatasetDict):
        ds = hf_ds.concatenate_datasets(list(ds.values()))
    ds = prepare_daily_hourly_hf(ds)  # (24, 19) time-first, NaN at missing
    return (
        np.asarray(ds["values"], dtype=np.float32),
        np.asarray(ds["mask"], dtype=np.float32),
    )
