"""SegmentBinder — bind each participant's eligible raw segments to a TaskData.

``TaskDataProvider`` decides the per-(task, split) cohort + labels + per-user
lookup row positions (``eligible_indices``) but leaves ``TaskData.inputs`` empty.
The binder fills it: it loads the row-aligned segment source once and, per user,
gathers that user's eligible segments.

The data handed to a model is **raw** — values with NaN at missing positions plus
the missingness mask. Normalization is the model's concern: encoders z-score with
train-split stats; the Linear baseline pools raw values directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from downstream_evaluation.data.provider import TaskData


@dataclass
class ParticipantSegments:
    """One participant's eligible segments (raw; ``values`` carry NaN at missing).

    Attributes:
        values: ``(n_segments, T, 19)`` sensor values, NaN where missing.
        mask: ``(n_segments, T, 19)`` missingness mask (1 = missing, 0 = observed).
    """

    values: np.ndarray
    mask: np.ndarray


class SegmentBinder:
    """Materialize per-participant eligible segments from lookup row positions.

    Loads the row-aligned segment source ONCE and reuses it across all tasks/splits.
    The daily lookup is row-aligned with ``daily_hourly_hf``, so a user's
    ``eligible_indices`` index the segment data directly (select-by-index, no join).
    """

    def __init__(self, data_dir: str | None, granularity: str = "daily") -> None:
        """Args:
        data_dir: dataset root (``MHC_DATA_DIR`` / openmhc cache if ``None``).
        granularity: only ``"daily"`` is supported in this phase.
        """
        if granularity != "daily":
            raise NotImplementedError(
                f"SegmentBinder supports 'daily' granularity for now, got {granularity!r}"
            )
        self.granularity = granularity

        import datasets as hf_ds

        from openmhc._evaluate import _DatasetPaths

        from downstream_evaluation.data.data_loader import prepare_daily_hourly_hf

        paths = _DatasetPaths.resolve(data_dir)
        ds = hf_ds.load_from_disk(str(paths.daily_hourly_hf))
        if isinstance(ds, hf_ds.DatasetDict):
            ds = hf_ds.concatenate_datasets(list(ds.values()))
        ds = prepare_daily_hourly_hf(ds)  # (24,19) time-first, NaN at missing

        # Keep as numpy for fast select-by-index; row order == lookup row order.
        self._values = np.asarray(ds["values"], dtype=np.float32)  # (N, 24, 19)
        self._mask = np.asarray(ds["mask"], dtype=np.float32)  # (N, 24, 19)

    def bind(self, td: TaskData) -> TaskData:
        """Fill ``td.inputs`` with one :class:`ParticipantSegments` per cohort user."""
        inputs = [
            ParticipantSegments(
                values=self._values[np.sort(np.asarray(rows, dtype=np.int64))],
                mask=self._mask[np.sort(np.asarray(rows, dtype=np.int64))],
            )
            for rows in td.eligible_indices
        ]
        td.inputs = inputs
        return td
