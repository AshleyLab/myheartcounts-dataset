"""DataLoader — the single per-participant segment materializer.

Loads the row-aligned segment source **once**, indexes it by ``(user_id, date)``, and
materializes each participant's eligible segments on demand. It supersedes both the
per-(task, split) :class:`SegmentBinder` (``bind`` below) and the per-model self-loads
(``user_segments`` / ``segment_store`` below) — so ``daily_hourly_hf`` is read a single
time per run instead of once per consumer.

Keyed by ``(user_id, date)`` (not lookup row positions), so the same selection works
regardless of which segment source / resolution backs it — the lookup↔segment
row-alignment is no longer assumed. Selection is identical to the binder's: under
full-history a participant's eligible dates are all their retained days, so
``bind`` returns the same segments ``SegmentBinder`` did.

See ``docs/data_loader_design.md`` for the full design.
"""

from __future__ import annotations

import numpy as np

from downstream_evaluation.data.binder import ParticipantSegments
from downstream_evaluation.data.provider import TaskData


def _day(d) -> str:
    """Normalize a date value to a ``YYYY-MM-DD`` key."""
    s = d if isinstance(d, str) else str(d)
    return s[:10]


class DataLoader:
    """Load the segment source once; materialize per-participant segments by date."""

    def __init__(self, data_dir: str | None, granularity: str = "daily") -> None:
        """Args:
        data_dir: dataset root (``MHC_DATA_DIR`` / openmhc cache if ``None``).
        granularity: only ``"daily"`` is supported (hourly daily segments).
        """
        if granularity != "daily":
            raise NotImplementedError(
                f"DataLoader supports 'daily' granularity for now, got {granularity!r}"
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

        # Keep as numpy for fast select-by-index.
        self._values = np.asarray(ds["values"], dtype=np.float32)  # (N, 24, 19)
        self._mask = np.asarray(ds["mask"], dtype=np.float32)  # (N, 24, 19)
        self._users = np.asarray(ds["user_id"], dtype=object).astype(str)  # per-row user_id
        dates = np.asarray(ds["date"], dtype=object)

        # (user_id, date) -> row, and user_id -> sorted row list (for whole-history
        # consumers). Built once; reused across all tasks/splits.
        self._row_by_key: dict[tuple[str, str], int] = {}
        self._rows_by_user: dict[str, list[int]] = {}
        for i, (u, d) in enumerate(zip(self._users, dates)):
            key = (u, _day(d))
            self._row_by_key[key] = i
            self._rows_by_user.setdefault(u, []).append(i)

    def participant(self, user_id, dates) -> ParticipantSegments:
        """One participant's eligible segments, selected by ``dates`` (date-ascending)."""
        u = str(user_id)
        rows = sorted(
            self._row_by_key[(u, k)] for d in dates if (u, k := _day(d)) in self._row_by_key
        )
        idx = np.asarray(rows, dtype=np.int64)
        return ParticipantSegments(values=self._values[idx], mask=self._mask[idx])

    def bind(self, td: TaskData) -> TaskData:
        """Fill ``td.inputs`` with one :class:`ParticipantSegments` per cohort user.

        Drop-in for :meth:`SegmentBinder.bind` — selects each user's eligible-date
        segments (``td.dates``) instead of lookup row positions; the same rows.
        """
        td.inputs = [self.participant(u, d) for u, d in zip(td.user_ids, td.dates)]
        return td

    # ----- whole-history access (for global-fit consumers: gru_d, multirocket) -----
    def user_segments(self, user_id) -> np.ndarray:
        """All of a participant's segments ``(n, 24, 19)`` (raw, NaN at missing)."""
        idx = np.asarray(sorted(self._rows_by_user.get(str(user_id), [])), dtype=np.int64)
        return self._values[idx]

    def segment_store(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The raw row-aligned store: ``(values, mask, users)`` — ``values`` / ``mask``
        are ``(N, 24, 19)`` (NaN at missing), ``users`` the per-row ``user_id`` array.

        For consumers that fit on the global train split or transform every segment
        (GRU-D, MultiRocket) and need the whole store, not just a cohort.
        """
        return self._values, self._mask, self._users
