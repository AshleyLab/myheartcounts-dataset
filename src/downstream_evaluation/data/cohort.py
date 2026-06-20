"""CohortView — the public, loader-free handle a streaming Method pulls data from.

The engine builds one :class:`CohortView` per ``(task, split)`` and either drains it
into a list (eager, for small specs) or hands it to a model whose
:class:`~openmhc.DataSpec` needs streaming. It exposes only the cohort's identity and a
per-participant :meth:`load` — never the loader, the lookup, or the segment store
(the design's public/internal boundary). Iterating yields one participant's array at a
time, so peak memory is one participant, not the whole cohort.

This concrete class is internal; it structurally satisfies the public
:class:`openmhc.CohortStream` protocol, which is the type submitters program against.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import numpy as np

from downstream_evaluation.data.windowing import series_window

if TYPE_CHECKING:
    from openmhc._data_spec import DataSpec


def _day(d) -> str:
    """Normalize a date value to a ``YYYY-MM-DD`` key."""
    s = d if isinstance(d, str) else str(d)
    return s[:10]


class CohortView:
    """Lazy, read-only per-participant data access for one ``(task, split)``.

    A streaming model consumes it exactly like a list — iterate for one participant at a
    time, or pull a specific participant by id — but the whole cohort never sits in RAM::

        def fit(self, data, labels, task_type):    # data is a CohortView
            for x in data:                          # one participant's array at a time
                emb = self._encode(x)
                ...
            # or pull a specific participant:  x = data.load(user_id)

    Attributes:
        user_ids: cohort user ids; their order defines iteration order.
        labels: labels aligned with ``user_ids`` at fit time, ``None`` at predict time.
        task_type: ``"binary"`` / ``"multiclass"`` / ``"ordinal"`` / ``"regression"``.
        task: the task name.
        split: ``"train"`` / ``"validation"`` / ``"test"``.
    """

    def __init__(self, loader, spec: "DataSpec", user_ids, dates, labels, task_type, task, split):
        self._loader = loader
        self._spec = spec
        self.user_ids = np.asarray(user_ids, dtype=object)
        self.labels = labels
        self.task_type = task_type
        self.task = task
        self.split = split
        # per-user eligible dates, keyed for random access by load().
        self._dates_by_uid = {str(u): d for u, d in zip(user_ids, dates)}

    def __len__(self) -> int:
        return len(self.user_ids)

    def load(self, user_id) -> np.ndarray:
        """One participant's data, shaped to the cohort's :class:`~openmhc.DataSpec`.

        - ``("hourly", "day")`` -> ``(n_days, 24, 38)``
        - ``("hourly", "series", N)`` -> ``(N, 38)`` continuous left-padded window
        - ``("minute", "day")`` -> ``(n_days, 1440, 38)`` (streamed from the mmap store)
        """
        spec = self._spec
        dates = self._dates_by_uid[str(user_id)]
        if spec.window == "series":
            return self._series(user_id, dates, spec.window_units)
        if spec.resolution == "hourly":  # day window, hourly resolution
            return self._loader.participant(user_id, dates).as_array()
        return self._minute_day(user_id, dates)  # day window, minute resolution

    def __iter__(self) -> Iterator[np.ndarray]:
        # Process-and-discard: one participant resident at a time.
        for u in self.user_ids:
            yield self.load(u)

    def _series(self, user_id, eligible_dates, window_hours) -> np.ndarray:
        """Build the continuous window from the user's eligible days only."""
        values, dates = self._loader.user_days(user_id)
        eligible = {_day(d) for d in eligible_dates}
        keep = [i for i, d in enumerate(dates) if _day(d) in eligible]
        return series_window(values[keep], [dates[i] for i in keep], window_hours)

    def _minute_day(self, user_id, dates) -> np.ndarray:
        """One participant's minute days as ``(n_days, 1440, 38)`` (streamed from the mmap
        ``daily_hf`` store, one participant at a time)."""
        return self._loader.participant_minute_public(user_id, dates)
