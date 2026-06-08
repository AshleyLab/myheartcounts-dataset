"""``WindowBuilder`` — one anchored hour-window per participant.

Reconstructs each cohort user's continuous hourly timeline from raw ``daily_hourly_hf``,
anchors it (``label_date + forward_window`` for ``anchor="window_end"``, or the label date
for ``anchor="label"``), and slices the last ``hours`` — the Toto/Chronos-2 data path,
generalized to any window length.

The numerically-critical windowing currently *reuses* the functions in
``downstream_evaluation.models.tsfm`` (``build_user_timeline`` / ``build_window`` /
``_group_indices`` / ``_label_timestamp``) so the output is byte-identical to today's
Toto/Chronos-2 extraction by construction. (A later cleanup moves those functions here and
flips the import direction; that is behaviour-preserving and golden-guarded.)

Empty/short windows follow the cohort-invariant rule: never drop a user (the lookup decides
the cohort) — pad + mask, and a user with no anchorable data gets an all-masked window.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from downstream_evaluation.data.provider import TaskData

from .base import InputBuilder, ParticipantData


class WindowBuilder(InputBuilder):
    """Per-participant anchored hour-window of shape ``(1, hours, 19)``."""

    def __init__(self, data_dir: str | None, spec, temporal=None) -> None:
        """Bind the spec and forward-window source for this builder.

        ``spec`` is a :class:`~...inputs.spec.Window`; ``temporal`` supplies the
        per-task forward window for ``anchor="window_end"`` (defaults to the shared policy).
        """
        self.spec = spec
        self._data_dir = data_dir
        self._temporal = temporal
        self._ds = None
        self._grouped = None
        self._n_channels: int | None = None

    # ----- lazy load + per-user grouping (mirrors tsfm's front-end) -----
    def _ensure_loaded(self) -> None:
        if self._ds is not None:
            return
        import datasets as hf_ds

        from data.processing.hf_config import N_CHANNELS
        from downstream_evaluation.data.splits import load_split_file
        from downstream_evaluation.models.tsfm import _group_indices
        from openmhc._evaluate import _DatasetPaths

        paths = _DatasetPaths.resolve(self._data_dir)
        self._ds = hf_ds.load_from_disk(str(paths.daily_hourly_hf))
        split_users = load_split_file(Path(paths.splits_file))
        user_to_split = {str(u): s for s, us in split_users.items() for u in us}
        self._grouped, _ = _group_indices(self._ds, user_to_split)
        self._n_channels = N_CHANNELS

    def _weeks_after(self, task: str) -> int:
        """Return the forward-window length for the anchor.

        0 for ``"label"``, the per-task policy for ``"window_end"``.
        """
        if getattr(self.spec, "anchor", "window_end") == "label":
            return 0
        if self._temporal is None:
            from downstream_evaluation.config import TemporalWindowConfig

            self._temporal = TemporalWindowConfig()
        return self._temporal.weeks_after(task)

    def iter_inputs(self, td: TaskData) -> Iterator[ParticipantData]:
        """Yield one anchored ``(1, hours, 19)`` window per cohort user in ``td``."""
        import pandas as pd

        from downstream_evaluation.config import LABEL_REFERENCE_DATE
        from downstream_evaluation.models.tsfm import (
            _label_timestamp,
            build_user_timeline,
            build_window,
        )

        if getattr(self.spec, "resolution", "hourly") != "hourly":
            raise NotImplementedError("WindowBuilder: minute resolution not yet supported")
        self._ensure_loaded()
        reference_ts = pd.Timestamp(LABEL_REFERENCE_DATE)
        weeks_after = self._weeks_after(td.task)
        hours, nc = int(self.spec.hours), int(self._n_channels)
        users = self._grouped.get(td.split, {})

        for raw_uid in td.user_ids:
            uid = str(raw_uid)
            ex = None
            indices = users.get(uid)
            if indices:
                timeline = build_user_timeline(self._ds, indices, nc)
                if timeline is not None:
                    label_ts = _label_timestamp(uid, td.task, reference_ts)
                    if label_ts is not None:
                        ex = build_window(
                            timeline,
                            uid,
                            td.task,
                            label_ts.strftime("%Y-%m-%d"),
                            hours,
                            nc,
                            weeks_after,
                        )
            yield _to_participant(ex, hours, nc)


def _to_participant(ex, hours: int, nc: int) -> ParticipantData:
    """Convert a ``WindowExample`` to standard ``(1, hours, 19)`` ParticipantData.

    Channel-first window with zeros at gaps becomes values with NaN at missing and a
    mask where 1 = missing. An empty (``None``) example yields an all-masked window.
    """
    if ex is None:
        return ParticipantData(
            values=np.full((1, hours, nc), np.nan, dtype=np.float32),
            mask=np.ones((1, hours, nc), dtype=np.float32),
        )
    real = ex.padding_mask  # (nc, hours) bool, True where real
    vals = np.where(real, ex.window, np.nan).astype(np.float32).T[None, :, :]
    mask = (~real).astype(np.float32).T[None, :, :]
    return ParticipantData(values=vals, mask=mask)
