"""TaskDataProvider — derives each task's cohort, eligibility, and labels from the lookup.

This is the single place the per-task cohort and temporal scope are decided, and it
reads them from the **shipped labels lookup** rather than recomputing them from raw
labels. A lookup cell is *non-sentinel* iff that segment is a valid wearable day that
passes the data-quality inclusion criteria, within the lookup's temporal scope; the
non-sentinel mask is therefore at once the cohort, the temporal scope, and the label
value. Deriving everything from one lookup keeps every method on an identical,
reproducible cohort.

The benchmark's default lookup applies the inclusion criteria over each participant's
full history; a forward-windowed ablation lookup additionally caps eligibility at
``label + weeks_after`` (see ``TemporalWindowConfig``).

The provider hands each model **eligible data + labels per (user, task)** at a
declared granularity; the model never sees the lookup, the segment grid, or the
mask. Granularity controls only *which* lookup supplies eligibility:

  - ``"weekly"`` → weekly lookup          (weekly-segment models, e.g. SSL)
  - ``"daily"``  → daily lookup           (daily-segment models, e.g. MAE)
  - ``"series"`` → daily lookup, eligibility broadcast to the continuous timeline
                   (models that window the raw series themselves, e.g. 5h/2048h)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Sentinels marking a missing/out-of-window label cell in the lookup.
_MISSING_INT = -1
_MISSING_FLOAT = -1.0

# Non-task columns in the labels lookup (everything else is a per-task label).
_KEY_COLS = {"user_id", "date", "week_start", "n_valid_days", "n_valid_hours"}

# Granularity → lookup filename (relative to the dataset's processed/ dir).
LOOKUP_BY_GRANULARITY: dict[str, str] = {
    "daily": "daily_labels_lookup.parquet",
    "weekly": "weekly_labels_lookup_stride7_windowed.parquet",
    # series eligibility = valid days (daily lookup), broadcast to the timeline.
    "series": "daily_labels_lookup.parquet",
}


def lookup_filename(granularity: str, full_history: bool = False) -> str:
    """Lookup parquet filename for ``granularity``.

    With ``full_history`` the variant built without a forward-window cap is used, so
    the cohort spans each participant's whole record. The data-quality inclusion
    criteria still apply — only the forward window is dropped.
    """
    name = LOOKUP_BY_GRANULARITY[granularity]
    return name.replace(".parquet", "_full_history.parquet") if full_history else name


@dataclass
class TaskData:
    """Eligible data + labels for one ``(task, split)``, handed to a model.

    ``user_ids`` / ``labels`` define the cohort and targets. ``dates`` gives, per
    user, the dates of that user's non-sentinel lookup cells — the eligible segments
    the :class:`~downstream_evaluation.data.loader.DataLoader` selects by
    ``(user_id, date)``. ``inputs`` holds the per-user eligible data once bound by
    the data layer (one entry per user); it is ``None`` until then.
    """

    task: str
    split: str
    granularity: str
    user_ids: np.ndarray
    labels: np.ndarray
    dates: list | None = None  # per-user eligible segment dates (daily) / week_starts (weekly)
    inputs: list | None = None


class TaskDataProvider:
    """Derive per-task cohort/labels/eligibility from the labels lookup."""

    def __init__(
        self,
        lookup_path: str,
        split_users: dict[str, list[str] | set[str]],
        granularity: str = "series",
    ) -> None:
        """Args:
        lookup_path: path to the labels lookup parquet for this granularity.
        split_users: ``{"train"/"validation"/"test": [user_id, ...]}``.
        granularity: ``"daily"`` / ``"weekly"`` / ``"series"``.
        """
        if granularity not in LOOKUP_BY_GRANULARITY:
            raise ValueError(
                f"granularity must be one of {sorted(LOOKUP_BY_GRANULARITY)}, got {granularity!r}"
            )
        self.granularity = granularity
        self._lookup = pd.read_parquet(lookup_path)
        self._date_col = "date" if "date" in self._lookup.columns else "week_start"
        self._split_users = {k: {str(u) for u in v} for k, v in split_users.items()}
        self._tasks = [c for c in self._lookup.columns if c not in _KEY_COLS]
        self._index_cache: dict[str, dict[str, tuple]] = {}

    @property
    def tasks(self) -> list[str]:
        """Task columns present in the lookup."""
        return list(self._tasks)

    def _task_index(self, task: str) -> dict[str, tuple]:
        """``{user_id: (label, dates)}`` from non-sentinel cells.

        A user appears iff they have ≥1 non-sentinel cell for ``task`` (passing the
        inclusion criteria); the dates are those of the lookup rows where the cell is
        non-sentinel (within the lookup's temporal scope) — the user's eligible
        segments; the label is the constant non-sentinel value (true for the
        cross-sectional tasks).
        """
        if task in self._index_cache:
            return self._index_cache[task]
        if task not in self._lookup.columns:
            raise KeyError(f"task {task!r} not in lookup")

        col = self._lookup[task].to_numpy()
        is_float = np.issubdtype(col.dtype, np.floating)
        if is_float:
            valid = ~(np.isnan(col) | (col == _MISSING_FLOAT))
        else:
            valid = col != _MISSING_INT

        sub = self._lookup.loc[valid, ["user_id", self._date_col, task]]
        index: dict[str, tuple] = {}
        for uid, grp in sub.groupby("user_id", sort=False):
            raw = grp[task].iloc[0]
            label = float(raw) if is_float else int(raw)
            index[str(uid)] = (label, grp[self._date_col].to_numpy())
        self._index_cache[task] = index
        return index

    def cohort(self, task: str) -> dict[str, list[str]]:
        """Per-split cohort users (split ∩ non-sentinel-for-task), user_id-sorted."""
        index = self._task_index(task)
        return {
            split: sorted(u for u in users if u in index)
            for split, users in self._split_users.items()
        }

    def task_data(self, task: str, split: str) -> TaskData:
        """Cohort + labels + eligible dates for ``(task, split)``.

        ``inputs`` is left ``None`` here; the data layer binds each user's eligible
        sensor data (at ``self.granularity``) by selecting their ``dates`` from the
        segment source.
        """
        index = self._task_index(task)
        split_key = "validation" if split in ("val", "validation") else split
        users = sorted(u for u in self._split_users.get(split_key, set()) if u in index)
        labels = np.array([index[u][0] for u in users])
        dates = [index[u][1] for u in users]
        return TaskData(
            task=task,
            split=split_key,
            granularity=self.granularity,
            user_ids=np.array(users, dtype=object),
            labels=labels,
            dates=dates,
        )
