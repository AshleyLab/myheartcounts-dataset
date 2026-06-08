"""``InputBuilder`` interface + the per-participant payload it produces.

The builder turns a :class:`~downstream_evaluation.data.provider.TaskData` (cohort +
eligibility from the lookup) into one :class:`ParticipantData` per cohort user. It
delivers *data*, never encodings — the model keeps its own transform and cache.

Two consumption modes share one interface:
  - ``iter_inputs`` — lazy; yields one participant at a time (scales to 2048 h / minute).
  - ``build`` — eager convenience; fills ``td.inputs`` with the full list.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from downstream_evaluation.data.provider import TaskData


@dataclass
class ParticipantData:
    """One participant's materialized input.

    Attributes:
        values: ``(n, T, 19)`` sensor values, NaN at missing positions.
        mask: ``(n, T, 19)`` missingness mask (1 = missing, 0 = observed).
    """

    values: np.ndarray
    mask: np.ndarray


class InputBuilder(ABC):
    """Materialize per-participant model input from a cohort ``TaskData``."""

    @abstractmethod
    def iter_inputs(self, td: TaskData) -> Iterator[ParticipantData]:
        """Yield one :class:`ParticipantData` per cohort user, in ``td.user_ids`` order."""
        raise NotImplementedError

    def build(self, td: TaskData) -> TaskData:
        """Eagerly fill ``td.inputs`` with one :class:`ParticipantData` per cohort user."""
        td.inputs = list(self.iter_inputs(td))
        return td

    def bind(self, td: TaskData) -> TaskData:
        """Alias for :meth:`build` — the evaluator calls ``.bind`` (the legacy binder name)."""
        return self.build(td)
