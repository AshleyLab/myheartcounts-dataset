"""Duck-typed protocols for MHC-Benchmark model interfaces.

Users implement these interfaces to plug custom models into the benchmark.
No base class inheritance required -- just implement the right methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, runtime_checkable

import numpy as np


@dataclass
class EvalContext:
    """Per-(task, split) context the benchmark hands a bundled baseline.

    The unified :class:`Method` contract passes only clean arrays + labels +
    ``task_type`` — everything a from-scratch external model needs. A *bundled*
    baseline may additionally need to know **which** cohort it is being fit on (to
    align a precomputed per-user feature/embedding cache) or the **task name** (the
    ``linear`` baseline drops a demographic covariate on its own task, e.g. age is
    not a feature for the ``age`` task). The engine delivers that here via the
    optional ``set_context`` hook — called once before ``fit`` (train context) and
    once before ``predict`` (test context) — so the public ``fit`` / ``predict``
    signatures stay minimal. External models simply don't implement ``set_context``
    and never see it.

    Attributes:
        task: the task name (e.g. ``"Diabetes"``); ``task_type`` is ``get_task_type(task)``.
        split: the cohort split — ``"train"`` / ``"validation"`` / ``"test"``.
        user_ids: cohort user ids, aligned with the ``data`` / ``labels`` passed to
            ``fit`` / ``predict``.
        dates: per-user eligible segment dates (daily) / week_starts (weekly), aligned
            with ``user_ids`` — for cache baselines that pool per-user over the cohort's
            eligible days (e.g. MAE). ``None`` when not provided.
    """

    task: str
    split: str
    user_ids: np.ndarray
    dates: list | None = None


@runtime_checkable
class Method(Protocol):
    """The prediction-track model contract — **fit on arrays + labels, return predictions**.

    One shape for every model, external submissions and bundled baselines alike: a
    method fits on the train cohort's per-participant data and labels, then returns
    one prediction per participant on a held-out cohort. Representation isolation is
    preserved by *convention*: an encoder-style method runs its own representation
    and then the benchmark's uniform head, :class:`openmhc.LinearProbe`, inside
    ``fit`` / ``predict`` — so its score still reflects the representation, not the
    choice of classifier — while an end-to-end method owns its head directly.

    **Choose your input shape** with ``data_spec = DataSpec(resolution, window)``
    (see :class:`~openmhc.DataSpec`):

    - ``DataSpec("hourly", "day")``       → each participant ``(n_days, 24, 38)``
    - ``DataSpec("hourly", "series", N)`` → each participant ``(N, 38)`` (one continuous window)
    - ``DataSpec("minute", "day")``       → each participant ``(n_days, 1440, 38)``

    Channels are always 0-18 raw sensor values (NaN at missing positions) and 19-37 the
    missingness mask (1 = missing). ``labels`` is a ``(n,)`` array aligned with the cohort;
    ``task_type`` is ``"binary"`` / ``"multiclass"`` / ``"ordinal"`` / ``"regression"``.

    **Consuming ``data``** — iterate it to get one participant's array at a time::

        for x in data: ...        # x is that participant's array, shaped to your data_spec

    For small (hourly) specs ``data`` is a plain ``list``; for large specs (``minute``, or
    whenever you set ``streaming = True``) it is a streamed :class:`~openmhc.CohortView`
    that yields the *same* per-participant arrays one at a time, so the whole cohort never
    sits in memory. **Iterate — don't index** (``data[i]``) **or stack the raw cohort**
    (``np.stack(data)``): those force everything into RAM and break streaming. Encoding each
    participant and stacking the small results (as below) is always safe.

    A model may also define the optional ``set_context(ctx: EvalContext)`` hook, called
    before ``fit`` and ``predict`` (bundled baselines only — see :class:`EvalContext`). A
    model without a ``data_spec`` falls back to the legacy ``input_granularity = "daily"``
    (equivalent to ``DataSpec("hourly", "day")``).

    Example (encoder-style, via the uniform probe — streaming-safe as written)::

        import numpy as np
        import openmhc
        from openmhc import DataSpec

        class MyMethod:
            data_spec = DataSpec("hourly", "day")            # (n_days, 24, 38) per participant

            def fit(self, data, labels, task_type):
                emb = np.stack([self._encode(x) for x in data])   # iterate; stack small embeddings
                self._probe = openmhc.LinearProbe(task_type).fit(emb, labels)

            def predict(self, data):
                return self._probe.predict(np.stack([self._encode(x) for x in data]))
    """

    def fit(self, data: Iterable[np.ndarray], labels: np.ndarray, task_type: str) -> None:
        """Fit on the train cohort: per-participant ``data`` (iterate it) + aligned ``labels``."""
        ...

    def predict(self, data: Iterable[np.ndarray]) -> np.ndarray:
        """Return one prediction per participant, in ``data`` order."""
        ...


@runtime_checkable
class Imputer(Protocol):
    """Protocol for imputation evaluation.

    Example:
        >>> class MeanImputer:
        ...     def fit(self, data, masks):
        ...         self.means = np.nanmean(data, axis=(0, 2))
        ...     def impute(self, data, observed_mask, target_mask):
        ...         result = data.copy()
        ...         for ch in range(19):
        ...             target = target_mask[:, ch, :] == 1
        ...             result[:, ch, :][target] = self.means[ch]
        ...         return result
    """

    def fit(self, data: np.ndarray, masks: np.ndarray) -> None:
        """Fit on training data.

        Args:
            data: Daily sensor values of shape (N, 19, 1440). NaN at missing
                positions.
            masks: Binary masks of shape (N, 19, 1440). 1 = observed,
                0 = missing.
        """
        ...

    def impute(
        self,
        data: np.ndarray,
        observed_mask: np.ndarray,
        target_mask: np.ndarray,
    ) -> np.ndarray:
        """Impute artificially masked positions.

        Args:
            data: Sensor values of shape (N, 19, 1440) with NaN at masked
                positions.
            observed_mask: Binary mask of shape (N, 19, 1440). 1 = originally
                observed, 0 = naturally missing.
            target_mask: Binary mask of shape (N, 19, 1440). 1 = positions to
                impute (a subset of observed_mask).

        Returns:
            Array of shape (N, 19, 1440) with imputed values at target
            positions. Must be float32.
        """
        ...


@runtime_checkable
class Forecaster(Protocol):
    """Protocol for forecasting evaluation (Track 3).

    Forecast future hours from a history window. The benchmark evaluates
    point predictions; quantile forecasts are optional via a return signature
    extension (see below).

    Example:
        >>> class LastValueForecaster:
        ...     def predict(self, history, horizon):
        ...         # history: (n_channels, history_length)
        ...         # returns: (n_channels, horizon)
        ...         last = history[:, -1:]
        ...         return np.tile(last, (1, horizon))
    """

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """Forecast ``horizon`` future hours given the history window.

        Args:
            history: Float array of shape ``(n_channels, history_length)``
                with the past observations. May contain NaN at missing
                positions.
            horizon: Number of future hours to predict.

        Returns:
            Float array of shape ``(n_channels, horizon)`` with the point
            forecast. Must be float32.
        """
        ...
