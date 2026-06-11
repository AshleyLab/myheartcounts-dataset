"""Duck-typed protocols for MHC-Benchmark model interfaces.

Users implement these interfaces to plug custom models into the benchmark.
No base class inheritance required -- just implement the right methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Encoder(Protocol):
    """Protocol for health-prediction encoders — the model contract for external
    submissions and the bundled baselines alike.

    An encoder maps **one participant's eligible wearable data** — already filtered
    to the task's cohort and temporal scope by the benchmark (the inclusion criteria,
    plus any forward window) — to a
    single fixed-size embedding. The benchmark then fits a *uniform* linear probe
    on the embeddings and scores it, so a model's result reflects its representation
    rather than its choice of probe. The same protocol is implemented by external
    encoders and by the baselines (MAE, SSL, Toto, Chronos-2, ...).

    Set ``input_granularity`` to choose how the benchmark hands you each
    participant's data. ``"daily"`` is currently wired by the segment binder:

      - ``"daily"``  — eligible daily segments ``(n_segments, 24, 38)``.
      - ``"series"`` / ``"weekly"`` — planned (hourly series / weekly segments).

    Channels 0-18 are the **raw** sensor values (NaN at missing positions) and 19-37
    the missingness mask (1 = missing, 0 = observed). Normalization is your model's
    concern — z-score with your own train-split statistics if you need it (the
    imputation track hands raw values the same way). The benchmark fits a PCA-50 probe
    on the embeddings, so return at least 50 dimensions.

    Example:
        >>> class MyEncoder:
        ...     input_granularity = "daily"
        ...     def encode(self, data: np.ndarray) -> np.ndarray:
        ...         # data: (n_segments, 24, 38) — eligible days; channels 0-18 raw
        ...         # values (NaN at missing), 19-37 the mask. Return D >= 50.
        ...         x = np.nan_to_num(data).reshape(-1, 38)
        ...         return np.concatenate([x.mean(0), x.std(0)])   # -> (76,)
    """

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Map one participant's eligible data to a 1-D float32 embedding.

        Args:
            data: the participant's eligible data at ``input_granularity`` (see the
                class docstring). Only in-cohort, in-window data is included, so it
                may be pooled / windowed freely.

        Returns:
            A 1-D float32 embedding ``(D,)`` of any dimensionality.
        """
        ...


@runtime_checkable
class Predictor(Protocol):
    """Protocol for end-to-end prediction models that own their classifier head.

    Unlike :class:`Encoder` (which returns a representation for the benchmark's
    uniform probe), a ``Predictor`` fits on the cohort's eligible data + labels and
    returns one prediction per participant; the benchmark scores those directly.
    Used by end-to-end models (e.g. GRU-D, MultiRocket). Set ``input_granularity``
    as for :class:`Encoder` (default ``"series"``).

    This is the *public* contract — plain per-participant arrays in, predictions out.
    The benchmark adapts it to its internal engine. (Bundled baselines implement the
    richer internal model interface directly, so they can key by user / task.)

    Example:
        >>> class MyPredictor:
        ...     def fit(self, data, labels): ...            # data: list of per-participant arrays
        ...     def predict(self, data) -> np.ndarray: ...   # (n,) predictions, aligned with data
    """

    def fit(self, data: list[np.ndarray], labels: np.ndarray) -> None:
        """Fit on the train cohort: per-participant eligible data + aligned labels."""
        ...

    def predict(self, data: list[np.ndarray]) -> np.ndarray:
        """Return one prediction per participant, aligned with ``data``."""
        ...


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
    """Unified prediction contract — **fit on arrays + labels, return predictions**.

    Supersedes :class:`Encoder` + :class:`Predictor` with one shape: a method fits on
    the train cohort's per-participant data and labels, then returns one prediction
    per participant on a held-out cohort. Representation isolation is preserved by
    *convention*: an encoder-style method runs its own representation and then the
    benchmark's uniform head, :class:`openmhc.LinearProbe`, inside ``fit`` /
    ``predict`` — so its score still reflects the representation, not the choice of
    classifier — while an end-to-end method owns its head directly.

    ``data`` is a list with one entry per participant, at ``input_granularity``
    (default ``"daily"`` → each entry ``(n_segments, 24, 38)``: channels 0-18 raw
    sensor values with NaN at missing positions, 19-37 the missingness mask).
    ``labels`` is a ``(n,)`` array aligned with ``data``. ``task_type`` is one of
    ``"binary"``, ``"multiclass"``, ``"ordinal"``, ``"regression"``.

    Engine opt-in: the benchmark routes a model through this contract when it sets the
    class attribute ``predicts_from_arrays = True``. A model may also define the
    optional ``set_context(ctx: EvalContext)`` hook; the engine calls it before
    ``fit`` and again before ``predict`` (bundled baselines only — see
    :class:`EvalContext`).

    Example (encoder-style, via the uniform probe)::

        import openmhc

        class MyMethod:
            input_granularity = "daily"
            predicts_from_arrays = True

            def fit(self, data, labels, task_type):
                emb = np.stack([self._encode(x) for x in data])
                self._probe = openmhc.LinearProbe(task_type).fit(emb, labels)

            def predict(self, data):
                return self._probe.predict(np.stack([self._encode(x) for x in data]))
    """

    def fit(self, data: list[np.ndarray], labels: np.ndarray, task_type: str) -> None:
        """Fit on the train cohort: per-participant ``data`` + aligned ``labels``."""
        ...

    def predict(self, data: list[np.ndarray]) -> np.ndarray:
        """Return one prediction per participant, aligned with ``data``."""
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
