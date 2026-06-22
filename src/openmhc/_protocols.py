"""Duck-typed protocols for MHC-Benchmark model interfaces.

Users implement these interfaces to plug custom models into the benchmark.
No base class inheritance required -- just implement the right methods.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

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
class CohortStream(Protocol):
    """The cohort handed to ``fit`` / ``predict`` — iterate for one participant at a time.

    A small (hourly) spec hands you a plain ``list`` of per-participant arrays; a large or
    ``streaming`` spec hands you a lazy view with the **same iteration surface**, so the
    whole cohort never sits in RAM. Both forms satisfy this protocol — program against it,
    not against any concrete type. **Iterate — don't index** (``data[i]``) **or stack**
    (``np.stack(data)``) the streaming form: those force the cohort into memory. ``load`` is
    available on the streaming form for random access by participant id.

    This is the public description of that handle; the engine's concrete streaming class
    satisfies it structurally, so submitters never import an internal type.
    """

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield one participant's array at a time, shaped to the model's ``data_spec``."""
        ...

    def __len__(self) -> int:
        """Number of participants in the cohort."""
        ...

    def load(self, user_id) -> np.ndarray:
        """Return one participant's array by id (streaming form only)."""
        ...


# What the ``data`` argument to ``fit`` / ``predict`` actually is: a materialized list for
# small specs, or a streamed :class:`CohortStream` for large / streaming specs.
ParticipantData = list | CohortStream


@runtime_checkable
class Method(Protocol):
    """The prediction-track model contract — **return one prediction per participant**.

    One shape for every model, external submissions and bundled baselines alike.
    ``predict`` is the only **required** method: it returns one prediction per
    participant on a held-out cohort. ``fit`` is **optional** — this is an evaluation
    suite, not training infrastructure, so a zero-shot or pretrained model simply omits
    it; when present, the engine calls it on the train cohort before ``predict`` (see
    *Optional hooks* below). Representation isolation is preserved by *convention*: an
    encoder-style method runs its own representation and then the benchmark's uniform
    head, :class:`openmhc.LinearProbe`, inside ``fit`` / ``predict`` — so its score
    still reflects the representation, not the choice of classifier — while an
    end-to-end method owns its head directly.

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
    whenever you set ``streaming = True``) it is a streamed :class:`~openmhc.CohortStream`
    that yields the *same* per-participant arrays one at a time, so the whole cohort never
    sits in memory. **Iterate — don't index** (``data[i]``) **or stack the raw cohort**
    (``np.stack(data)``): those force everything into RAM and break streaming. Encoding each
    participant and stacking the small results (as below) is always safe.

    **Optional hooks** — omit any of these and the engine simply skips that step:

    - ``fit(data, labels, task_type) -> None`` — train on the train cohort. ``data`` is
      the per-participant arrays (iterate it) shaped to your ``data_spec``; ``labels`` is a
      ``(n,)`` array aligned with the cohort; ``task_type`` is as above. Omit it for a
      zero-shot / pretrained model — the engine then never even builds the train inputs.
    - ``set_context(ctx: EvalContext)`` — called before ``fit`` and ``predict`` (bundled
      baselines only — see :class:`EvalContext`).

    A model without a ``data_spec`` falls back to the legacy ``input_granularity = "daily"``
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

    def predict(self, data: ParticipantData) -> np.ndarray:
        """Return one prediction per participant, in ``data`` order (the only required method).

        The optional ``fit(data, labels, task_type)`` / ``set_context(ctx)`` hooks are
        documented in the class docstring above; the engine calls each only if defined.
        """
        ...


@runtime_checkable
class Imputer(Protocol):
    """Protocol for imputation evaluation.

    Implementations only need to define ``impute``. All setup — loading
    checkpoints, computing statistics from the training split, building
    per-user state — is the implementation's responsibility, typically
    done in ``__init__``. The benchmark harness does not call any
    preparation method.

    The optional metadata kwargs are keyword-only with ``None`` defaults.
    The harness inspects each implementation's ``impute`` signature once
    and forwards only the kwargs the implementation actually declares,
    so simple methods can keep the three-argument form below.

    Example:
        >>> class MeanImputer:
        ...     def __init__(self):
        ...         import openmhc
        ...         data_sum, data_count = 0.0, 0
        ...         for data, mask in openmhc.iter_train_data(version="xs"):
        ...             obs = (mask > 0.5) & np.isfinite(data)
        ...             data_sum = data_sum + np.where(obs, data, 0.0).sum(axis=(0, 2))
        ...             data_count = data_count + obs.sum(axis=(0, 2))
        ...         self.means = data_sum / np.maximum(data_count, 1)
        ...     def impute(self, data, observed_mask, target_mask):
        ...         result = data.copy()
        ...         for ch in range(19):
        ...             result[:, ch, :][target_mask[:, ch, :] == 1] = self.means[ch]
        ...         return result.astype(np.float32, copy=False)

    See ``openmhc.imputers`` for ready-to-use reference implementations
    (mean, mode, linear, LOCF, temporal, personalized, and a generic
    ``TorchImputer`` wrapper).
    """

    def impute(
        self,
        data: np.ndarray,
        observed_mask: np.ndarray,
        target_mask: np.ndarray,
        *,
        sample_indices: np.ndarray | None = None,
        user_ids: list[str] | None = None,
        dates: list[str] | None = None,
        day_offsets: np.ndarray | None = None,
    ) -> np.ndarray:
        """Impute artificially masked positions.

        Args:
            data: Sensor values of shape (N, 19, T) with NaN at both
                naturally missing positions and artificially masked
                positions. ``T = 1440`` for daily evaluation (the default);
                ``T = n_days * 1440`` when the caller sets ``n_days > 1``
                in ``evaluate_imputation`` (1-7).
            observed_mask: Binary mask of shape (N, 19, T). 1 =
                originally observed, 0 = naturally missing.
            target_mask: Binary mask of shape (N, 19, T). 1 = positions
                to impute (always a subset of ``observed_mask``).
            sample_indices: Optional split-local indices, shape (N,).
                Useful for any implementation that keeps per-sample state.
            user_ids: Optional list of N user-identifier strings, one per
                sample. Used by personalized methods.
            dates: Optional list of N ISO date strings (``YYYY-MM-DD``),
                one per sample.
            day_offsets: Optional int64 array of shape ``(N, n_days)``,
                only forwarded when ``n_days > 1``. Each row gives the
                calendar-day deltas of that window's day slots from the
                first non-padded day; ``-1`` marks left-padded slots that
                have no real data. Used by calendar-aware models (e.g.
                RoPE day embeddings in ``LSM2WeeklySparseImputer``) to
                encode real-world gaps between days in a non-contiguous
                window. Declaring this kwarg in your ``impute`` signature
                opts your imputer in — the harness inspects the signature
                once and only forwards declared kwargs.

        Returns:
            Array of shape (N, 19, T) with imputed values at
            ``target_mask == 1`` positions. Must be float32. ``T`` matches
            the input ``T`` (i.e. ``1440`` or ``n_days * 1440``).
        """
        ...


@runtime_checkable
class Forecaster(Protocol):
    """Protocol for forecasting evaluation (Track 3) — the unified contract.

    A forecaster receives, for each in-scope window, the **full-prefix** history
    ``(n_channels, history_length)`` (selected by data-quality criteria only, so
    the window set is identical across all models) and the forecast ``horizon``.
    The model owns all context windowing / truncation / padding it needs.

    The harness **never** drops a window for model-capability reasons. If the
    model cannot predict a given window/channel/timestep it must emit ``NaN``
    there; the harness substitutes the Seasonal-Naive baseline for those
    positions before scoring and reports how often that happened
    (``ForecastingResults.overall_fallback_rate``).

    Optional metadata kwargs are keyword-only and forwarded only if declared
    (the harness inspects the signature once, the same duck-typed pattern as
    :class:`Method` / :class:`Imputer`): ``variable_names``,
    ``past_covariates``, ``future_covariates``, ``index_days``.

    The benchmark ranks point forecasts; quantile forecasts are optional by
    returning a ``(point, quantiles)`` tuple instead of a bare point array
    (``quantiles`` shape ``(n_channels, horizon, n_quantiles)``; expose the
    matching levels as a ``quantile_levels`` attribute).

    Example:
        >>> class LastValueForecaster:
        ...     def predict(self, history, horizon):
        ...         # history: (n_channels, history_length), full prefix
        ...         # returns: (n_channels, horizon)
        ...         last = history[:, -1:]
        ...         return np.tile(last, (1, horizon))
    """

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """Forecast ``horizon`` future hours given the full-prefix history.

        Args:
            history: Float array of shape ``(n_channels, history_length)`` with
                the past observations (full prefix up to the forecast origin).
                May contain NaN at missing positions and may be short.
            horizon: Number of future hours to predict.

        Returns:
            Float array of shape ``(n_channels, horizon)`` with the point
            forecast (float32), or a ``(point, quantiles)`` tuple. Use ``NaN``
            for any position the model cannot predict — those are filled by the
            Seasonal-Naive baseline before scoring.
        """
        ...
