"""Protocol describing the internal imputation-method contract.

The harness (``ImputationEvaluator`` + ``run_eval``) duck-types its method
argument: anything with the right attributes / methods works. This module
makes the contract explicit so the ``TYPE_CHECKING`` imports in
``runner.py`` and ``evaluation/evaluator.py`` resolve to a real symbol,
and so contributors writing new methods have one place to look.

This is intentionally a ``Protocol`` (not a base class). The public
``openmhc.evaluate_imputation`` wraps user imputers in
``openmhc._evaluate._ImputerMethodAdapter``, which satisfies the
protocol structurally — no inheritance needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np


@runtime_checkable
class ImputationMethod(Protocol):
    """Structural contract for objects passed to ``ImputationEvaluator``.

    Attributes:
        name: Human-readable identifier (logged at fit time and used in
            metric configs).
        channel_stds: Per-channel standard deviations ``(C,)`` used by
            ``MetricAccumulator`` to normalize continuous-channel metrics.
            Populated by ``fit``.
        fallback_fill: Optional per-channel ``(C,)`` global fill used by the
            harness to substitute non-finite cells the method left at
            target positions. ``None`` opts out of substitution; the
            downstream ``isfinite`` filters then silently drop those cells
            as they did historically.
    """

    name: str
    channel_stds: np.ndarray | None
    fallback_fill: np.ndarray | None

    def fit(self, train_loader) -> None:
        """Run any harness-owned setup against the train loader.

        Typically computes ``channel_stds`` and ``fallback_fill`` from a
        single train pass. Does not invoke user model code beyond what
        the adapter declares — all user setup happens in ``__init__``.
        """
        ...

    def impute(
        self,
        data: np.ndarray,
        original_masks: np.ndarray,
        artificial_masks: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Produce an imputed array of the same shape as ``data``.

        ``original_masks`` marks observed cells (1 = observed); ``artificial_masks``
        marks the target cells the harness wants the method to fill (1 = target).
        Cells the method cannot produce should be returned as ``NaN``; the
        harness will substitute them from ``fallback_fill`` and report the rate.
        """
        ...
