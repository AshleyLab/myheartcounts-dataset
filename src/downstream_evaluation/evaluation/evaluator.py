"""DownstreamEvaluator — the per-task evaluation loop for the prediction track.

``run_eval`` sets up the data provider + data loader and the model, then hands
them here. For each task this binds the cohort's eligible data, runs the model
through the :class:`~openmhc.Method` contract (the optional ``fit`` on the train
cohort, then ``predict`` on test), and scores the test split.
"""

from __future__ import annotations

import logging

import numpy as np

from downstream_evaluation.evaluation.metrics import (
    compute_binary_metrics,
    compute_multiclass_metrics,
    compute_ordinal_metrics,
    compute_regression_metrics,
    get_task_type,
)

logger = logging.getLogger(__name__)


def _metrics_for(task: str, y_true, y_pred, seed: int = 42) -> dict[str, float]:
    ttype = get_task_type(task)
    if ttype == "binary":
        return compute_binary_metrics(y_true, y_pred, seed=seed)
    # multiclass/ordinal scores discrete class predictions. The uniform ordinal probe
    # already predicts ints; an end-to-end method may hand back raw floats (e.g. the
    # hybrid's rank-combined scores, GRU-D's ordinal expected level), so round to int
    # before scoring (a no-op when predictions are already discrete).
    if ttype == "multiclass":
        return compute_multiclass_metrics(y_true, np.round(y_pred).astype(int), seed=seed)
    if ttype == "ordinal":
        return compute_ordinal_metrics(y_true, np.round(y_pred).astype(int), seed=seed)
    return compute_regression_metrics(y_true, y_pred, seed=seed)


def _public_inputs(td):
    """Per-participant data as the public ``(n_segments, 24, 38)`` arrays for a unified
    :class:`~openmhc.Method`, or ``None`` when the model self-serves from its own cache
    (``needs_segments=False`` → ``td.inputs`` is ``None``)."""
    if td.inputs is None:
        return None
    return [seg.as_array() for seg in td.inputs]


def _spec_inputs(loader, spec, td, ttype, streaming, with_labels):
    """Per-participant data for a DataSpec model: a streamed :class:`CohortView`, or its
    eager drain to a list.

    The eager drain is byte-identical to ``_public_inputs`` for an equivalent legacy
    hourly-day model — both ultimately call ``loader.participant(...).as_array()`` — so
    routing a hourly-day model through here is a no-op on the data it sees.
    """
    from downstream_evaluation.data.cohort import CohortView

    cohort = CohortView(
        loader,
        spec,
        td.user_ids,
        td.dates,
        td.labels if with_labels else None,
        ttype,
        td.task,
        td.split,
    )
    return cohort if streaming else [cohort.load(u) for u in cohort.user_ids]


def _set_context(model, td) -> None:
    """Hand a unified Method its per-(task, split) context, if it defines the hook.

    Mirrors the ``set_temporal_window`` duck-typed hook: external Methods omit it and
    never see cohort identity; bundled baselines read ``user_ids`` / ``task`` from it."""
    if hasattr(model, "set_context"):
        from openmhc._protocols import EvalContext

        model.set_context(
            EvalContext(task=td.task, split=td.split, user_ids=td.user_ids, dates=td.dates)
        )


class DownstreamEvaluator:
    """Run the per-task fit→predict→score loop for one model."""

    def __init__(self, predictions_dir: str | None = None, seed: int = 42):
        """Args:
        predictions_dir: when set, write each task's test predictions (uid, y_true,
            y_pred, y_proba) to ``<predictions_dir>/<method>/<task>/test.parquet``.
        seed: random seed for the per-task bootstrap standard errors (default 42,
            the canonical leaderboard seed).
        """
        self.predictions_dir = predictions_dir
        self.seed = seed

    def run(self, provider, loader, model, tasks: list[str], spec=None) -> dict[str, dict]:
        """Evaluate ``model`` on each task; returns ``{task: {**metrics, n_test}}``.

        ``spec`` (an :class:`~openmhc.DataSpec`) routes the model through the CohortView
        path; ``None`` is the legacy per-cohort ``bind`` path, unchanged.
        """
        streaming = bool(
            spec is not None and (spec.is_streaming_required or getattr(model, "streaming", False))
        )
        results: dict[str, dict] = {}
        for task in tasks:
            train_td = provider.task_data(task, "train")
            test_td = provider.task_data(task, "test")
            if spec is None and loader is not None:
                train_td, test_td = loader.bind(train_td), loader.bind(test_td)
            if len(train_td.user_ids) == 0 or len(test_td.user_ids) == 0:
                logger.warning("task %s: empty train/test split, skipping", task)
                continue
            y_true, y_pred = self._eval_task(
                model, task, train_td, test_td, loader, spec, streaming
            )
            results[task] = {
                **_metrics_for(task, y_true, y_pred, seed=self.seed),
                "n_test": int(len(y_true)),
            }
            if self.predictions_dir is not None:
                from downstream_evaluation.evaluation.predictions_io import (
                    write_task_predictions,
                )

                write_task_predictions(
                    self.predictions_dir,
                    getattr(model, "name", type(model).__name__),
                    task,
                    test_td.user_ids,
                    y_true,
                    y_pred,
                )
            logger.info("  %s: n_test=%d", task, len(y_true))
        return results

    def _eval_task(
        self, model, task: str, train_td, test_td, loader=None, spec=None, streaming=False
    ):
        """Evaluate one task; returns ``(y_true, y_pred)`` for the test split.

        The model sees clean per-participant arrays + labels + ``task_type`` (the
        :class:`~openmhc.Method` contract). With a ``spec`` the data arrives via a
        :class:`~downstream_evaluation.data.cohort.CohortView` (streamed, or drained to a
        list); without one it is the legacy bound ``_public_inputs``. ``fit`` is optional —
        a model that omits it (zero-shot / pretrained) skips fitting, and the train inputs
        are never built. The optional ``set_context`` hook hands bundled baselines their
        cohort identity (``user_ids`` / ``task``) before each call; external methods omit it.
        """
        ttype = get_task_type(task)
        # ``fit`` is an OPTIONAL hook: this is an evaluation suite, not training
        # infrastructure, so a zero-shot / pretrained model may implement only
        # ``predict``. When ``fit`` is absent we skip fitting entirely — and never
        # build the train inputs, so a predict-only model never streams the train
        # cohort for nothing. Every bundled baseline defines ``fit``, so its path is
        # unchanged (build train → build test → fit → predict, in this exact order).
        has_fit = hasattr(model, "fit")
        if spec is None:
            train_data = _public_inputs(train_td) if has_fit else None
            test_data = _public_inputs(test_td)
        else:
            train_data = (
                _spec_inputs(loader, spec, train_td, ttype, streaming, with_labels=True)
                if has_fit
                else None
            )
            test_data = _spec_inputs(loader, spec, test_td, ttype, streaming, with_labels=False)
        if has_fit:
            _set_context(model, train_td)
            model.fit(train_data, train_td.labels, ttype)
        _set_context(model, test_td)
        y_pred = np.asarray(model.predict(test_data))
        return test_td.labels, y_pred
