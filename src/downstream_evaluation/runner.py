"""Prediction engine — ``run_eval(config, model)``.

``run_eval`` sets up the data provider and segment binder, hands them to a
:class:`DownstreamEvaluator`, and attaches run provenance. It supports both an
external model (wrapped by the openmhc adapter) and the bundled baseline models
through one engine.

  - ``Encoder``   — ``encode(data) -> (D,)`` per participant; the evaluator fits a
                    *uniform* PCA + linear probe, so the score reflects the
                    representation, not the probe.
  - ``Predictor`` — end-to-end; the evaluator scores its predictions directly.

All cohort / temporal / label logic comes from :class:`TaskDataProvider` (the
embedded-temporal lookup). The model only ever sees a participant's *eligible* data,
at the granularity it declares via ``input_granularity`` (default series).
"""

from __future__ import annotations

import logging

from downstream_evaluation.config import EvalConfig, TemporalWindowConfig
from downstream_evaluation.data.binder import SegmentBinder
from downstream_evaluation.data.inputs import input_builder_for
from downstream_evaluation.data.provider import LOOKUP_BY_GRANULARITY, TaskDataProvider
from downstream_evaluation.evaluation.evaluator import DownstreamEvaluator

logger = logging.getLogger(__name__)

# Re-export the config so ``from ...runner import EvalConfig`` works; it canonically
# lives in config.py.
__all__ = ["EvalConfig", "TemporalWindowConfig", "run_eval"]


def run_eval(config: EvalConfig, model) -> dict[str, dict]:
    """Run the prediction eval for one model (``Encoder`` or ``Predictor``).

    Builds the :class:`TaskDataProvider` (and the :class:`SegmentBinder`, unless the
    model declares ``needs_segments=False``) at the model's declared granularity,
    runs the :class:`DownstreamEvaluator`, and attaches a ``"config"`` provenance key.

    Returns ``{task: {**metrics, "n_test": int}, "config": {...}}``.
    """
    # Pick the cohort lookup + the input builder. New models declare a declarative
    # ``input`` spec (Raw/Window) → input_builder_for; legacy models use
    # ``input_granularity`` + ``needs_segments`` (cache models opt out with False).
    spec = getattr(model, "input", None)
    if spec is not None:
        grain = spec.cohort
        builder = input_builder_for(spec, config.data_dir, config.temporal)
    else:
        grain = getattr(model, "input_granularity", "series")
        needs_segments = getattr(model, "needs_segments", True)
        builder = SegmentBinder(config.data_dir, granularity=grain) if needs_segments else None
    lookup = f"{config.data_dir}/processed/{LOOKUP_BY_GRANULARITY[grain]}"
    provider = TaskDataProvider(lookup, config.split_users, granularity=grain)

    # Hand the temporal-window policy to models that build their own windows from raw
    # (Toto/Chronos-2); cohort/lookup models ignore it (their window is baked into the
    # lookup parquet). Duck-typed so new from-raw models opt in with one method.
    if hasattr(model, "set_temporal_window"):
        model.set_temporal_window(config.temporal)

    logger.info("Running prediction eval (granularity=%s) on %d tasks", grain, len(config.tasks))

    evaluator = DownstreamEvaluator(seed=config.seed, pca_n_components=config.pca_n_components)
    results = evaluator.run(provider, builder, model, config.tasks)
    results["config"] = {
        "model": getattr(model, "name", type(model).__name__),
        "seed": config.seed,
    }
    return results
