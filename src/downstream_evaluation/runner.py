"""Prediction engine — ``run_eval(config, model)``.

Mirrors the imputation/forecasting runners: ``run_eval`` sets up the data provider
and segment binder, hands them to a :class:`DownstreamEvaluator`, and attaches run
provenance. It powers both surfaces — Surface 1 (an external model, wrapped by the
openmhc adapter) and Surface 2 (our bundled baselines) — through one engine.

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
from downstream_evaluation.data.provider import LOOKUP_BY_GRANULARITY, TaskDataProvider
from downstream_evaluation.evaluation.evaluator import DownstreamEvaluator

logger = logging.getLogger(__name__)

# Re-export the config here too, so ``from ...runner import EvalConfig`` keeps working
# (the config canonically lives in config.py, mirroring the imputation track).
__all__ = ["EvalConfig", "TemporalWindowConfig", "run_eval"]


def run_eval(config: EvalConfig, model) -> dict[str, dict]:
    """Run the prediction eval for one model (``Encoder`` or ``Predictor``).

    Builds the :class:`TaskDataProvider` (and the :class:`SegmentBinder`, unless the
    model declares ``needs_segments=False``) at the model's declared granularity,
    runs the :class:`DownstreamEvaluator`, and attaches a ``"config"`` provenance key.

    Returns ``{task: {**metrics, "n_test": int}, "config": {...}}``.
    """
    grain = getattr(model, "input_granularity", "series")
    lookup = f"{config.data_dir}/processed/{LOOKUP_BY_GRANULARITY[grain]}"
    provider = TaskDataProvider(lookup, config.split_users, granularity=grain)
    # Cache-based models (precomputed per-user features/embeddings) declare
    # needs_segments=False and skip the segment binder entirely.
    needs_segments = getattr(model, "needs_segments", True)
    binder = SegmentBinder(config.data_dir, granularity=grain) if needs_segments else None

    # Hand the temporal-window policy to models that build their own windows from raw
    # (Toto/Chronos-2); cohort/lookup models ignore it (their window is baked into the
    # lookup parquet). Duck-typed so new from-raw models opt in with one method.
    if hasattr(model, "set_temporal_window"):
        model.set_temporal_window(config.temporal)

    logger.info("Running prediction eval (granularity=%s) on %d tasks", grain, len(config.tasks))

    evaluator = DownstreamEvaluator(seed=config.seed, pca_n_components=config.pca_n_components)
    results = evaluator.run(provider, binder, model, config.tasks)
    results["config"] = {
        "model": getattr(model, "name", type(model).__name__),
        "seed": config.seed,
    }
    return results
