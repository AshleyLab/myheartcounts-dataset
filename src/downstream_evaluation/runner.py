"""Prediction engine — ``run_eval(config, model)``.

``run_eval`` sets up the data provider and data loader, hands them to a
:class:`DownstreamEvaluator`, and attaches run provenance. External models and the
bundled baselines run through one engine on one contract, ``openmhc.Method``:
``predict(data)`` (required) with an optional ``fit(data, labels, task_type)``,
per-participant arrays in and predictions out. A model that omits ``fit`` (zero-shot /
pretrained) is scored as-is. Encoder-style methods run the uniform ``openmhc.LinearProbe``
inside ``fit`` / ``predict``; end-to-end methods own their head.

All cohort / temporal / label logic comes from :class:`TaskDataProvider` (the
labels lookup). The model only ever sees a participant's *eligible* data,
at the granularity it declares via ``input_granularity`` (default daily).
"""

from __future__ import annotations

import logging

from downstream_evaluation.config import EvalConfig, TemporalWindowConfig
from downstream_evaluation.data.loader import DataLoader
from downstream_evaluation.data.provider import TaskDataProvider, lookup_filename
from downstream_evaluation.evaluation.evaluator import DownstreamEvaluator

logger = logging.getLogger(__name__)

# Re-export the config so ``from ...runner import EvalConfig`` works; it canonically
# lives in config.py.
__all__ = ["EvalConfig", "TemporalWindowConfig", "run_eval"]


def run_eval(config: EvalConfig, model) -> dict[str, dict]:
    """Run the prediction eval for one model (an ``openmhc.Method``).

    Builds the :class:`TaskDataProvider` (and the :class:`DataLoader`, unless the
    model declares ``needs_segments=False``) at the model's declared granularity,
    runs the :class:`DownstreamEvaluator`, and attaches a ``"config"`` provenance key.

    Returns ``{task: {**metrics, "n_test": int}, "config": {...}}``.
    """
    # A model declares its input shape either via the structured ``data_spec``
    # (:class:`~openmhc.DataSpec`, the single source of truth) or the legacy loose attrs
    # (``input_granularity`` / ``segment_resolution``). DataSpec is additive: models
    # without it follow the exact legacy path below, so existing baselines are unchanged.
    spec = getattr(model, "data_spec", None)
    grain = spec.provider_granularity if spec is not None else getattr(model, "input_granularity", "daily")
    lookup = f"{config.data_dir}/processed/{lookup_filename(grain, config.temporal.is_full_history)}"
    provider = TaskDataProvider(lookup, config.split_users, granularity=grain)

    if spec is not None:
        # DataSpec models: build a loader at the spec's resolution; the evaluator builds a
        # per-(task, split) CohortView and chooses eager-list vs streamed delivery from the
        # spec. The loader always serves daily segments (series is windowed on top).
        loader = DataLoader(config.data_dir, granularity="daily", resolution=spec.loader_resolution)
        loader_for_run = loader
    else:
        # Legacy path (unchanged). Cache-based models (precomputed per-user features/
        # embeddings) declare needs_segments=False and skip the loader's per-cohort binding;
        # global-fit models (GRU-D, MultiRocket) also set needs_segments=False but consume the
        # whole segment store via a set_loader hook. Build the single loader whenever either
        # path is needed, and inject it for whole-store consumers. The store resolution is the
        # model's choice: "hourly" (daily_hourly_hf, default) or "minute" (daily_hf).
        needs_segments = getattr(model, "needs_segments", True)
        wants_loader = hasattr(model, "set_loader")
        loader = (
            DataLoader(
                config.data_dir,
                granularity=grain,
                resolution=getattr(model, "segment_resolution", "hourly"),
            )
            if (needs_segments or wants_loader)
            else None
        )
        if loader is not None and wants_loader:
            model.set_loader(loader)
        loader_for_run = loader if needs_segments else None

    # Hand the temporal-window policy to models that build their own windows from raw
    # (Toto/Chronos-2); cohort/lookup models ignore it (their window is baked into the
    # lookup parquet). Duck-typed so new from-raw models opt in with one method.
    if hasattr(model, "set_temporal_window"):
        model.set_temporal_window(config.temporal)

    logger.info("Running prediction eval (granularity=%s) on %d tasks", grain, len(config.tasks))

    evaluator = DownstreamEvaluator(predictions_dir=config.predictions_dir, seed=config.seed)
    results = evaluator.run(provider, loader_for_run, model, config.tasks, spec=spec)

    # Predictions export: alongside the per-(method, task) parquets the evaluator
    # wrote, persist one shared per-user subgroup map (age_group + sex) for the
    # fairness bootstrap. Demographics come from the daily lookup regardless of the
    # model's granularity, so the map covers the widest set of users.
    if config.predictions_dir is not None:
        from downstream_evaluation.evaluation.predictions_io import write_subgroup_map

        test_users: set[str] = set()
        for task in config.tasks:
            try:
                test_users.update(provider.task_data(task, "test").user_ids.tolist())
            except KeyError:
                continue
        daily_lookup = f"{config.data_dir}/processed/{lookup_filename('daily', config.temporal.is_full_history)}"
        write_subgroup_map(config.predictions_dir, daily_lookup, test_users)

    results["config"] = {
        "model": getattr(model, "name", type(model).__name__),
        "seed": config.seed,
    }
    return results
