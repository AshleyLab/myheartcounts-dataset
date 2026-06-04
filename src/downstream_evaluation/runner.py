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
from dataclasses import dataclass, field

from downstream_evaluation.data.binder import SegmentBinder
from downstream_evaluation.data.provider import LOOKUP_BY_GRANULARITY, TaskDataProvider
from downstream_evaluation.evaluation.evaluator import DownstreamEvaluator

logger = logging.getLogger(__name__)


@dataclass
class TemporalWindowConfig:
    """Per-task forward window (weeks) — the before-label window every method shares.

    A task's eligible region runs from the start of a user's data up to ``label +
    weeks_after(task)`` weeks. This is baked into the prebuilt ``*_windowed`` label
    lookups the cohort methods read, and applied live by the from-raw window builders
    (Toto/Chronos-2). Keeping it here makes it the single source of truth: the runner
    owns the policy, and any from-raw model is handed the window rather than redefining
    it. age/BiologicalSex widen to 156 (the cohort-asymmetry TC expansion).
    """

    default_weeks_after: int = 52
    task_weeks_after: dict[str, int] = field(
        default_factory=lambda: {"age": 156, "BiologicalSex": 156}
    )

    def weeks_after(self, task: str) -> int:
        """Forward-window length (weeks) for ``task``."""
        return self.task_weeks_after.get(task, self.default_weeks_after)


@dataclass
class EvalConfig:
    """Config for the prediction engine.

    Args:
        data_dir: dataset root (its ``processed/`` holds the lookups + sensor data).
        split_users: ``{"train"/"validation"/"test": [user_id, ...]}``.
        tasks: tasks to evaluate.
        seed: random_state for the probe / model.
        pca_n_components: PCA dim for the encoder probe (``None`` to disable).
        temporal: the per-task forward-window policy (handed to from-raw models).
    """

    data_dir: str
    split_users: dict
    tasks: list[str] = field(default_factory=list)
    seed: int = 42
    pca_n_components: int | None = 50
    temporal: TemporalWindowConfig = field(default_factory=TemporalWindowConfig)


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
