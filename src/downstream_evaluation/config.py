"""Configuration for downstream evaluation.

The config lives here; the engine lives in ``runner.py``.

  - ``EvalConfig`` (+ ``TemporalWindowConfig``) — what ``run_eval(config, model)``
    consumes for the prediction engine.
  - ``ClassifierConfig`` + the per-probe hyperparameter dataclasses it composes —
    the uniform linear/tree probe the engine fits on encoder embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# Fixed reference timestamp for static enrollment-time label lookups. The Labels
# API returns the nearest-in-time match; this anchors all lookups to enrollment.
LABEL_REFERENCE_DATE = "2020-06-01"


# --------------------------------------------------------------------------- #
# Prediction-engine config (consumed by runner.run_eval)
# --------------------------------------------------------------------------- #
@dataclass
class TemporalWindowConfig:
    """Per-task forward window (weeks) — the before-label window every method shares.

    A task's eligible region runs from the start of a user's data up to ``label +
    weeks_after(task)`` weeks. This is baked into the prebuilt ``*_windowed`` label
    lookups the cohort methods read, and applied live by the from-raw window builders
    (Toto/Chronos-2). Keeping it here makes it the single source of truth: the runner
    owns the policy, and any from-raw model is handed the window rather than redefining
    it. age and BiologicalSex widen to a 156-week window (these demographic tasks
    include data further from the label date).
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


# --------------------------------------------------------------------------- #
# Probe hyperparameters (composed by ClassifierConfig, read by create_model)
# --------------------------------------------------------------------------- #
@dataclass
class LogRegConfig:
    """LogisticRegression hyperparameters."""

    max_iter: int = 4000
    class_weight: str | None = "balanced"
    C: float = 1.0
    solver: str = "liblinear"  # More robust for small datasets than lbfgs
    n_jobs: int = 1  # liblinear doesn't support n_jobs=-1


@dataclass
class LinearRegressionConfig:
    """Linear Regression hyperparameters."""

    fit_intercept: bool = True
    copy_X: bool = True
    n_jobs: int | None = None
    positive: bool = False


@dataclass
class ClassifierConfig:
    """Uniform-probe selection + hyperparameters.

    The engine fits only linear probes — logistic regression (binary/multiclass),
    a K−1 binary ordinal decomposition (``logreg_ordinal``, which reuses the
    logistic-regression hyperparameters), and OLS (regression). The tree-based
    XGBoost method is self-contained and does not use this probe.
    """

    type: Literal[
        "logistic_regression",
        "linear_regression",
        "logreg_ordinal",
    ] = "logistic_regression"
    use_scaler: bool = True
    scaler_type: Literal["robust", "standard"] = "robust"  # "robust" = z-score + clip ±10σ (prevents overflow), "standard" = z-score only
    pca_n_components: int | None = None  # PCA dim reduction before classifier (None = disabled)
    pca_whiten: bool = False  # If True, whiten PCA output (unit variance per component)
    use_l2_norm: bool = False  # L2-normalize each sample before classifier
    logistic_regression: LogRegConfig = field(default_factory=LogRegConfig)
    linear_regression: LinearRegressionConfig = field(default_factory=LinearRegressionConfig)
