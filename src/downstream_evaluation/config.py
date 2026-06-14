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
    """Forward-window policy — the before-label window every method shares.

    The benchmark uses each participant's full history: by default there is no forward
    window, so a participant's whole record is eligible (subject to the data-quality
    inclusion criteria). ``weeks_after`` returns ``None`` for every task, the cohort
    comes from the inclusion-criteria-only lookup, and from-raw models (Toto/Chronos-2)
    build their input over the full history.

    The forward-windowed policy is retained for ablation via :meth:`windowed`: it caps
    each task's eligible region at ``label + weeks_after`` weeks (52 by default; 156 for
    age and BiologicalSex, which include data further from the label date), and is baked
    into the prebuilt ``*_windowed`` label lookups the cohort methods read.
    """

    default_weeks_after: int | None = None
    task_weeks_after: dict[str, int] = field(default_factory=dict)

    def weeks_after(self, task: str) -> int | None:
        """Forward-window length in weeks for ``task`` (``None`` = no cap, full history)."""
        return self.task_weeks_after.get(task, self.default_weeks_after)

    @property
    def is_full_history(self) -> bool:
        """True when no forward-window cap applies (the benchmark default)."""
        return self.default_weeks_after is None and not self.task_weeks_after

    @classmethod
    def windowed(cls) -> TemporalWindowConfig:
        """Forward-windowed ablation policy (52 weeks; age and BiologicalSex 156)."""
        return cls(default_weeks_after=52, task_weeks_after={"age": 156, "BiologicalSex": 156})


@dataclass
class EvalConfig:
    """Config for the prediction engine.

    Args:
        data_dir: dataset root (its ``processed/`` holds the lookups + sensor data).
        split_users: ``{"train"/"validation"/"test": [user_id, ...]}``.
        tasks: tasks to evaluate.
        seed: recorded in run provenance (models own their seeds; the uniform
            probe runs inside each method via ``openmhc.LinearProbe``).
        temporal: the per-task forward-window policy (handed to from-raw models).
        predictions_dir: when set, the evaluator writes per-(method, task) test
            predictions + a shared ``_subgroups.json`` here (input to the
            paper-metrics bootstrap); ``None`` disables prediction export.
    """

    data_dir: str
    split_users: dict
    tasks: list[str] = field(default_factory=list)
    seed: int = 42
    temporal: TemporalWindowConfig = field(default_factory=TemporalWindowConfig)
    predictions_dir: str | None = None


# --------------------------------------------------------------------------- #
# Hydra CLI config tree (consumed by ``mhc-downstream-eval``; composed from
# ``configs/downstream/*``). Separate from ``EvalConfig`` above (the engine
# config): the CLI translates this into an ``openmhc.evaluate_prediction`` call.
# Mirrors the imputation track's ``ImputationEvalConfig`` so the two CLIs share
# the ``eval_hydra`` plumbing and feel identical to an open-source user.
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    """Dataset location for the CLI; ``None`` falls back to ``MHC_DATA_DIR``."""

    data_dir: str | None = None


@dataclass
class MethodConfig:
    """Which bundled model to run + its build-on-miss knobs.

    ``type`` selects the registry builder. The remaining fields are the
    per-method options that used to be env vars / constructor args: ``checkpoint``
    (MAE / WBM), ``features_dir`` + ``max_future_days`` (XGBoost). Builders ignore
    the fields they don't read.
    """

    type: str = "linear"
    checkpoint: str | None = None
    features_dir: str | None = None
    max_future_days: int = 0


@dataclass
class EvaluationConfig:
    """What to evaluate. ``tasks`` is ``"all"`` or an explicit list of task names."""

    tasks: str | list[str] = "all"


@dataclass
class OutputConfig:
    """Where results land.

    ``predictions_dir`` (when set) enables the per-task prediction parquets +
    ``_subgroups.json`` for the paper-metrics bootstrap.
    """

    results_dir: str = "results/eval"
    predictions_dir: str | None = None


@dataclass
class DownstreamEvalConfig:
    """Root config for the ``mhc-downstream-eval`` Hydra CLI."""

    data: DataConfig = field(default_factory=DataConfig)
    method: MethodConfig = field(default_factory=MethodConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    seed: int = 42


# --------------------------------------------------------------------------- #
# Probe hyperparameters (composed by ClassifierConfig, read by create_model)
# --------------------------------------------------------------------------- #
# Task type → the linear head used for it — the one mapping every probe shares
# (openmhc.LinearProbe and the model-internal probes alike).
PROBE_BY_TASK_TYPE: dict[str, str] = {
    "binary": "logistic_regression",
    "multiclass": "logistic_regression",
    "ordinal": "logreg_ordinal",
    "regression": "linear_regression",
}
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
