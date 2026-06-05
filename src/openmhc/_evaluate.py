"""Public evaluation functions for OpenMHC.

These functions provide a simple interface to the benchmark's evaluation
pipelines. They accept duck-typed Encoder/Imputer objects and return
structured results.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from openmhc._protocols import Encoder, Imputer

from openmhc._dataset import data_dir as _resolve_default_data_dir
from openmhc._results import ForecastingResults, ImputationResults, PredictionResults

logger = logging.getLogger(__name__)


@dataclass
class _DatasetPaths:
    """Resolved paths into the dataset directory.

    All paths are derived from a single ``root`` so the API stays consistent
    with what :func:`openmhc.download_dataset` produces. The expected layout
    matches DATASET.md.
    """

    root: Path
    daily_hourly_hf: Path
    daily_hf: Path
    window_index: Path
    weekly_labels_lookup: Path
    daily_labels_lookup: Path
    splits_file: Path
    norm_stats: Path
    labels_dir: Path
    hourly_trajectory: Path
    forecasting_sample_index_dir: Path

    @classmethod
    def resolve(cls, override: str | Path | None = None) -> _DatasetPaths:
        """Build the paths bundle from an explicit override or the default.

        Resolution order matches :func:`openmhc.data_dir`:

        1. ``override`` argument (if provided)
        2. ``MHC_DATA_DIR`` env var
        3. ``~/.cache/openmhc/data``
        """
        root = _resolve_default_data_dir(override)
        return cls(
            root=root,
            daily_hourly_hf=root / "processed" / "daily_hourly_hf",
            daily_hf=root / "processed" / "daily_hf",
            window_index=root / "processed" / "window_index_w7_s7_d5.parquet",
            weekly_labels_lookup=root / "processed" / "weekly_labels_lookup_stride7_windowed.parquet",
            daily_labels_lookup=root / "processed" / "daily_labels_lookup.parquet",
            splits_file=root / "splits" / "sharable_users_seed42_2026.json",
            norm_stats=root / "processed" / "normalization_stats_hourly.json",
            labels_dir=root / "labels",
            hourly_trajectory=root / "hourly_trajectory",
            forecasting_sample_index_dir=root / "forecasting_sample_index",
        )


def _ensure_labels_env(labels_dir: Path) -> None:
    """Point the bundled `labels.api` module at the downloaded labels dir.

    `labels.api` reads its data-file paths from env vars at import time and
    caches them in module-level Path constants. We set each var if the user
    hasn't, then reload the module if it was already imported (e.g. via
    ``openmhc.list_tasks()``) so the cached paths reflect the new values.

    Without this, paths fall back to the repo-local ``data/labels/`` (which
    ships only schema files), and `enrollment_info.json` / `label_validity.json`
    silently load as empty — breaking the imputation sensitivity pathway and
    the default ``return_valid_only=True`` behaviour of ``get_labels``.
    """
    env_files = {
        "LABELS_DATA_PATH": "last_labels.json",
        "CONTEXT_LABELS_PATH": "context_labels.json",
        "ENROLLMENT_DATA_PATH": "enrollment_info.json",
        "LABEL_VALIDITY_PATH": "label_validity.json",
        "HEALTHKIT_DAILY_PATH": "healthkit_daily.json",
    }
    changed = False
    for var, filename in env_files.items():
        if not os.getenv(var):
            os.environ[var] = str(labels_dir / filename)
            changed = True

    if changed:
        import importlib
        import sys

        if "labels.api" in sys.modules:
            importlib.reload(sys.modules["labels.api"])


# ---------------------------------------------------------------------------
# Prediction evaluation
# ---------------------------------------------------------------------------


def evaluate_prediction(
    encoder: Encoder,
    tasks: str | list[str] = "all",
    data_dir: str | Path | None = None,
    seed: int = 42,
) -> PredictionResults:
    """Run health-prediction evaluation with a custom encoder.

    Encodes weekly sensor tensors via `encoder.encode()` and evaluates the
    resulting embeddings on up to 33 health prediction tasks using linear
    probes (logistic regression for binary, ordinal logistic for ordinal,
    ridge regression for continuous).

    Args:
        encoder: Object with an `encode(weekly_tensors) -> embeddings` method.
            Input shape is (B, 168, 38), output shape is (B, D).
        tasks: "all" to run all 33 tasks, or a list of task name strings.
        data_dir: Override for the dataset root (the same root that
            ``download_dataset`` writes to). ``None`` uses the default
            (``MHC_DATA_DIR`` env var or ``~/.cache/openmhc/data``). All
            sub-paths (`processed/daily_hourly_hf/`, `splits/`, `labels/`,
            etc.) are derived from this root.
        seed: Random seed for classifiers and splits.

    Returns:
        A PredictionResults instance with per-task metrics and a global score
        (mean AUROC across binary tasks).
    """
    paths = _DatasetPaths.resolve(data_dir)
    _ensure_labels_env(paths.labels_dir)

    from downstream_evaluation.data.splits import load_split_file
    from downstream_evaluation.evaluation.metrics import get_task_type
    from downstream_evaluation.runner import EvalConfig, run_eval
    from labels.api import TARGET_NAMES

    if tasks == "all":
        task_list = sorted(TARGET_NAMES)
    elif isinstance(tasks, str):
        task_list = [tasks]
    else:
        task_list = list(tasks)

    split_users = load_split_file(paths.splits_file)

    # Both surfaces run through the one engine (mirrors evaluate_imputation →
    # run_eval): an external encoder exposes ``encode``; a bundled baseline exposes
    # ``encode_cohort``/``fit``. run_eval selects the path, applies the uniform
    # PCA-50 + linear probe, and reports the primary metric per task type. The
    # per-task temporal scope is baked into the windowed lookup (no eval-time knob).
    cfg = EvalConfig(
        data_dir=str(paths.root), split_users=split_users, tasks=task_list, seed=seed
    )
    # A pure external encoder implements only the public ``encode(data)`` contract, so
    # wrap it to translate the engine's per-participant ``ParticipantSegments`` into the
    # documented ``(n_segments, 24, 38)`` array (mirrors ``_ImputerMethodAdapter`` /
    # ``_ForecasterAdapter``). Bundled baselines speak the internal interface
    # (``encode_cohort`` / ``fit``) and pass through untouched.
    model = encoder
    if hasattr(encoder, "encode") and not hasattr(encoder, "encode_cohort"):
        model = _EncoderMethodAdapter(encoder)
    results = run_eval(cfg, model)

    # Flatten the engine's ``{task: {metric: value, n_test}}`` into long-format
    # records; the global score is the mean test AUPRC over binary tasks (AUPRC is
    # the primary binary metric).
    probe_by_type = {
        "binary": "logistic_regression",
        "multiclass": "logistic_regression",
        "ordinal": "logreg_ordinal",
        "regression": "linear_regression",
    }
    records: list[dict] = []
    binary_primary: list[float] = []
    for task_name, task_metrics in results.items():
        if task_name == "config":
            continue
        task_type = get_task_type(task_name)
        n_test = task_metrics.get("n_test")
        for metric_name, value in task_metrics.items():
            if metric_name == "n_test":
                continue
            records.append({
                "task": task_name,
                "task_type": task_type,
                "classifier": probe_by_type.get(task_type),
                "metric": metric_name,
                "value": value,
                "n_test": n_test,
            })
        if task_type == "binary" and "auprc" in task_metrics:
            binary_primary.append(task_metrics["auprc"])

    global_score = float(np.mean(binary_primary)) if binary_primary else 0.0
    return PredictionResults(records=records, global_score=global_score)


class _EncoderMethodAdapter:
    """Adapt a user's :class:`~openmhc.Encoder` to the internal engine interface.

    The internal evaluator hands a model one ``ParticipantSegments`` per participant —
    raw ``.values`` and ``.mask``, each ``(n_segments, 24, 19)``. The public
    ``Encoder.encode`` contract is a single ``(n_segments, 24, 38)`` array (channels
    0-18 raw sensor values with NaN at missing positions, 19-37 the missingness mask).
    This adapter performs that translation and forwards to the user's ``encode``,
    mirroring ``_ImputerMethodAdapter`` / ``_ForecasterAdapter`` so all three tracks
    expose the same clean-array public contract.
    """

    def __init__(self, encoder: Encoder) -> None:
        """Wrap ``encoder`` and inherit its declared input granularity."""
        self._encoder = encoder
        # Respect the encoder's declared granularity so the engine binds the matching
        # segments (the binder currently materializes daily segments).
        self.input_granularity = getattr(encoder, "input_granularity", "daily")

    @property
    def name(self) -> str:
        """Method name for run provenance."""
        return getattr(self._encoder, "name", "custom_encoder")

    def encode(self, segments) -> np.ndarray:
        """Translate one participant's ``ParticipantSegments`` to the public array."""
        values = np.asarray(segments.values, dtype=np.float32)  # (n, 24, 19) raw, NaN at missing
        mask = np.asarray(segments.mask, dtype=np.float32)  # (n, 24, 19)
        data = np.concatenate([values, mask], axis=-1)  # (n, 24, 38)
        return np.asarray(self._encoder.encode(data), dtype=np.float32)


# ---------------------------------------------------------------------------
# Imputation evaluation
# ---------------------------------------------------------------------------


def evaluate_imputation(
    imputer: Imputer,
    masking_scenarios: str | list[str] = "all",
    data_dir: str | Path | None = None,
    seed: int = 42,
) -> ImputationResults:
    """Run imputation evaluation with a custom imputer.

    Fits the imputer on training data, then evaluates on validation and test
    sets using up to 6 masking scenarios.

    Args:
        imputer: Object with `fit(data, masks)` and
            `impute(data, observed_mask, target_mask)` methods.
        masking_scenarios: "all" to run all 6 scenarios, or a list of scenario
            name strings.
        data_dir: Override for the dataset root (the same root that
            ``download_dataset`` writes to). ``None`` uses the default
            (``MHC_DATA_DIR`` env var or ``~/.cache/openmhc/data``). All
            sub-paths (`processed/daily_hf/`, `splits/`, `labels/`, etc.)
            are derived from this root.
        seed: Random seed for mask generation.

    Returns:
        An ImputationResults instance with per-scenario, per-split metrics.

    Raises:
        ValueError: If an unknown masking scenario name is provided.
    """
    from openmhc._constants import MASKING_SCENARIOS

    # Resolve scenarios.
    if masking_scenarios == "all":
        scenario_list = list(MASKING_SCENARIOS)
    elif isinstance(masking_scenarios, str):
        scenario_list = [masking_scenarios]
    else:
        scenario_list = list(masking_scenarios)

    # Validate scenario names.
    for s in scenario_list:
        if s not in MASKING_SCENARIOS:
            raise ValueError(
                f"Unknown masking scenario: {s!r}. "
                f"Valid scenarios: {MASKING_SCENARIOS}"
            )

    paths = _DatasetPaths.resolve(data_dir)
    _ensure_labels_env(paths.labels_dir)

    from imputation_evaluation.config import (
        DataConfig,
        EvalConfig,
        ImputationEvalConfig,
        MaskingConfig,
        MethodConfig,
        OutputConfig,
        SensitivityConfig,
        VisualizationConfig,
        WandbConfig,
    )
    from imputation_evaluation.runner import run_eval

    masking_cfg = MaskingConfig(mask_seed=seed)
    masking_cfg.random_noise.enabled = "random_noise" in scenario_list
    masking_cfg.temporal_slice.enabled = "temporal_slice" in scenario_list
    masking_cfg.signal_slice.enabled = "signal_slice" in scenario_list
    masking_cfg.sleep_gap.enabled = "sleep_gap" in scenario_list
    masking_cfg.workout_gap.enabled = "workout_gap" in scenario_list
    masking_cfg.intensity_failure.enabled = "intensity_failure" in scenario_list

    data_cfg = DataConfig(
        daily_hf_dir=str(paths.daily_hf),
        split_file=str(paths.splits_file),
        split_seed=seed,
        batch_size=5000,
        num_workers=4,
        num_eval_workers=1,
    )

    eval_cfg = EvalConfig(
        include_ks=False,
        include_wasserstein=False,
        compute_metrics=True,
        save_pairs=False,
    )

    cfg = ImputationEvalConfig(
        seed=seed,
        data=data_cfg,
        masking=masking_cfg,
        method=MethodConfig(type="mean"),  # placeholder; not used (custom adapter below)
        output=OutputConfig(),
        evaluation=eval_cfg,
        visualization=VisualizationConfig(),
        sensitivity=SensitivityConfig(),
        wandb=WandbConfig(),
    )

    adapter = _ImputerMethodAdapter(imputer)
    logger.info("Running imputation eval with custom imputer...")
    results = run_eval(cfg, method=adapter)
    return ImputationResults(scenarios=results.get("scenarios", results))


class _ImputerMethodAdapter:
    """Adapt a user's Imputer to the internal ImputationMethod interface.

    This adapter collects batched training data, computes channel standard
    deviations for normalized metrics, and translates argument names between
    the public Imputer protocol and the internal evaluation pipeline.

    Attributes:
        name: Name of the imputation method (defaults to "custom_imputer").
        channel_stds: Per-channel standard deviations computed during fit,
            or None before fit is called.
    """

    def __init__(self, imputer: Imputer) -> None:
        """Initialize the adapter.

        Args:
            imputer: Object implementing the Imputer protocol.
        """
        self._imputer = imputer
        self._channel_stds: np.ndarray | None = None

    @property
    def name(self) -> str:
        """Return the method name."""
        return getattr(self._imputer, "name", "custom_imputer")

    @property
    def channel_stds(self) -> np.ndarray | None:
        """Return per-channel standard deviations from training data."""
        return self._channel_stds

    def fit(self, train_loader) -> None:
        """Collect training data from the loader and call the user's fit.

        Args:
            train_loader: DataLoader yielding batches of training data with
                "values" and "mask" keys.
        """
        all_data = []
        all_masks = []
        for batch in train_loader:
            data = batch["values"] if isinstance(batch, dict) else batch[0]
            mask = batch["mask"] if isinstance(batch, dict) else batch[1]
            all_data.append(np.asarray(data, dtype=np.float32))
            all_masks.append(np.asarray(mask, dtype=np.float32))

        data = np.concatenate(all_data, axis=0)
        masks = np.concatenate(all_masks, axis=0)

        # Compute channel stds for normalized metrics.
        observed = masks > 0.5
        stds = []
        for c in range(data.shape[1]):
            vals = data[:, c, :][observed[:, c, :]]
            stds.append(float(np.std(vals)) if len(vals) > 1 else 1.0)
        self._channel_stds = np.array(stds, dtype=np.float32)

        self._imputer.fit(data, masks)

    def impute(
        self,
        data: np.ndarray,
        original_masks: np.ndarray,
        artificial_masks: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Delegate to the user's impute, translating argument names.

        Internal callers may pass extra kwargs (``sample_indices``,
        ``day_offsets``) used by personalized / RoPE-aware methods. The
        public ``Imputer`` protocol doesn't expose those, so we discard
        them silently.
        """
        return self._imputer.impute(
            data=data,
            observed_mask=original_masks,
            target_mask=artificial_masks,
        )

    def prepare_split(self, *args, **kwargs) -> None:
        """No-op; some internal methods use this hook."""


# ---------------------------------------------------------------------------
# Forecasting evaluation (Track 3)
# ---------------------------------------------------------------------------


def evaluate_forecasting(
    forecaster,
    forecasting_length: int = 24,
    data_dir: str | Path | None = None,
    seed: int = 42,
    max_samples: int | None = None,
) -> ForecastingResults:
    """Run forecasting evaluation (Track 3) with a custom forecaster.

    Args:
        forecaster: Object satisfying the :class:`Forecaster` protocol —
            has ``predict(history, horizon)`` returning a ``(n_channels,
            horizon)`` array.
        forecasting_length: Forecast horizon in hours. Defaults to 24
            (matching the paper's Track 3 sub-task).
        data_dir: Override for the dataset root. ``None`` uses the default
            (``MHC_DATA_DIR`` env var or ``~/.cache/openmhc/data``).
        seed: Random seed.
        max_samples: Limit prediction samples per user (debugging).

    Returns:
        :class:`ForecastingResults` with per-channel metrics.
    """
    from openmhc._results import ForecastingResults

    paths = _DatasetPaths.resolve(data_dir)
    _ensure_labels_env(paths.labels_dir)

    from forecasting_evaluation.config import (
        DataConfig,
        EvaluatorConfig,
        FeaturesConfig,
        ForecastingConfig,
        ForecastingEvalConfig,
        ForecastingModelConfig,
        OutputConfig,
    )
    from forecasting_evaluation.runner import run_eval

    # Pick a sample-index file matching the requested forecasting horizon.
    sample_index_file = paths.forecasting_sample_index_dir / "sample_index_raw.json"

    data_cfg = DataConfig(
        trajectory_hf_dir=str(paths.hourly_trajectory),
        split_file=str(paths.splits_file),
        day_remain_mask=str(paths.forecasting_sample_index_dir / "day_remain_mask.json"),
        sample_index_file=str(sample_index_file),
        split_seed=seed,
        max_samples=max_samples,
    )
    forecasting_cfg = ForecastingConfig(forecasting_length=forecasting_length)

    with tempfile.TemporaryDirectory(prefix="openmhc-fc-") as tmp_results:
        cfg = ForecastingEvalConfig(
            seed=seed,
            experiment_name="openmhc_run",
            debug_mode=False,
            data=data_cfg,
            forecasting=forecasting_cfg,
            model=ForecastingModelConfig(),  # ignored by _CustomModelEvaluator
            features=FeaturesConfig(),
            evaluator=EvaluatorConfig(),
            output=OutputConfig(results_dir=tmp_results),
        )
        adapter = _build_forecaster_adapter(forecaster)
        result = run_eval(cfg, model=adapter)

    return ForecastingResults(
        per_channel=result.get("per_channel", {}),
        run_dir=str(result.get("run_dir", "")),
        n_samples=int(result.get("n_samples", 0)),
    )


def _build_forecaster_adapter(forecaster):
    """Wrap a user's ``Forecaster`` as an internal ``BasePredictionModel``.

    Subclasses ``BasePredictionModel`` so we inherit ``predict_wrapper`` (which
    adds timing + memory tracking around the user's ``predict()`` call).
    """
    from forecasting_evaluation.models.base import BasePredictionModel

    class _ForecasterAdapter(BasePredictionModel):
        model_name = "openmhc_custom_forecaster"
        quantile_levels = None
        uses_standard_scaler = False
        scaler_stats = None

        def __init__(self, forecaster):
            self._forecaster = forecaster

        def predict(self, inputs):
            """Translate ``SubTrajectoryInput`` → user's ``predict(history, horizon)``."""
            point = self._forecaster.predict(inputs.history, inputs.prediction_hours)
            return np.asarray(point, dtype=np.float32), None

        def reset(self):
            if hasattr(self._forecaster, "reset"):
                self._forecaster.reset()

    return _ForecasterAdapter(forecaster)
