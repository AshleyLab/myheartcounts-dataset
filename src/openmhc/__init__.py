"""MHC-Benchmark: accessible evaluation API for wearable sensor models.

Quick start:

    >>> import openmhc
    >>> results = openmhc.evaluate_prediction(my_encoder)
    >>> results.summary()
    >>> results.global_score  # mean AUROC across binary tasks

    >>> results = openmhc.evaluate_imputation(my_imputer)
    >>> results.summary()

    >>> openmhc.list_tasks()              # 33 prediction tasks
    >>> openmhc.list_masking_scenarios()   # 6 masking scenarios
    >>> openmhc.SENSOR_CHANNELS           # 19 channel names
"""

from pathlib import Path

from openmhc._constants import MASKING_SCENARIOS, SENSOR_CHANNELS
from openmhc._data_utils import (
    iter_split_data,
    iter_train_data,
    load_sample_metadata,
)
from openmhc._dataset import (
    Version,
    data_dir,
    download_dataset,
    read_dataset_marker,
    write_dataset_marker,
)
from openmhc._protocols import Encoder, Forecaster, Imputer
from openmhc._results import (
    ForecastingResults,
    ImputationResults,
    PredictionResults,
)

__all__ = [
    # Protocols
    "Encoder",
    "Imputer",
    "Forecaster",
    # Evaluation functions
    "evaluate_prediction",
    "evaluate_imputation",
    "evaluate_forecasting",
    # Dataset
    "download_dataset",
    "data_dir",
    "read_dataset_marker",
    "write_dataset_marker",
    # Data utilities
    "iter_train_data",
    "iter_split_data",
    "load_sample_metadata",
    # Discovery
    "list_tasks",
    "list_masking_scenarios",
    "SENSOR_CHANNELS",
    # Result types
    "PredictionResults",
    "ImputationResults",
    "ForecastingResults",
]


def evaluate_prediction(
    encoder: Encoder,
    version: Version,
    tasks: str | list[str] = "all",
    data_dir: str | Path | None = None,
    seed: int = 42,
) -> PredictionResults:
    """Run health-prediction evaluation with a custom encoder.

    Args:
        encoder: Object implementing the Encoder protocol.
        version: ``"xs"`` or ``"full"``. Required — cross-checked against
            the dataset root's ``dataset_version.json`` marker.
        tasks: "all" to run all 33 tasks, or a list of task names.
        data_dir: Path to the dataset root. If omitted, ``MHC_DATA_DIR``
            must be set.
        seed: Random seed for classifiers and splits.

    Returns:
        PredictionResults with per-task metrics and a global score.
    """
    from openmhc._evaluate import evaluate_prediction as _eval

    return _eval(encoder, version=version, tasks=tasks, data_dir=data_dir, seed=seed)


def evaluate_imputation(
    imputer: Imputer,
    version: Version,
    masking_scenarios: str | list[str] = "all",
    data_dir: str | Path | None = None,
    seed: int = 42,
    *,
    n_days: int = 1,
    max_samples: int | None = None,
    num_workers: int = 0,
    num_eval_workers: int = 1,
    pin_memory: bool = False,
    output_dir: str | Path | None = None,
    baseline_errors: str | Path | None = None,
    keep_pairs: bool = False,
    method_name: str = "custom",
) -> ImputationResults:
    """Run imputation evaluation with a custom imputer.

    Args:
        imputer: Object implementing the Imputer protocol.
        version: ``"full"`` (11,894-user leaderboard split) or ``"xs"``
            (593-user reviewer subset). Required — cross-checked against
            the dataset root's ``dataset_version.json`` marker.
        masking_scenarios: "all" to run all 6 scenarios, or a list of
            scenario names.
        data_dir: Path to the dataset root directory. If omitted,
            ``MHC_DATA_DIR`` must be set.
        seed: Random seed for mask generation.
        n_days: Number of consecutive days per evaluation window (1-7).
            Defaults to ``1`` (single-day windows — matches all daily
            models). Set ``n_days=7`` for weekly models like
            ``LSM2WeeklySparseImputer`` or any 7-day PyPOTS variant; the
            imputer then receives tensors of shape ``(B, 19, n_days * 1440)``.
        max_samples: Limit samples per split for testing/debugging
            (None = no limit). Mirrors ``evaluate_forecasting``.
        num_workers: DataLoader worker processes (default ``0``). Raise toward
            your CPU count to overlap data loading with compute.
        num_eval_workers: Parallel processes for the evaluation loop (default
            ``1`` = sequential). ``> 1`` runs batches concurrently — much faster
            on the full split, with identical results. The imputer must be
            picklable (a class defined in a notebook cell can fail under the
            ``spawn`` start method; works under Linux ``fork``).
        pin_memory: DataLoader ``pin_memory`` flag (default ``False``).
        output_dir: Optional persistent directory for
            ``per_user_errors.parquet`` (and ``skill_scores.csv`` when
            ``baseline_errors`` is set). When ``None`` artifacts live
            in-memory on the returned ``ImputationResults`` only.
        baseline_errors: Path to a frozen single-method
            ``per_user_errors.parquet`` (typically the LOCF baseline
            shipped at ``src/openmhc/data/baselines/imputation_locf_per_user_errors.parquet``).
            Triggers paired-R skill-score computation against this
            baseline; result lands on ``ImputationResults.skill_scores``.
        keep_pairs: When ``True`` and ``output_dir`` is set, retain
            ``output_dir/pairs/`` after the call. Default ``False``
            deletes the pairs subdir once the producer has reduced it.
        method_name: Label embedded in the ``method`` column of emitted
            per-user errors. Defaults to ``"custom"``. Set to the
            canonical method identifier if you intend to concatenate
            with another method's file for downstream ranking via
            ``scripts/paper_results/compute_imputation_paper_metrics.py``.

    Returns:
        ImputationResults with per-scenario, per-split metrics; the
        additive ``per_user_errors`` / ``skill_scores`` fields are
        populated when ``output_dir`` or ``baseline_errors`` is set.

    Note:
        Cross-method **ranks** are not produced by the single-imputer
        public API. Run multiple ``evaluate_imputation(…,
        output_dir=outX, method_name="x")`` calls and then aggregate
        with::

            python scripts/paper_results/compute_imputation_paper_metrics.py \\
                --per-user-errors <dir of outX/per_user_errors.parquet files> \\
                --methods locf mean linear brits

        ``--methods`` selects the ranking pool; the baseline method must
        be in the selection.
    """
    from openmhc._evaluate import evaluate_imputation as _eval

    return _eval(
        imputer,
        version=version,
        masking_scenarios=masking_scenarios,
        data_dir=data_dir,
        seed=seed,
        n_days=n_days,
        max_samples=max_samples,
        num_workers=num_workers,
        num_eval_workers=num_eval_workers,
        pin_memory=pin_memory,
        output_dir=output_dir,
        baseline_errors=baseline_errors,
        keep_pairs=keep_pairs,
        method_name=method_name,
    )


def evaluate_forecasting(
    forecaster: Forecaster,
    version: Version,
    forecasting_length: int = 24,
    data_dir: str | Path | None = None,
    seed: int = 42,
    max_samples: int | None = None,
    *,
    num_workers: int = 4,
) -> ForecastingResults:
    """Run forecasting evaluation (Track 3) with a custom forecaster.

    Args:
        forecaster: Object implementing the Forecaster protocol —
            ``predict(history, horizon)`` returns ``(n_channels, horizon)``.
        version: ``"xs"`` or ``"full"``. Required — cross-checked against
            the dataset root's ``dataset_version.json`` marker.
        forecasting_length: Forecast horizon in hours. Defaults to 24.
        data_dir: Override for the dataset root. If omitted,
            ``MHC_DATA_DIR`` must be set.
        seed: Random seed.
        max_samples: Limit prediction samples per user (debugging only).
        num_workers: DataLoader worker processes for loading trajectories
            (default ``4``). The forecasting evaluator is sequential-only, so
            this only affects data loading; ``max_samples`` is the main speed
            lever.

    Returns:
        ForecastingResults with per-channel metrics.
    """
    from openmhc._evaluate import evaluate_forecasting as _eval

    return _eval(
        forecaster,
        version=version,
        forecasting_length=forecasting_length,
        data_dir=data_dir,
        seed=seed,
        max_samples=max_samples,
        num_workers=num_workers,
    )


def list_tasks() -> list[str]:
    """Return all 33 available prediction task names.

    Returns:
        Sorted list of task name strings.
    """
    from labels.api import TARGET_NAMES

    return sorted(TARGET_NAMES)


def list_masking_scenarios() -> list[str]:
    """Return all 6 available imputation masking scenario names.

    Returns:
        List of masking scenario name strings.
    """
    return list(MASKING_SCENARIOS)
