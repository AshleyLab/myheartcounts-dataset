"""MHC-Benchmark: accessible evaluation API for wearable sensor models.

Quick start:

    >>> import openmhc
    >>> results = openmhc.evaluate_prediction(my_model, version="full")
    >>> results.summary()

    >>> results = openmhc.evaluate_imputation(my_imputer, version="full")
    >>> results.summary()

    >>> openmhc.list_tasks()              # the 32 benchmark prediction tasks
    >>> openmhc.list_masking_scenarios()   # 6 masking scenarios
    >>> openmhc.SENSOR_CHANNELS           # 19 channel names
"""

from pathlib import Path

from openmhc._constants import MASKING_SCENARIOS, SENSOR_CHANNELS, TASK_DISPLAY_NAMES
from openmhc._data_spec import DataSpec
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
from openmhc._probe import LinearProbe
from openmhc._protocols import (
    CohortStream,
    EvalContext,
    Forecaster,
    Imputer,
    Method,
)
from openmhc._results import (
    ForecastingResults,
    ImputationResults,
    PredictionResults,
)

__all__ = [
    # Protocols
    "Method",
    "EvalContext",
    "DataSpec",
    "CohortStream",
    "Imputer",
    "Forecaster",
    # Standard probe (turns embeddings into predictions)
    "LinearProbe",
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
    "TASK_DISPLAY_NAMES",
    # Result types
    "PredictionResults",
    "ImputationResults",
    "ForecastingResults",
]


def evaluate_prediction(
    model: Method,
    version: Version,
    tasks: str | list[str] = "all",
    data_dir: str | Path | None = None,
    seed: int = 42,
    predictions_dir: str | Path | None = None,
) -> PredictionResults:
    """Run health-prediction evaluation with a custom model.

    Args:
        model: Object implementing the :class:`Method` protocol —
            ``fit(data, labels, task_type)`` / ``predict(data)`` on per-participant
            arrays. Declare input shape via ``data_spec`` (see :class:`DataSpec`); ``data``
            is a list, or a streamed :class:`CohortStream` for large specs. Encoder-style
            models run :class:`LinearProbe` inside ``fit`` / ``predict``.
        version: ``"xs"`` (593-user reviewer subset) or ``"full"``
            (11,894-user leaderboard split). Required — cross-checked against
            the dataset root's ``dataset_version.json`` marker.
        tasks: "all" to run the 32 benchmark tasks, or a list of task names.
        data_dir: Path to the dataset root. If omitted, ``MHC_DATA_DIR``
            must be set.
        seed: Random seed for classifiers and splits.
        predictions_dir: when set, write per-(method, task) test predictions +
            a shared ``_subgroups.json`` here, for the paper-metrics bootstrap.

    Returns:
        PredictionResults with per-task metrics.
    """
    from openmhc._evaluate import evaluate_prediction as _eval

    return _eval(
        model,
        version=version,
        tasks=tasks,
        data_dir=data_dir,
        seed=seed,
        predictions_dir=predictions_dir,
    )


def evaluate_imputation(
    imputer: Imputer,
    version: Version,
    masking_scenarios: str | list[str] = "all",
    data_dir: str | Path | None = None,
    seed: int = 42,
    *,
    n_days: int = 1,
    bootstrap: bool | dict = False,
    max_samples: int | None = None,
    num_workers: int = 0,
    num_eval_workers: int = 1,
    pin_memory: bool = False,
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
        bootstrap: Opt-in participant-level cluster bootstrap. See
            :func:`openmhc._evaluate.evaluate_imputation` for the full
            shape of the option.
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

    Returns:
        ImputationResults with per-scenario, per-split metrics.
    """
    from openmhc._evaluate import evaluate_imputation as _eval

    return _eval(
        imputer,
        version=version,
        masking_scenarios=masking_scenarios,
        data_dir=data_dir,
        seed=seed,
        n_days=n_days,
        bootstrap=bootstrap,
        max_samples=max_samples,
        num_workers=num_workers,
        num_eval_workers=num_eval_workers,
        pin_memory=pin_memory,
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
    """Return the 32 benchmark prediction task names.

    Returns:
        List of task name strings (the same set ``evaluate_prediction(tasks="all")`` runs).
    """
    from openmhc._constants import BENCHMARK_TASKS

    return list(BENCHMARK_TASKS)


def list_masking_scenarios() -> list[str]:
    """Return all 6 available imputation masking scenario names.

    Returns:
        List of masking scenario name strings.
    """
    return list(MASKING_SCENARIOS)
