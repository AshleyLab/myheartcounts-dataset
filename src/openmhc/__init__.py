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
    bootstrap: bool | dict = False,
    max_samples: int | None = None,
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
        bootstrap: Opt-in participant-level cluster bootstrap. See
            :func:`openmhc._evaluate.evaluate_imputation` for the full
            shape of the option.
        max_samples: Limit samples per split for testing/debugging
            (None = no limit). Mirrors ``evaluate_forecasting``.

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
        bootstrap=bootstrap,
        max_samples=max_samples,
    )


def evaluate_forecasting(
    forecaster: Forecaster,
    version: Version,
    forecasting_length: int = 24,
    data_dir: str | Path | None = None,
    seed: int = 42,
    max_samples: int | None = None,
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
