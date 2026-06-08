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

from openmhc._constants import MASKING_SCENARIOS, SENSOR_CHANNELS
from openmhc._dataset import data_dir, download_dataset
from openmhc._protocols import Encoder, Forecaster, Imputer, Predictor
from openmhc._results import (
    ForecastingResults,
    ImputationResults,
    PredictionResults,
)

# Input specs — what shape of data your model receives. Declare one as ``input`` on your
# Encoder/Predictor; the framework hands you each cohort participant's IC/TC-bounded data.
from downstream_evaluation.data.inputs import Raw, Window

__all__ = [
    # Protocols
    "Encoder",
    "Predictor",
    "Imputer",
    "Forecaster",
    # Input specs (what data your model receives)
    "Raw",
    "Window",
    # Evaluation functions
    "evaluate_prediction",
    "evaluate_imputation",
    "evaluate_forecasting",
    # Dataset
    "download_dataset",
    "data_dir",
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
    tasks: str | list[str] = "all",
    data_dir: str | None = None,
    seed: int = 42,
) -> PredictionResults:
    """Run health-prediction evaluation with a custom encoder.

    Args:
        encoder: Object implementing the Encoder protocol.
        tasks: "all" to run all 33 tasks, or a list of task names.
        data_dir: Path to the `daily_hourly_hf` dataset directory.
            None uses the default location.
        seed: Random seed for classifiers and splits.

    Returns:
        PredictionResults with per-task metrics and a global score.
    """
    from openmhc._evaluate import evaluate_prediction as _eval

    return _eval(encoder, tasks=tasks, data_dir=data_dir, seed=seed)


def evaluate_imputation(
    imputer: Imputer,
    masking_scenarios: str | list[str] = "all",
    data_dir: str | None = None,
    seed: int = 42,
) -> ImputationResults:
    """Run imputation evaluation with a custom imputer.

    Args:
        imputer: Object implementing the Imputer protocol.
        masking_scenarios: "all" to run all 6 scenarios, or a list of
            scenario names.
        data_dir: Path to the `daily_hf` dataset directory.
            None uses the default location.
        seed: Random seed for mask generation.

    Returns:
        ImputationResults with per-scenario, per-split metrics.
    """
    from openmhc._evaluate import evaluate_imputation as _eval

    return _eval(imputer, masking_scenarios=masking_scenarios, data_dir=data_dir, seed=seed)


def evaluate_forecasting(
    forecaster: Forecaster,
    forecasting_length: int = 24,
    data_dir: str | None = None,
    seed: int = 42,
    max_samples: int | None = None,
) -> ForecastingResults:
    """Run forecasting evaluation (Track 3) with a custom forecaster.

    Args:
        forecaster: Object implementing the Forecaster protocol —
            ``predict(history, horizon)`` returns ``(n_channels, horizon)``.
        forecasting_length: Forecast horizon in hours. Defaults to 24.
        data_dir: Override for the dataset root. None uses the default.
        seed: Random seed.
        max_samples: Limit prediction samples per user (debugging only).

    Returns:
        ForecastingResults with per-channel metrics.
    """
    from openmhc._evaluate import evaluate_forecasting as _eval

    return _eval(
        forecaster,
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
