"""MHC-Benchmark: accessible evaluation API for wearable sensor models.

Quick start:

    >>> import openmhc
    >>> results = openmhc.evaluate_prediction(my_encoder)
    >>> results.summary()

    >>> results = openmhc.evaluate_imputation(my_imputer)
    >>> results.summary()

    >>> openmhc.list_tasks()              # the 32 benchmark prediction tasks
    >>> openmhc.list_masking_scenarios()   # 6 masking scenarios
    >>> openmhc.SENSOR_CHANNELS           # 19 channel names
"""

from openmhc._constants import MASKING_SCENARIOS, SENSOR_CHANNELS
from openmhc._data_spec import DataSpec
from openmhc._dataset import data_dir, download_dataset
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
from openmhc.probe import LinearProbe

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
    model: Method,
    tasks: str | list[str] = "all",
    data_dir: str | None = None,
    seed: int = 42,
    predictions_dir: str | None = None,
) -> PredictionResults:
    """Run health-prediction evaluation with a custom model.

    Args:
        model: Object implementing the :class:`Method` protocol —
            ``fit(data, labels, task_type)`` / ``predict(data)`` on per-participant
            arrays. Declare input shape via ``data_spec`` (see :class:`DataSpec`); ``data``
            is a list, or a streamed :class:`CohortStream` for large specs. Encoder-style
            models run :class:`LinearProbe` inside ``fit`` / ``predict``.
        tasks: "all" to run the 32 benchmark tasks, or a list of task names.
        data_dir: Path to the `daily_hourly_hf` dataset directory.
            None uses the default location.
        seed: Random seed for classifiers and splits.
        predictions_dir: when set, write per-(method, task) test predictions +
            a shared ``_subgroups.json`` here, for the paper-metrics bootstrap.

    Returns:
        PredictionResults with per-task metrics.
    """
    from openmhc._evaluate import evaluate_prediction as _eval

    return _eval(
        model, tasks=tasks, data_dir=data_dir, seed=seed, predictions_dir=predictions_dir
    )


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
