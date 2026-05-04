"""MHC-Benchmark: accessible evaluation API for wearable sensor models.

Quick start:

    >>> import openmhc
    >>> results = openmhc.evaluate_downstream(my_encoder)
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
from openmhc._protocols import Encoder, Imputer
from openmhc._results import DownstreamResults, ImputationResults

__all__ = [
    # Protocols
    "Encoder",
    "Imputer",
    # Evaluation functions
    "evaluate_downstream",
    "evaluate_imputation",
    # Dataset
    "download_dataset",
    "data_dir",
    # Discovery
    "list_tasks",
    "list_masking_scenarios",
    "SENSOR_CHANNELS",
    # Result types
    "DownstreamResults",
    "ImputationResults",
]


def evaluate_downstream(
    encoder: Encoder,
    tasks: str | list[str] = "all",
    data_dir: str | None = None,
    seed: int = 42,
) -> DownstreamResults:
    """Run downstream health prediction evaluation with a custom encoder.

    Args:
        encoder: Object implementing the Encoder protocol.
        tasks: "all" to run all 33 tasks, or a list of task names.
        data_dir: Path to the `daily_hourly_hf` dataset directory.
            None uses the default location.
        seed: Random seed for classifiers and splits.

    Returns:
        DownstreamResults with per-task metrics and a global score.
    """
    from openmhc._evaluate import evaluate_downstream as _eval

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


def list_tasks() -> list[str]:
    """Return all 33 available downstream prediction task names.

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
