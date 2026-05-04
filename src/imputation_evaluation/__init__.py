"""Imputation evaluation for MHC benchmark.

This package provides a framework for evaluating imputation methods
on daily sensor data using various masking scenarios.
"""

__all__ = ["ImputationEvalConfig", "ImputationEvaluator"]


def __getattr__(name: str):
    """Lazy import to avoid dependencies at module load."""
    if name == "ImputationEvalConfig":
        from imputation_evaluation.config import ImputationEvalConfig

        return ImputationEvalConfig
    elif name == "ImputationEvaluator":
        from imputation_evaluation.evaluation.evaluator import ImputationEvaluator

        return ImputationEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
