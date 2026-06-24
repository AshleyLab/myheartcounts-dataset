"""Downstream outcome-prediction evaluation for the MHC benchmark.

This package evaluates ``Method``-protocol models — linear/logistic probes
(sklearn), gradient boosting, and neural encoders — on the benchmark prediction
tasks, sharing the existing data-loading and splitting infrastructure.
"""

# Lazy imports to avoid requiring sklearn at module load time
__all__ = ["EvalConfig", "DownstreamEvaluator"]


def __getattr__(name: str):
    """Lazy import to avoid sklearn dependency at module load."""
    if name == "EvalConfig":
        from downstream_evaluation.config import EvalConfig

        return EvalConfig
    elif name == "DownstreamEvaluator":
        from downstream_evaluation.evaluation.evaluator import DownstreamEvaluator

        return DownstreamEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
