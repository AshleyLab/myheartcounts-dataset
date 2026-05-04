"""Sklearn-based downstream evaluation for MHC benchmark.

This package provides a flexible evaluation framework using sklearn classifiers
instead of PyTorch Lightning, while maintaining consistency with the existing
data loading and splitting infrastructure.
"""

# Lazy imports to avoid requiring sklearn at module load time
__all__ = ["DownstreamEvalConfig", "DownstreamEvaluator"]


def __getattr__(name: str):
    """Lazy import to avoid sklearn dependency at module load."""
    if name == "DownstreamEvalConfig":
        from downstream_evaluation.config import DownstreamEvalConfig

        return DownstreamEvalConfig
    elif name == "DownstreamEvaluator":
        from downstream_evaluation.evaluation.evaluator import DownstreamEvaluator

        return DownstreamEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
