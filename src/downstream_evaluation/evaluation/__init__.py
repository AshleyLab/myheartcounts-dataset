"""Evaluation metrics and orchestration."""

__all__ = [
    "DownstreamEvaluator",
    "compute_binary_metrics",
    "compute_multiclass_metrics",
    "compute_regression_metrics",
    "compute_ordinal_metrics",
    "get_task_type",
]


def __getattr__(name: str):
    """Lazy import to avoid sklearn dependency at module load."""
    if name == "DownstreamEvaluator":
        from downstream_evaluation.evaluation.evaluator import DownstreamEvaluator

        return DownstreamEvaluator
    elif name in (
        "compute_binary_metrics",
        "compute_multiclass_metrics",
        "compute_regression_metrics",
        "compute_ordinal_metrics",
        "get_task_type",
    ):
        from downstream_evaluation.evaluation import metrics

        return getattr(metrics, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
