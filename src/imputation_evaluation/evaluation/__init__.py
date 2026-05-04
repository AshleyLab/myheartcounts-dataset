"""Evaluation module for imputation evaluation."""

__all__ = ["ImputationEvaluator", "PairWriter", "aggregate_pairs", "compute_scenario_metrics"]


def __getattr__(name: str):
    """Lazy import."""
    if name == "ImputationEvaluator":
        from imputation_evaluation.evaluation.evaluator import ImputationEvaluator

        return ImputationEvaluator
    elif name == "compute_scenario_metrics":
        from imputation_evaluation.evaluation.metrics import compute_scenario_metrics

        return compute_scenario_metrics
    elif name == "PairWriter":
        from imputation_evaluation.evaluation.pair_writer import PairWriter

        return PairWriter
    elif name == "aggregate_pairs":
        from imputation_evaluation.evaluation.pair_aggregator import aggregate_pairs

        return aggregate_pairs
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
