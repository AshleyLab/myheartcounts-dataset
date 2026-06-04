"""Evaluation module for imputation evaluation."""

__all__ = [
    "ImputationEvaluator",
    "PairWriter",
    "aggregate_pairs",
    "compute_scenario_metrics",
    "compute_per_draw_errors",
    "aggregate_skill_rank_fairness",
    "read_draws_parquet",
    "write_draws_parquet",
]


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
    elif name == "compute_per_draw_errors":
        from imputation_evaluation.evaluation.bootstrap_skill_rank import (
            compute_per_draw_errors,
        )

        return compute_per_draw_errors
    elif name == "aggregate_skill_rank_fairness":
        from imputation_evaluation.evaluation.bootstrap_skill_rank import (
            aggregate_skill_rank_fairness,
        )

        return aggregate_skill_rank_fairness
    elif name == "read_draws_parquet":
        from imputation_evaluation.evaluation.bootstrap_skill_rank import read_draws_parquet

        return read_draws_parquet
    elif name == "write_draws_parquet":
        from imputation_evaluation.evaluation.bootstrap_skill_rank import write_draws_parquet

        return write_draws_parquet
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
