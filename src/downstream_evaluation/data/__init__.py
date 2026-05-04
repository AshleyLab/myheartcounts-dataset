"""Data loading and aggregation utilities."""

__all__ = ["DownstreamDataLoader", "aggregate_by_user", "prepare_daily_hourly_hf"]


def __getattr__(name: str):
    """Lazy import to avoid heavy dependencies at module load."""
    if name == "DownstreamDataLoader":
        from downstream_evaluation.data.data_loader import DownstreamDataLoader

        return DownstreamDataLoader
    elif name == "aggregate_by_user":
        from downstream_evaluation.data.aggregation import aggregate_by_user

        return aggregate_by_user
    elif name == "prepare_daily_hourly_hf":
        from downstream_evaluation.data.data_loader import prepare_daily_hourly_hf

        return prepare_daily_hourly_hf
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
