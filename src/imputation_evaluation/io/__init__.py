"""IO utilities for imputation evaluation."""

__all__ = ["ResultsWriter"]


def __getattr__(name: str):
    """Lazy import."""
    if name == "ResultsWriter":
        from imputation_evaluation.io.writer import ResultsWriter

        return ResultsWriter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
