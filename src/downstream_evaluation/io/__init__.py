"""Results writing utilities."""

__all__ = ["ResultsWriter"]


def __getattr__(name: str):
    """Lazy import to avoid heavy dependencies at module load."""
    if name == "ResultsWriter":
        from downstream_evaluation.io.writer import ResultsWriter

        return ResultsWriter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
