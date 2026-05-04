"""Sklearn classifier registry and factory."""

__all__ = ["create_model"]


def __getattr__(name: str):
    """Lazy import to avoid sklearn dependency at module load."""
    if name == "create_model":
        from downstream_evaluation.models.registry import create_model

        return create_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
