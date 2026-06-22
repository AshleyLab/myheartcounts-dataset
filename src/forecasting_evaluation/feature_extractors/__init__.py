"""Feature extractors for the forecasting track."""

__all__ = ["FeatureExtractor", "MultivariateFeatureExtractor"]


def __getattr__(name: str):
    """Lazy import to avoid heavy dependencies at module load."""
    if name == "FeatureExtractor":
        from forecasting_evaluation.feature_extractors.base import FeatureExtractor

        return FeatureExtractor
    if name == "MultivariateFeatureExtractor":
        from forecasting_evaluation.feature_extractors.multivariate_extractor import (
            MultivariateFeatureExtractor,
        )

        return MultivariateFeatureExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
