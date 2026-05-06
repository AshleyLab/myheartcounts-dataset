"""Feature extractors for downstream evaluation."""

__all__ = ["FeatureExtractor", "BaselineFeatureExtractor", "EncoderFeatureExtractor"]


def __getattr__(name: str):
    """Lazy import to avoid heavy dependencies at module load."""
    if name == "FeatureExtractor":
        from downstream_evaluation.feature_extractors.base import FeatureExtractor

        return FeatureExtractor
    elif name == "BaselineFeatureExtractor":
        from downstream_evaluation.feature_extractors.baseline_extractor import (
            BaselineFeatureExtractor,
        )

        return BaselineFeatureExtractor
    elif name == "EncoderFeatureExtractor":
        from downstream_evaluation.feature_extractors.encoder_extractor import (
            EncoderFeatureExtractor,
        )

        return EncoderFeatureExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
