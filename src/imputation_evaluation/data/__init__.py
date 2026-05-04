"""Data loading utilities for imputation evaluation."""

__all__ = [
    "ImputationDataLoader",
    "ImputationDataset",
    "DailySample",
    "LoadedData",
    "MaskGenerationDataset",
    "load_split_file",
    "random_split_users",
]


def __getattr__(name: str):
    """Lazy import."""
    if name == "ImputationDataLoader":
        from imputation_evaluation.data.data_loader import ImputationDataLoader

        return ImputationDataLoader
    elif name == "ImputationDataset":
        from imputation_evaluation.data.data_loader import ImputationDataset

        return ImputationDataset
    elif name == "DailySample":
        from imputation_evaluation.data.data_loader import DailySample

        return DailySample
    elif name == "LoadedData":
        from imputation_evaluation.data.data_loader import LoadedData

        return LoadedData
    elif name == "MaskGenerationDataset":
        from imputation_evaluation.data.mask_dataset import MaskGenerationDataset

        return MaskGenerationDataset
    elif name in ("load_split_file", "random_split_users"):
        from imputation_evaluation.data import splits

        return getattr(splits, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
