"""PyPOTS imputer training pipeline for OpenMHC.

The companion to :mod:`imputation_evaluation`: same data layer, same splits,
and produces release bundles directly consumable by the eval Hydra CLI's
``method.release_dir=...`` flag.

Supported models (mirror the four neural imputers benchmarked in the
OpenMHC paper):
- BRITS
- DLinear
- TimesNet
- FEDformer

FEDformer training additionally captures each ``FourierBlock.index`` to a
``fourier_modes.json`` sidecar in the release bundle, so the resulting
checkpoint is reproducible across processes. Without that sidecar PyPOTS
re-draws the indices on load against an unknown ``np.random`` state and
the trained weights operate on the wrong frequency bins (the upstream
PyPOTS bug ``openmhc`` works around).

Public API:
    >>> from imputation_training import PyPOTSTrainingConfig, run_training
    >>> cfg = PyPOTSTrainingConfig(...)
    >>> release_dir = run_training(cfg)
    # release_dir is then consumable via FEDformerImputer.from_release(...)
"""

from __future__ import annotations

from imputation_training.config import (
    H5ExportConfig,
    ModelConfig,
    OutputConfig,
    PyPOTSTrainingConfig,
    TrainingConfig,
)
from imputation_training.runner import run_training
from imputation_training.seeding import seed_everything

__all__ = [
    "H5ExportConfig",
    "ModelConfig",
    "OutputConfig",
    "PyPOTSTrainingConfig",
    "TrainingConfig",
    "run_training",
    "seed_everything",
]
