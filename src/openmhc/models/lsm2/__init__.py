"""LSM2 (Large Sensor Model 2) — masked autoencoder ViT for 1D wearable data.

Adaptation of Google's LSM2 wearable foundation model: Xu et al., "LSM-2:
Learning from Incomplete Wearable Sensor Data" (2025), https://arxiv.org/abs/2506.05321.

The model code is vendored from the private MHC-benchmark repo. Two model
classes are exposed:

- :class:`LSM2ViT1D` — daily and weekly (configurable ``seq_length``, ``patch_size``)
  MAE-style encoder/decoder. Used by ``LSM2Imputer``.
- :class:`WeeklySparseDecoderLSM2` — frozen-daily-encoder + sparse cross-day decoder.
  Used by ``LSM2WeeklySparseImputer``.

The :func:`create_inherited_mask` helper builds the patch-level mask from the
sample-level boolean missingness mask used at inference time.
"""

from openmhc.models.lsm2.utils import create_inherited_mask
from openmhc.models.lsm2.vit1d import LSM2ViT1D
from openmhc.models.lsm2.weekly_sparse_decoder import WeeklySparseDecoderLSM2

__all__ = [
    "LSM2ViT1D",
    "WeeklySparseDecoderLSM2",
    "create_inherited_mask",
]
