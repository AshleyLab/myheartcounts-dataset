"""LSM2 downstream method: dense ViT encoder + cohort pooling.

``model.py`` holds the ``LSM2`` encoder, which extracts per-day 384-d embeddings
using the ``LSM2ViT1D`` architecture from ``openmhc.models.lsm2`` and mean-pools
them per cohort; the engine then adds the uniform PCA-50 + linear probe.
"""

from __future__ import annotations

from downstream_evaluation.models.lsm2.model import LSM2

__all__ = ["LSM2"]
