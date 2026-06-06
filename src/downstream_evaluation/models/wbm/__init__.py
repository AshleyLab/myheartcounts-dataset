"""WBM downstream method: contrastive (Mamba2) weekly encoder.

Self-contained package. ``model.py`` holds the ``WBM`` encoder, which loads the
pretrained contrastive checkpoint and the ``week_encoders_mamba2``/``tokenizers``
architecture in this package to produce per-week embeddings; the engine then adds
the uniform PCA-50 + linear probe.
"""

from __future__ import annotations

from downstream_evaluation.models.wbm.model import DEFAULT_CHECKPOINT, WBM

__all__ = ["WBM", "DEFAULT_CHECKPOINT"]
