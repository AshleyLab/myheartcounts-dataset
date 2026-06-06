"""MAE / LSM2 downstream method: dense ViT encoder + cohort pooling.

Self-contained package. ``model.py`` holds the ``MAE`` encoder, which extracts
per-day 384-d embeddings using the ``MaskedAutoencoderViT1D_LSM2`` architecture in
this package (``mae_vit1d``/``blocks``/``positional``/``utils``) and mean-pools them
per cohort; the engine then adds the uniform PCA-50 + linear probe.
"""

from __future__ import annotations

from downstream_evaluation.models.mae.model import MAE

__all__ = ["MAE"]
