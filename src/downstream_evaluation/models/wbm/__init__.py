"""WBM downstream method: contrastive (Mamba2) weekly encoder.

Self-contained package. ``model.py`` holds the bare ``WBM`` encoder (loads the
pretrained contrastive checkpoint + the ``week_encoders_mamba2``/``tokenizers``
architecture to produce per-week embeddings) and ``WBMProbe`` — the reported WBM
*model*: the encoder + the uniform :class:`openmhc.LinearProbe`, returning
non-finite for participants without a weekly embedding so the harness substitutes
the Linear baseline (the missing-prediction fallback, issue #38).
"""

from __future__ import annotations

from downstream_evaluation.models.wbm.model import DEFAULT_CHECKPOINT, WBM, WBMProbe

__all__ = ["WBM", "WBMProbe", "DEFAULT_CHECKPOINT"]
