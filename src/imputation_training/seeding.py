"""Deterministic seeding for PyPOTS training runs.

The MHC-benchmark training pipeline declared a ``seed: 42`` config field
but never actually applied it before model construction. Combined with
the upstream PyPOTS FourierBlock bug (``np.random.shuffle`` at
``__init__`` time, indices NOT in state_dict), that made FEDformer
checkpoints non-reproducible across processes.

``seed_everything`` must be called **before** :func:`create_model` so
that:

1. ``FourierBlock.__init__`` draws its random frequency indices from a
   known state. Combined with the ``fourier_modes.json`` sidecar the
   trainer writes (see :func:`imputation_training.release.write_release`)
   this gives an end-to-end reproducible FEDformer release.

2. PyTorch parameter init (``nn.Parameter`` and friends) is identical
   across runs at the same seed, useful for ablation and debugging.

It does NOT enable ``torch.use_deterministic_algorithms(True)`` —
PyPOTS uses cuDNN convs which would refuse to run in deterministic mode
on common hardware. The model arch is fixed by the seed; numerical
training steps still have minor GPU non-determinism, which is fine for
reproducible release-bundle authoring (the trained weights vary at the
sub-percent level run-to-run, but the FourierBlock index pinning is
exact).
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs before model construction.

    Args:
        seed: Non-negative 32-bit integer.
    """
    if seed < 0 or seed >= 2**32:
        raise ValueError(f"seed must fit in uint32; got {seed!r}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Seeded random/numpy/torch (+ cuda if available) with seed=%d", seed)
