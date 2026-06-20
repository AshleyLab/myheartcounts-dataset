"""Deterministic seeding shared across OpenMHC training pipelines.

A single ``seed_everything`` used by both :mod:`imputation_training` and
:mod:`forecasting_training`. It seeds Python's ``random``, NumPy, and PyTorch
(plus CUDA when available) **before** model construction, so randomized
architecture init (e.g. PyPOTS FEDformer's ``FourierBlock`` frequency-index
shuffle, which is not saved in ``state_dict``) is drawn from a known state and
parameter init is repeatable run-to-run at a fixed seed.

It intentionally does NOT enable ``torch.use_deterministic_algorithms(True)``:
PyPOTS uses cuDNN convolutions that refuse to run in deterministic mode on
common hardware. The model architecture is pinned by the seed; numerical
training steps still carry minor GPU non-determinism, which is acceptable for
reproducible release-bundle authoring.
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
