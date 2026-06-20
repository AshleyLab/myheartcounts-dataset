"""Deterministic seeding for forecasting PyPOTS training runs.

The implementation is shared with :mod:`imputation_training` and lives in
:mod:`openmhc._seeding`. Re-exported here so callers can use
``from forecasting_training import seed_everything`` /
``from forecasting_training.seeding import seed_everything``.

Must be called **before** model construction so PyTorch parameter init is
repeatable at a fixed seed.
"""

from __future__ import annotations

from openmhc._seeding import seed_everything

__all__ = ["seed_everything"]
