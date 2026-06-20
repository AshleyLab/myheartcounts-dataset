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

# The implementation is shared across the imputation and forecasting training
# pipelines; it lives in ``openmhc._seeding``. Re-exported here to preserve the
# ``from imputation_training import seed_everything`` /
# ``from imputation_training.seeding import seed_everything`` public surface.
from openmhc._seeding import seed_everything

__all__ = ["seed_everything"]
