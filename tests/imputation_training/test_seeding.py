"""Verify that seed_everything makes FourierBlock construction deterministic.

This is the property the upstream PyPOTS bug breaks: without seeding,
``np.random.shuffle`` in ``FourierBlock.__init__`` produces a different
index list every time the model is constructed. With seeding (and a
fixed import order), two constructions in the same process should
produce identical indices.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pypots")

from imputation_training.seeding import seed_everything


def _build_fourier_block():
    """Tiny FourierBlock construction; isolates the bug surface."""
    from pypots.nn.modules.fedformer.layers import FourierBlock

    return FourierBlock(
        in_channels=8,
        out_channels=8,
        seq_len=1440,
        modes=8,
        mode_select_method="random",
    )


def test_seed_everything_pins_fourier_indices() -> None:
    seed_everything(42)
    a = _build_fourier_block()
    seed_everything(42)
    b = _build_fourier_block()
    assert list(a.index) == list(b.index), (
        f"FourierBlock indices differ across seeded constructions: "
        f"{a.index} vs {b.index}"
    )


def test_different_seeds_produce_different_indices() -> None:
    seed_everything(0)
    a = _build_fourier_block()
    seed_everything(123)
    b = _build_fourier_block()
    assert list(a.index) != list(b.index), (
        "Different seeds unexpectedly produced identical FourierBlock indices — "
        "is np.random.seed actually being applied?"
    )


def test_unseeded_constructions_diverge() -> None:
    """Without seeding, two constructions in the same process differ.

    We seed both, but with different seeds, to keep this test
    self-contained and robust to test-order ordering effects. The point
    is to demonstrate that the index IS state-dependent (which is the
    upstream bug) — the seeded versions just sidestep it.
    """
    np.random.seed(7)
    a = _build_fourier_block()
    # No re-seed: the second construction sees a different np.random state.
    b = _build_fourier_block()
    # With overwhelmingly high probability these differ (1 - C(720-32, 32)/C(720, 32)).
    assert list(a.index) != list(b.index)
