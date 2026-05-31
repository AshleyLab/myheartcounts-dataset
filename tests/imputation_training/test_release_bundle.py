"""End-to-end test of the training-time release-bundle writer.

Builds a tiny in-memory FEDformer, extracts its FourierBlock indices,
writes a release bundle, then reloads via FEDformerImputer.from_release
and verifies the indices were restored byte-for-byte.

This is the critical contract that fixes the upstream PyPOTS bug: the
round-trip from training-time index → sidecar → fresh-process load must
preserve the exact same indices.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from imputation_training.release import (
    build_arch,
    extract_fourier_indices,
    write_release,
)
from imputation_training.seeding import seed_everything


def _build_tiny_fedformer(saving_path: Path):
    """Construct (but do NOT fit) a small FEDformer using the public factory.

    We don't train — the test only cares that the round-trip through
    write_release → from_release preserves FourierBlock.index. PyPOTS
    is happy to save/load a model that's only had its constructor run
    (state_dict is just random init).
    """
    from imputation_training.config import (
        ModelConfig,
        OutputConfig,
        TrainingConfig,
    )
    from imputation_training.model_registry import create_model

    model_cfg = ModelConfig(
        model_name="fedformer",
        n_steps=64,
        n_features=4,
        n_layers=1,
        d_model=16,
        n_heads=2,
        d_ffn=16,
        modes=4,
        mode_select="random",
        moving_avg_window_size=5,
    )
    train_cfg = TrainingConfig(epochs=1, batch_size=2, patience=1, device="cpu")
    out_cfg = OutputConfig(saving_path=str(saving_path))
    return create_model(model_cfg, train_cfg, out_cfg), model_cfg


def test_extract_fourier_indices_populated(tmp_path: Path) -> None:
    seed_everything(42)
    model, _ = _build_tiny_fedformer(tmp_path / "pypots")
    idx = extract_fourier_indices(model)
    assert idx, "expected at least one FourierBlock in FEDformer"
    for name, indices in idx.items():
        assert isinstance(name, str) and "." in name  # dotted path
        assert all(isinstance(i, int) for i in indices)


def test_build_arch_does_version_to_variant_rename() -> None:
    from imputation_training.config import ModelConfig

    arch = build_arch(ModelConfig(model_name="fedformer", version="Fourier", modes=8))
    assert arch["variant"] == "Fourier"
    assert "version" not in arch  # the openmhc inference side uses 'variant'


def test_extract_indices_empty_for_non_fedformer(tmp_path: Path) -> None:
    from imputation_training.config import (
        ModelConfig,
        OutputConfig,
        TrainingConfig,
    )
    from imputation_training.model_registry import create_model

    seed_everything(0)
    model_cfg = ModelConfig(
        model_name="dlinear", n_steps=64, n_features=4, moving_avg_window_size=5, d_model=16,
    )
    train_cfg = TrainingConfig(epochs=1, batch_size=2, patience=1, device="cpu")
    out_cfg = OutputConfig(saving_path=str(tmp_path / "pypots"))
    model = create_model(model_cfg, train_cfg, out_cfg)
    assert extract_fourier_indices(model) == {}


@pytest.mark.slow
def test_release_roundtrip_preserves_fourier_indices(tmp_path: Path) -> None:
    """The critical end-to-end test.

    Train-time indices → sidecar → load in a "fresh" state must
    produce byte-identical indices. We simulate the "fresh process"
    in-test by reseeding np.random with a different value between save
    and load — this guarantees the indices would diverge if the
    sidecar mechanism weren't restoring them.
    """
    pypots_dir = tmp_path / "pypots"
    release_dir = tmp_path / "release"

    # 1. Build under seed=42; capture indices.
    seed_everything(42)
    model, model_cfg = _build_tiny_fedformer(pypots_dir)
    train_indices = extract_fourier_indices(model)
    assert train_indices

    # 2. Save weights (PyPOTS's own save method). We don't go through
    #    model.fit() to keep this fast — random init weights are fine
    #    for the round-trip test.
    pypots_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = pypots_dir / "FEDformer.pypots"
    model.save(str(ckpt_path))

    # 3. Write a release bundle (also dumps fourier_modes.json).
    #    Skip normalization_stats — keep the test self-contained.
    write_release(
        model=model,
        model_config=model_cfg,
        release_dir=release_dir,
        pypots_checkpoint=ckpt_path,
        normalization_stats=None,
    )
    sidecar = release_dir / "fourier_modes.json"
    assert sidecar.exists()
    on_disk = json.loads(sidecar.read_text())
    assert on_disk == train_indices  # exact match

    # 4. Pollute np.random so a fresh construction would diverge.
    np.random.seed(99999)

    # 5. Load via FEDformerImputer.from_release.
    from openmhc.imputers import FEDformerImputer

    imputer = FEDformerImputer.from_release(release_dir, version="full", device="cpu")
    loaded_indices = extract_fourier_indices(imputer._model)

    # 6. The restored indices must match the training-time indices
    #    exactly — that's the whole point of the sidecar.
    assert loaded_indices == train_indices, (
        "Sidecar round-trip failed: loaded indices differ from training indices. "
        f"train={train_indices}, loaded={loaded_indices}"
    )
