"""Round-trip tests for the LSM2 (formerly MAE) imputer wrappers.

Each test builds a tiny LSM2 Lightning module, saves a Lightning-style
``.ckpt``, loads it through the wrapper, and runs ``impute()`` on a
synthetic batch. Verifies the protocol contract: output shape and dtype,
target positions imputed (no NaN), non-target positions untouched.

Skipped automatically when ``pytorch_lightning`` is not installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pl = pytest.importorskip("pytorch_lightning")
torch = pytest.importorskip("torch")


N_CHANNELS = 19
N_STEPS = 48  # small for test speed
PATCH_SIZE = 8


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_synthetic_batch(
    n: int, n_steps: int = N_STEPS, seed: int = 0, missing_frac: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, N_CHANNELS, n_steps)).astype(np.float32)
    data[:, 7:, :] = (rng.random((n, 12, n_steps)) < 0.3).astype(np.float32)
    mask = (rng.random((n, N_CHANNELS, n_steps)) > missing_frac).astype(np.float32)
    data = np.where(mask > 0.5, data, np.nan)
    return data, mask


def _make_target_mask(
    observed_mask: np.ndarray, frac: float = 0.2, seed: int = 1
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    target = (observed_mask > 0.5) & (rng.random(observed_mask.shape) < frac)
    return target.astype(np.float32)


def _assert_round_trip(out: np.ndarray, data: np.ndarray, target: np.ndarray) -> None:
    assert out.shape == data.shape
    assert out.dtype == np.float32
    target_bool = target > 0.5
    assert np.isfinite(out[target_bool]).all()
    keep = ~target_bool
    np.testing.assert_array_equal(np.isnan(out[keep]), np.isnan(data[keep]))
    finite_keep = keep & np.isfinite(data)
    np.testing.assert_array_equal(out[finite_keep], data[finite_keep])


def _identity_stats_json(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "means": [0.0] * N_CHANNELS,
                "stds": [1.0] * N_CHANNELS,
                "channels": list(range(7)),
                "epsilon": 1e-8,
            }
        )
    )


def _save_lsm2_daily_ckpt(tmp_path: Path) -> Path:
    """Build a tiny daily LSM2 module and save a Lightning-style .ckpt."""
    from openmhc.models.lsm2.modules import LSM2Module

    module = LSM2Module(
        learning_rate=1e-4,
        seq_length=N_STEPS,
        patch_size=PATCH_SIZE,
        in_channels=N_CHANNELS,
        embed_dim=16,
        depth=1,
        num_heads=1,
        decoder_embed_dim=16,
        decoder_depth=1,
        decoder_num_heads=1,
        mlp_ratio=2.0,
    )
    ckpt_path = tmp_path / "lsm2.ckpt"
    torch.save(
        {
            "state_dict": module.state_dict(),
            "hyper_parameters": dict(module.hparams),
            "pytorch-lightning_version": pl.__version__,
            "LightningDataModule": {
                "normalization_stats": {
                    "mean_prior": [0.0] * N_CHANNELS,
                    "std_prior": [1.0] * N_CHANNELS,
                    "prior_count": 1e12,
                }
            },
        },
        ckpt_path,
    )
    return ckpt_path


def _save_lsm2_weekly_sparse_ckpt(tmp_path: Path, num_days: int = 2) -> Path:
    """Build a tiny weekly-sparse LSM2 module and save a Lightning-style .ckpt."""
    from openmhc.models.lsm2.modules import WeeklySparseLSM2Module

    module = WeeklySparseLSM2Module(
        learning_rate=1e-4,
        seq_length=N_STEPS,
        patch_size=PATCH_SIZE,
        in_channels=N_CHANNELS,
        embed_dim=16,
        depth=1,
        num_heads=1,
        decoder_embed_dim=16,
        decoder_depth=2,  # must be even (alternating day-local / cross-day)
        decoder_num_heads=1,
        mlp_ratio=2.0,
        num_days=num_days,
        window_minutes=24,
        use_rope_day_embed=False,
    )
    ckpt_path = tmp_path / "lsm2_weekly_sparse.ckpt"
    torch.save(
        {
            "state_dict": module.state_dict(),
            "hyper_parameters": dict(module.hparams),
            "pytorch-lightning_version": pl.__version__,
        },
        ckpt_path,
    )
    return ckpt_path


# ---------------------------------------------------------------------------
# Daily / weekly tests (same model class)
# ---------------------------------------------------------------------------


def test_lsm2_daily_round_trip(tmp_path):
    from openmhc.imputers import LSM2Imputer

    ckpt = _save_lsm2_daily_ckpt(tmp_path)
    stats = tmp_path / "stats.json"
    _identity_stats_json(stats)

    imp = LSM2Imputer(
        model_path=ckpt,
        version="xs",
        seq_length=N_STEPS,
        patch_size=PATCH_SIZE,
        in_channels=N_CHANNELS,
        embed_dim=16,
        depth=1,
        num_heads=1,
        decoder_embed_dim=16,
        decoder_depth=1,
        decoder_num_heads=1,
        mlp_ratio=2.0,
        device="cpu",
        inference_batch_size=2,
        inference_dropout_removal_ratio=0.0,
        normalization_stats_path=stats,
    )
    assert imp.name == "lsm2_lsm2"

    data, mask = _make_synthetic_batch(3, seed=0)
    target = _make_target_mask(mask, frac=0.1, seed=1)
    out = imp.impute(data, mask, target)
    _assert_round_trip(out, data, target)


def test_lsm2_directory_path_resolution(tmp_path):
    """Passing the directory should resolve to the .ckpt inside."""
    from openmhc.imputers import LSM2Imputer

    _save_lsm2_daily_ckpt(tmp_path)
    stats = tmp_path / "stats.json"
    _identity_stats_json(stats)

    imp = LSM2Imputer(
        model_path=tmp_path,
        version="xs",
        seq_length=N_STEPS,
        patch_size=PATCH_SIZE,
        device="cpu",
        inference_batch_size=2,
        inference_dropout_removal_ratio=0.0,
        normalization_stats_path=stats,
    )
    assert imp._ckpt_file.suffix == ".ckpt"


def test_lsm2_missing_path_raises(tmp_path):
    from openmhc.imputers import LSM2Imputer

    with pytest.raises(FileNotFoundError):
        LSM2Imputer(model_path=tmp_path / "does_not_exist", version="xs", device="cpu")


def test_lsm2_weekly_sparse_round_trip(tmp_path):
    from openmhc.imputers import LSM2WeeklySparseImputer

    num_days = 2
    ckpt = _save_lsm2_weekly_sparse_ckpt(tmp_path, num_days=num_days)
    # weekly-sparse trained on raw inputs in this synthetic test (no stats file).

    imp = LSM2WeeklySparseImputer(
        model_path=ckpt,
        version="xs",
        seq_length=N_STEPS,
        patch_size=PATCH_SIZE,
        in_channels=N_CHANNELS,
        embed_dim=16,
        depth=1,
        num_heads=1,
        decoder_embed_dim=16,
        decoder_depth=2,
        decoder_num_heads=1,
        mlp_ratio=2.0,
        num_days=num_days,
        window_minutes=24,
        use_rope_day_embed=False,
        device="cpu",
        inference_batch_size=2,
        inference_dropout_removal_ratio=0.0,
    )
    assert imp.name == "lsm2_lsm2_weekly_sparse"

    weekly_len = num_days * N_STEPS
    data, mask = _make_synthetic_batch(2, n_steps=weekly_len, seed=2)
    target = _make_target_mask(mask, frac=0.05, seed=3)
    out = imp.impute(data, mask, target)
    _assert_round_trip(out, data, target)


# ---------------------------------------------------------------------------
# Release manifest (from_release)
# ---------------------------------------------------------------------------


def _build_lsm2_release(tmp_path: Path) -> Path:
    """Lay out a tiny LSM2 release directory and return its path."""
    from openmhc.imputers import write_manifest

    ckpt = _save_lsm2_daily_ckpt(tmp_path)
    # Rename to model.ckpt so the layout matches what build_manifest produces.
    (tmp_path / "model.ckpt").write_bytes(ckpt.read_bytes())
    ckpt.unlink()

    _identity_stats_json(tmp_path / "normalization_stats.json")

    write_manifest(
        tmp_path,
        kind="lsm2",
        arch={
            "seq_length": N_STEPS,
            "patch_size": PATCH_SIZE,
            "in_channels": N_CHANNELS,
            "embed_dim": 16,
            "depth": 1,
            "num_heads": 1,
            "decoder_embed_dim": 16,
            "decoder_depth": 1,
            "decoder_num_heads": 1,
            "mlp_ratio": 2.0,
            "mask_ratio": 0.5,
        },
        checkpoint="model.ckpt",
        normalization_stats="normalization_stats.json",
        provenance={"training_run": "test", "dataset_version": "synthetic"},
    )
    return tmp_path


def test_lsm2_from_release_round_trip(tmp_path):
    from openmhc.imputers import LSM2Imputer

    release = _build_lsm2_release(tmp_path)
    imp = LSM2Imputer.from_release(
        release,
        version="xs",
        device="cpu",
        inference_batch_size=2,
        inference_dropout_removal_ratio=0.0,
    )
    assert imp.name == "lsm2_lsm2"

    data, mask = _make_synthetic_batch(2, seed=4)
    target = _make_target_mask(mask, frac=0.1, seed=5)
    out = imp.impute(data, mask, target)
    _assert_round_trip(out, data, target)


def test_lsm2_from_release_kind_mismatch_raises(tmp_path):
    from openmhc.imputers import LSM2WeeklySparseImputer

    release = _build_lsm2_release(tmp_path)
    with pytest.raises(ValueError, match="kind 'lsm2'"):
        LSM2WeeklySparseImputer.from_release(release, version="xs", device="cpu")
