"""Release-bundle assembly: contract + both manifest loaders parse it.

``write_release`` must produce a bundle that is loadable by BOTH the public
``openmhc.forecasters._release.load_manifest`` and the eval harness's
``forecasting_evaluation.hydra.release.load_forecasting_manifest`` (same
on-disk schema), and the manifest ``arch`` must agree with the bundled
``training_config.json``.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pytest

from forecasting_evaluation.data.standard_scaler import ChannelStandardScalerStats
from forecasting_evaluation.hydra.release import load_forecasting_manifest
from forecasting_training.config import ForecastingTrainingConfig
from forecasting_training.release import write_release
from openmhc.forecasters._release import load_manifest


def _make_scaler_stats(tmp_path, n_features: int):
    stats = ChannelStandardScalerStats(
        means=np.zeros(n_features),
        stds=np.ones(n_features),
        valid_counts=np.ones(n_features, dtype=np.int64),
    )
    path = tmp_path / "scaler.json"
    stats.save_stats_json(path)
    return path


def test_release_bundle_roundtrip(tmp_path) -> None:
    ckpt = tmp_path / "src.pypots"
    ckpt.write_bytes(b"dummy-checkpoint")
    cfg = ForecastingTrainingConfig()  # default dlinear, whether_standardscaler=True
    stats_path = _make_scaler_stats(tmp_path, cfg.model.n_features)

    bundle = write_release(
        model_name="dlinear",
        arch={
            "n_steps": cfg.model.n_steps,
            "n_pred_steps": cfg.model.n_pred_steps,
            "n_features": cfg.model.n_features,
        },
        training_config_json=asdict(cfg),
        release_dir=tmp_path / "release",
        pypots_checkpoint=ckpt,
        scaler_stats_path=stats_path,
        provenance={"seed": cfg.seed},
    )

    for fname in (
        "model.pypots",
        "standard_scaler_stats.json",
        "training_config.json",
        "openmhc_manifest.json",
    ):
        assert (bundle / fname).exists(), f"missing {fname}"

    # Both loaders parse the same bundle.
    public = load_manifest(bundle)
    harness = load_forecasting_manifest(bundle)
    assert public.kind == "dlinear" == harness.kind
    assert public.checkpoint_path.name == "model.pypots"
    assert public.normalization_stats_path is not None
    assert public.normalization_stats_path.name == "standard_scaler_stats.json"

    # Manifest arch agrees with the bundled training_config.json (the arch contract).
    tc = json.loads((bundle / "training_config.json").read_text())
    assert tc["model"]["n_steps"] == public.arch["n_steps"]
    assert tc["model"]["n_pred_steps"] == public.arch["n_pred_steps"]
    assert tc["training"]["whether_standardscaler"] is True


def test_release_requires_scaler_when_standardized(tmp_path) -> None:
    ckpt = tmp_path / "src.pypots"
    ckpt.write_bytes(b"dummy")
    cfg = ForecastingTrainingConfig()  # whether_standardscaler=True
    with pytest.raises(ValueError, match="whether_standardscaler"):
        write_release(
            model_name="dlinear",
            arch={"n_steps": 168, "n_pred_steps": 24, "n_features": 19},
            training_config_json=asdict(cfg),
            release_dir=tmp_path / "release",
            pypots_checkpoint=ckpt,
            scaler_stats_path=None,
        )


def test_release_rejects_stats_when_not_standardized(tmp_path) -> None:
    ckpt = tmp_path / "src.pypots"
    ckpt.write_bytes(b"dummy")
    cfg = ForecastingTrainingConfig()
    cfg.training.whether_standardscaler = False
    stats_path = _make_scaler_stats(tmp_path, cfg.model.n_features)
    with pytest.raises(ValueError):
        write_release(
            model_name="dlinear",
            arch={"n_steps": 168, "n_pred_steps": 24, "n_features": 19},
            training_config_json=asdict(cfg),
            release_dir=tmp_path / "release",
            pypots_checkpoint=ckpt,
            scaler_stats_path=stats_path,
        )
