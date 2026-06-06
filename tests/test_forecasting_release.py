"""Tests for the forecasting release manifest (``openmhc.forecasters._release``).

Covers the local-disk round-trip through ``write_manifest`` / ``load_manifest``
and the ``ReleaseLoadableMixin.from_release`` kind-matching contract. Mirrors the
imputation manifest tests in ``test_release.py``; the Hugging Face Hub branch is
exercised by the shared ``snapshot_download`` path tested in ``test_release_hf.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmhc.forecasters._release import (
    SPEC_VERSION,
    Manifest,
    ReleaseLoadableMixin,
    load_manifest,
    write_manifest,
)


def _seed_neural(dst: Path) -> None:
    """Seed a neural-style bundle: a .pypots file plus co-located scaler."""
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "OnlineDLinear.pypots").write_bytes(b"\x00fake")
    (dst / "training_config.json").write_text(json.dumps({"model": {"n_steps": 336}}))
    (dst / "standard_scaler_stats.json").write_text(
        json.dumps({"means": [0.0], "stds": [1.0]})
    )


def _seed_chronos(dst: Path) -> None:
    """Seed a Chronos-style bundle: a merged model sub-directory, no stats."""
    ckpt = dst / "checkpoint"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "config.json").write_text("{}")
    (ckpt / "model.safetensors").write_bytes(b"\x00fake")


def test_spec_version_is_one() -> None:
    """Forecasting bundles share the harness manifest schema (spec_version 1)."""
    assert SPEC_VERSION == 1


def test_write_then_load_neural_bundle(tmp_path: Path) -> None:
    """Round-trip a neural bundle: dir checkpoint + co-located scaler + arch."""
    _seed_neural(tmp_path)
    write_manifest(
        tmp_path,
        kind="dlinear",
        checkpoint=".",
        arch={"n_steps": 336, "n_features": 19},
        normalization_stats="standard_scaler_stats.json",
        provenance={"trained_on": "MHC training split"},
    )
    manifest = load_manifest(tmp_path)
    assert isinstance(manifest, Manifest)
    assert manifest.spec_version == 1
    assert manifest.kind == "dlinear"
    assert manifest.checkpoint_path.is_dir()
    assert manifest.normalization_stats_path is not None
    assert manifest.normalization_stats_path.name == "standard_scaler_stats.json"
    assert manifest.arch == {"n_steps": 336, "n_features": 19}


def test_write_then_load_chronos_bundle_dir_checkpoint(tmp_path: Path) -> None:
    """Chronos-2 checkpoint is a directory and carries no normalization stats."""
    _seed_chronos(tmp_path)
    write_manifest(
        tmp_path,
        kind="chronos2",
        checkpoint="checkpoint",
        normalization_stats=None,
        provenance={"base_model": "amazon/chronos-2", "finetune_mode": "lora"},
    )
    manifest = load_manifest(tmp_path)
    assert manifest.kind == "chronos2"
    assert manifest.checkpoint_path.is_dir()
    assert (manifest.checkpoint_path / "config.json").exists()
    assert manifest.normalization_stats_path is None


def test_write_rejects_unknown_kind(tmp_path: Path) -> None:
    """write_manifest refuses a kind outside the known forecasting set."""
    _seed_neural(tmp_path)
    with pytest.raises(ValueError, match="Unknown manifest kind"):
        write_manifest(tmp_path, kind="not_a_model", checkpoint=".")


def test_load_rejects_unknown_kind(tmp_path: Path) -> None:
    """load_manifest refuses a hand-edited manifest with an unknown kind."""
    _seed_neural(tmp_path)
    (tmp_path / "openmhc_manifest.json").write_text(
        json.dumps(
            {
                "spec_version": 1,
                "kind": "bogus",
                "checkpoint": ".",
                "normalization_stats": None,
                "arch": {},
                "provenance": {},
            }
        )
    )
    with pytest.raises(ValueError, match="Unknown manifest kind"):
        load_manifest(tmp_path)


def test_load_rejects_unknown_spec_version(tmp_path: Path) -> None:
    """Only spec_version 1 manifests are accepted."""
    _seed_neural(tmp_path)
    (tmp_path / "openmhc_manifest.json").write_text(
        json.dumps(
            {
                "spec_version": 99,
                "kind": "dlinear",
                "checkpoint": ".",
                "normalization_stats": None,
                "arch": {},
                "provenance": {},
            }
        )
    )
    with pytest.raises(ValueError, match="Unsupported forecasting manifest spec_version"):
        load_manifest(tmp_path)


def test_load_rejects_missing_checkpoint(tmp_path: Path) -> None:
    """A manifest pointing at an absent checkpoint fails loudly."""
    (tmp_path / "openmhc_manifest.json").write_text(
        json.dumps(
            {
                "spec_version": 1,
                "kind": "toto",
                "checkpoint": "model.ckpt",  # not seeded
                "normalization_stats": None,
                "arch": {},
                "provenance": {},
            }
        )
    )
    with pytest.raises(FileNotFoundError, match="missing checkpoint"):
        load_manifest(tmp_path)


def test_from_release_rejects_mismatched_kind(tmp_path: Path) -> None:
    """A wrapper must refuse a bundle whose kind it does not handle."""
    _seed_neural(tmp_path)
    write_manifest(tmp_path, kind="dlinear", checkpoint=".")

    class _SegRNNLike(ReleaseLoadableMixin):
        model_name = "segrnn"

        def __init__(self, **_kwargs):
            pass

    with pytest.raises(ValueError, match="expects kind 'segrnn'"):
        _SegRNNLike.from_release(tmp_path)


def test_from_release_forwards_paths_and_arch(tmp_path: Path) -> None:
    """from_release splats checkpoint, stats, and arch into the constructor."""
    _seed_neural(tmp_path)
    write_manifest(
        tmp_path,
        kind="dlinear",
        checkpoint=".",
        arch={"n_steps": 336},
        normalization_stats="standard_scaler_stats.json",
    )
    captured: dict = {}

    class _DLinearLike(ReleaseLoadableMixin):
        model_name = "dlinear"

        def __init__(self, **kwargs):
            captured.update(kwargs)

    _DLinearLike.from_release(tmp_path, device="cpu")
    assert Path(captured["model_path"]).name == tmp_path.name
    assert captured["normalization_stats_path"].endswith("standard_scaler_stats.json")
    assert captured["n_steps"] == 336
    assert captured["device"] == "cpu"
