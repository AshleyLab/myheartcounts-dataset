"""Tests for the Track-1 encoder release manifest (``openmhc.encoders._release``).

Covers the local-disk path through ``load_manifest`` / ``write_manifest`` and the
``from_release`` kind guard. All CPU-only: no checkpoint is loaded (the Mamba2
encoder needs CUDA), only the manifest round-trips.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmhc.encoders import WBM
from openmhc.encoders._release import (
    SPEC_VERSION,
    Manifest,
    ReleaseLoadableMixin,
    load_manifest,
    write_manifest,
)

_ARCH = {
    "in_dim": 38,
    "embed_dim": 256,
    "hidden_dim": 64,
    "num_layers": 4,
    "proj_dim": 128,
    "dropout": 0.223,
}


def _seed_files(dst: Path) -> None:
    """Drop the sibling files a WBM manifest points at."""
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "model.ckpt").write_bytes(b"\x00fake")
    (dst / "normalization_stats.json").write_text(
        json.dumps({"means": [0.0] * 19, "stds": [1.0] * 19})
    )


def test_spec_version_is_one() -> None:
    """Encoder bundles share the spec v1 manifest schema with forecasting."""
    assert SPEC_VERSION == 1


def test_write_then_load_wbm(tmp_path: Path) -> None:
    """A wbm manifest round-trips with its checkpoint + normalization stats."""
    _seed_files(tmp_path)
    write_manifest(
        tmp_path,
        kind="wbm",
        checkpoint="model.ckpt",
        arch=_ARCH,
        normalization_stats="normalization_stats.json",
        provenance={"source_artifact": "wandb:MHC_Dataset/.../WBM_Final_HPO_best:v1"},
    )
    manifest = load_manifest(tmp_path)
    assert isinstance(manifest, Manifest)
    assert manifest.spec_version == 1
    assert manifest.kind == "wbm"
    assert manifest.arch == _ARCH
    assert manifest.checkpoint_path.name == "model.ckpt"
    assert manifest.normalization_stats_path is not None
    assert manifest.normalization_stats_path.name == "normalization_stats.json"
    assert manifest.provenance["source_artifact"].startswith("wandb:")


def test_write_rejects_unknown_kind(tmp_path: Path) -> None:
    """Only ``wbm`` is a known encoder kind."""
    _seed_files(tmp_path)
    with pytest.raises(ValueError, match="Unknown manifest kind"):
        write_manifest(tmp_path, kind="dlinear", checkpoint="model.ckpt")


def test_load_rejects_unknown_kind(tmp_path: Path) -> None:
    """A hand-edited manifest with a foreign kind is rejected at load."""
    _seed_files(tmp_path)
    (tmp_path / "openmhc_manifest.json").write_text(
        json.dumps(
            {
                "spec_version": 1,
                "kind": "dlinear",
                "checkpoint": "model.ckpt",
                "normalization_stats": "normalization_stats.json",
                "arch": {},
                "provenance": {},
            }
        )
    )
    with pytest.raises(ValueError, match="Unknown manifest kind"):
        load_manifest(tmp_path)


def test_load_rejects_unknown_spec_version(tmp_path: Path) -> None:
    """Loading a manifest with an unsupported spec_version raises ValueError."""
    _seed_files(tmp_path)
    (tmp_path / "openmhc_manifest.json").write_text(
        json.dumps(
            {
                "spec_version": 99,
                "kind": "wbm",
                "checkpoint": "model.ckpt",
                "normalization_stats": "normalization_stats.json",
                "arch": _ARCH,
                "provenance": {},
            }
        )
    )
    with pytest.raises(ValueError, match="Unsupported encoder manifest spec_version"):
        load_manifest(tmp_path)


def test_load_rejects_missing_checkpoint(tmp_path: Path) -> None:
    """If the manifest references a checkpoint that doesn't exist, fail loudly."""
    (tmp_path / "normalization_stats.json").write_text("{}")
    write_manifest(
        tmp_path,
        kind="wbm",
        checkpoint="model.ckpt",  # not seeded
        arch=_ARCH,
        normalization_stats="normalization_stats.json",
    )
    with pytest.raises(FileNotFoundError, match="missing checkpoint"):
        load_manifest(tmp_path)


def test_from_release_rejects_kind_mismatch(tmp_path: Path) -> None:
    """``from_release`` refuses a bundle whose kind != the wrapper's model_name.

    Uses a throwaway subclass so the guard is exercised without constructing the
    CUDA-only WBM encoder.
    """
    _seed_files(tmp_path)
    write_manifest(tmp_path, kind="wbm", checkpoint="model.ckpt", arch=_ARCH)

    class _Other(ReleaseLoadableMixin):
        model_name = "other"

    with pytest.raises(ValueError, match="expects kind 'other'"):
        _Other.from_release(tmp_path)


def test_wbm_wrapper_method_attributes() -> None:
    """The public WBM wrapper advertises the engine routing attrs (no construction)."""
    assert WBM.model_name == "wbm"
    assert WBM.name == "wbm"
    assert WBM.input_granularity == "daily"
    assert WBM.needs_segments is True
