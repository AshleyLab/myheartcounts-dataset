"""Tests for the spec v2 manifest format and the fourier_modes sidecar.

Covers the local-disk path through ``load_manifest`` / ``write_manifest``.
The Hugging Face Hub branch is tested separately in ``test_release_hf.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmhc.imputers._release import (
    _SUPPORTED_SPEC_VERSIONS,
    SPEC_VERSION,
    Manifest,
    load_manifest,
    write_manifest,
)


def _seed_files(dst: Path, *, fourier: bool = False) -> None:
    """Drop the sibling files a manifest typically points at."""
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "model.pypots").write_bytes(b"\x00fake")
    (dst / "normalization_stats.json").write_text(
        json.dumps({"means": [0.0], "stds": [1.0], "channels": [0], "epsilon": 1e-8})
    )
    if fourier:
        (dst / "fourier_modes.json").write_text(
            json.dumps(
                {
                    "encoder.attn_layers.0.attention.inner_correlation": [
                        0,
                        3,
                        7,
                        11,
                        18,
                        24,
                        29,
                        31,
                    ],
                }
            )
        )


def test_current_spec_version_is_two() -> None:
    """v2 is the live writer; v1 stays loadable for back-compat."""
    assert SPEC_VERSION == 2
    assert 1 in _SUPPORTED_SPEC_VERSIONS
    assert 2 in _SUPPORTED_SPEC_VERSIONS


def test_write_then_load_v2_with_sidecar(tmp_path: Path) -> None:
    """A fedformer manifest round-trips at spec v2 with its fourier_modes sidecar."""
    _seed_files(tmp_path, fourier=True)
    write_manifest(
        tmp_path,
        kind="fedformer",
        arch={"n_steps": 1440, "n_features": 19, "modes": 8},
        checkpoint="model.pypots",
        normalization_stats="normalization_stats.json",
        fourier_modes="fourier_modes.json",
        provenance={"trained_on": "openmhc-data-full"},
    )
    manifest = load_manifest(tmp_path)
    assert isinstance(manifest, Manifest)
    assert manifest.spec_version == 2
    assert manifest.kind == "fedformer"
    assert manifest.fourier_modes_path is not None
    assert manifest.fourier_modes_path.name == "fourier_modes.json"
    payload = json.loads(manifest.fourier_modes_path.read_text())
    assert "encoder.attn_layers.0.attention.inner_correlation" in payload


def test_write_then_load_v2_without_sidecar(tmp_path: Path) -> None:
    """Non-fedformer kinds skip the field; manifest still spec v2."""
    _seed_files(tmp_path)
    write_manifest(
        tmp_path,
        kind="brits",
        arch={"n_steps": 1440, "n_features": 19, "rnn_hidden_size": 128},
        checkpoint="model.pypots",
        normalization_stats="normalization_stats.json",
    )
    raw = json.loads((tmp_path / "openmhc_manifest.json").read_text())
    assert raw["spec_version"] == 2
    assert "fourier_modes" not in raw  # absent (not None) keeps the file compact
    manifest = load_manifest(tmp_path)
    assert manifest.fourier_modes_path is None


def test_load_v1_manifest_is_still_supported(tmp_path: Path) -> None:
    """All existing on-disk checkpoint bundles are spec_version 1."""
    _seed_files(tmp_path)
    (tmp_path / "openmhc_manifest.json").write_text(
        json.dumps(
            {
                "spec_version": 1,
                "kind": "brits",
                "checkpoint": "model.pypots",
                "normalization_stats": "normalization_stats.json",
                "arch": {"n_steps": 1440, "n_features": 19, "rnn_hidden_size": 128},
                "provenance": {},
            }
        )
    )
    manifest = load_manifest(tmp_path)
    assert manifest.spec_version == 1
    assert manifest.fourier_modes_path is None


def test_write_rejects_sidecar_for_non_fedformer_kind(tmp_path: Path) -> None:
    """Writing a fourier_modes sidecar for a non-fedformer kind raises ValueError."""
    _seed_files(tmp_path, fourier=True)
    with pytest.raises(ValueError, match="cannot carry a 'fourier_modes' sidecar"):
        write_manifest(
            tmp_path,
            kind="brits",
            arch={"n_steps": 1440, "n_features": 19, "rnn_hidden_size": 128},
            checkpoint="model.pypots",
            normalization_stats="normalization_stats.json",
            fourier_modes="fourier_modes.json",
        )


def test_load_rejects_sidecar_paired_with_wrong_kind(tmp_path: Path) -> None:
    """A hand-edited manifest mispairing brits + sidecar must be rejected."""
    _seed_files(tmp_path, fourier=True)
    (tmp_path / "openmhc_manifest.json").write_text(
        json.dumps(
            {
                "spec_version": 2,
                "kind": "brits",
                "checkpoint": "model.pypots",
                "normalization_stats": "normalization_stats.json",
                "fourier_modes": "fourier_modes.json",
                "arch": {"n_steps": 1440, "n_features": 19, "rnn_hidden_size": 128},
                "provenance": {},
            }
        )
    )
    with pytest.raises(ValueError, match="cannot carry a 'fourier_modes' sidecar"):
        load_manifest(tmp_path)


def test_load_rejects_unknown_spec_version(tmp_path: Path) -> None:
    """Loading a manifest with an unsupported spec_version raises ValueError."""
    _seed_files(tmp_path)
    (tmp_path / "openmhc_manifest.json").write_text(
        json.dumps(
            {
                "spec_version": 99,
                "kind": "brits",
                "checkpoint": "model.pypots",
                "normalization_stats": "normalization_stats.json",
                "arch": {"n_steps": 1440, "n_features": 19, "rnn_hidden_size": 128},
                "provenance": {},
            }
        )
    )
    with pytest.raises(ValueError, match="Unsupported manifest spec_version"):
        load_manifest(tmp_path)


def test_load_rejects_missing_sidecar_file(tmp_path: Path) -> None:
    """If the manifest references a sidecar that doesn't exist on disk, fail loudly."""
    _seed_files(tmp_path)
    (tmp_path / "openmhc_manifest.json").write_text(
        json.dumps(
            {
                "spec_version": 2,
                "kind": "fedformer",
                "checkpoint": "model.pypots",
                "normalization_stats": "normalization_stats.json",
                "fourier_modes": "fourier_modes.json",  # not seeded!
                "arch": {"n_steps": 1440, "n_features": 19, "modes": 8},
                "provenance": {},
            }
        )
    )
    with pytest.raises(FileNotFoundError, match="fourier_modes sidecar"):
        load_manifest(tmp_path)
