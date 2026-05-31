"""Tests for the ``hf://`` URI branch of ``_resolve_manifest_path``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from openmhc.imputers import load_manifest
from openmhc.imputers._release import _resolve_hf_manifest, write_manifest


def _seed_bundle(dst: Path) -> Path:
    """Write a minimal valid BRITS-shaped bundle into ``dst`` and return it."""
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "model.pypots").write_bytes(b"\x00fake")
    (dst / "normalization_stats.json").write_text(
        json.dumps({"means": [0.0], "stds": [1.0], "channels": [0], "epsilon": 1e-8})
    )
    write_manifest(
        dst,
        kind="brits",
        arch={"n_steps": 1440, "n_features": 19, "rnn_hidden_size": 128},
        checkpoint="model.pypots",
        normalization_stats="normalization_stats.json",
    )
    return dst


def test_resolve_hf_manifest_calls_snapshot_download(tmp_path, monkeypatch):
    """``hf://org/repo`` should snapshot-download and return the manifest path."""
    bundle = _seed_bundle(tmp_path / "snapshot")
    calls: dict = {}

    def fake_snapshot_download(*, repo_id, revision, allow_patterns):
        calls["repo_id"] = repo_id
        calls["revision"] = revision
        calls["allow_patterns"] = allow_patterns
        return str(bundle)

    fake_hub = type(sys)("huggingface_hub")
    fake_hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    manifest_path = _resolve_hf_manifest("hf://MyHeartCounts/openmhc-brits-imp")

    assert manifest_path == bundle / "openmhc_manifest.json"
    assert calls["repo_id"] == "MyHeartCounts/openmhc-brits-imp"
    assert calls["revision"] is None
    # Allowlist must cover both the manifest and the payload extensions.
    assert "openmhc_manifest.json" in calls["allow_patterns"]
    assert "*.pypots" in calls["allow_patterns"]
    assert "*.ckpt" in calls["allow_patterns"]


def test_resolve_hf_manifest_forwards_revision(tmp_path, monkeypatch):
    """``@revision`` suffix should be split off and forwarded to snapshot_download."""
    bundle = _seed_bundle(tmp_path / "snapshot")
    captured: dict = {}

    def fake_snapshot_download(*, repo_id, revision, allow_patterns):
        captured["repo_id"] = repo_id
        captured["revision"] = revision
        return str(bundle)

    fake_hub = type(sys)("huggingface_hub")
    fake_hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    _resolve_hf_manifest("hf://MyHeartCounts/openmhc-brits-imp@v1.0")

    assert captured["repo_id"] == "MyHeartCounts/openmhc-brits-imp"
    assert captured["revision"] == "v1.0"


def test_load_manifest_through_hf_uri_round_trips(tmp_path, monkeypatch):
    """End-to-end: load_manifest('hf://...') should parse the downloaded bundle."""
    bundle = _seed_bundle(tmp_path / "snapshot")

    fake_hub = type(sys)("huggingface_hub")
    fake_hub.snapshot_download = lambda **_: str(bundle)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    m = load_manifest("hf://MyHeartCounts/openmhc-brits-imp")

    assert m.kind == "brits"
    assert m.arch == {"n_steps": 1440, "n_features": 19, "rnn_hidden_size": 128}
    assert m.checkpoint_path == bundle / "model.pypots"
    assert m.normalization_stats_path == bundle / "normalization_stats.json"


def test_resolve_hf_manifest_missing_manifest_file(tmp_path, monkeypatch):
    """A snapshot dir without an openmhc_manifest.json should error clearly."""
    empty = tmp_path / "empty-snapshot"
    empty.mkdir()

    fake_hub = type(sys)("huggingface_hub")
    fake_hub.snapshot_download = lambda **_: str(empty)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    with pytest.raises(FileNotFoundError, match="contains no openmhc_manifest.json"):
        _resolve_hf_manifest("hf://MyHeartCounts/openmhc-brits-imp")


def test_resolve_hf_manifest_missing_hf_extra(monkeypatch):
    """When huggingface_hub isn't importable, point users at the [hf] extra."""
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    with pytest.raises(ImportError, match=r"openmhc\[hf\]"):
        _resolve_hf_manifest("hf://MyHeartCounts/openmhc-brits-imp")
