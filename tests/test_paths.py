"""Path-resolution tests for the openmhc API.

These verify that the API and the dataset downloader agree on where data
lives. Without this, ``download_dataset`` would write to one location and
``evaluate_*`` would look in another.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from openmhc._dataset import data_dir
from openmhc._evaluate import _DatasetPaths


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Clear dataset / labels env vars so each test starts from a known state.

    Without this, tests that set ``LABELS_DATA_PATH`` / ``CONTEXT_LABELS_PATH``
    would leak into sibling test modules (e.g. ``src/labels/test_api.py``)
    and make ordering matter.
    """
    for var in (
        "MHC_DATA_DIR",
        "LABELS_DATA_PATH",
        "CONTEXT_LABELS_PATH",
        "ENROLLMENT_DATA_PATH",
    ):
        monkeypatch.delenv(var, raising=False)
    yield


class TestDataDir:
    def test_default_under_user_cache(self):
        result = data_dir()
        assert result == Path.home() / ".cache" / "openmhc" / "data"

    def test_explicit_override_wins(self, tmp_path):
        result = data_dir(tmp_path / "explicit")
        assert result == (tmp_path / "explicit").resolve()

    def test_env_var_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MHC_DATA_DIR", str(tmp_path / "from-env"))
        result = data_dir()
        assert result == (tmp_path / "from-env").resolve()

    def test_explicit_wins_over_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MHC_DATA_DIR", str(tmp_path / "from-env"))
        result = data_dir(tmp_path / "explicit")
        assert result == (tmp_path / "explicit").resolve()

    def test_tilde_expansion(self):
        result = data_dir("~/some-mhc-data")
        assert result == (Path.home() / "some-mhc-data").resolve()


class TestDatasetPaths:
    """Confirm the eval pipeline routes through ``data_dir`` consistently."""

    def test_default_root_matches_data_dir(self):
        paths = _DatasetPaths.resolve()
        assert paths.root == data_dir()

    def test_all_subpaths_under_root(self, tmp_path):
        paths = _DatasetPaths.resolve(tmp_path)
        assert paths.daily_hourly_hf == tmp_path / "processed" / "daily_hourly_hf"
        assert paths.daily_hf == tmp_path / "processed" / "daily_hf"
        assert paths.window_index == tmp_path / "processed" / "window_index_w7_s7_d5.parquet"
        assert (
            paths.weekly_labels_lookup
            == tmp_path / "processed" / "weekly_labels_lookup_stride7.parquet"
        )
        assert paths.splits_file == tmp_path / "splits" / "sharable_users_seed42_2026.json"
        assert paths.norm_stats == tmp_path / "processed" / "normalization_stats_hourly.json"
        assert paths.clip_dates == tmp_path / "labels" / "clip_dates.json"
        assert paths.labels_dir == tmp_path / "labels"

    def test_env_override_propagates(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MHC_DATA_DIR", str(tmp_path))
        paths = _DatasetPaths.resolve()
        assert paths.root == tmp_path.resolve()
        assert paths.daily_hf.parent.parent == tmp_path.resolve()

    def test_explicit_override_propagates(self, tmp_path):
        paths = _DatasetPaths.resolve(tmp_path / "explicit")
        assert paths.root == (tmp_path / "explicit").resolve()


class TestLabelsEnvWiring:
    """Verify the labels.api env-var bridge is set when missing."""

    def test_sets_labels_path_when_unset(self, monkeypatch, tmp_path):
        monkeypatch.delenv("LABELS_DATA_PATH", raising=False)
        monkeypatch.delenv("CONTEXT_LABELS_PATH", raising=False)
        from openmhc._evaluate import _ensure_labels_env

        _ensure_labels_env(tmp_path / "labels")
        assert os.environ["LABELS_DATA_PATH"] == str(tmp_path / "labels" / "last_labels.json")
        assert os.environ["CONTEXT_LABELS_PATH"] == str(
            tmp_path / "labels" / "context_labels.json"
        )

    def test_respects_user_overrides(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LABELS_DATA_PATH", "/somewhere/else.json")
        monkeypatch.setenv("CONTEXT_LABELS_PATH", "/somewhere/ctx.json")
        from openmhc._evaluate import _ensure_labels_env

        _ensure_labels_env(tmp_path / "labels")
        assert os.environ["LABELS_DATA_PATH"] == "/somewhere/else.json"
        assert os.environ["CONTEXT_LABELS_PATH"] == "/somewhere/ctx.json"


class TestDownloadDatasetSurface:
    """Verify the download helper signature and error paths."""

    def test_unknown_version_raises(self):
        from openmhc import download_dataset

        with pytest.raises(ValueError, match="version must be one of"):
            download_dataset(version="bogus")

    def test_unpublished_version_raises(self):
        from openmhc import download_dataset

        with pytest.raises(ValueError, match="not yet published"):
            download_dataset(version="full")
