"""Path-resolution tests for the openmhc API.

These verify that the API and the dataset downloader agree on where data
lives. Without this, ``download_dataset`` would write to one location and
``evaluate_*`` would look in another.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pandas as pd
import pytest

import openmhc
from openmhc._dataset import bundled_metadata_dir, data_dir
from openmhc._evaluate import _DatasetPaths


def _write_minimal_split(root: Path, name: str = "sharable_users_seed42_2026.json") -> None:
    split_dir = root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / name).write_text('{"train": [], "validation": [], "test": []}')


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
        "LABEL_VALIDITY_PATH",
        "HEALTHKIT_DAILY_PATH",
        "LABEL_TYPES_PATH",
        "ORDINAL_DICTIONARY_PATH",
        "VALIDITY_CONFIG_PATH",
    ):
        monkeypatch.delenv(var, raising=False)
    yield


class TestDataDir:
    def test_raises_when_not_configured(self):
        with pytest.raises(ValueError, match="MHC_DATA_DIR"):
            data_dir()

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

    def test_all_subpaths_under_root(self, tmp_path):
        _write_minimal_split(tmp_path)
        paths = _DatasetPaths.resolve(tmp_path, version="full")
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
        _write_minimal_split(tmp_path)
        monkeypatch.setenv("MHC_DATA_DIR", str(tmp_path))
        paths = _DatasetPaths.resolve(version="full")
        assert paths.root == tmp_path.resolve()
        assert paths.daily_hf.parent.parent == tmp_path.resolve()

    def test_explicit_override_propagates(self, tmp_path):
        root = tmp_path / "explicit"
        _write_minimal_split(root)
        paths = _DatasetPaths.resolve(root, version="full")
        assert paths.root == (tmp_path / "explicit").resolve()


class TestLabelsEnvWiring:
    """Verify the labels.api env-var bridge is set when missing."""

    def test_sets_payload_paths_when_unset(self, monkeypatch, tmp_path):
        from openmhc._evaluate import _ensure_labels_env

        _ensure_labels_env(tmp_path / "labels")
        assert os.environ["LABELS_DATA_PATH"] == str(tmp_path / "labels" / "last_labels.json")
        assert os.environ["CONTEXT_LABELS_PATH"] == str(
            tmp_path / "labels" / "context_labels.json"
        )
        assert os.environ["ENROLLMENT_DATA_PATH"] == str(
            tmp_path / "labels" / "enrollment_info.json"
        )
        assert os.environ["LABEL_VALIDITY_PATH"] == str(
            tmp_path / "labels" / "label_validity.json"
        )
        assert os.environ["HEALTHKIT_DAILY_PATH"] == str(
            tmp_path / "labels" / "healthkit_daily.json"
        )

    def test_respects_user_overrides(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LABELS_DATA_PATH", "/somewhere/else.json")
        monkeypatch.setenv("CONTEXT_LABELS_PATH", "/somewhere/ctx.json")
        monkeypatch.setenv("ENROLLMENT_DATA_PATH", "/somewhere/enrollment.json")
        from openmhc._evaluate import _ensure_labels_env

        _ensure_labels_env(tmp_path / "labels")
        assert os.environ["LABELS_DATA_PATH"] == "/somewhere/else.json"
        assert os.environ["CONTEXT_LABELS_PATH"] == "/somewhere/ctx.json"
        assert os.environ["ENROLLMENT_DATA_PATH"] == "/somewhere/enrollment.json"
        assert os.environ["LABEL_VALIDITY_PATH"] == str(
            tmp_path / "labels" / "label_validity.json"
        )


class TestLabelsApiResolution:
    def test_bundled_metadata_remains_repo_default(self):
        import labels.api as api

        api = importlib.reload(api)

        assert api.BUNDLED_METADATA_DIR == bundled_metadata_dir()
        assert api.LABEL_TYPES_PATH == bundled_metadata_dir() / "label_types.json"
        assert api.ORDINAL_DICTIONARY_PATH == bundled_metadata_dir() / "ordinal_dictionary.json"
        assert api.VALIDITY_CONFIG_PATH == bundled_metadata_dir() / "validity_config.json"
        assert api.LABELS_PATH is None
        assert len(api.TARGET_NAMES) > 0

    def test_large_payloads_resolve_from_dataset_root(self, monkeypatch, tmp_path):
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        monkeypatch.setenv("MHC_DATA_DIR", str(tmp_path))
        import labels.api as api

        api = importlib.reload(api)

        assert api.LABELS_PATH == labels_dir / "last_labels.json"
        assert api.CONTEXT_LABELS_PATH == labels_dir / "context_labels.json"
        assert api.ENROLLMENT_PATH == labels_dir / "enrollment_info.json"
        assert api.LABEL_VALIDITY_PATH == labels_dir / "label_validity.json"
        assert api.HEALTHKIT_DAILY_PATH == labels_dir / "healthkit_daily.json"
        assert api.LABEL_TYPES_PATH == bundled_metadata_dir() / "label_types.json"

    def test_per_file_env_overrides_dataset_root(self, monkeypatch, tmp_path):
        custom = tmp_path / "custom" / "labels.json"
        custom.parent.mkdir(parents=True)
        monkeypatch.setenv("MHC_DATA_DIR", str(tmp_path / "dataset"))
        monkeypatch.setenv("LABELS_DATA_PATH", str(custom))
        import labels.api as api

        api = importlib.reload(api)

        assert api.LABELS_PATH == custom.resolve()
        assert api.ENROLLMENT_PATH == (tmp_path / "dataset" / "labels" / "enrollment_info.json")

    def test_large_payload_access_raises_without_explicit_root(self):
        import labels.api as api

        api = importlib.reload(api)

        with pytest.raises(ValueError, match="data_dir=|MHC_DATA_DIR"):
            api.get_labels(
                health_code="user-123",
                timestamp=pd.Timestamp("2020-01-01"),
                label=api.TARGET_NAMES[0],
            )


class TestDownloadDatasetSurface:
    """Verify the download helper signature and error paths."""

    def test_requires_explicit_destination(self):
        with pytest.raises(ValueError, match="MHC_DATA_DIR"):
            openmhc.download_dataset(version="xs")

    def test_unknown_version_raises(self):
        from openmhc import download_dataset

        with pytest.raises(ValueError, match="version must be one of"):
            download_dataset(version="bogus")

    def test_unpublished_version_raises(self):
        from openmhc import download_dataset

        with pytest.raises(ValueError, match="not yet published"):
            download_dataset(version="full")


class TestPublicApisRequireDatasetRoot:
    def test_evaluate_prediction_fails_fast(self):
        class DummyEncoder:
            def encode(self, weekly_tensors):
                return weekly_tensors

        with pytest.raises(ValueError, match="MHC_DATA_DIR"):
            openmhc.evaluate_prediction(DummyEncoder())

    def test_evaluate_imputation_fails_fast(self):
        class DummyImputer:
            def impute(self, data, observed_mask, target_mask):
                return data

        with pytest.raises(ValueError, match="MHC_DATA_DIR"):
            openmhc.evaluate_imputation(DummyImputer())

    def test_evaluate_forecasting_fails_fast(self):
        class DummyForecaster:
            def predict(self, history, horizon):
                return history

        with pytest.raises(ValueError, match="MHC_DATA_DIR"):
            openmhc.evaluate_forecasting(DummyForecaster())
