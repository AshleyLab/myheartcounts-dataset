"""Path-resolution tests for the openmhc API.

These verify that the API and the dataset downloader agree on where data
lives. Without this, ``download_dataset`` would write to one location and
``evaluate_*`` would look in another.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import pandas as pd
import pytest

import openmhc
from openmhc._dataset import (
    DATASET_VERSION_FILENAME,
    bundled_metadata_dir,
    data_dir,
    write_dataset_marker,
)
from openmhc._evaluate import _DatasetPaths


def _write_minimal_split(root: Path, version: str = "full") -> None:
    """Build a minimal version-tagged dataset root.

    Writes both the split file (with the canonical user count for the
    requested version) and the ``dataset_version.json`` marker, so
    ``_DatasetPaths.resolve(root, version=...)`` is happy without needing
    the full payload on disk.
    """
    expected = {"full": 11894, "xs": 593}[version]
    name = {
        "full": "sharable_users_seed42_2026.json",
        "xs": "sharable_users_seed42_2026_xs.json",
    }[version]
    split_dir = root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    users = [f"u{i}" for i in range(expected)]
    payload = {"train": users, "validation": [], "test": []}
    (split_dir / name).write_text(json.dumps(payload))
    write_dataset_marker(root, version=version, n_users=expected)


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
    """Resolution order and normalization of ``data_dir``."""

    def test_raises_when_not_configured(self):
        """With no argument and no env var, ``data_dir`` raises ``ValueError``."""
        with pytest.raises(ValueError, match="MHC_DATA_DIR"):
            data_dir()

    def test_explicit_override_wins(self, tmp_path):
        """An explicit path argument is resolved and returned."""
        result = data_dir(tmp_path / "explicit")
        assert result == (tmp_path / "explicit").resolve()

    def test_env_var_override(self, monkeypatch, tmp_path):
        """When unset explicitly, ``MHC_DATA_DIR`` supplies the root."""
        monkeypatch.setenv("MHC_DATA_DIR", str(tmp_path / "from-env"))
        result = data_dir()
        assert result == (tmp_path / "from-env").resolve()

    def test_explicit_wins_over_env(self, monkeypatch, tmp_path):
        """An explicit argument takes precedence over ``MHC_DATA_DIR``."""
        monkeypatch.setenv("MHC_DATA_DIR", str(tmp_path / "from-env"))
        result = data_dir(tmp_path / "explicit")
        assert result == (tmp_path / "explicit").resolve()

    def test_tilde_expansion(self):
        """A leading ``~`` is expanded to the user's home directory."""
        result = data_dir("~/some-mhc-data")
        assert result == (Path.home() / "some-mhc-data").resolve()


class TestDatasetPaths:
    """Confirm the eval pipeline routes through ``data_dir`` consistently."""

    def test_all_subpaths_under_root(self, tmp_path):
        """Every resolved dataset subpath sits under the configured root."""
        _write_minimal_split(tmp_path, version="full")
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
        """``MHC_DATA_DIR`` becomes the resolved root for the eval paths."""
        _write_minimal_split(tmp_path, version="full")
        monkeypatch.setenv("MHC_DATA_DIR", str(tmp_path))
        paths = _DatasetPaths.resolve(version="full")
        assert paths.root == tmp_path.resolve()
        assert paths.daily_hf.parent.parent == tmp_path.resolve()

    def test_explicit_override_propagates(self, tmp_path):
        """An explicit root argument becomes the resolved ``paths.root``."""
        root = tmp_path / "explicit"
        _write_minimal_split(root, version="full")
        paths = _DatasetPaths.resolve(root, version="full")
        assert paths.root == (tmp_path / "explicit").resolve()

    def test_version_required(self, tmp_path):
        """Resolving without a ``version`` argument raises ``ValueError``."""
        _write_minimal_split(tmp_path, version="full")
        with pytest.raises(ValueError, match="version='xs' or version='full'"):
            _DatasetPaths.resolve(tmp_path)

    def test_marker_required(self, tmp_path):
        """A missing ``dataset_version.json`` marker raises ``FileNotFoundError``."""
        # split file present but no marker
        split_dir = tmp_path / "splits"
        split_dir.mkdir()
        (split_dir / "sharable_users_seed42_2026.json").write_text("{}")
        with pytest.raises(FileNotFoundError, match=DATASET_VERSION_FILENAME):
            _DatasetPaths.resolve(tmp_path, version="full")

    def test_marker_version_mismatch_rejected(self, tmp_path):
        """A marker version differing from the requested version is rejected."""
        _write_minimal_split(tmp_path, version="xs")
        with pytest.raises(ValueError, match="is version 'xs'"):
            _DatasetPaths.resolve(tmp_path, version="full")

    def test_split_user_count_mismatch_rejected(self, tmp_path):
        """A split-file user count that disagrees with the marker is rejected."""
        # Marker claims full (11894 users), but split file holds only 5.
        split_dir = tmp_path / "splits"
        split_dir.mkdir()
        (split_dir / "sharable_users_seed42_2026.json").write_text(
            '{"train": ["u0","u1","u2"], "validation": ["u3"], "test": ["u4"]}'
        )
        write_dataset_marker(tmp_path, version="full")  # n_users defaults to 11894
        with pytest.raises(ValueError, match="contains 5 users"):
            _DatasetPaths.resolve(tmp_path, version="full")


class TestLabelsEnvWiring:
    """Verify the labels.api env-var bridge is set when missing."""

    def test_sets_payload_paths_when_unset(self, monkeypatch, tmp_path):
        """When unset, label payload env vars are pointed under the labels dir."""
        from openmhc._evaluate import _ensure_labels_env

        _ensure_labels_env(tmp_path / "labels")
        assert os.environ["LABELS_DATA_PATH"] == str(tmp_path / "labels" / "last_labels.json")
        assert os.environ["CONTEXT_LABELS_PATH"] == str(tmp_path / "labels" / "context_labels.json")
        assert os.environ["ENROLLMENT_DATA_PATH"] == str(
            tmp_path / "labels" / "enrollment_info.json"
        )
        assert os.environ["LABEL_VALIDITY_PATH"] == str(tmp_path / "labels" / "label_validity.json")
        assert os.environ["HEALTHKIT_DAILY_PATH"] == str(
            tmp_path / "labels" / "healthkit_daily.json"
        )

    def test_respects_user_overrides(self, monkeypatch, tmp_path):
        """Pre-set label env vars are left untouched; only unset ones are filled."""
        monkeypatch.setenv("LABELS_DATA_PATH", "/somewhere/else.json")
        monkeypatch.setenv("CONTEXT_LABELS_PATH", "/somewhere/ctx.json")
        monkeypatch.setenv("ENROLLMENT_DATA_PATH", "/somewhere/enrollment.json")
        from openmhc._evaluate import _ensure_labels_env

        _ensure_labels_env(tmp_path / "labels")
        assert os.environ["LABELS_DATA_PATH"] == "/somewhere/else.json"
        assert os.environ["CONTEXT_LABELS_PATH"] == "/somewhere/ctx.json"
        assert os.environ["ENROLLMENT_DATA_PATH"] == "/somewhere/enrollment.json"
        assert os.environ["LABEL_VALIDITY_PATH"] == str(tmp_path / "labels" / "label_validity.json")


class TestLabelsApiResolution:
    """How ``labels.api`` resolves bundled metadata vs. dataset-root payloads."""

    def test_bundled_metadata_remains_repo_default(self):
        """Bundled metadata paths default to the repo's bundled metadata dir."""
        import labels.api as api

        api = importlib.reload(api)

        assert api.BUNDLED_METADATA_DIR == bundled_metadata_dir()
        assert api.LABEL_TYPES_PATH == bundled_metadata_dir() / "label_types.json"
        assert api.ORDINAL_DICTIONARY_PATH == bundled_metadata_dir() / "ordinal_dictionary.json"
        assert api.VALIDITY_CONFIG_PATH == bundled_metadata_dir() / "validity_config.json"
        assert api.LABELS_PATH is None
        assert len(api.TARGET_NAMES) > 0

    def test_large_payloads_resolve_from_dataset_root(self, monkeypatch, tmp_path):
        """Large payload paths resolve under ``MHC_DATA_DIR``/labels when set."""
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
        """A per-file env var overrides the dataset root for that one payload."""
        custom = tmp_path / "custom" / "labels.json"
        custom.parent.mkdir(parents=True)
        monkeypatch.setenv("MHC_DATA_DIR", str(tmp_path / "dataset"))
        monkeypatch.setenv("LABELS_DATA_PATH", str(custom))
        import labels.api as api

        api = importlib.reload(api)

        assert api.LABELS_PATH == custom.resolve()
        assert api.ENROLLMENT_PATH == (tmp_path / "dataset" / "labels" / "enrollment_info.json")

    def test_large_payload_access_raises_without_explicit_root(self):
        """Accessing a large payload with no configured root raises ``ValueError``."""
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
        """Downloading with no destination or ``MHC_DATA_DIR`` raises ``ValueError``."""
        with pytest.raises(ValueError, match="MHC_DATA_DIR"):
            openmhc.download_dataset(version="xs")

    def test_unknown_version_raises(self):
        """An unrecognized ``version`` string raises ``ValueError``."""
        from openmhc import download_dataset

        with pytest.raises(ValueError, match="version must be one of"):
            download_dataset(version="bogus")

    def test_unpublished_version_raises(self):
        """Requesting the not-yet-published ``full`` version raises ``ValueError``."""
        from openmhc import download_dataset

        with pytest.raises(ValueError, match="not yet published"):
            download_dataset(version="full")


class TestPublicApisRequireDatasetRoot:
    """Each public ``evaluate_*`` entry point fails fast without a dataset root."""

    def test_evaluate_prediction_fails_fast(self):
        """``evaluate_prediction`` raises ``ValueError`` when no root is configured."""

        class DummyEncoder:
            def encode(self, weekly_tensors):
                return weekly_tensors

        with pytest.raises(ValueError, match="MHC_DATA_DIR"):
            openmhc.evaluate_prediction(DummyEncoder(), version="xs")

    def test_evaluate_imputation_fails_fast(self):
        """``evaluate_imputation`` raises ``ValueError`` when no root is configured."""

        class DummyImputer:
            def impute(self, data, observed_mask, target_mask):
                return data

        with pytest.raises(ValueError, match="MHC_DATA_DIR"):
            openmhc.evaluate_imputation(DummyImputer(), version="xs")

    def test_evaluate_forecasting_fails_fast(self):
        """``evaluate_forecasting`` raises ``ValueError`` when no root is configured."""

        class DummyForecaster:
            def predict(self, history, horizon):
                return history

        with pytest.raises(ValueError, match="MHC_DATA_DIR"):
            openmhc.evaluate_forecasting(DummyForecaster(), version="xs")
