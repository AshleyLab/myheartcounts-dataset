"""Tests for the context API module."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import pandas as pd
import pytest

_REAL_ORDINAL_DICT = (
    Path(__file__).resolve().parent.parent.parent / "data" / "labels" / "ordinal_dictionary.json"
)


def _load_context_api(
    tmp_path: Path,
    label_types: dict,
    context_data: dict,
    last_labels_data: dict | None = None,
):
    """Reload context.api with given fixture files."""
    lt_path = tmp_path / "label_types.json"
    lt_path.write_text(json.dumps(label_types))

    labels_path = tmp_path / "last_labels.json"
    labels_path.write_text(json.dumps(last_labels_data or {}))

    context_path = tmp_path / "context_labels.json"
    context_path.write_text(json.dumps(context_data))

    enrollment_path = tmp_path / "enrollment_info.json"
    enrollment_path.write_text(json.dumps({}))

    os.environ["LABELS_DATA_PATH"] = str(labels_path)
    os.environ["ENROLLMENT_DATA_PATH"] = str(enrollment_path)
    os.environ["LABEL_TYPES_PATH"] = str(lt_path)
    os.environ["ORDINAL_DICTIONARY_PATH"] = str(_REAL_ORDINAL_DICT)
    os.environ["LABEL_VALIDITY_PATH"] = str(tmp_path / "nope.json")
    os.environ["HEALTHKIT_DAILY_PATH"] = str(tmp_path / "nope.json")
    os.environ["CONTEXT_LABELS_PATH"] = str(context_path)

    # Reload labels.api first so STORE picks up the new env vars,
    # then reload context.api which imports from labels.api.
    import labels.api
    importlib.reload(labels.api)
    import context.api
    return importlib.reload(context.api)


def test_get_context_returns_label_result(tmp_path: Path) -> None:
    """get_context returns a LabelResult for a valid context label."""
    label_types = {
        "Diabetes": {"type": "binary", "target": True},
        "field_smoking": {"type": "binary", "target": False},
    }
    context_data = {
        "field_smoking": {
            "user-123": {"timestamps": ["2020-01-01T00:00:00"], "values": [False]},
        },
    }
    api = _load_context_api(tmp_path, label_types, context_data)

    result = api.get_context("user-123", pd.Timestamp("2020-01-01"), "field_smoking")
    assert result.value is False
    assert isinstance(result.value, bool)


def test_get_context_rejects_target_label(tmp_path: Path) -> None:
    """get_context raises ValueError when label is a target, not a context variable."""
    label_types = {
        "Diabetes": {"type": "binary", "target": True},
        "field_smoking": {"type": "binary", "target": False},
    }
    last_labels = {
        "Diabetes": {
            "user-123": {"timestamps": ["2020-01-01T00:00:00"], "values": [True]},
        },
    }
    context_data = {
        "field_smoking": {
            "user-123": {"timestamps": ["2020-01-01T00:00:00"], "values": [False]},
        },
    }
    api = _load_context_api(tmp_path, label_types, context_data, last_labels)

    with pytest.raises(ValueError, match="not a context label"):
        api.get_context("user-123", pd.Timestamp("2020-01-01"), "Diabetes")


def test_get_context_unknown_label_raises(tmp_path: Path) -> None:
    """get_context raises ValueError for an entirely unknown label."""
    label_types = {"field_smoking": {"type": "binary", "target": False}}
    context_data = {
        "field_smoking": {
            "user-123": {"timestamps": ["2020-01-01T00:00:00"], "values": [False]},
        },
    }
    api = _load_context_api(tmp_path, label_types, context_data)

    with pytest.raises(ValueError, match="Unknown label"):
        api.get_context("user-123", pd.Timestamp("2020-01-01"), "no_such_label")


def test_context_names_exported(tmp_path: Path) -> None:
    """CONTEXT_NAMES is re-exported from context module."""
    label_types = {
        "Diabetes": {"type": "binary", "target": True},
        "field_smoking": {"type": "binary", "target": False},
        "field_race": {"type": "multi_categorical", "target": False},
    }
    api = _load_context_api(tmp_path, label_types, {})
    assert set(api.CONTEXT_NAMES) == {"field_smoking", "field_race"}
