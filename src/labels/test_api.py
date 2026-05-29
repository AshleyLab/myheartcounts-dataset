"""Tests for the labels API module."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from time import perf_counter

import pandas as pd
import pytest


def _write_labels(path: Path) -> None:
    sample = {
        "sleep_diagnosis1": {
            "user-123": {
                "timestamps": ["2020-01-01T00:00:00", "2020-01-01T00:10:00"],
                "values": ["early", "late"],
            },
            "e0f99e16-cc25-44c4-8b9e-83efd6d0f923": {
                "timestamps": ["2015-03-09T12:47:16"],
                "values": [True],
            },
        },
        "age": {
            "user-123": {
                "timestamps": ["2020-01-01T00:00:00"],
                "values": [19],
            },
            "e0f99e16-cc25-44c4-8b9e-83efd6d0f923": {
                "timestamps": ["2015-03-09T12:47:16"],
                "values": [24],
            },
        },
    }
    path.write_text(json.dumps(sample))


def _write_enrollment(path: Path) -> None:
    sample = {
        "user-123": {
            "birthdate": "2000-01-15",
        },
        "e0f99e16-cc25-44c4-8b9e-83efd6d0f923": {
            "birthdate": "1990-05-01",
        },
    }
    path.write_text(json.dumps(sample))


_REAL_LABEL_TYPES = Path(__file__).resolve().parent.parent.parent / "data" / "labels" / "label_types.json"
_REAL_ORDINAL_DICT = Path(__file__).resolve().parent.parent.parent / "data" / "labels" / "ordinal_dictionary.json"


def _load_api(tmp_path: Path):
    os.environ["LABELS_DATA_PATH"] = str(tmp_path / "last_labels.json")
    os.environ["ENROLLMENT_DATA_PATH"] = str(tmp_path / "enrollment_info.json")
    os.environ["LABEL_TYPES_PATH"] = str(_REAL_LABEL_TYPES)
    os.environ["ORDINAL_DICTIONARY_PATH"] = str(_REAL_ORDINAL_DICT)
    os.environ["LABEL_VALIDITY_PATH"] = str(tmp_path / "nonexistent_validity.json")
    os.environ["HEALTHKIT_DAILY_PATH"] = str(tmp_path / "nonexistent_daily.json")
    os.environ["CONTEXT_LABELS_PATH"] = str(tmp_path / "nonexistent_context.json")
    import labels.api as api

    return importlib.reload(api)


def _load_api_with_paths(labels_path: Path, enrollment_path: Path):
    os.environ["LABELS_DATA_PATH"] = str(labels_path)
    os.environ["ENROLLMENT_DATA_PATH"] = str(enrollment_path)
    os.environ["LABEL_TYPES_PATH"] = str(_REAL_LABEL_TYPES)
    os.environ["ORDINAL_DICTIONARY_PATH"] = str(_REAL_ORDINAL_DICT)
    os.environ["LABEL_VALIDITY_PATH"] = str(labels_path.parent / "nonexistent_validity.json")
    os.environ["HEALTHKIT_DAILY_PATH"] = str(labels_path.parent / "nonexistent_daily.json")
    os.environ["CONTEXT_LABELS_PATH"] = str(labels_path.parent / "nonexistent_context.json")
    import labels.api as api

    return importlib.reload(api)


def test_nearest_prefers_earlier_on_tie(tmp_path: Path) -> None:
    """Test that nearest prefers earlier labels on tie."""
    labels_path = tmp_path / "last_labels.json"
    enrollment_path = tmp_path / "enrollment_info.json"
    _write_labels(labels_path)
    _write_enrollment(enrollment_path)
    api = _load_api(tmp_path)

    # Use enforce_type=False since this test uses mock string values
    # to test the nearest-time matching logic, not type enforcement
    result = api.get_labels(
        health_code="user-123",
        timestamp=pd.Timestamp("2020-01-01T00:05:00"),
        label="sleep_diagnosis1",
        enforce_type=False,
    )

    assert result.value == "early"
    assert result.matched_timestamp == pd.Timestamp("2020-01-01T00:00:00")


def test_unknown_health_code_raises(tmp_path: Path) -> None:
    """Test that unknown health code raises an error."""
    labels_path = tmp_path / "last_labels.json"
    enrollment_path = tmp_path / "enrollment_info.json"
    _write_labels(labels_path)
    _write_enrollment(enrollment_path)
    api = _load_api(tmp_path)

    with pytest.raises(KeyError):
        api.get_labels(
            health_code="missing",
            timestamp=pd.Timestamp("2020-01-01T00:05:00"),
            label="sleep_diagnosis1",
        )


def test_age_returns_static_value(tmp_path: Path) -> None:
    """Test that age is returned as a static label from last_labels.json."""
    labels_path = tmp_path / "last_labels.json"
    enrollment_path = tmp_path / "enrollment_info.json"
    _write_labels(labels_path)
    _write_enrollment(enrollment_path)
    api = _load_api(tmp_path)

    result = api.get_labels(
        health_code="user-123",
        timestamp=pd.Timestamp("2020-01-14"),
        label="age",
    )

    # Age is stored as int in JSON, enforced as continuous (float)
    assert result.value == 19.0
    assert isinstance(result.value, float)
    # matched_timestamp is the stored label timestamp, not the query timestamp
    assert result.matched_timestamp == pd.Timestamp("2020-01-01")

    # Same value regardless of query timestamp
    result2 = api.get_labels(
        health_code="user-123",
        timestamp=pd.Timestamp("2025-06-01"),
        label="age",
    )
    assert result2.value == 19.0


def test_store_returns_json_label_value(tmp_path: Path) -> None:
    """Test that store returns JSON label value."""
    labels_path = tmp_path / "last_labels.json"
    enrollment_path = tmp_path / "enrollment_info.json"
    _write_labels(labels_path)
    _write_enrollment(enrollment_path)
    api = _load_api(tmp_path)

    result = api.get_labels(
        health_code="e0f99e16-cc25-44c4-8b9e-83efd6d0f923",
        timestamp=pd.Timestamp("2015-03-09T12:47:16"),
        label="sleep_diagnosis1",
    )

    assert result.value is True


def test_performance_real_files(capsys) -> None:
    """Test performance with real files."""
    base = Path(__file__).resolve().parent
    labels_path = base / "last_labels.json"
    enrollment_path = base / "enrollment_info.json"

    if not labels_path.exists() or not enrollment_path.exists():
        pytest.skip("Real labels/enrollment files not available.")

    with labels_path.open("r") as handle:
        labels_data = json.load(handle)
    with enrollment_path.open("r") as handle:
        enrollment_data = json.load(handle)

    label_name, health_code, ts = _pick_sample_label(labels_data)
    age_code, age_ts = _pick_sample_enrollment(enrollment_data)

    api = _load_api_with_paths(labels_path, enrollment_path)

    # Warm the caches so JSON load/build is not timed.
    api.get_labels(health_code=health_code, timestamp=ts, label=label_name)
    api.get_labels(health_code=age_code, timestamp=age_ts, label="age")

    label_rate = _throughput(api, health_code, ts, label_name, iterations=5000)
    age_rate = _throughput(api, age_code, age_ts, "age", iterations=5000)

    # Only sanity check; surface throughput via stdout for manual inspection.
    with capsys.disabled():
        print(
            f"label throughput {label_rate:.1f} lookups/s; age throughput {age_rate:.1f} lookups/s"
        )
    assert label_rate > 0
    assert age_rate > 0


def _throughput(
    api, health_code: str, timestamp: pd.Timestamp, label: str, iterations: int
) -> float:
    start = perf_counter()
    for _ in range(iterations):
        api.get_labels(health_code=health_code, timestamp=timestamp, label=label)
    elapsed = perf_counter() - start
    return iterations / elapsed if elapsed > 0 else float("inf")


def _pick_sample_label(labels_data: dict) -> tuple[str, str, pd.Timestamp]:
    for label, per_label in labels_data.items():
        for health_code, payload in per_label.items():
            timestamps = payload.get("timestamps", [])
            if timestamps:
                return label, health_code, pd.Timestamp(timestamps[0])
    raise RuntimeError("No label entry with timestamps found in last_labels.json")


def _pick_sample_enrollment(enrollment_data: dict) -> tuple[str, pd.Timestamp]:
    for health_code, payload in enrollment_data.items():
        birthdate = payload.get("birthdate")
        created_on = payload.get("createdOn")
        if birthdate:
            ts = pd.Timestamp(created_on) if created_on else pd.Timestamp("2020-01-01")
            return health_code, ts
    raise RuntimeError("No enrollment entry with birthdate found in enrollment_info.json")


def test_print_labels_statistics_with_sample_data(tmp_path: Path, capsys) -> None:
    """Test printing labels statistics with sample data."""
    """Test that print_labels_statistics works with sample data."""
    labels_path = tmp_path / "last_labels.json"
    enrollment_path = tmp_path / "enrollment_info.json"

    # Create sample data with different types
    sample_labels = {
        "numeric_label": {
            "user-123": {
                "timestamps": ["2020-01-01T00:00:00"],
                "values": [10.5],
            },
            "user-456": {
                "timestamps": ["2020-01-02T00:00:00"],
                "values": [20.0],
            },
        },
        "string_label": {
            "user-123": {
                "timestamps": ["2020-01-01T00:00:00"],
                "values": ["option_a"],
            },
        },
        "bool_label": {
            "user-123": {
                "timestamps": ["2020-01-01T00:00:00"],
                "values": [True],
            },
            "user-456": {
                "timestamps": ["2020-01-02T00:00:00"],
                "values": [False],
            },
        },
    }

    sample_enrollment = {
        "user-123": {"birthdate": "2000-01-15"},
    }

    labels_path.write_text(json.dumps(sample_labels))
    enrollment_path.write_text(json.dumps(sample_enrollment))

    api = _load_api_with_paths(labels_path, enrollment_path)

    # Temporarily override LABEL_NAMES for this test
    original_labels = api.LABEL_NAMES[:]
    api.LABEL_NAMES[:] = ["numeric_label", "string_label", "bool_label"]

    try:
        # Capture output
        api.print_labels_statistics()

        captured = capsys.readouterr()
        output = captured.out

        # Check that the header is present
        assert "Label" in output
        assert "Type" in output
        assert "Min" in output
        assert "Max" in output
        assert "Median" in output
        assert "Unique" in output

        # Check that our test labels are present
        assert "numeric_label" in output
        assert "string_label" in output
        assert "bool_label" in output

        # Check that age is not included
        assert "age" not in output

        # Check that we have some actual statistics
        assert "15.25" in output  # median of [10.5, 20.0]
        assert "2" in output  # unique count for bool_label

    finally:
        # Restore original LABEL_NAMES
        api.LABEL_NAMES[:] = original_labels


def test_type_enforcement_binary(tmp_path: Path) -> None:
    """Test type enforcement for binary labels."""
    """Test that binary labels are converted to bool."""
    labels_path = tmp_path / "last_labels.json"
    enrollment_path = tmp_path / "enrollment_info.json"

    sample_labels = {
        "BiologicalSex": {
            "user-123": {
                "timestamps": ["2020-01-01T00:00:00"],
                "values": ["Male"],
            },
            "user-456": {
                "timestamps": ["2020-01-01T00:00:00"],
                "values": ["Female"],
            },
        },
        "Diabetes": {
            "user-123": {
                "timestamps": ["2020-01-01T00:00:00"],
                "values": [1.0],
            },
            "user-456": {
                "timestamps": ["2020-01-01T00:00:00"],
                "values": [0],
            },
        },
    }

    labels_path.write_text(json.dumps(sample_labels))
    enrollment_path.write_text(json.dumps({"user-123": {"birthdate": "2000-01-15"}}))

    api = _load_api_with_paths(labels_path, enrollment_path)

    # Test string "Male" -> True
    result = api.get_labels("user-123", pd.Timestamp("2020-01-01"), "BiologicalSex")
    assert result.value is True
    assert isinstance(result.value, bool)

    # Test string "Female" -> False
    result = api.get_labels("user-456", pd.Timestamp("2020-01-01"), "BiologicalSex")
    assert result.value is False
    assert isinstance(result.value, bool)

    # Test numeric 1.0 -> True
    result = api.get_labels("user-123", pd.Timestamp("2020-01-01"), "Diabetes")
    assert result.value is True
    assert isinstance(result.value, bool)

    # Test numeric 0 -> False
    result = api.get_labels("user-456", pd.Timestamp("2020-01-01"), "Diabetes")
    assert result.value is False
    assert isinstance(result.value, bool)


def test_type_enforcement_ordinal_and_categorical(tmp_path: Path) -> None:
    """Test type enforcement for ordinal and categorical labels."""
    """Test that ordinal and categorical labels are converted to int."""
    labels_path = tmp_path / "last_labels.json"
    enrollment_path = tmp_path / "enrollment_info.json"

    sample_labels = {
        "BMI_categories": {
            "user-123": {
                "timestamps": ["2020-01-01T00:00:00"],
                "values": ["Normal weight"],  # string - ordinal - to float
            },
        },
        "cardiovascular_disease": {
            "user-123": {
                "timestamps": ["2020-01-01T00:00:00"],
                "values": [True],  # string - categorical - to string
            },
        },
    }

    labels_path.write_text(json.dumps(sample_labels))
    enrollment_path.write_text(json.dumps({"user-123": {"birthdate": "2000-01-15"}}))

    api = _load_api_with_paths(labels_path, enrollment_path)

    # Test ordinal string -> int
    result = api.get_labels("user-123", pd.Timestamp("2020-01-01"), "BMI_categories")
    assert result.value == 1
    assert isinstance(result.value, int)

    # Test categorical bool -> int
    result = api.get_labels("user-123", pd.Timestamp("2020-01-01"), "cardiovascular_disease")
    print("test categorical", result.value)
    assert result.value == 1
    assert isinstance(result.value, int)


def test_type_enforcement_continuous(tmp_path: Path) -> None:
    """Test type enforcement for continuous labels."""
    """Test that continuous labels are converted to float."""
    labels_path = tmp_path / "last_labels.json"
    enrollment_path = tmp_path / "enrollment_info.json"

    sample_labels = {
        "WeightKilograms": {
            "user-123": {
                "timestamps": ["2020-01-01T00:00:00"],
                "values": [75],  # int
            },
        },
        "TotalCholesterol": {
            "user-123": {
                "timestamps": ["2020-01-01T00:00:00"],
                "values": [5],  # datetime string
            },
        },
    }

    labels_path.write_text(json.dumps(sample_labels))
    enrollment_path.write_text(json.dumps({"user-123": {"birthdate": "2000-01-15"}}))

    api = _load_api_with_paths(labels_path, enrollment_path)

    # Test int -> float
    result = api.get_labels("user-123", pd.Timestamp("2020-01-01"), "WeightKilograms")
    assert result.value == 75.0
    assert isinstance(result.value, float)

    # Test datetime string -> float (hour of day)
    result = api.get_labels("user-123", pd.Timestamp("2020-01-01"), "TotalCholesterol")
    assert result.value == 5.0  # 23:30 -> 23.5 hours
    assert isinstance(result.value, float)


def test_null_timestamp(tmp_path: Path) -> None:
    """Test type enforcement for continuous labels."""
    """Test that continuous labels are converted to float."""
    labels_path = tmp_path / "last_labels.json"
    enrollment_path = tmp_path / "enrollment_info.json"

    sample_labels = {
        "WeightKilograms": {
            "user-123": {
                "timestamps": [""],
                "values": [75],  # int
            },
        },
        "TotalCholesterol": {
            "user-123": {
                "timestamps": [None],
                "values": [5],  # datetime string
            },
        },
    }

    labels_path.write_text(json.dumps(sample_labels))
    enrollment_path.write_text(json.dumps({"user-123": {"birthdate": "2000-01-15"}}))

    api = _load_api_with_paths(labels_path, enrollment_path)

    # Test int -> float
    result = api.get_labels("user-123", None, "WeightKilograms")
    assert result.value == 75.0
    assert isinstance(result.value, float)

    # Test datetime string -> float (hour of day)
    result = api.get_labels("user-123", None, "TotalCholesterol")
    assert result.value == 5.0  # 23:30 -> 23.5 hours
    assert isinstance(result.value, float)


def test_type_enforcement_raises_on_none(tmp_path: Path) -> None:
    """Test that type enforcement raises on None values."""
    """Test that None values raise LabelValueError."""
    labels_path = tmp_path / "last_labels.json"
    enrollment_path = tmp_path / "enrollment_info.json"

    sample_labels = {
        "WeightKilograms": {
            "user-123": {
                "timestamps": ["2020-01-01T00:00:00"],
                "values": [None],
            },
        },
    }

    labels_path.write_text(json.dumps(sample_labels))
    enrollment_path.write_text(json.dumps({"user-123": {"birthdate": "2000-01-15"}}))

    api = _load_api_with_paths(labels_path, enrollment_path)

    with pytest.raises(api.LabelValueError):
        api.get_labels("user-123", pd.Timestamp("2020-01-01"), "WeightKilograms")


def test_type_enforcement_raises_on_nan_string(tmp_path: Path) -> None:
    """Test that type enforcement raises on NaN strings."""
    """Test that 'nan' string values raise LabelValueError."""
    labels_path = tmp_path / "last_labels.json"
    enrollment_path = tmp_path / "enrollment_info.json"

    sample_labels = {
        "WeightKilograms": {
            "user-123": {
                "timestamps": ["2020-01-01T00:00:00"],
                "values": ["nan"],
            },
        },
    }

    labels_path.write_text(json.dumps(sample_labels))
    enrollment_path.write_text(json.dumps({"user-123": {"birthdate": "2000-01-15"}}))

    api = _load_api_with_paths(labels_path, enrollment_path)

    with pytest.raises(api.LabelValueError):
        api.get_labels("user-123", pd.Timestamp("2020-01-01"), "WeightKilograms")


def test_type_enforcement_can_be_disabled(tmp_path: Path) -> None:
    """Test that type enforcement can be disabled."""
    """Test that enforce_type=False returns raw values."""
    labels_path = tmp_path / "last_labels.json"
    enrollment_path = tmp_path / "enrollment_info.json"

    sample_labels = {
        "BiologicalSex": {
            "user-123": {
                "timestamps": ["2020-01-01T00:00:00"],
                "values": ["Male"],
            },
        },
    }

    labels_path.write_text(json.dumps(sample_labels))
    enrollment_path.write_text(json.dumps({"user-123": {"birthdate": "2000-01-15"}}))

    api = _load_api_with_paths(labels_path, enrollment_path)

    result = api.get_labels(
        "user-123", pd.Timestamp("2020-01-01"), "BiologicalSex", enforce_type=False
    )
    assert result.value == "Male"
    assert isinstance(result.value, str)


def test_get_labels_statistics_with_sample_data(tmp_path: Path) -> None:
    """Test getting labels statistics with sample data."""
    """Test that get_labels_statistics returns a proper DataFrame with sample data."""
    labels_path = tmp_path / "last_labels.json"
    enrollment_path = tmp_path / "enrollment_info.json"

    # Create sample data with different types
    sample_labels = {
        "numeric_label": {
            "user-123": {
                "timestamps": ["2020-01-01T00:00:00"],
                "values": [10.5],
            },
            "user-456": {
                "timestamps": ["2020-01-02T00:00:00"],
                "values": [20.0],
            },
        },
        "string_label": {
            "user-123": {
                "timestamps": ["2020-01-01T00:00:00"],
                "values": ["option_a"],
            },
        },
        "bool_label": {
            "user-123": {
                "timestamps": ["2020-01-01T00:00:00"],
                "values": [True],
            },
            "user-456": {
                "timestamps": ["2020-01-02T00:00:00"],
                "values": [False],
            },
        },
    }

    sample_enrollment = {
        "user-123": {"birthdate": "2000-01-15"},
    }

    labels_path.write_text(json.dumps(sample_labels))
    enrollment_path.write_text(json.dumps(sample_enrollment))

    api = _load_api_with_paths(labels_path, enrollment_path)

    # Temporarily override LABEL_NAMES for this test
    original_labels = api.LABEL_NAMES[:]
    api.LABEL_NAMES[:] = ["numeric_label", "string_label", "bool_label"]

    try:
        # Get DataFrame
        df = api.get_labels_statistics()

        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.columns) == ["label", "type", "min", "max", "median", "unique"]

        # Check numeric_label row
        numeric_row = df[df["label"] == "numeric_label"].iloc[0]
        assert numeric_row["type"] == "float"
        assert numeric_row["min"] == 10.5
        assert numeric_row["max"] == 20.0
        assert numeric_row["median"] == 15.25
        assert numeric_row["unique"] == 2

        # Check string_label row
        string_row = df[df["label"] == "string_label"].iloc[0]
        assert string_row["type"] == "str"
        assert pd.isna(string_row["min"])
        assert pd.isna(string_row["max"])
        assert pd.isna(string_row["median"])
        assert string_row["unique"] == 1

        # Check bool_label row
        bool_row = df[df["label"] == "bool_label"].iloc[0]
        assert bool_row["type"] == "bool"
        assert bool_row["min"] == 0.0
        assert bool_row["max"] == 1.0
        assert bool_row["median"] == 0.5
        assert bool_row["unique"] == 2

    finally:
        # Restore original LABEL_NAMES
        api.LABEL_NAMES[:] = original_labels


# ---- Framingham risk label tests ---- #


def test_framingham_risk_in_label_names(tmp_path: Path) -> None:
    """Test that framingham_risk is discoverable in LABEL_NAMES and LABEL_TYPES."""
    labels_path = tmp_path / "last_labels.json"
    enrollment_path = tmp_path / "enrollment_info.json"
    labels_path.write_text(json.dumps({}))
    enrollment_path.write_text(json.dumps({}))
    api = _load_api_with_paths(labels_path, enrollment_path)

    assert "framingham_risk" in api.LABEL_NAMES
    assert api.LABEL_TYPES["framingham_risk"] == "continuous"


def test_framingham_risk_continuous_type_enforcement(tmp_path: Path) -> None:
    """Test that framingham_risk values are enforced as float (continuous)."""
    labels_path = tmp_path / "last_labels.json"
    enrollment_path = tmp_path / "enrollment_info.json"

    sample_labels = {
        "framingham_risk": {
            "user-123": {
                "timestamps": ["2020-06-01T00:00:00"],
                "values": [0.0735],
            },
            "user-456": {
                "timestamps": ["2019-12-15T00:00:00"],
                "values": [0],  # int zero — should become 0.0 float
            },
        },
    }

    labels_path.write_text(json.dumps(sample_labels))
    enrollment_path.write_text(json.dumps({"user-123": {"birthdate": "1980-01-01"}}))

    api = _load_api_with_paths(labels_path, enrollment_path)

    result = api.get_labels("user-123", pd.Timestamp("2020-06-01"), "framingham_risk")
    assert result.value == 0.0735
    assert isinstance(result.value, float)
    assert result.matched_timestamp == pd.Timestamp("2020-06-01")

    result2 = api.get_labels("user-456", pd.Timestamp("2020-01-01"), "framingham_risk")
    assert result2.value == 0.0
    assert isinstance(result2.value, float)


def test_framingham_risk_nearest_timestamp(tmp_path: Path) -> None:
    """Test nearest-timestamp matching for framingham_risk (multi-survey user)."""
    labels_path = tmp_path / "last_labels.json"
    enrollment_path = tmp_path / "enrollment_info.json"

    sample_labels = {
        "framingham_risk": {
            "user-123": {
                "timestamps": [
                    "2018-06-01T00:00:00",
                    "2020-01-01T00:00:00",
                ],
                "values": [0.05, 0.12],
            },
        },
    }

    labels_path.write_text(json.dumps(sample_labels))
    enrollment_path.write_text(json.dumps({"user-123": {"birthdate": "1970-01-01"}}))

    api = _load_api_with_paths(labels_path, enrollment_path)

    # Query closer to first timestamp
    result = api.get_labels("user-123", pd.Timestamp("2018-07-01"), "framingham_risk")
    assert result.value == 0.05

    # Query closer to second timestamp
    result = api.get_labels("user-123", pd.Timestamp("2019-11-01"), "framingham_risk")
    assert result.value == 0.12


def test_framingham_risk_missing_user_raises(tmp_path: Path) -> None:
    """Test that querying framingham_risk for a missing user raises KeyError."""
    labels_path = tmp_path / "last_labels.json"
    enrollment_path = tmp_path / "enrollment_info.json"

    sample_labels = {
        "framingham_risk": {
            "user-123": {
                "timestamps": ["2020-01-01T00:00:00"],
                "values": [0.08],
            },
        },
    }

    labels_path.write_text(json.dumps(sample_labels))
    enrollment_path.write_text(json.dumps({}))

    api = _load_api_with_paths(labels_path, enrollment_path)

    with pytest.raises(KeyError):
        api.get_labels("nonexistent-user", pd.Timestamp("2020-01-01"), "framingham_risk")


# ---- Label validity tests ---- #


def _write_validity(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data))


def _load_api_with_validity(labels_path: Path, enrollment_path: Path, validity_path: Path):
    os.environ["LABELS_DATA_PATH"] = str(labels_path)
    os.environ["ENROLLMENT_DATA_PATH"] = str(enrollment_path)
    os.environ["LABEL_TYPES_PATH"] = str(_REAL_LABEL_TYPES)
    os.environ["ORDINAL_DICTIONARY_PATH"] = str(_REAL_ORDINAL_DICT)
    os.environ["LABEL_VALIDITY_PATH"] = str(validity_path)
    os.environ["HEALTHKIT_DAILY_PATH"] = str(labels_path.parent / "nonexistent_daily.json")
    os.environ["CONTEXT_LABELS_PATH"] = str(labels_path.parent / "nonexistent_context.json")
    import labels.api as api

    return importlib.reload(api)


def _make_validity_fixtures(tmp_path: Path):
    """Create labels + enrollment + validity files for validity tests."""
    labels = {
        "Diabetes": {
            "user-valid": {
                "timestamps": ["2020-06-01T00:00:00"],
                "values": [True],
            },
            "user-invalid": {
                "timestamps": ["2020-06-01T00:00:00"],
                "values": [True],
            },
        },
        "happiness": {
            "user-mixed": {
                "timestamps": [
                    "2020-01-15T00:00:00",
                    "2020-03-20T00:00:00",
                    "2020-06-15T00:00:00",
                    "2020-09-10T00:00:00",
                ],
                "values": [7.0, 8.0, 6.0, 9.0],
            },
        },
        "age": {
            "user-valid": {
                "timestamps": ["2020-06-01T00:00:00"],
                "values": [40],
            },
            "user-invalid": {
                "timestamps": ["2020-06-01T00:00:00"],
                "values": [40],
            },
            "user-mixed": {
                "timestamps": ["2020-06-01T00:00:00"],
                "values": [40],
            },
        },
    }
    enrollment = {
        "user-valid": {"birthdate": "1980-01-01"},
        "user-invalid": {"birthdate": "1980-01-01"},
        "user-mixed": {"birthdate": "1980-01-01"},
    }
    validity = {
        "Diabetes": {
            "user-valid": [True],
            "user-invalid": [False],
        },
        "happiness": {
            "user-mixed": [False, True, False, True],
        },
    }

    labels_path = tmp_path / "last_labels.json"
    enrollment_path = tmp_path / "enrollment_info.json"
    validity_path = tmp_path / "label_validity.json"

    labels_path.write_text(json.dumps(labels))
    enrollment_path.write_text(json.dumps(enrollment))
    _write_validity(validity_path, validity)

    return labels_path, enrollment_path, validity_path


def test_validity_returns_valid_measurement(tmp_path: Path) -> None:
    """Valid single-measurement label is returned normally."""
    labels_path, enrollment_path, validity_path = _make_validity_fixtures(tmp_path)
    api = _load_api_with_validity(labels_path, enrollment_path, validity_path)

    result = api.get_labels(
        "user-valid", pd.Timestamp("2020-06-01"), "Diabetes", return_valid_only=True
    )
    assert result.value is True


def test_validity_raises_for_invalid_measurement(tmp_path: Path) -> None:
    """Single-measurement label with valid=False raises KeyError."""
    labels_path, enrollment_path, validity_path = _make_validity_fixtures(tmp_path)
    api = _load_api_with_validity(labels_path, enrollment_path, validity_path)

    with pytest.raises(KeyError):
        api.get_labels(
            "user-invalid", pd.Timestamp("2020-06-01"), "Diabetes", return_valid_only=True
        )


def test_validity_false_ignores_validity(tmp_path: Path) -> None:
    """return_valid_only=False returns the measurement regardless of validity."""
    labels_path, enrollment_path, validity_path = _make_validity_fixtures(tmp_path)
    api = _load_api_with_validity(labels_path, enrollment_path, validity_path)

    result = api.get_labels(
        "user-invalid", pd.Timestamp("2020-06-01"), "Diabetes", return_valid_only=False
    )
    assert result.value is True


def test_validity_multi_measurement_returns_nearest_valid(tmp_path: Path) -> None:
    """For happiness with mixed validity, returns nearest VALID measurement."""
    labels_path, enrollment_path, validity_path = _make_validity_fixtures(tmp_path)
    api = _load_api_with_validity(labels_path, enrollment_path, validity_path)

    # Query closest to Jan 15 (invalid). Nearest valid is Mar 20.
    result = api.get_labels(
        "user-mixed", pd.Timestamp("2020-01-20"), "happiness",
        enforce_type=False, return_valid_only=True,
    )
    assert result.value == 8.0
    assert result.matched_timestamp == pd.Timestamp("2020-03-20")


def test_validity_multi_measurement_picks_closest_valid(tmp_path: Path) -> None:
    """Query between two valid measurements picks the closer one."""
    labels_path, enrollment_path, validity_path = _make_validity_fixtures(tmp_path)
    api = _load_api_with_validity(labels_path, enrollment_path, validity_path)

    # Valid measurements: Mar 20 (value=8) and Sep 10 (value=9).
    # Query at Aug 1 is closer to Sep 10.
    result = api.get_labels(
        "user-mixed", pd.Timestamp("2020-08-01"), "happiness",
        enforce_type=False, return_valid_only=True,
    )
    assert result.value == 9.0
    assert result.matched_timestamp == pd.Timestamp("2020-09-10")


def test_validity_no_file_degrades_gracefully(tmp_path: Path) -> None:
    """When label_validity.json does not exist, behaves like return_valid_only=False."""
    labels_path = tmp_path / "last_labels.json"
    enrollment_path = tmp_path / "enrollment_info.json"
    nonexistent_validity = tmp_path / "nonexistent.json"

    labels_path.write_text(json.dumps({
        "Diabetes": {
            "user-123": {"timestamps": ["2020-06-01T00:00:00"], "values": [True]},
        },
    }))
    enrollment_path.write_text(json.dumps({"user-123": {"birthdate": "1980-01-01"}}))

    api = _load_api_with_validity(labels_path, enrollment_path, nonexistent_validity)

    # Should work fine — no validity data means no filtering
    result = api.get_labels("user-123", pd.Timestamp("2020-06-01"), "Diabetes")
    assert result.value is True


def test_validity_age_bypasses_check(tmp_path: Path) -> None:
    """Age is a static label (validity_config has null) so validity is not applied."""
    labels_path, enrollment_path, validity_path = _make_validity_fixtures(tmp_path)
    api = _load_api_with_validity(labels_path, enrollment_path, validity_path)

    result = api.get_labels("user-valid", pd.Timestamp("2020-06-01"), "age")
    assert result.value == 40.0


def test_validity_default_is_true(tmp_path: Path) -> None:
    """get_labels() defaults to return_valid_only=True."""
    labels_path, enrollment_path, validity_path = _make_validity_fixtures(tmp_path)
    api = _load_api_with_validity(labels_path, enrollment_path, validity_path)

    # user-invalid has Diabetes valid=[False]. Default should reject.
    with pytest.raises(KeyError):
        api.get_labels("user-invalid", pd.Timestamp("2020-06-01"), "Diabetes")


def test_validity_mask_length_mismatch_ignored(tmp_path: Path) -> None:
    """If validity mask length doesn't match timestamps, it's ignored (treated as no data)."""
    labels_path = tmp_path / "last_labels.json"
    enrollment_path = tmp_path / "enrollment_info.json"
    validity_path = tmp_path / "label_validity.json"

    labels_path.write_text(json.dumps({
        "Diabetes": {
            "user-123": {"timestamps": ["2020-06-01T00:00:00"], "values": [True]},
        },
    }))
    enrollment_path.write_text(json.dumps({"user-123": {"birthdate": "1980-01-01"}}))
    # Wrong length mask: 2 booleans for 1 timestamp
    _write_validity(validity_path, {"Diabetes": {"user-123": [True, False]}})

    api = _load_api_with_validity(labels_path, enrollment_path, validity_path)

    # Mask length mismatch → not attached → series.valid stays None → nearest() used
    result = api.get_labels("user-123", pd.Timestamp("2020-06-01"), "Diabetes")
    assert result.value is True


# ---- Windowed label query tests ---- #


def _load_api_with_daily(
    labels_path: Path, enrollment_path: Path, daily_path: Path,
):
    os.environ["LABELS_DATA_PATH"] = str(labels_path)
    os.environ["ENROLLMENT_DATA_PATH"] = str(enrollment_path)
    os.environ["LABEL_TYPES_PATH"] = str(_REAL_LABEL_TYPES)
    os.environ["ORDINAL_DICTIONARY_PATH"] = str(_REAL_ORDINAL_DICT)
    os.environ["LABEL_VALIDITY_PATH"] = str(labels_path.parent / "nonexistent.json")
    os.environ["HEALTHKIT_DAILY_PATH"] = str(daily_path)
    os.environ["CONTEXT_LABELS_PATH"] = str(labels_path.parent / "nonexistent_context.json")
    import labels.api as api

    return importlib.reload(api)


def _make_windowed_fixtures(tmp_path: Path):
    """Create fixtures with daily-resolution happiness + Watch data."""
    labels = {
        "happiness": {
            "user-daily": {
                "timestamps": [
                    "2020-03-01T00:00:00",
                    "2020-03-02T00:00:00",
                    "2020-03-03T00:00:00",
                    "2020-03-04T00:00:00",
                    "2020-03-05T00:00:00",
                ],
                "values": [6.0, 8.0, 7.0, 9.0, 5.0],
            },
        },
    }
    enrollment = {"user-daily": {"birthdate": "1980-01-01"}}

    # Daily HealthKit data
    daily = {
        "Watch_RestingHeartRate": {
            "user-daily": {
                "timestamps": [
                    "2020-03-01T00:00:00",
                    "2020-03-02T00:00:00",
                    "2020-03-03T00:00:00",
                    "2020-03-04T00:00:00",
                    "2020-03-05T00:00:00",
                    "2020-03-06T00:00:00",
                    "2020-03-07T00:00:00",
                ],
                "values": [60.0, 62.0, 58.0, 61.0, 63.0, 59.0, 64.0],
            },
        },
        "happiness": {
            "user-daily": {
                "timestamps": [
                    "2020-03-01T00:00:00",
                    "2020-03-02T00:00:00",
                    "2020-03-03T00:00:00",
                    "2020-03-04T00:00:00",
                    "2020-03-05T00:00:00",
                ],
                "values": [6.0, 8.0, 7.0, 9.0, 5.0],
            },
        },
    }

    labels_path = tmp_path / "last_labels.json"
    enrollment_path = tmp_path / "enrollment_info.json"
    daily_path = tmp_path / "healthkit_daily.json"

    labels_path.write_text(json.dumps(labels))
    enrollment_path.write_text(json.dumps(enrollment))
    daily_path.write_text(json.dumps(daily))

    return labels_path, enrollment_path, daily_path


def test_windowed_median(tmp_path: Path) -> None:
    """Windowed query returns median of values in window."""
    lp, ep, dp = _make_windowed_fixtures(tmp_path)
    api = _load_api_with_daily(lp, ep, dp)

    # ±2 days around March 3 → covers March 1-5 (5 values: 60,62,58,61,63)
    result = api.get_labels_windowed(
        "user-daily", pd.Timestamp("2020-03-03"), "Watch_RestingHeartRate",
        window_days=2, aggregation="median",
    )
    assert result.value == 61.0
    assert result.n_points == 5


def test_windowed_mean(tmp_path: Path) -> None:
    """Windowed query with mean aggregation."""
    lp, ep, dp = _make_windowed_fixtures(tmp_path)
    api = _load_api_with_daily(lp, ep, dp)

    # ±1 day around March 3 → covers March 2-4 (3 values: 62,58,61)
    result = api.get_labels_windowed(
        "user-daily", pd.Timestamp("2020-03-03"), "Watch_RestingHeartRate",
        window_days=1, aggregation="mean",
    )
    assert abs(result.value - 60.333333) < 0.01
    assert result.n_points == 3


def test_windowed_exact_day(tmp_path: Path) -> None:
    """window_days=0 returns only the exact calendar day."""
    lp, ep, dp = _make_windowed_fixtures(tmp_path)
    api = _load_api_with_daily(lp, ep, dp)

    result = api.get_labels_windowed(
        "user-daily", pd.Timestamp("2020-03-03"), "Watch_RestingHeartRate",
        window_days=0, aggregation="mean",
    )
    assert result.value == 58.0
    assert result.n_points == 1


def test_windowed_no_data_raises(tmp_path: Path) -> None:
    """Windowed query raises KeyError when no data in window."""
    lp, ep, dp = _make_windowed_fixtures(tmp_path)
    api = _load_api_with_daily(lp, ep, dp)

    with pytest.raises(KeyError):
        api.get_labels_windowed(
            "user-daily", pd.Timestamp("2021-01-01"), "Watch_RestingHeartRate",
            window_days=0,
        )


def test_windowed_happiness_from_daily(tmp_path: Path) -> None:
    """Happiness windowed query uses daily index when available."""
    lp, ep, dp = _make_windowed_fixtures(tmp_path)
    api = _load_api_with_daily(lp, ep, dp)

    result = api.get_labels_windowed(
        "user-daily", pd.Timestamp("2020-03-03"), "happiness",
        window_days=1, aggregation="median",
    )
    # March 2-4: values 8.0, 7.0, 9.0 → median = 8.0
    assert result.value == 8.0
    assert result.n_points == 3


def test_windowed_falls_back_to_labels(tmp_path: Path) -> None:
    """When daily index doesn't exist, windowed falls back to last_labels."""
    labels_path = tmp_path / "last_labels.json"
    enrollment_path = tmp_path / "enrollment_info.json"
    nonexistent_daily = tmp_path / "nonexistent_daily.json"

    labels_path.write_text(json.dumps({
        "happiness": {
            "user-123": {
                "timestamps": [
                    "2020-03-01T00:00:00",
                    "2020-03-02T00:00:00",
                    "2020-03-03T00:00:00",
                ],
                "values": [5.0, 7.0, 9.0],
            },
        },
    }))
    enrollment_path.write_text(json.dumps({"user-123": {"birthdate": "1980-01-01"}}))

    api = _load_api_with_daily(labels_path, enrollment_path, nonexistent_daily)

    result = api.get_labels_windowed(
        "user-123", pd.Timestamp("2020-03-02"), "happiness",
        window_days=1, aggregation="mean",
    )
    assert result.value == 7.0
    assert result.n_points == 3


def test_windowed_invalid_aggregation(tmp_path: Path) -> None:
    """Invalid aggregation method raises ValueError."""
    lp, ep, dp = _make_windowed_fixtures(tmp_path)
    api = _load_api_with_daily(lp, ep, dp)

    with pytest.raises(ValueError, match="Unknown aggregation"):
        api.get_labels_windowed(
            "user-daily", pd.Timestamp("2020-03-03"), "Watch_RestingHeartRate",
            window_days=1, aggregation="invalid",
        )


# ---- Context API tests ---- #


def _load_api_with_context(
    labels_path: Path,
    enrollment_path: Path,
    context_path: Path,
    label_types_path: Path | None = None,
):
    os.environ["LABELS_DATA_PATH"] = str(labels_path)
    os.environ["ENROLLMENT_DATA_PATH"] = str(enrollment_path)
    os.environ["LABEL_TYPES_PATH"] = str(label_types_path or _REAL_LABEL_TYPES)
    os.environ["ORDINAL_DICTIONARY_PATH"] = str(_REAL_ORDINAL_DICT)
    os.environ["LABEL_VALIDITY_PATH"] = str(labels_path.parent / "nonexistent.json")
    os.environ["HEALTHKIT_DAILY_PATH"] = str(labels_path.parent / "nonexistent.json")
    os.environ["CONTEXT_LABELS_PATH"] = str(context_path)
    import labels.api as api

    return importlib.reload(api)


def test_target_names_and_context_names_partition(tmp_path: Path) -> None:
    """TARGET_NAMES + CONTEXT_NAMES = LABEL_NAMES, no overlap."""
    label_types = {
        "Diabetes": {"type": "binary", "target": True},
        "age": {"type": "continuous", "target": True},
        "field_smoking": {"type": "binary", "target": False},
    }
    lt_path = tmp_path / "label_types.json"
    lt_path.write_text(json.dumps(label_types))

    labels_path = tmp_path / "last_labels.json"
    enrollment_path = tmp_path / "enrollment_info.json"
    labels_path.write_text(json.dumps({}))
    enrollment_path.write_text(json.dumps({}))

    os.environ["LABELS_DATA_PATH"] = str(labels_path)
    os.environ["ENROLLMENT_DATA_PATH"] = str(enrollment_path)
    os.environ["LABEL_TYPES_PATH"] = str(lt_path)
    os.environ["LABEL_VALIDITY_PATH"] = str(tmp_path / "nope.json")
    os.environ["HEALTHKIT_DAILY_PATH"] = str(tmp_path / "nope.json")
    os.environ["CONTEXT_LABELS_PATH"] = str(tmp_path / "nope.json")
    import labels.api as api
    api = importlib.reload(api)

    assert set(api.TARGET_NAMES) | set(api.CONTEXT_NAMES) == set(api.LABEL_NAMES)
    assert set(api.TARGET_NAMES) & set(api.CONTEXT_NAMES) == set()
    assert api.TARGET_NAMES == ["Diabetes", "age"]
    assert api.CONTEXT_NAMES == ["field_smoking"]


def test_context_labels_loaded_and_queryable(tmp_path: Path) -> None:
    """Context labels from context_labels.json are queryable via get_labels()."""
    label_types = {
        "Diabetes": {"type": "binary", "target": True},
        "field_smoking": {"type": "binary", "target": False},
    }
    lt_path = tmp_path / "label_types.json"
    lt_path.write_text(json.dumps(label_types))

    labels_path = tmp_path / "last_labels.json"
    labels_path.write_text(json.dumps({
        "Diabetes": {
            "user-123": {"timestamps": ["2020-01-01T00:00:00"], "values": [True]},
        },
    }))
    context_path = tmp_path / "context_labels.json"
    context_path.write_text(json.dumps({
        "field_smoking": {
            "user-123": {"timestamps": ["2020-01-01T00:00:00"], "values": [False]},
        },
    }))
    enrollment_path = tmp_path / "enrollment_info.json"
    enrollment_path.write_text(json.dumps({"user-123": {"birthdate": "1980-01-01"}}))

    api = _load_api_with_context(labels_path, enrollment_path, context_path, lt_path)

    # Target label works
    result = api.get_labels("user-123", pd.Timestamp("2020-01-01"), "Diabetes")
    assert result.value is True

    # Context label works identically
    result = api.get_labels("user-123", pd.Timestamp("2020-01-01"), "field_smoking")
    assert result.value is False
    assert isinstance(result.value, bool)


def test_context_overlap_target_takes_precedence(tmp_path: Path) -> None:
    """For overlapping labels, last_labels.json (targets) takes precedence."""
    label_types = {
        "Diabetes": {"type": "binary", "target": True},
    }
    lt_path = tmp_path / "label_types.json"
    lt_path.write_text(json.dumps(label_types))

    # Target says True at 2020-01-01
    labels_path = tmp_path / "last_labels.json"
    labels_path.write_text(json.dumps({
        "Diabetes": {
            "user-123": {"timestamps": ["2020-01-01T00:00:00"], "values": [True]},
        },
    }))
    # Context says False at 2020-01-01 (different survey)
    context_path = tmp_path / "context_labels.json"
    context_path.write_text(json.dumps({
        "Diabetes": {
            "user-123": {"timestamps": ["2020-01-01T00:00:00"], "values": [False]},
        },
    }))
    enrollment_path = tmp_path / "enrollment_info.json"
    enrollment_path.write_text(json.dumps({"user-123": {"birthdate": "1980-01-01"}}))

    api = _load_api_with_context(labels_path, enrollment_path, context_path, lt_path)

    # Should get target value (True), not context value (False)
    result = api.get_labels("user-123", pd.Timestamp("2020-01-01"), "Diabetes")
    assert result.value is True


def test_context_overlap_merges_new_users(tmp_path: Path) -> None:
    """Context file adds users that don't exist in target file for same label."""
    label_types = {
        "Diabetes": {"type": "binary", "target": True},
    }
    lt_path = tmp_path / "label_types.json"
    lt_path.write_text(json.dumps(label_types))

    labels_path = tmp_path / "last_labels.json"
    labels_path.write_text(json.dumps({
        "Diabetes": {
            "user-123": {"timestamps": ["2020-01-01T00:00:00"], "values": [True]},
        },
    }))
    context_path = tmp_path / "context_labels.json"
    context_path.write_text(json.dumps({
        "Diabetes": {
            "user-456": {"timestamps": ["2020-06-01T00:00:00"], "values": [False]},
        },
    }))
    enrollment_path = tmp_path / "enrollment_info.json"
    enrollment_path.write_text(json.dumps({
        "user-123": {"birthdate": "1980-01-01"},
        "user-456": {"birthdate": "1990-01-01"},
    }))

    api = _load_api_with_context(labels_path, enrollment_path, context_path, lt_path)

    # user-123 from targets
    r1 = api.get_labels("user-123", pd.Timestamp("2020-01-01"), "Diabetes")
    assert r1.value is True
    # user-456 from context (merged in)
    r2 = api.get_labels("user-456", pd.Timestamp("2020-06-01"), "Diabetes")
    assert r2.value is False


def test_no_context_file_works(tmp_path: Path) -> None:
    """API works fine when context_labels.json does not exist."""
    labels_path = tmp_path / "last_labels.json"
    enrollment_path = tmp_path / "enrollment_info.json"
    nonexistent_ctx = tmp_path / "nonexistent_context.json"

    labels_path.write_text(json.dumps({
        "Diabetes": {
            "user-123": {"timestamps": ["2020-06-01T00:00:00"], "values": [True]},
        },
    }))
    enrollment_path.write_text(json.dumps({"user-123": {"birthdate": "1980-01-01"}}))

    api = _load_api_with_context(labels_path, enrollment_path, nonexistent_ctx)

    result = api.get_labels("user-123", pd.Timestamp("2020-06-01"), "Diabetes")
    assert result.value is True


def test_backward_compat_old_label_types_format(tmp_path: Path) -> None:
    """Old flat label_types.json format still loads correctly."""
    old_format = {"Diabetes": "binary", "age": "continuous"}
    lt_path = tmp_path / "label_types.json"
    lt_path.write_text(json.dumps(old_format))

    labels_path = tmp_path / "last_labels.json"
    labels_path.write_text(json.dumps({}))
    enrollment_path = tmp_path / "enrollment_info.json"
    enrollment_path.write_text(json.dumps({}))

    os.environ["LABELS_DATA_PATH"] = str(labels_path)
    os.environ["ENROLLMENT_DATA_PATH"] = str(enrollment_path)
    os.environ["LABEL_TYPES_PATH"] = str(lt_path)
    os.environ["LABEL_VALIDITY_PATH"] = str(tmp_path / "nope.json")
    os.environ["HEALTHKIT_DAILY_PATH"] = str(tmp_path / "nope.json")
    os.environ["CONTEXT_LABELS_PATH"] = str(tmp_path / "nope.json")
    import labels.api as api
    api = importlib.reload(api)

    assert api.LABEL_TYPES == {"Diabetes": "binary", "age": "continuous"}
    assert api.TARGET_NAMES == ["Diabetes", "age"]
    assert api.CONTEXT_NAMES == []


# ---- Defensive KeyError wrap (safeguard S6) ---- #


def test_unknown_ordinal_string_raises_label_type_error(tmp_path: Path) -> None:
    """Unknown string in an ordinal/categorical column wraps KeyError as LabelTypeError."""
    labels_path = tmp_path / "last_labels.json"
    enrollment_path = tmp_path / "enrollment_info.json"
    labels_path.write_text(json.dumps({
        "BMI_categories": {
            "user-123": {
                "timestamps": ["2020-01-01T00:00:00"],
                "values": ["__not_a_known_string__"],
            },
        },
    }))
    enrollment_path.write_text(json.dumps({"user-123": {"birthdate": "2000-01-15"}}))

    api = _load_api_with_paths(labels_path, enrollment_path)

    with pytest.raises(api.LabelTypeError):
        api.get_labels("user-123", pd.Timestamp("2020-01-01"), "BMI_categories")


# ---- _to_tuple_of_int converter (T1-T7 from scoping doc) ---- #


def test_to_tuple_of_int_basic(tmp_path: Path) -> None:
    """T1: basic list of ints sorts ascending."""
    api = _load_api(tmp_path)
    assert api._to_tuple_of_int([1, 3], "field_race") == (1, 3)


def test_to_tuple_of_int_sorts_unsorted(tmp_path: Path) -> None:
    """T2: unsorted input is sorted ascending so {3,1} == {1,3}."""
    api = _load_api(tmp_path)
    assert api._to_tuple_of_int([3, 1], "field_race") == (1, 3)


def test_to_tuple_of_int_floats_with_integer_value(tmp_path: Path) -> None:
    """T3: float-valued ints are coerced (source parquet is list<double>)."""
    api = _load_api(tmp_path)
    assert api._to_tuple_of_int([1.0, 3.0], "field_race") == (1, 3)


def test_to_tuple_of_int_rejects_non_integer_float(tmp_path: Path) -> None:
    """T4: non-integer float raises LabelTypeError."""
    api = _load_api(tmp_path)
    with pytest.raises(api.LabelTypeError):
        api._to_tuple_of_int([1.5], "field_race")


def test_to_tuple_of_int_rejects_empty(tmp_path: Path) -> None:
    """T5: empty list raises LabelTypeError (storage layer should have emitted None)."""
    api = _load_api(tmp_path)
    with pytest.raises(api.LabelTypeError):
        api._to_tuple_of_int([], "field_race")


def test_to_tuple_of_int_rejects_scalar(tmp_path: Path) -> None:
    """T6: non-list raises LabelTypeError."""
    api = _load_api(tmp_path)
    with pytest.raises(api.LabelTypeError):
        api._to_tuple_of_int(5, "field_race")


def test_to_tuple_of_int_rejects_bool_element(tmp_path: Path) -> None:
    """T7: bool element rejected (since isinstance(True, int) is True in Python)."""
    api = _load_api(tmp_path)
    with pytest.raises(api.LabelTypeError):
        api._to_tuple_of_int([True, 1], "field_race")


def test_enforce_type_multi_categorical_dispatches_to_tuple(tmp_path: Path) -> None:
    """T8: _enforce_type routes multi_categorical labels through _to_tuple_of_int."""
    label_types = {
        "field_race": {"type": "multi_categorical", "target": False},
    }
    lt_path = tmp_path / "label_types.json"
    lt_path.write_text(json.dumps(label_types))

    labels_path = tmp_path / "last_labels.json"
    enrollment_path = tmp_path / "enrollment_info.json"
    labels_path.write_text(json.dumps({}))
    enrollment_path.write_text(json.dumps({}))

    os.environ["LABELS_DATA_PATH"] = str(labels_path)
    os.environ["ENROLLMENT_DATA_PATH"] = str(enrollment_path)
    os.environ["LABEL_TYPES_PATH"] = str(lt_path)
    os.environ["LABEL_VALIDITY_PATH"] = str(tmp_path / "nope.json")
    os.environ["HEALTHKIT_DAILY_PATH"] = str(tmp_path / "nope.json")
    os.environ["CONTEXT_LABELS_PATH"] = str(tmp_path / "nope.json")
    import labels.api as api
    api = importlib.reload(api)

    assert api._enforce_type("field_race", [1, 3]) == (1, 3)


def test_multi_categorical_names_export_is_derived_from_label_types(tmp_path: Path) -> None:
    """MULTI_CATEGORICAL_NAMES is the list of labels with type=multi_categorical."""
    label_types = {
        "Diabetes": {"type": "binary", "target": True},
        "field_race": {"type": "multi_categorical", "target": False},
        "field_family_history": {"type": "multi_categorical", "target": False},
    }
    lt_path = tmp_path / "label_types.json"
    lt_path.write_text(json.dumps(label_types))

    labels_path = tmp_path / "last_labels.json"
    enrollment_path = tmp_path / "enrollment_info.json"
    labels_path.write_text(json.dumps({}))
    enrollment_path.write_text(json.dumps({}))

    os.environ["LABELS_DATA_PATH"] = str(labels_path)
    os.environ["ENROLLMENT_DATA_PATH"] = str(enrollment_path)
    os.environ["LABEL_TYPES_PATH"] = str(lt_path)
    os.environ["LABEL_VALIDITY_PATH"] = str(tmp_path / "nope.json")
    os.environ["HEALTHKIT_DAILY_PATH"] = str(tmp_path / "nope.json")
    os.environ["CONTEXT_LABELS_PATH"] = str(tmp_path / "nope.json")
    import labels.api as api
    api = importlib.reload(api)

    assert sorted(api.MULTI_CATEGORICAL_NAMES) == sorted(["field_race", "field_family_history"])


# ---- multi_categorical helpers (T9-T12) ---- #


def _ord_dict_with_field_race(tmp_path: Path) -> Path:
    """Write an ordinal_dictionary.json with field_race int-keyed mappings."""
    path = tmp_path / "ordinal_dictionary.json"
    path.write_text(json.dumps({
        "field_race": {
            "1": "White", "2": "Black", "3": "American Indian",
            "4": "Alaska Native", "5": "Asian Indian",
        },
    }))
    return path


def _setup_helpers_api(tmp_path: Path):
    """Stand up an api module with field_race=multi_categorical for helper tests."""
    label_types = {
        "field_race": {"type": "multi_categorical", "target": False},
        "Diabetes": {"type": "binary", "target": True},
    }
    lt_path = tmp_path / "label_types.json"
    lt_path.write_text(json.dumps(label_types))
    od_path = _ord_dict_with_field_race(tmp_path)
    labels_path = tmp_path / "last_labels.json"
    labels_path.write_text(json.dumps({
        "field_race": {
            "user-123": {"timestamps": ["2020-01-01T00:00:00"], "values": [[1, 3]]},
        },
        "Diabetes": {
            "user-123": {"timestamps": ["2020-01-01T00:00:00"], "values": [True]},
        },
    }))
    enrollment_path = tmp_path / "enrollment_info.json"
    enrollment_path.write_text(json.dumps({"user-123": {"birthdate": "2000-01-15"}}))

    os.environ["LABELS_DATA_PATH"] = str(labels_path)
    os.environ["ENROLLMENT_DATA_PATH"] = str(enrollment_path)
    os.environ["LABEL_TYPES_PATH"] = str(lt_path)
    os.environ["ORDINAL_DICTIONARY_PATH"] = str(od_path)
    os.environ["LABEL_VALIDITY_PATH"] = str(tmp_path / "nope.json")
    os.environ["HEALTHKIT_DAILY_PATH"] = str(tmp_path / "nope.json")
    os.environ["CONTEXT_LABELS_PATH"] = str(tmp_path / "nope.json")
    import labels.api as api
    return importlib.reload(api)


def test_get_labels_one_hot_returns_per_option_bool(tmp_path: Path) -> None:
    """T9: one-hot returns dict[option_code -> bool] over options from dict file."""
    api = _setup_helpers_api(tmp_path)
    out = api.get_labels_one_hot("user-123", pd.Timestamp("2020-01-01"), "field_race")
    assert out == {1: True, 2: False, 3: True, 4: False, 5: False}


def test_get_labels_count_returns_number_selected(tmp_path: Path) -> None:
    """T10: count returns the number of selected options."""
    api = _setup_helpers_api(tmp_path)
    n = api.get_labels_count("user-123", pd.Timestamp("2020-01-01"), "field_race")
    assert n == 2


def test_get_labels_contains_membership(tmp_path: Path) -> None:
    """T11: contains returns True iff option is in the selected tuple."""
    api = _setup_helpers_api(tmp_path)
    assert api.get_labels_contains(
        "user-123", pd.Timestamp("2020-01-01"), "field_race", option=1
    ) is True
    assert api.get_labels_contains(
        "user-123", pd.Timestamp("2020-01-01"), "field_race", option=2
    ) is False


def test_helpers_reject_non_multi_categorical_label(tmp_path: Path) -> None:
    """T12: helpers raise ValueError on non-multi_categorical labels."""
    api = _setup_helpers_api(tmp_path)
    with pytest.raises(ValueError, match="multi_categorical"):
        api.get_labels_one_hot("user-123", pd.Timestamp("2020-01-01"), "Diabetes")
    with pytest.raises(ValueError, match="multi_categorical"):
        api.get_labels_count("user-123", pd.Timestamp("2020-01-01"), "Diabetes")
    with pytest.raises(ValueError, match="multi_categorical"):
        api.get_labels_contains(
            "user-123", pd.Timestamp("2020-01-01"), "Diabetes", option=1
        )


def test_get_labels_windowed_rejects_multi_categorical(tmp_path: Path) -> None:
    """windowed() over a multi_categorical label raises ValueError, not TypeError."""
    api = _setup_helpers_api(tmp_path)
    with pytest.raises(ValueError, match="multi_categorical"):
        api.get_labels_windowed(
            "user-123", pd.Timestamp("2020-01-01"), "field_race",
            window_days=7, aggregation="mean",
        )


def test_get_labels_statistics_skips_multi_categorical(tmp_path: Path) -> None:
    """get_labels_statistics omits multi_categorical labels (no silent NaN coercion)."""
    api = _setup_helpers_api(tmp_path)
    df = api.get_labels_statistics()
    assert "field_race" not in df["label"].values
    # sanity: non-multi_categorical labels still appear
    assert "Diabetes" in df["label"].values


# ---- Privacy migration (birth_year) tests ---- #


def test_get_birth_year_from_birth_year_field(tmp_path: Path) -> None:
    """When enrollment has 'birth_year', get_birth_year returns it as int."""
    labels_path = tmp_path / "last_labels.json"
    enrollment_path = tmp_path / "enrollment_info.json"
    labels_path.write_text(json.dumps({}))
    enrollment_path.write_text(json.dumps({"user-123": {"birth_year": 1985}}))
    api = _load_api_with_paths(labels_path, enrollment_path)

    assert api.STORE.enrollment.get_birth_year("user-123") == 1985


def test_get_birth_year_transition_fallback_from_birthdate(tmp_path: Path) -> None:
    """Legacy 'birthdate' field is accepted: birth year derived from it."""
    labels_path = tmp_path / "last_labels.json"
    enrollment_path = tmp_path / "enrollment_info.json"
    labels_path.write_text(json.dumps({}))
    enrollment_path.write_text(json.dumps({"user-123": {"birthdate": "1990-05-01"}}))
    api = _load_api_with_paths(labels_path, enrollment_path)

    assert api.STORE.enrollment.get_birth_year("user-123") == 1990


def test_get_birth_year_missing_raises(tmp_path: Path) -> None:
    """Neither birth_year nor birthdate -> KeyError."""
    labels_path = tmp_path / "last_labels.json"
    enrollment_path = tmp_path / "enrollment_info.json"
    labels_path.write_text(json.dumps({}))
    enrollment_path.write_text(json.dumps({"user-123": {}}))
    api = _load_api_with_paths(labels_path, enrollment_path)

    with pytest.raises(KeyError):
        api.STORE.enrollment.get_birth_year("user-123")


def test_get_birthdate_no_longer_present(tmp_path: Path) -> None:
    """get_birthdate is removed; only get_birth_year exists post-migration."""
    labels_path = tmp_path / "last_labels.json"
    enrollment_path = tmp_path / "enrollment_info.json"
    labels_path.write_text(json.dumps({}))
    enrollment_path.write_text(json.dumps({"user-123": {"birth_year": 1985}}))
    api = _load_api_with_paths(labels_path, enrollment_path)

    assert not hasattr(api.STORE.enrollment, "get_birthdate")


def test_years_between_birth_year_basic(tmp_path: Path) -> None:
    """years_between_birth_year returns calendar-year age."""
    api = _load_api(tmp_path)
    assert api.years_between_birth_year(2000, pd.Timestamp("2020-12-31")) == 20
    assert api.years_between_birth_year(2000, pd.Timestamp("2020-01-01")) == 20
    assert api.years_between_birth_year(1990, pd.Timestamp("2026-06-15")) == 36


def test_years_between_removed(tmp_path: Path) -> None:
    """years_between (birthdate-based) is removed; use years_between_birth_year."""
    api = _load_api(tmp_path)
    assert not hasattr(api, "years_between")


def test_vlm_path_exports_present() -> None:
    """LABEL_TYPES_PATH and ORDINAL_DICTIONARY_PATH are re-exported from labels package."""
    import labels
    assert hasattr(labels, "LABEL_TYPES_PATH")
    assert hasattr(labels, "ORDINAL_DICTIONARY_PATH")


def test_years_between_birth_year_export_present() -> None:
    """years_between_birth_year is re-exported from labels package."""
    import labels
    assert hasattr(labels, "years_between_birth_year")
