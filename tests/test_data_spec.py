"""Unit tests for openmhc.DataSpec — the public input-shape declaration.

Pure-stdlib (no dataset needed): exercises the closed menu, cross-field invariants,
the spec -> internal mapping, and immutability.
"""

import pytest

from openmhc import DataSpec
from openmhc._data_spec import SUPPORTED_SPECS


def test_public_import_path():
    # DataSpec is exported from the package root, not just the private module.
    import openmhc

    assert openmhc.DataSpec is DataSpec


@pytest.mark.parametrize(
    "spec,expected_resolution,expected_granularity,streaming",
    [
        (DataSpec("hourly", "day"), "hourly", "daily", False),
        (DataSpec("hourly", "series", 2048), "hourly", "series", False),
        (DataSpec("minute", "day"), "minute", "daily", True),
    ],
)
def test_supported_specs_map_to_internals(
    spec, expected_resolution, expected_granularity, streaming
):
    assert spec.loader_resolution == expected_resolution
    assert spec.provider_granularity == expected_granularity
    assert spec.is_streaming_required is streaming


def test_menu_is_exactly_the_three_pinned_pairs():
    assert SUPPORTED_SPECS == {("hourly", "day"), ("hourly", "series"), ("minute", "day")}


def test_series_requires_positive_window_units():
    with pytest.raises(ValueError, match="window_units"):
        DataSpec("hourly", "series")  # missing
    with pytest.raises(ValueError, match="window_units"):
        DataSpec("hourly", "series", 0)  # non-positive
    with pytest.raises(ValueError, match="window_units"):
        DataSpec("hourly", "series", 12.5)  # not an int


def test_day_rejects_window_units():
    with pytest.raises(ValueError, match="no window_units"):
        DataSpec("hourly", "day", 2048)


def test_unsupported_combo_fails_at_construction():
    # minute x series is deliberately not shipped -> illegal state is unrepresentable.
    with pytest.raises(ValueError, match="unsupported"):
        DataSpec("minute", "series", 2048)


def test_bad_axis_values():
    with pytest.raises(ValueError, match="resolution"):
        DataSpec("daily", "day")
    with pytest.raises(ValueError, match="window"):
        DataSpec("hourly", "week")


def test_frozen_is_immutable():
    spec = DataSpec("hourly", "day")
    with pytest.raises(Exception):
        spec.resolution = "minute"  # frozen dataclass
