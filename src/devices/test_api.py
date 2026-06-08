"""Tests for the devices API module."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import pandas as pd
import pytest

_SAMPLE_DATA: dict[str, dict[str, list[list[str]]]] = {
    "sample-user-phone-and-watch": {
        "phone": [
            ["2017-12-18", "2023-05-18", "iPhone 8 Plus"],
            ["2023-05-19", "2024-09-30", "iPhone 14 Pro"],
        ],
        "watch": [
            ["2019-04-05", "2023-12-19", "Apple Watch Series 5 44mm GPS"],
        ],
    },
    "sample-user-phone-only": {
        "phone": [
            ["2016-09-01", "2019-08-31", "iPhone 6s"],
        ],
        "watch": [],
    },
    "sample-user-watch-only": {
        "phone": [],
        "watch": [
            ["2021-01-15", "2022-06-30", "Apple Watch Series 6 44mm GPS"],
        ],
    },
    "sample-user-unknown-bucket": {
        "phone": [
            ["2018-05-10", "2020-12-31", "iPhone Unknown Model"],
        ],
        "watch": [
            ["2018-05-10", "2020-12-31", "Apple Watch Unknown Model"],
        ],
    },
    "sample-user-single-short-interval": {
        "phone": [
            ["2022-07-01", "2022-07-07", "iPhone 13"],
        ],
        "watch": [],
    },
}


@pytest.fixture
def sample_fixture(tmp_path: Path) -> Path:
    """Write the 5-user sample dict to a tmp file and return its path."""
    path = tmp_path / "user_device_info.json"
    path.write_text(json.dumps(_SAMPLE_DATA))
    return path


@pytest.fixture(autouse=True)
def _restore_device_info_env():
    """Snapshot USER_DEVICE_INFO_PATH and restore it after each test.

    Tests in this module overwrite the env var to point at fixtures; this
    keeps that mutation from leaking into sibling tests that import
    ``devices`` without going through ``_load_devices_api``.
    """
    saved = os.environ.get("USER_DEVICE_INFO_PATH")
    yield
    if saved is None:
        os.environ.pop("USER_DEVICE_INFO_PATH", None)
    else:
        os.environ["USER_DEVICE_INFO_PATH"] = saved


def _load_devices_api(device_info_path: Path):
    """Reload devices.api with the given device-info file path."""
    os.environ["USER_DEVICE_INFO_PATH"] = str(device_info_path)
    import devices.api
    return importlib.reload(devices.api)


def test_get_devices_returns_snapshot_with_phone_and_watch(sample_fixture: Path) -> None:
    """get_devices returns a DeviceSnapshot with both fields populated."""
    api = _load_devices_api(sample_fixture)

    snap = api.get_devices(
        "sample-user-phone-and-watch",
        pd.Timestamp("2020-06-01"),
    )
    assert snap.phone is not None
    assert snap.watch is not None
    assert snap.phone.name == "iPhone 8 Plus"
    assert snap.watch.name == "Apple Watch Series 5 44mm GPS"


def test_get_devices_picks_correct_interval_after_upgrade(sample_fixture: Path) -> None:
    """After a phone upgrade, the later interval is returned."""
    api = _load_devices_api(sample_fixture)

    snap = api.get_devices(
        "sample-user-phone-and-watch",
        pd.Timestamp("2024-01-15"),
    )
    assert snap.phone.name == "iPhone 14 Pro"
    # Watch coverage ended 2023-12-19.
    assert snap.watch is None


def test_get_devices_returns_none_outside_coverage(sample_fixture: Path) -> None:
    """Querying before the first record yields None for both classes."""
    api = _load_devices_api(sample_fixture)

    snap = api.get_devices(
        "sample-user-phone-and-watch",
        pd.Timestamp("2010-01-01"),
    )
    assert snap.phone is None
    assert snap.watch is None


def test_get_devices_phone_only_user(sample_fixture: Path) -> None:
    """A user with no watch records yields watch=None."""
    api = _load_devices_api(sample_fixture)

    snap = api.get_devices(
        "sample-user-phone-only",
        pd.Timestamp("2017-06-15"),
    )
    assert snap.phone.name == "iPhone 6s"
    assert snap.watch is None


def test_get_devices_watch_only_user(sample_fixture: Path) -> None:
    """A user with no phone records yields phone=None."""
    api = _load_devices_api(sample_fixture)

    snap = api.get_devices(
        "sample-user-watch-only",
        pd.Timestamp("2021-06-01"),
    )
    assert snap.phone is None
    assert snap.watch.name == "Apple Watch Series 6 44mm GPS"


def test_get_devices_unknown_bucket_returned_as_is(sample_fixture: Path) -> None:
    """'Unknown Model' strings flow through unchanged."""
    api = _load_devices_api(sample_fixture)

    snap = api.get_devices(
        "sample-user-unknown-bucket",
        pd.Timestamp("2019-06-15"),
    )
    assert snap.phone.name == "iPhone Unknown Model"
    assert snap.watch.name == "Apple Watch Unknown Model"


def test_get_devices_inclusive_endpoints(sample_fixture: Path) -> None:
    """Start and end dates of an interval are both included."""
    api = _load_devices_api(sample_fixture)

    start = api.get_devices(
        "sample-user-single-short-interval",
        pd.Timestamp("2022-07-01"),
    )
    end = api.get_devices(
        "sample-user-single-short-interval",
        pd.Timestamp("2022-07-07"),
    )
    assert start.phone.name == "iPhone 13"
    assert end.phone.name == "iPhone 13"


def test_get_devices_unknown_healthcode_raises(sample_fixture: Path) -> None:
    """A healthCode not in the file raises KeyError."""
    api = _load_devices_api(sample_fixture)

    with pytest.raises(KeyError, match="Unknown healthCode"):
        api.get_devices("not-a-real-user", pd.Timestamp("2020-01-01"))


def test_get_device_timeline_returns_sorted_intervals(sample_fixture: Path) -> None:
    """get_device_timeline returns DeviceInterval tuples sorted by start."""
    api = _load_devices_api(sample_fixture)

    timeline = api.get_device_timeline("sample-user-phone-and-watch")
    assert timeline.health_code == "sample-user-phone-and-watch"
    assert len(timeline.phone) == 2
    assert timeline.phone[0].name == "iPhone 8 Plus"
    assert timeline.phone[1].name == "iPhone 14 Pro"
    assert timeline.phone[0].start < timeline.phone[1].start
    assert len(timeline.watch) == 1


def test_missing_data_file_raises_filenotfound(tmp_path: Path) -> None:
    """First lookup against a missing file raises FileNotFoundError."""
    api = _load_devices_api(tmp_path / "does_not_exist.json")

    with pytest.raises(FileNotFoundError, match="user_device_info.json"):
        api.get_devices("sample-user-phone-and-watch", pd.Timestamp("2020-01-01"))


def test_import_succeeds_without_data_file(tmp_path: Path) -> None:
    """The module imports cleanly even when the data file is absent."""
    api = _load_devices_api(tmp_path / "still_does_not_exist.json")

    # Public surface should be available; only first query fails.
    assert hasattr(api, "get_devices")
    assert hasattr(api, "get_device_timeline")
    assert hasattr(api, "DeviceInterval")
    assert hasattr(api, "DeviceSnapshot")
    assert hasattr(api, "DeviceTimeline")


def test_get_devices_using_built_timeline_snapshot_at(sample_fixture: Path) -> None:
    """DeviceTimeline.snapshot_at matches get_devices for the same query."""
    api = _load_devices_api(sample_fixture)

    timeline = api.get_device_timeline("sample-user-phone-and-watch")
    ts = pd.Timestamp("2020-06-01")
    direct = api.get_devices("sample-user-phone-and-watch", ts)
    via_timeline = timeline.snapshot_at(ts)
    assert direct == via_timeline


def test_overlapping_intervals_raise_on_load(tmp_path: Path) -> None:
    """A user with overlapping intervals fails loudly at first query."""
    data = {
        "user-overlap": {
            "phone": [
                ["2020-01-01", "2020-12-31", "iPhone 11"],
                ["2020-06-01", "2021-06-30", "iPhone 12"],
            ],
            "watch": [],
        }
    }
    path = tmp_path / "user_device_info.json"
    path.write_text(json.dumps(data))
    api = _load_devices_api(path)

    with pytest.raises(ValueError, match="Overlapping phone intervals"):
        api.get_devices("user-overlap", pd.Timestamp("2020-08-01"))


def test_intervals_loaded_out_of_order_are_sorted(tmp_path: Path) -> None:
    """Intervals provided out of order in the JSON are sorted on load."""
    data = {
        "user-x": {
            "phone": [
                ["2022-01-01", "2022-06-30", "iPhone 14"],
                ["2018-01-01", "2018-12-31", "iPhone 8"],
            ],
            "watch": [],
        }
    }
    path = tmp_path / "user_device_info.json"
    path.write_text(json.dumps(data))
    api = _load_devices_api(path)

    timeline = api.get_device_timeline("user-x")
    assert timeline.phone[0].name == "iPhone 8"
    assert timeline.phone[1].name == "iPhone 14"
