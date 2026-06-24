"""Forecasting fallback rate: to_json preserves it and the uploader extracts it."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from openmhc import ForecastingResults

_TOOLS = Path(__file__).resolve().parent.parent / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import upload_leaderboard_substrate as uploader  # noqa: E402


def test_to_json_preserves_fallback_rate(tmp_path):
    res = ForecastingResults(
        per_channel={"ch_0": {"mae": 1.0}},
        run_dir="/x",
        n_samples=10,
        overall_fallback_rate=0.0123,
        fallback_rate={"ch_0": 0.0},
    )
    p = tmp_path / "results.json"
    res.to_json(p)
    d = json.loads(p.read_text())
    # The rate survives the round-trip at the top level (unlike the old
    # behaviour, which dumped only per_channel and dropped it).
    assert d["overall_fallback_rate"] == pytest.approx(0.0123)
    assert d["fallback_rate"] == {"ch_0": 0.0}
    assert d["per_channel"] == {"ch_0": {"mae": 1.0}}
    # And the uploader extracts it from the exported file (the leaderboard path).
    assert uploader._worst_case_fallback_rate(p) == pytest.approx(0.0123)


def test_to_json_default_zero(tmp_path):
    res = ForecastingResults(per_channel={})
    p = tmp_path / "results.json"
    res.to_json(p)
    assert json.loads(p.read_text())["overall_fallback_rate"] == 0.0
