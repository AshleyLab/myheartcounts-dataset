"""Uploader downstream method-column guard + fallback-rate extraction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

_TOOLS = Path(__file__).resolve().parent.parent / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import upload_leaderboard_substrate as uploader  # noqa: E402


def test_validate_method_column_downstream_pass(tmp_path):
    pq = tmp_path / "linear.parquet"
    pd.DataFrame({"method": ["linear", "linear"], "y_true": [0, 1]}).to_parquet(pq)
    # Mirrors the uploader gate for --track downstream: must not raise.
    uploader.validate_method_column(pq, "linear")


def test_validate_method_column_mismatch_raises(tmp_path):
    pq = tmp_path / "linear.parquet"
    pd.DataFrame({"method": ["custom"], "y_true": [0]}).to_parquet(pq)
    with pytest.raises(SystemExit):
        uploader.validate_method_column(pq, "linear")


def _write_results(tmp_path, payload):
    p = tmp_path / "results.json"
    p.write_text(json.dumps(payload))
    return p


def test_worst_case_fallback_rate_forecasting_top_level(tmp_path):
    # Forecasting / downstream shape: a single top-level scalar.
    p = _write_results(
        tmp_path,
        {"per_channel": {"ch_0": {"mae": 1.0}}, "overall_fallback_rate": 0.0123},
    )
    assert uploader._worst_case_fallback_rate(p) == pytest.approx(0.0123)


def test_worst_case_fallback_rate_forecasting_zero(tmp_path):
    p = _write_results(tmp_path, {"per_channel": {}, "overall_fallback_rate": 0.0})
    assert uploader._worst_case_fallback_rate(p) == 0.0


def test_worst_case_fallback_rate_imputation_nested_max(tmp_path):
    # Imputation shape: max across all scenarios[*][*].overall_fallback_rate cells.
    p = _write_results(
        tmp_path,
        {
            "random_noise": {"test": {"overall_fallback_rate": 0.01}},
            "sleep_gap": {
                "test": {"overall_fallback_rate": 0.05},
                "val": {"overall_fallback_rate": 0.20},
            },
        },
    )
    assert uploader._worst_case_fallback_rate(p) == pytest.approx(0.20)


def test_worst_case_fallback_rate_legacy_absent_returns_none(tmp_path):
    # Legacy run with no fallback field anywhere -> None (not 0.0).
    p = _write_results(tmp_path, {"per_channel": {"ch_0": {"mae": 1.0}}})
    assert uploader._worst_case_fallback_rate(p) is None
