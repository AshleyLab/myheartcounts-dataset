"""Uploader fallback-rate persistence (issue #39) + downstream method-column guard."""

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


def test_resolve_fallback_rate_flag_wins(tmp_path):
    pq = tmp_path / "m.parquet"
    pq.write_bytes(b"")  # presence only; the flag short-circuits any sidecar read
    Path(f"{pq}.meta.json").write_text(json.dumps({"overall_fallback_rate": 0.9}))
    rate, source = uploader.resolve_fallback_rate(pq, 0.25)
    assert rate == 0.25
    assert source == "--fallback-rate"


def test_resolve_fallback_rate_from_sidecar(tmp_path):
    pq = tmp_path / "m.parquet"
    Path(f"{pq}.meta.json").write_text(json.dumps({"overall_fallback_rate": 0.125}))
    rate, source = uploader.resolve_fallback_rate(pq, None)
    assert rate == 0.125
    assert source == "m.parquet.meta.json"


def test_resolve_fallback_rate_absent(tmp_path):
    pq = tmp_path / "m.parquet"  # no sidecar, no flag → None → leaderboard shows n/a
    rate, source = uploader.resolve_fallback_rate(pq, None)
    assert rate is None
    assert source == ""


def test_resolve_fallback_rate_sidecar_without_key(tmp_path):
    pq = tmp_path / "m.parquet"
    Path(f"{pq}.meta.json").write_text(json.dumps({"method": "m"}))  # no rate key
    rate, _ = uploader.resolve_fallback_rate(pq, None)
    assert rate is None


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
