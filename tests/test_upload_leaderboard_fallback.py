"""Uploader downstream method-column guard."""

from __future__ import annotations

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
