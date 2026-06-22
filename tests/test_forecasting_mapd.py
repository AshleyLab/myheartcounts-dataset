"""Unit checks for the fairness per-task disparity primitive (MAPD).

The fairness skill score's per-task disparity switched from max-min to the mean
absolute pairwise difference (MAPD). For a 2-subgroup attribute (e.g. ``sex``)
MAPD is identical to max-min, so those rows are unchanged; for >=3 subgroups
(e.g. ``age_group`` with up to 5 buckets) it smooths over every pair.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecasting_evaluation.metrics.fair_skill_score import _mapd


def test_mapd_g2_equals_max_minus_min():
    """|G| = 2 collapses to |E_a - E_b| — bit-identical to the old max-min."""
    assert _mapd(pd.Series([1.0, 3.0])) == pytest.approx(2.0)
    assert _mapd(pd.Series([3.0, 1.0])) == pytest.approx(2.0)


def test_mapd_g3_smooths_over_all_pairs():
    """[1,2,4]: pairs |1-2|,|1-4|,|2-4| = 1,3,2 -> mean 2.0 (max-min would be 3.0)."""
    assert _mapd(pd.Series([1.0, 2.0, 4.0])) == pytest.approx(2.0)


def test_mapd_g4():
    """[0,1,2,5]: 6 pairs sum to 16 -> 16/6."""
    assert _mapd(pd.Series([0.0, 1.0, 2.0, 5.0])) == pytest.approx(16.0 / 6.0)


def test_mapd_ignores_nan_and_requires_two_finite():
    """Non-finite values are dropped; <2 finite -> NaN (mirrors the task drop)."""
    assert _mapd(pd.Series([1.0, np.nan, 3.0])) == pytest.approx(2.0)
    assert np.isnan(_mapd(pd.Series([5.0])))
    assert np.isnan(_mapd(pd.Series([np.nan, np.nan])))
