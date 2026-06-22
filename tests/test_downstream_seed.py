"""The user seed must reach the downstream bootstrap standard errors.

Regression guard for the PR #7 fix: ``evaluate_prediction(seed=...)`` threads the
seed through ``DownstreamEvaluator`` → ``_metrics_for`` → ``compute_*_metrics`` →
``bootstrap_se``. A different seed must move the SE; the default must reproduce
seed 42 (the canonical leaderboard seed) so published numbers are unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from downstream_evaluation.evaluation.metrics import (
    compute_binary_metrics,
    compute_multiclass_metrics,
    compute_ordinal_metrics,
    compute_regression_metrics,
)


def _data():
    rng = np.random.RandomState(0)
    n = 200
    y_bin = (rng.rand(n) > 0.5).astype(int)
    # Deliberately weak signal: overlapping scores keep AUPRC off 1.0 so the
    # bootstrap SE is non-degenerate and the seed visibly moves it.
    y_prob = np.clip(y_bin * 0.25 + rng.rand(n) * 0.75, 0, 1)
    y_cls = rng.randint(0, 3, n)
    y_cls_pred = np.where(rng.rand(n) > 0.3, y_cls, rng.randint(0, 3, n))
    y_ord = rng.randint(0, 5, n)
    y_ord_pred = np.clip(y_ord + rng.randint(-1, 2, n), 0, 4)
    y_reg = rng.rand(n)
    y_reg_pred = y_reg * 0.7 + rng.rand(n) * 0.3
    return {
        "binary": (compute_binary_metrics, (y_bin, y_prob), "auprc_se"),
        "multiclass": (compute_multiclass_metrics, (y_cls, y_cls_pred), "accuracy_se"),
        "ordinal": (compute_ordinal_metrics, (y_ord, y_ord_pred), "spearman_r_se"),
        "regression": (compute_regression_metrics, (y_reg, y_reg_pred), "pearson_r_se"),
    }


@pytest.mark.parametrize("ttype", ["binary", "multiclass", "ordinal", "regression"])
def test_seed_threads_into_bootstrap_se(ttype):
    """A different seed moves the SE; the default reproduces seed 42."""
    fn, args, se_key = _data()[ttype]
    se_42 = fn(*args, seed=42)[se_key]
    se_7 = fn(*args, seed=7)[se_key]
    se_default = fn(*args)[se_key]

    assert np.isfinite(se_42) and se_42 > 0, "test data must yield a non-degenerate SE"
    # A different seed moves the SE (the seed is actually used).
    assert abs(se_7 - se_42) > 1e-9, f"{ttype}: seed not honored ({se_7} == {se_42})"
    # The default reproduces seed 42 — published leaderboard SEs are unchanged.
    assert se_default == se_42, f"{ttype}: default seed drifted from 42"
    # Same seed is deterministic.
    assert fn(*args, seed=7)[se_key] == se_7


def test_evaluator_exposes_seed():
    """DownstreamEvaluator carries the seed (defaulting to the canonical 42)."""
    from downstream_evaluation.evaluation.evaluator import DownstreamEvaluator

    assert DownstreamEvaluator(seed=7).seed == 7
    assert DownstreamEvaluator().seed == 42  # canonical default
