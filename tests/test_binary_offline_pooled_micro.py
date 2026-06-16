"""Pooled within-user micro for binary classification metrics (f1/auroc/auprc).

These metrics are non-decomposable, so true micro means concatenating all of a
user's windows' (truth, score) pairs per channel and scoring the pool once. These
tests lock in that the producer does the pool (not a mean of per-window scalars),
which differs whenever the per-window scores disagree.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from forecasting_evaluation.metrics.binary_offline_calculate import (
    _compute_binary_metrics_for_sample,
    _compute_pooled_binary_metrics_for_user,
)


def _ch(*cells: float) -> np.ndarray:
    """A single-channel (1, horizon) row."""
    return np.asarray([list(cells)], dtype=float)


def test_pooled_auroc_differs_from_mean_of_window_auroc():
    # Window 1: perfectly separated -> AUROC 1.0; Window 2: reversed -> AUROC 0.0.
    gt1, pred1 = _ch(1, 1, 0, 0), _ch(0.9, 0.8, 0.2, 0.1)
    gt2, pred2 = _ch(1, 0), _ch(0.3, 0.7)

    w1 = _compute_binary_metrics_for_sample(point_predictions=pred1, ground_truth=gt1, threshold=0.5)
    w2 = _compute_binary_metrics_for_sample(point_predictions=pred2, ground_truth=gt2, threshold=0.5)
    mean_of_windows = np.mean([w1["auroc"][0], w2["auroc"][0]])
    assert w1["auroc"][0] == pytest.approx(1.0)
    assert w2["auroc"][0] == pytest.approx(0.0)
    assert mean_of_windows == pytest.approx(0.5)

    pooled = _compute_pooled_binary_metrics_for_user(
        windows=[(gt1, pred1), (gt2, pred2)], n_features=1, threshold=0.5
    )
    # Pool: pos scores {0.9,0.8,0.3}, neg {0.2,0.1,0.7} -> 8/9 concordant pairs.
    assert pooled["auroc"][0] == pytest.approx(8.0 / 9.0)
    assert pooled["auroc"][0] != pytest.approx(mean_of_windows, abs=1e-3)


def test_pooled_f1_matches_summed_confusion_counts():
    gt1, pred1 = _ch(1, 1, 0, 0), _ch(0.9, 0.8, 0.2, 0.1)
    gt2, pred2 = _ch(1, 0), _ch(0.3, 0.7)
    pooled = _compute_pooled_binary_metrics_for_user(
        windows=[(gt1, pred1), (gt2, pred2)], n_features=1, threshold=0.5
    )
    # Pooled @0.5: TP={0.9,0.8}=2, FP={0.7}=1, FN={0.3}=1 -> F1 = 2*2/(2*2+1+1).
    tp, fp, fn = 2, 1, 1
    assert pooled["f1"][0] == pytest.approx((2.0 * tp) / (2.0 * tp + fp + fn))
    assert int(pooled["binary_valid_count"][0]) == 6
    assert int(pooled["binary_positive_count"][0]) == 3
    assert int(pooled["binary_negative_count"][0]) == 3


def test_pooling_recovers_auroc_when_a_window_lacks_negatives():
    # Window 1 has only positives -> per-window AUROC is undefined (NaN);
    # pooling with window 2 (a negative) makes the user-level AUROC computable.
    gt1, pred1 = _ch(1, 1), _ch(0.8, 0.9)
    gt2, pred2 = _ch(0), _ch(0.2)

    w1 = _compute_binary_metrics_for_sample(point_predictions=pred1, ground_truth=gt1, threshold=0.5)
    assert math.isnan(w1["auroc"][0])

    pooled = _compute_pooled_binary_metrics_for_user(
        windows=[(gt1, pred1), (gt2, pred2)], n_features=1, threshold=0.5
    )
    assert pooled["auroc"][0] == pytest.approx(1.0)  # both positives outrank the one negative
    assert int(pooled["binary_positive_count"][0]) == 2
    assert int(pooled["binary_negative_count"][0]) == 1


def test_single_window_pool_equals_per_window():
    gt, pred = _ch(1, 0, 1, 0), _ch(0.7, 0.2, 0.6, 0.4)
    per_window = _compute_binary_metrics_for_sample(
        point_predictions=pred, ground_truth=gt, threshold=0.5
    )
    pooled = _compute_pooled_binary_metrics_for_user(
        windows=[(gt, pred)], n_features=1, threshold=0.5
    )
    for key in ("f1", "auroc", "auprc", "binary_valid_count"):
        assert pooled[key][0] == pytest.approx(per_window[key][0])
