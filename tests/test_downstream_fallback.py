"""Missing-prediction fallback (Track 1, issue #38).

The harness substitutes the Linear baseline for participants a model leaves
non-finite, reproducing the per-user routing the ``Hybrid`` (WBM) model used to
do internally. These tests pin the *combine* algorithm byte-identical to that
former internal logic (so reported WBM numbers cannot move) without needing the
dataset — the end-to-end byte-parity vs the golden is a separate cluster run.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import rankdata

from downstream_evaluation.evaluation.evaluator import _combine_with_fallback
from openmhc import PredictionResults


def _hybrid_reference(ssl_by_user, fb_by_user, daily_users, ssl_users, ttype):
    """Verbatim re-implementation of the pre-#38 ``Hybrid.predict`` branch combine.

    The ``Hybrid`` model was removed in #38 (its routing moved to the harness);
    this copy is preserved as the parity oracle so any drift in the harness
    ``_combine_with_fallback`` helper is caught.
    """
    ssl_set = set(ssl_users)
    is_ssl = np.array([u in ssl_set for u in daily_users])
    out = np.zeros(len(daily_users), dtype=np.float64)
    ssl_idx = np.where(is_ssl)[0]
    fb_idx = np.where(~is_ssl)[0]
    ssl_vals = np.array([ssl_by_user[daily_users[i]] for i in ssl_idx], dtype=np.float64)
    fb_vals = np.array([fb_by_user[daily_users[i]] for i in fb_idx], dtype=np.float64)
    if ttype in ("binary", "ordinal") and len(ssl_idx) and len(fb_idx):
        ssl_vals = rankdata(ssl_vals) / max(len(ssl_vals), 1)
        fb_vals = rankdata(fb_vals) / max(len(fb_vals), 1)
    out[ssl_idx] = ssl_vals
    out[fb_idx] = fb_vals
    return out


@pytest.mark.parametrize("ttype", ["binary", "ordinal", "regression", "multiclass"])
@pytest.mark.parametrize("seed", [0, 1, 7, 42, 123])
def test_combine_matches_hybrid_reference(ttype, seed):
    """The harness combine reproduces Hybrid's per-branch rank-then-merge."""
    rng = np.random.default_rng(seed)
    n = 60
    daily_users = [f"u{i}" for i in range(n)]
    # Random SSL cohort = a subset with weekly embeddings.
    is_ssl = rng.random(n) < 0.6
    ssl_users = [daily_users[i] for i in range(n) if is_ssl[i]]

    ssl_raw = rng.normal(size=n)
    fb_raw = rng.normal(size=n)
    ssl_by_user = {daily_users[i]: ssl_raw[i] for i in range(n) if is_ssl[i]}
    fb_by_user = {daily_users[i]: fb_raw[i] for i in range(n)}

    expected = _hybrid_reference(ssl_by_user, fb_by_user, daily_users, ssl_users, ttype)

    # Harness inputs: y_pred = SSL prediction where available else NaN; fb =
    # Linear baseline for every user (the harness only reads its non-finite rows).
    y_pred = np.where(is_ssl, ssl_raw, np.nan)
    out, n_sub = _combine_with_fallback(y_pred, fb_raw, ttype)

    assert np.array_equal(out, expected)
    assert n_sub == int((~is_ssl).sum())


@pytest.mark.parametrize("ttype", ["binary", "ordinal", "regression"])
def test_combine_all_fallback_no_rank(ttype):
    """No finite cohort → raw baseline substitution, no ranking (matches Hybrid)."""
    y_pred = np.full(10, np.nan)
    fb = np.arange(10, dtype=np.float64)
    out, n_sub = _combine_with_fallback(y_pred, fb, ttype)
    assert np.array_equal(out, fb)
    assert n_sub == 10


def test_combine_no_nonfinite_is_identity():
    """All-finite predictions are returned unchanged with zero substitutions."""
    y_pred = np.array([0.1, 0.9, 0.5, 0.3])
    out, n_sub = _combine_with_fallback(y_pred, np.zeros(4), "binary")
    assert np.array_equal(out, y_pred)
    assert n_sub == 0


def test_wbmprobe_predict_aligns_and_nans_missing():
    """WBMProbe scatters SSL preds onto the daily cohort, NaN where no weekly embedding.

    The NaN positions are exactly what the harness fallback then substitutes with
    the Linear baseline. Exercised with mocked encoder/probe (no dataset needed).
    """
    import types

    from downstream_evaluation.models.wbm import WBMProbe

    m = WBMProbe("/tmp/x")
    # Daily cohort = u0..u3; weekly (encodable) cohort = u1, u3 only.
    m._ctx = types.SimpleNamespace(task="Diabetes", split="test", user_ids=["u0", "u1", "u2", "u3"])
    weekly = types.SimpleNamespace(user_ids=["u1", "u3"])
    m._weekly_td = lambda task, split: weekly
    m._wbm = types.SimpleNamespace(encode_cohort=lambda task, td: np.zeros((len(td.user_ids), 4)))
    m._probe = types.SimpleNamespace(predict=lambda X: np.array([0.7, 0.9]))

    out = m.predict(None)

    assert out.shape == (4,)
    assert np.isnan(out[0]) and np.isnan(out[2])  # no weekly embedding → harness fallback
    assert out[1] == 0.7 and out[3] == 0.9  # weekly users keep the SSL probe output


def test_prediction_results_fallback_fields_default():
    """PredictionResults stays constructible from records alone (additive fields)."""
    pr = PredictionResults(records=[{"task": "t", "metric": "auprc", "value": 0.5}])
    assert pr.overall_fallback_rate == 0.0
    assert pr.fallback_rate == {}
    assert "overall_fallback_rate=0.0000" in repr(pr)


def test_prediction_results_fallback_fields_populated():
    pr = PredictionResults(
        records=[],
        overall_fallback_rate=0.25,
        fallback_rate={"task_a": 0.5},
    )
    assert pr.overall_fallback_rate == 0.25
    assert pr.fallback_rate == {"task_a": 0.5}
    # Fallback stays out of the metrics table so eval_<method>.csv is unaffected.
    assert "fallback" not in {c for c in pr.to_dataframe().columns}
