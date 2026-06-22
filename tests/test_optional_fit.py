"""``fit`` is an optional hook on the public ``Method`` contract.

This is an evaluation suite, not training infrastructure, so a zero-shot /
pretrained model may implement only ``predict`` and omit ``fit`` entirely. These
tests drive the real engine path (``DownstreamEvaluator._eval_task``) to prove:

1. a predict-only model runs end-to-end (no ``AttributeError`` on the missing
   ``fit``), and its train inputs are NEVER built (so it never streams/binds the
   train cohort for nothing);
2. a model that DOES define ``fit`` is unchanged — it is still fit on the train
   cohort, with the train inputs built and handed in.

Data-free + fast: ``TaskData`` and the per-participant segments are faked down to
exactly the fields the evaluator reads.
"""

from __future__ import annotations

import numpy as np

from downstream_evaluation.evaluation.evaluator import DownstreamEvaluator


class _Seg:
    """A per-participant segment stand-in: ``_public_inputs`` calls ``.as_array()``."""

    def __init__(self, arr):
        self._arr = arr

    def as_array(self):
        return self._arr


class _ExplodingSeg:
    """A segment whose materialization blows up — proves it is never touched."""

    def as_array(self):
        raise AssertionError("train inputs must NOT be built for a predict-only model")


class _TD:
    """Stands in for TaskData: just the fields the evaluator reads."""

    def __init__(self, user_ids, inputs, labels, task="Diabetes", split="train"):
        self.user_ids = np.array(user_ids, dtype=object)
        self.dates = None
        self.inputs = inputs
        self.labels = labels
        self.task = task
        self.split = split


class _PredictOnly:
    """A zero-shot model: implements ``predict``, NOT ``fit``."""

    def predict(self, data):
        return np.zeros(len(data), dtype=float)


class _WithFit:
    """A trained model: records the train data it was fit on."""

    def __init__(self):
        self.fit_data = None

    def fit(self, data, labels, task_type):
        self.fit_data = list(data)

    def predict(self, data):
        return np.zeros(len(data), dtype=float)


def _test_td():
    arr = np.zeros((1, 24, 38), dtype=np.float32)
    return _TD(["u_a", "u_b"], [_Seg(arr), _Seg(arr)], np.array([0, 1]), split="test")


def test_predict_only_model_skips_fit_and_never_builds_train():
    model = _PredictOnly()
    assert not hasattr(model, "fit")  # the whole point: no fit defined

    # Train inputs would raise if materialized — they must never be touched.
    train_td = _TD(["u_x"], [_ExplodingSeg()], np.array([1]), split="train")
    test_td = _test_td()

    y_true, y_pred = DownstreamEvaluator()._eval_task(
        model, "Diabetes", train_td, test_td, spec=None
    )

    np.testing.assert_array_equal(y_true, [0, 1])
    assert len(y_pred) == 2  # one prediction per test participant, no crash on fit


def test_model_with_fit_still_fits_on_train():
    model = _WithFit()
    train_arr = np.ones((1, 24, 38), dtype=np.float32)
    train_td = _TD(["u_x"], [_Seg(train_arr)], np.array([1]), split="train")
    test_td = _test_td()

    y_true, y_pred = DownstreamEvaluator()._eval_task(
        model, "Diabetes", train_td, test_td, spec=None
    )

    # fit was called and the train inputs were built + handed in (path unchanged).
    assert model.fit_data is not None
    assert len(model.fit_data) == 1
    np.testing.assert_array_equal(model.fit_data[0], train_arr)
    assert len(y_pred) == 2
