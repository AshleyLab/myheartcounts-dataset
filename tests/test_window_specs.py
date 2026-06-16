"""Unit tests for the series slicer + CohortView (Phase 2a).

Synthetic days + a fake loader — no dataset required. Covers calendar layout,
gap NaN-fill, crop/left-pad, and CohortView's spec dispatch + eager==streaming.
"""

import numpy as np
import pytest

from downstream_evaluation.data.cohort import CohortView
from downstream_evaluation.data.windowing import series_window
from downstream_evaluation.evaluation.evaluator import _public_inputs, _spec_inputs
from openmhc import DataSpec


def _days(n, value=1.0, n_ch=19):
    """n fully-observed days of shape (n, 24, n_ch) at a constant value."""
    return np.full((n, 24, n_ch), value, dtype=np.float32)


# --------------------------- series_window ---------------------------

def test_series_left_pads_when_history_short():
    out = series_window(_days(1, 1.0), ["2020-01-01"], window_hours=48)
    assert out.shape == (48, 38)
    # leading 24h padded: values NaN, mask = 1 (missing)
    assert np.isnan(out[:24, :19]).all()
    assert (out[:24, 19:] == 1.0).all()
    # trailing 24h = the observed day: values 1.0, mask 0
    assert (out[24:, :19] == 1.0).all()
    assert (out[24:, 19:] == 0.0).all()


def test_series_crops_to_last_window_hours():
    out = series_window(
        _days(3, 1.0), ["2020-01-01", "2020-01-02", "2020-01-03"], window_hours=24
    )
    assert out.shape == (24, 38)
    assert (out[:, :19] == 1.0).all()  # the last day, fully observed
    assert (out[:, 19:] == 0.0).all()


def test_series_nan_fills_calendar_gaps():
    # 2020-01-02 skipped -> a gap day in the middle of the calendar span.
    out = series_window(_days(2, 1.0), ["2020-01-01", "2020-01-03"], window_hours=72)
    assert out.shape == (72, 38)
    assert (out[:24, :19] == 1.0).all()       # day 1 observed
    assert np.isnan(out[24:48, :19]).all()    # gap day -> NaN
    assert (out[24:48, 19:] == 1.0).all()     # gap day -> mask missing
    assert (out[48:72, :19] == 1.0).all()     # day 3 observed


def test_series_empty_is_all_missing():
    out = series_window(np.zeros((0, 24, 19), np.float32), [], window_hours=10)
    assert out.shape == (10, 38)
    assert np.isnan(out[:, :19]).all()
    assert (out[:, 19:] == 1.0).all()


# --------------------------- CohortView ---------------------------

class _FakeSeg:
    def __init__(self, v, m):
        self.v, self.m = v, m

    def as_array(self):
        return np.concatenate([self.v, self.m], axis=-1)


class FakeLoader:
    """Stands in for DataLoader: serves synthetic per-user days by date."""

    def __init__(self, per_user):  # {uid: (values (n,24,19), [dates])}
        self._d = {str(k): v for k, v in per_user.items()}

    def participant(self, uid, dates):
        vals, ds = self._d[str(uid)]
        eligible = {d[:10] for d in dates}
        keep = [i for i, d in enumerate(ds) if d[:10] in eligible]
        v = vals[keep]
        return _FakeSeg(v, np.isnan(v).astype(np.float32))

    def user_days(self, uid):
        return self._d[str(uid)]


def _cohort(spec, loader, users, dates, labels=None):
    return CohortView(
        loader, spec, np.array(users, dtype=object), dates, labels,
        task_type="binary", task="t", split="train",
    )


def test_cohortview_hourly_day_load_and_iter():
    loader = FakeLoader({
        "u1": (_days(2, 1.0), ["2020-01-01", "2020-01-02"]),
        "u2": (_days(1, 2.0), ["2020-02-01"]),
    })
    cv = _cohort(
        DataSpec("hourly", "day"), loader, ["u1", "u2"],
        [["2020-01-01", "2020-01-02"], ["2020-02-01"]],
    )
    assert len(cv) == 2
    assert cv.load("u1").shape == (2, 24, 38)
    assert [x.shape for x in cv] == [(2, 24, 38), (1, 24, 38)]


def test_cohortview_eager_equals_streaming():
    loader = FakeLoader({"u1": (_days(2, 1.0), ["2020-01-01", "2020-01-02"])})
    cv = _cohort(DataSpec("hourly", "day"), loader, ["u1"], [["2020-01-01", "2020-01-02"]])
    eager = [cv.load(u) for u in cv.user_ids]      # drained list
    streamed = list(cv)                            # one-at-a-time
    assert len(eager) == len(streamed)
    assert np.array_equal(eager[0], streamed[0])


def test_cohortview_series_load():
    loader = FakeLoader({
        "u1": (_days(3, 1.0), ["2020-01-01", "2020-01-02", "2020-01-03"]),
    })
    cv = _cohort(
        DataSpec("hourly", "series", 48), loader, ["u1"],
        [["2020-01-01", "2020-01-02", "2020-01-03"]],
    )
    assert cv.load("u1").shape == (48, 38)


def test_to_public_minute_layout_and_mask():
    # (n=2, C=3, T=4) channel-first NaN-at-missing -> public (n, T, 2C) time-first + mask.
    from downstream_evaluation.data.loader import _to_public_minute

    vals = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    vals[0, 1, 2] = np.nan  # one missing position (n=0, channel=1, t=2)
    out = _to_public_minute(vals)
    assert out.shape == (2, 4, 6)
    # values half = the transpose (0,2,1)
    assert np.array_equal(np.nan_to_num(out[..., :3]), np.nan_to_num(vals.transpose(0, 2, 1)))
    # mask half = isnan: the missing position -> 1, an observed one -> 0
    assert out[0, 2, 3 + 1] == 1.0
    assert out[0, 0, 3 + 0] == 0.0


def test_cohortview_minute_dispatches_to_public_loader():
    # CohortView delegates minute delivery to loader.participant_minute_public (which owns
    # the ZeroToNaN + mask production); here we verify the dispatch + shape with a fake.
    class _MinuteLoader(FakeLoader):
        def participant_minute_public(self, uid, dates):
            return np.zeros((len(dates), 1440, 38), dtype=np.float32)

    loader = _MinuteLoader({"u1": (_days(1), ["2020-01-01", "2020-01-02"])})
    cv = _cohort(DataSpec("minute", "day"), loader, ["u1"], [["2020-01-01", "2020-01-02"]])
    out = cv.load("u1")
    assert out.shape == (2, 1440, 38)


# --------------------------- engine routing (_spec_inputs) ---------------------------

class _FakeTD:
    """Stands in for TaskData: just the fields _spec_inputs / _public_inputs read."""

    def __init__(self, user_ids, dates, labels=None, task="Diabetes", split="train"):
        self.user_ids = np.array(user_ids, dtype=object)
        self.dates = dates
        self.labels = labels
        self.inputs = None
        self.task = task
        self.split = split


def test_spec_eager_drain_equals_legacy_public_inputs():
    # The no-op gate: routing a hourly-day model through _spec_inputs (eager) must yield
    # arrays byte-identical to the legacy bound _public_inputs path.
    loader = FakeLoader({
        "u1": (_days(2, 1.0), ["2020-01-01", "2020-01-02"]),
        "u2": (_days(1, 3.0), ["2020-02-01"]),
    })
    users = ["u1", "u2"]
    dates = [["2020-01-01", "2020-01-02"], ["2020-02-01"]]
    td = _FakeTD(users, dates, labels=np.array([0, 1]))

    # legacy: bind then _public_inputs
    bound = _FakeTD(users, dates, labels=np.array([0, 1]))
    bound.inputs = [loader.participant(u, d) for u, d in zip(users, dates)]
    legacy = _public_inputs(bound)

    eager = _spec_inputs(
        loader, DataSpec("hourly", "day"), td, "binary", streaming=False, with_labels=True
    )
    assert isinstance(eager, list) and len(eager) == len(legacy)
    for a, b in zip(legacy, eager):
        assert np.array_equal(a, b)


def test_spec_streaming_returns_cohortview_with_label_gating():
    loader = FakeLoader({"u1": (_days(1), ["2020-01-01"])})
    td = _FakeTD(["u1"], [["2020-01-01"]], labels=np.array([1]))
    # fit-time: labels present
    fit_data = _spec_inputs(
        loader, DataSpec("minute", "day"), td, "binary", streaming=True, with_labels=True
    )
    assert isinstance(fit_data, CohortView)
    assert fit_data.labels is not None
    # predict-time: labels withheld (no leakage of test y_true into the handle)
    pred_data = _spec_inputs(
        loader, DataSpec("hourly", "day"), td, "binary", streaming=True, with_labels=False
    )
    assert isinstance(pred_data, CohortView)
    assert pred_data.labels is None
