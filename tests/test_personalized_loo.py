"""Leave-one-sample-out leakage regression tests for personalized imputers.

The personalized imputers fit per-user stats during ``__init__`` by scanning
the official val + test splits. Before the LOSO fix, those stats included the
exact cells the harness would later mask for scoring — the imputer's fill
value for sample ``i`` was informed by sample ``i``'s own held-out cells.

These tests pin the LOSO behavior end-to-end: given a controlled synthetic
val/test split with hand-computable user means, we assert that

  (a) when ``sample_indices`` are supplied, the fill equals the mean computed
      from the user's *other* samples — not the full per-user mean, and
  (b) when ``sample_indices`` are NOT supplied, the fill equals the full
      per-user mean (the non-LOSO fallback path).
"""

from __future__ import annotations

import numpy as np
import pytest

N_CHANNELS = 19
SEQ_LEN = 1440


# ---------------------------------------------------------------------------
# Synthetic split fixture: 3 users, 2 daily samples each, all in "val".
# Sample values are constant per (user, sample) so user means are trivial.
# ---------------------------------------------------------------------------


def _user_value(user_idx: int, sample_offset: int) -> float:
    """Deterministic constant for (user_idx, per-user sample offset)."""
    # Distinct, easily distinguishable values.
    return float(10 * (user_idx + 1) + sample_offset)


N_USERS = 3
SAMPLES_PER_USER = 2
N_VAL = N_USERS * SAMPLES_PER_USER  # 6 samples, all in val

# All-observed mask (no natural missingness) keeps the math simple.
_OBSERVED_FRAC = 1.0


def _make_val_batch():
    """Build the (data, mask) for the val split, ordered to match metadata."""
    data = np.zeros((N_VAL, N_CHANNELS, SEQ_LEN), dtype=np.float32)
    mask = np.ones((N_VAL, N_CHANNELS, SEQ_LEN), dtype=np.float32)
    for u in range(N_USERS):
        for s in range(SAMPLES_PER_USER):
            sample_idx = u * SAMPLES_PER_USER + s
            data[sample_idx, :, :] = _user_value(u, s)
    return data, mask


def _make_val_metadata():
    """One metadata entry per sample, ordered to match `_make_val_batch`."""
    meta = []
    for u in range(N_USERS):
        for s in range(SAMPLES_PER_USER):
            sample_idx = u * SAMPLES_PER_USER + s
            meta.append(
                {
                    "sample_idx": sample_idx,
                    "user_id": f"user_{u}",
                    "date": f"2024-01-{sample_idx + 1:02d}",
                }
            )
    return meta


@pytest.fixture
def stub_personalized_streams(monkeypatch):
    """Patch the streams the personalized imputer reads in ``__init__``."""

    def _fake_iter_split_data(
        split, version=None, data_dir=None, batch_size=5000, num_workers=4, seed=42
    ):
        if split == "train":
            # Two small constant-zero train batches so global fallbacks are 0.
            yield (
                np.zeros((4, N_CHANNELS, SEQ_LEN), dtype=np.float32),
                np.ones((4, N_CHANNELS, SEQ_LEN), dtype=np.float32),
            )
        elif split == "val":
            yield _make_val_batch()
        elif split == "test":
            # Empty test split: no samples.
            yield (
                np.zeros((0, N_CHANNELS, SEQ_LEN), dtype=np.float32),
                np.zeros((0, N_CHANNELS, SEQ_LEN), dtype=np.float32),
            )

    def _fake_load_sample_metadata(split, version=None, data_dir=None, seed=42):
        if split == "train":
            return [
                {"sample_idx": i, "user_id": f"train_user_{i}", "date": "2023-01-01"}
                for i in range(4)
            ]
        if split == "val":
            return _make_val_metadata()
        if split == "test":
            return []
        raise ValueError(split)

    import openmhc._data_utils as _du
    import openmhc.imputers._base as _base
    import openmhc.imputers._personalized_base as _pbase

    monkeypatch.setattr(_du, "iter_split_data", _fake_iter_split_data)
    monkeypatch.setattr(_du, "iter_train_data", lambda **kw: _fake_iter_split_data("train", **kw))
    monkeypatch.setattr(_du, "load_sample_metadata", _fake_load_sample_metadata)

    monkeypatch.setattr(_base, "iter_train_data", lambda **kw: _fake_iter_split_data("train", **kw))
    monkeypatch.setattr(_base, "load_sample_metadata", _fake_load_sample_metadata)

    monkeypatch.setattr(_pbase, "iter_split_data", _fake_iter_split_data)


# ---------------------------------------------------------------------------
# PersonalizedMeanImputer
# ---------------------------------------------------------------------------


def _make_target(sample_idx: int) -> np.ndarray:
    """A target mask that holds out a few cells in channel 0 of one sample."""
    target = np.zeros((1, N_CHANNELS, SEQ_LEN), dtype=np.float32)
    # Pick a handful of held-out minutes; the fill is constant per channel so
    # the exact positions only matter for placement, not value.
    target[0, 0, :5] = 1.0
    return target


class TestPersonalizedMeanLOO:
    def test_loo_fill_excludes_self_sample(self, stub_personalized_streams):
        """LOSO fill must equal the mean of the user's OTHER samples."""
        from openmhc.imputers import PersonalizedMeanImputer

        imp = PersonalizedMeanImputer(version="xs")

        # User 0 sample 0 (split-local sample_idx=0): values _user_value(0, 0)=10.
        # User 0 sample 1 (split-local sample_idx=1): values _user_value(0, 1)=11.
        # Full per-user mean = (10+11)/2 = 10.5 (LEAKY).
        # LOSO mean for sample 0 = 11 (other sample only). EXPECTED.
        data = np.full((1, N_CHANNELS, SEQ_LEN), np.nan, dtype=np.float32)
        observed = np.ones_like(data)
        target = _make_target(sample_idx=0)
        out = imp.impute(
            data,
            observed,
            target,
            user_ids=["user_0"],
            sample_indices=np.array([0], dtype=np.int64),
        )
        t = target[0, 0] > 0.5
        np.testing.assert_allclose(out[0, 0][t], 11.0, rtol=1e-5)

        # And for sample 1, LOSO fill = 10 (other sample only).
        out2 = imp.impute(
            data,
            observed,
            target,
            user_ids=["user_0"],
            sample_indices=np.array([1], dtype=np.int64),
        )
        np.testing.assert_allclose(out2[0, 0][t], 10.0, rtol=1e-5)

    def test_no_sample_indices_uses_full_user_mean(self, stub_personalized_streams):
        """Without sample_indices, falls back to the (leaky) full per-user mean."""
        from openmhc.imputers import PersonalizedMeanImputer

        imp = PersonalizedMeanImputer(version="xs")
        data = np.full((1, N_CHANNELS, SEQ_LEN), np.nan, dtype=np.float32)
        observed = np.ones_like(data)
        target = _make_target(sample_idx=0)
        out = imp.impute(data, observed, target, user_ids=["user_0"])
        t = target[0, 0] > 0.5
        np.testing.assert_allclose(out[0, 0][t], 10.5, rtol=1e-5)

    def test_unknown_user_falls_back_to_global(self, stub_personalized_streams):
        """Unknown user_id uses the global (train) fallback."""
        from openmhc.imputers import PersonalizedMeanImputer

        imp = PersonalizedMeanImputer(version="xs")
        data = np.full((1, N_CHANNELS, SEQ_LEN), np.nan, dtype=np.float32)
        observed = np.ones_like(data)
        target = _make_target(sample_idx=0)
        out = imp.impute(
            data,
            observed,
            target,
            user_ids=["nobody"],
            sample_indices=np.array([999], dtype=np.int64),
        )
        t = target[0, 0] > 0.5
        # Train batch is all zeros, so global fallback channel mean is 0.
        np.testing.assert_allclose(out[0, 0][t], imp._global_fallback[0], rtol=1e-5)


# ---------------------------------------------------------------------------
# PersonalizedTemporalMeanImputer
# ---------------------------------------------------------------------------


class TestPersonalizedTemporalMeanLOO:
    def test_loo_minute_mean_excludes_self_sample(self, stub_personalized_streams):
        """LOSO temporal mean equals the other-samples per-minute mean."""
        from openmhc.imputers import PersonalizedTemporalMeanImputer

        imp = PersonalizedTemporalMeanImputer(version="xs")

        # Sample is constant 10.0 on user_0 sample 0; sample 1 is constant 11.0.
        # Per-minute mean (LOSO for sample 0) = 11.0 at every minute.
        data = np.full((1, N_CHANNELS, SEQ_LEN), np.nan, dtype=np.float32)
        observed = np.ones_like(data)
        target = _make_target(sample_idx=0)
        out = imp.impute(
            data,
            observed,
            target,
            user_ids=["user_0"],
            sample_indices=np.array([0], dtype=np.int64),
        )
        t = target[0, 0] > 0.5
        np.testing.assert_allclose(out[0, 0][t], 11.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# PersonalizedModeImputer
# ---------------------------------------------------------------------------


class TestPersonalizedModeLOO:
    def test_loo_mode_drops_self_only_value(self, stub_personalized_streams):
        """LOSO mode falls back to the other sample's value, not its own."""
        from openmhc.imputers import PersonalizedModeImputer

        imp = PersonalizedModeImputer(version="xs")

        # User 0 sample 0 has all values = 10.0, sample 1 has all values = 11.0.
        # Under macro-of-per-sample-modes: sample 0's mode is 10.0, sample 1's
        # mode is 11.0. Without LOSO, the per-user pool is [10.0, 11.0] —
        # np.unique sorted ascending + np.argmax tie-break picks 10.0. With
        # LOSO for sample 0, only [11.0] remains; mode is 11.0.
        data = np.full((1, N_CHANNELS, SEQ_LEN), np.nan, dtype=np.float32)
        observed = np.ones_like(data)
        target = _make_target(sample_idx=0)
        out = imp.impute(
            data,
            observed,
            target,
            user_ids=["user_0"],
            sample_indices=np.array([0], dtype=np.int64),
        )
        t = target[0, 0] > 0.5
        np.testing.assert_allclose(out[0, 0][t], 11.0, rtol=1e-5)

        out2 = imp.impute(
            data,
            observed,
            target,
            user_ids=["user_0"],
            sample_indices=np.array([1], dtype=np.int64),
        )
        np.testing.assert_allclose(out2[0, 0][t], 10.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# Single-sample-user LOSO edge case (review M6)
#
# A user with exactly one sample, under LOSO, has nothing left to compute
# a per-user statistic from. Each imputer must fall through to the global
# fallback rather than dividing by zero / producing NaN.
# ---------------------------------------------------------------------------


class TestSingleSampleUserLOO:
    def _data_mask_target(self):
        """Tiny inputs for a single-sample impute call."""
        data = np.full((1, N_CHANNELS, SEQ_LEN), np.nan, dtype=np.float32)
        mask = np.ones_like(data)
        target = _make_target(sample_idx=0)
        return data, mask, target

    def test_mean_falls_back_to_global_on_single_sample(self):
        """Per-user mean with one sample, LOSO-excluded → global fallback."""
        from openmhc.imputers import PersonalizedMeanImputer

        with _stub_single_sample_split() as imp_cls_args:
            imp = PersonalizedMeanImputer(*imp_cls_args[0], **imp_cls_args[1])
            data, observed, target = self._data_mask_target()
            out = imp.impute(
                data,
                observed,
                target,
                user_ids=["lone_user"],
                sample_indices=np.array([0], dtype=np.int64),
            )
            t = target[0, 0] > 0.5
            # Train batch is all zeros → global channel mean is 0.
            np.testing.assert_allclose(out[0, 0][t], imp._global_fallback[0], rtol=1e-5)

    def test_temporal_mean_falls_back_to_global_on_single_sample(self):
        """Per-user temporal mean with one sample, LOSO-excluded → global."""
        from openmhc.imputers import PersonalizedTemporalMeanImputer

        with _stub_single_sample_split() as imp_cls_args:
            imp = PersonalizedTemporalMeanImputer(*imp_cls_args[0], **imp_cls_args[1])
            data, observed, target = self._data_mask_target()
            out = imp.impute(
                data,
                observed,
                target,
                user_ids=["lone_user"],
                sample_indices=np.array([0], dtype=np.int64),
            )
            t = target[0, 0] > 0.5
            # Global temporal mean for the all-zero train batch is 0 at every minute.
            np.testing.assert_allclose(
                out[0, 0][t], imp._global_fallback[0, :5], rtol=1e-5
            )

    def test_mode_falls_back_to_global_on_single_sample(self):
        """Per-user mode with one sample, LOSO-excluded → global fallback."""
        from openmhc.imputers import PersonalizedModeImputer

        with _stub_single_sample_split() as imp_cls_args:
            imp = PersonalizedModeImputer(*imp_cls_args[0], **imp_cls_args[1])
            data, observed, target = self._data_mask_target()
            out = imp.impute(
                data,
                observed,
                target,
                user_ids=["lone_user"],
                sample_indices=np.array([0], dtype=np.int64),
            )
            t = target[0, 0] > 0.5
            np.testing.assert_allclose(out[0, 0][t], imp._global_fallback[0], rtol=1e-5)


# Helper for the single-sample-user tests: stubs iter_split_data and
# load_sample_metadata so val has exactly one user with exactly one sample.

import contextlib  # noqa: E402


@contextlib.contextmanager
def _stub_single_sample_split():
    """Yield (args, kwargs) for instantiating an imputer with the stubs active."""
    from unittest.mock import patch

    def _fake_iter_split_data(
        split, version=None, data_dir=None, batch_size=5000, num_workers=4, seed=42
    ):
        if split == "train":
            yield (
                np.zeros((2, N_CHANNELS, SEQ_LEN), dtype=np.float32),
                np.ones((2, N_CHANNELS, SEQ_LEN), dtype=np.float32),
            )
        elif split == "val":
            # One user, one sample, all values = 7.0.
            data = np.full((1, N_CHANNELS, SEQ_LEN), 7.0, dtype=np.float32)
            mask = np.ones_like(data)
            yield data, mask
        elif split == "test":
            yield (
                np.zeros((0, N_CHANNELS, SEQ_LEN), dtype=np.float32),
                np.zeros((0, N_CHANNELS, SEQ_LEN), dtype=np.float32),
            )

    def _fake_load_sample_metadata(split, version=None, data_dir=None, seed=42):
        if split == "train":
            return [
                {"sample_idx": i, "user_id": f"train_user_{i}", "date": "2023-01-01"}
                for i in range(2)
            ]
        if split == "val":
            return [{"sample_idx": 0, "user_id": "lone_user", "date": "2024-01-01"}]
        return []

    import openmhc._data_utils as _du
    import openmhc.imputers._base as _base
    import openmhc.imputers._personalized_base as _pbase

    with patch.object(_du, "iter_split_data", _fake_iter_split_data), \
         patch.object(_du, "iter_train_data", lambda **kw: _fake_iter_split_data("train", **kw)), \
         patch.object(_du, "load_sample_metadata", _fake_load_sample_metadata), \
         patch.object(_base, "iter_train_data", lambda **kw: _fake_iter_split_data("train", **kw)), \
         patch.object(_base, "load_sample_metadata", _fake_load_sample_metadata), \
         patch.object(_pbase, "iter_split_data", _fake_iter_split_data):
        yield ((), {"version": "xs"})
