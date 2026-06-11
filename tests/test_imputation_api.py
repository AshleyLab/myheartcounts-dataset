"""Tests for the imputation API: protocol, adapter, utilities, baselines."""

from __future__ import annotations

import numpy as np
import pytest

from openmhc import Imputer
from openmhc._evaluate import _ImputerMethodAdapter

N_CHANNELS = 19
SEQ_LEN = 1440


# ---------------------------------------------------------------------------
# Synthetic data fixtures (no real dataset required)
# ---------------------------------------------------------------------------


def _make_synthetic_batch(
    n: int, seed: int = 0, missing_frac: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    """Build (data, mask) pair of shape (n, 19, 1440)."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, N_CHANNELS, SEQ_LEN)).astype(np.float32)
    # Channels 7-18 are binary (0/1).
    data[:, 7:, :] = (rng.random((n, 12, SEQ_LEN)) < 0.3).astype(np.float32)
    mask = (rng.random((n, N_CHANNELS, SEQ_LEN)) > missing_frac).astype(np.float32)
    data = np.where(mask > 0.5, data, np.nan)
    return data, mask.astype(np.float32)


def _make_target_mask(
    observed_mask: np.ndarray, frac: float = 0.2, seed: int = 1
) -> np.ndarray:
    """Subset of observed_mask == 1 marked as target_mask == 1."""
    rng = np.random.default_rng(seed)
    pool = observed_mask > 0.5
    target = pool & (rng.random(observed_mask.shape) < frac)
    return target.astype(np.float32)


@pytest.fixture
def stub_iter_train_data(monkeypatch):
    """Patch ``openmhc.iter_train_data`` to yield synthetic batches."""
    def _fake_iter_split_data(
        split, version=None, data_dir=None, batch_size=5000, num_workers=4, seed=42
    ):
        if split == "train":
            yield _make_synthetic_batch(40, seed=10)
            yield _make_synthetic_batch(40, seed=11)
        elif split == "val":
            yield _make_synthetic_batch(20, seed=20)
        elif split == "test":
            yield _make_synthetic_batch(20, seed=30)

    # Patch in the places imputers actually look it up.
    import openmhc._data_utils as _du

    monkeypatch.setattr(_du, "iter_split_data", _fake_iter_split_data)
    monkeypatch.setattr(
        _du, "iter_train_data", lambda **kw: _fake_iter_split_data("train", **kw)
    )

    import openmhc.imputers._base as _base

    monkeypatch.setattr(_base, "iter_train_data", lambda **kw: _fake_iter_split_data("train", **kw))

    import openmhc.imputers._personalized_base as _pbase

    monkeypatch.setattr(_pbase, "iter_split_data", _fake_iter_split_data)


@pytest.fixture
def stub_metadata(monkeypatch):
    """Patch ``load_sample_metadata`` so personalized imputers see synthetic users."""
    def _fake(split, version=None, data_dir=None, seed=42):
        n = {"train": 80, "val": 20, "test": 20}[split]
        # Five distinct users, cycling.
        return [
            {
                "sample_idx": i,
                "user_id": f"user_{i % 5}",
                "date": f"2024-01-{(i % 28) + 1:02d}",
            }
            for i in range(n)
        ]

    import openmhc._data_utils as _du

    monkeypatch.setattr(_du, "load_sample_metadata", _fake)
    import openmhc.imputers._base as _base

    monkeypatch.setattr(_base, "load_sample_metadata", _fake)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class TestImputerProtocol:
    def test_fake_imputer_satisfies_protocol(self):
        class FakeImputer:
            def impute(self, data, observed_mask, target_mask):
                return data

        assert isinstance(FakeImputer(), Imputer)

    def test_imputer_with_metadata_kwargs_satisfies_protocol(self):
        class FakeImputer:
            def impute(
                self,
                data,
                observed_mask,
                target_mask,
                *,
                sample_indices=None,
                user_ids=None,
                dates=None,
            ):
                return data

        assert isinstance(FakeImputer(), Imputer)

    def test_object_without_impute_is_not_imputer(self):
        class NotAnImputer:
            def fit(self, data, masks):
                pass

        assert not isinstance(NotAnImputer(), Imputer)


# ---------------------------------------------------------------------------
# Adapter — hard-break, signature filtering, fit, prepare_split, impute
# ---------------------------------------------------------------------------


class TestAdapterHardBreak:
    def test_rejects_imputer_without_impute(self):
        class OldStyle:
            def fit(self, data, masks):
                pass

        with pytest.raises(TypeError, match="impute"):
            _ImputerMethodAdapter(OldStyle())


class _RecordingImputer:
    """Records the kwargs it was called with."""

    def __init__(self, signature_form="minimal"):
        # Build an impute that matches the requested signature shape.
        if signature_form == "minimal":
            def impute(self, data, observed_mask, target_mask):
                self.called_with = {
                    "data": data,
                    "observed_mask": observed_mask,
                    "target_mask": target_mask,
                }
                return data

        elif signature_form == "personalized":
            def impute(
                self, data, observed_mask, target_mask, *, user_ids=None,
            ):
                self.called_with = {
                    "data": data,
                    "observed_mask": observed_mask,
                    "target_mask": target_mask,
                    "user_ids": user_ids,
                }
                return data

        elif signature_form == "full":
            def impute(
                self,
                data,
                observed_mask,
                target_mask,
                *,
                sample_indices=None,
                user_ids=None,
                dates=None,
            ):
                self.called_with = {
                    "data": data,
                    "observed_mask": observed_mask,
                    "target_mask": target_mask,
                    "sample_indices": sample_indices,
                    "user_ids": user_ids,
                    "dates": dates,
                }
                return data

        else:
            raise ValueError(signature_form)

        # Bind the right impute as a method.
        self.impute = impute.__get__(self)
        self.called_with: dict = {}


class _FakeHfDataset:
    """Minimal HF-dataset stand-in supporting column access via __getitem__."""

    def __init__(self, user_ids, dates):
        self._cols = {"user_id": user_ids, "date": dates}

    def __getitem__(self, key):
        return self._cols[key]


class TestAdapterSignatureFiltering:
    def test_minimal_signature_gets_only_three_args(self):
        imp = _RecordingImputer("minimal")
        adapter = _ImputerMethodAdapter(imp)
        adapter.prepare_split(
            _FakeHfDataset(["u0", "u1"], ["2024-01-01", "2024-01-02"]),
            [0, 1],
            None,
        )
        data, mask = _make_synthetic_batch(2, seed=0)
        target = _make_target_mask(mask)
        adapter.impute(
            data, mask, target, sample_indices=np.array([0, 1])
        )
        assert set(imp.called_with) == {"data", "observed_mask", "target_mask"}

    def test_personalized_signature_gets_user_ids(self):
        imp = _RecordingImputer("personalized")
        adapter = _ImputerMethodAdapter(imp)
        adapter.prepare_split(
            _FakeHfDataset(["alice", "bob"], ["2024-01-01", "2024-01-02"]),
            [0, 1],
            None,
        )
        data, mask = _make_synthetic_batch(2, seed=0)
        target = _make_target_mask(mask)
        adapter.impute(
            data, mask, target, sample_indices=np.array([0, 1])
        )
        assert imp.called_with["user_ids"] == ["alice", "bob"]
        assert "sample_indices" not in imp.called_with
        assert "dates" not in imp.called_with

    def test_full_signature_gets_all_metadata(self):
        imp = _RecordingImputer("full")
        adapter = _ImputerMethodAdapter(imp)
        adapter.prepare_split(
            _FakeHfDataset(["alice", "bob"], ["2024-01-01", "2024-01-02"]),
            [0, 1],
            None,
        )
        data, mask = _make_synthetic_batch(2, seed=0)
        target = _make_target_mask(mask)
        adapter.impute(
            data, mask, target, sample_indices=np.array([0, 1])
        )
        assert imp.called_with["user_ids"] == ["alice", "bob"]
        assert imp.called_with["dates"] == ["2024-01-01", "2024-01-02"]
        assert imp.called_with["sample_indices"].tolist() == [0, 1]


class TestAdapterFitComputesStds:
    def test_fit_streams_loader_and_sets_channel_stds(self):
        class FakeImputer:
            def impute(self, data, observed_mask, target_mask):
                return data

        adapter = _ImputerMethodAdapter(FakeImputer())
        loader = [
            _make_synthetic_batch(20, seed=0),
            _make_synthetic_batch(20, seed=1),
        ]
        adapter.fit(loader)
        stds = adapter.channel_stds
        assert stds is not None
        assert stds.shape == (N_CHANNELS,)
        assert stds.dtype == np.float32
        assert np.all(np.isfinite(stds))
        assert np.all(stds >= 1e-6)

    def test_fit_does_not_invoke_user_methods(self):
        calls = []

        class FakeImputer:
            def impute(self, data, observed_mask, target_mask):
                calls.append("impute")
                return data

            def fit(self, *args, **kwargs):
                calls.append("fit")

        adapter = _ImputerMethodAdapter(FakeImputer())
        adapter.fit([_make_synthetic_batch(5, seed=0)])
        assert calls == []  # neither impute nor any hypothetical fit was called


# ---------------------------------------------------------------------------
# BaseImputer + baselines
# ---------------------------------------------------------------------------


class TestBaseImputer:
    def test_default_impute_raises(self):
        from openmhc.imputers import BaseImputer

        bi = BaseImputer(version="xs")
        with pytest.raises(NotImplementedError):
            bi.impute(np.zeros((1, 19, 10)), np.ones((1, 19, 10)), np.zeros((1, 19, 10)))


# The ``stub_iter_train_data`` fixture yields exactly these two batches as
# the training stream (see ``_fake_iter_split_data`` above). Independently
# reconstructing them here lets the mean/mode tests compute expected fill
# values from the *training data*, not from the imputer's stored statistics
# — otherwise the assertion would be tautological (filling-with-X equals X).
_FIXTURE_TRAIN_BATCH_SEEDS: tuple[int, ...] = (10, 11)
_FIXTURE_TRAIN_BATCH_SIZE: int = 40


def _materialize_fixture_train_data() -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the concatenated (data, mask) the stub fixture streams."""
    batches = [
        _make_synthetic_batch(_FIXTURE_TRAIN_BATCH_SIZE, seed=s)
        for s in _FIXTURE_TRAIN_BATCH_SEEDS
    ]
    data = np.concatenate([b[0] for b in batches], axis=0)
    mask = np.concatenate([b[1] for b in batches], axis=0)
    return data, mask


class TestMeanImputer:
    def test_fills_only_target_positions_with_channel_mean(self, stub_iter_train_data):
        from openmhc.imputers import MeanImputer

        imp = MeanImputer(version="xs")
        data, mask = _make_synthetic_batch(5, seed=42)
        target = _make_target_mask(mask, frac=0.1)
        # Wipe target positions (simulate the harness's NaN injection).
        data_corrupted = data.copy()
        data_corrupted[target > 0.5] = np.nan

        out = imp.impute(data_corrupted, mask, target)

        assert out.shape == data.shape
        assert out.dtype == np.float32
        # Non-target observed positions are untouched (NaN handling aside).
        observed_keep = (mask > 0.5) & (target < 0.5)
        np.testing.assert_array_equal(out[observed_keep], data[observed_keep])
        # Every target position got a finite value.
        target_bool = target > 0.5
        assert np.all(np.isfinite(out[target_bool]))

        # Compute the expected channel mean directly from the training batches
        # the fixture streams — using ``np.nanmean``, which is independent of
        # the imputer's streaming sum/count implementation.
        train_data, train_mask = _materialize_fixture_train_data()
        train_obs = train_data.copy()
        train_obs[train_mask < 0.5] = np.nan
        expected_means = np.nanmean(train_obs, axis=(0, 2)).astype(np.float32)

        for ch in range(N_CHANNELS):
            ch_target = target_bool[:, ch, :]
            if ch_target.any():
                vals = out[:, ch, :][ch_target]
                np.testing.assert_allclose(vals, expected_means[ch], rtol=1e-5)


class TestModeImputer:
    def test_fills_with_per_channel_mode(self, stub_iter_train_data):
        from openmhc.imputers import ModeImputer

        imp = ModeImputer(version="xs")
        data, mask = _make_synthetic_batch(5, seed=7)
        target = _make_target_mask(mask, frac=0.1)
        out = imp.impute(data, mask, target)
        assert out.shape == data.shape
        assert out.dtype == np.float32

        # Recompute the per-channel value distribution from the same training
        # batches the fixture streams, then verify each channel's fill value
        # is *a* mode (has maximum count among rounded values). We can't
        # assert equality with a single expected value because round-to-1dp
        # ties are common; we instead check "filled value is one of the
        # most-frequent values" — which is the contract ``ModeImputer``
        # implements. ``np.unique`` + ``argmax`` is independent of the
        # imputer's ``Counter``-based path.
        train_data, train_mask = _materialize_fixture_train_data()
        target_bool = target > 0.5
        for ch in range(N_CHANNELS):
            ch_target = target_bool[:, ch, :]
            if not ch_target.any():
                continue
            vals = out[:, ch, :][ch_target]
            # All target positions in this channel get the same fill value.
            assert np.all(vals == vals[0]), f"ch={ch}: fill not constant: {vals}"
            filled = float(vals[0])

            ch_valid = (train_mask[:, ch, :] > 0.5) & np.isfinite(train_data[:, ch, :])
            if not ch_valid.any():
                continue
            rounded = np.round(train_data[:, ch, :][ch_valid], imp.decimal_precision)
            unique, counts = np.unique(rounded, return_counts=True)
            max_count = int(counts.max())
            match = np.where(np.isclose(unique, filled, atol=1e-5))[0]
            assert match.size == 1, (
                f"ch={ch}: filled value {filled!r} is not among the observed "
                f"rounded training values {unique.tolist()}"
            )
            assert int(counts[match[0]]) == max_count, (
                f"ch={ch}: filled={filled} has count {int(counts[match[0]])}, "
                f"max count={max_count} — not a mode"
            )


class TestLinearImputer:
    def test_interpolates_between_known_anchors(self, stub_iter_train_data):
        from openmhc.imputers import LinearImputer

        imp = LinearImputer(version="xs")
        # Construct a deterministic per-(n, c) ramp so we know the right answer.
        data = np.arange(SEQ_LEN, dtype=np.float32)[None, None, :].repeat(2, axis=0)
        data = np.broadcast_to(data, (2, N_CHANNELS, SEQ_LEN)).copy()
        observed = np.ones_like(data)
        target = np.zeros_like(data)
        # Mask out positions 100..200 (target) for sample 0, channel 0.
        target[0, 0, 100:201] = 1
        data_corrupted = data.copy()
        data_corrupted[target > 0.5] = np.nan
        out = imp.impute(data_corrupted, observed, target)
        # Linear interpolation on a perfect ramp recovers the ramp exactly.
        np.testing.assert_allclose(out[0, 0, 100:201], data[0, 0, 100:201], atol=1e-4)


class TestLOCFImputer:
    def test_carries_last_known_value_forward(self, stub_iter_train_data):
        from openmhc.imputers import LOCFImputer

        imp = LOCFImputer(version="xs")
        data = np.zeros((1, N_CHANNELS, SEQ_LEN), dtype=np.float32)
        data[0, 0, :] = 5.0  # constant value
        observed = np.ones_like(data)
        target = np.zeros_like(data)
        target[0, 0, 50:100] = 1  # mask a contiguous block
        data_corrupted = data.copy()
        data_corrupted[target > 0.5] = np.nan
        out = imp.impute(data_corrupted, observed, target)
        np.testing.assert_allclose(out[0, 0, 50:100], 5.0)


class TestTemporalMeanImputer:
    def test_output_shape_dtype_and_fill_positions(self, stub_iter_train_data):
        from openmhc.imputers import TemporalMeanImputer

        imp = TemporalMeanImputer(version="xs")
        data, mask = _make_synthetic_batch(3, seed=99)
        target = _make_target_mask(mask, frac=0.1)
        out = imp.impute(data, mask, target)
        assert out.shape == data.shape
        assert out.dtype == np.float32
        # Target positions are finite (filled).
        assert np.all(np.isfinite(out[target > 0.5]))


# ---------------------------------------------------------------------------
# Personalized
# ---------------------------------------------------------------------------


class TestPersonalizedMeanImputer:
    def test_per_user_dispatch(self, stub_iter_train_data, stub_metadata):
        from openmhc.imputers import PersonalizedMeanImputer

        imp = PersonalizedMeanImputer(version="xs")
        data, mask = _make_synthetic_batch(2, seed=0)
        target = _make_target_mask(mask, frac=0.1)
        # Two different users -> two different fill values (in general).
        out_a = imp.impute(
            data.copy(), mask.copy(), target.copy(),
            user_ids=["user_0", "user_1"],
        )
        out_b = imp.impute(
            data.copy(), mask.copy(), target.copy(),
            user_ids=["user_2", "user_3"],
        )
        # At least one (sample, channel) should differ — different users
        # have different per-channel means.
        diffs = []
        for i, ch in [(0, 0), (1, 0), (0, 1)]:
            t = target[i, ch] > 0.5
            if t.any():
                diffs.append(not np.allclose(out_a[i, ch][t], out_b[i, ch][t]))
        assert any(diffs)

    def test_unknown_user_falls_back_to_global(self, stub_iter_train_data, stub_metadata):
        from openmhc.imputers import PersonalizedMeanImputer

        imp = PersonalizedMeanImputer(version="xs")
        data, mask = _make_synthetic_batch(1, seed=0)
        target = _make_target_mask(mask, frac=0.1)
        out = imp.impute(data, mask, target, user_ids=["nobody_special"])
        target_bool = target > 0.5
        for ch in range(N_CHANNELS):
            t = target_bool[0, ch]
            if t.any():
                np.testing.assert_allclose(
                    out[0, ch][t], imp._global_fallback[ch], rtol=1e-5
                )


# ---------------------------------------------------------------------------
# TorchImputer
# ---------------------------------------------------------------------------


class TestTorchImputer:
    def test_tiny_model_round_trip(self, stub_iter_train_data):
        torch = pytest.importorskip("torch")
        from openmhc.imputers import TorchImputer

        class IdentityConv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv1d(N_CHANNELS, N_CHANNELS, kernel_size=1)
                # Initialise to (approximately) identity so the model output is
                # close to its input (in normalized space).
                with torch.no_grad():
                    self.conv.weight.zero_()
                    for i in range(N_CHANNELS):
                        self.conv.weight[i, i, 0] = 1.0
                    self.conv.bias.zero_()

            def forward(self, x, mask=None):
                return self.conv(x)

        imp = TorchImputer(
            IdentityConv(),
            version="xs",
            device="cpu",
            inference_batch_size=4,
            forward_signature="x_mask",
            normalize=False,
            nan_fill="zero",
            binary_channels=(),  # disable sigmoid for a pure identity check
        )
        data, mask = _make_synthetic_batch(3, seed=0)
        target = _make_target_mask(mask, frac=0.1)
        data_corrupted = data.copy()
        data_corrupted[target > 0.5] = np.nan
        out = imp.impute(data_corrupted, mask, target)
        assert out.shape == data.shape
        assert out.dtype == np.float32
        # Target positions are finite and reflect the model's output (zero fill -> 0).
        target_bool = target > 0.5
        np.testing.assert_allclose(out[target_bool], 0.0, atol=1e-5)
        # Non-target observed positions are untouched.
        keep = (mask > 0.5) & (target < 0.5)
        np.testing.assert_array_equal(out[keep], data[keep])

    def test_sigmoid_applied_on_binary_channels(self, stub_iter_train_data):
        torch = pytest.importorskip("torch")
        from openmhc.imputers import TorchImputer

        class ZeroModel(torch.nn.Module):
            def forward(self, x, mask=None):
                return torch.zeros_like(x)

        imp = TorchImputer(
            ZeroModel(),
            version="xs",
            device="cpu",
            inference_batch_size=4,
            forward_signature="x_mask",
            normalize=False,
            nan_fill="zero",
            binary_channels=tuple(range(7, 19)),
        )
        data, mask = _make_synthetic_batch(1, seed=0)
        target = _make_target_mask(mask, frac=0.2)
        out = imp.impute(data, mask, target)
        target_bool = target > 0.5
        # Binary channels: sigmoid(0) = 0.5
        for ch in range(7, 19):
            t = target_bool[0, ch]
            if t.any():
                np.testing.assert_allclose(out[0, ch][t], 0.5, atol=1e-5)
        # Continuous channels: untouched by sigmoid, stay at 0
        for ch in range(0, 7):
            t = target_bool[0, ch]
            if t.any():
                np.testing.assert_allclose(out[0, ch][t], 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Fallback substitution: parity for finite imputers + visibility for NaN
# ---------------------------------------------------------------------------


class TestAdapterFallbackFill:
    """The adapter's single fit pass should populate ``fallback_fill`` correctly."""

    def test_fit_sets_fallback_fill_with_correct_shape_and_dtype(self):
        class FakeImputer:
            def impute(self, data, observed_mask, target_mask):
                return data

        adapter = _ImputerMethodAdapter(FakeImputer())
        adapter.fit([_make_synthetic_batch(20, seed=0), _make_synthetic_batch(20, seed=1)])
        fill = adapter.fallback_fill
        assert fill is not None
        assert fill.shape == (N_CHANNELS,)
        assert fill.dtype == np.float32
        assert np.all(np.isfinite(fill))

    def test_binary_channels_are_zero_or_one(self):
        class FakeImputer:
            def impute(self, data, observed_mask, target_mask):
                return data

        adapter = _ImputerMethodAdapter(FakeImputer())
        adapter.fit([_make_synthetic_batch(40, seed=0), _make_synthetic_batch(40, seed=1)])
        fill = adapter.fallback_fill
        for ch in range(7, N_CHANNELS):
            assert fill[ch] in (0.0, 1.0), f"channel {ch} fill={fill[ch]} not majority class"

    def test_binary_majority_class_matches_observed_mean(self):
        """If observed mean > 0.5 → fill=1.0; else fill=0.0."""
        class FakeImputer:
            def impute(self, data, observed_mask, target_mask):
                return data

        # Construct a synthetic batch where binary channel 7 is mostly 1 and channel 8 is mostly 0.
        rng = np.random.default_rng(0)
        data = rng.standard_normal((10, N_CHANNELS, SEQ_LEN)).astype(np.float32)
        mask = np.ones_like(data)  # everything observed
        data[:, 7, :] = 1.0  # majority class 1
        data[:, 8, :] = 0.0  # majority class 0
        # Other binary channels mixed.
        data[:, 9:, :] = (rng.random((10, N_CHANNELS - 9, SEQ_LEN)) < 0.3).astype(np.float32)

        adapter = _ImputerMethodAdapter(FakeImputer())
        adapter.fit([(data, mask)])
        fill = adapter.fallback_fill
        assert fill[7] == 1.0
        assert fill[8] == 0.0


class TestApplyFallbackParity:
    """A finite imputed array must be byte-identical after _apply_fallback (no-op)."""

    def test_finite_imputed_is_unchanged_and_counts_are_zero(self):
        from imputation_evaluation.evaluation.evaluator import _apply_fallback

        rng = np.random.default_rng(0)
        imputed = rng.standard_normal((4, N_CHANNELS, 100)).astype(np.float32)
        artificial = (rng.random(imputed.shape) < 0.2).astype(np.float32)
        fill = np.arange(N_CHANNELS, dtype=np.float32)

        before = imputed.copy()
        sub, asked = _apply_fallback(imputed, artificial, fill)

        # No non-finite cells to substitute → array byte-identical.
        np.testing.assert_array_equal(imputed, before)
        # sub counts must all be zero.
        assert sub.dtype == np.int64
        assert sub.shape == (N_CHANNELS,)
        assert int(sub.sum()) == 0
        # asked counts equal the target-cell count per channel.
        expected_asked = (artificial == 1).sum(axis=(0, 2)).astype(np.int64)
        np.testing.assert_array_equal(asked, expected_asked)

    def test_fill_none_is_complete_noop(self):
        from imputation_evaluation.evaluation.evaluator import _apply_fallback

        rng = np.random.default_rng(0)
        imputed = rng.standard_normal((2, N_CHANNELS, 50)).astype(np.float32)
        # Inject NaN at some target cells.
        artificial = (rng.random(imputed.shape) < 0.2).astype(np.float32)
        imputed[artificial == 1] = np.nan
        before = imputed.copy()

        sub, asked = _apply_fallback(imputed, artificial, None)

        # fill=None → array unchanged (NaN survives), counts zero.
        np.testing.assert_array_equal(
            np.isnan(imputed), np.isnan(before)
        )
        assert int(sub.sum()) == 0
        assert int(asked.sum()) == 0


class TestApplyFallbackSubstitution:
    """When the imputer returns NaN at target cells, those cells must be substituted and counted."""

    def test_nan_cells_at_target_get_filled_and_counted(self):
        from imputation_evaluation.evaluation.evaluator import _apply_fallback

        # Build a controlled scenario: 1 sample, target spans first 10 timesteps on every channel.
        T = 20
        imputed = np.zeros((1, N_CHANNELS, T), dtype=np.float32)
        artificial = np.zeros_like(imputed)
        artificial[0, :, :10] = 1  # target = first 10 timesteps per channel
        # Imputer returns NaN at ALL target cells on channels 0 and 7; finite (0.0) elsewhere.
        imputed[0, 0, :10] = np.nan
        imputed[0, 7, :10] = np.nan
        fill = np.arange(N_CHANNELS, dtype=np.float32) + 1.0  # ch0→1.0, ch7→8.0, etc.

        sub, asked = _apply_fallback(imputed, artificial, fill)

        # Cells substituted: ch 0 → 1.0, ch 7 → 8.0.
        np.testing.assert_allclose(imputed[0, 0, :10], 1.0)
        np.testing.assert_allclose(imputed[0, 7, :10], 8.0)
        # No remaining NaN at target cells.
        assert not np.any(np.isnan(imputed[artificial == 1]))
        # Per-channel sub counts: ch 0 and ch 7 each contributed 10 substitutions.
        assert int(sub[0]) == 10
        assert int(sub[7]) == 10
        assert int(sub.sum()) == 20
        # Asked counts: 10 per channel (target spans 10 timesteps on every channel).
        np.testing.assert_array_equal(asked, np.full(N_CHANNELS, 10, dtype=np.int64))


class TestMetricAccumulatorFallback:
    """The accumulator's fallback fields must merge correctly and surface in compute()."""

    def _make_acc(self):
        from imputation_evaluation.evaluation.evaluator import MetricAccumulator

        return MetricAccumulator(channel_stds=np.ones(N_CHANNELS, dtype=np.float32))

    def test_compute_emits_zero_rate_when_no_substitutions(self):
        acc = self._make_acc()
        # Drive a single batch through the accumulator with finite values.
        rng = np.random.default_rng(0)
        gt = rng.standard_normal((2, N_CHANNELS, 10)).astype(np.float32)
        imputed = gt.copy()
        mask = np.ones_like(gt)
        acc.update(gt, imputed, mask)
        # Record the no-substitution accounting.
        asked = mask.sum(axis=(0, 2)).astype(np.int64)
        acc.add_fallback(np.zeros(N_CHANNELS, dtype=np.int64), asked)

        metrics = acc.compute()
        assert metrics["overall_fallback_rate"] == 0.0
        for ch in range(N_CHANNELS):
            assert metrics["fallback_rate"][f"ch_{ch}"] == 0.0

    def test_compute_emits_nonzero_rate_when_substitutions_occur(self):
        acc = self._make_acc()
        rng = np.random.default_rng(1)
        gt = rng.standard_normal((2, N_CHANNELS, 10)).astype(np.float32)
        imputed = gt.copy()
        mask = np.ones_like(gt)
        acc.update(gt, imputed, mask)
        asked = mask.sum(axis=(0, 2)).astype(np.int64)
        # Pretend channel 0 had half its target cells substituted.
        sub = np.zeros(N_CHANNELS, dtype=np.int64)
        sub[0] = asked[0] // 2
        acc.add_fallback(sub, asked)

        metrics = acc.compute()
        # Overall rate = sub.sum() / asked.sum() = (asked[0]/2) / (N_CHANNELS * asked[0])
        expected_overall = float(sub.sum()) / float(asked.sum())
        assert metrics["overall_fallback_rate"] == pytest.approx(expected_overall)
        # Per-channel: ch_0 == 0.5, others == 0.0.
        assert metrics["fallback_rate"]["ch_0"] == pytest.approx(0.5)
        for ch in range(1, N_CHANNELS):
            assert metrics["fallback_rate"][f"ch_{ch}"] == 0.0

    def test_merge_sums_fallback_counters(self):
        acc_a = self._make_acc()
        acc_b = self._make_acc()
        sub_a = np.zeros(N_CHANNELS, dtype=np.int64); sub_a[0] = 3
        sub_b = np.zeros(N_CHANNELS, dtype=np.int64); sub_b[0] = 5
        asked_a = np.full(N_CHANNELS, 10, dtype=np.int64)
        asked_b = np.full(N_CHANNELS, 10, dtype=np.int64)
        acc_a.add_fallback(sub_a, asked_a)
        acc_b.add_fallback(sub_b, asked_b)
        acc_a.merge(acc_b)
        assert int(acc_a.fallback_substituted[0]) == 8
        assert int(acc_a.fallback_asked[0]) == 20

    def test_parity_existing_metrics_unchanged_when_no_substitutions(self):
        """A finite imputer (no fallback) must produce byte-identical existing metric values."""
        acc_with_fb = self._make_acc()
        acc_without_fb = self._make_acc()
        rng = np.random.default_rng(0)
        gt = rng.standard_normal((4, N_CHANNELS, 50)).astype(np.float32)
        # Make binary channels actually binary so balanced_accuracy is well-defined.
        gt[:, 7:, :] = (rng.random((4, 12, 50)) < 0.5).astype(np.float32)
        imputed = gt + 0.1 * rng.standard_normal(gt.shape).astype(np.float32)
        mask = np.ones_like(gt)
        for acc in (acc_with_fb, acc_without_fb):
            acc.update(gt, imputed, mask)
        # Only one accumulator records the (zero) fallback counts.
        asked = mask.sum(axis=(0, 2)).astype(np.int64)
        acc_with_fb.add_fallback(np.zeros(N_CHANNELS, dtype=np.int64), asked)

        m_with = acc_with_fb.compute()
        m_without = acc_without_fb.compute()

        # The new keys exist on m_with and not on m_without.
        assert "overall_fallback_rate" in m_with
        # Existing metric values must be identical (parity).
        assert m_with["n_samples"] == m_without["n_samples"]
        np.testing.assert_array_equal(
            [m_with["continuous"]["mean_normalized_rmse"]],
            [m_without["continuous"]["mean_normalized_rmse"]],
        )
        for ch in range(N_CHANNELS):
            ch_w = m_with["per_channel"][f"ch_{ch}"]
            ch_wo = m_without["per_channel"][f"ch_{ch}"]
            # Every key present in the without-fb dict must match.
            for k, v in ch_wo.items():
                assert ch_w[k] == v or (
                    isinstance(v, float) and np.isnan(v) and np.isnan(ch_w[k])
                )
