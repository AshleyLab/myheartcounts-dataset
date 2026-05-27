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
        # Per-channel: filled value equals the imputer's stored channel mean.
        for ch in range(N_CHANNELS):
            ch_target = target_bool[:, ch, :]
            if ch_target.any():
                vals = out[:, ch, :][ch_target]
                np.testing.assert_allclose(vals, imp._channel_means[ch], rtol=1e-5)


class TestModeImputer:
    def test_fills_with_per_channel_mode(self, stub_iter_train_data):
        from openmhc.imputers import ModeImputer

        imp = ModeImputer(version="xs")
        data, mask = _make_synthetic_batch(5, seed=7)
        target = _make_target_mask(mask, frac=0.1)
        out = imp.impute(data, mask, target)
        assert out.shape == data.shape
        assert out.dtype == np.float32
        target_bool = target > 0.5
        for ch in range(N_CHANNELS):
            ch_target = target_bool[:, ch, :]
            if ch_target.any():
                vals = out[:, ch, :][ch_target]
                np.testing.assert_allclose(vals, imp._channel_modes[ch], rtol=1e-5)


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
