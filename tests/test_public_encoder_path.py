"""Smoke tests for the public ``Encoder.encode`` path.

The external-submission contract is ``Encoder.encode(data) -> embedding`` where
``data`` is a single ``(n_segments, 24, 38)`` array (channels 0-18 raw sensor values
with NaN at missing positions, 19-37 the missingness mask). The engine, however,
materializes one :class:`ParticipantSegments` (split raw ``.values`` / ``.mask``) per
participant, so :class:`_EncoderMethodAdapter` bridges the two.

No bundled baseline exercises this path (they use ``encode_cohort`` / ``fit``), so
without these tests it can silently break against its documented contract — which is
exactly what happened before the adapter existed. Keep them fast + data-free.
"""

from __future__ import annotations

import numpy as np

from downstream_evaluation.data.binder import ParticipantSegments
from openmhc._evaluate import _EncoderMethodAdapter


class _RecordingEncoder:
    """A public-contract encoder that records exactly what the benchmark hands it."""

    input_granularity = "daily"
    name = "recording"

    def __init__(self) -> None:
        self.seen: np.ndarray | None = None

    def encode(self, data: np.ndarray) -> np.ndarray:
        self.seen = data
        return data.reshape(-1, data.shape[-1]).mean(axis=0)  # (38,)


def _make_segments(n: int = 3, t: int = 24):
    """Return a fake ParticipantSegments plus its raw values/mask arrays."""
    rng = np.random.RandomState(0)
    values = rng.randn(n, t, 19).astype(np.float32)
    mask = (rng.rand(n, t, 19) > 0.5).astype(np.float32)
    return ParticipantSegments(values=values, mask=mask), values, mask


class TestEncoderMethodAdapter:
    """Guard the ParticipantSegments -> (n, 24, 38) array translation."""

    def test_encode_receives_single_n_24_38_array(self):
        """The adapter hands the encoder one ``(n, 24, 38)`` array."""
        seg, _, _ = _make_segments()
        enc = _RecordingEncoder()
        _EncoderMethodAdapter(enc).encode(seg)
        assert isinstance(enc.seen, np.ndarray)
        assert enc.seen.shape == (3, 24, 38)

    def test_channels_are_values_then_mask(self):
        """Channels 0-18 carry raw values and 19-37 carry the mask."""
        seg, values, mask = _make_segments()
        enc = _RecordingEncoder()
        _EncoderMethodAdapter(enc).encode(seg)
        np.testing.assert_array_equal(enc.seen[..., :19], values)
        np.testing.assert_array_equal(enc.seen[..., 19:], mask)

    def test_raw_nan_values_pass_through(self):
        """NaN raw values reach the encoder unfilled."""
        # The contract is RAW values (NaN at missing); the adapter must not pre-fill.
        seg, _, _ = _make_segments()
        seg.values[0, 0, 0] = np.nan
        enc = _RecordingEncoder()
        _EncoderMethodAdapter(enc).encode(seg)
        assert np.isnan(enc.seen[0, 0, 0])

    def test_returns_embedding_as_float32(self):
        """The adapter returns the embedding as a float32 array."""
        seg, _, _ = _make_segments()
        out = _EncoderMethodAdapter(_RecordingEncoder()).encode(seg)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float32
        assert out.shape == (38,)

    def test_inherits_granularity_and_name(self):
        """The adapter inherits the encoder's granularity and name."""
        enc = _RecordingEncoder()
        adapter = _EncoderMethodAdapter(enc)
        assert adapter.input_granularity == "daily"
        assert adapter.name == "recording"

    def test_defaults_when_encoder_omits_metadata(self):
        """The adapter falls back to default granularity and name."""

        class Bare:
            def encode(self, data):
                return data.reshape(-1, 38).mean(axis=0)

        adapter = _EncoderMethodAdapter(Bare())
        assert adapter.input_granularity == "daily"
        assert adapter.name == "custom_encoder"
