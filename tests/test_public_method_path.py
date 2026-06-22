"""Smoke tests for the public ``Method`` data contract.

The external-submission contract is ``fit(data, labels, task_type)`` /
``predict(data)`` where each participant's entry in ``data`` is a single
``(n_segments, 24, 38)`` array (channels 0-18 raw sensor values with NaN at missing
positions, 19-37 the missingness mask). The engine, however, materializes one
:class:`ParticipantSegments` (split raw ``.values`` / ``.mask``) per participant;
``ParticipantSegments.as_array`` is the translation the evaluator applies before
every ``fit`` / ``predict``.

No bundled baseline depends on that exact layout (they unpack values/mask
themselves), so without these tests it can silently break against its documented
contract. Keep them fast + data-free.
"""

from __future__ import annotations

import numpy as np

from downstream_evaluation.data.loader import ParticipantSegments


def _make_segments(n: int = 3, t: int = 24):
    """Return a fake ParticipantSegments plus its raw values/mask arrays."""
    rng = np.random.RandomState(0)
    values = rng.randn(n, t, 19).astype(np.float32)
    mask = (rng.rand(n, t, 19) > 0.5).astype(np.float32)
    return ParticipantSegments(values=values, mask=mask), values, mask


class TestParticipantSegmentsAsArray:
    """Guard the ParticipantSegments -> (n, 24, 38) array translation."""

    def test_shape_is_n_24_38(self):
        seg, _, _ = _make_segments()
        assert seg.as_array().shape == (3, 24, 38)

    def test_channels_are_values_then_mask(self):
        seg, values, mask = _make_segments()
        arr = seg.as_array()
        np.testing.assert_array_equal(arr[..., :19], values)
        np.testing.assert_array_equal(arr[..., 19:], mask)

    def test_raw_nan_values_pass_through(self):
        # The contract is RAW values (NaN at missing); the translation must not pre-fill.
        seg, _, _ = _make_segments()
        seg.values[0, 0, 0] = np.nan
        assert np.isnan(seg.as_array()[0, 0, 0])

    def test_is_float32(self):
        seg, _, _ = _make_segments()
        assert seg.as_array().dtype == np.float32
