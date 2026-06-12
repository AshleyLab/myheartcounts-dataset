"""Unit tests for the user-macro reducer in bootstrap_skill_rank.

The reducer aggregates per-(user, channel) sums into per-channel error E
under the "users-then-channels" framing: per-task E is the macroaverage
over users (rather than the cell-pooled micro mean of the prior
implementation).
"""

from __future__ import annotations

import numpy as np

from imputation_evaluation.evaluation.bootstrap_skill_rank import (
    CellStats,
    _per_method_cell_errors,
    _per_user_auc_from_cell_stats,
)


N_CHANNELS = 19
CONTINUOUS_CH = 0  # ch_0 is continuous
BINARY_CH = 7  # ch_7 is binary (sleep:asleep)


def _make_cell_stats(
    n_users: int,
    n_u_ch: np.ndarray | None = None,
    sse_u_ch: np.ndarray | None = None,
    has_data_channels: list[int] | None = None,
) -> CellStats:
    """Build a CellStats with the given (U, C) sums; binary channels empty."""
    if n_u_ch is None:
        n_u_ch = np.zeros((n_users, N_CHANNELS), dtype=np.int64)
    if sse_u_ch is None:
        sse_u_ch = np.zeros((n_users, N_CHANNELS), dtype=np.float64)
    has_data = np.zeros(N_CHANNELS, dtype=bool)
    for ch in has_data_channels or []:
        has_data[ch] = True
    return CellStats(
        user_ids=[f"u{i}" for i in range(n_users)],
        n=n_u_ch,
        sse=sse_u_ch,
        sae=np.zeros_like(sse_u_ch),
        tp=np.zeros((n_users, N_CHANNELS), dtype=np.int64),
        tn=np.zeros((n_users, N_CHANNELS), dtype=np.int64),
        fp=np.zeros((n_users, N_CHANNELS), dtype=np.int64),
        fn=np.zeros((n_users, N_CHANNELS), dtype=np.int64),
        has_data=has_data,
        binary_rows={},
    )


class TestUserMacroContinuousE:
    def test_balanced_data_matches_cell_pooled(self):
        """When each user contributes equally, user-macro == cell-pooled."""
        # 3 users, each with N=100 cells in channel 0 and SSE=100 (so per-user
        # RMSE = sqrt(100/100) = 1.0 for everyone).
        n_users = 3
        n_u_ch = np.zeros((n_users, N_CHANNELS), dtype=np.int64)
        sse_u_ch = np.zeros((n_users, N_CHANNELS), dtype=np.float64)
        for u in range(n_users):
            n_u_ch[u, CONTINUOUS_CH] = 100
            sse_u_ch[u, CONTINUOUS_CH] = 100.0
        cs = _make_cell_stats(n_users, n_u_ch, sse_u_ch, [CONTINUOUS_CH])

        boot_idx = np.array([[0, 1, 2]])  # one draw, no resampling
        E = _per_method_cell_errors(cs, boot_idx, channel_stds=np.ones(N_CHANNELS), include_auc=False)
        # Expected: nanmean of [1.0, 1.0, 1.0] = 1.0.
        np.testing.assert_allclose(E[0, CONTINUOUS_CH], 1.0, rtol=1e-9)

    def test_heavy_user_does_not_dominate(self):
        """Under user-macro, a heavy-cell-count user has the same weight as a light one."""
        # User 0: 10,000 cells, all with err 0.1 → per-user RMSE = 0.1.
        # User 1: 10 cells, all with err 10.0 → per-user RMSE = 10.0.
        # Cell-micro pool: SSE = 10000*0.01 + 10*100 = 100+1000 = 1100; N=10010.
        #   pooled RMSE = sqrt(1100 / 10010) ≈ 0.331. Dominated by user 0.
        # User-macro: mean(0.1, 10.0) = 5.05. Both users counted equally.
        n_users = 2
        n_u_ch = np.zeros((n_users, N_CHANNELS), dtype=np.int64)
        sse_u_ch = np.zeros((n_users, N_CHANNELS), dtype=np.float64)
        n_u_ch[0, CONTINUOUS_CH] = 10000
        sse_u_ch[0, CONTINUOUS_CH] = 100.0  # → RMSE 0.1
        n_u_ch[1, CONTINUOUS_CH] = 10
        sse_u_ch[1, CONTINUOUS_CH] = 1000.0  # → RMSE 10.0
        cs = _make_cell_stats(n_users, n_u_ch, sse_u_ch, [CONTINUOUS_CH])

        boot_idx = np.array([[0, 1]])
        E = _per_method_cell_errors(cs, boot_idx, channel_stds=np.ones(N_CHANNELS), include_auc=False)
        np.testing.assert_allclose(E[0, CONTINUOUS_CH], 5.05, rtol=1e-9)

    def test_resampled_users_count_with_multiplicity(self):
        """Cluster bootstrap: a user resampled twice contributes twice to the mean."""
        n_users = 2
        n_u_ch = np.zeros((n_users, N_CHANNELS), dtype=np.int64)
        sse_u_ch = np.zeros((n_users, N_CHANNELS), dtype=np.float64)
        n_u_ch[0, CONTINUOUS_CH] = 100
        sse_u_ch[0, CONTINUOUS_CH] = 100.0  # RMSE 1.0
        n_u_ch[1, CONTINUOUS_CH] = 100
        sse_u_ch[1, CONTINUOUS_CH] = 400.0  # RMSE 2.0
        cs = _make_cell_stats(n_users, n_u_ch, sse_u_ch, [CONTINUOUS_CH])

        # Two draws: even balance, then user 0 picked twice.
        boot_idx = np.array([[0, 1], [0, 0]])
        E = _per_method_cell_errors(cs, boot_idx, channel_stds=np.ones(N_CHANNELS), include_auc=False)
        np.testing.assert_allclose(E[0, CONTINUOUS_CH], 1.5, rtol=1e-9)  # mean(1.0, 2.0)
        np.testing.assert_allclose(E[1, CONTINUOUS_CH], 1.0, rtol=1e-9)  # mean(1.0, 1.0)

    def test_zero_data_user_drops_via_nanmean(self):
        """User with N=0 for a channel contributes NaN, drops from the mean."""
        n_users = 3
        n_u_ch = np.zeros((n_users, N_CHANNELS), dtype=np.int64)
        sse_u_ch = np.zeros((n_users, N_CHANNELS), dtype=np.float64)
        # User 0 has data; users 1 and 2 don't.
        n_u_ch[0, CONTINUOUS_CH] = 100
        sse_u_ch[0, CONTINUOUS_CH] = 100.0  # RMSE 1.0
        cs = _make_cell_stats(n_users, n_u_ch, sse_u_ch, [CONTINUOUS_CH])

        boot_idx = np.array([[0, 1, 2]])
        E = _per_method_cell_errors(cs, boot_idx, channel_stds=np.ones(N_CHANNELS), include_auc=False)
        np.testing.assert_allclose(E[0, CONTINUOUS_CH], 1.0, rtol=1e-9)


class TestUserMacroBinaryE:
    def test_per_user_auc_drops_single_class_users(self):
        """Users with only one class for a channel contribute NaN AUC."""
        from imputation_evaluation.evaluation.bootstrap_skill_rank import BinaryRows

        # Three users for ch_7 (binary):
        # u0 has [0, 0, 1, 1] gt with discriminating pred → AUC well-defined.
        # u1 has [1, 1, 1] gt (all positive) → single-class → NaN.
        # u2 has [0, 0] gt (all negative) → single-class → NaN.
        cs = _make_cell_stats(3, has_data_channels=[BINARY_CH])
        cs.binary_rows[BINARY_CH] = BinaryRows(
            gt=np.array([False, False, True, True, True, True, True, False, False]),
            pred=np.array([0.1, 0.2, 0.9, 0.8, 0.7, 0.8, 0.9, 0.1, 0.2], dtype=np.float32),
            u_rows=np.array([0, 0, 0, 0, 1, 1, 1, 2, 2], dtype=np.int64),
        )

        per_user_auc = _per_user_auc_from_cell_stats(cs, N_CHANNELS)
        assert np.isfinite(per_user_auc[0, BINARY_CH])  # well-defined
        assert np.isnan(per_user_auc[1, BINARY_CH])  # single positive class
        assert np.isnan(per_user_auc[2, BINARY_CH])  # single negative class
        # User 0's AUC: predictions perfectly separate (0.1/0.2 < 0.7/0.8/0.9).
        np.testing.assert_allclose(per_user_auc[0, BINARY_CH], 1.0, rtol=1e-9)

    def test_user_macro_binary_e_excludes_single_class_users(self):
        """Single-class (user, channel) AUC drops from the per-task mean."""
        from imputation_evaluation.evaluation.bootstrap_skill_rank import BinaryRows

        cs = _make_cell_stats(3, has_data_channels=[BINARY_CH])
        cs.binary_rows[BINARY_CH] = BinaryRows(
            gt=np.array([False, True, False, True, True, True]),
            pred=np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.8], dtype=np.float32),
            u_rows=np.array([0, 0, 1, 1, 2, 2], dtype=np.int64),  # u2 all-positive
        )
        boot_idx = np.array([[0, 1, 2]])
        E = _per_method_cell_errors(
            cs, boot_idx, channel_stds=np.ones(N_CHANNELS), include_auc=True
        )
        # Expected: per-user 1-AUC for u0 and u1 (both 0.0; perfect separation),
        # u2 dropped → mean(0.0, 0.0) = 0.0 → E = 1 - 0.0 = ... wait,
        # E = 1 - mean(AUC) where mean(AUC) = mean(1.0, 1.0) = 1.0 → E = 0.0.
        np.testing.assert_allclose(E[0, BINARY_CH], 0.0, atol=1e-9)
