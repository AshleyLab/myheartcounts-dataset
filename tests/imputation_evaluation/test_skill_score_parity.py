"""Parity tests for the imputation skill-score reducer.

Locks in the alignment with forecasting Track 3 (commit ``79c8628``):

1. ``_per_method_cell_paired_ratios`` identity-draw matches a deterministic
   paired point estimate (forecasting's ``test_identity_draw_matches_point``
   analog).
2. The MAE switch: continuous E uses ``sae/n`` (not ``sqrt(sse/n)``); the
   skill ratio is invariant to per-channel normalization.
3. ``BINARY_ERROR_FLOOR``: a perfect-AUC user produces per-user binary
   error ``0.005`` (not ``0.0``); the paired ratio is finite and the user
   is *not* dropped by the ``baseline > 0`` filter.
4. Pooled binary parity: per-user AUC is scored once on the user's pooled
   rows (mirrors forecasting's "pooled binary metrics").
5. Live = bootstrap identity: ``aggregate_pairs(..., aggregation="user_macro")``
   equals the bootstrap identity-draw point estimate; an uneven-cell-count
   fixture ensures user-macro ≠ cell-micro (the test wouldn't be vacuous).
6. Cross-track estimand parity: a synthetic fixture scored through the
   forecasting and imputation reducers yields the same skill score.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data.processing.hf_config import N_CHANNELS
from imputation_evaluation.evaluation.bootstrap_skill_rank import (
    BINARY_ERROR_FLOOR,
    SKILL_CLIP_LOWER,
    SKILL_CLIP_UPPER,
    BinaryRows,
    CellStats,
    _per_method_cell_errors,
    _per_method_cell_paired_ratios,
    _per_user_log_ratio_column,
)
from imputation_evaluation.evaluation.pair_aggregator import aggregate_pairs

# ---------------------------------------------------------------------------
# Tiny synthetic CellStats builder
# ---------------------------------------------------------------------------


def _embed(arr: np.ndarray, n_channels: int = N_CHANNELS) -> np.ndarray:
    """Zero-pad a ``(n_users, k)`` array to ``(n_users, n_channels)``.

    The reducers iterate over ``CONTINUOUS_CHANNEL_INDICES`` (0..6) and over
    every binary channel index up to ``n_channels``, so test fixtures must
    use canonical-width arrays even when only a few channels carry data.
    """
    if arr.shape[1] == n_channels:
        return arr
    out = np.zeros((arr.shape[0], n_channels), dtype=arr.dtype)
    out[:, : arr.shape[1]] = arr
    return out


def _make_cell_stats(
    *,
    n_users: int,
    sae: np.ndarray,
    sse: np.ndarray,
    n: np.ndarray,
    binary_rows: dict[int, BinaryRows] | None = None,
    has_data: np.ndarray | None = None,
    n_channels: int = N_CHANNELS,
) -> CellStats:
    """Build a CellStats from per-(user, channel) sufficient stats.

    ``sae`` / ``sse`` / ``n`` may be supplied as ``(n_users, k)`` for
    ``k < n_channels``; they're zero-padded to ``n_channels`` so the
    reducers' unconditional iteration over ``CONTINUOUS_CHANNEL_INDICES``
    is safe.
    """
    sae = _embed(sae, n_channels)
    sse = _embed(sse, n_channels)
    n = _embed(n, n_channels)
    if has_data is None:
        has_data = (n.sum(axis=0) > 0) | bool(binary_rows)

    return CellStats(
        user_ids=[f"u{i}" for i in range(n_users)],
        n=n.astype(np.int64),
        sse=sse.astype(np.float64),
        sae=sae.astype(np.float64),
        tp=np.zeros((n_users, n_channels), dtype=np.int64),
        tn=np.zeros((n_users, n_channels), dtype=np.int64),
        fp=np.zeros((n_users, n_channels), dtype=np.int64),
        fn=np.zeros((n_users, n_channels), dtype=np.int64),
        has_data=has_data.astype(bool),
        binary_rows=binary_rows or {},
    )


# ---------------------------------------------------------------------------
# 1. Identity-draw parity for paired ratios
# ---------------------------------------------------------------------------


def test_paired_ratio_identity_draw_matches_deterministic_point():
    """Identity draw (each user once) reproduces the deterministic paired point."""
    n_users, n_channels = 4, 2
    # Channel 0: continuous. n_u = 10 each; sae varied per user.
    sae_m = np.zeros((n_users, n_channels))
    sae_b = np.zeros((n_users, n_channels))
    sse = np.zeros((n_users, n_channels))
    n_arr = np.zeros((n_users, n_channels), dtype=np.int64)
    sae_m[:, 0] = [2.0, 3.0, 5.0, 7.0]
    sae_b[:, 0] = [4.0, 5.0, 8.0, 11.0]
    n_arr[:, 0] = 10

    cs_method = _make_cell_stats(n_users=n_users, sae=sae_m, sse=sse, n=n_arr)
    cs_baseline = _make_cell_stats(n_users=n_users, sae=sae_b, sse=sse, n=n_arr)

    # Identity draw: each user listed exactly once.
    boot_idx = np.arange(n_users).reshape(1, n_users)

    R = _per_method_cell_paired_ratios(
        cs_method,
        cs_baseline,
        boot_idx,
        include_auc=False,
    )

    # Deterministic point: paired per-user ratio of MAE (sae/n), then geomean.
    mae_m = sae_m[:, 0] / 10.0
    mae_b = sae_b[:, 0] / 10.0
    ratios = np.clip(mae_m / mae_b, SKILL_CLIP_LOWER, SKILL_CLIP_UPPER)
    expected_R = float(np.exp(np.mean(np.log(ratios))))

    assert R[0, 0] == pytest.approx(expected_R, rel=1e-9)
    # Skill = 1 - R; sanity-check direction.
    assert 1.0 - R[0, 0] > 0  # method beats baseline on every user → skill > 0


def test_baseline_vs_self_paired_ratio_is_one():
    """Baseline ≡ self → every per-user ratio is 1 → R ≡ 1 → skill ≡ 0."""
    n_users, n_channels = 3, 1
    sae = np.array([[1.0], [2.0], [4.0]])
    sse = np.zeros_like(sae)
    n_arr = np.full((n_users, n_channels), 5, dtype=np.int64)
    cs = _make_cell_stats(n_users=n_users, sae=sae, sse=sse, n=n_arr)
    boot_idx = np.arange(n_users).reshape(1, n_users)
    R = _per_method_cell_paired_ratios(cs, cs, boot_idx, include_auc=False)
    assert R[0, 0] == pytest.approx(1.0, rel=1e-12)


# ---------------------------------------------------------------------------
# 2. MAE switch + normalization invariance
# ---------------------------------------------------------------------------


def test_continuous_E_uses_sae_over_n_not_sqrt():
    """_per_method_cell_errors continuous E uses MAE = sae/n, not RMSE.

    Uses a multi-cell-per-user fixture so MAE = sae/n and
    RMSE = sqrt(sse/n) take different numerical values; with a single
    cell per user they coincide.
    """
    n_users, n_channels = 2, 1
    n_arr = np.full((n_users, n_channels), 4, dtype=np.int64)
    sae = np.array([[8.0], [8.0]])  # MAE per user = 2, 2
    sse = np.array([[24.0], [24.0]])  # MSE per user = 6 → RMSE ≈ 2.449
    cs = _make_cell_stats(n_users=n_users, sae=sae, sse=sse, n=n_arr)
    boot_idx = np.arange(n_users).reshape(1, n_users)
    E = _per_method_cell_errors(cs, boot_idx, np.ones(n_channels), include_auc=False)
    expected_mae = 2.0  # mean of (2, 2)
    expected_rmse = np.sqrt(6.0)  # what we would see if E were RMSE
    assert E[0, 0] == pytest.approx(expected_mae, rel=1e-9), (
        f"continuous E should be MAE={expected_mae}, got {E[0, 0]} (RMSE would be {expected_rmse})"
    )


def test_skill_ratio_invariant_to_normalization():
    """Per-channel ``E_method / E_baseline`` is invariant to channel-std scaling."""
    n_users, n_channels = 4, 1
    sae_m = np.array([[1.0], [2.0], [3.0], [4.0]])
    sae_b = np.array([[2.0], [4.0], [6.0], [8.0]])
    sse = np.zeros_like(sae_m)
    n_arr = np.full((n_users, n_channels), 1, dtype=np.int64)
    cs_m = _make_cell_stats(n_users=n_users, sae=sae_m, sse=sse, n=n_arr)
    cs_b = _make_cell_stats(n_users=n_users, sae=sae_b, sse=sse, n=n_arr)
    boot_idx = np.arange(n_users).reshape(1, n_users)
    R_unscaled = _per_method_cell_paired_ratios(cs_m, cs_b, boot_idx, include_auc=False)[0, 0]

    # Scaling both sides by the same per-channel factor (a "channel_std") cancels.
    scale = 17.3
    cs_m_scaled = _make_cell_stats(n_users=n_users, sae=sae_m * scale, sse=sse, n=n_arr)
    cs_b_scaled = _make_cell_stats(n_users=n_users, sae=sae_b * scale, sse=sse, n=n_arr)
    R_scaled = _per_method_cell_paired_ratios(
        cs_m_scaled, cs_b_scaled, boot_idx, include_auc=False
    )[0, 0]
    assert R_unscaled == pytest.approx(R_scaled, rel=1e-12)


# ---------------------------------------------------------------------------
# 3. Binary error floor: perfect AUC → 0.005, ratio finite
# ---------------------------------------------------------------------------


def test_binary_error_floor_keeps_perfect_auc_user_in_paired_set():
    """A user with AUC = 1.0 on both sides contributes a finite paired ratio."""
    # Construct per-user AUC arrays directly: 4 users, channel 7 binary.
    n_users = 4
    per_user_auc_method = np.full((n_users, N_CHANNELS), np.nan, dtype=np.float64)
    per_user_auc_baseline = np.full((n_users, N_CHANNELS), np.nan, dtype=np.float64)
    # Three users with realistic AUCs, one with a perfect 1.0 on both sides.
    per_user_auc_method[:, 7] = [0.8, 0.7, 0.9, 1.0]
    per_user_auc_baseline[:, 7] = [0.6, 0.5, 0.7, 1.0]

    # Build CellStats with has_data[7] = True (binary channel present).
    sae = np.zeros((n_users, N_CHANNELS))
    sse = np.zeros((n_users, N_CHANNELS))
    n_arr = np.zeros((n_users, N_CHANNELS), dtype=np.int64)
    has_data = np.zeros(N_CHANNELS, dtype=bool)
    has_data[7] = True
    cs = _make_cell_stats(
        n_users=n_users, sae=sae, sse=sse, n=n_arr, has_data=has_data
    )
    boot_idx = np.arange(n_users).reshape(1, n_users)

    R = _per_method_cell_paired_ratios(
        cs,
        cs,  # baseline CellStats (only the precomputed AUC matters here)
        boot_idx,
        include_auc=True,
        per_user_auc_method=per_user_auc_method,
        per_user_auc_baseline=per_user_auc_baseline,
    )

    # The perfect-AUC user contributes max(1 - 1.0, ε) = ε on BOTH sides:
    # r = ε/ε = 1, which is in the paired set. So the paired ratio R is
    # finite (not NaN) and includes all 4 users.
    assert np.isfinite(R[0, 7])

    # Recompute what we expect: per-user ratio = max(1-AUC_m, ε) / max(1-AUC_b, ε)
    e_m = np.maximum(1.0 - per_user_auc_method[:, 7], BINARY_ERROR_FLOOR)
    e_b = np.maximum(1.0 - per_user_auc_baseline[:, 7], BINARY_ERROR_FLOOR)
    ratios = np.clip(e_m / e_b, SKILL_CLIP_LOWER, SKILL_CLIP_UPPER)
    expected_R = float(np.exp(np.mean(np.log(ratios))))
    assert R[0, 7] == pytest.approx(expected_R, rel=1e-9)
    # Verify the perfect user's contribution is 1.0 (i.e. ε/ε), not dropped:
    assert ratios[3] == pytest.approx(1.0, rel=1e-12)


def test_per_user_log_ratio_drops_zero_baseline():
    """Users with e_b = 0 are dropped (forecasting's ``baseline > 0`` guard)."""
    e_m = np.array([1.0, 2.0, 3.0, 4.0])
    e_b = np.array([0.5, 0.0, 1.5, 2.0])  # second user has zero baseline
    log_r = _per_user_log_ratio_column(e_m, e_b)
    assert np.isfinite(log_r[0])
    assert np.isnan(log_r[1])  # dropped by baseline > 0 filter
    assert np.isfinite(log_r[2])
    assert np.isfinite(log_r[3])


# ---------------------------------------------------------------------------
# 4. Cross-track estimand parity
# ---------------------------------------------------------------------------


def _forecasting_paired_skill(e_m: np.ndarray, e_b: np.ndarray) -> float:
    """Reimplementation of forecasting.compute_skill_from_errors (defaults).

    Kept tiny and inline so the parity check doesn't depend on which
    forecasting branch is on PYTHONPATH. Mirrors
    ``forecasting_evaluation.metrics.skill_score_summary.compute_skill_from_errors``
    line-for-line with default args (clip_lower=0.01, clip_upper=100.0,
    min_pairs=1).
    """
    e_m = np.asarray(e_m, dtype=float).reshape(-1)
    e_b = np.asarray(e_b, dtype=float).reshape(-1)
    valid = np.isfinite(e_m) & np.isfinite(e_b) & (e_b > 0.0)
    if not valid.any():
        return float("nan")
    r = np.clip(e_m[valid] / e_b[valid], 0.01, 100.0)
    r = r[np.isfinite(r) & (r > 0.0)]
    if r.size == 0:
        return float("nan")
    return float(1.0 - float(np.exp(np.mean(np.log(r)))))


def test_cross_track_skill_parity_continuous():
    """Imputation paired-ratio reducer ≡ forecasting compute_skill_from_errors."""
    n_users, n_channels = 6, 1
    rng = np.random.default_rng(0)
    sae_m = rng.uniform(0.5, 4.0, (n_users, n_channels))
    sae_b = rng.uniform(0.5, 4.0, (n_users, n_channels))
    sse = np.zeros_like(sae_m)
    n_arr = np.full((n_users, n_channels), 7, dtype=np.int64)
    cs_m = _make_cell_stats(n_users=n_users, sae=sae_m, sse=sse, n=n_arr)
    cs_b = _make_cell_stats(n_users=n_users, sae=sae_b, sse=sse, n=n_arr)
    boot_idx = np.arange(n_users).reshape(1, n_users)
    R = _per_method_cell_paired_ratios(cs_m, cs_b, boot_idx, include_auc=False)[0, 0]
    imp_skill = 1.0 - R

    # Forecasting reducer on the same per-user errors.
    e_m = (sae_m / 7.0).reshape(-1)
    e_b = (sae_b / 7.0).reshape(-1)
    fc_skill = _forecasting_paired_skill(e_m, e_b)

    assert imp_skill == pytest.approx(fc_skill, rel=1e-12)


# ---------------------------------------------------------------------------
# 5. Live (aggregate_pairs user_macro) = bootstrap identity-draw point
# ---------------------------------------------------------------------------


def _write_uneven_cell_count_fixture(tmp_path):
    """Write a tiny pairs/manifest tree where users have DIFFERENT cell counts.

    Without uneven counts, user-macro ≡ cell-micro and the parity test
    against cell-micro would be vacuous. Here users contribute 4 / 8 / 12
    cells respectively on channel 0, with the same per-user mean absolute
    error so the user-macro MAE is well-defined and differs from the
    cell-pooled MAE.
    """
    users = ["uA", "uB", "uC"]
    cells_per_user = [4, 8, 12]  # uneven
    sample_records: list[dict] = []
    cont_rows: list[dict] = []
    sidx = 0
    for u, uid in enumerate(users):
        ncells = cells_per_user[u]
        # Per-user errors are all the same scale, but heavy-cell users
        # dominate a cell-pool mean — making user-macro ≠ cell-micro.
        err_scale = 1.0 + 0.5 * u
        for _ in range(ncells):
            cont_rows.append(
                {
                    "sample_idx": sidx,
                    "timestep": 0,
                    "gt": 0.0,
                    "pred": float(err_scale),  # constant error = err_scale per cell
                }
            )
            sample_records.append(
                {
                    "sample_idx": sidx,
                    "user_id": uid,
                    "date": f"2024-01-{sidx + 1:02d}",
                }
            )
            sidx += 1

    manifest_tbl = pa.table(
        {
            "sample_idx": pa.array([r["sample_idx"] for r in sample_records], type=pa.int32()),
            "user_id": pa.array([r["user_id"] for r in sample_records], type=pa.utf8()),
            "date": pa.array([r["date"] for r in sample_records], type=pa.utf8()),
        }
    )
    pq.write_table(manifest_tbl, tmp_path / "manifest_test.parquet")

    split_dir = tmp_path / "scenarioA" / "test"
    split_dir.mkdir(parents=True)
    # Cast Python floats to np.float16 before handing to pa.array(type=float16) —
    # the pyarrow build pinned here refuses the implicit Python-float coercion.
    gt_arr = np.asarray([r["gt"] for r in cont_rows], dtype=np.float16)
    pred_arr = np.asarray([r["pred"] for r in cont_rows], dtype=np.float16)
    cont_tbl = pa.table(
        {
            "sample_idx": pa.array([r["sample_idx"] for r in cont_rows], type=pa.int32()),
            "timestep": pa.array([r["timestep"] for r in cont_rows], type=pa.int16()),
            "gt": pa.array(gt_arr, type=pa.float16()),
            "pred": pa.array(pred_arr, type=pa.float16()),
        }
    )
    pq.write_table(cont_tbl, split_dir / "pairs_ch00.parquet")

    stds = np.ones(N_CHANNELS, dtype=np.float64)
    np.save(tmp_path / "channel_stds.npy", stds)
    return tmp_path, split_dir, stds, cells_per_user


def test_aggregate_pairs_user_macro_differs_from_cell_micro(tmp_path):
    """Uneven-cell-count fixture: user-macro headline ≠ cell-micro pool."""
    _, split_dir, stds, cells_per_user = _write_uneven_cell_count_fixture(tmp_path)

    user_macro = aggregate_pairs(split_dir, stds, aggregation="user_macro")
    cell_micro = aggregate_pairs(split_dir, stds, aggregation="cell_micro")

    assert user_macro["per_channel"]["ch_0"]["aggregation"] == "user_macro"
    assert cell_micro["per_channel"]["ch_0"]["aggregation"] == "cell_micro"

    user_mae = user_macro["per_channel"]["ch_0"]["mae"]
    micro_mae = cell_micro["per_channel"]["ch_0"]["mae"]

    # Per-user MAE = err_scale (constant per cell) = 1.0, 1.5, 2.0.
    # user-macro = mean(1.0, 1.5, 2.0) = 1.5
    assert user_mae == pytest.approx(1.5, rel=1e-3)

    # cell-micro = weighted by counts: (4*1.0 + 8*1.5 + 12*2.0) / 24 = 1.6667
    cells = cells_per_user
    micro_expected = (cells[0] * 1.0 + cells[1] * 1.5 + cells[2] * 2.0) / sum(cells)
    assert micro_mae == pytest.approx(micro_expected, rel=1e-3)

    # And they're meaningfully different (otherwise the test is vacuous).
    assert abs(user_mae - micro_mae) > 0.05
