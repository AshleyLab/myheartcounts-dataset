"""Tests for participant-level bootstrap of imputation pair metrics."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data.processing.hf_config import N_CHANNELS
from imputation_evaluation.evaluation.bootstrap import (
    bootstrap_pairs_dir,
    bootstrap_split,
    compute_user_sufficient_stats,
)
from imputation_evaluation.evaluation.pair_aggregator import aggregate_pairs


def _write_synthetic_pairs(tmp_path, rng_seed: int = 0):
    """Write a tiny synthetic pairs/manifest tree.

    5 users x 4 samples each (20 samples), 8 timesteps. Channel 0 (continuous):
    pred = gt + N(0, sigma_u**2) where sigma_u differs per user. Channel 7
    (binary): predicted probability is logistic of a linear function of gt.
    """
    rng = np.random.default_rng(rng_seed)
    n_users = 5
    samples_per_user = 4
    T = 8

    user_ids = [f"u{i}" for i in range(n_users)]
    sigmas = np.linspace(0.5, 2.0, n_users)  # heterogeneous noise per user

    sample_records = []
    cont_rows: list[dict] = []
    bin_rows: list[dict] = []

    sidx = 0
    for u, uid in enumerate(user_ids):
        for s in range(samples_per_user):
            gt_cont = rng.normal(0, 1, size=T).astype(np.float32)
            pred_cont = gt_cont + rng.normal(0, sigmas[u], size=T).astype(np.float32)
            for t in range(T):
                cont_rows.append({
                    "sample_idx": sidx, "timestep": t,
                    "gt": float(gt_cont[t]), "pred": float(pred_cont[t]),
                })
            gt_bin = rng.integers(0, 2, size=T).astype(bool)
            pred_bin = np.where(
                gt_bin,
                rng.uniform(0.30, 0.90, size=T),
                rng.uniform(0.10, 0.70, size=T),
            ).astype(np.float32)
            for t in range(T):
                bin_rows.append({
                    "sample_idx": sidx, "timestep": t,
                    "gt": bool(gt_bin[t]), "pred": float(pred_bin[t]),
                })
            sample_records.append({
                "sample_idx": sidx,
                "user_id": uid,
                "date": f"2024-01-{sidx + 1:02d}",
            })
            sidx += 1

    manifest_tbl = pa.table({
        "sample_idx": pa.array([r["sample_idx"] for r in sample_records], type=pa.int32()),
        "user_id":    pa.array([r["user_id"] for r in sample_records], type=pa.utf8()),
        "date":       pa.array([r["date"] for r in sample_records], type=pa.utf8()),
    })
    pq.write_table(manifest_tbl, tmp_path / "manifest_test.parquet")

    split_dir = tmp_path / "scenarioA" / "test"
    split_dir.mkdir(parents=True)

    cont_tbl = pa.table({
        "sample_idx": pa.array([r["sample_idx"] for r in cont_rows], type=pa.int32()),
        "timestep":   pa.array([r["timestep"] for r in cont_rows], type=pa.int16()),
        "gt":         pa.array([r["gt"] for r in cont_rows], type=pa.float16()),
        "pred":       pa.array([r["pred"] for r in cont_rows], type=pa.float16()),
    })
    pq.write_table(cont_tbl, split_dir / "pairs_ch00.parquet")

    bin_tbl = pa.table({
        "sample_idx": pa.array([r["sample_idx"] for r in bin_rows], type=pa.int32()),
        "timestep":   pa.array([r["timestep"] for r in bin_rows], type=pa.int16()),
        "gt":         pa.array([r["gt"] for r in bin_rows], type=pa.bool_()),
        "pred":       pa.array([r["pred"] for r in bin_rows], type=pa.float16()),
    })
    pq.write_table(bin_tbl, split_dir / "pairs_ch07.parquet")

    stds = np.ones(N_CHANNELS, dtype=np.float64)
    np.save(tmp_path / "channel_stds.npy", stds)

    return tmp_path, manifest_tbl, split_dir, stds


def test_sufficient_stats_match_aggregator_point(tmp_path):
    """The bootstrap point estimate should match aggregate_pairs() exactly.

    Both sides are pinned to ``aggregation="cell_micro"`` here: ``bootstrap.py``'s
    participant-cluster point estimate pools per-user (SSE, SAE, N) sums before
    dividing (cell-micro), so this parity is meaningful only against the
    cell-micro path of ``aggregate_pairs``. The default is now ``"user_macro"``
    (matching the bootstrap-skill-rank leaderboard estimand); a separate
    ``test_live_equals_bootstrap_identity`` (see ``test_skill_score_parity``)
    pins that the user-macro live path equals the
    ``bootstrap_skill_rank`` identity-draw point.
    """
    pairs_dir, manifest, split_dir, stds = _write_synthetic_pairs(tmp_path, rng_seed=1)

    point = aggregate_pairs(split_dir, stds, aggregation="cell_micro")
    res = bootstrap_split(split_dir, manifest, stds, n_boot=10, seed=0, include_auc=True)

    assert res["per_channel"]["ch_0"]["rmse"]["point"] == pytest.approx(
        point["per_channel"]["ch_0"]["rmse"], rel=1e-5
    )
    assert res["per_channel"]["ch_0"]["mae"]["point"] == pytest.approx(
        point["per_channel"]["ch_0"]["mae"], rel=1e-5
    )
    assert res["per_channel"]["ch_7"]["balanced_accuracy"]["point"] == pytest.approx(
        point["per_channel"]["ch_7"]["balanced_accuracy"], rel=1e-5
    )
    assert res["per_channel"]["ch_7"]["roc_auc"]["point"] == pytest.approx(
        point["per_channel"]["ch_7"]["roc_auc"], rel=1e-5
    )


def test_bootstrap_returns_se_and_ci(tmp_path):
    """Bootstrap returns finite SE and CI fields for both RMSE and AUC."""
    pairs_dir, manifest, split_dir, stds = _write_synthetic_pairs(tmp_path, rng_seed=2)
    res = bootstrap_split(split_dir, manifest, stds, n_boot=200, seed=42, include_auc=True)
    rmse = res["per_channel"]["ch_0"]["rmse"]
    for k in ("point", "bootstrap_mean", "bootstrap_se", "ci_lo", "ci_hi", "n_valid_boot"):
        assert k in rmse, f"missing {k}"
    assert rmse["bootstrap_se"] > 0
    assert rmse["ci_lo"] < rmse["ci_hi"]
    assert rmse["ci_lo"] <= rmse["bootstrap_mean"] <= rmse["ci_hi"]
    assert rmse["n_valid_boot"] <= 200

    auc = res["per_channel"]["ch_7"]["roc_auc"]
    assert auc["bootstrap_se"] > 0
    assert auc["ci_lo"] < auc["ci_hi"]


def test_user_sufficient_stats_shapes(tmp_path):
    """User-level sufficient statistics arrays have the expected shapes/sums."""
    _, manifest, split_dir, stds = _write_synthetic_pairs(tmp_path, rng_seed=3)
    stats = compute_user_sufficient_stats(split_dir, manifest, n_channels=N_CHANNELS)
    assert len(stats.user_ids) == 5
    assert stats.n.shape == (5, N_CHANNELS)
    assert stats.has_data[0] and stats.has_data[7]
    # Each user: 4 samples * 8 timesteps = 32 masked positions per channel
    assert (stats.n[:, 0] == 32).all()
    total = stats.tp[:, 7] + stats.tn[:, 7] + stats.fp[:, 7] + stats.fn[:, 7]
    assert (total == 32).all()


def test_bootstrap_no_auc_skips_auc(tmp_path):
    """When include_auc=False, AUC fields are absent."""
    _, manifest, split_dir, stds = _write_synthetic_pairs(tmp_path, rng_seed=4)
    res = bootstrap_split(split_dir, manifest, stds, n_boot=20, seed=0, include_auc=False)
    assert "roc_auc" not in res["per_channel"]["ch_7"]
    assert "macro_roc_auc" not in res["binary"]


def test_aggregate_macro_metrics_present(tmp_path):
    """Macro-aggregate continuous and binary metrics are populated."""
    _, manifest, split_dir, stds = _write_synthetic_pairs(tmp_path, rng_seed=5)
    res = bootstrap_split(split_dir, manifest, stds, n_boot=50, seed=0, include_auc=True)
    mean_rmse = res["continuous"]["mean_rmse"]
    nrmse = res["continuous"]["mean_normalized_rmse"]
    for k in ("point", "bootstrap_mean", "bootstrap_se", "ci_lo", "ci_hi", "n_valid_boot"):
        assert k in mean_rmse
        assert k in nrmse
    # Stds are 1.0 in the fixture, so normalized == raw.
    assert mean_rmse["point"] == pytest.approx(nrmse["point"], rel=1e-6)
    assert res["continuous"]["n_channels"] == 1
    assert res["binary"]["n_channels"] == 1


def test_bootstrap_pairs_dir_wrapper(tmp_path):
    """``bootstrap_pairs_dir`` auto-discovers scenarios + splits and uses channel_stds.npy."""
    _write_synthetic_pairs(tmp_path, rng_seed=6)
    out = bootstrap_pairs_dir(tmp_path, splits=("test",), n_boot=10, seed=0, include_auc=False)
    assert "scenarioA" in out["scenarios"]
    assert "test" in out["scenarios"]["scenarioA"]
    assert out["config"]["n_boot"] == 10
    assert out["config"]["include_auc"] is False
    rmse = out["scenarios"]["scenarioA"]["test"]["per_channel"]["ch_0"]["rmse"]
    assert "ci_lo" in rmse and "ci_hi" in rmse


def _row_bootstrap_rmse_se(split_dir, n_boot: int, seed: int) -> float:
    """Naive row-level bootstrap SE for RMSE on ch 0 — for the cluster-vs-row test."""
    tbl = pq.read_table(split_dir / "pairs_ch00.parquet", columns=["gt", "pred"])
    gt = tbl.column("gt").to_numpy().astype(np.float64)
    pred = tbl.column("pred").to_numpy().astype(np.float64)
    err2 = (pred - gt) ** 2
    rng = np.random.default_rng(seed)
    n = len(err2)
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        out[b] = float(np.sqrt(err2[idx].mean()))
    return float(np.std(out, ddof=1))


def _write_two_user_skewed_pairs(tmp_path):
    """One user with high-error rows, one with tiny-error rows. Equal row counts.

    Used to show that cluster bootstrap SE meaningfully exceeds row-level SE
    when within-user error is consistent but between-user error differs.
    """
    n_per_user = 64
    rng = np.random.default_rng(0)
    rows = []
    manifest_records = []
    for uid_idx, (uid, err_scale) in enumerate([("uA", 5.0), ("uB", 0.1)]):
        for s in range(n_per_user):
            sidx = uid_idx * n_per_user + s
            gt = float(rng.normal())
            pred = gt + float(rng.normal() * err_scale)
            rows.append({"sample_idx": sidx, "timestep": 0, "gt": gt, "pred": pred})
            manifest_records.append({"sample_idx": sidx, "user_id": uid, "date": "2024-01-01"})

    manifest_tbl = pa.table({
        "sample_idx": pa.array([r["sample_idx"] for r in manifest_records], type=pa.int32()),
        "user_id":    pa.array([r["user_id"] for r in manifest_records], type=pa.utf8()),
        "date":       pa.array([r["date"] for r in manifest_records], type=pa.utf8()),
    })
    pq.write_table(manifest_tbl, tmp_path / "manifest_test.parquet")

    split_dir = tmp_path / "scenarioA" / "test"
    split_dir.mkdir(parents=True)
    cont_tbl = pa.table({
        "sample_idx": pa.array([r["sample_idx"] for r in rows], type=pa.int32()),
        "timestep":   pa.array([r["timestep"] for r in rows], type=pa.int16()),
        "gt":         pa.array([r["gt"] for r in rows], type=pa.float16()),
        "pred":       pa.array([r["pred"] for r in rows], type=pa.float16()),
    })
    pq.write_table(cont_tbl, split_dir / "pairs_ch00.parquet")

    stds = np.ones(N_CHANNELS, dtype=np.float64)
    np.save(tmp_path / "channel_stds.npy", stds)
    return manifest_tbl, split_dir, stds


def test_cluster_se_exceeds_row_se_for_skewed_clusters(tmp_path):
    """Cluster bootstrap captures between-user variance the row bootstrap misses.

    With two users whose per-row errors differ by 50x, the cluster bootstrap
    SE should be much larger than a naive row-level bootstrap SE.
    """
    manifest, split_dir, stds = _write_two_user_skewed_pairs(tmp_path)
    cluster_res = bootstrap_split(
        split_dir, manifest, stds, n_boot=500, seed=0, include_auc=False
    )
    cluster_se = cluster_res["per_channel"]["ch_0"]["rmse"]["bootstrap_se"]
    row_se = _row_bootstrap_rmse_se(split_dir, n_boot=500, seed=0)
    assert cluster_se > 2.0 * row_se, (
        f"cluster SE ({cluster_se:.4f}) should be > 2x row SE ({row_se:.4f}); "
        "the cluster bootstrap is supposed to capture between-user variance."
    )
