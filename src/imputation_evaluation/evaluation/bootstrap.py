"""Participant-level (cluster) bootstrap for imputation pair metrics.

Resamples users with replacement and recomputes metrics on the resampled rows.
RMSE / MSE / MAE / normalized variants / balanced accuracy are obtained from
per-user additive sufficient statistics in O(|U|) per iteration. ROC AUC is
non-decomposable so it uses cluster-weighted Mann-Whitney U via a one-time
global sort of predictions; per iteration we walk the sorted rows with per-user
multiplicities. Tied predictions get half-credit.

Adds one wrapper
(``bootstrap_pairs_dir``) that auto-discovers scenarios/splits + loads
``channel_stds.npy`` + per-split manifests so callers don't need to plumb
those arguments through the runner.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score

from data.processing.hf_config import CONTINUOUS_CHANNEL_INDICES, N_CHANNELS
from imputation_evaluation.evaluation.pair_writer import load_sample_manifest

logger = logging.getLogger(__name__)


def _channel_file(pairs_dir: Path, ch: int) -> Path:
    return pairs_dir / f"pairs_ch{ch:02d}.parquet"


@dataclass
class UserStats:
    """Per-user additive sufficient statistics for a single (scenario, split).

    Shapes are ``(n_users, n_channels)`` for the per-channel arrays. Channels
    that don't appear in the pairs (no rows) have all zeros and ``has_data=False``.
    """

    user_ids: list[str]  # length n_users; stable ordering
    user_idx: dict[str, int]  # user_id -> row in the arrays
    has_data: np.ndarray  # (C,) bool — channel present in pairs
    # Continuous (channels in CONTINUOUS_CHANNEL_INDICES)
    n: np.ndarray  # (n_users, C) int64
    sse: np.ndarray  # (n_users, C) float64
    sae: np.ndarray  # (n_users, C) float64
    # Binary (other channels), threshold 0.5
    tp: np.ndarray  # (n_users, C) int64
    tn: np.ndarray
    fp: np.ndarray
    fn: np.ndarray


def _build_sample_to_user(manifest) -> tuple[list[str], dict[str, int], np.ndarray]:
    """Return (user_ids, user_idx, sample_to_user_row).

    ``sample_to_user_row[s]`` gives the user-row index for ``sample_idx=s``.
    """
    sample_idx_arr = manifest.column("sample_idx").to_numpy()
    user_id_arr = manifest.column("user_id").to_pylist()

    user_ids: list[str] = []
    user_idx: dict[str, int] = {}
    for uid in user_id_arr:
        if uid not in user_idx:
            user_idx[uid] = len(user_ids)
            user_ids.append(uid)

    max_sidx = int(sample_idx_arr.max()) if len(sample_idx_arr) else -1
    sample_to_user_row = np.full(max_sidx + 1, -1, dtype=np.int64)
    for sidx, uid in zip(sample_idx_arr, user_id_arr):
        sample_to_user_row[int(sidx)] = user_idx[uid]
    return user_ids, user_idx, sample_to_user_row


def compute_user_sufficient_stats(
    pairs_dir: Path,
    manifest,
    n_channels: int = N_CHANNELS,
) -> UserStats:
    """Single streaming pass per channel; accumulates per-user additive stats."""
    pairs_dir = Path(pairs_dir)
    user_ids, user_idx, sample_to_user_row = _build_sample_to_user(manifest)
    U = len(user_ids)

    n = np.zeros((U, n_channels), dtype=np.int64)
    sse = np.zeros((U, n_channels), dtype=np.float64)
    sae = np.zeros((U, n_channels), dtype=np.float64)
    tp = np.zeros((U, n_channels), dtype=np.int64)
    tn = np.zeros((U, n_channels), dtype=np.int64)
    fp = np.zeros((U, n_channels), dtype=np.int64)
    fn = np.zeros((U, n_channels), dtype=np.int64)
    has_data = np.zeros(n_channels, dtype=bool)

    for ch in range(n_channels):
        ch_file = _channel_file(pairs_dir, ch)
        if not ch_file.exists():
            continue
        table = pq.read_table(ch_file, columns=["sample_idx", "gt", "pred"])
        if table.num_rows == 0:
            continue
        has_data[ch] = True

        sidx = table.column("sample_idx").to_numpy()
        u_rows = sample_to_user_row[sidx]
        if (u_rows < 0).any():
            missing = int((u_rows < 0).sum())
            raise ValueError(f"{ch_file.name}: {missing} rows have sample_idx not in manifest")

        if ch in CONTINUOUS_CHANNEL_INDICES:
            gt_ch = table.column("gt").to_numpy().astype(np.float32)
            pred_ch = table.column("pred").to_numpy().astype(np.float32)
            err = (pred_ch - gt_ch).astype(np.float64)
            np.add.at(n[:, ch], u_rows, 1)
            np.add.at(sse[:, ch], u_rows, err * err)
            np.add.at(sae[:, ch], u_rows, np.abs(err))
        else:
            gt_bool = table.column("gt").to_numpy().astype(bool)
            pred_ch = table.column("pred").to_numpy().astype(np.float32)
            pred_bool = pred_ch > 0.5
            tp_mask = gt_bool & pred_bool
            tn_mask = (~gt_bool) & (~pred_bool)
            fp_mask = (~gt_bool) & pred_bool
            fn_mask = gt_bool & (~pred_bool)
            np.add.at(tp[:, ch], u_rows, tp_mask.astype(np.int64))
            np.add.at(tn[:, ch], u_rows, tn_mask.astype(np.int64))
            np.add.at(fp[:, ch], u_rows, fp_mask.astype(np.int64))
            np.add.at(fn[:, ch], u_rows, fn_mask.astype(np.int64))

        del table

    return UserStats(
        user_ids=user_ids,
        user_idx=user_idx,
        has_data=has_data,
        n=n,
        sse=sse,
        sae=sae,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
    )


def _continuous_metrics_from_sums(
    N: np.ndarray, SSE: np.ndarray, SAE: np.ndarray, channel_stds: np.ndarray
) -> dict:
    """Compute RMSE/MSE/MAE + normalized variants from aggregated sums.

    Inputs are shape (B, C) (B = bootstrap iters or 1). Returns dict of (B, C)
    arrays; entries with N==0 are NaN.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        mse = np.where(N > 0, SSE / N, np.nan)
        mae = np.where(N > 0, SAE / N, np.nan)
        rmse = np.sqrt(mse)
        stds = np.asarray(channel_stds, dtype=np.float64).reshape(1, -1)
        safe_stds = np.where(stds > 0, stds, 1.0)
        nrmse = rmse / safe_stds
        nmse = mse / (safe_stds**2)
        nmae = mae / safe_stds
    return {"rmse": rmse, "mse": mse, "mae": mae, "nrmse": nrmse, "nmse": nmse, "nmae": nmae}


def _balanced_accuracy_from_confusion(
    TP: np.ndarray, TN: np.ndarray, FP: np.ndarray, FN: np.ndarray
) -> np.ndarray:
    """Compute balanced accuracy from confusion-matrix sums.

    Inputs shape (B, C). Returns (B, C); NaN where a class is empty (matches
    sklearn's behaviour on single-class draws being treated as undefined).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        n_pos = TP + FN
        n_neg = TN + FP
        sens = np.where(n_pos > 0, TP / n_pos, np.nan)
        spec = np.where(n_neg > 0, TN / n_neg, np.nan)
        single_class = (n_pos == 0) | (n_neg == 0)
        bal_acc = np.where(single_class, np.nan, (sens + spec) / 2.0)
    return bal_acc


def _summarize(values: np.ndarray, ci_level: float) -> dict:
    """Reduce a (B,) array of bootstrap statistic values to {mean, se, ci_lo, ci_hi, n_valid}.

    NaN values are dropped before computing summaries.
    """
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    n_valid = int(finite.size)
    if n_valid == 0:
        return {
            "bootstrap_mean": float("nan"),
            "bootstrap_se": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "n_valid_boot": 0,
        }
    alpha = 1.0 - ci_level
    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)
    return {
        "bootstrap_mean": float(np.mean(finite)),
        "bootstrap_se": float(np.std(finite, ddof=1)) if n_valid > 1 else 0.0,
        "ci_lo": float(np.percentile(finite, lo_q)),
        "ci_hi": float(np.percentile(finite, hi_q)),
        "n_valid_boot": n_valid,
    }


def _bootstrap_indices(n_users: int, n_boot: int, seed: int) -> np.ndarray:
    """Draw an (n_boot, n_users) matrix of user indices sampled with replacement."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_users, size=(n_boot, n_users), dtype=np.int64)


def _bootstrap_continuous_and_binary(
    stats: UserStats,
    boot_idx: np.ndarray,
    channel_stds: np.ndarray,
) -> dict:
    """Vectorised bootstrap over the additive sufficient stats.

    Returns dict: {metric_name: (B, C) array}.
    """
    n_boot, n_users = boot_idx.shape
    n_channels = stats.n.shape[1]

    n_arr = stats.n
    sse_arr = stats.sse
    sae_arr = stats.sae
    tp_arr = stats.tp
    tn_arr = stats.tn
    fp_arr = stats.fp
    fn_arr = stats.fn

    rmse_b = np.full((n_boot, n_channels), np.nan, dtype=np.float64)
    mse_b = np.full_like(rmse_b, np.nan)
    mae_b = np.full_like(rmse_b, np.nan)
    nrmse_b = np.full_like(rmse_b, np.nan)
    nmse_b = np.full_like(rmse_b, np.nan)
    nmae_b = np.full_like(rmse_b, np.nan)
    balacc_b = np.full_like(rmse_b, np.nan)

    # Process in chunks to bound peak memory: each chunk gathers a (chunk, n_users, C) view
    # via fancy indexing then sums along axis 1.
    bytes_per_elem = 8
    target_mem_bytes = 2 * 1024**3  # ~2 GB peak for the gathered arrays
    chunk = max(1, target_mem_bytes // (n_users * n_channels * bytes_per_elem))
    chunk = min(chunk, n_boot)

    for b0 in range(0, n_boot, chunk):
        b1 = min(b0 + chunk, n_boot)
        idx = boot_idx[b0:b1]  # (c, n_users)

        N_b = n_arr[idx].sum(axis=1)  # (c, C) int64
        SSE_b = sse_arr[idx].sum(axis=1)
        SAE_b = sae_arr[idx].sum(axis=1)
        cont = _continuous_metrics_from_sums(N_b, SSE_b, SAE_b, channel_stds)
        rmse_b[b0:b1] = cont["rmse"]
        mse_b[b0:b1] = cont["mse"]
        mae_b[b0:b1] = cont["mae"]
        nrmse_b[b0:b1] = cont["nrmse"]
        nmse_b[b0:b1] = cont["nmse"]
        nmae_b[b0:b1] = cont["nmae"]

        TP_b = tp_arr[idx].sum(axis=1)
        TN_b = tn_arr[idx].sum(axis=1)
        FP_b = fp_arr[idx].sum(axis=1)
        FN_b = fn_arr[idx].sum(axis=1)
        balacc_b[b0:b1] = _balanced_accuracy_from_confusion(TP_b, TN_b, FP_b, FN_b)

    cont_mask = np.zeros(n_channels, dtype=bool)
    cont_mask[CONTINUOUS_CHANNEL_INDICES] = True
    rmse_b[:, ~cont_mask] = np.nan
    mse_b[:, ~cont_mask] = np.nan
    mae_b[:, ~cont_mask] = np.nan
    nrmse_b[:, ~cont_mask] = np.nan
    nmse_b[:, ~cont_mask] = np.nan
    nmae_b[:, ~cont_mask] = np.nan
    balacc_b[:, cont_mask] = np.nan

    return {
        "rmse": rmse_b,
        "mse": mse_b,
        "mae": mae_b,
        "nrmse": nrmse_b,
        "nmse": nmse_b,
        "nmae": nmae_b,
        "balanced_accuracy": balacc_b,
    }


def _bootstrap_auc_one_channel(
    pairs_dir: Path,
    ch: int,
    sample_to_user_row: np.ndarray,
    n_users: int,
    boot_idx: np.ndarray,
) -> np.ndarray:
    """Cluster bootstrap of ROC AUC for a single binary channel.

    Uses Mann-Whitney U via a one-time global sort of predictions, then for each
    bootstrap iteration computes weighted U in O(N) by walking the globally sorted
    rows with per-user multiplicities. Tied predictions get half-credit, matching
    sklearn's ``roc_auc_score``.

    Returns (n_boot,) AUC values, NaN where the resample is single-class or the
    channel has no data.
    """
    ch_file = _channel_file(pairs_dir, ch)
    n_boot = boot_idx.shape[0]
    if not ch_file.exists():
        return np.full(n_boot, np.nan)
    table = pq.read_table(ch_file, columns=["sample_idx", "gt", "pred"])
    if table.num_rows == 0:
        return np.full(n_boot, np.nan)

    sidx = table.column("sample_idx").to_numpy()
    gt = table.column("gt").to_numpy().astype(bool)
    pred = table.column("pred").to_numpy().astype(np.float32)
    del table

    u_rows = sample_to_user_row[sidx]
    if (u_rows < 0).any():
        raise ValueError(f"{ch_file.name}: rows with sample_idx not in manifest")

    order = np.argsort(pred, kind="stable")
    sorted_pred = pred[order]
    sorted_gt = gt[order]
    sorted_user = u_rows[order]
    sorted_pos = sorted_gt.astype(np.float64)
    sorted_neg = (~sorted_gt).astype(np.float64)

    is_new_group = np.empty(sorted_pred.shape[0], dtype=bool)
    is_new_group[0] = True
    is_new_group[1:] = sorted_pred[1:] != sorted_pred[:-1]
    group_id = np.cumsum(is_new_group) - 1
    G = int(group_id[-1]) + 1

    out = np.full(n_boot, np.nan)

    pos_per_user = np.bincount(sorted_user, weights=sorted_pos, minlength=n_users)
    neg_per_user = np.bincount(sorted_user, weights=sorted_neg, minlength=n_users)

    S_pos = sp.csr_matrix(
        (sorted_pos, (sorted_user, group_id)),
        shape=(n_users, G),
    )
    S_neg = sp.csr_matrix(
        (sorted_neg, (sorted_user, group_id)),
        shape=(n_users, G),
    )

    B = n_boot
    cap_bytes = 1 * 1024**3
    batch = max(1, cap_bytes // (max(G, 1) * 8))
    batch = min(batch, B)

    for b0 in range(0, B, batch):
        b1 = min(b0 + batch, B)
        bs = b1 - b0
        M = np.empty((bs, n_users), dtype=np.float64)
        for j, b in enumerate(range(b0, b1)):
            M[j] = np.bincount(boot_idx[b], minlength=n_users).astype(np.float64)
        W_pos = M @ S_pos  # (bs, G)
        W_neg = M @ S_neg
        cumneg = np.cumsum(W_neg, axis=1)
        cumneg_before = np.empty_like(cumneg)
        cumneg_before[:, 0] = 0.0
        cumneg_before[:, 1:] = cumneg[:, :-1]
        numer = (W_pos * cumneg_before).sum(axis=1) + 0.5 * (W_pos * W_neg).sum(axis=1)
        N_pos = M @ pos_per_user
        N_neg = M @ neg_per_user
        denom = N_pos * N_neg
        with np.errstate(divide="ignore", invalid="ignore"):
            auc_b = np.where(denom > 0, numer / denom, np.nan)
        auc_b = np.where((N_pos == 0) | (N_neg == 0), np.nan, auc_b)
        out[b0:b1] = auc_b
    return out


def bootstrap_split(
    pairs_dir: Path,
    manifest,
    channel_stds: np.ndarray,
    n_boot: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
    include_auc: bool = True,
) -> dict:
    """Run participant-level bootstrap on a single (scenario, split) pairs dir.

    Returns a dict mirroring ``aggregate_pairs`` but with each metric value
    replaced by ``{point, bootstrap_mean, bootstrap_se, ci_lo, ci_hi, n_valid_boot}``.

    Args:
        pairs_dir: Directory containing ``pairs_ch*.parquet`` for one scenario x split.
        manifest: PyArrow table with ``(sample_idx, user_id, date)``.
        channel_stds: ``(N_CHANNELS,)`` array used to normalize errors.
        n_boot: Number of bootstrap iterations.
        ci_level: e.g. 0.95 for a 95% percentile CI.
        seed: RNG seed.
        include_auc: If False, skip the (slower) AUC bootstrap.
    """
    pairs_dir = Path(pairs_dir)
    n_channels = N_CHANNELS

    logger.info("Computing per-user sufficient statistics ...")
    stats = compute_user_sufficient_stats(pairs_dir, manifest, n_channels)
    n_users = len(stats.user_ids)
    if n_users == 0:
        return {"error": "no_users", "n_users": 0}

    logger.info(f"Bootstrap over {n_users} users, B={n_boot}, seed={seed}")
    boot_idx = _bootstrap_indices(n_users, n_boot, seed)
    decomp = _bootstrap_continuous_and_binary(stats, boot_idx, channel_stds)

    auc_b = np.full((n_boot, n_channels), np.nan, dtype=np.float64)
    if include_auc:
        _, _, sample_to_user_row = _build_sample_to_user(manifest)
        for ch in range(n_channels):
            if ch in CONTINUOUS_CHANNEL_INDICES:
                continue
            if not stats.has_data[ch]:
                continue
            logger.info(f"  AUC bootstrap channel {ch}")
            auc_b[:, ch] = _bootstrap_auc_one_channel(
                pairs_dir, ch, sample_to_user_row, n_users, boot_idx
            )

    cont_channels = [c for c in CONTINUOUS_CHANNEL_INDICES if stats.has_data[c]]
    bin_channels = [
        c for c in range(n_channels) if c not in CONTINUOUS_CHANNEL_INDICES and stats.has_data[c]
    ]

    def _macro(arr_2d: np.ndarray, channels: list[int]) -> np.ndarray:
        if not channels:
            return np.full(n_boot, np.nan)
        return np.nanmean(arr_2d[:, channels], axis=1)

    agg_rmse = _macro(decomp["rmse"], cont_channels)
    agg_nrmse = _macro(decomp["nrmse"], cont_channels)
    agg_nmse = _macro(decomp["nmse"], cont_channels)
    agg_nmae = _macro(decomp["nmae"], cont_channels)
    agg_balacc = _macro(decomp["balanced_accuracy"], bin_channels)
    agg_auc = _macro(auc_b, bin_channels) if include_auc else np.full(n_boot, np.nan)

    N_full = stats.n.sum(axis=0, keepdims=True)
    SSE_full = stats.sse.sum(axis=0, keepdims=True)
    SAE_full = stats.sae.sum(axis=0, keepdims=True)
    point_cont = _continuous_metrics_from_sums(N_full, SSE_full, SAE_full, channel_stds)
    TP_full = stats.tp.sum(axis=0, keepdims=True)
    TN_full = stats.tn.sum(axis=0, keepdims=True)
    FP_full = stats.fp.sum(axis=0, keepdims=True)
    FN_full = stats.fn.sum(axis=0, keepdims=True)
    point_balacc = _balanced_accuracy_from_confusion(TP_full, TN_full, FP_full, FN_full)

    point_auc = np.full(n_channels, np.nan)
    if include_auc:
        for ch in bin_channels:
            ch_file = _channel_file(pairs_dir, ch)
            table = pq.read_table(ch_file, columns=["gt", "pred"])
            gt = table.column("gt").to_numpy().astype(bool)
            pred = table.column("pred").to_numpy().astype(np.float32)
            del table
            if gt.all() or not gt.any():
                continue
            try:
                point_auc[ch] = float(roc_auc_score(gt, pred))
            except Exception:
                pass

    result: dict = {
        "n_users": n_users,
        "n_boot": n_boot,
        "seed": seed,
        "ci_level": ci_level,
        "per_channel": {},
        "continuous": {},
        "binary": {},
    }

    def _entry(point: float, values: np.ndarray) -> dict:
        summary = _summarize(values, ci_level)
        return {"point": float(point) if np.isfinite(point) else float("nan"), **summary}

    for ch in range(n_channels):
        ch_metrics: dict = {"channel_idx": ch}
        if not stats.has_data[ch]:
            ch_metrics["error"] = "no_masked_positions"
            result["per_channel"][f"ch_{ch}"] = ch_metrics
            continue
        if ch in CONTINUOUS_CHANNEL_INDICES:
            ch_metrics["n_masked"] = int(stats.n[:, ch].sum())
            ch_metrics["rmse"] = _entry(point_cont["rmse"][0, ch], decomp["rmse"][:, ch])
            ch_metrics["mse"] = _entry(point_cont["mse"][0, ch], decomp["mse"][:, ch])
            ch_metrics["mae"] = _entry(point_cont["mae"][0, ch], decomp["mae"][:, ch])
            ch_metrics["normalized_rmse"] = _entry(
                point_cont["nrmse"][0, ch], decomp["nrmse"][:, ch]
            )
            ch_metrics["normalized_mse"] = _entry(point_cont["nmse"][0, ch], decomp["nmse"][:, ch])
            ch_metrics["normalized_mae"] = _entry(point_cont["nmae"][0, ch], decomp["nmae"][:, ch])
        else:
            ch_metrics["n_masked"] = int(
                (stats.tp[:, ch] + stats.tn[:, ch] + stats.fp[:, ch] + stats.fn[:, ch]).sum()
            )
            ch_metrics["balanced_accuracy"] = _entry(
                point_balacc[0, ch], decomp["balanced_accuracy"][:, ch]
            )
            if include_auc:
                ch_metrics["roc_auc"] = _entry(point_auc[ch], auc_b[:, ch])
        result["per_channel"][f"ch_{ch}"] = ch_metrics

    if cont_channels:
        rmse_pt = float(np.nanmean(point_cont["rmse"][0, cont_channels]))
        nrmse_pt = float(np.nanmean(point_cont["nrmse"][0, cont_channels]))
        nmse_pt = float(np.nanmean(point_cont["nmse"][0, cont_channels]))
        nmae_pt = float(np.nanmean(point_cont["nmae"][0, cont_channels]))
    else:
        rmse_pt = nrmse_pt = nmse_pt = nmae_pt = float("nan")
    result["continuous"]["mean_rmse"] = _entry(rmse_pt, agg_rmse)
    result["continuous"]["mean_normalized_rmse"] = _entry(nrmse_pt, agg_nrmse)
    result["continuous"]["mean_normalized_mse"] = _entry(nmse_pt, agg_nmse)
    result["continuous"]["mean_normalized_mae"] = _entry(nmae_pt, agg_nmae)
    result["continuous"]["n_channels"] = len(cont_channels)

    if bin_channels:
        balacc_pt = float(np.nanmean(point_balacc[0, bin_channels]))
        if include_auc:
            auc_pt = float(np.nanmean(point_auc[bin_channels]))
        else:
            auc_pt = float("nan")
    else:
        balacc_pt = auc_pt = float("nan")
    result["binary"]["macro_balanced_accuracy"] = _entry(balacc_pt, agg_balacc)
    if include_auc:
        result["binary"]["macro_roc_auc"] = _entry(auc_pt, agg_auc)
    result["binary"]["n_channels"] = len(bin_channels)

    return result


def bootstrap_pairs_dir(
    pairs_dir: str | Path,
    *,
    splits: tuple[str, ...] = ("val", "test"),
    scenarios: list[str] | None = None,
    n_boot: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
    include_auc: bool = True,
    channel_stds: np.ndarray | None = None,
) -> dict:
    """Discover scenarios + splits under ``pairs_dir`` and bootstrap each.

    Auto-loads ``channel_stds.npy`` from ``pairs_dir`` (written by
    ``ImputationEvaluator.run`` when ``save_pairs=True``) and per-split manifests
    via ``load_sample_manifest``. Wraps ``bootstrap_split`` per (scenario, split).

    Args:
        pairs_dir: Root directory containing ``{scenario}/{split}/pairs_ch*.parquet``
            and ``manifest_{split}.parquet`` files, plus ``channel_stds.npy``.
        splits: Which splits to bootstrap. Skips ones without a manifest.
        scenarios: If given, restrict to these scenario names. Otherwise discover
            all subdirectories of ``pairs_dir`` that contain at least one split
            with pair files.
        n_boot: Number of bootstrap iterations.
        ci_level: Percentile CI level.
        seed: RNG seed (same seed across scenarios/splits is fine — each split
            has independent users, so cross-split correlation is moot).
        include_auc: If False, skip the slower AUC bootstrap.
        channel_stds: Override ``pairs_dir/channel_stds.npy``. Pass when the
            stds live elsewhere.

    Returns:
        ``{"config": {...}, "scenarios": {scenario: {split: bootstrap_split_result}}}``
        — matches the layout written by the private repo's
        ``bootstrap_imputation_pairs.py`` driver.
    """
    pairs_dir = Path(pairs_dir)
    if not pairs_dir.is_dir():
        raise FileNotFoundError(f"pairs_dir does not exist: {pairs_dir}")

    if channel_stds is None:
        stds_path = pairs_dir / "channel_stds.npy"
        if not stds_path.exists():
            raise FileNotFoundError(
                f"channel_stds.npy not found at {stds_path}. "
                "Re-run evaluation with save_pairs=True, or pass channel_stds explicitly."
            )
        channel_stds = np.load(stds_path)

    if scenarios is None:
        scenarios = sorted(
            d.name
            for d in pairs_dir.iterdir()
            if d.is_dir() and any((d / s).is_dir() for s in splits)
        )

    out: dict = {
        "config": {
            "n_boot": n_boot,
            "ci_level": ci_level,
            "seed": seed,
            "include_auc": include_auc,
        },
        "scenarios": {},
    }

    for scenario in scenarios:
        out["scenarios"][scenario] = {}
        for split in splits:
            split_dir = pairs_dir / scenario / split
            if not split_dir.is_dir():
                continue
            if not any(_channel_file(split_dir, ch).exists() for ch in range(N_CHANNELS)):
                continue
            manifest = load_sample_manifest(pairs_dir, split)
            if manifest is None:
                logger.warning(
                    "No manifest at %s; skipping bootstrap for %s/%s",
                    pairs_dir / f"manifest_{split}.parquet",
                    scenario,
                    split,
                )
                continue
            logger.info("Bootstrapping %s/%s ...", scenario, split)
            out["scenarios"][scenario][split] = bootstrap_split(
                split_dir,
                manifest,
                channel_stds,
                n_boot=n_boot,
                ci_level=ci_level,
                seed=seed,
                include_auc=include_auc,
            )

    return out
