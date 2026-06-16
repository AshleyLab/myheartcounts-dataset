"""Cross-task bootstrap for skill score, average rank, and fairness CIs.

The participant-level cluster bootstrap in ``bootstrap.py`` produces CIs for
**per-(scenario, split, channel)** metrics. This module raises that one level:
it resamples users, recomputes per-(method, scenario, channel) error E for
**every method together** (paired across methods via a shared resample
matrix), then re-runs the existing point-flow aggregations (skill score,
average rank, fairness) on each draw. Stacking draws yields mean / SE /
percentile-CI for every cell of the paper's headline table.

Pipeline
--------

Phase 1 — :func:`compute_per_draw_errors` reads each method's saved
pairs/, builds per-(method, scenario, split, channel, subgroup) per-user
sufficient stats, generates a single shared resample matrix per split
(preserving cross-scenario and cross-subgroup covariance within each
draw), and writes a long-format Parquet of per-draw E values.

Phase 2 — :func:`aggregate_skill_rank_fairness` consumes that Parquet and
runs ``compute_skill_scores`` / ``compute_average_rankings`` /
``compute_fairness`` per draw; new disparity metrics or λ values rerun
phase 2 in seconds without touching pairs/.
"""

from __future__ import annotations

import hashlib
import json
import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data.processing.hf_config import CONTINUOUS_CHANNEL_INDICES, N_CHANNELS

# Importing from bootstrap.py — these helpers are battle-tested.
from imputation_evaluation.evaluation.bootstrap import (
    _bootstrap_indices,
    _summarize,
)
from imputation_evaluation.evaluation.disparity_metrics import (
    DISPARITY_FUNCTIONS,
    FAIRNESS_COMBINE,
    disparity_higher_is_better,
)
from imputation_evaluation.evaluation.paper_metrics_core import (
    BASELINE_CONTINUOUS,
    BINARY_CATEGORIES_ORDERED,
    EXCLUDE_BINARY_SCENARIOS,
    build_baseline_errors,
    compute_average_rankings,
    compute_skill_scores,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Constants — kept in sync with compute_imputation_paper_metrics.py
# --------------------------------------------------------------------------

ALL_KEY = ("all", "all")  # sentinel cell for the global / non-subgroup metrics

# Skill-score constants — mirror forecasting_evaluation/metrics/metric_spec.py
# so the two tracks share an estimand. ``BINARY_ERROR_FLOOR`` is the ε on
# binary E = 1 − AUC so a perfect-AUC user contributes a finite paired ratio
# instead of being dropped by the ``baseline > 0`` filter. Continuous E is
# unfloored. Clip bounds apply to the paired ratio, not to E directly.
SKILL_CLIP_LOWER = 0.01
SKILL_CLIP_UPPER = 100.0
SKILL_MIN_PAIRS = 1
BINARY_ERROR_FLOOR = 0.005


# --------------------------------------------------------------------------
# Subgroup-aware per-user sufficient statistics
# --------------------------------------------------------------------------


@dataclass
class BinaryRows:
    """Raw rows for a single binary channel, used for the AUC bootstrap."""

    gt: np.ndarray  # bool, (R,)
    pred: np.ndarray  # float32, (R,)
    u_rows: np.ndarray  # int64,  (R,)  — index into cell's user_ids


@dataclass
class CellStats:
    """Per-user sufficient stats for one (scenario, split, subgroup) cell.

    ``user_ids`` is the canonical user ordering shared with the cell's
    bootstrap-index matrix. ``has_data[ch]`` is True iff any row exists for
    that channel (after subgroup filtering). For binary channels we also
    keep the raw row arrays (gt, pred, u_rows) so the AUC bootstrap can
    operate on subgroup-filtered rows.
    """

    user_ids: list[str]
    n: np.ndarray  # (U, C) int64    — continuous sample counts
    sse: np.ndarray  # (U, C) float64
    sae: np.ndarray  # (U, C) float64
    tp: np.ndarray  # (U, C) int64    — binary confusion sums (kept for parity)
    tn: np.ndarray
    fp: np.ndarray
    fn: np.ndarray
    has_data: np.ndarray  # (C,) bool
    # Binary raw rows per channel — for AUC; lazy, only set when needed
    binary_rows: dict[int, BinaryRows]


def _channel_file(scenario_split_dir: Path, ch: int) -> Path:
    return scenario_split_dir / f"pairs_ch{ch:02d}.parquet"


def _build_cell_user_index(
    sample_idx_arr: np.ndarray,
    user_id_arr: list[str],
    canonical_user_index: dict[str, int],
    keep_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Map manifest rows to canonical user-row indices, optionally filtering."""
    if keep_mask is not None:
        sample_idx_arr = sample_idx_arr[keep_mask]
        user_id_arr = [user_id_arr[i] for i in np.flatnonzero(keep_mask)]
    u_rows = np.array([canonical_user_index[u] for u in user_id_arr], dtype=np.int64)
    return sample_idx_arr, u_rows


def _subgroup_cells_from_mapping(
    subgroup_mapping: dict[int, dict[str, str]] | None,
    exclude_unknown: bool,
) -> list[tuple[str, str]]:
    """Discover ``(attr, value)`` cells present in the per-sample subgroup map.

    Always includes the ``ALL_KEY`` sentinel for the global cell.
    """
    cells: set[tuple[str, str]] = {ALL_KEY}
    if subgroup_mapping:
        for demo in subgroup_mapping.values():
            for attr, val in demo.items():
                if exclude_unknown and val == "unknown":
                    continue
                cells.add((attr, val))
    return sorted(cells)


def compute_user_stats_per_cell(
    scenario_split_dir: Path,
    manifest: pa.Table,
    canonical_user_ids: list[str],
    canonical_user_index: dict[str, int],
    cells: list[tuple[str, str]],
    sample_to_subgroup: dict[int, dict[str, str]] | None,
    n_channels: int = N_CHANNELS,
    *,
    include_binary_rows: bool = True,
) -> dict[tuple[str, str], CellStats]:
    """Per-user sufficient stats per (subgroup_attr, subgroup_value) cell.

    Streaming per-channel pass that accumulates per-user sufficient stats
    for every cell in ``cells``.
    All cells share the same canonical user ordering so that downstream
    code can use a single ``idx_b`` matrix per split.

    Args:
        scenario_split_dir: e.g. ``pairs/random_noise/test/``.
        manifest: PyArrow table with ``(sample_idx, user_id, date)``.
        canonical_user_ids: ordered list of every user that appears in
            *any* method's manifest for this split. Users absent from a
            particular scenario/method just leave that user's row at zero.
        canonical_user_index: ``user_id -> row in canonical_user_ids``.
        cells: list of ``(attr, value)`` tuples to populate.
        sample_to_subgroup: maps ``sample_idx`` to attribute dicts. May be
            ``None`` if only the ``ALL_KEY`` cell is requested.
        n_channels: number of channels to scan (default ``N_CHANNELS``).
        include_binary_rows: keep raw (gt, pred, u_rows) per binary channel
            so the AUC bootstrap can run later. Costs RAM proportional to
            row count; turn off if AUC isn't needed.
    """
    sidx_arr = manifest.column("sample_idx").to_numpy()
    uid_arr = manifest.column("user_id").to_pylist()

    # Map manifest sample_idx -> (canonical user row) once.
    full_u_rows = np.array([canonical_user_index[u] for u in uid_arr], dtype=np.int64)

    # Build per-cell row masks over the manifest rows.
    cell_masks: dict[tuple[str, str], np.ndarray] = {}
    n_manifest = len(uid_arr)
    cell_masks[ALL_KEY] = np.ones(n_manifest, dtype=bool)
    for cell in cells:
        if cell == ALL_KEY:
            continue
        attr, value = cell
        mask = np.zeros(n_manifest, dtype=bool)
        if sample_to_subgroup is None:
            cell_masks[cell] = mask
            continue
        for i, sidx in enumerate(sidx_arr):
            demo = sample_to_subgroup.get(int(sidx))
            if demo is None:
                continue
            if demo.get(attr) == value:
                mask[i] = True
        cell_masks[cell] = mask

    U = len(canonical_user_ids)
    out: dict[tuple[str, str], CellStats] = {}
    for cell in cells:
        out[cell] = CellStats(
            user_ids=canonical_user_ids,
            n=np.zeros((U, n_channels), dtype=np.int64),
            sse=np.zeros((U, n_channels), dtype=np.float64),
            sae=np.zeros((U, n_channels), dtype=np.float64),
            tp=np.zeros((U, n_channels), dtype=np.int64),
            tn=np.zeros((U, n_channels), dtype=np.int64),
            fp=np.zeros((U, n_channels), dtype=np.int64),
            fn=np.zeros((U, n_channels), dtype=np.int64),
            has_data=np.zeros(n_channels, dtype=bool),
            binary_rows={},
        )

    # sample_idx -> manifest-row, so we can map pair rows to manifest masks.
    sidx_to_manifest_row = np.full(
        int(sidx_arr.max()) + 1 if len(sidx_arr) else 0, -1, dtype=np.int64
    )
    for i, sidx in enumerate(sidx_arr):
        sidx_to_manifest_row[int(sidx)] = i

    cont_set = set(CONTINUOUS_CHANNEL_INDICES)

    for ch in range(n_channels):
        ch_file = _channel_file(scenario_split_dir, ch)
        if not ch_file.exists():
            continue
        table = pq.read_table(ch_file, columns=["sample_idx", "gt", "pred"])
        if table.num_rows == 0:
            continue
        pair_sidx = table.column("sample_idx").to_numpy()
        pair_manifest_rows = sidx_to_manifest_row[pair_sidx]
        if (pair_manifest_rows < 0).any():
            raise ValueError(f"{ch_file.name}: rows with sample_idx not in manifest")

        u_rows_pair = full_u_rows[pair_manifest_rows]

        if ch in cont_set:
            gt_ch = table.column("gt").to_numpy().astype(np.float32)
            pred_ch = table.column("pred").to_numpy().astype(np.float32)
            err = (pred_ch - gt_ch).astype(np.float64)
            err_sq = err * err
            err_abs = np.abs(err)
            for cell, mask in cell_masks.items():
                if not mask.any():
                    continue
                row_mask = mask[pair_manifest_rows]
                if not row_mask.any():
                    continue
                cs = out[cell]
                cs.has_data[ch] = True
                u_sub = u_rows_pair[row_mask]
                np.add.at(cs.n[:, ch], u_sub, 1)
                np.add.at(cs.sse[:, ch], u_sub, err_sq[row_mask])
                np.add.at(cs.sae[:, ch], u_sub, err_abs[row_mask])
        else:
            gt_bool = table.column("gt").to_numpy().astype(bool)
            pred_ch = table.column("pred").to_numpy().astype(np.float32)
            pred_bool = pred_ch > 0.5
            tp_mask_all = gt_bool & pred_bool
            tn_mask_all = (~gt_bool) & (~pred_bool)
            fp_mask_all = (~gt_bool) & pred_bool
            fn_mask_all = gt_bool & (~pred_bool)
            for cell, mask in cell_masks.items():
                if not mask.any():
                    continue
                row_mask = mask[pair_manifest_rows]
                if not row_mask.any():
                    continue
                cs = out[cell]
                cs.has_data[ch] = True
                u_sub = u_rows_pair[row_mask]
                np.add.at(cs.tp[:, ch], u_sub, tp_mask_all[row_mask].astype(np.int64))
                np.add.at(cs.tn[:, ch], u_sub, tn_mask_all[row_mask].astype(np.int64))
                np.add.at(cs.fp[:, ch], u_sub, fp_mask_all[row_mask].astype(np.int64))
                np.add.at(cs.fn[:, ch], u_sub, fn_mask_all[row_mask].astype(np.int64))
                if include_binary_rows:
                    cs.binary_rows[ch] = BinaryRows(
                        gt=gt_bool[row_mask],
                        pred=pred_ch[row_mask],
                        u_rows=u_sub.astype(np.int64),
                    )

    return out


# --------------------------------------------------------------------------
# AUC bootstrap from in-memory arrays
# --------------------------------------------------------------------------


def _bootstrap_auc_from_arrays(
    gt: np.ndarray,
    pred: np.ndarray,
    u_rows: np.ndarray,
    n_users: int,
    boot_idx: np.ndarray,
) -> np.ndarray:
    """Cluster bootstrap of ROC AUC via Mann-Whitney U with per-user multiplicities.

    Mirrors :func:`bootstrap._bootstrap_auc_one_channel` but operates on
    in-memory arrays (avoids re-reading the parquet file when the same row
    set is bootstrapped many times under different subgroup masks).
    """
    import scipy.sparse as sp

    n_boot = boot_idx.shape[0]
    if gt.size == 0:
        return np.full(n_boot, np.nan)

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

    out = np.full(n_boot, np.nan, dtype=np.float64)
    cap_bytes = 1 * 1024**3
    batch = max(1, cap_bytes // (max(G, 1) * 8))
    batch = min(batch, n_boot)
    for b0 in range(0, n_boot, batch):
        b1 = min(b0 + batch, n_boot)
        bs = b1 - b0
        M = np.empty((bs, n_users), dtype=np.float64)
        for j, b in enumerate(range(b0, b1)):
            M[j] = np.bincount(boot_idx[b], minlength=n_users).astype(np.float64)
        W_pos = M @ S_pos
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


# --------------------------------------------------------------------------
# Per-draw error reconstruction (phase 1 core)
# --------------------------------------------------------------------------


def _seed_for_split(seed: int, split: str) -> int:
    """Stable per-split seed for the shared resample matrix.

    Every scenario and subgroup cell within the same split shares one
    resample matrix so that cross-scenario covariance and within-draw
    fairness pairing are preserved.

    Uses SHA-256 rather than Python's built-in ``hash()`` (which is
    process-randomized by ``PYTHONHASHSEED`` for str keys).
    """
    key = f"{seed}|{split}".encode()
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _per_user_auc_from_cell_stats(
    cell_stats: CellStats, n_channels: int
) -> np.ndarray:
    """Precompute per-(user, channel) AUC for binary channels (user-macro path).

    Returns shape ``(n_users, n_channels)`` float64; NaN at channels that are
    continuous, at (user, channel) pairs with zero data, and at (user, channel)
    pairs where the user has only one class present (AUC undefined).

    Cached upfront because per-user AUC is **invariant across bootstrap draws** —
    each draw merely resamples user rows (with replacement). Replacing
    ``_bootstrap_auc_from_arrays``'s per-draw Mann–Whitney recomputation with
    a single precompute + ``nanmean(per_user_auc[boot_idx], axis=1)`` brings
    the per-draw binary work from O(draws × Mann-Whitney) to O(draws × users).
    """
    from sklearn.metrics import roc_auc_score

    n_users = cell_stats.n.shape[0]
    out = np.full((n_users, n_channels), np.nan, dtype=np.float64)
    for ch in range(n_channels):
        if ch in CONTINUOUS_CHANNEL_INDICES:
            continue
        if not cell_stats.has_data[ch]:
            continue
        br = cell_stats.binary_rows.get(ch)
        if br is None or br.gt.size == 0:
            continue
        u_rows = br.u_rows
        gt_arr = br.gt
        pred_arr = br.pred
        for u in np.unique(u_rows):
            mask = u_rows == u
            user_gt = gt_arr[mask]
            if user_gt.size == 0 or user_gt.all() or not user_gt.any():
                continue  # single-class user → AUC undefined → NaN
            try:
                out[u, ch] = float(roc_auc_score(user_gt, pred_arr[mask]))
            except Exception:
                pass  # leave NaN on sklearn edge cases
    return out


def _per_method_cell_collapsed_errors(
    per_user_auc: np.ndarray,
    boot_idx: np.ndarray,
    binary_categories: tuple[tuple[str, tuple[int, ...]], ...],
) -> np.ndarray:
    """Compute (n_boot, n_binary_categories) per-draw E for collapsed scopes.

    Reuses the precomputed per-(user, channel) AUC matrix from
    :func:`_per_user_auc_from_cell_stats` — no recomputation across draws.

    For each binary category:

    1. Per-(user, category) E = ``nanmean`` over the category's channels of
       ``1 − AUC[user, ch]``. Users with only one class in every channel of
       the category → NaN.
    2. Per-(draw, category) E = ``nanmean`` over the draw's resampled
       users of per-(user, category) E. Users with NaN per-category E drop
       from the macro mean (their replicates simply don't contribute).

    Returned array preserves the column order of ``binary_categories``.
    """
    n_users = per_user_auc.shape[0]
    n_boot = boot_idx.shape[0]
    n_cats = len(binary_categories)
    out = np.full((n_boot, n_cats), np.nan, dtype=np.float64)

    # Precompute per-(user, category) E once across all draws.
    per_user_E_cat = np.full((n_users, n_cats), np.nan, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for cat_idx, (_cat_name, ch_indices) in enumerate(binary_categories):
            # Restrict to columns that actually have AUC data (some channels
            # may be all-NaN for this cell — e.g. a category with no
            # qualifying users on a rare scenario).
            cols = np.asarray(ch_indices, dtype=np.int64)
            sub = per_user_auc[:, cols]  # (n_users, n_channels_in_cat)
            per_user_E_cat[:, cat_idx] = np.nanmean(1.0 - sub, axis=1)

        for cat_idx in range(n_cats):
            per_draw_rows = per_user_E_cat[boot_idx, cat_idx]
            out[:, cat_idx] = np.nanmean(per_draw_rows, axis=1)
    return out


def _per_method_cell_errors(
    cell_stats: CellStats,
    boot_idx: np.ndarray,
    channel_stds: np.ndarray,
    include_auc: bool,
    per_user_auc: np.ndarray | None = None,
) -> np.ndarray:
    """Compute (n_boot, n_channels) per-draw error E under **user-macro** aggregation.

    Per-channel task error is the arithmetic macroaverage **over users** of the
    per-(user, channel) error. The bootstrap draws are over users (cluster
    bootstrap), so per-user errors are invariant across draws — precompute
    once, then take ``nanmean`` over the resampled user rows per draw. This is
    both correct (matches the "users-then-channels" framing) and substantially
    faster than the prior cell-micro path's per-draw Mann–Whitney AUC
    recomputation.

    Continuous channels: ``E = MAE`` (per-user macro mean of per-user MAE).
    Per-user MAE is ``sae_u / n_u`` — sum of absolute errors over all the
    user's cells, divided by the cell count, then converted once. This is
    equivalent to forecasting's ``within_user_aggregation="micro"`` by
    construction; imputation has no per-window intermediary, so no
    aggregation-mode knob is needed. Per-channel
    ``MAE_method / MAE_baseline = nMAE_method / nMAE_baseline`` exactly
    (``channel_std`` cancels in the per-task ratio), so the skill score is
    numerically identical to the normalized version. ``channel_stds`` is
    retained on the signature for backward compatibility but no longer
    affects the returned E.

    Binary channels: ``E = 1 − AUC`` where AUC is the per-user AUC macro
    mean. Per-(user, channel) AUC requires both classes within that user;
    single-class users contribute NaN and drop from the per-task mean.
    Unfloored here — the absolute metric. The
    :data:`BINARY_ERROR_FLOOR` is applied only inside the paired-ratio
    reducer (see :func:`_per_method_cell_paired_ratios`).

    Channels with no data anywhere (``cell_stats.has_data[ch] == False``) are
    NaN columns.
    """
    n_channels = cell_stats.n.shape[1]
    n_boot = boot_idx.shape[0]
    E = np.full((n_boot, n_channels), np.nan, dtype=np.float64)

    # --- Continuous: per-(user, channel) MAE, then per-draw user-macro -----
    n_u_ch = cell_stats.n  # (U, C) int64
    sae_u_ch = cell_stats.sae  # (U, C) float64
    with np.errstate(divide="ignore", invalid="ignore"):
        per_user_mae = np.where(n_u_ch > 0, sae_u_ch / np.maximum(n_u_ch, 1), np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # silence all-NaN slice warnings
        for ch in CONTINUOUS_CHANNEL_INDICES:
            if not cell_stats.has_data[ch]:
                continue
            # boot_idx: (n_boot, n_users_per_draw). Fancy-index user axis.
            per_draw_rows = per_user_mae[boot_idx, ch]  # (n_boot, n_users_per_draw)
            E[:, ch] = np.nanmean(per_draw_rows, axis=1)

    # --- Binary: per-(user, channel) AUC, then per-draw user-macro ---------
    if include_auc:
        if per_user_auc is None:
            per_user_auc = _per_user_auc_from_cell_stats(cell_stats, n_channels)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for ch in range(n_channels):
                if ch in CONTINUOUS_CHANNEL_INDICES:
                    continue
                if not cell_stats.has_data[ch]:
                    continue
                per_draw_rows = per_user_auc[boot_idx, ch]
                mean_auc = np.nanmean(per_draw_rows, axis=1)
                E[:, ch] = 1.0 - mean_auc

    return E


def _per_user_log_ratio_column(e_m: np.ndarray, e_b: np.ndarray) -> np.ndarray:
    """Per-user clipped log-ratio column; NaN where the pair filter fails.

    Mirrors ``forecasting_evaluation.metrics.skill_score_summary
    .compute_skill_from_errors``'s per-pair filtering for a single
    ``(model, baseline)`` per-user error vector — returns a per-user array
    of length ``n_users`` with ``log(clip(e_m / e_b, lo, hi))`` where the
    pair is valid, NaN elsewhere.
    """
    n = e_m.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    valid = np.isfinite(e_m) & np.isfinite(e_b) & (e_b > 0.0)
    if not np.any(valid):
        return out
    ratios = e_m[valid] / e_b[valid]
    ratios = np.clip(ratios, SKILL_CLIP_LOWER, SKILL_CLIP_UPPER)
    finite = np.isfinite(ratios) & (ratios > 0.0)
    valid_idx = np.flatnonzero(valid)
    keep = valid_idx[finite]
    out[keep] = np.log(ratios[finite])
    return out


def _per_method_cell_paired_ratios(
    cell_stats_method: CellStats,
    cell_stats_baseline: CellStats,
    boot_idx: np.ndarray,
    *,
    include_auc: bool,
    per_user_auc_method: np.ndarray | None = None,
    per_user_auc_baseline: np.ndarray | None = None,
) -> np.ndarray:
    """Compute ``(n_boot, n_channels)`` per-draw geometric-mean ratio R.

    ``R[b, ch] = exp(nanmean(log(clip(e^M_u / e^B_u, lo, hi))))`` over the
    surviving resampled users, mirroring ``forecasting_evaluation.metrics
    .skill_score_summary.compute_skill_from_errors`` at the per-user grain.

    - Continuous per-user error: ``e_u = sae_u / n_u`` (unfloored), NaN where
      ``n_u == 0``. Same micro-by-construction reasoning as
      :func:`_per_method_cell_errors`.
    - Binary per-user error: ``e_u = max(1 − AUC_u, BINARY_ERROR_FLOOR)``.
      Single-class users have NaN ``AUC_u``; ``np.maximum`` preserves NaN
      so they drop from the join.
    - Inner-join on the canonical user index (both ``CellStats`` share the
      ordering from :func:`compute_user_stats_per_cell`). Drop users where
      either side is non-finite or ``e^B_u ≤ 0``; clip the ratio to
      ``[SKILL_CLIP_LOWER, SKILL_CLIP_UPPER]``; drop non-finite or
      non-positive clipped ratios.
    - Baseline ≡ self → R ≡ 1, so skill = 1 − R = 0 by construction.

    Channels with no data on either side return NaN columns. Per-draw
    cells with fewer than :data:`SKILL_MIN_PAIRS` finite log-ratios are
    NaN.
    """
    n_users = cell_stats_method.n.shape[0]
    n_channels = cell_stats_method.n.shape[1]
    n_boot = boot_idx.shape[0]
    R = np.full((n_boot, n_channels), np.nan, dtype=np.float64)
    log_r = np.full((n_users, n_channels), np.nan, dtype=np.float64)

    # --- Continuous: per-user MAE both sides, then per-user log-ratio -------
    n_m = cell_stats_method.n
    sae_m = cell_stats_method.sae
    n_b = cell_stats_baseline.n
    sae_b = cell_stats_baseline.sae
    with np.errstate(divide="ignore", invalid="ignore"):
        e_m_cont = np.where(n_m > 0, sae_m / np.maximum(n_m, 1), np.nan)
        e_b_cont = np.where(n_b > 0, sae_b / np.maximum(n_b, 1), np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for ch in CONTINUOUS_CHANNEL_INDICES:
            if not cell_stats_method.has_data[ch] or not cell_stats_baseline.has_data[ch]:
                continue
            log_r[:, ch] = _per_user_log_ratio_column(e_m_cont[:, ch], e_b_cont[:, ch])

    # --- Binary: per-user AUC both sides, ε-floored, then per-user log-ratio -
    if include_auc:
        if per_user_auc_method is None:
            per_user_auc_method = _per_user_auc_from_cell_stats(cell_stats_method, n_channels)
        if per_user_auc_baseline is None:
            per_user_auc_baseline = _per_user_auc_from_cell_stats(
                cell_stats_baseline, n_channels
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for ch in range(n_channels):
                if ch in CONTINUOUS_CHANNEL_INDICES:
                    continue
                if not cell_stats_method.has_data[ch] or not cell_stats_baseline.has_data[ch]:
                    continue
                e_m_bin = np.maximum(1.0 - per_user_auc_method[:, ch], BINARY_ERROR_FLOOR)
                e_b_bin = np.maximum(1.0 - per_user_auc_baseline[:, ch], BINARY_ERROR_FLOOR)
                log_r[:, ch] = _per_user_log_ratio_column(e_m_bin, e_b_bin)

    # --- Reduce per-user log-ratios over each draw to per-draw R -----------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for ch in range(n_channels):
            if np.all(np.isnan(log_r[:, ch])):
                continue
            per_draw_rows = log_r[boot_idx, ch]  # (n_boot, n_users_per_draw)
            n_valid_per_draw = np.sum(np.isfinite(per_draw_rows), axis=1)
            mean_log = np.nanmean(per_draw_rows, axis=1)
            R[:, ch] = np.where(
                n_valid_per_draw >= SKILL_MIN_PAIRS, np.exp(mean_log), np.nan
            )

    return R


def _per_method_cell_paired_collapsed_ratios(
    per_user_auc_method: np.ndarray,
    per_user_auc_baseline: np.ndarray,
    boot_idx: np.ndarray,
    binary_categories: tuple[tuple[str, tuple[int, ...]], ...],
) -> np.ndarray:
    """Compute ``(n_boot, n_binary_categories)`` per-draw paired-ratio R.

    Mirrors :func:`_per_method_cell_collapsed_errors` (per-method absolute
    E for collapsed binary categories) at the paired-ratio level:

    1. Per-(user, category) E for each side = ``nanmean`` over the
       category's channels of ``max(1 − AUC_u, BINARY_ERROR_FLOOR)``. Users
       with NaN per-(user, channel) AUC on every channel of the category →
       NaN per-(user, category) E → drop from the join.
    2. Inner-join on user, filter ``e^B > 0`` and finiteness, clip the
       ratio to ``[SKILL_CLIP_LOWER, SKILL_CLIP_UPPER]`` →
       per-(user, category) log-ratio column.
    3. Per draw: ``R[b, cat] = exp(nanmean(log_r[boot_idx[b], cat]))`` over
       the surviving resampled users.

    Returned array preserves the column order of ``binary_categories``.
    """
    n_users = per_user_auc_method.shape[0]
    n_boot = boot_idx.shape[0]
    n_cats = len(binary_categories)
    R = np.full((n_boot, n_cats), np.nan, dtype=np.float64)
    log_r_cat = np.full((n_users, n_cats), np.nan, dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for cat_idx, (_cat_name, ch_indices) in enumerate(binary_categories):
            cols = np.asarray(ch_indices, dtype=np.int64)
            sub_m = per_user_auc_method[:, cols]
            sub_b = per_user_auc_baseline[:, cols]
            e_u_m = np.nanmean(np.maximum(1.0 - sub_m, BINARY_ERROR_FLOOR), axis=1)
            e_u_b = np.nanmean(np.maximum(1.0 - sub_b, BINARY_ERROR_FLOOR), axis=1)
            log_r_cat[:, cat_idx] = _per_user_log_ratio_column(e_u_m, e_u_b)

        for cat_idx in range(n_cats):
            per_draw_rows = log_r_cat[boot_idx, cat_idx]
            n_valid_per_draw = np.sum(np.isfinite(per_draw_rows), axis=1)
            mean_log = np.nanmean(per_draw_rows, axis=1)
            R[:, cat_idx] = np.where(
                n_valid_per_draw >= SKILL_MIN_PAIRS, np.exp(mean_log), np.nan
            )

    return R


def _channel_type(ch: int) -> str:
    return "continuous" if ch in CONTINUOUS_CHANNEL_INDICES else "binary"


_REQUIRED_MANIFEST_COLUMNS: tuple[str, ...] = ("sample_idx", "user_id", "date")


def _assert_manifests_agree(
    method_manifests: dict[str, pa.Table],
    *,
    split: str,
    reference_method: str | None = None,
    n_examples: int = 5,
) -> None:
    """Fail fast if methods disagree on ``sample_idx -> (user_id, date)``.

    Fairness subgroup mappings are built from one reference method's manifest
    and then applied to every method's per-method manifest. If a method's
    ``sample_idx`` indexes a different ``(user_id, date)`` pair than the
    reference, fairness subgroup rows are silently mis-assigned to the wrong
    demographic. This validator enforces the invariant that all methods share
    the same ``sample_idx -> (user_id, date)`` mapping before any metric work
    happens.

    Args:
        method_manifests: Mapping ``{method: pyarrow.Table}`` for one split.
        split: Split name, used only for error messages.
        reference_method: Method to treat as the source of truth. If ``None``,
            uses the first key of ``method_manifests`` (matches the
            "first method" contract in ``bootstrap_imputation_draws.py``).
        n_examples: Max number of example ``sample_idx`` rows per mismatch
            category to include in the error message.

    Raises:
        ValueError: When any manifest is missing a required column, contains
            duplicate ``sample_idx`` values, or disagrees with the reference
            on the set of ``sample_idx`` or the ``(user_id, date)`` mapping.
    """
    if not method_manifests:
        return

    methods = list(method_manifests.keys())
    if reference_method is None:
        reference_method = methods[0]
    if reference_method not in method_manifests:
        raise ValueError(
            f"[split={split}] reference_method={reference_method!r} not in "
            f"method_manifests (have {methods})"
        )

    def _check_columns(method: str, table: pa.Table) -> None:
        missing_cols = [c for c in _REQUIRED_MANIFEST_COLUMNS if c not in table.column_names]
        if missing_cols:
            raise ValueError(
                f"[split={split}] manifest for method {method!r} is missing "
                f"required columns: {missing_cols}. Have: {list(table.column_names)}"
            )

    def _build_map(method: str, table: pa.Table) -> dict[int, tuple[str, str]]:
        sample_idxs = table.column("sample_idx").to_pylist()
        user_ids = table.column("user_id").to_pylist()
        dates = table.column("date").to_pylist()
        out: dict[int, tuple[str, str]] = {}
        duplicates: list[int] = []
        for sidx, uid, date_str in zip(sample_idxs, user_ids, dates):
            key = int(sidx)
            if key in out:
                duplicates.append(key)
                continue
            out[key] = (str(uid), str(date_str))
        if duplicates:
            dup_examples = duplicates[:n_examples]
            raise ValueError(
                f"[split={split}] manifest for method {method!r} has "
                f"duplicate sample_idx values (not unique): "
                f"{len(duplicates)} duplicates, examples: {dup_examples}"
            )
        return out

    # Validate every manifest's columns first so column errors surface together
    # with reference-vs-method errors below.
    for method, table in method_manifests.items():
        _check_columns(method, table)

    ref_map = _build_map(reference_method, method_manifests[reference_method])
    ref_keys = set(ref_map.keys())

    for method, table in method_manifests.items():
        if method == reference_method:
            continue
        method_map = _build_map(method, table)
        method_keys = set(method_map.keys())

        missing = sorted(ref_keys - method_keys)  # in ref, not in method
        extra = sorted(method_keys - ref_keys)  # in method, not in ref
        common = ref_keys & method_keys
        mismatched = sorted(s for s in common if ref_map[s] != method_map[s])

        if not (missing or extra or mismatched):
            continue

        lines = [
            f"Manifest mismatch in [split={split}] between reference method "
            f"{reference_method!r} and method {method!r}:",
            f"  missing sample_idx (in reference, not in {method!r}): {len(missing)}",
            f"  extra sample_idx   (in {method!r}, not in reference): {len(extra)}",
            f"  user_id/date mismatched: {len(mismatched)} of {len(common)}",
        ]
        if missing:
            lines.append("Examples (missing):")
            for s in missing[:n_examples]:
                lines.append(f"  sample_idx={s}   ref={ref_map[s]}")
        if extra:
            lines.append("Examples (extra):")
            for s in extra[:n_examples]:
                lines.append(f"  sample_idx={s}   {method!r}={method_map[s]}")
        if mismatched:
            lines.append("Examples (mismatched):")
            for s in mismatched[:n_examples]:
                lines.append(
                    f"  sample_idx={s}   ref={ref_map[s]}   "
                    f"{method!r}={method_map[s]}"
                )
        raise ValueError("\n".join(lines))


def compute_per_draw_errors(
    method_dirs: dict[str, Path],
    scenarios: list[str],
    splits: list[str],
    n_boot: int,
    seed: int,
    *,
    baseline_method: str = BASELINE_CONTINUOUS,
    subgroup_mappings: dict[str, dict[int, dict[str, str]]] | None = None,
    channel_stds: np.ndarray | None = None,
    channel_stds_path: Path | None = None,
    include_auc: bool = True,
    exclude_unknown: bool = False,
    n_channels: int = N_CHANNELS,
    progress_logger: Callable[[str], None] = logger.info,
) -> pd.DataFrame:
    """Phase-1 core: build a long-format DataFrame of per-draw errors and ratios.

    Two value columns are emitted per row:

    - ``E`` — per-method absolute, user-macro MAE (continuous) or ``1 − AUC``
      (binary). This is what :func:`compute_average_rankings` ranks on.
    - ``R`` — paired geometric-mean ratio of per-user error vs the baseline
      (``baseline_method``), formed from per-user arrays inside
      :func:`_per_method_cell_paired_ratios`. This is what
      :func:`compute_skill_scores` consumes: ``skill = 1 − exp(mean(log R))``
      over tasks in scope. Baseline-vs-self → R = 1 by construction (each
      user's ratio is 1), so the baseline's skill is 0.

    Args:
        method_dirs: ``{method: pairs_root}`` where ``pairs_root`` contains
            ``manifest_<split>.parquet``, ``channel_stds.npy`` and per-scenario
            subdirs with ``<scenario>/<split>/pairs_ch{NN}.parquet``.
        scenarios: scenario names to process.
        splits: split names to process (e.g. ``["test"]``).
        n_boot: number of bootstrap draws.
        seed: master RNG seed; per-split seeds are derived deterministically.
        baseline_method: name of the method whose ``CellStats`` is used as the
            paired denominator when forming ``R``. Defaults to
            :data:`BASELINE_CONTINUOUS` (= ``"locf"``). Rows with no baseline
            CellStats in their (scenario, cell) keep ``R = NaN``.
        subgroup_mappings: optional ``{split_name: {sample_idx: {attr: val}}}``
            built externally (mirrors ``aggregate_imputation_pairs.py``).
            When provided, per-method manifests are required to agree on the
            ``sample_idx -> (user_id, date)`` mapping; mismatches raise
            ``ValueError`` before any metric work happens.
        channel_stds: array of length ``n_channels``. If ``None``, loaded
            from each method's ``pairs_root/channel_stds.npy`` and asserted
            to match across methods.
        channel_stds_path: optional explicit path overriding all method dirs.
        include_auc: enable the (slower) AUC bootstrap for binary channels.
        exclude_unknown: skip subgroup_value=="unknown" cells.
        n_channels: number of channels per pairs/<scenario>/<split>/ dir.
        progress_logger: logger callable invoked with progress messages.

    Returns:
        Long-format DataFrame with columns
        ``[method, scenario, split, channel, channel_type, subgroup_attr,
        subgroup_value, draw, E, R]``.
    """
    methods = list(method_dirs.keys())
    if not methods:
        raise ValueError("method_dirs is empty")

    # ------------------ resolve channel_stds ------------------
    if channel_stds is None and channel_stds_path is None:
        ref_method = methods[0]
        channel_stds_path = Path(method_dirs[ref_method]) / "channel_stds.npy"
    if channel_stds is None:
        channel_stds = np.load(channel_stds_path)
    channel_stds = np.asarray(channel_stds, dtype=np.float64)
    if channel_stds.shape[0] < n_channels:
        raise ValueError(f"channel_stds has {channel_stds.shape[0]} entries, need {n_channels}")

    rows: list[dict] = []

    for split in splits:
        sg_map = (subgroup_mappings or {}).get(split)
        cells = _subgroup_cells_from_mapping(sg_map, exclude_unknown=exclude_unknown)
        progress_logger(f"[split={split}] cells = {[f'{a}:{v}' for a, v in cells]}")

        # ---------- Per-split canonical user list (union across methods) ----------
        # Manifests are scenario-independent, so we load them once per split.
        method_manifests: dict[str, pa.Table] = {}
        for method, root in method_dirs.items():
            manifest_path = Path(root) / f"manifest_{split}.parquet"
            if not manifest_path.exists():
                progress_logger(f"  WARN method={method}: {manifest_path} missing — skipping.")
                continue
            method_manifests[method] = pq.read_table(manifest_path)

        if not method_manifests:
            continue

        # When fairness subgroups are requested, the shared sample_idx -> demo
        # map only makes sense if every method's manifest agrees on
        # sample_idx -> (user_id, date). Fail fast if not.
        if sg_map is not None and len(method_manifests) >= 2:
            _assert_manifests_agree(method_manifests, split=split)

        all_user_ids: list[str] = []
        seen: set[str] = set()
        for manifest in method_manifests.values():
            for uid in manifest.column("user_id").to_pylist():
                if uid not in seen:
                    seen.add(uid)
                    all_user_ids.append(uid)
        canonical_user_index = {u: i for i, u in enumerate(all_user_ids)}
        U = len(all_user_ids)
        progress_logger(f"[split={split}] U={U} (union across all methods)")

        # ---------- One resample matrix per split ----------
        split_seed = _seed_for_split(seed, split)
        idx_b = _bootstrap_indices(U, n_boot, split_seed)

        for scenario in scenarios:
            progress_logger(f"[{scenario}/{split}] checking scenario dirs …")

            method_payload: dict[str, dict] = {}
            for method, manifest in method_manifests.items():
                ssd = Path(method_dirs[method]) / scenario / split
                if not ssd.exists():
                    progress_logger(f"  WARN method={method}: {ssd} missing — skipping this cell.")
                    continue
                method_payload[method] = {
                    "scenario_split_dir": ssd,
                    "manifest": manifest,
                }
            if not method_payload:
                continue

            progress_logger(f"[{scenario}/{split}] methods={list(method_payload.keys())}")

            # ------------------ Per-method, per-cell stats ------------------
            method_cell_stats: dict[str, dict[tuple[str, str], CellStats]] = {}
            for method, payload in method_payload.items():
                progress_logger(f"  computing UserStats for method={method}")
                method_cell_stats[method] = compute_user_stats_per_cell(
                    payload["scenario_split_dir"],
                    payload["manifest"],
                    canonical_user_ids=all_user_ids,
                    canonical_user_index=canonical_user_index,
                    cells=cells,
                    sample_to_subgroup=sg_map,
                    n_channels=n_channels,
                    include_binary_rows=include_auc,
                )

            # ------------------ Per-cell paired bootstrap ------------------
            for cell in cells:
                attr, value = cell

                # Precompute baseline-side per_user_auc once per cell (NaN if
                # the baseline is missing from this scenario/split). The
                # baseline's CellStats shares the canonical user index with
                # every method, so a single cache entry serves all methods.
                baseline_cs = method_cell_stats.get(baseline_method, {}).get(cell)
                if baseline_cs is not None and baseline_cs.has_data.any():
                    if include_auc:
                        baseline_per_user_auc = _per_user_auc_from_cell_stats(
                            baseline_cs, n_channels
                        )
                    else:
                        baseline_per_user_auc = None
                else:
                    baseline_per_user_auc = None

                for method, cell_stats_map in method_cell_stats.items():
                    cs = cell_stats_map[cell]
                    if not cs.has_data.any():
                        continue
                    # Precompute the per-(user, channel) AUC matrix once per
                    # (method, scenario, cell); reused by both the per-channel
                    # reducer (for binary task E) and the Part D collapsed
                    # reducer below.
                    if method == baseline_method:
                        per_user_auc = baseline_per_user_auc
                    elif include_auc:
                        per_user_auc = _per_user_auc_from_cell_stats(cs, n_channels)
                    else:
                        per_user_auc = None

                    E = _per_method_cell_errors(
                        cs,
                        idx_b,
                        channel_stds,
                        include_auc=include_auc,
                        per_user_auc=per_user_auc,
                    )

                    # Paired ratio R vs baseline. NaN matrix if baseline is
                    # missing for this cell (paired skill undefined).
                    if baseline_cs is not None:
                        R = _per_method_cell_paired_ratios(
                            cell_stats_method=cs,
                            cell_stats_baseline=baseline_cs,
                            boot_idx=idx_b,
                            include_auc=include_auc,
                            per_user_auc_method=per_user_auc,
                            per_user_auc_baseline=baseline_per_user_auc,
                        )
                    else:
                        R = np.full((n_boot, n_channels), np.nan, dtype=np.float64)

                    for ch in range(n_channels):
                        if not cs.has_data[ch]:
                            continue
                        ch_type = _channel_type(ch)
                        if ch_type == "binary" and scenario in EXCLUDE_BINARY_SCENARIOS:
                            continue
                        col_E = E[:, ch].astype(np.float32)
                        col_R = R[:, ch].astype(np.float32)
                        for b in range(n_boot):
                            val_E = col_E[b]
                            if not np.isfinite(val_E):
                                continue
                            val_R = col_R[b]
                            rows.append(
                                {
                                    "method": method,
                                    "scenario": scenario,
                                    "split": split,
                                    "channel": f"ch_{ch}",
                                    "channel_type": ch_type,
                                    "subgroup_attr": attr,
                                    "subgroup_value": value,
                                    "draw": int(b),
                                    "E": float(val_E),
                                    "R": float(val_R) if np.isfinite(val_R) else float("nan"),
                                }
                            )

                    # --- Part D: emit per-(draw, binary_category) rows -----
                    # Skip on semantic scenarios (same exclusion as the
                    # per-channel binary path) and when AUC isn't requested.
                    if (
                        include_auc
                        and per_user_auc is not None
                        and scenario not in EXCLUDE_BINARY_SCENARIOS
                    ):
                        E_cat = _per_method_cell_collapsed_errors(
                            per_user_auc, idx_b, BINARY_CATEGORIES_ORDERED,
                        )
                        if baseline_per_user_auc is not None:
                            R_cat = _per_method_cell_paired_collapsed_ratios(
                                per_user_auc_method=per_user_auc,
                                per_user_auc_baseline=baseline_per_user_auc,
                                boot_idx=idx_b,
                                binary_categories=BINARY_CATEGORIES_ORDERED,
                            )
                        else:
                            R_cat = np.full(
                                (n_boot, len(BINARY_CATEGORIES_ORDERED)),
                                np.nan,
                                dtype=np.float64,
                            )
                        for cat_idx, (cat_name, _) in enumerate(BINARY_CATEGORIES_ORDERED):
                            col_E = E_cat[:, cat_idx].astype(np.float32)
                            col_R = R_cat[:, cat_idx].astype(np.float32)
                            for b in range(n_boot):
                                val_E = col_E[b]
                                if not np.isfinite(val_E):
                                    continue
                                val_R = col_R[b]
                                rows.append(
                                    {
                                        "method": method,
                                        "scenario": scenario,
                                        "split": split,
                                        "channel": f"cat_collapsed:{cat_name}",
                                        "channel_type": "binary_collapsed",
                                        "subgroup_attr": attr,
                                        "subgroup_value": value,
                                        "draw": int(b),
                                        "E": float(val_E),
                                        "R": float(val_R) if np.isfinite(val_R) else float("nan"),
                                    }
                                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "method",
                "scenario",
                "split",
                "channel",
                "channel_type",
                "subgroup_attr",
                "subgroup_value",
                "draw",
                "E",
                "R",
            ]
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Parquet IO
# --------------------------------------------------------------------------

DRAWS_PARQUET_COLUMNS = [
    "method",
    "scenario",
    "split",
    "channel",
    "channel_type",
    "subgroup_attr",
    "subgroup_value",
    "draw",
    "E",
    "R",
]


def write_draws_parquet(
    df: pd.DataFrame,
    path: Path,
    meta: dict | None = None,
) -> None:
    """Write the per-draw errors DataFrame plus a sidecar JSON of metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df[DRAWS_PARQUET_COLUMNS].copy()
    df["draw"] = df["draw"].astype("int32")
    df["E"] = df["E"].astype("float32")
    df["R"] = df["R"].astype("float32")
    for col in (
        "method",
        "scenario",
        "split",
        "channel",
        "channel_type",
        "subgroup_attr",
        "subgroup_value",
    ):
        df[col] = df[col].astype("category")
    df.to_parquet(path, compression="zstd")
    if meta is not None:
        meta_path = path.with_suffix(path.suffix + ".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2, default=str))


def read_draws_parquet(path: Path) -> tuple[pd.DataFrame, dict | None]:
    """Read the per-draw errors Parquet and its sidecar metadata if present."""
    path = Path(path)
    df = pd.read_parquet(path)
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    meta: dict | None = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    return df, meta


# --------------------------------------------------------------------------
# Phase-2 aggregation: skill, rank, fairness summaries from the draws table
# --------------------------------------------------------------------------


def _skill_for_draw(
    draw_errors: pd.DataFrame,
    *,
    baseline_errors: pd.DataFrame | None = None,
    clip_lower: float,
    clip_upper: float,
) -> pd.DataFrame:
    """Skill score for one draw slice.

    Preferred path: ``draw_errors`` carries the per-task ``R`` column emitted
    by :func:`compute_per_draw_errors` (paired user-bootstrap geomean ratio
    vs LOCF baseline at the per-user grain). ``compute_skill_scores``
    consumes ``R`` directly when present and ``baseline_errors`` is ignored.

    Legacy path: when ``R`` is absent (e.g. the deprecated fairness
    ``S − λ·D`` loop's subgroup-vs-global pairing), pass the global
    ``baseline_errors`` and the function falls back to forming ``E_M /
    E_B`` per task.
    """
    if "R" in draw_errors.columns:
        cols = ["method", "scenario", "channel", "channel_type", "E", "R"]
        return compute_skill_scores(
            draw_errors[cols],
            None,
            clip_lower=clip_lower,
            clip_upper=clip_upper,
        )
    cols = ["method", "scenario", "channel", "channel_type", "E"]
    if baseline_errors is None:
        raise ValueError(
            "_skill_for_draw: draw_errors has no 'R' column and "
            "baseline_errors is None — cannot form the skill ratio."
        )
    return compute_skill_scores(
        draw_errors[cols],
        baseline_errors[cols],
        clip_lower=clip_lower,
        clip_upper=clip_upper,
    )


def _rank_for_draw(draw_errors: pd.DataFrame) -> pd.DataFrame:
    cols = ["method", "scenario", "channel", "channel_type", "E"]
    return compute_average_rankings(draw_errors[cols])


def aggregate_skill_rank_fairness(
    draws_df: pd.DataFrame,
    *,
    baseline_method: str = "locf",
    clip_lower: float = 1e-2,
    clip_upper: float = 100.0,
    lambda_fairness: float = 0.5,
    disparity_fns: dict[str, Callable[[dict[str, float]], float]] | None = None,
    fairness_combine_name: str = "linear_penalty",
    ci_level: float = 0.95,
) -> dict[str, pd.DataFrame]:
    """Phase-2 core: per-draw skill / rank / fairness, summarised across draws.

    ``disparity_fns`` is a ``{name: callable}`` dict — multiple disparities
    can coexist in one pass. Defaults to all four built-ins.

    Returns a dict with four DataFrames:
    - ``skill_scores``       — columns ``method, scope, mean, se, ci_lo, ci_hi, n_boot, n_tasks``
    - ``avg_rankings``       — same shape (mean of ``avg_rank``)
    - ``fairness_subgroups`` — columns ``method, demographic_attr, subgroup,
                                mean, se, ci_lo, ci_hi, n_boot``
    - ``fairness_summary``   — columns ``method, demographic_attr,
                                S_overall_{stats}``, then for each disparity
                                ``disparity_<name>_{stats}`` and
                                ``fairness_adjusted_<name>_{stats}``,
                                plus ``lambda, fairness_combine, n_boot``.

    .. deprecated::
        The ``fairness_subgroups`` / ``fairness_summary`` outputs implement
        the legacy ``S − λ·D`` fairness-adjusted skill score (Family B).
        The leaderboard now uses the disparity-ratio "Fairness Skill Score"
        produced by
        :func:`scripts.paper_results.aggregate_fairness_skill_score.compute_fairness_skill_scores`
        (``fairness_skill_score_bootstrap.csv``). The ``skill_scores`` and
        ``avg_rankings`` outputs of this function are **not** deprecated.
    """
    warnings.warn(
        "aggregate_skill_rank_fairness's fairness outputs "
        "(fairness_subgroups, fairness_summary) implement the deprecated "
        "S − λ·D fairness-adjusted skill score. Use "
        "aggregate_fairness_skill_score.compute_fairness_skill_scores for "
        "the leaderboard's disparity-ratio Fairness Skill Score. The "
        "skill_scores / avg_rankings outputs of this function remain supported.",
        DeprecationWarning,
        stacklevel=2,
    )
    if disparity_fns is None:
        disparity_fns = {n: spec.fn for n, spec in DISPARITY_FUNCTIONS.items()}
    fairness_combine_fn = FAIRNESS_COMBINE[fairness_combine_name]

    if draws_df.empty:
        empty = pd.DataFrame()
        return {
            "skill_scores": empty.copy(),
            "avg_rankings": empty.copy(),
            "fairness_subgroups": empty.copy(),
            "fairness_summary": empty.copy(),
        }

    splits = sorted(draws_df["split"].unique())
    if len(splits) > 1:
        logger.warning(
            "draws_df has multiple splits %s — aggregating each independently",
            splits,
        )

    skill_per_draw_records: list[dict] = []
    rank_per_draw_records: list[dict] = []
    fairness_subgroup_records: list[dict] = []
    fairness_summary_records: list[dict] = []

    for split in splits:
        df_split = draws_df[draws_df["split"] == split]
        df_all = df_split[df_split["subgroup_attr"] == "all"]
        if df_all.empty:
            continue

        # Pre-compute per-draw global baseline + overall skill for
        # fairness_adjusted lookup.
        all_by_draw: dict[int, pd.DataFrame] = {
            int(d): grp for d, grp in df_all.groupby("draw", observed=True)
        }

        for draw, df_draw in all_by_draw.items():
            bl = build_baseline_errors(
                df_draw,
                baseline_continuous=baseline_method,
                baseline_binary=baseline_method,
            )
            if bl.empty:
                continue
            skill = _skill_for_draw(
                df_draw,
                baseline_errors=bl,
                clip_lower=clip_lower,
                clip_upper=clip_upper,
            )
            for _, row in skill.iterrows():
                skill_per_draw_records.append(
                    {
                        "method": row["method"],
                        "scope": row["scope"],
                        "split": split,
                        "skill_score": float(row["skill_score"]),
                        "n_tasks": int(row["n_tasks"]),
                        "draw": int(draw),
                    }
                )
            rank = _rank_for_draw(df_draw)
            for _, row in rank.iterrows():
                rank_per_draw_records.append(
                    {
                        "method": row["method"],
                        "scope": row["scope"],
                        "split": split,
                        "avg_rank": float(row["avg_rank"]),
                        "n_tasks": int(row["n_tasks"]),
                        "draw": int(draw),
                    }
                )

        # Fairness — subgroup S_g uses subgroup model errors against the
        # global baseline at the same draw (keeps pairing coherent).
        sg_attrs = sorted(v for v in df_split["subgroup_attr"].unique() if v != "all")
        for attr in sg_attrs:
            df_attr = df_split[df_split["subgroup_attr"] == attr]
            attr_values = sorted(df_attr["subgroup_value"].unique())
            for draw in sorted(int(d) for d in df_attr["draw"].unique()):
                df_draw_all = all_by_draw.get(draw)
                if df_draw_all is None or df_draw_all.empty:
                    continue
                bl_global = build_baseline_errors(
                    df_draw_all,
                    baseline_continuous=baseline_method,
                    baseline_binary=baseline_method,
                )
                if bl_global.empty:
                    continue
                overall_skill_draw = _skill_for_draw(
                    df_draw_all,
                    baseline_errors=bl_global,
                    clip_lower=clip_lower,
                    clip_upper=clip_upper,
                )
                overall_lookup = (
                    overall_skill_draw[overall_skill_draw["scope"] == "overall"]
                    .set_index("method")["skill_score"]
                    .to_dict()
                )

                df_draw_attr = df_attr[df_attr["draw"] == draw]
                methods = sorted(df_draw_attr["method"].unique())

                for method in methods:
                    group_scores: dict[str, float] = {}
                    for sg_val in attr_values:
                        sub = df_draw_attr[
                            (df_draw_attr["method"] == method)
                            & (df_draw_attr["subgroup_value"] == sg_val)
                        ]
                        if sub.empty:
                            continue
                        skill_sg = _skill_for_draw(
                            sub,
                            baseline_errors=bl_global,
                            clip_lower=clip_lower,
                            clip_upper=clip_upper,
                        )
                        ov = skill_sg[
                            (skill_sg["method"] == method) & (skill_sg["scope"] == "overall")
                        ]
                        if ov.empty:
                            continue
                        s_g = float(ov["skill_score"].iloc[0])
                        group_scores[sg_val] = s_g
                        fairness_subgroup_records.append(
                            {
                                "method": method,
                                "demographic_attr": attr,
                                "subgroup": sg_val,
                                "split": split,
                                "S_g": s_g,
                                "draw": int(draw),
                            }
                        )

                    if len(group_scores) < 2:
                        continue
                    s_overall = overall_lookup.get(method, np.nan)
                    rec: dict = {
                        "method": method,
                        "demographic_attr": attr,
                        "split": split,
                        "draw": int(draw),
                        "S_overall": float(s_overall) if np.isfinite(s_overall) else float("nan"),
                    }
                    for name, fn in disparity_fns.items():
                        d = float(fn(group_scores))
                        rec[f"disparity_{name}"] = d
                        if np.isfinite(s_overall) and np.isfinite(d):
                            # The combine fn applies ``S − λ·d_eff``. For
                            # disparities where *higher* values mean *fairer*
                            # (e.g. ``worst_group`` = min(S_g)), flip the sign
                            # so the fairness-adjusted score still rewards
                            # better fairness. Lower-is-fairer disparities
                            # (``max_minus_min``, ``std``, ``relative_drop``)
                            # are passed through unchanged.
                            d_eff = -d if disparity_higher_is_better(name) else d
                            rec[f"fairness_adjusted_{name}"] = float(
                                fairness_combine_fn(
                                    float(s_overall),
                                    d_eff,
                                    lambda_fairness,
                                )
                            )
                        else:
                            rec[f"fairness_adjusted_{name}"] = float("nan")
                    fairness_summary_records.append(rec)

    # ------------------ summarise across draws ------------------
    def _summary_table(
        records: list[dict], value_col: str, key_cols: list[str], extra_cols: list[str]
    ) -> pd.DataFrame:
        if not records:
            return pd.DataFrame(
                columns=key_cols
                + extra_cols
                + [
                    "mean",
                    "se",
                    "ci_lo",
                    "ci_hi",
                    "n_boot",
                ]
            )
        df = pd.DataFrame(records)
        out_rows = []
        for keys, grp in df.groupby(key_cols, observed=True):
            summary = _summarize(grp[value_col].to_numpy(), ci_level)
            row = dict(zip(key_cols, keys if isinstance(keys, tuple) else (keys,)))
            for ec in extra_cols:
                if ec in grp.columns:
                    # take first (constant per group expected)
                    row[ec] = grp[ec].iloc[0]
            row.update(
                {
                    "mean": summary["bootstrap_mean"],
                    "se": summary["bootstrap_se"],
                    "ci_lo": summary["ci_lo"],
                    "ci_hi": summary["ci_hi"],
                    "n_boot": summary["n_valid_boot"],
                }
            )
            out_rows.append(row)
        return pd.DataFrame(out_rows)

    skill_table = _summary_table(
        skill_per_draw_records,
        "skill_score",
        key_cols=["method", "scope", "split"],
        extra_cols=["n_tasks"],
    )
    rank_table = _summary_table(
        rank_per_draw_records,
        "avg_rank",
        key_cols=["method", "scope", "split"],
        extra_cols=["n_tasks"],
    )
    fairness_subgroup_table = _summary_table(
        fairness_subgroup_records,
        "S_g",
        key_cols=["method", "demographic_attr", "subgroup", "split"],
        extra_cols=[],
    )

    # Fairness summary: multiple value cols (S_overall + per-disparity)
    if fairness_summary_records:
        fs_df = pd.DataFrame(fairness_summary_records)
        value_cols = [
            c
            for c in fs_df.columns
            if c.startswith(("S_overall", "disparity_", "fairness_adjusted_"))
        ]
        out_rows = []
        for keys, grp in fs_df.groupby(["method", "demographic_attr", "split"], observed=True):
            row = {
                "method": keys[0],
                "demographic_attr": keys[1],
                "split": keys[2],
                "lambda": lambda_fairness,
                "fairness_combine": fairness_combine_name,
            }
            n_boot_seen = len(grp)
            row["n_boot"] = n_boot_seen
            for vc in value_cols:
                summary = _summarize(grp[vc].to_numpy(), ci_level)
                row[f"{vc}_mean"] = summary["bootstrap_mean"]
                row[f"{vc}_se"] = summary["bootstrap_se"]
                row[f"{vc}_ci_lo"] = summary["ci_lo"]
                row[f"{vc}_ci_hi"] = summary["ci_hi"]
            out_rows.append(row)
        fairness_summary_table = pd.DataFrame(out_rows)
    else:
        fairness_summary_table = pd.DataFrame()

    return {
        "skill_scores": skill_table,
        "avg_rankings": rank_table,
        "fairness_subgroups": fairness_subgroup_table,
        "fairness_summary": fairness_summary_table,
    }
