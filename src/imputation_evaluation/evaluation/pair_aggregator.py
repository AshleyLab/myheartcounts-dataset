"""Aggregate metrics from saved per-channel (gt, pred) pair Parquet files.

Reads per-channel Parquet files written by PairWriter, and computes user-macro
metrics: per-user metrics first (``sae_u / n_u`` for MAE; pooled per-user AUC
for binary), then ``nanmean`` across users.

Each channel file is loaded and freed independently, keeping peak memory to
~1/19th of the old single-file approach.

The user-macro reducer mirrors the leaderboard estimand produced by the
:mod:`imputation_evaluation.evaluation.bootstrap_skill_rank` reducers — so
the live point estimate equals the bootstrap identity-draw point estimate.

Phase-A note: the legacy cell-micro path has been deleted. The aggregator
requires a sample manifest at ``<pairs_root>/manifest_<split>.parquet``; if
missing, the call raises ``FileNotFoundError``.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

from data.processing.hf_config import CONTINUOUS_CHANNEL_INDICES, N_CHANNELS
from imputation_evaluation.evaluation.paper_metrics_core import (
    BINARY_CATEGORIES_ORDERED,
)

logger = logging.getLogger(__name__)


def _channel_file(pairs_dir: Path, ch: int) -> Path:
    return pairs_dir / f"pairs_ch{ch:02d}.parquet"


# ---------------------------------------------------------------------------
# Manifest discovery
# ---------------------------------------------------------------------------


def _discover_manifest(pairs_dir: Path) -> Path | None:
    """Return the manifest sibling for ``pairs_dir = <root>/<scenario>/<split>``.

    Returns ``None`` if no manifest is found at the conventional location.
    """
    split = pairs_dir.name
    candidate = pairs_dir.parent.parent / f"manifest_{split}.parquet"
    return candidate if candidate.exists() else None


def _load_sample_to_user(manifest_path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load ``(canonical_user_ids, sidx_arr, full_u_rows)`` from a manifest.

    ``canonical_user_ids`` is the ordered unique user list; ``sidx_arr`` is
    the manifest's ``sample_idx`` column; ``full_u_rows`` maps each manifest
    row to its position in ``canonical_user_ids``.
    """
    table = pq.read_table(manifest_path, columns=["sample_idx", "user_id"])
    sidx_arr = table.column("sample_idx").to_numpy()
    uid_arr = table.column("user_id").to_pylist()

    seen: set[str] = set()
    canonical_user_ids: list[str] = []
    for uid in uid_arr:
        if uid not in seen:
            seen.add(uid)
            canonical_user_ids.append(uid)
    canonical_user_index = {u: i for i, u in enumerate(canonical_user_ids)}
    full_u_rows = np.array([canonical_user_index[u] for u in uid_arr], dtype=np.int64)
    return canonical_user_ids, sidx_arr, full_u_rows


# ---------------------------------------------------------------------------
# Per-user accumulators (user-macro reducer)
# ---------------------------------------------------------------------------


def _aggregate_user_macro_one_cell(
    pairs_dir: Path,
    channel_stds: np.ndarray,
    *,
    canonical_user_ids: list[str],
    sidx_arr: np.ndarray,
    full_u_rows: np.ndarray,
    keep_mask: np.ndarray | None = None,
    n_channels: int = N_CHANNELS,
    return_per_user: bool = False,
) -> dict:
    """User-macro per-channel metrics for one cell (all rows, or a subgroup mask).

    Mirrors the bootstrap reducer (:func:`bootstrap_skill_rank.compute_user_stats_per_cell`
    + :func:`bootstrap_skill_rank._per_method_cell_errors`) at the deterministic
    point grain — i.e. per-user MAE/RMSE/AUC computed once, then ``nanmean``
    over users.

    ``keep_mask`` is a manifest-row-aligned boolean mask used by the subgroup
    variant. ``None`` means "keep every manifest row" (the ``ALL_KEY`` cell).

    When ``return_per_user=True``, also emit ``metrics["per_user"]`` as a
    ``{ch_key: {user_id: E_user}}`` map covering the 19 real per-channel
    tasks and the two synthetic collapsed-binary tasks
    (``cat_collapsed:sleep`` / ``cat_collapsed:workouts``). Per-user E is the
    same unfloored quantity the bootstrap consumes:

      * continuous: ``E_user = sae[user, ch] / n[user, ch]`` (per-user MAE).
      * binary per-channel: ``E_user = 1 − AUC[user, ch]`` (unfloored — the
        ``BINARY_ERROR_FLOOR`` applies only to paired skill ratios).
      * binary collapsed (``cat_collapsed:sleep`` / ``cat_collapsed:workouts``):
        ``E_user = nanmean`` over the category's real binary channels of
        ``1 − AUC[user, ch]`` — mirrors
        :func:`bootstrap_skill_rank._per_method_cell_collapsed_errors` at the
        per-user grain.
    """
    n_channels = min(n_channels, len(channel_stds))
    U = len(canonical_user_ids)

    # Per-(user, channel) sufficient stats — mirror CellStats's relevant
    # fields. We only need sse / sae / n for continuous and per-user AUC for
    # binary; balanced_accuracy is computed from per-user tp/tn/fp/fn.
    sse = np.zeros((U, n_channels), dtype=np.float64)
    sae = np.zeros((U, n_channels), dtype=np.float64)
    n = np.zeros((U, n_channels), dtype=np.int64)
    tp = np.zeros((U, n_channels), dtype=np.int64)
    tn = np.zeros((U, n_channels), dtype=np.int64)
    fp = np.zeros((U, n_channels), dtype=np.int64)
    fn = np.zeros((U, n_channels), dtype=np.int64)
    has_data = np.zeros(n_channels, dtype=bool)

    # Per-(user, channel) raw rows for AUC — pooled across the user's cells
    # before scoring once (mirrors forecasting "pooled binary metrics").
    binary_rows_gt: dict[int, list[np.ndarray]] = {}
    binary_rows_pred: dict[int, list[np.ndarray]] = {}
    binary_rows_user: dict[int, list[np.ndarray]] = {}

    # Map sample_idx -> manifest-row so we can lift the keep_mask onto pair rows.
    sidx_to_manifest_row = np.full(
        int(sidx_arr.max()) + 1 if sidx_arr.size else 0, -1, dtype=np.int64
    )
    for i, sidx in enumerate(sidx_arr):
        sidx_to_manifest_row[int(sidx)] = i

    unique_samples: set[int] = set()

    for ch in range(n_channels):
        ch_file = _channel_file(pairs_dir, ch)
        if not ch_file.exists():
            continue
        table = pq.read_table(ch_file, columns=["sample_idx", "gt", "pred"])
        if table.num_rows == 0:
            continue

        pair_sidx = table.column("sample_idx").to_numpy()
        pair_manifest_rows = sidx_to_manifest_row[pair_sidx]
        # Pair rows whose sample_idx isn't in the manifest are dropped — same
        # tolerance as MetricAccumulator. (compute_user_stats_per_cell raises;
        # the live path is more permissive on partial inputs.)
        in_manifest = pair_manifest_rows >= 0
        if not in_manifest.all():
            pair_sidx = pair_sidx[in_manifest]
            pair_manifest_rows = pair_manifest_rows[in_manifest]
            keep_indices = np.flatnonzero(in_manifest)
        else:
            keep_indices = None
        if keep_mask is not None:
            row_mask = keep_mask[pair_manifest_rows]
            if not row_mask.any():
                continue
        else:
            row_mask = np.ones(pair_sidx.size, dtype=bool)
        u_rows_pair = full_u_rows[pair_manifest_rows][row_mask]
        unique_samples.update(np.unique(pair_sidx[row_mask]).tolist())

        gt_col = table.column("gt")
        pred_col = table.column("pred")

        if ch in CONTINUOUS_CHANNEL_INDICES:
            gt_ch = gt_col.to_numpy().astype(np.float32)
            pred_ch = pred_col.to_numpy().astype(np.float32)
            if keep_indices is not None:
                gt_ch = gt_ch[keep_indices]
                pred_ch = pred_ch[keep_indices]
            gt_ch = gt_ch[row_mask]
            pred_ch = pred_ch[row_mask]
            err = (pred_ch - gt_ch).astype(np.float64)
            err_sq = err * err
            err_abs = np.abs(err)
            has_data[ch] = True
            np.add.at(n[:, ch], u_rows_pair, 1)
            np.add.at(sse[:, ch], u_rows_pair, err_sq)
            np.add.at(sae[:, ch], u_rows_pair, err_abs)
        else:
            gt_bool = gt_col.to_numpy().astype(bool)
            pred_ch = pred_col.to_numpy().astype(np.float32)
            if keep_indices is not None:
                gt_bool = gt_bool[keep_indices]
                pred_ch = pred_ch[keep_indices]
            gt_bool = gt_bool[row_mask]
            pred_ch = pred_ch[row_mask]
            pred_bool = pred_ch > 0.5

            has_data[ch] = True
            tp_mask = gt_bool & pred_bool
            tn_mask = (~gt_bool) & (~pred_bool)
            fp_mask = (~gt_bool) & pred_bool
            fn_mask = gt_bool & (~pred_bool)
            np.add.at(tp[:, ch], u_rows_pair, tp_mask.astype(np.int64))
            np.add.at(tn[:, ch], u_rows_pair, tn_mask.astype(np.int64))
            np.add.at(fp[:, ch], u_rows_pair, fp_mask.astype(np.int64))
            np.add.at(fn[:, ch], u_rows_pair, fn_mask.astype(np.int64))
            binary_rows_gt.setdefault(ch, []).append(gt_bool)
            binary_rows_pred.setdefault(ch, []).append(pred_ch)
            binary_rows_user.setdefault(ch, []).append(u_rows_pair)

        del table, gt_col, pred_col

    # --- Build per_channel dict + macro aggregates -----------------------
    metrics: dict = {
        "n_samples": len(unique_samples),
        "per_channel": {},
        "continuous": {},
        "binary": {},
    }

    user_macro_maes: list[float] = []
    user_macro_normalized_maes: list[float] = []
    user_macro_rmses: list[float] = []
    user_macro_normalized_rmses: list[float] = []
    user_macro_normalized_mses: list[float] = []
    binary_balanced_accs: list[float] = []
    binary_roc_aucs: list[float] = []

    # When return_per_user is set, keep the per-(user, channel) error matrices
    # so we can emit the long ``per_user`` map and build collapsed-binary
    # per-user E at the end. Continuous channels keep per_user_mae[U, ch];
    # binary channels keep per_user_auc[U, ch] (used unfloored as 1−AUC for
    # E and as the input to the collapsed-category nanmean).
    per_user_mae_matrix: np.ndarray | None = None
    per_user_auc_matrix: np.ndarray | None = None
    if return_per_user:
        per_user_mae_matrix = np.full((U, n_channels), np.nan, dtype=np.float64)
        per_user_auc_matrix = np.full((U, n_channels), np.nan, dtype=np.float64)

    for ch in range(n_channels):
        ch_metrics: dict = {"channel_idx": ch}
        if not has_data[ch]:
            ch_metrics["n_masked"] = 0
            ch_metrics["error"] = "no_masked_positions"
            metrics["per_channel"][f"ch_{ch}"] = ch_metrics
            continue

        if ch in CONTINUOUS_CHANNEL_INDICES:
            n_total = int(n[:, ch].sum())
            ch_metrics["n_masked"] = n_total
            ch_std = float(channel_stds[ch]) if channel_stds[ch] > 0 else 1.0

            with np.errstate(divide="ignore", invalid="ignore"):
                per_user_mae = np.where(n[:, ch] > 0, sae[:, ch] / np.maximum(n[:, ch], 1), np.nan)
                per_user_mse = np.where(n[:, ch] > 0, sse[:, ch] / np.maximum(n[:, ch], 1), np.nan)
            per_user_rmse = np.sqrt(per_user_mse)

            if per_user_mae_matrix is not None:
                per_user_mae_matrix[:, ch] = per_user_mae

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                user_mae = float(np.nanmean(per_user_mae))
                user_mse = float(np.nanmean(per_user_mse))
                user_rmse = float(np.nanmean(per_user_rmse))

            ch_metrics["mae"] = user_mae
            ch_metrics["mse"] = user_mse
            ch_metrics["rmse"] = user_rmse
            ch_metrics["normalized_mae"] = user_mae / ch_std
            ch_metrics["normalized_mse"] = user_mse / (ch_std * ch_std)
            ch_metrics["normalized_rmse"] = user_rmse / ch_std
            ch_metrics["aggregation"] = "user_macro"

            if np.isfinite(user_mae):
                user_macro_maes.append(user_mae)
                user_macro_normalized_maes.append(ch_metrics["normalized_mae"])
                user_macro_rmses.append(user_rmse)
                user_macro_normalized_rmses.append(ch_metrics["normalized_rmse"])
                user_macro_normalized_mses.append(ch_metrics["normalized_mse"])
        else:
            # --- Binary: per-user pooled AUC, then user-macro --------------
            tp_ch = tp[:, ch]
            tn_ch = tn[:, ch]
            fp_ch = fp[:, ch]
            fn_ch = fn[:, ch]
            n_total = int((tp_ch + tn_ch + fp_ch + fn_ch).sum())
            ch_metrics["n_masked"] = n_total

            with np.errstate(divide="ignore", invalid="ignore"):
                tpr = np.where((tp_ch + fn_ch) > 0, tp_ch / np.maximum(tp_ch + fn_ch, 1), np.nan)
                tnr = np.where((tn_ch + fp_ch) > 0, tn_ch / np.maximum(tn_ch + fp_ch, 1), np.nan)
            per_user_bal_acc = 0.5 * (tpr + tnr)

            # Per-user AUC via pooled rows.
            gt_concat = (
                np.concatenate(binary_rows_gt.get(ch, []))
                if binary_rows_gt.get(ch)
                else np.array([], dtype=bool)
            )
            pred_concat = (
                np.concatenate(binary_rows_pred.get(ch, []))
                if binary_rows_pred.get(ch)
                else np.array([], dtype=np.float32)
            )
            u_concat = (
                np.concatenate(binary_rows_user.get(ch, []))
                if binary_rows_user.get(ch)
                else np.array([], dtype=np.int64)
            )

            per_user_auc = np.full(U, np.nan, dtype=np.float64)
            if u_concat.size:
                for u in np.unique(u_concat):
                    mask_u = u_concat == u
                    gt_u = gt_concat[mask_u]
                    if gt_u.size == 0 or gt_u.all() or not gt_u.any():
                        continue  # single-class → AUC undefined
                    try:
                        per_user_auc[u] = float(roc_auc_score(gt_u, pred_concat[mask_u]))
                    except Exception:
                        pass

            if per_user_auc_matrix is not None:
                per_user_auc_matrix[:, ch] = per_user_auc

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                user_balanced_acc = float(np.nanmean(per_user_bal_acc))
                user_auc = float(np.nanmean(per_user_auc))

            ch_metrics["balanced_accuracy"] = user_balanced_acc
            ch_metrics["roc_auc"] = user_auc
            ch_metrics["aggregation"] = "user_macro"

            if np.isfinite(user_balanced_acc):
                binary_balanced_accs.append(user_balanced_acc)
            if np.isfinite(user_auc):
                binary_roc_aucs.append(user_auc)

        metrics["per_channel"][f"ch_{ch}"] = ch_metrics

    # --- Cross-channel macros (headline) ---------------------------------
    if user_macro_normalized_maes:
        metrics["continuous"]["mean_normalized_mae"] = float(np.mean(user_macro_normalized_maes))
        metrics["continuous"]["mean_normalized_rmse"] = float(np.mean(user_macro_normalized_rmses))
        metrics["continuous"]["mean_normalized_mse"] = float(np.mean(user_macro_normalized_mses))
        metrics["continuous"]["mean_mae"] = float(np.mean(user_macro_maes))
        metrics["continuous"]["mean_rmse"] = float(np.mean(user_macro_rmses))
        metrics["continuous"]["n_channels"] = len(user_macro_maes)
    else:
        metrics["continuous"]["mean_normalized_mae"] = float("nan")
        metrics["continuous"]["mean_normalized_rmse"] = float("nan")
        metrics["continuous"]["mean_normalized_mse"] = float("nan")
        metrics["continuous"]["mean_mae"] = float("nan")
        metrics["continuous"]["mean_rmse"] = float("nan")
        metrics["continuous"]["n_channels"] = 0
    metrics["continuous"]["aggregation"] = "user_macro"

    if binary_balanced_accs:
        metrics["binary"]["macro_balanced_accuracy"] = float(np.mean(binary_balanced_accs))
        metrics["binary"]["n_channels"] = len(binary_balanced_accs)
    else:
        metrics["binary"]["macro_balanced_accuracy"] = float("nan")
        metrics["binary"]["n_channels"] = 0
    metrics["binary"]["macro_roc_auc"] = (
        float(np.mean(binary_roc_aucs)) if binary_roc_aucs else float("nan")
    )
    metrics["binary"]["aggregation"] = "user_macro"

    # --- Optional: per-user long emission for the aligned rank/skill flows -
    if return_per_user:
        per_user_map: dict[str, dict[str, float]] = {}
        # Per-channel tasks — same E definition as bootstrap's
        # ``_per_method_cell_errors``: continuous = MAE = sae/n, binary =
        # 1 − AUC (unfloored).
        for ch in range(n_channels):
            if not has_data[ch]:
                continue
            ch_key = f"ch_{ch}"
            user_map: dict[str, float] = {}
            if ch in CONTINUOUS_CHANNEL_INDICES:
                col = per_user_mae_matrix[:, ch]  # type: ignore[index]
                for u_idx, val in enumerate(col):
                    if np.isfinite(val):
                        user_map[canonical_user_ids[u_idx]] = float(val)
            else:
                col = per_user_auc_matrix[:, ch]  # type: ignore[index]
                for u_idx, val in enumerate(col):
                    if np.isfinite(val):
                        user_map[canonical_user_ids[u_idx]] = float(1.0 - val)
            if user_map:
                per_user_map[ch_key] = user_map

        # Collapsed-binary tasks — mirror
        # ``_per_method_cell_collapsed_errors`` at the per-user grain:
        # ``E_user = nanmean over the category's channels of (1 − AUC[user, ch])``.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for cat_name, ch_indices in BINARY_CATEGORIES_ORDERED:
                cols = np.asarray(ch_indices, dtype=np.int64)
                # Restrict to channels with any data in this cell.
                cols = cols[has_data[cols]]
                if cols.size == 0:
                    continue
                sub = per_user_auc_matrix[:, cols]  # type: ignore[index]
                per_user_cat_E = np.nanmean(1.0 - sub, axis=1)
                user_map = {}
                for u_idx, val in enumerate(per_user_cat_E):
                    if np.isfinite(val):
                        user_map[canonical_user_ids[u_idx]] = float(val)
                if user_map:
                    per_user_map[f"cat_collapsed:{cat_name}"] = user_map

        metrics["per_user"] = per_user_map

    return metrics


def aggregate_pairs(
    pairs_path: str | Path,
    channel_stds: np.ndarray,
    *,
    manifest_path: str | Path | None = None,
    return_per_user: bool = False,
) -> dict:
    """Compute user-macro metrics from saved per-channel pair Parquet files.

    Reads per-channel files one at a time, computing metrics and freeing
    memory before moving to the next channel.

    Args:
        pairs_path: Path to the pairs directory (containing
            ``pairs_ch*.parquet``).
        channel_stds: Per-channel standard deviations for normalization,
            shape ``(C,)``.
        manifest_path: Optional explicit path to the manifest parquet (cols:
            ``sample_idx``, ``user_id``). When ``None``, looks for
            ``<pairs_root>/manifest_<split>.parquet`` by convention.
        return_per_user: When ``True``, also include a ``"per_user"`` entry
            mapping each channel key to a ``{user_id: E_user}`` map of
            unfloored per-user errors.

    Returns:
        Metrics dict with per-channel + aggregate fields. The reducer is
        always user-macro (the only supported aggregation after Phase A).

    Raises:
        FileNotFoundError: If no per-channel pair files are found, or the
            sample manifest is missing.
    """
    pairs_path = Path(pairs_path)
    if pairs_path.is_file():
        pairs_dir = pairs_path.parent
    else:
        pairs_dir = pairs_path

    n_channels = min(N_CHANNELS, len(channel_stds))
    any_channel_file = any(_channel_file(pairs_dir, ch).exists() for ch in range(n_channels))

    if not any_channel_file:
        raise FileNotFoundError(
            f"No per-channel pair files (pairs_ch*.parquet) found under {pairs_dir}"
        )

    resolved = (
        Path(manifest_path) if manifest_path is not None else _discover_manifest(pairs_dir)
    )
    if resolved is None or not resolved.exists():
        searched = manifest_path or (
            pairs_dir.parent.parent / f"manifest_{pairs_dir.name}.parquet"
        )
        raise FileNotFoundError(
            f"aggregate_pairs requires a sample manifest at {searched}; cell-micro "
            "fallback was removed in Phase A."
        )
    canonical_user_ids, sidx_arr, full_u_rows = _load_sample_to_user(resolved)
    return _aggregate_user_macro_one_cell(
        pairs_dir,
        channel_stds,
        canonical_user_ids=canonical_user_ids,
        sidx_arr=sidx_arr,
        full_u_rows=full_u_rows,
        keep_mask=None,
        n_channels=n_channels,
        return_per_user=return_per_user,
    )


def aggregate_pairs_by_subgroup(
    pairs_path: str | Path,
    channel_stds: np.ndarray,
    subgroup_mapping: dict[int, dict[str, str]],
    *,
    manifest_path: str | Path | None = None,
    return_per_user: bool = False,
) -> dict[str, dict[str, dict]]:
    """Per-subgroup user-macro metrics from saved per-channel pair Parquet files.

    Reads each per-channel file once, partitions rows by subgroup via
    ``sample_idx`` lookup, and computes user-macro metrics per group —
    following the same channel-by-channel memory pattern as
    :func:`aggregate_pairs`.

    Args:
        pairs_path: Directory containing ``pairs_ch*.parquet`` files.
        channel_stds: Per-channel standard deviations for normalization,
            shape ``(C,)``.
        subgroup_mapping: Maps ``sample_idx`` to attribute dicts,
            e.g. ``{0: {"age_group": "30-39", "sex": "male"}, ...}``.
        manifest_path: As in :func:`aggregate_pairs`.
        return_per_user: As in :func:`aggregate_pairs`; forwarded to each
            subgroup's metrics computation.

    Returns:
        ``{attr: {group_name: metrics_dict}}`` where ``metrics_dict``
        matches the format of :func:`aggregate_pairs`.

    Raises:
        FileNotFoundError: If the manifest sibling is missing.
    """
    pairs_path = Path(pairs_path)
    if pairs_path.is_file():
        pairs_path = pairs_path.parent

    attrs: set[str] = set()
    for demo in subgroup_mapping.values():
        attrs.update(demo.keys())
    attrs_list = sorted(attrs)
    if not attrs_list:
        return {}

    n_channels = min(N_CHANNELS, len(channel_stds))

    resolved = (
        Path(manifest_path) if manifest_path is not None else _discover_manifest(pairs_path)
    )
    if resolved is None or not resolved.exists():
        searched = manifest_path or (
            pairs_path.parent.parent / f"manifest_{pairs_path.name}.parquet"
        )
        raise FileNotFoundError(
            f"aggregate_pairs_by_subgroup requires a sample manifest at {searched}; "
            "cell-micro fallback was removed in Phase A."
        )

    canonical_user_ids, sidx_arr, full_u_rows = _load_sample_to_user(resolved)

    # Build (attr, group) -> keep_mask over manifest rows once.
    n_manifest = sidx_arr.size
    cells_by_attr: dict[str, dict[str, np.ndarray]] = {a: {} for a in attrs_list}
    for attr in attrs_list:
        group_masks: dict[str, np.ndarray] = {}
        for i, sidx in enumerate(sidx_arr):
            demo = subgroup_mapping.get(int(sidx), {})
            group = demo.get(attr, "unknown")
            mask = group_masks.get(group)
            if mask is None:
                mask = np.zeros(n_manifest, dtype=bool)
                group_masks[group] = mask
            mask[i] = True
        cells_by_attr[attr] = group_masks

    result: dict[str, dict[str, dict]] = {}
    for attr in attrs_list:
        result[attr] = {}
        for group_name, keep_mask in cells_by_attr[attr].items():
            if not keep_mask.any():
                continue
            result[attr][group_name] = _aggregate_user_macro_one_cell(
                pairs_path,
                channel_stds,
                canonical_user_ids=canonical_user_ids,
                sidx_arr=sidx_arr,
                full_u_rows=full_u_rows,
                keep_mask=keep_mask,
                n_channels=n_channels,
                return_per_user=return_per_user,
            )
    return result
