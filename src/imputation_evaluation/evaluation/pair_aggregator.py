"""Aggregate metrics from saved per-channel (gt, pred) pair Parquet files.

Reads per-channel Parquet files written by PairWriter, and computes the same
metrics as MetricAccumulator.compute(). Each channel file is loaded and freed
independently, keeping peak memory to ~1/19th of the old single-file approach.

Default aggregation is **user-macro**: per-user metrics first
(``sae_u / n_u`` for MAE; pooled per-user AUC for binary), then ``nanmean``
across users. This mirrors the leaderboard estimand produced by the
:mod:`imputation_evaluation.evaluation.bootstrap_skill_rank` reducers — so
the live point estimate equals the bootstrap identity-draw point estimate.
The legacy **cell-micro** path (pool all rows globally per channel) is kept
as a fallback when no manifest is available.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from data.processing.hf_config import CONTINUOUS_CHANNEL_INDICES, N_CHANNELS

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
) -> dict:
    """User-macro per-channel metrics for one cell (all rows, or a subgroup mask).

    Mirrors the bootstrap reducer (:func:`bootstrap_skill_rank.compute_user_stats_per_cell`
    + :func:`bootstrap_skill_rank._per_method_cell_errors`) at the deterministic
    point grain — i.e. per-user MAE/RMSE/AUC computed once, then ``nanmean``
    over users.

    ``keep_mask`` is a manifest-row-aligned boolean mask used by the subgroup
    variant. ``None`` means "keep every manifest row" (the ``ALL_KEY`` cell).
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

    return metrics


def aggregate_pairs(
    pairs_path: str | Path,
    channel_stds: np.ndarray,
    *,
    manifest_path: str | Path | None = None,
    aggregation: str = "user_macro",
) -> dict:
    """Compute metrics from saved per-channel pair Parquet files.

    Reads per-channel files one at a time, computing metrics and freeing
    memory before moving to the next channel.

    Args:
        pairs_path: Path to the pairs directory (containing
            ``pairs_ch*.parquet``). Also accepts a file path for backward
            compatibility with old ``pairs.parquet`` format (logs a warning).
        channel_stds: Per-channel standard deviations for normalization,
            shape ``(C,)``.
        manifest_path: Optional explicit path to the manifest parquet (cols:
            ``sample_idx``, ``user_id``). When ``None`` and
            ``aggregation="user_macro"``, looks for
            ``<pairs_root>/manifest_<split>.parquet`` by convention. If still
            missing, logs a warning and falls back to cell-micro.
        aggregation: ``"user_macro"`` (default — leaderboard estimand,
            matches the bootstrap identity draw) or ``"cell_micro"`` (legacy
            global pool over all rows per channel; kept for the dev-tools
            comparison scripts and as a fallback when no manifest is
            available).

    Returns:
        Metrics dict matching the format of
        ``MetricAccumulator.compute()``, with per-channel
        ``"aggregation"`` and aggregate ``continuous.aggregation`` /
        ``binary.aggregation`` keys distinguishing the two modes.
    """
    if aggregation not in {"user_macro", "cell_micro"}:
        raise ValueError(f"aggregation must be 'user_macro' or 'cell_micro', got {aggregation!r}")

    pairs_path = Path(pairs_path)
    if pairs_path.is_file():
        pairs_dir = pairs_path.parent
    else:
        pairs_dir = pairs_path

    n_channels = min(N_CHANNELS, len(channel_stds))
    any_channel_file = any(_channel_file(pairs_dir, ch).exists() for ch in range(n_channels))

    if not any_channel_file:
        old_file = pairs_dir / "pairs.parquet"
        if old_file.exists():
            logger.warning(
                "Found old-format pairs.parquet. Re-run imputation eval to generate "
                "per-channel files for lower memory usage."
            )
            return _aggregate_legacy(old_file, channel_stds)
        return {"error": "pairs_file_not_found", "n_samples": 0}

    if aggregation == "user_macro":
        resolved = (
            Path(manifest_path) if manifest_path is not None else _discover_manifest(pairs_dir)
        )
        if resolved is None or not resolved.exists():
            logger.warning(
                "aggregate_pairs: aggregation='user_macro' requested but no manifest "
                "found (searched %s). Falling back to cell_micro.",
                manifest_path or (pairs_dir.parent.parent / f"manifest_{pairs_dir.name}.parquet"),
            )
            return _aggregate_cell_micro(pairs_dir, channel_stds)
        canonical_user_ids, sidx_arr, full_u_rows = _load_sample_to_user(resolved)
        return _aggregate_user_macro_one_cell(
            pairs_dir,
            channel_stds,
            canonical_user_ids=canonical_user_ids,
            sidx_arr=sidx_arr,
            full_u_rows=full_u_rows,
            keep_mask=None,
            n_channels=n_channels,
        )

    return _aggregate_cell_micro(pairs_dir, channel_stds)


def aggregate_pairs_by_subgroup(
    pairs_path: str | Path,
    channel_stds: np.ndarray,
    subgroup_mapping: dict[int, dict[str, str]],
    *,
    manifest_path: str | Path | None = None,
    aggregation: str = "user_macro",
) -> dict[str, dict[str, dict]]:
    """Per-subgroup metrics from saved per-channel pair Parquet files.

    Reads each per-channel file once, partitions rows by subgroup via
    ``sample_idx`` lookup, and computes metrics per group — following
    the same channel-by-channel memory pattern as :func:`aggregate_pairs`.

    Aggregation semantics mirror :func:`aggregate_pairs`: default
    ``"user_macro"`` reduces over users within each subgroup cell;
    ``"cell_micro"`` keeps the legacy global-pool behaviour.

    Args:
        pairs_path: Directory containing ``pairs_ch*.parquet`` files.
        channel_stds: Per-channel standard deviations for normalization,
            shape ``(C,)``.
        subgroup_mapping: Maps ``sample_idx`` to attribute dicts,
            e.g. ``{0: {"age_group": "30-39", "sex": "male"}, ...}``.
        manifest_path: As in :func:`aggregate_pairs`.
        aggregation: As in :func:`aggregate_pairs`.

    Returns:
        ``{attr: {group_name: metrics_dict}}`` where ``metrics_dict``
        matches the format of :func:`aggregate_pairs`.
    """
    if aggregation not in {"user_macro", "cell_micro"}:
        raise ValueError(f"aggregation must be 'user_macro' or 'cell_micro', got {aggregation!r}")

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

    if aggregation == "user_macro":
        resolved = (
            Path(manifest_path) if manifest_path is not None else _discover_manifest(pairs_path)
        )
        if resolved is None or not resolved.exists():
            logger.warning(
                "aggregate_pairs_by_subgroup: user_macro requested but no manifest "
                "found — falling back to cell_micro."
            )
            return _aggregate_by_subgroup_cell_micro(
                pairs_path, channel_stds, subgroup_mapping, attrs_list, n_channels
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
                )
        return result

    return _aggregate_by_subgroup_cell_micro(
        pairs_path, channel_stds, subgroup_mapping, attrs_list, n_channels
    )


# ---------------------------------------------------------------------------
# Cell-micro legacy reducers (kept as fallback / dev-tools comparison)
# ---------------------------------------------------------------------------


def _aggregate_cell_micro(
    pairs_dir: Path,
    channel_stds: np.ndarray,
) -> dict:
    """Cell-micro reducer (legacy): pool all rows per channel, score once."""
    n_channels = min(N_CHANNELS, len(channel_stds))

    metrics = {
        "n_samples": 0,
        "per_channel": {},
        "continuous": {},
        "binary": {},
    }

    normalized_rmses = []
    normalized_mses = []
    normalized_maes = []
    binary_balanced_accs = []
    binary_roc_aucs = []
    unique_samples: set[int] = set()

    for ch in range(n_channels):
        ch_file = _channel_file(pairs_dir, ch)
        ch_metrics: dict = {"channel_idx": ch}

        if not ch_file.exists():
            ch_metrics["n_masked"] = 0
            ch_metrics["error"] = "no_masked_positions"
            metrics["per_channel"][f"ch_{ch}"] = ch_metrics
            continue

        table = pq.read_table(ch_file)

        if table.num_rows == 0:
            ch_metrics["n_masked"] = 0
            ch_metrics["error"] = "no_masked_positions"
            metrics["per_channel"][f"ch_{ch}"] = ch_metrics
            del table
            continue

        sample_idx_col = table.column("sample_idx")
        unique_samples.update(pc.unique(sample_idx_col).to_numpy())

        gt_col = table.column("gt")
        pred_col = table.column("pred")

        ch_metrics["n_masked"] = table.num_rows

        if ch in CONTINUOUS_CHANNEL_INDICES:
            gt_ch = gt_col.to_numpy().astype(np.float32)
            pred_ch = pred_col.to_numpy().astype(np.float32)
            del table, gt_col, pred_col, sample_idx_col

            errors = pred_ch - gt_ch
            mse = float(np.mean(errors**2))
            rmse = float(np.sqrt(mse))
            mae = float(np.mean(np.abs(errors)))

            ch_metrics["rmse"] = rmse
            ch_metrics["mse"] = mse
            ch_metrics["mae"] = mae

            ch_std = float(channel_stds[ch]) if channel_stds[ch] > 0 else 1.0
            normalized_rmse = rmse / ch_std
            normalized_mse = mse / (ch_std**2)
            normalized_mae = mae / ch_std
            ch_metrics["normalized_rmse"] = normalized_rmse
            ch_metrics["normalized_mse"] = normalized_mse
            ch_metrics["normalized_mae"] = normalized_mae
            ch_metrics["aggregation"] = "cell_micro"
            normalized_rmses.append(normalized_rmse)
            normalized_mses.append(normalized_mse)
            normalized_maes.append(normalized_mae)

        else:
            gt_bool = gt_col.to_numpy()
            pred_ch = pred_col.to_numpy().astype(np.float32)
            del table, gt_col, pred_col, sample_idx_col

            gt_binary = gt_bool.astype(np.int32)
            pred_binary = (pred_ch > 0.5).astype(np.int32)

            unique_gt = np.unique(gt_binary)
            if len(unique_gt) < 2:
                ch_metrics["balanced_accuracy"] = float("nan")
                ch_metrics["roc_auc"] = float("nan")
                ch_metrics["warning"] = "single_class"
            else:
                try:
                    balanced_acc = balanced_accuracy_score(gt_binary, pred_binary)
                    ch_metrics["balanced_accuracy"] = float(balanced_acc)
                    binary_balanced_accs.append(balanced_acc)
                except Exception as e:
                    ch_metrics["balanced_accuracy"] = float("nan")
                    ch_metrics["balanced_accuracy_error"] = str(e)

                try:
                    roc_auc = roc_auc_score(gt_binary, pred_ch)
                    ch_metrics["roc_auc"] = float(roc_auc)
                    binary_roc_aucs.append(roc_auc)
                except Exception as e:
                    ch_metrics["roc_auc"] = float("nan")
                    ch_metrics["roc_auc_error"] = str(e)
            ch_metrics["aggregation"] = "cell_micro"

        metrics["per_channel"][f"ch_{ch}"] = ch_metrics

    metrics["n_samples"] = len(unique_samples)

    if normalized_rmses:
        metrics["continuous"]["mean_normalized_rmse"] = float(np.mean(normalized_rmses))
        metrics["continuous"]["mean_normalized_mse"] = float(np.mean(normalized_mses))
        metrics["continuous"]["mean_normalized_mae"] = float(np.mean(normalized_maes))
        metrics["continuous"]["n_channels"] = len(normalized_rmses)
    else:
        metrics["continuous"]["mean_normalized_rmse"] = float("nan")
        metrics["continuous"]["mean_normalized_mse"] = float("nan")
        metrics["continuous"]["mean_normalized_mae"] = float("nan")
        metrics["continuous"]["n_channels"] = 0
    metrics["continuous"]["aggregation"] = "cell_micro"

    if binary_balanced_accs:
        metrics["binary"]["macro_balanced_accuracy"] = float(np.mean(binary_balanced_accs))
        metrics["binary"]["n_channels"] = len(binary_balanced_accs)
    else:
        metrics["binary"]["macro_balanced_accuracy"] = float("nan")
        metrics["binary"]["n_channels"] = 0
    metrics["binary"]["macro_roc_auc"] = (
        float(np.mean(binary_roc_aucs)) if binary_roc_aucs else float("nan")
    )
    metrics["binary"]["aggregation"] = "cell_micro"

    return metrics


def _aggregate_by_subgroup_cell_micro(
    pairs_path: Path,
    channel_stds: np.ndarray,
    subgroup_mapping: dict[int, dict[str, str]],
    attrs_list: list[str],
    n_channels: int,
) -> dict[str, dict[str, dict]]:
    """Cell-micro reducer with subgroup partitioning."""
    from collections import defaultdict

    group_data: dict[str, dict[str, dict[int, dict]]] = {
        attr: defaultdict(lambda: {}) for attr in attrs_list
    }
    group_unique_samples: dict[str, dict[str, set[int]]] = {
        attr: defaultdict(set) for attr in attrs_list
    }

    for ch in range(n_channels):
        ch_file = _channel_file(pairs_path, ch)
        if not ch_file.exists():
            continue
        table = pq.read_table(ch_file)
        if table.num_rows == 0:
            del table
            continue

        sample_idx_arr = table.column("sample_idx").to_numpy()
        gt_col = table.column("gt")
        pred_col = table.column("pred")
        is_binary = ch not in CONTINUOUS_CHANNEL_INDICES
        if is_binary:
            gt_arr = gt_col.to_numpy()
            pred_arr = pred_col.to_numpy().astype(np.float32)
        else:
            gt_arr = gt_col.to_numpy().astype(np.float32)
            pred_arr = pred_col.to_numpy().astype(np.float32)
        del table, gt_col, pred_col

        for attr in attrs_list:
            row_groups: dict[str, list[int]] = defaultdict(list)
            for i, sidx in enumerate(sample_idx_arr):
                demo = subgroup_mapping.get(int(sidx), {})
                group = demo.get(attr, "unknown")
                row_groups[group].append(i)

            for group_name, row_indices in row_groups.items():
                idx = np.array(row_indices)
                if ch not in group_data[attr][group_name]:
                    group_data[attr][group_name][ch] = {"gt": [], "pred": []}
                group_data[attr][group_name][ch]["gt"].append(gt_arr[idx])
                group_data[attr][group_name][ch]["pred"].append(pred_arr[idx])
                group_unique_samples[attr][group_name].update(sample_idx_arr[idx].tolist())

    result: dict[str, dict[str, dict]] = {}
    for attr in attrs_list:
        result[attr] = {}
        for group_name in sorted(group_data[attr].keys()):
            ch_data = group_data[attr][group_name]
            metrics: dict = {
                "n_samples": len(group_unique_samples[attr][group_name]),
                "per_channel": {},
                "continuous": {},
                "binary": {},
            }
            normalized_rmses = []
            normalized_mses = []
            normalized_maes = []
            binary_balanced_accs = []
            binary_roc_aucs = []

            for ch in range(n_channels):
                ch_metrics: dict = {"channel_idx": ch}
                if ch not in ch_data:
                    ch_metrics["n_masked"] = 0
                    ch_metrics["error"] = "no_masked_positions"
                    metrics["per_channel"][f"ch_{ch}"] = ch_metrics
                    continue

                gt_ch = np.concatenate(ch_data[ch]["gt"])
                pred_ch = np.concatenate(ch_data[ch]["pred"])
                ch_metrics["n_masked"] = len(gt_ch)

                if ch in CONTINUOUS_CHANNEL_INDICES:
                    errors = pred_ch - gt_ch
                    mse = float(np.mean(errors**2))
                    rmse = float(np.sqrt(mse))
                    mae = float(np.mean(np.abs(errors)))
                    ch_metrics["rmse"] = rmse
                    ch_metrics["mse"] = mse
                    ch_metrics["mae"] = mae
                    ch_std = float(channel_stds[ch]) if channel_stds[ch] > 0 else 1.0
                    normalized_rmse = rmse / ch_std
                    normalized_mse = mse / (ch_std**2)
                    normalized_mae = mae / ch_std
                    ch_metrics["normalized_rmse"] = normalized_rmse
                    ch_metrics["normalized_mse"] = normalized_mse
                    ch_metrics["normalized_mae"] = normalized_mae
                    ch_metrics["aggregation"] = "cell_micro"
                    normalized_rmses.append(normalized_rmse)
                    normalized_mses.append(normalized_mse)
                    normalized_maes.append(normalized_mae)
                else:
                    gt_binary = (
                        gt_ch.astype(np.int32)
                        if gt_ch.dtype == bool
                        else (gt_ch > 0.5).astype(np.int32)
                    )
                    pred_binary = (pred_ch > 0.5).astype(np.int32)
                    unique_gt = np.unique(gt_binary)
                    if len(unique_gt) < 2:
                        ch_metrics["balanced_accuracy"] = float("nan")
                        ch_metrics["roc_auc"] = float("nan")
                        ch_metrics["warning"] = "single_class"
                    else:
                        try:
                            balanced_acc = balanced_accuracy_score(gt_binary, pred_binary)
                            ch_metrics["balanced_accuracy"] = float(balanced_acc)
                            binary_balanced_accs.append(balanced_acc)
                        except Exception as e:
                            ch_metrics["balanced_accuracy"] = float("nan")
                            ch_metrics["balanced_accuracy_error"] = str(e)
                        try:
                            roc_auc = roc_auc_score(gt_binary, pred_ch)
                            ch_metrics["roc_auc"] = float(roc_auc)
                            binary_roc_aucs.append(roc_auc)
                        except Exception as e:
                            ch_metrics["roc_auc"] = float("nan")
                            ch_metrics["roc_auc_error"] = str(e)
                    ch_metrics["aggregation"] = "cell_micro"

                metrics["per_channel"][f"ch_{ch}"] = ch_metrics

            if normalized_rmses:
                metrics["continuous"]["mean_normalized_rmse"] = float(np.mean(normalized_rmses))
                metrics["continuous"]["mean_normalized_mse"] = float(np.mean(normalized_mses))
                metrics["continuous"]["mean_normalized_mae"] = float(np.mean(normalized_maes))
                metrics["continuous"]["n_channels"] = len(normalized_rmses)
            else:
                metrics["continuous"]["mean_normalized_rmse"] = float("nan")
                metrics["continuous"]["mean_normalized_mse"] = float("nan")
                metrics["continuous"]["mean_normalized_mae"] = float("nan")
                metrics["continuous"]["n_channels"] = 0
            metrics["continuous"]["aggregation"] = "cell_micro"

            if binary_balanced_accs:
                metrics["binary"]["macro_balanced_accuracy"] = float(np.mean(binary_balanced_accs))
                metrics["binary"]["n_channels"] = len(binary_balanced_accs)
            else:
                metrics["binary"]["macro_balanced_accuracy"] = float("nan")
                metrics["binary"]["n_channels"] = 0
            metrics["binary"]["macro_roc_auc"] = (
                float(np.mean(binary_roc_aucs)) if binary_roc_aucs else float("nan")
            )
            metrics["binary"]["aggregation"] = "cell_micro"

            result[attr][group_name] = metrics

    return result


def _aggregate_legacy(
    pairs_file: Path,
    channel_stds: np.ndarray,
) -> dict:
    """Backward-compatible aggregation from old single-file pairs.parquet.

    Always cell-micro — the old format predates the per-channel manifest
    convention.
    """
    table = pq.read_table(pairs_file)
    if table.num_rows == 0:
        return {"error": "no_pairs", "n_samples": 0}

    sample_idx = table.column("sample_idx").to_numpy()
    channel = table.column("channel").to_numpy()
    gt = table.column("gt").to_numpy()
    pred = table.column("pred").to_numpy()

    n_applicable = len(np.unique(sample_idx))
    n_channels = min(N_CHANNELS, len(channel_stds))

    metrics: dict = {
        "n_samples": n_applicable,
        "per_channel": {},
        "continuous": {},
        "binary": {},
    }

    normalized_rmses = []
    normalized_mses = []
    normalized_maes = []
    binary_balanced_accs = []
    binary_roc_aucs = []

    for ch in range(n_channels):
        ch_metrics: dict = {"channel_idx": ch}
        ch_mask = channel == ch
        gt_ch = gt[ch_mask]
        pred_ch = pred[ch_mask]

        ch_metrics["n_masked"] = len(gt_ch)

        if len(gt_ch) == 0:
            ch_metrics["error"] = "no_masked_positions"
            metrics["per_channel"][f"ch_{ch}"] = ch_metrics
            continue

        if ch in CONTINUOUS_CHANNEL_INDICES:
            errors = pred_ch - gt_ch
            mse = float(np.mean(errors**2))
            rmse = float(np.sqrt(mse))
            mae = float(np.mean(np.abs(errors)))

            ch_metrics["rmse"] = rmse
            ch_metrics["mse"] = mse
            ch_metrics["mae"] = mae

            ch_std = float(channel_stds[ch]) if channel_stds[ch] > 0 else 1.0
            normalized_rmse = rmse / ch_std
            normalized_mse = mse / (ch_std**2)
            normalized_mae = mae / ch_std
            ch_metrics["normalized_rmse"] = normalized_rmse
            ch_metrics["normalized_mse"] = normalized_mse
            ch_metrics["normalized_mae"] = normalized_mae
            ch_metrics["aggregation"] = "cell_micro"
            normalized_rmses.append(normalized_rmse)
            normalized_mses.append(normalized_mse)
            normalized_maes.append(normalized_mae)

        else:
            gt_binary = (gt_ch > 0.5).astype(int)
            pred_binary = (pred_ch > 0.5).astype(int)

            unique_gt = np.unique(gt_binary)
            if len(unique_gt) < 2:
                ch_metrics["balanced_accuracy"] = float("nan")
                ch_metrics["roc_auc"] = float("nan")
                ch_metrics["warning"] = "single_class"
            else:
                try:
                    balanced_acc = balanced_accuracy_score(gt_binary, pred_binary)
                    ch_metrics["balanced_accuracy"] = float(balanced_acc)
                    binary_balanced_accs.append(balanced_acc)
                except Exception as e:
                    ch_metrics["balanced_accuracy"] = float("nan")
                    ch_metrics["balanced_accuracy_error"] = str(e)

                try:
                    roc_auc = roc_auc_score(gt_binary, pred_ch)
                    ch_metrics["roc_auc"] = float(roc_auc)
                    binary_roc_aucs.append(roc_auc)
                except Exception as e:
                    ch_metrics["roc_auc"] = float("nan")
                    ch_metrics["roc_auc_error"] = str(e)
            ch_metrics["aggregation"] = "cell_micro"

        metrics["per_channel"][f"ch_{ch}"] = ch_metrics

    if normalized_rmses:
        metrics["continuous"]["mean_normalized_rmse"] = float(np.mean(normalized_rmses))
        metrics["continuous"]["mean_normalized_mse"] = float(np.mean(normalized_mses))
        metrics["continuous"]["mean_normalized_mae"] = float(np.mean(normalized_maes))
        metrics["continuous"]["n_channels"] = len(normalized_rmses)
    else:
        metrics["continuous"]["mean_normalized_rmse"] = float("nan")
        metrics["continuous"]["mean_normalized_mse"] = float("nan")
        metrics["continuous"]["mean_normalized_mae"] = float("nan")
        metrics["continuous"]["n_channels"] = 0
    metrics["continuous"]["aggregation"] = "cell_micro"

    if binary_balanced_accs:
        metrics["binary"]["macro_balanced_accuracy"] = float(np.mean(binary_balanced_accs))
        metrics["binary"]["n_channels"] = len(binary_balanced_accs)
    else:
        metrics["binary"]["macro_balanced_accuracy"] = float("nan")
        metrics["binary"]["n_channels"] = 0
    metrics["binary"]["macro_roc_auc"] = (
        float(np.mean(binary_roc_aucs)) if binary_roc_aucs else float("nan")
    )
    metrics["binary"]["aggregation"] = "cell_micro"

    return metrics
