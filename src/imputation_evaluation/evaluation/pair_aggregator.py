"""Aggregate metrics from saved per-channel (gt, pred) pair Parquet files.

Reads per-channel Parquet files written by PairWriter, and computes the same
metrics as MetricAccumulator.compute(). Each channel file is loaded and freed
independently, keeping peak memory to ~1/19th of the old single-file approach.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from data.processing.hf_config import CONTINUOUS_CHANNEL_INDICES, N_CHANNELS

logger = logging.getLogger(__name__)


def _channel_file(pairs_dir: Path, ch: int) -> Path:
    return pairs_dir / f"pairs_ch{ch:02d}.parquet"


def aggregate_pairs(
    pairs_path: str | Path,
    channel_stds: np.ndarray,
    include_ks: bool = True,
    include_wasserstein: bool = True,
) -> dict:
    """Compute metrics from saved per-channel pair Parquet files.

    Reads per-channel files one at a time, computing metrics and freeing memory
    before moving to the next channel.

    Args:
        pairs_path: Path to the pairs directory (containing ``pairs_ch*.parquet``).
            Also accepts a file path for backward compatibility with old
            ``pairs.parquet`` format (logs a warning).
        channel_stds: Per-channel standard deviations for normalization, shape (C,).
        include_ks: Whether to compute KS statistic for continuous channels.
        include_wasserstein: Whether to compute Wasserstein distance.

    Returns:
        Metrics dict matching the format of ``MetricAccumulator.compute()``.
    """
    pairs_path = Path(pairs_path)

    # Resolve to directory
    if pairs_path.is_file():
        pairs_dir = pairs_path.parent
    else:
        pairs_dir = pairs_path

    # Check for per-channel files
    n_channels = min(N_CHANNELS, len(channel_stds))
    any_channel_file = any(_channel_file(pairs_dir, ch).exists() for ch in range(n_channels))

    if not any_channel_file:
        # Backward compat: check for old single-file format
        old_file = pairs_dir / "pairs.parquet"
        if old_file.exists():
            logger.warning(
                "Found old-format pairs.parquet. Re-run imputation eval to generate "
                "per-channel files for lower memory usage."
            )
            return _aggregate_legacy(old_file, channel_stds, include_ks, include_wasserstein)
        return {"error": "pairs_file_not_found", "n_samples": 0}

    metrics = {
        "n_samples": 0,
        "per_channel": {},
        "continuous": {},
        "binary": {},
    }

    normalized_rmses = []
    normalized_mses = []
    normalized_maes = []
    ks_statistics = []
    wasserstein_distances = []
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
            # float16 → float32 for computation
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
            normalized_rmses.append(normalized_rmse)
            normalized_mses.append(normalized_mse)
            normalized_maes.append(normalized_mae)

            if include_ks and len(gt_ch) >= 2:
                try:
                    ks_result = ks_2samp(gt_ch, pred_ch, nan_policy="omit")
                    ch_metrics["ks_statistic"] = float(ks_result.statistic)
                    ks_statistics.append(ks_result.statistic)
                except Exception:
                    ch_metrics["ks_statistic"] = float("nan")
            elif include_ks:
                ch_metrics["ks_statistic"] = float("nan")

            if include_wasserstein and len(gt_ch) >= 1:
                try:
                    wd = wasserstein_distance(gt_ch, pred_ch)
                    ch_metrics["wasserstein_distance"] = float(wd)
                    wasserstein_distances.append(wd)
                except Exception:
                    ch_metrics["wasserstein_distance"] = float("nan")
            elif include_wasserstein:
                ch_metrics["wasserstein_distance"] = float("nan")

        else:
            # Binary channel — gt is bool in Parquet
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

        metrics["per_channel"][f"ch_{ch}"] = ch_metrics

    metrics["n_samples"] = len(unique_samples)

    # Aggregate continuous
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

    if ks_statistics:
        metrics["continuous"]["mean_ks_statistic"] = float(np.mean(ks_statistics))
    else:
        metrics["continuous"]["mean_ks_statistic"] = float("nan")

    if wasserstein_distances:
        metrics["continuous"]["mean_wasserstein_distance"] = float(np.mean(wasserstein_distances))
    else:
        metrics["continuous"]["mean_wasserstein_distance"] = float("nan")

    # Aggregate binary
    if binary_balanced_accs:
        metrics["binary"]["macro_balanced_accuracy"] = float(np.mean(binary_balanced_accs))
        metrics["binary"]["n_channels"] = len(binary_balanced_accs)
    else:
        metrics["binary"]["macro_balanced_accuracy"] = float("nan")
        metrics["binary"]["n_channels"] = 0

    if binary_roc_aucs:
        metrics["binary"]["macro_roc_auc"] = float(np.mean(binary_roc_aucs))
    else:
        metrics["binary"]["macro_roc_auc"] = float("nan")

    return metrics


def aggregate_pairs_by_subgroup(
    pairs_path: str | Path,
    channel_stds: np.ndarray,
    subgroup_mapping: dict[int, dict[str, str]],
    include_ks: bool = False,
    include_wasserstein: bool = False,
) -> dict[str, dict[str, dict]]:
    """Compute per-subgroup metrics from saved per-channel pair Parquet files.

    Reads each per-channel file once, partitions rows by subgroup via
    ``sample_idx`` lookup, and computes metrics per group — following
    the same channel-by-channel memory pattern as :func:`aggregate_pairs`.

    Args:
        pairs_path: Directory containing ``pairs_ch*.parquet`` files.
        channel_stds: Per-channel standard deviations for normalization, shape ``(C,)``.
        subgroup_mapping: Maps ``sample_idx`` to attribute dicts,
            e.g. ``{0: {"age_group": "30-39", "sex": "male"}, ...}``.
        include_ks: Whether to compute KS statistic for continuous channels.
        include_wasserstein: Whether to compute Wasserstein distance.

    Returns:
        ``{attr: {group_name: metrics_dict}}`` where ``metrics_dict``
        matches the format of :func:`aggregate_pairs`.
    """
    pairs_path = Path(pairs_path)
    if pairs_path.is_file():
        pairs_path = pairs_path.parent

    n_channels = min(N_CHANNELS, len(channel_stds))

    # Discover attributes and groups from the mapping
    attrs: set[str] = set()
    for demo in subgroup_mapping.values():
        attrs.update(demo.keys())
    attrs_list = sorted(attrs)

    if not attrs_list:
        return {}

    # Per-attribute, per-group accumulators keyed by (attr, group) -> per-channel data
    # We accumulate raw arrays per group per channel, then compute metrics at the end.
    # Structure: {attr: {group: {ch: {"gt": [], "pred": [], "sample_idxs": set}}}}
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

        # Partition rows by subgroup
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

    # Compute metrics per group
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
            ks_statistics = []
            wasserstein_distances = []
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
                    normalized_rmses.append(normalized_rmse)
                    normalized_mses.append(normalized_mse)
                    normalized_maes.append(normalized_mae)

                    if include_ks and len(gt_ch) >= 2:
                        try:
                            ks_result = ks_2samp(gt_ch, pred_ch, nan_policy="omit")
                            ch_metrics["ks_statistic"] = float(ks_result.statistic)
                            ks_statistics.append(ks_result.statistic)
                        except Exception:
                            ch_metrics["ks_statistic"] = float("nan")
                    elif include_ks:
                        ch_metrics["ks_statistic"] = float("nan")

                    if include_wasserstein and len(gt_ch) >= 1:
                        try:
                            wd = wasserstein_distance(gt_ch, pred_ch)
                            ch_metrics["wasserstein_distance"] = float(wd)
                            wasserstein_distances.append(wd)
                        except Exception:
                            ch_metrics["wasserstein_distance"] = float("nan")
                    elif include_wasserstein:
                        ch_metrics["wasserstein_distance"] = float("nan")

                else:
                    gt_binary = gt_ch.astype(np.int32) if gt_ch.dtype == bool else (gt_ch > 0.5).astype(np.int32)
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

                metrics["per_channel"][f"ch_{ch}"] = ch_metrics

            # Aggregate continuous
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

            if ks_statistics:
                metrics["continuous"]["mean_ks_statistic"] = float(np.mean(ks_statistics))
            else:
                metrics["continuous"]["mean_ks_statistic"] = float("nan")

            if wasserstein_distances:
                metrics["continuous"]["mean_wasserstein_distance"] = float(
                    np.mean(wasserstein_distances)
                )
            else:
                metrics["continuous"]["mean_wasserstein_distance"] = float("nan")

            # Aggregate binary
            if binary_balanced_accs:
                metrics["binary"]["macro_balanced_accuracy"] = float(
                    np.mean(binary_balanced_accs)
                )
                metrics["binary"]["n_channels"] = len(binary_balanced_accs)
            else:
                metrics["binary"]["macro_balanced_accuracy"] = float("nan")
                metrics["binary"]["n_channels"] = 0

            if binary_roc_aucs:
                metrics["binary"]["macro_roc_auc"] = float(np.mean(binary_roc_aucs))
            else:
                metrics["binary"]["macro_roc_auc"] = float("nan")

            result[attr][group_name] = metrics

    return result


def _aggregate_legacy(
    pairs_file: Path,
    channel_stds: np.ndarray,
    include_ks: bool,
    include_wasserstein: bool,
) -> dict:
    """Backward-compatible aggregation from old single-file pairs.parquet."""
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
    ks_statistics = []
    wasserstein_dists = []
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
            normalized_rmses.append(normalized_rmse)
            normalized_mses.append(normalized_mse)
            normalized_maes.append(normalized_mae)

            if include_ks and len(gt_ch) >= 2:
                try:
                    ks_result = ks_2samp(gt_ch, pred_ch, nan_policy="omit")
                    ch_metrics["ks_statistic"] = float(ks_result.statistic)
                    ks_statistics.append(ks_result.statistic)
                except Exception:
                    ch_metrics["ks_statistic"] = float("nan")
            elif include_ks:
                ch_metrics["ks_statistic"] = float("nan")

            if include_wasserstein and len(gt_ch) >= 1:
                try:
                    wd = wasserstein_distance(gt_ch, pred_ch)
                    ch_metrics["wasserstein_distance"] = float(wd)
                    wasserstein_dists.append(wd)
                except Exception:
                    ch_metrics["wasserstein_distance"] = float("nan")
            elif include_wasserstein:
                ch_metrics["wasserstein_distance"] = float("nan")

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

    if ks_statistics:
        metrics["continuous"]["mean_ks_statistic"] = float(np.mean(ks_statistics))
    else:
        metrics["continuous"]["mean_ks_statistic"] = float("nan")

    if wasserstein_dists:
        metrics["continuous"]["mean_wasserstein_distance"] = float(np.mean(wasserstein_dists))
    else:
        metrics["continuous"]["mean_wasserstein_distance"] = float("nan")

    if binary_balanced_accs:
        metrics["binary"]["macro_balanced_accuracy"] = float(np.mean(binary_balanced_accs))
        metrics["binary"]["n_channels"] = len(binary_balanced_accs)
    else:
        metrics["binary"]["macro_balanced_accuracy"] = float("nan")
        metrics["binary"]["n_channels"] = 0

    if binary_roc_aucs:
        metrics["binary"]["macro_roc_auc"] = float(np.mean(binary_roc_aucs))
    else:
        metrics["binary"]["macro_roc_auc"] = float("nan")

    return metrics
