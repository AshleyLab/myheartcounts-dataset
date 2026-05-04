"""Metrics for imputation evaluation.

Computes per-channel and aggregate metrics for continuous and binary channels.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from data.processing.hf_config import CONTINUOUS_CHANNEL_INDICES

logger = logging.getLogger(__name__)


def compute_scenario_metrics(
    ground_truths: np.ndarray,
    imputations: np.ndarray,
    artificial_masks: np.ndarray,
    channel_stds: np.ndarray,
    include_ks: bool = True,
    include_wasserstein: bool = True,
) -> dict:
    """Compute metrics for a single masking scenario.

    Collects all artificially-masked positions per channel and computes:
    - Continuous channels (0-6): RMSE, MAE, KS statistic per channel + Mean Normalized RMSE, Mean KS
    - Binary channels (7-18): Balanced Accuracy, ROC AUC per channel + macro-average

    Args:
        ground_truths: Ground truth values of shape (N, C, T).
        imputations: Imputed values of shape (N, C, T).
        artificial_masks: Binary masks of shape (N, C, T), 1=was masked.
        channel_stds: Per-channel stds from training, shape (C,).
        include_ks: Whether to compute KS statistic.
        include_wasserstein: Whether to compute Wasserstein distance.

    Returns:
        Dictionary with per-channel and aggregate metrics.
    """
    n_samples, n_channels, n_timesteps = ground_truths.shape

    metrics = {
        "n_samples": n_samples,
        "per_channel": {},
        "continuous": {},
        "binary": {},
    }

    # Track for aggregate metrics
    normalized_rmses = []
    normalized_mses = []
    normalized_maes = []
    ks_statistics = []
    wasserstein_distances = []
    binary_balanced_accs = []
    binary_roc_aucs = []

    for ch in range(n_channels):
        ch_metrics = {"channel_idx": ch}

        # Get masked positions for this channel
        mask = artificial_masks[:, ch, :] == 1
        gt_values = ground_truths[:, ch, :][mask]
        pred_values = imputations[:, ch, :][mask]

        # Filter out non-finite values
        finite_mask = np.isfinite(gt_values) & np.isfinite(pred_values)
        gt_values = gt_values[finite_mask]
        pred_values = pred_values[finite_mask]

        ch_metrics["n_masked"] = len(gt_values)

        if len(gt_values) == 0:
            ch_metrics["error"] = "no_masked_positions"
            metrics["per_channel"][f"ch_{ch}"] = ch_metrics
            continue

        if ch in CONTINUOUS_CHANNEL_INDICES:
            # Continuous channel: RMSE, MAE, KS statistic
            errors = pred_values - gt_values
            rmse = np.sqrt(np.mean(errors**2))
            mae = np.mean(np.abs(errors))

            mse = np.mean(errors**2)
            ch_metrics["rmse"] = float(rmse)
            ch_metrics["mse"] = float(mse)
            ch_metrics["mae"] = float(mae)

            # Normalized metrics (by training std)
            ch_std = channel_stds[ch] if channel_stds[ch] > 0 else 1.0
            normalized_rmse = rmse / ch_std
            normalized_mse = mse / (ch_std**2)
            normalized_mae = mae / ch_std
            ch_metrics["normalized_rmse"] = float(normalized_rmse)
            ch_metrics["normalized_mse"] = float(normalized_mse)
            ch_metrics["normalized_mae"] = float(normalized_mae)
            normalized_rmses.append(normalized_rmse)
            normalized_mses.append(normalized_mse)
            normalized_maes.append(normalized_mae)

            # KS statistic: compare distributions of gt vs imputed at masked positions
            if include_ks:
                if len(gt_values) >= 2:
                    ks_result = ks_2samp(gt_values, pred_values, nan_policy="omit")
                    ch_metrics["ks_statistic"] = float(ks_result.statistic)
                    ks_statistics.append(ks_result.statistic)
                else:
                    ch_metrics["ks_statistic"] = float("nan")

            # Wasserstein distance: 1-Wasserstein between gt and imputed distributions
            if include_wasserstein:
                if len(gt_values) >= 1:
                    wd = wasserstein_distance(gt_values, pred_values)
                    ch_metrics["wasserstein_distance"] = float(wd)
                    wasserstein_distances.append(wd)
                else:
                    ch_metrics["wasserstein_distance"] = float("nan")

        else:
            # Binary channel: Balanced Accuracy, ROC AUC
            # Round predictions to 0/1 for classification metrics
            gt_binary = (gt_values > 0.5).astype(int)
            pred_binary = (pred_values > 0.5).astype(int)

            # Check for single-class case
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
                    # Use continuous predictions for ROC AUC
                    roc_auc = roc_auc_score(gt_binary, pred_values)
                    ch_metrics["roc_auc"] = float(roc_auc)
                    binary_roc_aucs.append(roc_auc)
                except Exception as e:
                    ch_metrics["roc_auc"] = float("nan")
                    ch_metrics["roc_auc_error"] = str(e)

        metrics["per_channel"][f"ch_{ch}"] = ch_metrics

    # Aggregate metrics for continuous channels
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

    if include_ks:
        if ks_statistics:
            metrics["continuous"]["mean_ks_statistic"] = float(np.mean(ks_statistics))
        else:
            metrics["continuous"]["mean_ks_statistic"] = float("nan")

    if include_wasserstein:
        if wasserstein_distances:
            metrics["continuous"]["mean_wasserstein_distance"] = float(
                np.mean(wasserstein_distances)
            )
        else:
            metrics["continuous"]["mean_wasserstein_distance"] = float("nan")

    # Aggregate metrics for binary channels
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


def compute_per_sample_metrics(
    ground_truth: np.ndarray,
    imputed: np.ndarray,
    masks: np.ndarray,
    include_ks: bool = True,
    include_wasserstein: bool = True,
) -> dict[str, list[float]]:
    """Compute metrics for each sample in the batch for a single channel.

    Args:
        ground_truth: Ground truth values of shape (B, T).
        imputed: Imputed values of shape (B, T).
        masks: Binary masks of shape (B, T), 1=masked.
        include_ks: Whether to compute KS statistic.
        include_wasserstein: Whether to compute Wasserstein distance.

    Returns:
        Dictionary with lists of metric values (one per applicable sample).
    """
    results = {
        "rmse": [],
        "mse": [],
        "mae": [],
        "ks_statistic": [],
        "wasserstein_distance": [],
        "n_masked": [],
    }

    batch_size = len(ground_truth)
    for i in range(batch_size):
        # Extract masked values for this sample
        mask_i = masks[i] == 1
        gt_i = ground_truth[i][mask_i]
        pred_i = imputed[i][mask_i]

        # Filter finite
        valid = np.isfinite(gt_i) & np.isfinite(pred_i)
        gt_i = gt_i[valid]
        pred_i = pred_i[valid]

        n = len(gt_i)
        results["n_masked"].append(float(n))

        if n == 0:
            results["rmse"].append(float("nan"))
            results["mse"].append(float("nan"))
            results["mae"].append(float("nan"))
            results["ks_statistic"].append(float("nan"))
            results["wasserstein_distance"].append(float("nan"))
            continue

        # RMSE/MSE/MAE
        errors = pred_i - gt_i
        mse = np.mean(errors**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(errors))
        results["rmse"].append(float(rmse))
        results["mse"].append(float(mse))
        results["mae"].append(float(mae))

        # KS
        if include_ks and n >= 2:
            try:
                ks = ks_2samp(gt_i, pred_i, nan_policy="omit").statistic
                results["ks_statistic"].append(float(ks))
            except Exception:
                results["ks_statistic"].append(float("nan"))
        else:
            results["ks_statistic"].append(float("nan"))

        # Wasserstein
        if include_wasserstein and n >= 1:
            try:
                wd = wasserstein_distance(gt_i, pred_i)
                results["wasserstein_distance"].append(float(wd))
            except Exception:
                results["wasserstein_distance"].append(float("nan"))
        else:
            results["wasserstein_distance"].append(float("nan"))

    return results
