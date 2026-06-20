"""Main evaluator for imputation evaluation.

Orchestrates data loading, imputation, and metric computation using pre-generated masks.
Uses batch-by-batch processing for memory efficiency.

Parallelism strategy:
- When num_eval_workers > 1, uses ProcessPoolExecutor for batch-level parallelism
- Each worker process handles ALL scenarios for its assigned batch
- This avoids GIL contention (vs ThreadPoolExecutor) since each process has its own interpreter
"""

from __future__ import annotations

import gc
import logging
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

from data.processing.hf_config import CONTINUOUS_CHANNEL_INDICES
from imputation_evaluation.evaluation.metrics import compute_per_sample_metrics

if TYPE_CHECKING:
    from imputation_evaluation.evaluation.pair_writer import PairWriter
    from imputation_evaluation.masking.generator import MaskCache
    from imputation_evaluation.methods.base import ImputationMethod

logger = logging.getLogger(__name__)

# Module-level globals for worker processes (set via pool initializer)
_worker_mask_cache: MaskCache | None = None
_worker_method: ImputationMethod | None = None
_worker_scenarios: list[str] | None = None
_worker_channel_stds: np.ndarray | None = None
_worker_fallback_fill: np.ndarray | None = None
_worker_split_name: str | None = None
_worker_subgroup_mapping: dict[int, dict[str, str]] | None = None
_worker_n_days: int = 1
_worker_window_descriptors: list[list[int]] | None = None
_worker_window_day_offsets: list[list[int]] | None = None
_worker_compute_metrics: bool = True
_worker_save_pairs: bool = True

# Number of channels
N_CHANNELS = 19


def _apply_fallback(
    imputed: np.ndarray,
    artificial_masks: np.ndarray,
    fill: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Substitute non-finite cells at target positions with a per-channel fill.

    Counts every target cell (``artificial_masks == 1``) the imputer left
    non-finite, and replaces those cells in-place with the corresponding
    channel's value from ``fill``. Mirrors the forecasting harness's
    fallback substitution — the model's inability is now visible
    instead of being silently dropped at downstream ``isfinite`` filters.

    Args:
        imputed: Imputed values, shape ``(N, C, T)``. Modified in place.
        artificial_masks: Binary masks of shape ``(N, C, T)``; ``1`` = was
            artificially masked (target cell).
        fill: Per-channel fill values, shape ``(C,)``. If ``None`` the
            function is a no-op and returns zero counts (legacy passthrough).

    Returns:
        A pair ``(sub, asked)`` of per-channel ``int64`` arrays of length ``C``:
            ``sub[c]`` is the number of target cells substituted in channel ``c``;
            ``asked[c]`` is the total number of target cells in channel ``c``.
    """
    if fill is None:
        return (
            np.zeros(N_CHANNELS, dtype=np.int64),
            np.zeros(N_CHANNELS, dtype=np.int64),
        )
    target = artificial_masks == 1
    nan_at_target = target & ~np.isfinite(imputed)
    # Per-channel substitution: broadcast fill[ch] across (N, T) where needed.
    for ch in range(N_CHANNELS):
        ch_nan = nan_at_target[:, ch, :]
        if ch_nan.any():
            imputed[:, ch, :][ch_nan] = fill[ch]
    sub = nan_at_target.sum(axis=(0, 2)).astype(np.int64)
    asked = target.sum(axis=(0, 2)).astype(np.int64)
    return sub, asked


class MetricAccumulator:
    """Accumulator for incremental metric computation.

    Computes metrics incrementally to avoid storing full arrays in memory.
    - Continuous channels: running sums for global RMSE/MAE; lists for per-sample RMSE/MAE.
    - Binary channels: stores only masked (gt, pred) pairs per channel (for now).
    """

    def __init__(
        self,
        channel_stds: np.ndarray,
    ):
        """Initialize empty accumulator.

        Args:
            channel_stds: Per-channel standard deviations for normalization.
        """
        self.channel_stds = channel_stds
        self.n_applicable: int = 0
        self.n_total: int = 0

        # Fallback substitution counters (per-channel).
        # ``fallback_substituted[c]`` = number of target cells in channel ``c`` that
        # were filled because the imputer returned non-finite there.
        # ``fallback_asked[c]``       = total number of target cells in channel ``c``.
        # The ratio is the model-capability gap, mirroring Track 3 forecasting.
        self.fallback_substituted = np.zeros(N_CHANNELS, dtype=np.int64)
        self.fallback_asked = np.zeros(N_CHANNELS, dtype=np.int64)

        # Continuous channel accumulators (channels 0-6)
        # Global stats (running sums)
        self._cont_sum_sq_errors = np.zeros(N_CHANNELS)
        self._cont_sum_abs_errors = np.zeros(N_CHANNELS)
        self._cont_counts = np.zeros(N_CHANNELS, dtype=np.int64)

        # Per-sample stats (lists of floats)
        self.per_sample_metrics = {
            "rmse": {ch: [] for ch in CONTINUOUS_CHANNEL_INDICES},
            "mse": {ch: [] for ch in CONTINUOUS_CHANNEL_INDICES},
            "mae": {ch: [] for ch in CONTINUOUS_CHANNEL_INDICES},
        }

        # Binary channel accumulators (channels 7-18)
        # Per-channel lists of (gt, pred) for masked positions only
        self._binary_gt: dict[int, list[np.ndarray]] = {ch: [] for ch in range(7, N_CHANNELS)}
        self._binary_pred: dict[int, list[np.ndarray]] = {ch: [] for ch in range(7, N_CHANNELS)}

    def update(
        self,
        ground_truth: np.ndarray,
        imputed: np.ndarray,
        artificial_masks: np.ndarray,
    ) -> None:
        """Add a batch of applicable samples to the accumulator.

        Args:
            ground_truth: Ground truth values of shape (N, C, T).
            imputed: Imputed values of shape (N, C, T).
            artificial_masks: Binary masks of shape (N, C, T), 1=was masked.
        """
        if len(ground_truth) == 0:
            return

        self.n_applicable += len(ground_truth)

        for ch in range(N_CHANNELS):
            # Get masked positions for this channel
            mask = artificial_masks[:, ch, :] == 1
            gt_values = ground_truth[:, ch, :][mask]
            pred_values = imputed[:, ch, :][mask]

            # Filter out non-finite values
            finite_mask = np.isfinite(gt_values) & np.isfinite(pred_values)
            gt_values = gt_values[finite_mask]
            pred_values = pred_values[finite_mask]

            if len(gt_values) == 0:
                continue

            if ch in CONTINUOUS_CHANNEL_INDICES:
                # Continuous channel: accumulate global error statistics
                errors = pred_values - gt_values
                self._cont_sum_sq_errors[ch] += np.sum(errors**2)
                self._cont_sum_abs_errors[ch] += np.sum(np.abs(errors))
                self._cont_counts[ch] += len(errors)

                # Continuous channel: compute and store per-sample metrics
                batch_metrics = compute_per_sample_metrics(
                    ground_truth[:, ch, :],
                    imputed[:, ch, :],
                    artificial_masks[:, ch, :],
                )
                self.per_sample_metrics["rmse"][ch].extend(batch_metrics["rmse"])
                self.per_sample_metrics["mse"][ch].extend(batch_metrics["mse"])
                self.per_sample_metrics["mae"][ch].extend(batch_metrics["mae"])

            else:
                # Binary channel: store masked values for later
                # Use int8 for ground truth (0/1) and float16 for predictions to save memory
                self._binary_gt[ch].append(gt_values.astype(np.int8))
                self._binary_pred[ch].append(pred_values.astype(np.float16))

    def increment_total(self, count: int) -> None:
        """Increment the total sample count (including non-applicable)."""
        self.n_total += count

    def add_fallback(self, substituted: np.ndarray, asked: np.ndarray) -> None:
        """Record per-channel fallback substitution counts for a batch.

        Args:
            substituted: Per-channel ``(C,)`` int counts of target cells the
                imputer left non-finite (and the harness filled in).
            asked: Per-channel ``(C,)`` int counts of total target cells.
        """
        self.fallback_substituted += np.asarray(substituted, dtype=np.int64)
        self.fallback_asked += np.asarray(asked, dtype=np.int64)

    def merge(self, other: MetricAccumulator) -> None:
        """Merge another accumulator into this one.

        Args:
            other: Another MetricAccumulator to merge into this one.
        """
        self.n_applicable += other.n_applicable
        self.n_total += other.n_total
        self.fallback_substituted += other.fallback_substituted
        self.fallback_asked += other.fallback_asked
        self._cont_sum_sq_errors += other._cont_sum_sq_errors
        self._cont_sum_abs_errors += other._cont_sum_abs_errors
        self._cont_counts += other._cont_counts

        # Merge per-sample metrics
        for metric in ["rmse", "mse", "mae"]:
            for ch in CONTINUOUS_CHANNEL_INDICES:
                self.per_sample_metrics[metric][ch].extend(other.per_sample_metrics[metric][ch])

        for ch in range(7, N_CHANNELS):
            self._binary_gt[ch].extend(other._binary_gt[ch])
            self._binary_pred[ch].extend(other._binary_pred[ch])

    def compute(self) -> dict:
        """Compute final metrics from accumulated statistics.

        Returns:
            Metrics dictionary matching the format of compute_scenario_metrics.
        """
        if self.n_applicable == 0:
            return {"error": "no_applicable_samples", "n_samples": 0}

        metrics = {
            "n_samples": self.n_applicable,
            "per_channel": {},
            "continuous": {},
            "binary": {},
        }

        normalized_rmses = []
        normalized_mses = []
        normalized_maes = []
        binary_balanced_accs = []
        binary_roc_aucs = []

        for ch in range(N_CHANNELS):
            ch_metrics = {"channel_idx": ch}

            if ch in CONTINUOUS_CHANNEL_INDICES:
                # Continuous channel metrics from running statistics
                count = self._cont_counts[ch]
                ch_metrics["n_masked"] = int(count)

                if count == 0:
                    ch_metrics["error"] = "no_masked_positions"
                else:
                    # Global RMSE/MSE/MAE
                    mse = self._cont_sum_sq_errors[ch] / count
                    rmse = np.sqrt(mse)
                    mae = self._cont_sum_abs_errors[ch] / count

                    ch_metrics["rmse"] = float(rmse)
                    ch_metrics["mse"] = float(mse)
                    ch_metrics["mae"] = float(mae)

                    # Normalized metrics
                    ch_std = self.channel_stds[ch] if self.channel_stds[ch] > 0 else 1.0
                    normalized_rmse = rmse / ch_std
                    normalized_mse = mse / (ch_std**2)
                    normalized_mae = mae / ch_std
                    ch_metrics["normalized_rmse"] = float(normalized_rmse)
                    ch_metrics["normalized_mse"] = float(normalized_mse)
                    ch_metrics["normalized_mae"] = float(normalized_mae)
                    normalized_rmses.append(normalized_rmse)
                    normalized_mses.append(normalized_mse)
                    normalized_maes.append(normalized_mae)

                    # Per-sample metrics aggregation
                    ps_rmse = np.array(self.per_sample_metrics["rmse"][ch])
                    ps_mse = np.array(self.per_sample_metrics["mse"][ch])
                    ps_mae = np.array(self.per_sample_metrics["mae"][ch])

                    # Filter NaNs
                    ps_rmse = ps_rmse[np.isfinite(ps_rmse)]
                    ps_mse = ps_mse[np.isfinite(ps_mse)]
                    ps_mae = ps_mae[np.isfinite(ps_mae)]

                    if len(ps_rmse) > 0:
                        ch_metrics["per_sample_rmse_mean"] = float(np.mean(ps_rmse))
                        ch_metrics["per_sample_rmse_std"] = float(np.std(ps_rmse))
                    else:
                        ch_metrics["per_sample_rmse_mean"] = float("nan")

                    if len(ps_mse) > 0:
                        ch_metrics["per_sample_mse_mean"] = float(np.mean(ps_mse))
                        ch_metrics["per_sample_mse_std"] = float(np.std(ps_mse))
                    else:
                        ch_metrics["per_sample_mse_mean"] = float("nan")

                    if len(ps_mae) > 0:
                        ch_metrics["per_sample_mae_mean"] = float(np.mean(ps_mae))
                        ch_metrics["per_sample_mae_std"] = float(np.std(ps_mae))
                    else:
                        ch_metrics["per_sample_mae_mean"] = float("nan")

            else:
                # Binary channel metrics from stored values
                gt_list = self._binary_gt[ch]
                pred_list = self._binary_pred[ch]

                if not gt_list:
                    ch_metrics["n_masked"] = 0
                    ch_metrics["error"] = "no_masked_positions"
                else:
                    gt_values = np.concatenate(gt_list)
                    pred_values = np.concatenate(pred_list)
                    ch_metrics["n_masked"] = len(gt_values)

                    # Round predictions to 0/1 for classification
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

        # Fallback substitution visibility (model-capability gap, orthogonal to
        # n_applicable/n_total which is data-quality). 0.0 when the imputer
        # produced finite values at every target cell — preserves parity for
        # the historical contract.
        asked_total = int(self.fallback_asked.sum())
        sub_total = int(self.fallback_substituted.sum())
        metrics["overall_fallback_rate"] = (
            float(sub_total) / asked_total if asked_total > 0 else 0.0
        )
        metrics["fallback_rate"] = {
            f"ch_{ch}": (
                float(self.fallback_substituted[ch]) / int(self.fallback_asked[ch])
                if self.fallback_asked[ch] > 0
                else 0.0
            )
            for ch in range(N_CHANNELS)
        }

        return metrics


@dataclass
class _ExtractedPairs:
    """Pre-extracted masked pairs — compact columns for PairWriter.

    Instead of shipping full (N, C, T) arrays back from workers, we extract
    only the ~20% of positions where artificial_masks == 1 inside the worker.
    This reduces per-batch pickle size from ~4.9 GB to ~150 MB.
    """

    sample_idx: np.ndarray  # (M,) int32, split-local
    channel: np.ndarray  # (M,) int8
    timestep: np.ndarray  # (M,) int16
    gt: np.ndarray  # (M,) float32
    pred: np.ndarray  # (M,) float32


def _extract_pairs(
    ground_truth: np.ndarray,
    imputed: np.ndarray,
    artificial_masks: np.ndarray,
    sample_indices: np.ndarray,
) -> _ExtractedPairs:
    """Extract masked (gt, pred) pairs into compact column arrays.

    Mirrors the extraction logic in PairWriter.write_batch but returns
    column arrays instead of writing to Parquet, so it can run in a worker.

    Args:
        ground_truth: Shape (N, C, T) — original unmasked values.
        imputed: Shape (N, C, T) — imputed values.
        artificial_masks: Shape (N, C, T) — binary, 1 = was artificially masked.
        sample_indices: Length-N array of split-local sample indices (int32).

    Returns:
        _ExtractedPairs with flat column arrays for masked positions only.
    """
    sample_indices = np.asarray(sample_indices, dtype=np.int32)

    # Vectorized extraction of masked positions
    mask_bool = artificial_masks == 1
    s_idx, c_idx, t_idx = np.where(mask_bool)

    if len(s_idx) == 0:
        return _ExtractedPairs(
            sample_idx=np.empty(0, dtype=np.int32),
            channel=np.empty(0, dtype=np.int8),
            timestep=np.empty(0, dtype=np.int16),
            gt=np.empty(0, dtype=np.float32),
            pred=np.empty(0, dtype=np.float32),
        )

    gt_vals = ground_truth[s_idx, c_idx, t_idx].astype(np.float32)
    pred_vals = imputed[s_idx, c_idx, t_idx].astype(np.float32)

    # Filter out non-finite gt or pred
    finite_mask = np.isfinite(gt_vals) & np.isfinite(pred_vals)
    if not np.all(finite_mask):
        s_idx = s_idx[finite_mask]
        c_idx = c_idx[finite_mask]
        t_idx = t_idx[finite_mask]
        gt_vals = gt_vals[finite_mask]
        pred_vals = pred_vals[finite_mask]

    if len(s_idx) == 0:
        return _ExtractedPairs(
            sample_idx=np.empty(0, dtype=np.int32),
            channel=np.empty(0, dtype=np.int8),
            timestep=np.empty(0, dtype=np.int16),
            gt=np.empty(0, dtype=np.float32),
            pred=np.empty(0, dtype=np.float32),
        )

    # Map batch-local sample index to split-local sample index
    mapped_sample_idx = sample_indices[s_idx]

    return _ExtractedPairs(
        sample_idx=mapped_sample_idx,
        channel=c_idx.astype(np.int8),
        timestep=t_idx.astype(np.int16),
        gt=gt_vals,
        pred=pred_vals,
    )


@dataclass
class BatchScenarioResult:
    """Result of evaluating all scenarios on a single batch.

    Used in parallel evaluation to return both overall and subgroup accumulators,
    plus optionally raw pair data for the main process PairWriter.
    """

    overall: MetricAccumulator | None
    subgroups: dict[str, dict[str, MetricAccumulator]] = field(
        default_factory=dict
    )  # attr -> group -> acc
    pair_data_list: list[_ExtractedPairs] = field(default_factory=list)


def _init_worker(
    mask_cache: MaskCache,
    method: ImputationMethod,
    scenarios: list[str],
    channel_stds: np.ndarray,
    split_name: str,
    subgroup_mapping: dict[int, dict[str, str]] | None = None,
    n_days: int = 1,
    window_descriptors: list[list[int]] | None = None,
    window_day_offsets: list[list[int]] | None = None,
    compute_metrics: bool = True,
    save_pairs: bool = True,
    fallback_fill: np.ndarray | None = None,
) -> None:
    """Initialize worker process with shared read-only data.

    Called once per worker process via ProcessPoolExecutor's initializer.
    Sets module-level globals to avoid serializing these objects per task.

    Args:
        mask_cache: Pre-generated masks for all scenarios and splits.
        method: Fitted imputation method.
        scenarios: List of scenario names to evaluate.
        channel_stds: Per-channel standard deviations for metric normalization.
        split_name: Name of the split (e.g. "val", "test").
        subgroup_mapping: Optional mapping from split-local index to demographic
            attributes for sensitivity analysis.
        n_days: Number of days per sample window (1 = single-day).
        window_descriptors: Per-split window descriptors for multi-day evaluation.
        window_day_offsets: Parallel structure to ``window_descriptors`` carrying
            per-day calendar offsets (``-1`` for padded slots). Forwarded to
            RoPE-aware imputation methods via ``impute(... day_offsets=...)``.
        compute_metrics: Whether to accumulate metrics in-memory.
        save_pairs: Whether to extract pairs for the main-process PairWriter.
        fallback_fill: Optional per-channel ``(C,)`` float32 array. When set,
            NaN cells at target positions are substituted in place and counted
            via ``MetricAccumulator.add_fallback``. ``None`` disables substitution.
    """
    global _worker_mask_cache, _worker_method, _worker_scenarios
    global _worker_channel_stds, _worker_fallback_fill, _worker_split_name
    global _worker_subgroup_mapping, _worker_n_days, _worker_window_descriptors
    global _worker_window_day_offsets, _worker_compute_metrics, _worker_save_pairs
    _worker_mask_cache = mask_cache
    _worker_method = method
    _worker_scenarios = scenarios
    _worker_channel_stds = channel_stds
    _worker_fallback_fill = fallback_fill
    _worker_split_name = split_name
    _worker_subgroup_mapping = subgroup_mapping
    _worker_n_days = n_days
    _worker_window_descriptors = window_descriptors
    _worker_window_day_offsets = window_day_offsets
    _worker_compute_metrics = compute_metrics
    _worker_save_pairs = save_pairs


def _evaluate_batch_all_scenarios(
    batch_data: tuple[np.ndarray, np.ndarray, list[int], int],
) -> dict[str, BatchScenarioResult]:
    """Evaluate ALL scenarios on one batch. Worker function for batch-level parallelism.

    Uses module-level globals for mask_cache, method, scenarios, channel_stds, split_name
    (set via pool initializer to avoid per-call serialization).

    When n_days > 1, assembles per-day masks into full-window masks, imputes the
    full multi-day window (model gets cross-day context), then slices back into
    per-day chunks for fair per-day metric computation.

    Args:
        batch_data: Tuple of (data, original_masks, batch_global_indices, batch_idx).

    Returns:
        Dict mapping scenario_name -> BatchScenarioResult for this batch.
    """
    data, original_masks, batch_global_indices, batch_idx = batch_data
    batch_len = len(data)
    results: dict[str, BatchScenarioResult] = {}
    n_days = _worker_n_days
    compute_metrics = _worker_compute_metrics

    for scenario_name in _worker_scenarios:
        accumulator = None
        if compute_metrics:
            accumulator = MetricAccumulator(_worker_channel_stds)

        sg_accs: dict[str, dict[str, MetricAccumulator]] = {}
        pair_data_list: list[_ExtractedPairs] = []

        if n_days == 1:
            # === SINGLE-DAY PATH ===
            if accumulator is not None:
                accumulator.increment_total(batch_len)

            applicable_local_indices, batch_art_masks = _worker_mask_cache.get_batch_masks(
                _worker_split_name, scenario_name, batch_global_indices
            )

            if len(applicable_local_indices) > 0:
                applicable_data = data[applicable_local_indices]
                applicable_orig = original_masks[applicable_local_indices]
                applicable_split_indices = [
                    batch_global_indices[li] for li in applicable_local_indices
                ]

                corrupted = applicable_data.copy()
                corrupted[batch_art_masks == 1] = np.nan

                impute_kwargs = {}
                if hasattr(_worker_method, "prepare_split"):
                    impute_kwargs["sample_indices"] = np.array(applicable_split_indices)
                imputed = _worker_method.impute(
                    corrupted, applicable_orig, batch_art_masks, **impute_kwargs
                )

                # Substitute NaN target cells with the channel-aware fallback fill
                # so they get scored (not silently dropped) and report visibility.
                fb_sub, fb_asked = _apply_fallback(imputed, batch_art_masks, _worker_fallback_fill)
                if accumulator is not None:
                    accumulator.add_fallback(fb_sub, fb_asked)

                if accumulator is not None:
                    accumulator.update(
                        ground_truth=applicable_data,
                        imputed=imputed,
                        artificial_masks=batch_art_masks,
                    )

                # Extract only masked pairs (compact) for main-process PairWriter
                if _worker_save_pairs:
                    pair_data_list.append(
                        _extract_pairs(
                            applicable_data,
                            imputed,
                            batch_art_masks,
                            np.array(applicable_split_indices, dtype=np.int32),
                        )
                    )

                # Subgroup accumulation
                if _worker_subgroup_mapping is not None and accumulator is not None:
                    sample_demo = _worker_subgroup_mapping.get(applicable_split_indices[0], {})
                    attributes = list(sample_demo.keys())

                    for attr in attributes:
                        groups: dict[str, list[int]] = defaultdict(list)
                        for i, split_idx in enumerate(applicable_split_indices):
                            group = _worker_subgroup_mapping.get(split_idx, {}).get(attr, "unknown")
                            groups[group].append(i)

                        if attr not in sg_accs:
                            sg_accs[attr] = {}

                        for group_name, group_indices in groups.items():
                            if group_name not in sg_accs[attr]:
                                sg_accs[attr][group_name] = MetricAccumulator(
                                    _worker_channel_stds,
                                )
                            idx = np.array(group_indices)
                            sg_accs[attr][group_name].update(
                                applicable_data[idx], imputed[idx], batch_art_masks[idx]
                            )
        else:
            # === MULTI-DAY PATH ===
            total_real_days = 0
            for window_idx in batch_global_indices:
                window_desc = _worker_window_descriptors[window_idx]
                total_real_days += sum(1 for d in window_desc if d != -1)
            if accumulator is not None:
                accumulator.increment_total(total_real_days)

            # Build full-window masks from per-day masks
            applicable_windows = []
            full_masks_list = []

            for w_local, window_idx in enumerate(batch_global_indices):
                window_desc = _worker_window_descriptors[window_idx]
                full_mask = np.zeros((N_CHANNELS, n_days * 1440), dtype=np.float32)
                has_any_mask = False

                for day_offset, day_split_idx in enumerate(window_desc):
                    if day_split_idx == -1:
                        continue
                    day_mask = _worker_mask_cache.get_single_mask(
                        _worker_split_name, scenario_name, day_split_idx
                    )
                    if day_mask is not None:
                        t_start = day_offset * 1440
                        full_mask[:, t_start : t_start + 1440] = day_mask
                        has_any_mask = True

                if has_any_mask:
                    applicable_windows.append(w_local)
                    full_masks_list.append(full_mask)

            if applicable_windows:
                applicable_data = data[applicable_windows]
                applicable_orig = original_masks[applicable_windows]
                batch_art_masks = np.stack(full_masks_list)

                # Corrupt and impute full multi-day windows
                corrupted = applicable_data.copy()
                corrupted[batch_art_masks == 1] = np.nan

                impute_kwargs = {}
                if hasattr(_worker_method, "prepare_split"):
                    window_sample_indices = []
                    for w_local in applicable_windows:
                        window_idx = batch_global_indices[w_local]
                        window_desc = _worker_window_descriptors[window_idx]
                        rep_idx = next((d for d in window_desc if d != -1), -1)
                        window_sample_indices.append(rep_idx)
                    impute_kwargs["sample_indices"] = np.array(window_sample_indices)
                if _worker_window_day_offsets is not None:
                    impute_kwargs["day_offsets"] = np.array(
                        [
                            _worker_window_day_offsets[batch_global_indices[w_local]]
                            for w_local in applicable_windows
                        ],
                        dtype=np.int64,
                    )
                imputed = _worker_method.impute(
                    corrupted, applicable_orig, batch_art_masks, **impute_kwargs
                )

                # Substitute NaN target cells once on the full multi-day window.
                # Per-channel counts sum across days via the (0,2) reduction.
                fb_sub, fb_asked = _apply_fallback(imputed, batch_art_masks, _worker_fallback_fill)
                if accumulator is not None:
                    accumulator.add_fallback(fb_sub, fb_asked)

                # Per-day metric computation and pair data
                for i, w_local in enumerate(applicable_windows):
                    window_idx = batch_global_indices[w_local]
                    window_desc = _worker_window_descriptors[window_idx]
                    for day_offset, day_split_idx in enumerate(window_desc):
                        if day_split_idx == -1:
                            continue
                        t_start = day_offset * 1440
                        t_end = t_start + 1440

                        day_art_mask = batch_art_masks[i : i + 1, :, t_start:t_end]
                        if day_art_mask.sum() == 0:
                            continue

                        day_gt = applicable_data[i : i + 1, :, t_start:t_end]
                        day_imputed = imputed[i : i + 1, :, t_start:t_end]

                        if accumulator is not None:
                            accumulator.update(day_gt, day_imputed, day_art_mask)

                        if _worker_save_pairs:
                            pair_data_list.append(
                                _extract_pairs(
                                    day_gt,
                                    day_imputed,
                                    day_art_mask,
                                    np.array([day_split_idx], dtype=np.int32),
                                )
                            )

        results[scenario_name] = BatchScenarioResult(
            overall=accumulator, subgroups=sg_accs, pair_data_list=pair_data_list
        )

    return results


class ImputationEvaluator:
    """Evaluator for imputation benchmarks using pre-generated masks.

    Orchestrates:
    1. Batch-by-batch loading of data
    2. Mask lookup from pre-generated MaskCache
    3. Imputation and incremental metric computation
    4. Optionally streaming raw (gt, pred) pairs to Parquet

    Key optimization: Each sample is loaded exactly once (via DataLoader),
    then evaluated for all scenarios before moving to the next batch.
    """

    def __init__(
        self,
        scenarios: list[str],
        num_eval_workers: int = 1,
        n_days: int = 1,
        compute_metrics: bool = True,
        save_pairs: bool = True,
        pairs_dir: str | Path | None = None,
    ):
        """Initialize the evaluator.

        Args:
            scenarios: List of scenario names to evaluate.
            num_eval_workers: Number of parallel workers for batch evaluation.
            n_days: Number of days per sample window (1 = single-day).
            compute_metrics: If True, accumulate and compute metrics in-memory.
                If False, only save pairs (requires save_pairs=True).
            save_pairs: If True, stream raw (gt, pred) pairs to Parquet.
            pairs_dir: Base directory for pair files. Structure:
                ``{pairs_dir}/{scenario}/{split}/pairs_ch{ch:02d}.parquet``
        """
        self.scenarios = scenarios
        self.num_eval_workers = num_eval_workers
        self.n_days = n_days
        self.compute_metrics = compute_metrics
        self.save_pairs = save_pairs
        self.pairs_dir = Path(pairs_dir) if pairs_dir is not None else None
        # Per-channel fallback fill for non-finite target cells; populated in ``run``.
        self._fallback_fill: np.ndarray | None = None

        if save_pairs and pairs_dir is None:
            raise ValueError("pairs_dir is required when save_pairs=True")

        if not compute_metrics and not save_pairs:
            raise ValueError("At least one of compute_metrics or save_pairs must be True")

    def run(
        self,
        val_loader: DataLoader | None,
        test_loader: DataLoader | None,
        mask_cache: MaskCache,
        method: ImputationMethod,
        channel_stds: np.ndarray,
        subgroup_mappings: dict[str, dict[int, dict[str, str]]] | None = None,
        window_descriptors: dict[str, list[list[int]]] | None = None,
        window_day_offsets: dict[str, list[list[int]]] | None = None,
        hf_dataset=None,
        split_indices: dict[str, list[int]] | None = None,
        zero_to_nan_transform=None,
        fallback_fill: np.ndarray | None = None,
    ) -> dict:
        """Run the imputation evaluation on val and test splits.

        Args:
            val_loader: DataLoader for validation split, or ``None`` to skip.
            test_loader: DataLoader for test split, or ``None`` to skip.
                Pass ``None`` from the runner when ``evaluation.eval_splits``
                excludes a split.
            mask_cache: Pre-generated masks for all scenarios and splits.
            method: Fitted imputation method.
            channel_stds: Per-channel standard deviations for metric normalization.
            subgroup_mappings: Optional mapping per split for sensitivity analysis.
            window_descriptors: Per-split window descriptors for multi-day evaluation.
            window_day_offsets: Per-split parallel structure to ``window_descriptors``
                carrying calendar-day offsets for each window's days. Forwarded to
                RoPE-aware imputation methods.
            hf_dataset: Optional HuggingFace dataset (for personalized methods).
            split_indices: Optional dict mapping split name to list of global indices.
            zero_to_nan_transform: Optional preprocessing transform (for personalized methods).
            fallback_fill: Optional per-channel ``(C,)`` float32 array used to
                substitute NaN cells at target positions the imputer failed
                to produce. ``None`` disables substitution (legacy passthrough);
                downstream ``isfinite`` filters then silently drop those cells
                as they did historically.

        Returns:
            Results dictionary with per-scenario metrics for val and test splits.
        """
        # Stash fallback_fill for sequential impute sites and the worker initializer.
        self._fallback_fill = fallback_fill

        # Save channel_stds alongside pairs for post-hoc aggregation
        if self.save_pairs and self.pairs_dir is not None:
            stds_path = self.pairs_dir / "channel_stds.npy"
            stds_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(stds_path, channel_stds)
            logger.info(f"Saved channel_stds to {stds_path}")

            # Save sample manifests for post-hoc sensitivity analysis
            if split_indices is not None and hf_dataset is not None:
                from imputation_evaluation.evaluation.pair_writer import write_sample_manifest

                # Extract lightweight columns once (avoid re-reading 3.8M rows per split)
                # Materialize to plain lists — HF datasets 4.x returns lazy Column
                # objects whose per-element random access is ~100x slower than list.
                all_user_ids = list(hf_dataset["user_id"])
                all_dates = list(hf_dataset["date"])

                for manifest_split in ("val", "test"):
                    if manifest_split in split_indices:
                        write_sample_manifest(
                            self.pairs_dir,
                            split_indices[manifest_split],
                            manifest_split,
                            all_user_ids,
                            all_dates,
                        )

                del all_user_ids, all_dates

        results = {"scenarios": {}}

        # Filter out splits whose loader is None — ``runner.py`` sets them to
        # None when ``evaluation.eval_splits`` excludes that split, so we can
        # skip the entire split block (prepare_split, worker init, batch
        # iteration) and never spawn the DataLoader's prefetch workers.
        for split_name, loader in [("val", val_loader), ("test", test_loader)]:
            if loader is None:
                logger.info(
                    "Skipping %s split (loader is None — see evaluation.eval_splits)", split_name
                )
                continue
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Evaluating {split_name} split...")
            logger.info(f"{'=' * 60}")

            # Prepare per-user statistics for personalized methods
            if hasattr(method, "prepare_split") and hf_dataset is not None:
                method.prepare_split(
                    hf_dataset,
                    split_indices[split_name],
                    zero_to_nan_transform,
                )

            split_subgroup_mapping = None
            if subgroup_mappings is not None:
                split_subgroup_mapping = subgroup_mappings.get(split_name)

            split_window_descriptors = None
            if window_descriptors is not None:
                split_window_descriptors = window_descriptors.get(split_name)

            split_window_day_offsets = None
            if window_day_offsets is not None:
                split_window_day_offsets = window_day_offsets.get(split_name)

            # Create PairWriters for this split (one per scenario)
            pair_writers: dict[str, PairWriter] = {}
            if self.save_pairs and self.pairs_dir is not None:
                from imputation_evaluation.evaluation.pair_writer import PairWriter as PW

                for scenario_name in self.scenarios:
                    pw_path = self.pairs_dir / scenario_name / split_name
                    pair_writers[scenario_name] = PW(pw_path)

            try:
                split_metrics = self._evaluate_split(
                    loader,
                    mask_cache,
                    method,
                    channel_stds,
                    split_name,
                    subgroup_mapping=split_subgroup_mapping,
                    window_descriptors=split_window_descriptors,
                    window_day_offsets=split_window_day_offsets,
                    pair_writers=pair_writers,
                )
            finally:
                # Always close PairWriters
                for pw in pair_writers.values():
                    pw.close()

            # Add to results structure (per-scenario)
            for scenario_name, metrics in split_metrics.items():
                if scenario_name not in results["scenarios"]:
                    results["scenarios"][scenario_name] = {}
                results["scenarios"][scenario_name][split_name] = metrics

            # Free memory from this split before starting the next one
            del split_metrics
            if split_subgroup_mapping is not None:
                del split_subgroup_mapping
            gc.collect()
            logger.info(f"Freed memory after {split_name} split evaluation.")

        return results

    def _evaluate_split(
        self,
        dataloader: DataLoader,
        mask_cache: MaskCache,
        method: ImputationMethod,
        channel_stds: np.ndarray,
        split_name: str,
        subgroup_mapping: dict[int, dict[str, str]] | None = None,
        window_descriptors: list[list[int]] | None = None,
        window_day_offsets: list[list[int]] | None = None,
        pair_writers: dict[str, PairWriter] | None = None,
    ) -> dict[str, dict]:
        """Evaluate a split using batch-by-batch processing.

        Args:
            dataloader: DataLoader for the split.
            mask_cache: Pre-generated masks.
            method: Fitted imputation method.
            channel_stds: Per-channel stds for metric normalization.
            split_name: Name of the split (for logging).
            subgroup_mapping: Optional mapping from split-local index to demographic
                attributes for sensitivity analysis.
            window_descriptors: Per-split window descriptors for multi-day evaluation.
            window_day_offsets: Parallel to ``window_descriptors`` carrying calendar
                offsets per day; forwarded to RoPE-aware methods.
            pair_writers: Optional dict of scenario_name -> PairWriter for streaming pairs.

        Returns:
            Dict mapping scenario names to their metrics.
        """
        compute_metrics = self.compute_metrics

        # Per-scenario metric accumulators (only if computing metrics)
        accumulators = {}
        if compute_metrics:
            accumulators = {name: MetricAccumulator(channel_stds) for name in self.scenarios}

        # Per-scenario subgroup accumulators: scenario -> attr -> group -> MetricAccumulator
        subgroup_accs: dict[str, dict[str, dict[str, MetricAccumulator]]] = {}
        if subgroup_mapping is not None and compute_metrics:
            for name in self.scenarios:
                subgroup_accs[name] = defaultdict(dict)

        n_samples = len(dataloader.dataset)
        use_parallel = self.num_eval_workers > 1

        if use_parallel:
            logger.info(
                f"Evaluating {split_name} split ({n_samples} samples) "
                f"with {self.num_eval_workers} parallel workers (batch-level)..."
            )
        else:
            logger.info(f"Evaluating {split_name} split ({n_samples} samples)...")

        if use_parallel:
            # Streaming batch-level parallelism
            batch_offset = 0
            in_flight: dict = {}  # future -> batch_idx

            with ProcessPoolExecutor(
                max_workers=self.num_eval_workers,
                initializer=_init_worker,
                initargs=(
                    mask_cache,
                    method,
                    self.scenarios,
                    channel_stds,
                    split_name,
                    subgroup_mapping,
                    self.n_days,
                    window_descriptors,
                    window_day_offsets,
                    compute_metrics,
                    self.save_pairs,
                    self._fallback_fill,
                ),
            ) as executor:
                for batch_idx, batch_data in enumerate(dataloader):
                    # Unpack batch — 3 elements when using SubsetWithOriginalIndices
                    if len(batch_data) == 3:
                        data, original_masks, original_indices = batch_data
                        data = data.numpy()
                        original_masks = original_masks.numpy()
                        batch_global_indices = original_indices.tolist()
                    else:
                        data, original_masks = batch_data
                        data = data.numpy()
                        original_masks = original_masks.numpy()
                        batch_global_indices = list(range(batch_offset, batch_offset + len(data)))
                    batch_len = len(data)
                    batch_offset += batch_len

                    # Throttle: wait for a slot before submitting
                    if len(in_flight) >= self.num_eval_workers:
                        done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                        for f in done:
                            self._merge_batch_result(
                                f.result(),
                                accumulators,
                                subgroup_accs,
                                channel_stds,
                                pair_writers=pair_writers,
                            )
                            del in_flight[f]

                    future = executor.submit(
                        _evaluate_batch_all_scenarios,
                        (data, original_masks, batch_global_indices, batch_idx),
                    )
                    in_flight[future] = batch_idx

                    if batch_idx % 10 == 0:
                        logger.info(
                            f"  Submitted batch {batch_idx + 1}, "
                            f"samples {batch_offset - batch_len} - {batch_offset}"
                        )

                # Drain remaining futures
                for f in in_flight:
                    self._merge_batch_result(
                        f.result(),
                        accumulators,
                        subgroup_accs,
                        channel_stds,
                        pair_writers=pair_writers,
                    )

        else:
            # Sequential evaluation (no multiprocessing overhead)
            batch_offset = 0
            n_days = self.n_days

            for batch_idx, batch_data in enumerate(dataloader):
                # Unpack batch — 3 elements when using SubsetWithOriginalIndices
                if len(batch_data) == 3:
                    data, original_masks, original_indices = batch_data
                    data = data.numpy()
                    original_masks = original_masks.numpy()
                    batch_global_indices = original_indices.tolist()
                else:
                    data, original_masks = batch_data
                    data = data.numpy()
                    original_masks = original_masks.numpy()
                    batch_global_indices = list(range(batch_offset, batch_offset + len(data)))
                batch_len = len(data)

                # Progress logging
                if batch_idx % 10 == 0 or batch_offset + batch_len == n_samples:
                    logger.info(
                        f"  Processing batch {batch_idx + 1}, "
                        f"samples {batch_offset} - {batch_offset + batch_len}"
                    )

                if n_days == 1:
                    # === SINGLE-DAY PATH ===
                    for scenario_name in self.scenarios:
                        if compute_metrics:
                            accumulators[scenario_name].increment_total(batch_len)

                        applicable_local_indices, batch_art_masks = mask_cache.get_batch_masks(
                            split_name, scenario_name, batch_global_indices
                        )

                        if len(applicable_local_indices) == 0:
                            continue

                        applicable_data = data[applicable_local_indices]
                        applicable_orig = original_masks[applicable_local_indices]

                        corrupted = applicable_data.copy()
                        corrupted[batch_art_masks == 1] = np.nan

                        impute_kwargs = {}
                        if hasattr(method, "prepare_split"):
                            applicable_split_indices = np.array(
                                [batch_global_indices[li] for li in applicable_local_indices]
                            )
                            impute_kwargs["sample_indices"] = applicable_split_indices
                        imputed = method.impute(
                            corrupted, applicable_orig, batch_art_masks, **impute_kwargs
                        )

                        # Substitute NaN target cells with the channel-aware fallback fill.
                        fb_sub, fb_asked = _apply_fallback(
                            imputed, batch_art_masks, self._fallback_fill
                        )
                        if compute_metrics:
                            accumulators[scenario_name].add_fallback(fb_sub, fb_asked)

                        if compute_metrics:
                            accumulators[scenario_name].update(
                                ground_truth=applicable_data,
                                imputed=imputed,
                                artificial_masks=batch_art_masks,
                            )

                        # Write pairs
                        if pair_writers and scenario_name in pair_writers:
                            sample_indices = np.array(
                                [batch_global_indices[li] for li in applicable_local_indices],
                                dtype=np.int32,
                            )
                            pair_writers[scenario_name].write_batch(
                                applicable_data, imputed, batch_art_masks, sample_indices
                            )

                        # Subgroup accumulation
                        if subgroup_mapping is not None and compute_metrics:
                            applicable_split_indices = [
                                batch_global_indices[li] for li in applicable_local_indices
                            ]
                            sample_demo = subgroup_mapping.get(applicable_split_indices[0], {})
                            attributes = list(sample_demo.keys())

                            for attr in attributes:
                                groups: dict[str, list[int]] = defaultdict(list)
                                for i, split_idx in enumerate(applicable_split_indices):
                                    group = subgroup_mapping.get(split_idx, {}).get(attr, "unknown")
                                    groups[group].append(i)

                                for group_name, group_indices in groups.items():
                                    if group_name not in subgroup_accs[scenario_name][attr]:
                                        subgroup_accs[scenario_name][attr][group_name] = (
                                            MetricAccumulator(channel_stds)
                                        )
                                    idx = np.array(group_indices)
                                    subgroup_accs[scenario_name][attr][group_name].update(
                                        applicable_data[idx],
                                        imputed[idx],
                                        batch_art_masks[idx],
                                    )
                else:
                    # === MULTI-DAY PATH ===
                    for scenario_name in self.scenarios:
                        # Count total real (non-padding) days
                        total_real_days = 0
                        for window_idx in batch_global_indices:
                            window_desc = window_descriptors[window_idx]
                            total_real_days += sum(1 for d in window_desc if d != -1)
                        if compute_metrics:
                            accumulators[scenario_name].increment_total(total_real_days)

                        # Build full-window masks from per-day masks
                        applicable_windows = []
                        full_masks_list = []

                        for w_local, window_idx in enumerate(batch_global_indices):
                            window_desc = window_descriptors[window_idx]
                            full_mask = np.zeros((N_CHANNELS, n_days * 1440), dtype=np.float32)
                            has_any_mask = False

                            for day_offset, day_split_idx in enumerate(window_desc):
                                if day_split_idx == -1:
                                    continue
                                day_mask = mask_cache.get_single_mask(
                                    split_name, scenario_name, day_split_idx
                                )
                                if day_mask is not None:
                                    t_start = day_offset * 1440
                                    full_mask[:, t_start : t_start + 1440] = day_mask
                                    has_any_mask = True

                            if has_any_mask:
                                applicable_windows.append(w_local)
                                full_masks_list.append(full_mask)

                        if not applicable_windows:
                            continue

                        applicable_data = data[applicable_windows]
                        applicable_orig = original_masks[applicable_windows]
                        batch_art_masks = np.stack(full_masks_list)

                        # Corrupt and impute full multi-day windows
                        corrupted = applicable_data.copy()
                        corrupted[batch_art_masks == 1] = np.nan

                        impute_kwargs = {}
                        if hasattr(method, "prepare_split"):
                            window_sample_indices = []
                            for w_local in applicable_windows:
                                window_idx = batch_global_indices[w_local]
                                window_desc = window_descriptors[window_idx]
                                rep_idx = next((d for d in window_desc if d != -1), -1)
                                window_sample_indices.append(rep_idx)
                            impute_kwargs["sample_indices"] = np.array(window_sample_indices)
                        if window_day_offsets is not None:
                            impute_kwargs["day_offsets"] = np.array(
                                [
                                    window_day_offsets[batch_global_indices[w_local]]
                                    for w_local in applicable_windows
                                ],
                                dtype=np.int64,
                            )
                        imputed = method.impute(
                            corrupted, applicable_orig, batch_art_masks, **impute_kwargs
                        )

                        # Substitute NaN target cells once on the full multi-day window.
                        fb_sub, fb_asked = _apply_fallback(
                            imputed, batch_art_masks, self._fallback_fill
                        )
                        if compute_metrics:
                            accumulators[scenario_name].add_fallback(fb_sub, fb_asked)

                        # Per-day metric computation and pair writing
                        for i, w_local in enumerate(applicable_windows):
                            window_idx = batch_global_indices[w_local]
                            window_desc = window_descriptors[window_idx]
                            for day_offset, day_split_idx in enumerate(window_desc):
                                if day_split_idx == -1:
                                    continue
                                t_start = day_offset * 1440
                                t_end = t_start + 1440

                                day_art_mask = batch_art_masks[i : i + 1, :, t_start:t_end]
                                if day_art_mask.sum() == 0:
                                    continue

                                day_gt = applicable_data[i : i + 1, :, t_start:t_end]
                                day_imputed = imputed[i : i + 1, :, t_start:t_end]

                                if compute_metrics:
                                    accumulators[scenario_name].update(
                                        day_gt, day_imputed, day_art_mask
                                    )

                                if pair_writers and scenario_name in pair_writers:
                                    pair_writers[scenario_name].write_batch(
                                        day_gt,
                                        day_imputed,
                                        day_art_mask,
                                        np.array([day_split_idx], dtype=np.int32),
                                    )

                batch_offset += batch_len

        # Finalize metrics for all scenarios
        scenario_metrics = {}
        for scenario_name in self.scenarios:
            if compute_metrics:
                accumulator = accumulators[scenario_name]
                metrics = accumulator.compute()
                metrics["n_applicable"] = accumulator.n_applicable
                metrics["n_total"] = accumulator.n_total

                # Add subgroup metrics if available
                if scenario_name in subgroup_accs:
                    metrics["subgroups"] = {}
                    for attr, groups in subgroup_accs[scenario_name].items():
                        metrics["subgroups"][attr] = {}
                        for group_name, sg_acc in sorted(groups.items()):
                            metrics["subgroups"][attr][group_name] = sg_acc.compute()

                scenario_metrics[scenario_name] = metrics
                logger.info(
                    f"  {scenario_name}: "
                    f"{accumulator.n_applicable}/{accumulator.n_total} applicable"
                )
            else:
                # Pairs-only mode: return placeholder metrics
                scenario_metrics[scenario_name] = {
                    "pairs_only": True,
                    "pairs_dir": str(self.pairs_dir / scenario_name / split_name)
                    if self.pairs_dir
                    else None,
                }
                logger.info(f"  {scenario_name}: pairs saved (metrics skipped)")

        return scenario_metrics

    @staticmethod
    def _merge_batch_result(
        batch_results: dict[str, BatchScenarioResult],
        accumulators: dict[str, MetricAccumulator],
        subgroup_accs: dict[str, dict[str, dict[str, MetricAccumulator]]],
        channel_stds: np.ndarray,
        pair_writers: dict[str, PairWriter] | None = None,
    ) -> None:
        """Merge a batch result (from parallel worker) into the main accumulators."""
        for scenario_name, batch_result in batch_results.items():
            if batch_result.overall is not None and scenario_name in accumulators:
                accumulators[scenario_name].merge(batch_result.overall)

            # Write pre-extracted pairs in main process
            if pair_writers and scenario_name in pair_writers:
                for pd in batch_result.pair_data_list:
                    pair_writers[scenario_name].write_extracted_pairs(
                        pd.sample_idx, pd.channel, pd.timestep, pd.gt, pd.pred
                    )

            if batch_result.subgroups and scenario_name in subgroup_accs:
                for attr, groups in batch_result.subgroups.items():
                    for group_name, sg_acc in groups.items():
                        if group_name not in subgroup_accs[scenario_name][attr]:
                            subgroup_accs[scenario_name][attr][group_name] = MetricAccumulator(
                                channel_stds,
                            )
                        subgroup_accs[scenario_name][attr][group_name].merge(sg_acc)
