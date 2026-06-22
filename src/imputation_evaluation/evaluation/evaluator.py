"""Main evaluator for imputation evaluation.

Orchestrates data loading, imputation, and pair-streaming using pre-generated masks.
Uses batch-by-batch processing for memory efficiency.

Parallelism strategy:
- When num_eval_workers > 1, uses ProcessPoolExecutor for batch-level parallelism
- Each worker process handles ALL scenarios for its assigned batch
- This avoids GIL contention (vs ThreadPoolExecutor) since each process has its own interpreter

Phase-A note: cell-micro metric accumulation has been deleted. The evaluator now
only writes per-channel pair files + a fallback sidecar; the canonical user-macro
producer (``build_per_user_errors``) reads those artifacts back to populate
display metrics in the runner.
"""

from __future__ import annotations

import gc
import logging
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from torch.utils.data import DataLoader

from imputation_evaluation.evaluation.pair_writer import write_fallback_sidecar

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
_worker_n_days: int = 1
_worker_window_descriptors: list[list[int]] | None = None
_worker_window_day_offsets: list[list[int]] | None = None

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


def _empty_counts() -> dict:
    """Return a fresh per-scenario counts dict (zeroed)."""
    return {
        "n_applicable": 0,
        "n_total": 0,
        "fallback_substituted": np.zeros(N_CHANNELS, dtype=np.int64),
        "fallback_asked": np.zeros(N_CHANNELS, dtype=np.int64),
    }


def _merge_counts(dst: dict, src: dict) -> None:
    """Sum integers + numpy arrays from ``src`` into ``dst`` in place."""
    dst["n_applicable"] += int(src["n_applicable"])
    dst["n_total"] += int(src["n_total"])
    dst["fallback_substituted"] += np.asarray(src["fallback_substituted"], dtype=np.int64)
    dst["fallback_asked"] += np.asarray(src["fallback_asked"], dtype=np.int64)


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

    Carries the per-scenario counts (applicability + fallback) and the
    list of extracted pair columns for the main-process PairWriter. The
    counts dict has the same shape as :func:`_empty_counts`.
    """

    counts: dict
    pair_data_list: list[_ExtractedPairs] = field(default_factory=list)


def _init_worker(
    mask_cache: MaskCache,
    method: ImputationMethod,
    scenarios: list[str],
    channel_stds: np.ndarray,
    split_name: str,
    n_days: int = 1,
    window_descriptors: list[list[int]] | None = None,
    window_day_offsets: list[list[int]] | None = None,
    fallback_fill: np.ndarray | None = None,
) -> None:
    """Initialize worker process with shared read-only data.

    Called once per worker process via ProcessPoolExecutor's initializer.
    Sets module-level globals to avoid serializing these objects per task.

    Args:
        mask_cache: Pre-generated masks for all scenarios and splits.
        method: Fitted imputation method.
        scenarios: List of scenario names to evaluate.
        channel_stds: Per-channel standard deviations (forwarded for future
            use; not consumed by the worker after Phase-A).
        split_name: Name of the split (e.g. "val", "test").
        n_days: Number of days per sample window (1 = single-day).
        window_descriptors: Per-split window descriptors for multi-day evaluation.
        window_day_offsets: Parallel structure to ``window_descriptors`` carrying
            per-day calendar offsets (``-1`` for padded slots). Forwarded to
            RoPE-aware imputation methods via ``impute(... day_offsets=...)``.
        fallback_fill: Optional per-channel ``(C,)`` float32 array. When set,
            NaN cells at target positions are substituted in place and counted
            into the batch counts. ``None`` disables substitution.
    """
    global _worker_mask_cache, _worker_method, _worker_scenarios
    global _worker_channel_stds, _worker_fallback_fill, _worker_split_name
    global _worker_n_days, _worker_window_descriptors, _worker_window_day_offsets
    _worker_mask_cache = mask_cache
    _worker_method = method
    _worker_scenarios = scenarios
    _worker_channel_stds = channel_stds
    _worker_fallback_fill = fallback_fill
    _worker_split_name = split_name
    _worker_n_days = n_days
    _worker_window_descriptors = window_descriptors
    _worker_window_day_offsets = window_day_offsets


def _evaluate_batch_all_scenarios(
    batch_data: tuple[np.ndarray, np.ndarray, list[int], int],
) -> dict[str, BatchScenarioResult]:
    """Evaluate ALL scenarios on one batch. Worker function for batch-level parallelism.

    Uses module-level globals for mask_cache, method, scenarios, channel_stds, split_name
    (set via pool initializer to avoid per-call serialization).

    When n_days > 1, assembles per-day masks into full-window masks, imputes the
    full multi-day window (model gets cross-day context), then slices back into
    per-day chunks for fair per-day pair extraction.

    Args:
        batch_data: Tuple of (data, original_masks, batch_global_indices, batch_idx).

    Returns:
        Dict mapping scenario_name -> BatchScenarioResult for this batch.
    """
    data, original_masks, batch_global_indices, batch_idx = batch_data
    batch_len = len(data)
    results: dict[str, BatchScenarioResult] = {}
    n_days = _worker_n_days

    for scenario_name in _worker_scenarios:
        counts = _empty_counts()
        pair_data_list: list[_ExtractedPairs] = []

        if n_days == 1:
            # === SINGLE-DAY PATH ===
            counts["n_total"] += batch_len

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
                counts["fallback_substituted"] += fb_sub
                counts["fallback_asked"] += fb_asked
                counts["n_applicable"] += len(applicable_local_indices)

                # Extract only masked pairs (compact) for main-process PairWriter
                pair_data_list.append(
                    _extract_pairs(
                        applicable_data,
                        imputed,
                        batch_art_masks,
                        np.array(applicable_split_indices, dtype=np.int32),
                    )
                )
        else:
            # === MULTI-DAY PATH ===
            total_real_days = 0
            for window_idx in batch_global_indices:
                window_desc = _worker_window_descriptors[window_idx]
                total_real_days += sum(1 for d in window_desc if d != -1)
            counts["n_total"] += total_real_days

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
                counts["fallback_substituted"] += fb_sub
                counts["fallback_asked"] += fb_asked

                # Per-day pair extraction
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
                        counts["n_applicable"] += 1

                        pair_data_list.append(
                            _extract_pairs(
                                day_gt,
                                day_imputed,
                                day_art_mask,
                                np.array([day_split_idx], dtype=np.int32),
                            )
                        )

        results[scenario_name] = BatchScenarioResult(
            counts=counts, pair_data_list=pair_data_list
        )

    return results


class ImputationEvaluator:
    """Evaluator for imputation benchmarks using pre-generated masks.

    Orchestrates:
    1. Batch-by-batch loading of data
    2. Mask lookup from pre-generated MaskCache
    3. Imputation, fallback substitution, and per-channel pair streaming
    4. Per-(scenario, split) counts (applicability + fallback) emitted via a
       sidecar JSON next to the pair files

    Key optimization: Each sample is loaded exactly once (via DataLoader),
    then evaluated for all scenarios before moving to the next batch.
    """

    def __init__(
        self,
        scenarios: list[str],
        num_eval_workers: int = 1,
        n_days: int = 1,
        pairs_dir: str | Path | None = None,
    ):
        """Initialize the evaluator.

        Args:
            scenarios: List of scenario names to evaluate.
            num_eval_workers: Number of parallel workers for batch evaluation.
            n_days: Number of days per sample window (1 = single-day).
            pairs_dir: Base directory for pair files. Structure:
                ``{pairs_dir}/{scenario}/{split}/pairs_ch{ch:02d}.parquet``.
                Required — pairs are always written.
        """
        if pairs_dir is None:
            raise ValueError("pairs_dir is required (pairs are always written).")
        self.scenarios = scenarios
        self.num_eval_workers = num_eval_workers
        self.n_days = n_days
        self.pairs_dir = Path(pairs_dir)
        # Per-channel fallback fill for non-finite target cells; populated in ``run``.
        self._fallback_fill: np.ndarray | None = None

    def run(
        self,
        val_loader: DataLoader | None,
        test_loader: DataLoader | None,
        mask_cache: MaskCache,
        method: ImputationMethod,
        channel_stds: np.ndarray,
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
            channel_stds: Per-channel standard deviations for metric normalization;
                persisted to ``pairs_dir/channel_stds.npy`` for post-hoc producers.
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
            ``{"scenarios": {scenario: {split: counts_dict}}}`` — the
            ``counts_dict`` matches the shape persisted by
            :func:`write_fallback_sidecar`. The runner reads it back via
            :func:`read_fallback_sidecar` to populate the display dict.
        """
        # Stash fallback_fill for sequential impute sites and the worker initializer.
        self._fallback_fill = fallback_fill

        # Save channel_stds alongside pairs for post-hoc aggregation
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

            split_window_descriptors = None
            if window_descriptors is not None:
                split_window_descriptors = window_descriptors.get(split_name)

            split_window_day_offsets = None
            if window_day_offsets is not None:
                split_window_day_offsets = window_day_offsets.get(split_name)

            # Create PairWriters for this split (one per scenario)
            pair_writers: dict[str, PairWriter] = {}
            from imputation_evaluation.evaluation.pair_writer import PairWriter as PW

            for scenario_name in self.scenarios:
                pw_path = self.pairs_dir / scenario_name / split_name
                pair_writers[scenario_name] = PW(pw_path)

            try:
                split_counts = self._evaluate_split(
                    loader,
                    mask_cache,
                    method,
                    channel_stds,
                    split_name,
                    window_descriptors=split_window_descriptors,
                    window_day_offsets=split_window_day_offsets,
                    pair_writers=pair_writers,
                )
            finally:
                # Always close PairWriters
                for pw in pair_writers.values():
                    pw.close()

            # Persist the fallback sidecar for this split (canonical producer reads it back).
            write_fallback_sidecar(self.pairs_dir, split_name, split_counts)

            # Add to results structure (per-scenario)
            for scenario_name, counts in split_counts.items():
                if scenario_name not in results["scenarios"]:
                    results["scenarios"][scenario_name] = {}
                results["scenarios"][scenario_name][split_name] = counts

            # Free memory from this split before starting the next one
            del split_counts
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
        window_descriptors: list[list[int]] | None = None,
        window_day_offsets: list[list[int]] | None = None,
        pair_writers: dict[str, PairWriter] | None = None,
    ) -> dict[str, dict]:
        """Evaluate a split using batch-by-batch processing.

        Args:
            dataloader: DataLoader for the split.
            mask_cache: Pre-generated masks.
            method: Fitted imputation method.
            channel_stds: Per-channel stds (forwarded to workers for future use).
            split_name: Name of the split (for logging).
            window_descriptors: Per-split window descriptors for multi-day evaluation.
            window_day_offsets: Parallel to ``window_descriptors`` carrying calendar
                offsets per day; forwarded to RoPE-aware methods.
            pair_writers: Dict of scenario_name -> PairWriter for streaming pairs.

        Returns:
            ``{scenario: counts_dict}`` — counts_dict shape matches
            :func:`_empty_counts`.
        """
        counts: dict[str, dict] = {name: _empty_counts() for name in self.scenarios}

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
                    self.n_days,
                    window_descriptors,
                    window_day_offsets,
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
                                counts,
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
                        counts,
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
                        counts[scenario_name]["n_total"] += batch_len

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
                        counts[scenario_name]["fallback_substituted"] += fb_sub
                        counts[scenario_name]["fallback_asked"] += fb_asked
                        counts[scenario_name]["n_applicable"] += len(applicable_local_indices)

                        # Write pairs
                        if pair_writers and scenario_name in pair_writers:
                            sample_indices = np.array(
                                [batch_global_indices[li] for li in applicable_local_indices],
                                dtype=np.int32,
                            )
                            pair_writers[scenario_name].write_batch(
                                applicable_data, imputed, batch_art_masks, sample_indices
                            )
                else:
                    # === MULTI-DAY PATH ===
                    for scenario_name in self.scenarios:
                        # Count total real (non-padding) days
                        total_real_days = 0
                        for window_idx in batch_global_indices:
                            window_desc = window_descriptors[window_idx]
                            total_real_days += sum(1 for d in window_desc if d != -1)
                        counts[scenario_name]["n_total"] += total_real_days

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
                        counts[scenario_name]["fallback_substituted"] += fb_sub
                        counts[scenario_name]["fallback_asked"] += fb_asked

                        # Per-day pair writing
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
                                counts[scenario_name]["n_applicable"] += 1

                                if pair_writers and scenario_name in pair_writers:
                                    pair_writers[scenario_name].write_batch(
                                        day_gt,
                                        day_imputed,
                                        day_art_mask,
                                        np.array([day_split_idx], dtype=np.int32),
                                    )

                batch_offset += batch_len

        for scenario_name in self.scenarios:
            logger.info(
                f"  {scenario_name}: "
                f"{counts[scenario_name]['n_applicable']}/{counts[scenario_name]['n_total']} "
                "applicable"
            )

        return counts

    @staticmethod
    def _merge_batch_result(
        batch_results: dict[str, BatchScenarioResult],
        counts: dict[str, dict],
        pair_writers: dict[str, PairWriter] | None = None,
    ) -> None:
        """Merge a batch result (from parallel worker) into the main counts dict."""
        for scenario_name, batch_result in batch_results.items():
            if scenario_name in counts:
                _merge_counts(counts[scenario_name], batch_result.counts)

            # Write pre-extracted pairs in main process
            if pair_writers and scenario_name in pair_writers:
                for pd in batch_result.pair_data_list:
                    pair_writers[scenario_name].write_extracted_pairs(
                        pd.sample_idx, pd.channel, pd.timestep, pd.gt, pd.pred
                    )
