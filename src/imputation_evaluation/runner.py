"""Library entry-point for the imputation eval pipeline.

This is the orchestration code that lives only in
``scripts/run_imputation_eval.py`` in the private repo. We extracted it
here so the public ``openmhc`` API can call the eval pipeline without
shelling out to a script.

The function accepts an already-built ``ImputationEvalConfig`` plus an
``ImputationMethod``-compatible object (anything with ``name``,
``channel_stds``, ``fit``, ``impute``). It returns the raw results dict
produced by :class:`imputation_evaluation.evaluation.evaluator.ImputationEvaluator`
plus the canonical producer's display metrics merged in.

Skipped vs. the script: writer/output writing, W&B logging,
visualization plots, sensitivity-analysis subgroup mappings (callers can
pass ``subgroup_mappings`` if they want).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from imputation_evaluation.data.data_loader import ImputationDataLoader
from imputation_evaluation.evaluation.evaluator import ImputationEvaluator
from imputation_evaluation.evaluation.pair_writer import (
    merge_counts_and_fallback,
    read_fallback_sidecar,
)
from imputation_evaluation.evaluation.per_user_errors import (
    build_per_user_errors,
    write_per_user_errors_parquet,
)
from imputation_evaluation.masking import MaskCache, MaskCacheGenerator, create_mask_generators

if TYPE_CHECKING:
    from imputation_evaluation.config import ImputationEvalConfig
    from imputation_evaluation.methods.base import ImputationMethod

logger = logging.getLogger(__name__)

N_CHANNELS = 19


def run_eval(
    config: ImputationEvalConfig,
    method: ImputationMethod,
    *,
    subgroup_mappings: dict | None = None,
) -> dict:
    """Run the imputation evaluation pipeline end-to-end.

    Mirrors the orchestration in ``scripts/run_imputation_eval.py`` but
    skips writer/W&B/viz so it's safe to call from a Python process that
    just wants the metrics dict back.

    Args:
        config: Fully-populated ``ImputationEvalConfig``.
        method: An object implementing the :class:`ImputationMethod`
            protocol. The OpenMHC public API wraps user-supplied
            ``Imputer`` objects in an internal adapter to satisfy this.
        subgroup_mappings: Optional pre-built subgroup mappings for
            sensitivity analysis. Pass ``None`` to skip.

    Returns:
        The raw results dict (scenario → split → group → metric → value)
        emitted by the canonical user-macro producer + fallback sidecar.
        Plus a ``"config"`` key.
    """
    # 1. Load data.
    logger.info("Loading data...")
    data_loader = ImputationDataLoader(config.data)
    loaded_data = data_loader.load_splits(
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )

    # 2. Create mask generators.
    generators = create_mask_generators(config.masking)
    scenario_names = [g.name for g in generators]
    logger.info("Created %d mask generators: %s", len(generators), scenario_names)

    # 3. Load pre-computed masks or generate fresh ones.
    # ``evaluation.eval_splits`` lets a focused rerun skip val (default
    # ``["val", "test"]`` preserves historical behavior). Skipping val
    # shrinks the in-memory mask cache and halves the parallel-worker
    # fork-CoW cost when memory is tight.
    eval_splits = list(config.evaluation.eval_splits)
    if not eval_splits:
        raise ValueError("evaluation.eval_splits must contain at least one of {'val','test'}")
    for s in eval_splits:
        if s not in ("val", "test"):
            raise ValueError(
                f"evaluation.eval_splits got {s!r}; only 'val' and 'test' are supported"
            )
    if eval_splits != ["val", "test"]:
        logger.info("evaluation.eval_splits=%s (default is ['val','test'])", eval_splits)
    if config.masking.masks_file:
        masks_path = Path(config.masking.masks_file)
        logger.info("Loading pre-computed masks from %s...", masks_path)
        mask_cache = MaskCache.load(
            masks_dir=masks_path,
            scenarios=scenario_names,
            splits=eval_splits,
        )
    else:
        logger.info("Generating masks...")
        mask_generator = MaskCacheGenerator(
            hf_dataset=loaded_data.hf_dataset,
            zero_to_nan_transform=loaded_data.zero_to_nan_transform,
            num_workers=config.data.num_workers,
            batch_size=config.data.batch_size,
        )
        mask_cache = mask_generator.generate(
            split_indices={s: loaded_data.split_indices[s] for s in eval_splits},
            generators=generators,
            base_seed=config.masking.mask_seed,
        )

    # 4. Fit the method on training data.
    logger.info("Fitting %s on training data...", method.name)
    method.fit(loaded_data.train_loader)

    channel_stds = method.channel_stds
    if channel_stds is None:
        channel_stds = np.ones(N_CHANNELS)

    # Optional per-channel fallback fill: when the method exposes it, the
    # evaluator substitutes non-finite imputed cells at target positions and
    # surfaces the substitution rate. Adapters built by the public OpenMHC API
    # populate this from the same train-pass means.
    fallback_fill = getattr(method, "fallback_fill", None)

    # 5. Pre-filter eval samples to only those with at least one mask.
    applicable_indices = None
    if mask_cache is not None:
        applicable_indices = {}
        for split_name in eval_splits:
            indices = mask_cache.get_applicable_indices(split_name)
            if indices:
                applicable_indices[split_name] = indices
        if not applicable_indices:
            applicable_indices = None

    # 6. Build eval-specific data loaders.
    eval_dl_workers = (
        config.data.num_eval_dl_workers
        if config.data.num_eval_dl_workers is not None
        else config.data.num_workers
    )
    # Personalized imputers under the lazy per-user state contract need
    # user-grouped batches to avoid evict-and-reload thrashing inside
    # impute(). The flag propagates through the imputer adapter via the
    # ``requires_user_grouped_batches`` attribute on the wrapped imputer.
    user_grouped_batches = bool(
        getattr(method, "requires_user_grouped_batches", False)
        or getattr(getattr(method, "_imputer", None), "requires_user_grouped_batches", False)
    )
    if user_grouped_batches:
        logger.info(
            "Method %s requires user-grouped eval batches; flipping "
            "create_eval_loaders into user-grouped mode.",
            method.name,
        )
    eval_val_loader, eval_test_loader = data_loader.create_eval_loaders(
        split_indices=loaded_data.split_indices,
        hf_dataset=loaded_data.hf_dataset,
        batch_size=config.data.batch_size,
        num_workers=eval_dl_workers,
        pin_memory=config.data.pin_memory,
        window_descriptors=loaded_data.window_descriptors,
        window_day_offsets=loaded_data.window_day_offsets,
        applicable_indices=applicable_indices,
        user_grouped_batches=user_grouped_batches,
    )
    # Drop loaders for splits we're not evaluating so the unused DataLoader's
    # prefetch workers (and their batches) are never spawned and the
    # evaluator skips that split entirely.
    if "val" not in eval_splits:
        eval_val_loader = None
    if "test" not in eval_splits:
        eval_test_loader = None

    # 7. Run evaluation. Pairs are always written; the canonical producer
    # reduces them to display metrics below.
    pairs_dir = Path(config.output.results_dir) / "pairs"
    logger.info("Pairs dir: %s", pairs_dir)

    evaluator = ImputationEvaluator(
        scenarios=scenario_names,
        num_eval_workers=config.data.num_eval_workers,
        n_days=config.data.n_days,
        pairs_dir=pairs_dir,
    )
    evaluator.run(
        val_loader=eval_val_loader,
        test_loader=eval_test_loader,
        mask_cache=mask_cache,
        method=method,
        channel_stds=channel_stds,
        window_descriptors=loaded_data.window_descriptors,
        window_day_offsets=loaded_data.window_day_offsets,
        hf_dataset=loaded_data.hf_dataset,
        split_indices=loaded_data.split_indices,
        zero_to_nan_transform=loaded_data.zero_to_nan_transform,
        fallback_fill=fallback_fill,
    )

    # 8. Canonical producer: read pair files + fallback sidecars back into
    # display metrics + a per-user errors frame. The producer owns both
    # the user-macro reduction and the subgroup partitioning — the worker no
    # longer accumulates subgroup metrics.
    per_user_df, display = build_per_user_errors(
        method_pairs_dir=pairs_dir,
        method_name=method.name,
        scenarios=scenario_names,
        splits=eval_splits,
        subgroup_mappings=subgroup_mappings,
        include_auc=True,
        exclude_unknown=False,
    )

    # Merge fallback sidecar (model-capability counts) with the producer's
    # display metrics to rebuild the public ``scenarios`` dict shape.
    sidecars: dict[str, dict | None] = {
        split: read_fallback_sidecar(pairs_dir, split) for split in eval_splits
    }

    scenarios_out: dict[str, dict] = {}
    for scenario in scenario_names:
        scenarios_out[scenario] = {}
        for split in eval_splits:
            sidecar = sidecars.get(split) or {}
            scenario_sidecar = sidecar.get(scenario)
            counts_block = merge_counts_and_fallback(scenario_sidecar, n_channels=N_CHANNELS)
            disp = display.get((scenario, split), {})
            merged = {
                "per_channel": disp.get("per_channel", {}),
                "continuous": disp.get("continuous", {}),
                "binary": disp.get("binary", {}),
                "n_samples": disp.get("n_samples", 0),
            }
            merged.update(counts_block)
            scenarios_out[scenario][split] = merged

    results: dict = {"scenarios": scenarios_out}

    # 9. Persist the canonical per-user errors substrate alongside results.
    results_dir = Path(config.output.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    per_user_path = results_dir / "per_user_errors.parquet"
    write_per_user_errors_parquet(per_user_df, per_user_path)
    logger.info("Wrote per-user errors (%d rows) to %s", len(per_user_df), per_user_path)

    # TODO(phase-a): wire optional skill_scores against a baseline_errors path
    # once the runner signature exposes it; for now downstream callers (paper
    # pipeline, public API) compute skill scores themselves from per_user_df.

    _summarize_fallback(results)

    results["config"] = {
        "method": method.name,
        "seed": config.seed,
        "mask_seed": config.masking.mask_seed,
    }
    return results


def _summarize_fallback(results: dict) -> None:
    """Emit a prominent warning for any (scenario, split) with non-zero fallback rate.

    The fallback rate reports the fraction of target cells the imputer left
    non-finite and that the harness substituted with the channel-aware global
    fill. A non-zero rate is a model-capability gap — orthogonal to data-quality
    drops (``n_applicable`` / ``n_total``). Mirrors Track 3 forecasting's
    fallback summary.
    """
    scenarios = results.get("scenarios", {})
    flagged: list[tuple[str, str, float]] = []
    for scenario, split_map in scenarios.items():
        if not isinstance(split_map, dict):
            continue
        for split, metrics in split_map.items():
            if not isinstance(metrics, dict):
                continue
            rate = metrics.get("overall_fallback_rate")
            if isinstance(rate, (int, float)) and rate > 0.0:
                flagged.append((scenario, split, float(rate)))
    if not flagged:
        return
    logger.warning(
        "Imputer fallback substitution detected (model returned non-finite at "
        "target cells; harness filled with channel-aware global baseline). "
        "Per-(scenario, split) rates:"
    )
    for scenario, split, rate in flagged:
        logger.warning("  %s/%s: overall_fallback_rate=%.4f", scenario, split, rate)
