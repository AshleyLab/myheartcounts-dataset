"""Library entry-point for the imputation eval pipeline.

This is the orchestration code that lives only in
``scripts/run_imputation_eval.py`` in the private repo. We extracted it
here so the public ``openmhc`` API can call the eval pipeline without
shelling out to a script.

The function accepts an already-built ``ImputationEvalConfig`` plus an
``ImputationMethod``-compatible object (anything with ``name``,
``channel_stds``, ``fit``, ``impute``). It returns the raw results dict
produced by :class:`imputation_evaluation.evaluation.evaluator.ImputationEvaluator`.

Skipped vs. the script: writer/output writing, W&B logging,
visualization plots, sensitivity-analysis subgroup mappings (callers can
pass ``subgroup_mappings`` if they want).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from imputation_evaluation.data.data_loader import ImputationDataLoader
from imputation_evaluation.evaluation.evaluator import ImputationEvaluator
from imputation_evaluation.masking import MaskCacheGenerator, create_mask_generators

if TYPE_CHECKING:
    from imputation_evaluation.config import ImputationEvalConfig
    from imputation_evaluation.methods.base import ImputationMethod

logger = logging.getLogger(__name__)


def run_eval(
    config: "ImputationEvalConfig",
    method: "ImputationMethod",
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
        emitted by :class:`ImputationEvaluator`. Plus a ``"config"`` key.
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

    # 3. Generate masks (no pre-cached masks supported in this entry point).
    logger.info("Generating masks...")
    mask_generator = MaskCacheGenerator(
        hf_dataset=loaded_data.hf_dataset,
        zero_to_nan_transform=loaded_data.zero_to_nan_transform,
        num_workers=config.data.num_workers,
        batch_size=config.data.batch_size,
    )
    mask_cache = mask_generator.generate(
        split_indices={
            "val": loaded_data.split_indices["val"],
            "test": loaded_data.split_indices["test"],
        },
        generators=generators,
        base_seed=config.masking.mask_seed,
    )

    # 4. Fit the method on training data.
    logger.info("Fitting %s on training data...", method.name)
    method.fit(loaded_data.train_loader)

    channel_stds = method.channel_stds
    if channel_stds is None:
        channel_stds = np.ones(19)

    # 5. Pre-filter eval samples to only those with at least one mask.
    applicable_indices = None
    if mask_cache is not None:
        applicable_indices = {}
        for split_name in ("val", "test"):
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
    eval_val_loader, eval_test_loader = data_loader.create_eval_loaders(
        split_indices=loaded_data.split_indices,
        hf_dataset=loaded_data.hf_dataset,
        batch_size=config.data.batch_size,
        num_workers=eval_dl_workers,
        pin_memory=config.data.pin_memory,
        window_descriptors=loaded_data.window_descriptors,
        window_day_offsets=loaded_data.window_day_offsets,
        applicable_indices=applicable_indices,
    )

    # 7. Run evaluation.
    evaluator = ImputationEvaluator(
        scenarios=scenario_names,
        num_eval_workers=config.data.num_eval_workers,
        include_ks=config.evaluation.include_ks,
        include_wasserstein=config.evaluation.include_wasserstein,
        n_days=config.data.n_days,
        compute_metrics=config.evaluation.compute_metrics,
        save_pairs=False,  # public API doesn't need raw pairs on disk
        pairs_dir=None,
    )
    results = evaluator.run(
        val_loader=eval_val_loader,
        test_loader=eval_test_loader,
        mask_cache=mask_cache,
        method=method,
        channel_stds=channel_stds,
        subgroup_mappings=subgroup_mappings,
        window_descriptors=loaded_data.window_descriptors,
        window_day_offsets=loaded_data.window_day_offsets,
        hf_dataset=loaded_data.hf_dataset,
        split_indices=loaded_data.split_indices,
        zero_to_nan_transform=loaded_data.zero_to_nan_transform,
    )

    results["config"] = {
        "method": method.name,
        "seed": config.seed,
        "mask_seed": config.masking.mask_seed,
    }
    return results
