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

import json
import logging
from typing import TYPE_CHECKING

import numpy as np

from pathlib import Path

from imputation_evaluation.data.data_loader import ImputationDataLoader
from imputation_evaluation.evaluation.evaluator import ImputationEvaluator
from imputation_evaluation.masking import MaskCache, MaskCacheGenerator, create_mask_generators

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

    # 3. Load pre-computed masks or generate fresh ones.
    if config.masking.masks_file:
        masks_path = Path(config.masking.masks_file)
        logger.info("Loading pre-computed masks from %s...", masks_path)
        mask_cache = MaskCache.load(
            masks_dir=masks_path,
            scenarios=scenario_names,
            splits=["val", "test"],
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

    # Optional per-channel fallback fill: when the method exposes it, the
    # evaluator substitutes non-finite imputed cells at target positions and
    # surfaces the substitution rate. Adapters built by the public OpenMHC API
    # populate this from the same train-pass means.
    fallback_fill = getattr(method, "fallback_fill", None)

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

    # 7. Run evaluation. Pair-saving is enabled when either evaluation.save_pairs is
    # set in the config (e.g. for the paper pipeline) or when bootstrap is enabled
    # (the cluster bootstrap operates post-hoc against pair files).
    bootstrap_cfg = getattr(config, "bootstrap", None)
    bootstrap_enabled = bool(bootstrap_cfg is not None and bootstrap_cfg.enabled)
    save_pairs = bool(config.evaluation.save_pairs) or bootstrap_enabled
    pairs_dir: Path | None = None
    if save_pairs:
        pairs_dir = Path(config.output.results_dir) / "pairs"
        logger.info(
            "save_pairs=True (pairs_dir=%s, bootstrap_enabled=%s)",
            pairs_dir,
            bootstrap_enabled,
        )

    evaluator = ImputationEvaluator(
        scenarios=scenario_names,
        num_eval_workers=config.data.num_eval_workers,
        n_days=config.data.n_days,
        compute_metrics=config.evaluation.compute_metrics,
        save_pairs=save_pairs,
        pairs_dir=pairs_dir,
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
        fallback_fill=fallback_fill,
    )

    _summarize_fallback(results)

    if bootstrap_enabled:
        _run_bootstrap(
            results=results,
            pairs_dir=pairs_dir,
            results_dir=Path(config.output.results_dir),
            bootstrap_cfg=bootstrap_cfg,
        )

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


# Suffixes appended to each scalar metric when bootstrap is enabled.
_BOOTSTRAP_FIELDS = ("bootstrap_mean", "bootstrap_se", "ci_lo", "ci_hi", "n_valid_boot")


def _merge_bootstrap_entry(target: dict, metric_name: str, entry: dict) -> None:
    """Merge bootstrap entry ``{point, bootstrap_mean, ...}`` as sibling fields.

    ``target`` is e.g. ``results["scenarios"][s]["test"]["per_channel"]["ch_0"]``;
    ``entry`` is the dict returned by ``bootstrap_split``'s ``_entry``. Adds
    ``{metric}_bootstrap_mean``, ``{metric}_bootstrap_se``, ``{metric}_ci_lo``,
    ``{metric}_ci_hi``, ``{metric}_n_valid_boot`` next to the existing scalar.
    """
    for field in _BOOTSTRAP_FIELDS:
        if field in entry:
            target[f"{metric_name}_{field}"] = entry[field]


def _merge_split_bootstrap(split_results: dict, split_bootstrap: dict) -> None:
    """Merge ``bootstrap_split`` output into a per-split results dict (sibling fields).

    Both inputs share the same per_channel / continuous / binary layout.
    """
    boot_per_ch = split_bootstrap.get("per_channel", {})
    for ch_key, ch_entry in split_results.get("per_channel", {}).items():
        boot_ch = boot_per_ch.get(ch_key)
        if not isinstance(boot_ch, dict):
            continue
        for metric_name, value in boot_ch.items():
            if isinstance(value, dict) and "point" in value:
                _merge_bootstrap_entry(ch_entry, metric_name, value)

    for group_name in ("continuous", "binary"):
        boot_group = split_bootstrap.get(group_name, {})
        target_group = split_results.get(group_name, {})
        for metric_name, value in boot_group.items():
            if isinstance(value, dict) and "point" in value:
                _merge_bootstrap_entry(target_group, metric_name, value)


def _run_bootstrap(
    *,
    results: dict,
    pairs_dir: Path,
    results_dir: Path,
    bootstrap_cfg,
) -> None:
    """Run participant-level bootstrap and merge CIs into ``results``.

    Side effects:
        - Mutates ``results["scenarios"][s][split]`` to add ``*_ci_lo``,
          ``*_ci_hi``, ``*_bootstrap_se``, etc. fields.
        - Writes the structured per-channel form to ``bootstrap_metrics.json``
          (path from ``bootstrap_cfg.output_path``, default ``<results_dir>``).
    """
    from imputation_evaluation.evaluation.bootstrap import bootstrap_pairs_dir

    logger.info(
        "Running participant-level bootstrap (n_boot=%d, ci_level=%g, seed=%d)",
        bootstrap_cfg.n_boot,
        bootstrap_cfg.ci_level,
        bootstrap_cfg.seed,
    )
    scenario_names = list(results.get("scenarios", {}).keys())
    boot = bootstrap_pairs_dir(
        pairs_dir,
        scenarios=scenario_names or None,
        n_boot=bootstrap_cfg.n_boot,
        ci_level=bootstrap_cfg.ci_level,
        seed=bootstrap_cfg.seed,
        include_auc=bootstrap_cfg.include_auc,
    )

    for scenario, split_map in boot.get("scenarios", {}).items():
        scenario_results = results.get("scenarios", {}).get(scenario)
        if not isinstance(scenario_results, dict):
            continue
        for split, split_boot in split_map.items():
            split_results = scenario_results.get(split)
            if isinstance(split_results, dict) and isinstance(split_boot, dict):
                _merge_split_bootstrap(split_results, split_boot)

    out_path = (
        Path(bootstrap_cfg.output_path)
        if bootstrap_cfg.output_path
        else results_dir / "bootstrap_metrics.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(boot, indent=2, default=_bootstrap_json_default))
    logger.info("Wrote bootstrap metrics to %s", out_path)
    results["bootstrap"] = boot


def _bootstrap_json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
