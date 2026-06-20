"""Weights & Biases logging for imputation evaluation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from imputation_evaluation.config import ImputationEvalConfig

logger = logging.getLogger(__name__)


def _config_to_wandb_dict(config: Any) -> dict[str, Any]:
    """Convert a dataclass or DictConfig snapshot into a wandb-friendly dict.

    Uses the shared shim from :mod:`eval_hydra` when available (i.e. when the
    ``[hydra]`` extra is installed) and falls back to ``dataclasses.asdict``
    so this module still works in the non-Hydra path.
    """
    try:
        from eval_hydra.wandb_shim import to_wandb_config

        return to_wandb_config(config)
    except ImportError:
        from dataclasses import asdict

        return asdict(config)


def init_wandb(config: ImputationEvalConfig, run_name: str | None = None) -> None:
    """Initialize a wandb run with the given config.

    Args:
        config: Full imputation evaluation config.
        run_name: Optional pre-resolved run name.
    """
    import wandb

    from imputation_evaluation.io.writer import resolve_experiment_name

    if run_name is None:
        run_name = resolve_experiment_name(config.output, config)

    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=run_name,
        tags=config.wandb.tags,
        config=_config_to_wandb_dict(config),
    )
    logger.info(f"Initialized wandb run: {wandb.run.name}")


def log_results(results: dict) -> None:
    """Log evaluation results as flat wandb metrics.

    Flattens the nested results dict into keys like:
        {scenario}/{split}/continuous/mean_nRMSE
        {scenario}/{split}/binary/macro_bal_acc
        {scenario}/{split}/channel/ch_{i}/nRMSE

    Args:
        results: Results dictionary from ImputationEvaluator.run().
    """
    import wandb

    flat_metrics: dict[str, float | int] = {}

    for scenario_name, scenario_results in results.get("scenarios", {}).items():
        for split_name, metrics in scenario_results.items():
            if "error" in metrics:
                continue

            prefix = f"{scenario_name}/{split_name}"

            # n_samples
            if "n_samples" in metrics:
                flat_metrics[f"{prefix}/n_samples"] = metrics["n_samples"]
            if "n_total" in metrics:
                flat_metrics[f"{prefix}/n_total"] = metrics["n_total"]

            # Fallback substitution visibility (model-capability gap).
            if "overall_fallback_rate" in metrics:
                flat_metrics[f"{prefix}/overall_fallback_rate"] = metrics[
                    "overall_fallback_rate"
                ]

            # Continuous aggregate metrics
            cont = metrics.get("continuous", {})
            if "mean_normalized_rmse" in cont:
                flat_metrics[f"{prefix}/continuous/mean_nRMSE"] = cont["mean_normalized_rmse"]
            if "mean_normalized_mse" in cont:
                flat_metrics[f"{prefix}/continuous/mean_nMSE"] = cont["mean_normalized_mse"]
            if "mean_normalized_mae" in cont:
                flat_metrics[f"{prefix}/continuous/mean_nMAE"] = cont["mean_normalized_mae"]

            # Binary aggregate metrics
            binary = metrics.get("binary", {})
            if "macro_balanced_accuracy" in binary:
                flat_metrics[f"{prefix}/binary/macro_bal_acc"] = binary["macro_balanced_accuracy"]
            if "macro_roc_auc" in binary:
                flat_metrics[f"{prefix}/binary/macro_roc_auc"] = binary["macro_roc_auc"]

            # Per-channel metrics
            for ch_key, ch_metrics in metrics.get("per_channel", {}).items():
                ch_prefix = f"{prefix}/channel/{ch_key}"
                if "normalized_rmse" in ch_metrics:
                    flat_metrics[f"{ch_prefix}/nRMSE"] = ch_metrics["normalized_rmse"]
                if "normalized_mse" in ch_metrics:
                    flat_metrics[f"{ch_prefix}/nMSE"] = ch_metrics["normalized_mse"]
                if "normalized_mae" in ch_metrics:
                    flat_metrics[f"{ch_prefix}/nMAE"] = ch_metrics["normalized_mae"]
                if "balanced_accuracy" in ch_metrics:
                    flat_metrics[f"{ch_prefix}/bal_acc"] = ch_metrics["balanced_accuracy"]

            # Per-channel fallback rates (model-capability gap, per channel).
            for ch_key, rate in metrics.get("fallback_rate", {}).items():
                flat_metrics[f"{prefix}/channel/{ch_key}/fallback_rate"] = rate

            # Subgroup metrics (sensitivity analysis)
            subgroups = metrics.get("subgroups")
            if subgroups:
                for attr, groups in subgroups.items():
                    for group_name, sg_metrics in groups.items():
                        sg_prefix = f"{prefix}/subgroups/{attr}/{group_name}"
                        if "n_samples" in sg_metrics:
                            flat_metrics[f"{sg_prefix}/n_samples"] = sg_metrics["n_samples"]
                        sg_cont = sg_metrics.get("continuous", {})
                        if "mean_normalized_rmse" in sg_cont:
                            flat_metrics[f"{sg_prefix}/mean_nRMSE"] = sg_cont[
                                "mean_normalized_rmse"
                            ]
                        if "mean_normalized_mse" in sg_cont:
                            flat_metrics[f"{sg_prefix}/mean_nMSE"] = sg_cont["mean_normalized_mse"]
                        if "mean_normalized_mae" in sg_cont:
                            flat_metrics[f"{sg_prefix}/mean_nMAE"] = sg_cont["mean_normalized_mae"]
                        sg_bin = sg_metrics.get("binary", {})
                        if "macro_balanced_accuracy" in sg_bin:
                            flat_metrics[f"{sg_prefix}/macro_bal_acc"] = sg_bin[
                                "macro_balanced_accuracy"
                            ]

    # Filter out NaN values (wandb doesn't handle them well)
    import math

    flat_metrics = {
        k: v for k, v in flat_metrics.items() if not (isinstance(v, float) and math.isnan(v))
    }

    wandb.log(flat_metrics)
    logger.info(f"Logged {len(flat_metrics)} metrics to wandb")

    # Create summary table
    _log_summary_table(results)


def _log_summary_table(results: dict) -> None:
    """Log a summary table of scenario x split results.

    Args:
        results: Results dictionary from ImputationEvaluator.run().
    """
    import wandb

    columns = ["scenario", "split", "n_samples", "mean_nRMSE", "macro_roc_auc"]
    table = wandb.Table(columns=columns)

    for scenario_name, scenario_results in results.get("scenarios", {}).items():
        for split_name, metrics in scenario_results.items():
            if "error" in metrics:
                continue
            n_samples = metrics.get("n_samples", 0)
            cont = metrics.get("continuous", {})
            binary = metrics.get("binary", {})
            nrmse = cont.get("mean_normalized_rmse")
            roc_auc = binary.get("macro_roc_auc")
            table.add_data(scenario_name, split_name, n_samples, nrmse, roc_auc)

    wandb.log({"summary": table})


def log_plots(output_dir: Path) -> None:
    """Log visualization plots as wandb images.

    Args:
        output_dir: Output directory containing plots/ subdirectory.
    """
    import wandb

    plots_dir = output_dir / "plots"
    if not plots_dir.exists():
        logger.warning(f"Plots directory not found: {plots_dir}")
        return

    images: dict[str, list[wandb.Image]] = {}
    for png_path in sorted(plots_dir.rglob("*.png")):
        # Group by scenario (parent directory name)
        scenario = png_path.parent.name if png_path.parent != plots_dir else "plots"
        key = f"plots/{scenario}"
        if key not in images:
            images[key] = []
        images[key].append(wandb.Image(str(png_path), caption=png_path.stem))

    for key, image_list in images.items():
        wandb.log({key: image_list})

    total = sum(len(v) for v in images.values())
    logger.info(f"Logged {total} plots to wandb across {len(images)} scenarios")


def finish() -> None:
    """Finish the wandb run."""
    import wandb

    wandb.finish()
    logger.info("Finished wandb run")
