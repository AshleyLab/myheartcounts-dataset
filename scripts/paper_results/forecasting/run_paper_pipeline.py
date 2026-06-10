#!/usr/bin/env python
r"""End-to-end driver for the forecasting (Track 3) paper-metrics pipeline.

Mirrors ``scripts/paper_results/run_paper_pipeline.py`` (imputation) for
forecasting. Stages (each skippable):

1. **Phase 0 (eval)** — for each PRE-SPECIFIED model in the sweep config, run
   ``mhc-forecast-eval`` under a shared run label so the point + binary metrics
   co-locate in ``<name>_metrics/<LABEL>/`` (see ``config.MetricsConfig``).
2. **Phase 1 (discover + validate)** — select exactly the configured models from
   the metrics tree; **error** if any expected model's metrics are missing;
   ignore any extra model dirs that happen to be present.
3. **Phase 2 (skill + rank)** — ``skill_score_summary`` + ``grouped_metric_rank_summary``
   (continuous=mae, binary=auprc, vs ``baseline``) → ``output_root``.
4. **Phase 3 (bootstrap CIs + fairness)** — paired user-level bootstrap CIs for
   skill score + mean rank; when ``bootstrap.fairness`` is set, also the
   disparity-ratio fairness skill score (deterministic point CSV + bootstrap-CI
   CSV). Gated on ``bootstrap.enabled``.

Usage::

    python scripts/paper_results/forecasting/run_paper_pipeline.py \
        --sweep-config configs/paper/sweep_forecasting.yaml

    # re-aggregate only (metrics already produced under the run label, e.g. by SLURM):
    python scripts/paper_results/forecasting/run_paper_pipeline.py \
        --sweep-config configs/paper/sweep_forecasting.yaml --skip-eval
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# scripts/paper_results/forecasting/run_paper_pipeline.py -> parents[3] = repo root.
REPO_ROOT = Path(__file__).resolve().parents[3]


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Run the forecasting paper pipeline end-to-end.")
    p.add_argument("--sweep-config", type=Path, required=True, help="Path to sweep_forecasting.yaml")
    p.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip Phase 0 (metrics already exist under the run label, e.g. produced by SLURM)",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Restrict to a subset of the configured models (still validated against the tree)",
    )
    p.add_argument(
        "--run-label",
        default=None,
        help="Override run_label from the sweep (output_root derived as <runs_root>/summary/<label>)",
    )
    p.add_argument("--output-root", default=None, help="Override output_root from the sweep")
    p.add_argument("--dry-run", action="store_true", help="Print commands; do not run them")
    return p.parse_args()


def _run(cmd: list[str], dry_run: bool) -> None:
    """Run a subprocess; exit non-zero on failure."""
    pretty = " ".join(str(c) for c in cmd)
    logger.info("$ %s", pretty)
    if dry_run:
        return
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        logger.error("Command failed (exit=%d): %s", res.returncode, pretty)
        sys.exit(res.returncode)


def _metrics_dir(runs_root: str, name: str, label: str) -> Path:
    """Canonical per-model metrics dir for a run label."""
    return Path(runs_root) / name / "predictions" / label / f"{name}_metrics" / label


def _phase0_eval(cfg: dict, models: list[dict], dry_run: bool) -> None:
    """Phase 0 — run mhc-forecast-eval for each model under the shared run label."""
    runs_root = cfg["runs_root"]
    label = cfg["run_label"]
    common = cfg.get("common_overrides", []) or []
    for m in models:
        name = m["name"]
        model_type = m.get("type", name)
        out = Path(runs_root) / name / "predictions"
        cmd = [
            "mhc-forecast-eval",
            f"model={model_type}",
            f"model.name={name}",
            f"experiment_name={label}",
            f"output.results_dir={out}",
            f"hydra.run.dir={Path(runs_root) / name / 'hydra'}",
            "hydra.job.chdir=false",
        ]
        if m.get("release_dir"):
            cmd.append(f"model.release_dir={m['release_dir']}")
        cmd += list(common) + list(m.get("overrides", []) or [])
        _run(cmd, dry_run)


def _discover_and_validate(cfg: dict, models: list[dict]) -> dict[str, str]:
    """Phase 1 — select exactly the configured models; error if any are missing.

    Returns ``{name: metrics_dir}`` for the expected models. Extra model dirs
    present under the run label are ignored. Raises if any expected model lacks
    its metrics (``mae``, and ``auprc`` when binary metrics are requested).
    """
    runs_root = cfg["runs_root"]
    label = cfg["run_label"]
    need_binary = bool(cfg.get("binary_metrics"))
    selected: dict[str, str] = {}
    missing: list[str] = []
    for m in models:
        name = m["name"]
        md = _metrics_dir(runs_root, name, label)
        ok = (md / "mae").is_dir() and ((md / "auprc").is_dir() if need_binary else True)
        if ok:
            selected[name] = str(md)
        else:
            missing.append(name)
    if missing:
        raise SystemExit(
            f"Expected models missing metrics under run_label '{label}': {sorted(missing)}\n"
            f"  looked for <runs_root>/<model>/predictions/{label}/<model>_metrics/{label}/"
            f"{{mae{',auprc' if need_binary else ''}}}\n"
            f"  (runs_root={runs_root}) — run Phase 0 (drop --skip-eval) or fix the sweep config."
        )
    logger.info("Discovered + validated %d expected models: %s", len(selected), ", ".join(sorted(selected)))
    return selected


def _phase_skill_rank(cfg: dict, selected: dict[str, str], dry_run: bool) -> None:
    """Phase 2 — skill score + grouped mean-rank from the validated metrics."""
    out = Path(cfg["output_root"])
    out.mkdir(parents=True, exist_ok=True)
    models_json = out / "skill_rank_models.json"
    logger.info("Writing model map (%d models) to %s", len(selected), models_json)
    if not dry_run:
        models_json.write_text(json.dumps({"models": selected}, indent=2))

    cont = cfg.get("continuous_metrics", ["mae"])
    binm = cfg.get("binary_metrics", ["auprc"])
    agg = cfg.get("aggregation_unit", "user")
    metrics = REPO_ROOT / "src" / "forecasting_evaluation" / "metrics"

    _run(
        [
            sys.executable, str(metrics / "skill_score_summary.py"),
            "--config", str(models_json),
            "--baseline", cfg["baseline"],
            "--continuous-metrics", *cont,
            "--binary-metrics", *binm,
            "--aggregation-unit", agg,
            "--output-dir", str(out),
            "--output-prefix", "forecasting_skill_score",
        ],
        dry_run,
    )
    _run(
        [
            sys.executable, str(metrics / "grouped_metric_rank_summary.py"),
            "--config", str(models_json),
            "--continuous-metrics", *cont,
            "--binary-metrics", *binm,
            "--output-dir", str(out),
            "--output-prefix", "forecasting_grouped_metric_rank",
        ],
        dry_run,
    )


def _phase_bootstrap(cfg: dict, selected: dict[str, str], dry_run: bool) -> None:
    """Phase 3 — paired user-level bootstrap CIs for skill score + mean rank.

    When ``bootstrap.fairness`` is set, also writes the disparity-ratio fairness
    skill score as both a deterministic point CSV and a bootstrap-CI CSV. Writes
    alongside the Phase-2 point summaries.
    """
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from forecasting_evaluation.metrics import metric_spec as _spec
    from forecasting_evaluation.metrics.bootstrap_skill_rank import bootstrap_skill_rank

    bs = cfg.get("bootstrap") or {}
    out = Path(cfg["output_root"])
    out.mkdir(parents=True, exist_ok=True)
    models = {name: {"path": path, "display_name": name} for name, path in selected.items()}

    n_boot = int(bs.get("n_boot", 1000))
    seed = int(bs.get("seed", 42))
    ci_level = float(bs.get("ci_level", 0.95))
    logger.info(
        "Phase 3 bootstrap: B=%d seed=%d ci=%.2f over %d models (baseline=%s)",
        n_boot, seed, ci_level, len(models), cfg["baseline"],
    )
    if dry_run:
        return

    tables = bootstrap_skill_rank(
        models=models,
        baseline_model=cfg["baseline"],
        continuous_metrics=cfg.get("continuous_metrics", ["mae"]),
        binary_metrics=cfg.get("binary_metrics", ["auprc"]),
        continuous_channel_indices=_spec.CONTINUOUS_CHANNELS,
        binary_channel_indices=_spec.BINARY_CHANNELS,
        binary_groups=[(name, tuple(idx)) for name, idx in _spec.BINARY_GROUPS],
        n_boot=n_boot,
        seed=seed,
        ci_level=ci_level,
    )
    skill_path = out / "forecasting_skill_score_bootstrap.csv"
    rank_path = out / "forecasting_grouped_metric_rank_bootstrap.csv"
    tables["skill_scores"].to_csv(skill_path, index=False)
    tables["avg_rankings"].to_csv(rank_path, index=False)
    logger.info("Wrote bootstrap CIs: %s , %s", skill_path, rank_path)

    if not bs.get("fairness"):
        return

    from forecasting_evaluation.metrics.bootstrap_fair_skill_score import (
        bootstrap_fair_skill_score,
    )

    # Demographics default to <dataset_root>/labels/ (resolved by the labels API
    # via MHC_DATA_DIR); the sweep config may override either path explicitly.
    from labels.api import ENROLLMENT_PATH as _ENROLL_DEFAULT
    from labels.api import LABELS_PATH as _LABELS_DEFAULT

    labels_path = bs.get("labels_path") or (str(_LABELS_DEFAULT) if _LABELS_DEFAULT else None)
    enrollment_path = bs.get("enrollment_path") or (
        str(_ENROLL_DEFAULT) if _ENROLL_DEFAULT else None
    )
    if labels_path is None or enrollment_path is None:
        raise SystemExit(
            "Fairness bootstrap needs demographics: set bootstrap.labels_path / "
            "bootstrap.enrollment_path in the sweep, or MHC_DATA_DIR so "
            "<dataset_root>/labels/{last_labels,enrollment_info}.json resolves."
        )

    logger.info("Phase 3 fairness: disparity-ratio fair skill score (point + bootstrap)")
    ftables = bootstrap_fair_skill_score(
        models=models,
        baseline_model=cfg["baseline"],
        continuous_metrics=cfg.get("continuous_metrics", ["mae"]),
        binary_metrics=cfg.get("binary_metrics", ["auprc"]),
        continuous_channel_indices=_spec.CONTINUOUS_CHANNELS,
        binary_channel_indices=_spec.BINARY_CHANNELS,
        labels_path=labels_path,
        enrollment_path=enrollment_path,
        age_bins=tuple(bs.get("age_bins", [18, 30, 40, 50, 60])),
        n_boot=n_boot,
        seed=seed,
        ci_level=ci_level,
    )
    fair_point_path = out / "forecasting_fairness_skill_score.csv"
    fair_boot_path = out / "forecasting_fairness_skill_score_bootstrap.csv"
    ftables["fairness_skill_scores_point"].to_csv(fair_point_path, index=False)
    ftables["fairness_skill_scores"].to_csv(fair_boot_path, index=False)
    logger.info(
        "Wrote fairness skill score (point + bootstrap): %s , %s", fair_point_path, fair_boot_path
    )


def main() -> int:
    """CLI entry point — see module docstring for usage."""
    args = _parse_args()
    cfg = yaml.safe_load(args.sweep_config.read_text())

    if args.run_label:
        cfg["run_label"] = args.run_label
        cfg["output_root"] = args.output_root or str(
            Path(cfg["runs_root"]) / "summary" / args.run_label
        )
    elif args.output_root:
        cfg["output_root"] = args.output_root

    models = cfg["models"]
    if args.models:
        wanted = set(args.models)
        models = [m for m in models if m["name"] in wanted]
        unknown = wanted - {m["name"] for m in cfg["models"]}
        if unknown:
            raise SystemExit(f"--models names not in the sweep config: {sorted(unknown)}")
        if not models:
            raise SystemExit("--models left no entries")

    if not args.skip_eval:
        _phase0_eval(cfg, models, args.dry_run)

    selected = _discover_and_validate(cfg, models)
    _phase_skill_rank(cfg, selected, args.dry_run)

    if (cfg.get("bootstrap") or {}).get("enabled", False):
        _phase_bootstrap(cfg, selected, args.dry_run)

    logger.info("Done. Summary CSVs in %s", cfg["output_root"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
