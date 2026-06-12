"""Generic from-raw downstream-eval driver — the real entrypoint.

Picks a bundled model by ``METHOD`` and runs it through the public API
``openmhc.evaluate_prediction(model)`` — the *same* call an external submitter
makes with their own model. No bespoke per-method orchestration; this replaces
the old ``run_*_eval.py`` drivers.

    METHOD=toto MHC_DATA_DIR=/path sbatch jobs/imperial/slurm/run_eval.slurm
"""

import os

from openmhc._constants import BENCHMARK_TASKS


def build_model(method: str, data_dir: str):
    """Instantiate a bundled baseline by name."""
    if method in ("wbm", "hybrid"):
        from downstream_evaluation.models.hybrid_wbm import Hybrid
        # WBM_CHECKPOINT overrides the default wandb SSL ref with a local .ckpt path
        # (e.g. the wandb-cached copy) so the run needs no network on the compute node.
        ckpt = os.environ.get("WBM_CHECKPOINT")
        return Hybrid(data_dir, checkpoint=ckpt) if ckpt else Hybrid(data_dir)
    if method == "linear":
        from downstream_evaluation.models.linear import Linear
        return Linear(data_dir=data_dir)
    if method == "toto":
        from downstream_evaluation.models.toto import Toto
        return Toto(data_dir=data_dir)
    if method == "chronos2":
        from downstream_evaluation.models.chronos2 import Chronos2
        return Chronos2(data_dir=data_dir)
    if method == "multirocket":
        from downstream_evaluation.models.multirocket import MultiRocket
        return MultiRocket(data_dir=data_dir, tasks=BENCHMARK_TASKS)
    if method == "mae":
        from downstream_evaluation.models.mae import MAE
        # MAE_CHECKPOINT overrides the default registry ref with a local .ckpt path
        # (or an alternate wandb ref) — e.g. when the registry is access-restricted.
        ckpt = os.environ.get("MAE_CHECKPOINT")
        return MAE(data_dir=data_dir, checkpoint=ckpt) if ckpt else MAE(data_dir=data_dir)
    if method == "xgboost":
        from downstream_evaluation.models.xgboost import XGBoost
        return XGBoost(data_dir=data_dir)
    if method == "gru_d":
        from downstream_evaluation.models.grud import GRUD
        return GRUD(data_dir=data_dir, tasks=BENCHMARK_TASKS)
    raise SystemExit(f"unknown METHOD={method!r}")


def main() -> None:
    import openmhc
    from openmhc._evaluate import _DatasetPaths

    method = os.environ.get("METHOD", "linear")
    paths = _DatasetPaths.resolve(os.environ.get("MHC_DATA_DIR"))
    model = build_model(method, str(paths.root))

    # PREDICTIONS_DIR (optional): emit per-(method, task) test predictions +
    # _subgroups.json for the paper-metrics bootstrap (skill / rank / fairness CIs).
    results = openmhc.evaluate_prediction(
        model,
        tasks=BENCHMARK_TASKS,
        data_dir=str(paths.root),
        predictions_dir=os.environ.get("PREDICTIONS_DIR"),
    )

    out = os.environ.get("OUT_CSV", f"eval_{method}.csv")
    results.to_csv(out)
    print(f"wrote {out}: {len(results.records)} records")


if __name__ == "__main__":
    main()
