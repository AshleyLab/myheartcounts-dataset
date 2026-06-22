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
        from downstream_evaluation.models.wbm import WBMProbe
        # WBM_CHECKPOINT overrides the default wandb SSL ref with a local .ckpt path
        # (e.g. the wandb-cached copy) so the run needs no network on the compute node.
        ckpt = os.environ.get("WBM_CHECKPOINT")
        return WBMProbe(data_dir, checkpoint=ckpt) if ckpt else WBMProbe(data_dir)
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
    if method == "lsm2":
        from downstream_evaluation.models.lsm2 import LSM2
        # LSM2_CHECKPOINT overrides the default registry ref with a local .ckpt path
        # (or an alternate wandb ref) — e.g. when the registry is access-restricted.
        ckpt = os.environ.get("LSM2_CHECKPOINT")
        return LSM2(data_dir=data_dir, checkpoint=ckpt) if ckpt else LSM2(data_dir=data_dir)
    if method == "xgboost":
        from downstream_evaluation.models.xgboost import XGBoost
        return XGBoost(data_dir=data_dir)
    if method == "gru_d":
        from downstream_evaluation.models.grud import GRUD
        return GRUD(data_dir=data_dir, tasks=BENCHMARK_TASKS)
    raise SystemExit(f"unknown METHOD={method!r}")


def main() -> None:
    """Run ``METHOD`` through ``evaluate_prediction``; write ``OUT_CSV`` (+ optional preds).

    Env vars: ``METHOD`` (default ``linear``), ``VERSION`` (``xs``|``full``, default
    ``full``), ``MHC_DATA_DIR``, ``PREDICTIONS_DIR`` (optional), ``OUTPUT_DIR``
    (optional — write the leaderboard substrate ``<OUTPUT_DIR>/<METHOD>.parquet``),
    ``OUT_CSV`` (default ``eval_<METHOD>.csv``).
    """
    import openmhc
    from openmhc._evaluate import _DatasetPaths

    method = os.environ.get("METHOD", "linear")
    version = os.environ.get("VERSION", "full")
    paths = _DatasetPaths.resolve(os.environ.get("MHC_DATA_DIR"), version=version)
    model = build_model(method, str(paths.root))

    # PREDICTIONS_DIR (optional): emit per-(method, task) test predictions +
    # _subgroups.json for the paper-metrics bootstrap (skill / rank / fairness CIs).
    # OUTPUT_DIR (optional): emit the per-method leaderboard substrate
    # <OUTPUT_DIR>/<METHOD>.parquet (raw per-user pairs) + meta sidecar for HF upload.
    output_dir = os.environ.get("OUTPUT_DIR")
    results = openmhc.evaluate_prediction(
        model,
        version=version,
        tasks=BENCHMARK_TASKS,
        data_dir=str(paths.root),
        predictions_dir=os.environ.get("PREDICTIONS_DIR"),
        output_dir=output_dir,
        method_name=method if output_dir else None,
    )

    out = os.environ.get("OUT_CSV", f"eval_{method}.csv")
    results.to_csv(out)
    print(f"wrote {out}: {len(results.records)} records")
    if output_dir:
        print(
            f"wrote {output_dir}/{method}.parquet "
            f"(overall_fallback_rate={results.overall_fallback_rate:.4f})"
        )


if __name__ == "__main__":
    main()
