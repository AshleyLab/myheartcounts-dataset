"""Generic from-raw downstream-eval driver — the real entrypoint.

Picks a bundled model by ``METHOD`` and runs it through the public API
``openmhc.evaluate_prediction(model)`` — the *same* call an external submitter
makes with their own encoder (which wraps the one ``run_eval`` engine). No bespoke
per-method orchestration; this replaces the old ``run_*_eval.py`` drivers.

    METHOD=toto MHC_DATA_DIR=/path sbatch jobs/imperial/slurm/run_eval.slurm
"""

import os

HEADLINE_TASKS = [
    "Atrial fibrillation (Afib)", "BMI_categories", "BMI_values", "BiologicalSex", "CAD",
    "Cerebrovascular Disease", "Congenital Heart", "Diabetes", "GoSleepTime_categories", "Hdl",
    "Heart Failure or CHF", "Hypertension", "Ldl", "PH", "Peripheral/Systemic Vascular Disease",
    "SystolicBloodPressure", "TotalCholesterol", "WakeUpTime_categories", "WeightKilograms", "age",
    "blood_pressure_categories", "cardiovascular_disease", "feel_worthwhile1", "feel_worthwhile2",
    "feel_worthwhile3", "feel_worthwhile4", "framingham_risk", "satisfiedwith_life",
    "sleep_diagnosis1", "sleep_time_categories", "vigorous_act", "work",
]


def build_model(method: str, data_dir: str):
    """Instantiate a bundled baseline by name."""
    if method in ("wbm", "hybrid"):
        from downstream_evaluation.models.hybrid_wbm import Hybrid
        return Hybrid(data_dir)
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
        return MultiRocket(data_dir=data_dir, tasks=HEADLINE_TASKS)
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
        return GRUD(data_dir=data_dir, tasks=HEADLINE_TASKS)
    raise SystemExit(f"unknown METHOD={method!r}")


def main() -> None:
    import openmhc
    from openmhc._evaluate import _DatasetPaths

    method = os.environ.get("METHOD", "linear")
    paths = _DatasetPaths.resolve(os.environ.get("MHC_DATA_DIR"))
    model = build_model(method, str(paths.root))

    results = openmhc.evaluate_prediction(
        model, tasks=HEADLINE_TASKS, data_dir=str(paths.root)
    )

    out = os.environ.get("OUT_CSV", f"eval_{method}.csv")
    results.to_csv(out)
    print(f"wrote {out}: {len(results.records)} records, global_score={results.global_score:.4f}")


if __name__ == "__main__":
    main()
