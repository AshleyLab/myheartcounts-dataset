"""Dev driver: run a TSFM encoder (Toto / Chronos-2) through the single engine flow.

``run_eval(Model())`` extracts the per-(split, task) last-latent embeddings from raw
on a cache miss (GPU), channel-mean-pools them, and runs the uniform PCA-50 + probe —
all inside one flow. Temporary jobs/ scaffolding for cluster validation.

    METHOD=toto      python jobs/imperial/slurm/run_tsfm_eval.py
    METHOD=chronos2  python jobs/imperial/slurm/run_tsfm_eval.py
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


def main() -> None:
    import csv

    from openmhc._evaluate import _DatasetPaths, _ensure_labels_env

    from downstream_evaluation.data.splits import load_split_file
    from downstream_evaluation.runner import EvalConfig, run_eval

    method = os.environ.get("METHOD", "toto")
    if method == "toto":
        from downstream_evaluation.models.toto import Toto as Model
    elif method == "chronos2":
        from downstream_evaluation.models.chronos2 import Chronos2 as Model
    else:
        raise SystemExit(f"unknown METHOD={method!r} (expected toto|chronos2)")

    paths = _DatasetPaths.resolve(os.environ.get("MHC_DATA_DIR"))
    _ensure_labels_env(paths.labels_dir)
    split_users = load_split_file(paths.splits_file)

    cfg = EvalConfig(
        data_dir=str(paths.root), split_users=split_users, tasks=HEADLINE_TASKS, seed=42
    )
    results = run_eval(cfg, Model(data_dir=str(paths.root)))

    out = os.environ.get("OUT_CSV", f"eval_{method}.csv")
    rows = {t: r for t, r in results.items() if t != "config"}
    fields = ["task"] + sorted(
        {("n_test" if k == "n_test" else f"test_{k}") for r in rows.values() for k in r}
    )
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for t in sorted(rows):
            row = {"task": t}
            for k, v in rows[t].items():
                row["n_test" if k == "n_test" else f"test_{k}"] = v
            w.writerow(row)
    print(f"wrote {len(rows)} task rows -> {out}")


if __name__ == "__main__":
    main()
