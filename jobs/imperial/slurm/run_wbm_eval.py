"""Dev driver: run the WBM downstream eval through the single engine flow.

`run_eval(WBM())` extracts the embeddings from raw on a cache miss (GPU), saves
them, and reuses them on a hit — all inside one flow — then runs the uniform
PCA-50 + probe. Temporary jobs/ scaffolding for cluster validation.
"""

import os

# 32 headline cross-sectional tasks (from final_eval.yaml).
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
    from downstream_evaluation.models.hybrid import Hybrid
    from downstream_evaluation.runner import EvalConfig, run_eval

    paths = _DatasetPaths.resolve(os.environ.get("MHC_DATA_DIR"))
    _ensure_labels_env(paths.labels_dir)
    split_users = load_split_file(paths.splits_file)

    cfg = EvalConfig(
        data_dir=str(paths.root), split_users=split_users, tasks=HEADLINE_TASKS, seed=42
    )
    # Headline WBM = the hybrid. On a cache miss the WBM (SSL) branch extracts its
    # embeddings from raw (GPU) inside run_eval; the Linear fallback builds from raw.
    results = run_eval(cfg, Hybrid(str(paths.root)))

    out = os.environ.get("OUT_CSV", "eval_hybrid.csv")
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
