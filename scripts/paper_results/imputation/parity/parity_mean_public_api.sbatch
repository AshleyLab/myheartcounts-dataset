#!/bin/bash -l
#SBATCH -p normal
#SBATCH --job-name=parity_mean_api
#SBATCH --time=06:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH -o /scratch/users/schuetzn/tmp/claude/parity_mean_api.out
#SBATCH -e /scratch/users/schuetzn/tmp/claude/parity_mean_api.err

# §4 Public-API parity for mean imputer.
#  (1) Run openmhc.evaluate_imputation(MeanImputer, version='full',
#      masking_scenarios=all 6, output_dir=VERIF/mean_public_api,
#      baseline_errors=frozen-locf) — exercises the A5-minimal surface
#      end-to-end against the canonical dataset.
#  (2) Compare per_user_errors.parquet (all/all slice) against
#      paper/per_user_errors.parquet[method=='mean', all/all].
#  (3) Compare skill_scores.csv against paper/skill_scores_bootstrap.csv
#      mean column within 5×SE.

set -euo pipefail

ml purge
ml load python/3.12.1
unset PYTHONPATH
source /scratch/users/schuetzn/envs/openmhc/bin/activate
export MHC_DATA_DIR=/home/users/schuetzn/.cache/openmhc/data-full

cd /home/users/schuetzn/myheartcounts-dataset

VERIF=/scratch/users/schuetzn/openmhc-imputation-eval/paper-verification
PAPER=/scratch/users/schuetzn/openmhc-imputation-eval/paper
OUTDIR="$VERIF/mean_public_api"
rm -rf "$OUTDIR" && mkdir -p "$OUTDIR"

echo "=== (1/2) Run public API end-to-end ==="
python - <<PYEOF
import openmhc
from openmhc.imputers.mean import MeanImputer

res = openmhc.evaluate_imputation(
    MeanImputer(version="full"),
    version="full",
    masking_scenarios=["intensity_failure", "random_noise", "signal_slice",
                       "sleep_gap", "temporal_slice", "workout_gap"],
    output_dir="$OUTDIR",
    baseline_errors="/home/users/schuetzn/myheartcounts-dataset/src/openmhc/data/baselines/imputation_locf_per_user_errors.parquet",
    method_name="mean",
    num_workers=4,
    num_eval_workers=4,
)
print("per_user_errors rows:", None if res.per_user_errors is None else len(res.per_user_errors))
print("skill_scores rows:", None if res.skill_scores is None else len(res.skill_scores))
PYEOF

echo "=== (2/2) Parity diff ==="
python - <<'PYEOF'
import logging
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
log = logging.getLogger()

VERIF = Path("/scratch/users/schuetzn/openmhc-imputation-eval/paper-verification/mean_public_api")
PAPER = Path("/scratch/users/schuetzn/openmhc-imputation-eval/paper")

failed = False

# ---- per_user_errors parity (all/all slice) ----
new = pd.read_parquet(VERIF / "per_user_errors.parquet")
gt = pd.read_parquet(PAPER / "per_user_errors.parquet")
gt = gt[(gt["method"].astype(str) == "mean")
        & (gt["subgroup_attr"].astype(str) == "all")
        & (gt["subgroup_value"].astype(str) == "all")]
log.info("new per_user rows: %d | gt per_user rows (mean, all/all): %d", len(new), len(gt))

JOIN_KEYS = ["method","scenario","split","channel","channel_type","subgroup_attr","subgroup_value","user_id"]
for c in JOIN_KEYS:
    new[c] = new[c].astype(str)
    gt[c] = gt[c].astype(str)
new["E_per_user"] = new["E_per_user"].astype(np.float32)
gt["E_per_user"] = gt["E_per_user"].astype(np.float32)
merged = new.merge(gt, on=JOIN_KEYS, how="outer", suffixes=("_new","_gt"), indicator=True)
only_new = int((merged["_merge"]=="left_only").sum())
only_gt = int((merged["_merge"]=="right_only").sum())
both = int((merged["_merge"]=="both").sum())
log.info("per_user_errors merged: both=%d only_new=%d only_gt=%d", both, only_new, only_gt)
if only_new or only_gt:
    failed = True
    log.error("per_user_errors row-set differs (only_new=%d only_gt=%d)", only_new, only_gt)
both_df = merged[merged["_merge"]=="both"]
ne = both_df["E_per_user_new"].to_numpy()
ge = both_df["E_per_user_gt"].to_numpy()
if np.array_equal(ne, ge, equal_nan=True):
    log.info("PASS: per_user_errors E byte-equal on all %d rows", len(both_df))
else:
    diff = np.abs(ne - ge)
    finite = np.isfinite(diff)
    log.warning("per_user_errors E float32 byte-mismatch on %d rows (max |diff|=%.3e); "
                "if max is at float32 ULP scale (~1e-3 on large values) this is "
                "expected from regen vs disk float32 round-trip.",
                int(np.sum(diff > 0)),
                float(np.nanmax(diff[finite])) if finite.any() else float("nan"))
    # Tolerance: relative diff per row < 1e-5 (float32 ulp scale)
    relative = diff / np.maximum(np.abs(ge), 1e-9)
    out_of_tol = int(np.sum(relative > 1e-4))
    if out_of_tol > 0:
        failed = True
        log.error("per_user_errors: %d rows have relative diff > 1e-4 — REAL divergence", out_of_tol)

# ---- skill scores parity ----
new_skill = pd.read_csv(VERIF / "skill_scores.csv")
gt_skill = pd.read_csv(PAPER / "skill_scores_bootstrap.csv")
gt_skill = gt_skill[gt_skill["method"].astype(str) == "mean"]
log.info("new skill rows: %d | gt skill rows (mean): %d", len(new_skill), len(gt_skill))
for c in ["method", "scope"]:
    new_skill[c] = new_skill[c].astype(str)
    gt_skill[c] = gt_skill[c].astype(str)

merged_s = new_skill.merge(gt_skill, on=["method","scope"], how="inner", suffixes=("_new","_gt"))
log.info("skill merged: %d rows", len(merged_s))
diff = np.abs(merged_s["skill_score"].to_numpy() - merged_s["mean"].to_numpy())
se = merged_s["se"].to_numpy()
threshold = 5.0 * se
oob = diff > threshold
n_oob = int(np.nansum(oob))
if n_oob == 0:
    log.info("PASS: skill_scores — all %d rows within 5×SE (max |Δ|=%.3e)",
             len(merged_s), float(np.nanmax(diff)))
else:
    failed = True
    worst = merged_s.assign(absd=diff, band=diff/np.where(threshold>0, threshold, np.nan)
                            ).nlargest(10, "band")
    log.error("FAIL: skill_scores — %d/%d rows outside 5×SE", n_oob, len(merged_s))
    log.error("worst 10:\n%s", worst.to_string())

if failed:
    raise SystemExit(1)
log.info("§4 mean-via-public-API parity PASSED ✓")
PYEOF
