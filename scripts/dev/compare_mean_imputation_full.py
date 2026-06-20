"""Compare public MeanImputer against the canonical private max91d mean run.

Runs evaluate_imputation(MeanImputer(), version="full") against the full
11,894-user split with pre-computed max91d masks, then compares the streaming
MetricAccumulator results against post-hoc aggregated_metrics.json from the
private repo's baselines_max91d_21679652 run.

Tolerance is atol=1e-3 (vs 1e-4 for the XS test) because the two sides use
different accumulation paths: streaming sums (public) vs post-hoc
aggregate_pairs() from float32 Parquet (private).

Usage:
    python tests/compare_mean_imputation_full.py

Expected output: per-scenario metrics from the public side, private ground
truth values, and a parity report.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Full 11,894-user dataset extracted from the openmhc-full Dataverse bundle
# into the public cache. Layout matches _DatasetPaths expectations:
#   processed/daily_hf/  — sensor data
#   splits/sharable_users_seed42_2026.json — full split
#   labels/ — label registry
#   dataset_version.json — version marker (required by the strict resolver)
DATA_DIR = Path.home() / ".cache" / "openmhc" / "data-full"

# Override via MHC_BENCHMARK_MEAN_RUN env var to point at your local canonical
# private mean run (the script is dev-only; the reference run lives in a
# private repo).
PRIVATE_RUN_DIR = Path(os.environ.get(
    "MHC_BENCHMARK_MEAN_RUN",
    str(
        Path.home()
        / "MHC-benchmark/results/imputation_eval"
        / "baselines_max91d_21679652_imputation_mean_20260416_044624"
    ),
))
PRIVATE_METRICS_FILE = PRIVATE_RUN_DIR / "pairs" / "aggregated_metrics.json"

ATOL = 1e-3

# ---------------------------------------------------------------------------
# Step 1: Run public evaluator
# ---------------------------------------------------------------------------

logger.info("=== Step 1: Running public evaluate_imputation (version='full') ===")

import openmhc
from openmhc.imputers.mean import MeanImputer

imputer = MeanImputer(version="full", data_dir=DATA_DIR)
pub_results = openmhc.evaluate_imputation(imputer, version="full", data_dir=DATA_DIR)

logger.info("Public eval complete.")

# ---------------------------------------------------------------------------
# Step 2: Load private ground truth
# ---------------------------------------------------------------------------

logger.info("=== Step 2: Loading private ground truth ===")

if not PRIVATE_METRICS_FILE.exists():
    logger.error("Private metrics file not found: %s", PRIVATE_METRICS_FILE)
    sys.exit(1)

priv_metrics = json.load(open(PRIVATE_METRICS_FILE))["scenarios"]
logger.info("Loaded private metrics for scenarios: %s", list(priv_metrics.keys()))

# ---------------------------------------------------------------------------
# Step 3: Parity report
# ---------------------------------------------------------------------------

logger.info("=== Step 3: Parity report (atol=%.0e) ===", ATOL)

CONT_METRICS = ("mean_normalized_rmse", "mean_normalized_mse", "mean_normalized_mae")
BIN_METRICS  = ("macro_balanced_accuracy", "macro_roc_auc")

scenario_names = list(priv_metrics.keys())
all_match = True

for scenario in scenario_names:
    for split in ("val", "test"):
        pub_split = pub_results.scenarios.get(scenario, {}).get(split, {})
        priv_split = priv_metrics.get(scenario, {}).get(split, {})

        pub_cont  = pub_split.get("continuous", {})
        priv_cont = priv_split.get("continuous", {})
        pub_bin   = pub_split.get("binary", {})
        priv_bin  = priv_split.get("binary", {})

        for key in CONT_METRICS:
            pv = pub_cont.get(key, float("nan"))
            rv = priv_cont.get(key, float("nan"))
            match = (np.isnan(pv) and np.isnan(rv)) or abs(pv - rv) <= ATOL
            status = "OK" if match else "MISMATCH"
            if not match:
                all_match = False
            logger.info(
                "  [%s] %s/%s/continuous %s: pub=%.6f  priv=%.6f  diff=%.2e  %s",
                scenario, split, "continuous", key,
                pv, rv,
                abs(pv - rv) if not (np.isnan(pv) or np.isnan(rv)) else float("nan"),
                status,
            )

        for key in BIN_METRICS:
            pv = pub_bin.get(key, float("nan"))
            rv = priv_bin.get(key, float("nan"))
            match = (np.isnan(pv) and np.isnan(rv)) or abs(pv - rv) <= ATOL
            status = "OK" if match else "MISMATCH"
            if not match:
                all_match = False
            logger.info(
                "  [%s] %s/%s/binary %s: pub=%.6f  priv=%.6f  diff=%.2e  %s",
                scenario, split, "binary", key,
                pv, rv,
                abs(pv - rv) if not (np.isnan(pv) or np.isnan(rv)) else float("nan"),
                status,
            )

logger.info("")
if all_match:
    logger.info("RESULT: All metrics match within atol=%.0e — PARITY CONFIRMED.", ATOL)
else:
    logger.warning("RESULT: Some metrics diverge — see MISMATCH lines above.")
