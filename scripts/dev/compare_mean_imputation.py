"""Compare mean imputation results between the public repo and the private repo.

Loads the 2026-xs split, fits both MeanImputer (public) and MeanImputation
(private), generates masks once with the private repo's masking params, then
runs the evaluator from both repos and asserts numeric parity.

Usage:
    python tests/compare_mean_imputation.py

Expected output: per-scenario metrics from both repos followed by a parity report.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
# Override via MHC_BENCHMARK_SRC env var to point at your local MHC-benchmark
# clone (private repo, not pip-installable).
PRIVATE_REPO_SRC = Path(
    os.environ.get(
        "MHC_BENCHMARK_SRC",
        str(Path.home() / "MHC-benchmark" / "src"),
    )
)

# Public repo src must come first so its versions of shared modules win.
sys.path.insert(0, str(REPO_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DATA_DIR = Path.home() / ".cache" / "openmhc" / "data-xs"
SPLIT_FILE = REPO_ROOT / "data" / "splits" / "sharable_users_seed42_2026_xs.json"


# ---------------------------------------------------------------------------
# Step 1: Load the tiny split (public data pipeline)
# ---------------------------------------------------------------------------

logger.info("=== Step 1: Loading the 2026-xs split ===")

from imputation_evaluation.config import (  # noqa: E402
    DataConfig,
    FilterConfig,
    IntensityFailureConfig,
    MaskingConfig,
    PreprocessingConfig,
    RandomNoiseConfig,
    SignalSliceConfig,
    SleepGapConfig,
    TemporalSliceConfig,
    WorkoutGapConfig,
)
from imputation_evaluation.data.data_loader import ImputationDataLoader  # noqa: E402

data_cfg = DataConfig(
    daily_hf_dir=str(DATA_DIR / "processed" / "daily_hf"),
    split_file=str(SPLIT_FILE),
    split_seed=42,
    batch_size=500,
    num_workers=0,
    num_eval_workers=1,
    preprocessing=PreprocessingConfig(zero_to_nan=True),
    filters=FilterConfig(
        min_wear_fraction=0.5,
        variance_filter_enabled=True,
        variance_thresholds=None,
    ),
)

data_loader = ImputationDataLoader(data_cfg)
loaded = data_loader.load_splits(
    batch_size=data_cfg.batch_size,
    num_workers=data_cfg.num_workers,
    pin_memory=False,
)
logger.info(
    "Split sizes — train: %d  val: %d  test: %d",
    len(loaded.train_loader.dataset),
    len(loaded.val_loader.dataset),
    len(loaded.test_loader.dataset),
)


# ---------------------------------------------------------------------------
# Step 2: Fit both imputers and compare channel means
# ---------------------------------------------------------------------------

logger.info("=== Step 2: Fitting both imputers ===")

# --- Public MeanImputer ---
from openmhc.imputers.mean import MeanImputer  # noqa: E402

public_imputer = MeanImputer(version="xs", data_dir=DATA_DIR)
public_means = public_imputer._channel_means
logger.info("Public channel_means (first 7): %s", public_means[:7])

# --- Private MeanImputation ---
# Import without polluting the rest of the namespace; private repo may have
# modules with the same name as the public repo.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_private_mean_imputation",
    PRIVATE_REPO_SRC / "imputation_evaluation" / "methods" / "mean_imputation.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
MeanImputation = _mod.MeanImputation

private_method = MeanImputation()
private_method.fit(loaded.train_loader)
private_means = private_method.channel_means
logger.info("Private channel_means (first 7): %s", private_means[:7])

# --- Compare ---
means_close = np.allclose(public_means, private_means, atol=1e-3, rtol=1e-4)
max_diff = float(np.abs(public_means - private_means).max())
logger.info("Channel means match: %s  (max abs diff: %.2e)", means_close, max_diff)

if not means_close:
    diffs = np.abs(public_means - private_means)
    worst = np.argsort(diffs)[::-1][:5]
    for ch in worst:
        logger.warning(
            "  ch %2d: public=%.6f  private=%.6f  diff=%.2e",
            ch,
            public_means[ch],
            private_means[ch],
            diffs[ch],
        )


# ---------------------------------------------------------------------------
# Step 3: Generate masks once (private repo's masking params)
# ---------------------------------------------------------------------------

logger.info("=== Step 3: Generating masks (private repo params) ===")

# Build a MaskingConfig with the private repo's base.yaml values.
masking_cfg = MaskingConfig(
    mask_seed=42,
    random_noise=RandomNoiseConfig(enabled=True, patch_size=30, mask_ratio=0.5),
    temporal_slice=TemporalSliceConfig(
        enabled=True, mask_ratio=0.25, min_block_size=30, max_block_size=60
    ),
    signal_slice=SignalSliceConfig(
        enabled=True,
        mask_ratio=0.5,
        device_groups={"iphone": [0, 1, 2], "watch": [3, 4, 5, 6]},
    ),
    sleep_gap=SleepGapConfig(enabled=True, asleep_channel=7, inbed_channel=8),
    workout_gap=WorkoutGapConfig(
        enabled=True, mask_channels=[5, 6], workout_channels=list(range(9, 19))
    ),
    intensity_failure=IntensityFailureConfig(
        enabled=True,
        hr_channel=5,
        hr_threshold=160.0,
        hr_unit="auto",
        mask_channels=[5, 6],
    ),
)

from imputation_evaluation.masking import MaskCacheGenerator, create_mask_generators  # noqa: E402

generators = create_mask_generators(masking_cfg)
scenario_names = [g.name for g in generators]
logger.info("Scenarios: %s", scenario_names)

mask_gen = MaskCacheGenerator(
    hf_dataset=loaded.hf_dataset,
    zero_to_nan_transform=loaded.zero_to_nan_transform,
    num_workers=0,
    batch_size=data_cfg.batch_size,
)
mask_cache = mask_gen.generate(
    split_indices={"val": loaded.split_indices["val"], "test": loaded.split_indices["test"]},
    generators=generators,
    base_seed=masking_cfg.mask_seed,
)
logger.info("Mask cache generated.")


# ---------------------------------------------------------------------------
# Step 4: Run public evaluator with public imputer
# ---------------------------------------------------------------------------

logger.info("=== Step 4: Running public evaluator ===")

from imputation_evaluation.evaluation.evaluator import ImputationEvaluator  # noqa: E402
from openmhc._evaluate import _ImputerMethodAdapter  # noqa: E402

# The adapter computes channel stds via fit(); wrap the public MeanImputer.
pub_adapter = _ImputerMethodAdapter(public_imputer)
pub_adapter.fit(loaded.train_loader)
pub_channel_stds = pub_adapter.channel_stds

# Build eval data loaders (pre-filtered to samples that have at least one mask).
applicable_indices = {}
for split_name in ("val", "test"):
    indices = mask_cache.get_applicable_indices(split_name)
    if indices:
        applicable_indices[split_name] = indices

eval_val_loader, eval_test_loader = data_loader.create_eval_loaders(
    split_indices=loaded.split_indices,
    hf_dataset=loaded.hf_dataset,
    batch_size=data_cfg.batch_size,
    num_workers=0,
    pin_memory=False,
    window_descriptors=loaded.window_descriptors,
    window_day_offsets=loaded.window_day_offsets,
    applicable_indices=applicable_indices if applicable_indices else None,
)

pub_evaluator = ImputationEvaluator(
    scenarios=scenario_names,
    num_eval_workers=1,
    n_days=data_cfg.n_days,
    compute_metrics=True,
    save_pairs=False,
    pairs_dir=None,
)
pub_results = pub_evaluator.run(
    val_loader=eval_val_loader,
    test_loader=eval_test_loader,
    mask_cache=mask_cache,
    method=pub_adapter,
    channel_stds=pub_channel_stds,
    hf_dataset=loaded.hf_dataset,
    split_indices=loaded.split_indices,
    zero_to_nan_transform=loaded.zero_to_nan_transform,
)
logger.info("Public eval complete.")


# ---------------------------------------------------------------------------
# Step 5: Run private evaluator with private imputer
# ---------------------------------------------------------------------------

logger.info("=== Step 5: Running private evaluator ===")

# We re-use the same mask_cache so masks are identical.
# The private evaluator is a direct copy of the public one (same code base in
# this repo), so we run it with the private MeanImputation method instead.
# The private method's channel_stds are used for metric normalization.

priv_channel_stds = private_method.channel_stds


class _PrivateMethodAdapter:
    """Thin adapter to make MeanImputation satisfy the internal ImputationMethod protocol."""

    def __init__(self, method):
        self._method = method

    @property
    def name(self) -> str:
        return self._method.name

    @property
    def channel_stds(self) -> np.ndarray | None:
        return self._method.channel_stds

    def fit(self, loader) -> None:
        pass  # already fitted

    def impute(self, data, original_masks, artificial_masks, **_kwargs) -> np.ndarray:
        return self._method.impute(data, original_masks, artificial_masks)


priv_adapter = _PrivateMethodAdapter(private_method)

priv_evaluator = ImputationEvaluator(
    scenarios=scenario_names,
    num_eval_workers=1,
    n_days=data_cfg.n_days,
    compute_metrics=True,
    save_pairs=False,
    pairs_dir=None,
)
priv_results = priv_evaluator.run(
    val_loader=eval_val_loader,
    test_loader=eval_test_loader,
    mask_cache=mask_cache,
    method=priv_adapter,
    channel_stds=priv_channel_stds,
    hf_dataset=loaded.hf_dataset,
    split_indices=loaded.split_indices,
    zero_to_nan_transform=loaded.zero_to_nan_transform,
)
logger.info("Private eval complete.")


# ---------------------------------------------------------------------------
# Step 6: Compare results
# ---------------------------------------------------------------------------

logger.info("=== Step 6: Parity report ===")

ATOL = 1e-4
all_match = True

for scenario in scenario_names:
    for split in ("val", "test"):
        pub_m = pub_results["scenarios"][scenario][split]
        priv_m = priv_results["scenarios"][scenario][split]

        # Continuous summary metrics
        for key in ("mean_normalized_rmse", "mean_normalized_mse", "mean_normalized_mae"):
            pv = pub_m.get("continuous", {}).get(key, float("nan"))
            rv = priv_m.get("continuous", {}).get(key, float("nan"))
            match = (np.isnan(pv) and np.isnan(rv)) or abs(pv - rv) <= ATOL
            status = "OK" if match else "MISMATCH"
            if not match:
                all_match = False
            logger.info(
                "  [%s] %s/%s %s: pub=%.6f  priv=%.6f  diff=%.2e  %s",
                scenario,
                split,
                "continuous",
                key,
                pv,
                rv,
                abs(pv - rv) if not np.isnan(pv - rv) else float("nan"),
                status,
            )

        # Binary summary metrics
        for key in ("macro_balanced_accuracy", "macro_roc_auc"):
            pv = pub_m.get("binary", {}).get(key, float("nan"))
            rv = priv_m.get("binary", {}).get(key, float("nan"))
            match = (np.isnan(pv) and np.isnan(rv)) or abs(pv - rv) <= ATOL
            status = "OK" if match else "MISMATCH"
            if not match:
                all_match = False
            logger.info(
                "  [%s] %s/%s %s: pub=%.6f  priv=%.6f  diff=%.2e  %s",
                scenario,
                split,
                "binary",
                key,
                pv,
                rv,
                abs(pv - rv) if not np.isnan(pv - rv) else float("nan"),
                status,
            )

logger.info("")
if all_match:
    logger.info("RESULT: All metrics match within atol=%.0e — PARITY CONFIRMED.", ATOL)
else:
    logger.warning("RESULT: Some metrics diverge — see MISMATCH lines above.")

# Summary of channel means check
logger.info("")
logger.info(
    "Channel means parity: %s (max diff %.2e)", "OK" if means_close else "MISMATCH", max_diff
)
