"""Precompute the canonical XS imputation masks and save them into the repo.

Mirrors the precomputed ``full`` masks shipped at
``data/imputation/masks/sharable_users_seed42_2026_max91d/``. Generating XS masks
on the fly costs ~20 min over the full test split on every
``evaluate_imputation(version="xs")`` call; shipping them lets the public API
load them instead (the same mechanism ``full`` uses).

The masks are valid for the canonical config the public API uses for XS:
``mask_seed=42``, ``n_days=1``, all six scenarios, the XS user split. The eval
path guards on those before loading (see ``openmhc/_evaluate.py``); any other
config falls back to on-the-fly generation.

Run once (the dataset root must hold the XS bundle):

    MHC_DATA_DIR=~/.cache/openmhc/data-xs python scripts/precompute_xs_masks.py
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

from imputation_evaluation.config import DataConfig, MaskingConfig  # noqa: E402
from imputation_evaluation.data.data_loader import ImputationDataLoader  # noqa: E402
from imputation_evaluation.masking import create_mask_generators  # noqa: E402
from imputation_evaluation.masking.generator import MaskCacheGenerator  # noqa: E402
from openmhc._evaluate import _DatasetPaths  # noqa: E402

SEED = 42
SPLITS = ("val", "test")
OUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "imputation"
    / "masks"
    / "sharable_users_seed42_2026_xs"
)


def main() -> None:
    """Generate and save the XS val/test masks for all six scenarios."""
    paths = _DatasetPaths.resolve(None, version="xs")  # honors MHC_DATA_DIR / data_dir
    data_cfg = DataConfig(
        daily_hf_dir=str(paths.daily_hf),
        split_file=str(paths.splits_file),
        version="xs",
        split_seed=SEED,
        batch_size=5000,
        num_workers=4,
        n_days=1,
    )
    loaded = ImputationDataLoader(data_cfg).load_splits(
        batch_size=5000, num_workers=4, pin_memory=False
    )

    masking_cfg = MaskingConfig(mask_seed=SEED)  # all six scenarios enabled by default
    generators = create_mask_generators(masking_cfg)
    logging.info("Scenarios: %s", [g.name for g in generators])

    generator = MaskCacheGenerator(
        hf_dataset=loaded.hf_dataset,
        zero_to_nan_transform=loaded.zero_to_nan_transform,
        num_workers=4,
        batch_size=5000,
    )
    t0 = time.time()
    cache = generator.generate(
        split_indices={s: loaded.split_indices[s] for s in SPLITS},
        generators=generators,
        base_seed=SEED,
    )
    logging.info("Generated masks in %.1fs", time.time() - t0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache.save(OUT_DIR)
    logging.info("Saved XS masks to %s", OUT_DIR)


if __name__ == "__main__":
    main()
