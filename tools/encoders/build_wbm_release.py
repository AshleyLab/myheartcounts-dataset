#!/usr/bin/env python
"""Stage the Track-1 WBM encoder release bundle for Hugging Face publishing.

Produces, under ``--staging-dir`` (default ``releases-enc/``), a self-contained
bundle for the reported **WBM** model, with an ``openmhc_manifest.json``
(loadable by ``openmhc.encoders.WBM.from_release``) plus the checkpoint payload
and a model card::

    releases-enc/openmhc-wbm-dp/
    ├── model.ckpt                 # Mamba2 contrastive encoder (from the W&B artifact)
    ├── normalization_stats.json   # canonical hourly z-score constants (from the dataset)
    ├── openmhc_manifest.json
    └── README.md

The checkpoint is pulled from the W&B artifact via ``resolve_checkpoint_path``;
the normalization stats are copied from the dataset's
``normalization_stats_hourly.json`` (or ``--norm-stats``). Publish with::

    python tools/encoders/build_wbm_release.py --data-dir /path/to/openmhc-data
    python tools/publish_to_hf.py --staging-dir releases-enc --only openmhc-wbm-dp --tag v1.0
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from openmhc.encoders import write_manifest

DEFAULT_ARTIFACT = "wandb:MHC_Dataset/mhc-apple-contrastive-transformer/WBM_Final_HPO_best:v1"
BUNDLE_NAME = "openmhc-wbm-dp"


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _wbm_arch() -> dict:
    """The encoder architecture echoed into the manifest (source of truth = model.py)."""
    from downstream_evaluation.models.wbm.model import _ARCH

    return dict(_ARCH)


def _resolve_norm_stats(data_dir: str | None, override: Path | None) -> Path:
    """Locate the canonical hourly normalization stats to ship in the bundle."""
    if override is not None:
        stats = override.expanduser()
        if not stats.exists():
            raise FileNotFoundError(f"--norm-stats file not found: {stats}")
        return stats
    from openmhc._evaluate import _DatasetPaths

    paths = _DatasetPaths.from_root(data_dir)
    stats = paths.daily_hourly_hf.parent / "normalization_stats_hourly.json"
    if not stats.exists():
        raise FileNotFoundError(
            f"normalization_stats_hourly.json not found at {stats}. Pass --norm-stats "
            "to point at the canonical hourly stats file."
        )
    return stats


def stage_wbm(
    artifact: str,
    data_dir: str | None,
    staging: Path,
    norm_stats: Path | None = None,
    release_tag: str = "v1.0",
) -> Path:
    """Stage the WBM bundle: encoder .ckpt + normalization stats + manifest + card."""
    from utils.checkpoints import resolve_checkpoint_path

    ckpt = resolve_checkpoint_path(artifact)
    stats = _resolve_norm_stats(data_dir, norm_stats)

    bundle = staging / BUNDLE_NAME
    _reset_dir(bundle)
    shutil.copy2(ckpt, bundle / "model.ckpt")
    shutil.copy2(stats, bundle / "normalization_stats.json")
    write_manifest(
        bundle,
        kind="wbm",
        checkpoint="model.ckpt",
        arch=_wbm_arch(),
        normalization_stats="normalization_stats.json",
        provenance={
            "model": "wbm",
            "trained_on": "MHC training split (Apple-contrastive SSL pretraining)",
            "source_artifact": artifact,
            "wandb_entity": "MHC_Dataset",
            "wandb_project": "mhc-apple-contrastive-transformer",
        },
    )
    _write_readme(bundle, release_tag)
    return bundle


def _write_readme(bundle: Path, release_tag: str = "v1.0") -> None:
    repo = f"MyHeartCounts/{BUNDLE_NAME}"
    (bundle / "README.md").write_text(
        f"""---
license: cc-by-4.0
library_name: openmhc
tags:
- representation-learning
- wearables
- openmhc
---

# OpenMHC Outcome Prediction — WBM

Track 1 (outcome prediction) reference checkpoint for the **MyHeartCounts /
OpenMHC** wearable-health benchmark (NeurIPS 2026).

**This checkpoint is the WBM encoder** — a bi-directional Mamba2 contrastive
self-supervised model that maps a week of wearable sensor data (168 hourly steps,
19 channels) to a 256-d representation. The reported **WBM** model pairs this
encoder (per-user pooled → PCA-50 → linear probe) with a Linear fallback for
users without a weekly embedding.

**Pretrained** with a contrastive objective on the MHC training split.

- **Checkpoint format:** PyTorch Lightning checkpoint (`model.ckpt`) +
  `normalization_stats.json` (canonical hourly z-score constants; channels 0–6
  normalized, 7–18 identity).
- **Outcome-prediction tasks:** 33 health & behavior labels (classification,
  ordinal, regression).

## Requirements

Running the encoder needs the CUDA-only Mamba2 kernels (`mamba-ssm`) and a GPU.

## Usage

```python
import openmhc
from openmhc.encoders import WBM

# pip install "openmhc[hf]"  (+ mamba-ssm on a CUDA machine)
enc = WBM.from_release("hf://{repo}@{release_tag}")
results = openmhc.evaluate_prediction(enc, version="full")
```

See `openmhc_manifest.json` for provenance (source W&B artifact, training
details) and architecture metadata.

## Citation

If you use this checkpoint, please cite the OpenMHC benchmark.
""",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    """Stage the WBM bundle under ``--staging-dir``."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--staging-dir", type=Path, default=Path("releases-enc"))
    p.add_argument(
        "--artifact",
        default=DEFAULT_ARTIFACT,
        help=f"Checkpoint reference (wandb:/hf:///local). Default: {DEFAULT_ARTIFACT}",
    )
    p.add_argument("--data-dir", default=None, help="dataset root (else MHC_DATA_DIR)")
    p.add_argument(
        "--norm-stats",
        type=Path,
        default=None,
        help="Override path to normalization_stats_hourly.json (else resolved from --data-dir).",
    )
    p.add_argument(
        "--release-tag",
        default="v1.0",
        help="Version tag referenced in the model card's usage example (hf://...@<tag>). "
        "Match this to publish_to_hf.py --tag. Default: v1.0.",
    )
    args = p.parse_args(argv)

    args.staging_dir.mkdir(parents=True, exist_ok=True)
    bundle = stage_wbm(
        artifact=args.artifact,
        data_dir=args.data_dir,
        staging=args.staging_dir,
        norm_stats=args.norm_stats,
        release_tag=args.release_tag,
    )
    files = sorted(x.name for x in bundle.iterdir())
    print(f"Staged bundle under {args.staging_dir}:")
    print(f"  {bundle.name}: {files}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
