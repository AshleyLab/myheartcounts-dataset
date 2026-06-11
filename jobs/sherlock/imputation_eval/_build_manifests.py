#!/usr/bin/env python3
"""Generate openmhc_manifest.json + copy normalization_stats.json into each release dir.

The W&B artifacts that 00_setup.sh downloads are bare training dumps (no
manifest). This script materializes the openmhc release-bundle layout the
imputer wrappers expect:

    <release_dir>/
    ├── <checkpoint>.{pypots,ckpt}
    ├── normalization_stats.json   <- copied from the openmhc data cache
    └── openmhc_manifest.json      <- written by this script

Arch params reflect the exact training-time config recorded under
MHC-benchmark/configs/imputation_eval/methods/*.yaml; they MUST match what
PyPOTS used to fit each model or load_model raises size mismatches.

Run from 00_setup.sh; idempotent.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# Import openmhc's write_manifest so the schema stays in sync.
from openmhc.imputers._release import write_manifest

# Allow overrides via env vars so anyone with a Sherlock-style layout can run
# this without editing the file. Defaults match the layout produced by
# ``jobs/sherlock/_env.sh`` + ``00_setup.sh``.
_SCRATCH = Path(os.environ.get("SCRATCH_RUN_ROOT", f"/scratch/users/{os.environ.get('USER', 'unknown')}"))
RELEASES = Path(os.environ.get("RELEASES", str(_SCRATCH / "releases")))
_MHC_CACHE = Path(os.environ.get("MHC_CACHE", str(_SCRATCH / ".myheartcounts-dataset-cache/data-full")))
STATS_SRC = Path(
    os.environ.get(
        "MHC_NORMALIZATION_STATS",
        str(_MHC_CACHE / "processed" / "normalization_stats.json"),
    )
)

# Per-model arch params extracted from
# MHC-benchmark/configs/imputation_eval/methods/pypots_*.yaml. Defaults that
# match openmhc's PyPOTSMethodConfig are omitted unless training used a
# non-default value.
SPECS = {
    "brits": {
        "kind": "brits",
        "checkpoint": "BRITS.pypots",
        "arch": {"n_steps": 1440, "n_features": 19, "rnn_hidden_size": 128},
        "provenance": {"wandb_artifact": "MHC_Dataset/mhc-pypots-brits/brits:v19"},
    },
    "dlinear": {
        "kind": "dlinear",
        "checkpoint": "DLinear.pypots",
        # MHC-benchmark trained with d_model=256, moving_avg_window_size=51
        "arch": {
            "n_steps": 1440, "n_features": 19,
            "d_model": 256, "moving_avg_window_size": 51,
        },
        "provenance": {"wandb_artifact": "MHC_Dataset/mhc-pypots-dlinear/dlinear:v49"},
    },
    "fedformer": {
        "kind": "fedformer",
        "checkpoint": "FEDformer.pypots",
        "arch": {
            "n_steps": 1440, "n_features": 19,
            "n_layers": 2, "d_model": 512, "n_heads": 8, "d_ffn": 128,
            "moving_avg_window_size": 25, "dropout": 0.1,
            "variant": "Fourier", "modes": 32, "mode_select": "random",
        },
        "provenance": {"wandb_artifact": "MHC_Dataset/mhc-pypots-fedformer/fedformer:v31"},
    },
    "timesnet": {
        "kind": "timesnet",
        "checkpoint": "TimesNet.pypots",
        "arch": {
            "n_steps": 1440, "n_features": 19,
            "n_layers": 2, "top_k": 5,
            "d_model": 128, "d_ffn": 512, "n_kernels": 6,
            "dropout": 0.4, "apply_nonstationary_norm": False,
        },
        "provenance": {"wandb_artifact": "MHC_Dataset/mhc-pypots-timesnet/timesnet:v31"},
    },
    "lsm2": {
        "kind": "lsm2",
        # Picked dynamically below — there's exactly one *.ckpt.
        "checkpoint": None,
        # Arch is embedded in the Lightning checkpoint; LSM2Imputer.__init__
        # accepts **_extra so an empty arch dict is fine.
        "arch": {},
        "provenance": {"wandb_artifact": "MHC_Dataset/mhc-mae-ssl-daily/mae-daily:v0"},
    },
    "lsm2_weekly_sparse": {
        "kind": "lsm2_weekly_sparse",
        "checkpoint": None,
        "arch": {},
        "provenance": {
            "wandb_artifact": "MHC_Dataset/mhc-mae-ssl/mae-weekly-sparse-d4:v0",
        },
    },
}


def _pick_ckpt(release_dir: Path, declared: str | None) -> str:
    """Pick the checkpoint filename inside release_dir. Prefer ``declared``
    if present; else for LSM2-style bundles, pick the lone *.ckpt."""
    if declared and (release_dir / declared).exists():
        return declared
    ckpts = sorted(p.name for p in release_dir.glob("*.ckpt"))
    if len(ckpts) == 1:
        return ckpts[0]
    pypots = sorted(
        p.name for p in release_dir.glob("*.pypots")
        if "_epoch" not in p.name and p.is_file()
    )
    if pypots:
        return pypots[0]
    raise FileNotFoundError(
        f"No suitable checkpoint in {release_dir} "
        f"(declared={declared!r}; .ckpt={ckpts})"
    )


def main() -> int:
    if not STATS_SRC.exists():
        print(f"[FATAL] normalization stats source missing: {STATS_SRC}", file=sys.stderr)
        return 1
    if not RELEASES.exists():
        print(f"[FATAL] releases root missing: {RELEASES}", file=sys.stderr)
        return 1

    for name, spec in SPECS.items():
        release_dir = RELEASES / name
        if not release_dir.is_dir():
            print(f"[skip] {name}: no release dir at {release_dir}")
            continue

        ckpt = _pick_ckpt(release_dir, spec["checkpoint"])

        # Copy normalization_stats.json into the release dir (idempotent).
        stats_dst = release_dir / "normalization_stats.json"
        if not stats_dst.exists():
            shutil.copy2(STATS_SRC, stats_dst)
            print(f"[copy] {STATS_SRC.name} -> {stats_dst}")
        else:
            print(f"[skip] stats already present: {stats_dst}")

        manifest = write_manifest(
            release_dir,
            kind=spec["kind"],
            checkpoint=ckpt,
            normalization_stats="normalization_stats.json",
            arch=spec["arch"],
            provenance=spec["provenance"],
        )
        print(f"[ok]  wrote {manifest}  (kind={spec['kind']}, ckpt={ckpt})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
