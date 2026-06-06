#!/usr/bin/env python
"""Stage the 5 forecasting release bundles for Hugging Face publishing.

Produces, under ``--staging-dir`` (default ``releases-fc/``), one self-contained
bundle per model, each with an ``openmhc_manifest.json`` (loadable by both the
public ``openmhc.forecasters`` API and the evaluation harness) plus the
checkpoint payload and a model card::

    releases-fc/openmhc-chronos2-fc/   checkpoint/{config.json,model.safetensors} + manifest   (merged from LoRA)
    releases-fc/openmhc-toto-fc/       model.ckpt + manifest                                    (full finetune)
    releases-fc/openmhc-dlinear-fc/    OnlineDLinear.pypots + training_config.json + standard_scaler_stats.json + manifest
    releases-fc/openmhc-segrnn-fc/     OnlineSegRNN.pypots  + training_config.json + standard_scaler_stats.json + manifest
    releases-fc/openmhc-mixlinear-fc/  OnlineMixLinear.pypots + training_config.json + standard_scaler_stats.json + manifest

Neural bundles co-locate ``training_config.json`` (architecture source of truth)
and ``standard_scaler_stats.json`` (train-split scaler for inverse-transform).
Chronos-2 / Toto normalize internally, so they carry no normalization stats.

Run from the public repo root with the base env (needs ``openmhc.forecasters``)::

    python tools/forecasting/build_forecasting_release.py \
        --ckpt-root forecasting_model_ckpt \
        --scaler .merge_cache/regen_scaler_stats.json \
        --chronos-merged .merge_cache/chronos2_FT_merged \
        --toto-ckpt ~/MHC-benchmark/models/foundational/toto/toto-full-20260419-0220/toto-epoch=24-step=116225-val_loss=-1.3597.ckpt
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from openmhc.forecasters import write_manifest

# Neural models: (bundle kind, source subdir under --ckpt-root, .pypots filename).
NEURAL = {
    "dlinear": ("dlinear_HPO_scale", "OnlineDLinear.pypots"),
    "segrnn": ("segrnn_HPO_scale", "OnlineSegRNN.pypots"),
    "mixlinear": ("mixlinear_HPO_scale", "OnlineMixLinear.pypots"),
}

PAPER_TABLE = "tab:forecasting_grouped_model_summary"


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _neural_arch(training_config: dict) -> dict:
    """Minimal arch echoed into the manifest (training_config.json is authoritative)."""
    model = training_config.get("model", {})
    return {
        k: model[k]
        for k in ("n_steps", "n_pred_steps", "n_features")
        if model.get(k) is not None
    }


def stage_neural(kind: str, ckpt_root: Path, scaler: Path, staging: Path) -> Path:
    """Stage a neural (.pypots) bundle: checkpoint + training_config + scaler + manifest."""
    src_dir_name, pypots_name = NEURAL[kind]
    src = ckpt_root / src_dir_name
    pypots = src / pypots_name
    train_cfg = src / "training_config.json"
    for f in (pypots, train_cfg):
        if not f.exists():
            raise FileNotFoundError(f"Missing neural source file: {f}")

    bundle = staging / f"openmhc-{kind}-fc"
    _reset_dir(bundle)
    shutil.copy2(pypots, bundle / pypots_name)
    shutil.copy2(train_cfg, bundle / "training_config.json")
    shutil.copy2(scaler, bundle / "standard_scaler_stats.json")

    training_config = json.loads(train_cfg.read_text())
    write_manifest(
        bundle,
        kind=kind,
        checkpoint=".",
        arch=_neural_arch(training_config),
        normalization_stats="standard_scaler_stats.json",
        provenance={
            "model": kind,
            "trained_on": "MHC training split (from-scratch)",
            "source_checkpoint": f"forecasting_model_ckpt/{src_dir_name}/{pypots_name}",
            "wandb_entity": "MHC_Dataset",
            "wandb_project": "mhc-forecasting",
            "paper_table": PAPER_TABLE,
        },
    )
    _write_readme(bundle, kind)
    return bundle


def stage_chronos(merged_dir: Path, staging: Path) -> Path:
    """Stage the Chronos-2 bundle from the merged full-model directory."""
    if not (merged_dir / "config.json").exists():
        raise FileNotFoundError(
            f"Merged Chronos-2 model not found at {merged_dir} "
            "(run tools/forecasting/merge_chronos_lora.py first)."
        )
    bundle = staging / "openmhc-chronos2-fc"
    _reset_dir(bundle)
    shutil.copytree(merged_dir, bundle / "checkpoint")
    write_manifest(
        bundle,
        kind="chronos2",
        checkpoint="checkpoint",
        arch={},
        normalization_stats=None,
        provenance={
            "model": "chronos2",
            "base_model": "amazon/chronos-2",
            "finetune_mode": "lora",
            "lora_r": 8,
            "lora_alpha": 16,
            "context_length": 168,
            "merged_from_adapter": True,
            "wandb_run": "kvxw0ty9",
            "wandb_entity": "MHC_Dataset",
            "wandb_project": "mhc-forecasting",
            "paper_table": PAPER_TABLE,
        },
    )
    _write_readme(bundle, "chronos2")
    return bundle


def stage_toto(toto_ckpt: Path, staging: Path) -> Path:
    """Stage the Toto bundle from the full-finetune Lightning .ckpt."""
    if not toto_ckpt.exists():
        raise FileNotFoundError(f"Toto checkpoint not found: {toto_ckpt}")
    bundle = staging / "openmhc-toto-fc"
    _reset_dir(bundle)
    shutil.copy2(toto_ckpt, bundle / "model.ckpt")
    write_manifest(
        bundle,
        kind="toto",
        checkpoint="model.ckpt",
        arch={},
        normalization_stats=None,
        provenance={
            "model": "toto",
            "base_model": "Datadog/Toto-Open-Base-1.0",
            "finetune_mode": "full",
            "source_checkpoint": toto_ckpt.name,
            "wandb_entity": "MHC_Dataset",
            "wandb_project": "mhc-forecasting",
            "paper_table": PAPER_TABLE,
        },
    )
    _write_readme(bundle, "toto")
    return bundle


_WRAPPERS = {
    "chronos2": "Chronos2Forecaster",
    "toto": "TotoForecaster",
    "dlinear": "DLinearForecaster",
    "segrnn": "SegRNNForecaster",
    "mixlinear": "MixLinearForecaster",
}


def _write_readme(bundle: Path, kind: str) -> None:
    repo = f"MyHeartCounts/openmhc-{kind}-fc"
    wrapper = _WRAPPERS[kind]
    (bundle / "README.md").write_text(
        f"""---
license: cc-by-4.0
tags:
- time-series-forecasting
- wearables
- openmhc
---

# OpenMHC Forecasting — {kind}

Track 3 (forecasting) reference checkpoint for the **MyHeartCounts / OpenMHC**
wearable-health benchmark, trained/fine-tuned on the benchmark training split.

## Usage

```python
import openmhc
from openmhc.forecasters import {wrapper}

fc = {wrapper}.from_release("hf://{repo}")
results = openmhc.evaluate_forecasting(fc, version="full")
```

See `openmhc_manifest.json` for provenance and architecture metadata.
""",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    """Stage the requested forecasting bundles under ``--staging-dir``."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--staging-dir", type=Path, default=Path("releases-fc"))
    p.add_argument("--ckpt-root", type=Path, default=Path("forecasting_model_ckpt"))
    p.add_argument(
        "--scaler",
        type=Path,
        default=None,
        help="train-split standard_scaler_stats.json (required for neural bundles)",
    )
    p.add_argument("--chronos-merged", type=Path, default=Path(".merge_cache/chronos2_FT_merged"))
    p.add_argument("--toto-ckpt", type=Path, default=None, help="Toto .ckpt (required to stage toto)")
    p.add_argument(
        "--only",
        action="append",
        default=None,
        help="Restrict to specific kinds (repeatable): dlinear/segrnn/mixlinear/chronos2/toto.",
    )
    args = p.parse_args(argv)

    staging = args.staging_dir
    staging.mkdir(parents=True, exist_ok=True)
    want = set(args.only) if args.only else None

    neural_wanted = [k for k in NEURAL if want is None or k in want]
    if neural_wanted and args.scaler is None:
        p.error("--scaler is required when staging neural bundles: " + ", ".join(neural_wanted))

    built: list[Path] = []
    for kind in neural_wanted:
        built.append(stage_neural(kind, args.ckpt_root, args.scaler.expanduser(), staging))
    if want is None or "chronos2" in want:
        built.append(stage_chronos(args.chronos_merged.expanduser(), staging))
    if want is None or "toto" in want:
        if args.toto_ckpt is None:
            p.error("--toto-ckpt is required when staging the toto bundle")
        built.append(stage_toto(args.toto_ckpt.expanduser(), staging))

    print(f"Staged {len(built)} bundle(s) under {staging}:")
    for b in built:
        files = sorted(x.name for x in b.iterdir())
        print(f"  {b.name}: {files}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
