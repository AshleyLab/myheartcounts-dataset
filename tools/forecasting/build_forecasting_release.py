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
        k: model[k] for k in ("n_steps", "n_pred_steps", "n_features") if model.get(k) is not None
    }


def stage_neural(
    kind: str, ckpt_root: Path, scaler: Path, staging: Path, release_tag: str = "v1.0"
) -> Path:
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
    _write_readme(bundle, kind, pypots_name, release_tag)
    return bundle


def stage_neural_from_training_bundle(
    kind: str, src_bundle: Path, staging: Path, release_tag: str = "v1.0"
) -> Path:
    """Stage a neural bundle from a ``forecasting_training`` release directory.

    Unlike :func:`stage_neural` (which packages the externally-trained HPO
    checkpoints and hard-codes provenance), this consumes a bundle already
    emitted by ``forecasting_training.release.write_release``: ``model.pypots``
    plus ``training_config.json``, ``standard_scaler_stats.json``, and a manifest
    carrying rich machine-derived provenance (seed, hyperparameters, source
    checkpoint). That provenance and the ``model.pypots`` layout are preserved
    verbatim; we only add the publish-time ``paper_table`` breadcrumb and the HF
    model card that ``forecasting_training`` does not write.
    """
    src_bundle = Path(src_bundle)
    src_manifest = src_bundle / "openmhc_manifest.json"
    if not src_manifest.exists():
        raise FileNotFoundError(f"No openmhc_manifest.json in training bundle: {src_bundle}")
    manifest = json.loads(src_manifest.read_text())
    if manifest.get("kind") != kind:
        raise ValueError(
            f"Training bundle {src_bundle} has kind {manifest.get('kind')!r}, expected {kind!r}"
        )

    ckpt_name = manifest["checkpoint"]  # "model.pypots"
    stats_name = manifest.get("normalization_stats")
    src_ckpt = src_bundle / ckpt_name
    src_cfg = src_bundle / "training_config.json"
    for f in (src_ckpt, src_cfg):
        if not f.exists():
            raise FileNotFoundError(f"Missing file in training bundle: {f}")

    bundle = staging / f"openmhc-{kind}-fc"
    _reset_dir(bundle)
    shutil.copy2(src_ckpt, bundle / ckpt_name)
    shutil.copy2(src_cfg, bundle / "training_config.json")
    if stats_name:
        src_stats = src_bundle / stats_name
        if not src_stats.exists():
            raise FileNotFoundError(f"Manifest references missing stats file: {src_stats}")
        shutil.copy2(src_stats, bundle / stats_name)

    # Preserve the rich training provenance; only add the publish-time paper table.
    provenance = dict(manifest.get("provenance") or {})
    provenance.setdefault("paper_table", PAPER_TABLE)

    write_manifest(
        bundle,
        kind=kind,
        checkpoint=ckpt_name,
        arch=manifest.get("arch") or {},
        normalization_stats=stats_name,
        provenance=provenance,
    )
    _write_readme(bundle, kind, ckpt_name, release_tag)
    return bundle


def stage_chronos(merged_dir: Path, staging: Path, release_tag: str = "v1.0") -> Path:
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
    _write_readme(bundle, "chronos2", release_tag=release_tag)
    return bundle


def stage_toto(toto_ckpt: Path, staging: Path, release_tag: str = "v1.0") -> Path:
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
    _write_readme(bundle, "toto", release_tag=release_tag)
    return bundle


# Per-model card metadata: wrapper class, display name, one-line architecture
# summary, how the checkpoint was produced, the install extra, and the upstream
# links (implementation + paper) for the model the checkpoint belongs to.
_MODEL_INFO = {
    "chronos2": {
        "wrapper": "Chronos2Forecaster",
        "title": "Chronos-2",
        "extra": "chronos",
        "format": "HuggingFace model directory (`checkpoint/`: `config.json` + `model.safetensors`)",
        "summary": (
            "Chronos-2 is Amazon's universal time-series foundation model, supporting "
            "multivariate and covariate-informed probabilistic forecasting."
        ),
        "produced": (
            "**Fine-tuned** from the pretrained `amazon/chronos-2` base on the MHC training "
            "split (LoRA, rank 8, alpha 16; the adapter has been merged into the base so this "
            "is a standalone full model — no PEFT runtime dependency)."
        ),
        "links": [
            (
                "Chronos (official implementation)",
                "https://github.com/amazon-science/chronos-forecasting",
            ),
            ("Base model `amazon/chronos-2`", "https://huggingface.co/amazon/chronos-2"),
            (
                "Paper: *Chronos-2: From Univariate to Universal Forecasting* (Ansari et al., 2025)",
                "https://arxiv.org/abs/2510.15821",
            ),
        ],
    },
    "toto": {
        "wrapper": "TotoForecaster",
        "title": "Toto",
        "extra": "toto",
        "format": "PyTorch Lightning checkpoint (`model.ckpt`)",
        "summary": (
            "Toto (Time-series Optimized Transformer for Observability) is Datadog's "
            "decoder-only time-series foundation model with a probabilistic output head."
        ),
        "produced": (
            "**Fine-tuned** (full) from the pretrained `Datadog/Toto-Open-Base-1.0` base on the "
            "MHC training split."
        ),
        "links": [
            ("Toto (official implementation)", "https://github.com/DataDog/toto"),
            (
                "Base model `Datadog/Toto-Open-Base-1.0`",
                "https://huggingface.co/Datadog/Toto-Open-Base-1.0",
            ),
            (
                "Paper: *Toto: Time Series Optimized Transformer for Observability* (Cohen et al., 2024)",
                "https://arxiv.org/abs/2407.07874",
            ),
        ],
    },
    "dlinear": {
        "wrapper": "DLinearForecaster",
        "title": "DLinear",
        "extra": "pypots",
        "format": "PyPOTS checkpoint (`{ckpt}`) + `standard_scaler_stats.json` + `training_config.json`",
        "summary": (
            "DLinear is a lightweight linear forecaster that decomposes the series into trend "
            "and seasonal components and applies a separate linear projection to each."
        ),
        "produced": "**Trained from scratch** on the MHC training split using the PyPOTS implementation.",
        "links": [
            ("PyPOTS toolkit (implementation)", "https://github.com/WenjieDu/PyPOTS"),
            ("PyPOTS documentation", "https://docs.pypots.com"),
            (
                "Paper: *Are Transformers Effective for Time Series Forecasting?* (Zeng et al., AAAI 2023)",
                "https://github.com/cure-lab/LTSF-Linear",
            ),
        ],
    },
    "segrnn": {
        "wrapper": "SegRNNForecaster",
        "title": "SegRNN",
        "extra": "pypots",
        "format": "PyPOTS checkpoint (`{ckpt}`) + `standard_scaler_stats.json` + `training_config.json`",
        "summary": (
            "SegRNN is an RNN-based long-horizon forecaster that uses segment-wise iterations "
            "and parallel multi-step decoding."
        ),
        "produced": "**Trained from scratch** on the MHC training split using the PyPOTS implementation.",
        "links": [
            ("PyPOTS toolkit (implementation)", "https://github.com/WenjieDu/PyPOTS"),
            ("PyPOTS documentation", "https://docs.pypots.com"),
            ("SegRNN (original implementation)", "https://github.com/lss-1138/SegRNN"),
            (
                "Paper: *SegRNN: Segment Recurrent Neural Network for Long-Term Time Series Forecasting* (Lin et al., IEEE IoT-J 2025)",
                "https://github.com/lss-1138/SegRNN",
            ),
        ],
    },
    "mixlinear": {
        "wrapper": "MixLinearForecaster",
        "title": "MixLinear",
        "extra": "pypots",
        "format": "PyPOTS checkpoint (`{ckpt}`) + `standard_scaler_stats.json` + `training_config.json`",
        "summary": (
            "MixLinear is an extremely lightweight forecaster combining segment-based linear "
            "modeling in the time domain with adaptive low-rank filtering in the frequency domain."
        ),
        "produced": (
            "**Trained from scratch** on the MHC training split using the PyPOTS implementation "
            "(requires `pypots>=1.2`)."
        ),
        "links": [
            ("PyPOTS toolkit (implementation)", "https://github.com/WenjieDu/PyPOTS"),
            ("PyPOTS documentation", "https://docs.pypots.com"),
            (
                "Paper: *MixLinear: Extreme Low Resource Multivariate Time Series Forecasting with 0.1K Parameters* (Ma et al., 2024)",
                "https://arxiv.org/abs/2410.02081",
            ),
        ],
    },
}

# Back-compat alias used elsewhere in this module.
_WRAPPERS = {k: v["wrapper"] for k, v in _MODEL_INFO.items()}


def _write_readme(
    bundle: Path, kind: str, ckpt_filename: str = "", release_tag: str = "v1.0"
) -> None:
    info = _MODEL_INFO[kind]
    repo = f"MyHeartCounts/openmhc-{kind}-fc"
    wrapper = info["wrapper"]
    # Neural cards parametrize the checkpoint filename ({ckpt}); chronos/toto
    # carry a fixed format string with no placeholder, so .format() is a no-op.
    ckpt_format = info["format"].format(ckpt=ckpt_filename)
    links = "\n".join(f"- [{label}]({url})" for label, url in info["links"])
    (bundle / "README.md").write_text(
        f"""---
license: cc-by-4.0
library_name: openmhc
tags:
- time-series-forecasting
- wearables
- openmhc
---

# OpenMHC Forecasting — {info["title"]}

Track 3 (forecasting) reference checkpoint for the **MyHeartCounts / OpenMHC**
wearable-health benchmark (NeurIPS 2026).

**This checkpoint is a {info["title"]} model.** {info["summary"]}

{info["produced"]}

- **Checkpoint format:** {ckpt_format}
- **Forecasting task:** 24-hour-ahead, 19 sensor channels, hourly resolution.

## Model & implementation

{links}

## Usage

```python
import openmhc
from openmhc.forecasters import {wrapper}

# pip install "openmhc[{info["extra"]}]"
fc = {wrapper}.from_release("hf://{repo}@{release_tag}")
results = openmhc.evaluate_forecasting(fc, version="full")
```

The same bundle also loads in the evaluation harness via
`model.release_dir=hf://{repo}@{release_tag}`. See `openmhc_manifest.json` for
provenance (training run, base model, fine-tuning details) and architecture
metadata.

## Citation

If you use this checkpoint, please cite the OpenMHC benchmark and the original
{info["title"]} work (linked above).
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
        help="train-split standard_scaler_stats.json (required for neural bundles "
        "staged from --ckpt-root)",
    )
    p.add_argument(
        "--neural-bundle",
        action="append",
        default=None,
        metavar="KIND=DIR",
        help="Stage a neural bundle from a forecasting_training release dir "
        "(repeatable), e.g. dlinear=results/.../dlinear_<ts>. Carries its own "
        "checkpoint + scaler + rich provenance, so --ckpt-root/--scaler are "
        "ignored for that kind.",
    )
    p.add_argument(
        "--release-tag",
        default="v1.0",
        help="Version tag referenced in generated model cards' usage examples "
        "(hf://...@<tag>). Match this to publish_to_hf.py --tag. Default: v1.0.",
    )
    p.add_argument("--chronos-merged", type=Path, default=Path(".merge_cache/chronos2_FT_merged"))
    p.add_argument(
        "--toto-ckpt", type=Path, default=None, help="Toto .ckpt (required to stage toto)"
    )
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

    overrides: dict[str, Path] = {}
    for item in args.neural_bundle or []:
        if "=" not in item:
            p.error(f"--neural-bundle expects KIND=DIR, got {item!r}")
        k, v = item.split("=", 1)
        if k not in NEURAL:
            p.error(f"--neural-bundle KIND must be one of {sorted(NEURAL)}; got {k!r}")
        overrides[k] = Path(v)
    if want is not None:
        stray = sorted(k for k in overrides if k not in want)
        if stray:
            p.error(f"--neural-bundle given for {stray} but excluded by --only")

    neural_wanted = [k for k in NEURAL if want is None or k in want]
    legacy_neural = [k for k in neural_wanted if k not in overrides]
    if legacy_neural and args.scaler is None:
        p.error(
            "--scaler is required when staging neural bundles from --ckpt-root: "
            + ", ".join(legacy_neural)
        )

    built: list[Path] = []
    for kind in neural_wanted:
        if kind in overrides:
            built.append(
                stage_neural_from_training_bundle(kind, overrides[kind], staging, args.release_tag)
            )
        else:
            built.append(
                stage_neural(
                    kind, args.ckpt_root, args.scaler.expanduser(), staging, args.release_tag
                )
            )
    if want is None or "chronos2" in want:
        built.append(
            stage_chronos(args.chronos_merged.expanduser(), staging, release_tag=args.release_tag)
        )
    if want is None or "toto" in want:
        if args.toto_ckpt is None:
            p.error("--toto-ckpt is required when staging the toto bundle")
        built.append(stage_toto(args.toto_ckpt.expanduser(), staging, release_tag=args.release_tag))

    print(f"Staged {len(built)} bundle(s) under {staging}:")
    for b in built:
        files = sorted(x.name for x in b.iterdir())
        print(f"  {b.name}: {files}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
