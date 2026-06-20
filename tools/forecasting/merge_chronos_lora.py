#!/usr/bin/env python
"""Merge a Chronos-2 LoRA adapter into the base model and save a full checkpoint.

The fine-tuned Chronos-2 checkpoint is distributed by the training pipeline as a
PEFT/LoRA adapter (``adapter_config.json`` + ``adapter_model.safetensors``).
``Chronos2Pipeline.from_pretrained(<adapter_dir>)`` loads the base
``amazon/chronos-2`` weights and folds the adapter in at load time, so we can
materialize a standalone, self-contained checkpoint simply by re-saving the
loaded pipeline.

This produces a full HuggingFace model directory (``config.json`` +
``model.safetensors``) that the public ``Chronos2Forecaster`` /
``forecasting_evaluation`` loader consumes without any PEFT dependency.

Run with an environment that has ``chronos`` installed (e.g. the
``mhc-benchmark`` conda env)::

    /opt/conda/envs/mhc-benchmark/bin/python tools/forecasting/merge_chronos_lora.py \
        --adapter-dir ~/MHC-benchmark/models/foundational/chronos2/kvxw0ty9/20260422_T025002/finetuned-ckpt \
        --output-dir  ~/myheartcounts-dataset/.merge_cache/chronos2_FT_merged

A self-check compares forecasts from the adapter pipeline and the re-loaded
merged model on random inputs; it aborts on mismatch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def _forecast(pipeline, context: np.ndarray, horizon: int) -> np.ndarray:
    _quantiles, mean = pipeline.predict_quantiles(
        inputs=[{"target": context}],
        prediction_length=horizon,
        quantile_levels=[0.1, 0.5, 0.9],
    )
    return mean[0].cpu().numpy()


def main(argv: list[str] | None = None) -> int:
    """Merge the LoRA adapter into the base model and save a full checkpoint."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--adapter-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--tolerance", type=float, default=1e-4)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--context-length", type=int, default=168)
    args = p.parse_args(argv)

    from chronos import Chronos2Pipeline

    adapter_dir = args.adapter_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter dir not found: {adapter_dir}")

    # Merge weights in float32 for maximum fidelity; inference can re-cast.
    print(f"Loading base + adapter from {adapter_dir} ...")
    pipe = Chronos2Pipeline.from_pretrained(
        str(adapter_dir), device_map="cpu", torch_dtype=torch.float32
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged full model to {output_dir} ...")
    pipe.save_pretrained(str(output_dir))

    saved = sorted(x.name for x in output_dir.iterdir())
    print("  wrote:", saved)
    if not (output_dir / "config.json").exists() or not any(
        (output_dir / f).exists() for f in ("model.safetensors", "pytorch_model.bin")
    ):
        raise RuntimeError(
            "Merged dir is missing config.json or weights; cannot be loaded as a full model."
        )

    # Self-check: merged model must reproduce the adapter pipeline's forecasts.
    rng = np.random.default_rng(0)
    contexts = [
        np.sin(np.arange(args.context_length) / 6.0).astype("float32")[None, :].repeat(3, 0),
        rng.standard_normal((5, args.context_length)).astype("float32"),
    ]
    merged_pipe = Chronos2Pipeline.from_pretrained(
        str(output_dir), device_map="cpu", torch_dtype=torch.float32
    )
    max_diff = 0.0
    for ctx in contexts:
        a = _forecast(pipe, ctx, args.horizon)
        b = _forecast(merged_pipe, ctx, args.horizon)
        max_diff = max(max_diff, float(np.abs(a - b).max()))
    print(f"adapter-vs-merged max|diff| = {max_diff:.3e} (tolerance {args.tolerance:.1e})")
    if max_diff > args.tolerance:
        print("MERGE SELF-CHECK FAILED — merged model does not match the adapter pipeline.")
        return 1
    print("MERGE OK — merged model matches the adapter pipeline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
