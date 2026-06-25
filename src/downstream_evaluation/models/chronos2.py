"""Chronos-2 encoder (channel-wise last-latent time-series foundation model).

Twin of :mod:`downstream_evaluation.models.toto`: identical extraction path, only
the model load and batch forward differ. On a cache miss ``fit`` / ``predict`` extract
per-(split, task) ``(N, 19, 768)`` ``predict_last_latent`` features (GPU),
channel-mean-pool to 768, and run the uniform PCA-50 probe.

The architecture comes from the stock PyPI ``chronos-forecasting`` package; the
fine-tuned weights come from the Hugging Face Hub
(``hf://MyHeartCounts/openmhc-chronos2-fc``). ``CHRONOS2_CHECKPOINT`` overrides the
default (a local path also works). The history window length defaults to the model's
``model_context_length``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from downstream_evaluation.models.tsfm import TSFMEncoder

logger = logging.getLogger(__name__)

PREDICTION_LENGTH = 24
DEFAULT_CHECKPOINT = os.environ.get("CHRONOS2_CHECKPOINT", "hf://MyHeartCounts/openmhc-chronos2-fc")


def _resolve_checkpoint(checkpoint: str | Path) -> Path:
    """Resolve a checkpoint reference to a local path.

    ``hf://ORG/REPO[@rev]`` snapshot-downloads the release bundle from the Hugging Face
    Hub and returns the local checkpoint (a directory for Chronos-2). A relative local
    path resolves against the repo root; an absolute path is returned as-is.
    """
    ref = str(checkpoint)
    if ref.startswith("hf://"):
        return _download_hf_release(ref)
    p = Path(ref)
    return p if p.is_absolute() else Path(__file__).resolve().parents[3] / p


def _download_hf_release(uri: str) -> Path:
    """Snapshot-download an ``hf://org/repo[@rev]`` OpenMHC release bundle; return the
    local path named by its manifest's ``checkpoint`` entry."""
    import json

    from huggingface_hub import snapshot_download

    repo_id, _, revision = uri[len("hf://") :].partition("@")
    local = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision or None,
            allow_patterns=[
                "openmhc_manifest.json", "*.ckpt", "checkpoint/**", "config.json", "*.safetensors",
            ],
        )
    )
    manifest = json.loads((local / "openmhc_manifest.json").read_text())
    return (local / manifest["checkpoint"]).resolve()


def _load_chronos2_pipeline(checkpoint, device) -> Any:
    """Load the fine-tuned Chronos-2 pipeline from the stock ``chronos`` package."""
    from chronos import Chronos2Pipeline  # noqa: PLC0415

    ckpt = _resolve_checkpoint(checkpoint)
    logger.info("loading Chronos-2 %s device_map=%s", ckpt, device)
    pipeline = Chronos2Pipeline.from_pretrained(str(ckpt), device_map=str(device))
    pipeline.model.eval()
    return pipeline




# ---------------------------------------------------------------------------
# Vendored last-latent extraction.
#
# Reproduces ``Chronos2Pipeline.predict_last_latent`` so OpenMHC can depend on the
# stock PyPI ``chronos-forecasting==2.3.0`` release rather than a fork. That method
# is NOT part of stock Chronos-2 — it was added in the (now-retired) fork
# ``github.com/ligeaaa/chronos-forecasting@d56b70c``, whose base is identical to
# v2.3.0 in the extraction path (model.py/pipeline.py/base.py have zero diff vs the
# fork base; only the TEST-mode-inert ``future_covariates`` change differs in
# dataset.py), so this function re-derives that extraction against the stock release.
# It is a faithful port, not a verified bit-exact match to the retired fork's own run —
# downstream features differ measurably from the earlier fork-based extraction.
# Adapted from the fork verbatim: ``self`` -> ``pipeline`` and the fork's
# ``_prepare_predict_step_kwargs`` helper inlined (stock 2.3.0 has no such helper).
# Caller wraps this in ``torch.no_grad()``.
#
# Derived from amazon/chronos-forecasting (Apache-2.0):
#   Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#   SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------------
def _predict_last_latent(
    pipeline,
    inputs,
    prediction_length: int | None = None,
    batch_size: int = 256,
    context_length: int | None = None,
    cross_learning: bool = False,
    limit_prediction_length: bool = False,
) -> list:
    """Last latent of the final Chronos-2 encoder layer, per input item.

    Returns a list of tensors, each shape ``(n_target_variates, d_model)``, using
    the same input formats and batching path as ``predict``/``predict_quantiles``.
    """
    import math
    import warnings

    import torch
    from torch.utils.data import DataLoader

    from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode

    model_prediction_length = pipeline.model_prediction_length
    if prediction_length is None:
        prediction_length = model_prediction_length

    max_output_patches = pipeline.max_output_patches

    if prediction_length > model_prediction_length:
        msg = (
            f"We recommend keeping prediction length <= {model_prediction_length}. "
            "The quality of longer predictions may degrade since the model is not "
            "optimized for it. "
        )
        if limit_prediction_length:
            msg += "You can turn off this check by setting `limit_prediction_length=False`."
            raise ValueError(msg)
        warnings.warn(msg)

    if context_length is None:
        context_length = pipeline.model_context_length

    if context_length > pipeline.model_context_length:
        warnings.warn(
            f"The specified context_length {context_length} is greater than the model's "
            f"default context length {pipeline.model_context_length}. "
            f"Resetting context_length to {pipeline.model_context_length}."
        )
        context_length = pipeline.model_context_length

    test_dataset = Chronos2Dataset(
        inputs,
        context_length=context_length,
        prediction_length=prediction_length,
        batch_size=batch_size,
        output_patch_size=pipeline.model_output_patch_size,
        mode=DatasetMode.TEST,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=None,
        pin_memory=pipeline.model.device.type == "cuda",
        shuffle=False,
        drop_last=False,
    )

    all_last_latents: list = []
    for batch in test_loader:
        assert batch["future_target"] is None
        batch_context = batch["context"].to(device=pipeline.model.device, dtype=torch.float32)
        batch_group_ids = batch["group_ids"].to(device=pipeline.model.device)
        batch_future_covariates = batch["future_covariates"].to(
            device=pipeline.model.device, dtype=torch.float32
        )
        batch_target_idx_ranges = batch["target_idx_ranges"]

        if cross_learning:
            batch_group_ids = torch.zeros_like(batch_group_ids)

        num_output_patches = math.ceil(prediction_length / pipeline.model_output_patch_size)
        num_output_patches = min(num_output_patches, max_output_patches)

        # Inlined fork helper ``_prepare_predict_step_kwargs`` (absent in stock 2.3.0).
        encode_kwargs: dict = {}
        if batch_future_covariates is not None:
            output_size = num_output_patches * pipeline.model_output_patch_size
            if output_size > batch_future_covariates.shape[1]:
                fc_bsz = len(batch_future_covariates)
                padding_size = output_size - batch_future_covariates.shape[1]
                padding_tensor = torch.full(
                    (fc_bsz, padding_size),
                    fill_value=torch.nan,
                    device=batch_future_covariates.device,
                )
                batch_future_covariates = torch.cat([batch_future_covariates, padding_tensor], dim=1)
            else:
                batch_future_covariates = batch_future_covariates[..., :output_size]
            encode_kwargs["future_covariates"] = batch_future_covariates

        encoder_outputs, *_ = pipeline.model.encode(
            context=batch_context,
            group_ids=batch_group_ids,
            num_output_patches=num_output_patches,
            **encode_kwargs,
        )
        last_latents = encoder_outputs.last_hidden_state[:, -1, :].to(
            dtype=torch.float32, device="cpu"
        )
        all_last_latents.extend(last_latents[start:end] for (start, end) in batch_target_idx_ranges)

    return all_last_latents



class Chronos2(TSFMEncoder):
    """Chronos-2 channel-wise last-latent encoder for the engine."""

    name = "chronos2"
    pooling_label = "chronos2_predict_last_latent"

    def __init__(self, data_dir=None, cache_dir=None, checkpoint=DEFAULT_CHECKPOINT,
                 prediction_length=PREDICTION_LENGTH, window_hours=None, batch_size=32, seed=42):
        super().__init__(data_dir=data_dir, cache_dir=cache_dir, batch_size=batch_size, seed=seed)
        self.checkpoint = checkpoint
        self.prediction_length = prediction_length
        self._window_override = window_hours

    def _load_model(self, device):
        pipeline = _load_chronos2_pipeline(self.checkpoint, device)
        ctx = int(pipeline.model_context_length)
        window_hours = ctx if self._window_override is None else min(int(self._window_override), ctx)
        return pipeline, window_hours

    def _run_batch(self, handle, examples, window_hours) -> np.ndarray:
        import torch

        x_np = np.stack(
            [np.where(e.padding_mask, e.window, np.nan).astype(np.float32, copy=False)
             for e in examples]
        )
        with torch.no_grad():
            latents = _predict_last_latent(
                handle, x_np, prediction_length=self.prediction_length,
                batch_size=len(examples), context_length=window_hours,
            )
        embeddings = torch.stack([latent.detach().float().cpu() for latent in latents], dim=0)
        return embeddings.numpy().astype(np.float32)
