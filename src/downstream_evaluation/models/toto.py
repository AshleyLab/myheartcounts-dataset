"""Toto encoder (channel-wise last-latent time-series foundation model).

Concrete :class:`~downstream_evaluation.models.tsfm.TSFMEncoder`: shares the whole
extraction path (label-aligned 2048 h history windows from ``daily_hourly_hf``) and
supplies only the Toto-specific model load + batch forward. On a cache miss
``fit`` / ``predict`` extract per-(split, task) ``(N, 19, 768)`` last-latent
features (GPU), channel-mean-pool to 768, and run the uniform PCA-50 probe.

The architecture comes from the stock PyPI ``toto-ts`` package; the fine-tuned weights
come from the Hugging Face Hub (``hf://MyHeartCounts/openmhc-toto-fc`` — base
``Datadog/Toto-Open-Base-1.0`` with the fine-tune merged on top). ``TOTO_CHECKPOINT``
overrides the default (a local ``.ckpt`` also works).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from downstream_evaluation.models.tsfm import TSFMEncoder

logger = logging.getLogger(__name__)

WINDOW_HOURS = 2048
DEFAULT_BASE_CHECKPOINT = "Datadog/Toto-Open-Base-1.0"
DEFAULT_CHECKPOINT = os.environ.get("TOTO_CHECKPOINT", "hf://MyHeartCounts/openmhc-toto-fc")


def _resolve_checkpoint(checkpoint: str | Path) -> Path:
    """Resolve a checkpoint reference to a local path.

    ``hf://ORG/REPO[@rev]`` snapshot-downloads the release bundle from the Hugging Face
    Hub and returns the local checkpoint (a ``.ckpt`` file for Toto). A relative local
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
    # Don't .resolve(): the snapshot entry is a symlink into blobs/, and resolving it
    # strips the manifest filename — including the ".ckpt" suffix that _load_toto_backbone
    # keys on to pick the base-load + state-dict-merge path. ``local`` is already absolute
    # and torch.load follows the symlink.
    return local / manifest["checkpoint"]


def _strip_lightning_prefix(key: str) -> str:
    for prefix in ("model.base_model.", "base_model."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _convert_lightning_state_dict(state_dict, lora_alpha=None):
    """Convert Lightning/LoRA Toto keys to the external wrapper layout (merge LoRA)."""
    converted, lora_a, lora_b = {}, {}, {}
    for raw_key, value in state_dict.items():
        key = _strip_lightning_prefix(raw_key)
        if ".lora_A.default.weight" in key:
            lora_a[key.replace(".lora_A.default.weight", ".weight")] = value.detach().cpu()
            continue
        if ".lora_B.default.weight" in key:
            lora_b[key.replace(".lora_B.default.weight", ".weight")] = value.detach().cpu()
            continue
        key = key.replace(".base_layer.weight", ".weight").replace(".base_layer.bias", ".bias")
        converted[key] = value.detach().cpu()
    for base_key, a_weight in lora_a.items():
        b_weight = lora_b.get(base_key)
        base_weight = converted.get(base_key)
        if b_weight is None or base_weight is None:
            logger.warning("Skipping incomplete Toto LoRA weights for %s", base_key)
            continue
        rank = int(a_weight.shape[0])
        alpha = float(lora_alpha) if lora_alpha is not None else float(rank)
        scaling = alpha / max(rank, 1)
        converted[base_key] = base_weight + (b_weight @ a_weight).to(base_weight.dtype) * scaling
    return converted


def _load_toto_backbone(checkpoint, device, base_checkpoint, lora_alpha=None):
    import torch

    from toto.model.toto import Toto  # noqa: PLC0415  (stock toto-ts)

    ckpt_path = _resolve_checkpoint(checkpoint)
    if ckpt_path.exists() and ckpt_path.suffix == ".ckpt":
        logger.info("loading base Toto %s + merging %s", base_checkpoint, ckpt_path)
        toto = Toto.from_pretrained(base_checkpoint, map_location=str(device))
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        merged = _convert_lightning_state_dict(state, lora_alpha=lora_alpha)
        inc = toto.load_state_dict(merged, strict=False)
        missing = [k for k in inc.missing_keys if not k.endswith("rotary_emb.freqs")]
        if missing:
            logger.warning("Toto load missing %d keys; first: %s", len(missing), missing[:5])
    else:
        logger.info("loading Toto checkpoint %s", ckpt_path)
        toto = Toto.from_pretrained(str(ckpt_path), map_location=str(device))
    toto = toto.to(device).eval()
    return toto.model.eval()


# --------------------------------------------------------------------------- #
# Vendored last-latent extraction.
#
# Reproduces ``TotoBackbone.last_latent()`` / ``encode()`` so OpenMHC can depend on
# the stock PyPI ``toto-ts==0.2.0`` release rather than a fork. Those methods are NOT
# in stock Toto — they were added in the (now-retired) fork
# ``github.com/ligeaaa/toto@25f885e``, whose base is byte-identical to v0.2.0 across
# ``toto/model/`` (backbone/scaler/embedding/transformer all blob-identical), so this
# yields identical latents on the stock release. Specialized to the no-kv-cache
# inference path ``_run_batch`` uses (kv_cache is never passed); loc/scale discarded.
#
# Derived from DataDog/toto (Apache-2.0):
#   This product includes software developed at Datadog (https://www.datadoghq.com/)
#   Copyright 2025 Datadog, Inc.
# --------------------------------------------------------------------------- #
def _last_latent(
    backbone,
    inputs,
    input_padding_mask,
    id_mask,
    scaling_prefix_length=None,
    num_exogenous_variables: int = 0,
):
    """Final-patch latent of the last transformer layer, per variate.

    Output shape ``(batch, variate, embed_dim)``.
    """
    import torch

    # Standard scaling operation, same API but without ID mask.
    scaled_inputs, _, _ = backbone.scaler(
        inputs,
        weights=torch.ones_like(inputs, device=inputs.device),
        padding_mask=input_padding_mask,
        prefix_length=scaling_prefix_length,
    )

    embeddings, reduced_id_mask = backbone.patch_embed(scaled_inputs, id_mask)

    # Build variate label embeddings (one per variate) if enabled.
    variate_label_embeds = backbone.build_variate_label_embeds(num_exogenous_variables, embeddings)

    # Apply the transformer (fusion handles prepending at layer 0).
    original_seq_len = embeddings.shape[2]
    transformed = backbone.transformer(
        embeddings, reduced_id_mask, None, variate_label_embeds=variate_label_embeds
    )

    # Crop out any prepended condition tokens, then take the final patch latent.
    added_tokens = transformed.shape[2] - original_seq_len
    if added_tokens > 0:
        transformed = transformed[:, :, added_tokens:]

    return transformed[:, :, -1, :]



# --------------------------------------------------------------------------- #
class Toto(TSFMEncoder):
    """Toto channel-wise last-latent encoder for the engine."""

    name = "toto"
    pooling_label = "none_channelwise_last_latent"

    def __init__(self, data_dir=None, cache_dir=None, checkpoint=DEFAULT_CHECKPOINT,
                 base_checkpoint=DEFAULT_BASE_CHECKPOINT, lora_alpha=None, batch_size=32, seed=42):
        super().__init__(data_dir=data_dir, cache_dir=cache_dir, batch_size=batch_size, seed=seed)
        self.checkpoint = checkpoint
        self.base_checkpoint = base_checkpoint
        self.lora_alpha = lora_alpha

    def _load_model(self, device):
        backbone = _load_toto_backbone(
            self.checkpoint, device, self.base_checkpoint, self.lora_alpha
        )
        return backbone, WINDOW_HOURS

    def _run_batch(self, handle, examples, window_hours) -> np.ndarray:
        import torch

        device = next(handle.parameters()).device
        x = torch.from_numpy(np.stack([e.window for e in examples])).to(device)
        padding_mask = torch.from_numpy(np.stack([e.padding_mask for e in examples])).to(
            device=device, dtype=torch.bool
        )
        id_mask = torch.zeros_like(x)
        with torch.no_grad():
            latents = _last_latent(handle, x, padding_mask, id_mask)
        return latents.detach().float().cpu().numpy().astype(np.float32)

