"""Toto encoder (channel-wise last-latent time-series foundation model).

Concrete :class:`~downstream_evaluation.models.tsfm.TSFMEncoder`: shares the whole
extraction path (label-aligned 2048 h history windows from ``daily_hourly_hf``) and
supplies only the Toto-specific model load + batch forward. On a cache miss the
``encode_cohort`` call extracts per-(split, task) ``(N, 19, 768)`` last-latent
features (GPU), then the engine channel-mean-pools to 768 and runs PCA-50 + probe.

The pretrained model is an external dependency (a git submodule checkout + a
fine-tuned Lightning checkpoint), resolved by reference like any model weight —
``TOTO_REPO`` / ``TOTO_CHECKPOINT`` env vars override the repo-relative defaults.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np

from downstream_evaluation.data.inputs import Window
from downstream_evaluation.models.tsfm import TSFMEncoder

logger = logging.getLogger(__name__)

WINDOW_HOURS = 2048
DEFAULT_BASE_CHECKPOINT = "Datadog/Toto-Open-Base-1.0"
DEFAULT_TOTO_REPO = os.environ.get("TOTO_REPO", "external/toto")
DEFAULT_CHECKPOINT = os.environ.get(
    "TOTO_CHECKPOINT", "models/final/toto_FT/toto-epoch=24-step=116225-val_loss=-1.3597.ckpt"
)


def _resolve(path: str | Path) -> Path:
    """Resolve a path against the repo root."""
    p = Path(path)
    if p.is_absolute():
        return p
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / p


# --------------------------------------------------------------------------- #
# Model load: base Toto + merged LoRA fine-tune.
# --------------------------------------------------------------------------- #
def _add_toto_repo_to_path(toto_repo: Path) -> None:
    if not (toto_repo / "toto" / "model" / "toto.py").exists():
        raise FileNotFoundError(
            f"Toto package not found under {toto_repo}. Set TOTO_REPO or run "
            "`git submodule update --init external/toto`."
        )
    toto_repo = toto_repo.resolve()
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != toto_repo]
    sys.path.insert(0, str(toto_repo))
    loaded = sys.modules.get("toto")
    loaded_file = getattr(loaded, "__file__", None)
    if loaded_file is not None and not Path(loaded_file).resolve().is_relative_to(toto_repo):
        for name in list(sys.modules):
            if name == "toto" or name.startswith("toto."):
                del sys.modules[name]


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


def _load_toto_backbone(checkpoint, toto_repo, device, base_checkpoint, lora_alpha=None):
    import torch

    _add_toto_repo_to_path(toto_repo)
    from toto.model.toto import Toto  # noqa: PLC0415

    ckpt_path = _resolve(checkpoint)
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
        logger.info("loading Toto checkpoint %s", checkpoint)
        toto = Toto.from_pretrained(checkpoint, map_location=str(device))
    toto = toto.to(device).eval()
    return toto.model.eval()


# --------------------------------------------------------------------------- #
class Toto(TSFMEncoder):
    """Toto channel-wise last-latent encoder for the engine."""

    name = "toto"
    pooling_label = "none_channelwise_last_latent"
    input = Window(WINDOW_HOURS, anchor="window_end")  # materializer builds the 2048h windows

    def __init__(
        self,
        data_dir=None,
        cache_dir=None,
        checkpoint=DEFAULT_CHECKPOINT,
        base_checkpoint=DEFAULT_BASE_CHECKPOINT,
        toto_repo=DEFAULT_TOTO_REPO,
        lora_alpha=None,
        batch_size=32,
        seed=42,
    ):
        """Configure the TSFM base plus the Toto checkpoint / repo / LoRA-merge settings."""
        super().__init__(data_dir=data_dir, cache_dir=cache_dir, batch_size=batch_size, seed=seed)
        self.checkpoint = checkpoint
        self.base_checkpoint = base_checkpoint
        self.toto_repo = toto_repo
        self.lora_alpha = lora_alpha

    def _load_model(self, device):
        backbone = _load_toto_backbone(
            self.checkpoint, _resolve(self.toto_repo), device, self.base_checkpoint, self.lora_alpha
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
            latents = handle.last_latent(x, padding_mask, id_mask)
        return latents.detach().float().cpu().numpy().astype(np.float32)
