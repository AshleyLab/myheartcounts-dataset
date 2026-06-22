"""Chronos-2 encoder (channel-wise last-latent time-series foundation model).

Twin of :mod:`downstream_evaluation.models.toto`: identical extraction path, only
the model load and batch forward differ. On a cache miss ``fit`` / ``predict`` extract
per-(split, task) ``(N, 19, 768)`` ``predict_last_latent`` features (GPU),
channel-mean-pool to 768, and run the uniform PCA-50 probe.

The pretrained model is an external dependency (a git submodule checkout + a
fine-tuned LoRA adapter directory), resolved by reference — ``CHRONOS_REPO`` /
``CHRONOS2_CHECKPOINT`` env vars override the repo-relative defaults. The history
window length defaults to the model's ``model_context_length``.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from downstream_evaluation.models.tsfm import TSFMEncoder

logger = logging.getLogger(__name__)

PREDICTION_LENGTH = 24
DEFAULT_BASE_CHECKPOINT = "amazon/chronos-2"
DEFAULT_CHRONOS_REPO = os.environ.get("CHRONOS_REPO", "external/chronos-forecasting")
DEFAULT_CHECKPOINT = os.environ.get("CHRONOS2_CHECKPOINT", "models/final/chronos2_FT")


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return Path(__file__).resolve().parents[3] / p


def _add_chronos_repo_to_path(chronos_repo: Path) -> None:
    if not (chronos_repo / "src" / "chronos" / "chronos2" / "pipeline.py").exists():
        raise FileNotFoundError(
            f"Chronos-2 package not found under {chronos_repo}. Set CHRONOS_REPO or run "
            "`git submodule update --init external/chronos-forecasting`."
        )
    chronos_src = (chronos_repo / "src").resolve()
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != chronos_src]
    sys.path.insert(0, str(chronos_src))
    loaded = sys.modules.get("chronos")
    loaded_file = getattr(loaded, "__file__", None)
    if loaded_file is not None and not Path(loaded_file).resolve().is_relative_to(chronos_src):
        for name in list(sys.modules):
            if name == "chronos" or name.startswith("chronos."):
                del sys.modules[name]


def _load_chronos2_pipeline(checkpoint, chronos_repo, device) -> Any:
    _add_chronos_repo_to_path(chronos_repo)
    module = importlib.import_module("chronos")
    if not Path(module.__file__).resolve().is_relative_to((chronos_repo / "src").resolve()):
        raise ImportError(f"Imported chronos from {module.__file__}, expected under {chronos_repo}")
    from chronos import Chronos2Pipeline  # noqa: PLC0415

    ckpt_path = _resolve(checkpoint)
    checkpoint_ref = str(ckpt_path) if ckpt_path.exists() else checkpoint
    logger.info("loading Chronos-2 %s device_map=%s", checkpoint_ref, device)
    pipeline = Chronos2Pipeline.from_pretrained(checkpoint_ref, device_map=str(device))
    pipeline.model.eval()
    return pipeline


class Chronos2(TSFMEncoder):
    """Chronos-2 channel-wise last-latent encoder for the engine."""

    name = "chronos2"
    pooling_label = "chronos2_predict_last_latent"

    def __init__(self, data_dir=None, cache_dir=None, checkpoint=DEFAULT_CHECKPOINT,
                 base_checkpoint=DEFAULT_BASE_CHECKPOINT, chronos_repo=DEFAULT_CHRONOS_REPO,
                 prediction_length=PREDICTION_LENGTH, window_hours=None, batch_size=32, seed=42):
        super().__init__(data_dir=data_dir, cache_dir=cache_dir, batch_size=batch_size, seed=seed)
        self.checkpoint = checkpoint
        self.base_checkpoint = base_checkpoint
        self.chronos_repo = chronos_repo
        self.prediction_length = prediction_length
        self._window_override = window_hours

    def _load_model(self, device):
        pipeline = _load_chronos2_pipeline(self.checkpoint, _resolve(self.chronos_repo), device)
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
            latents = handle.predict_last_latent(
                x_np, prediction_length=self.prediction_length,
                batch_size=len(examples), context_length=window_hours,
            )
        embeddings = torch.stack([latent.detach().float().cpu() for latent in latents], dim=0)
        return embeddings.numpy().astype(np.float32)
