"""WBM (Wearable Behavior Model — Mamba2 contrastive SSL encoder).

Two stages, both run from raw data:

  - **Stage 1 (extraction, GPU):** ``extract_wbm_embeddings`` loads the pretrained
    Mamba2 week encoder checkpoint and runs a forward pass over the weekly tensors
    built on-the-fly from ``daily_hourly_hf`` + the window index, writing the
    per-(user, week) 256-d embeddings. This regenerates the embeddings from raw data
    rather than loading a prebuilt cache. ``mamba_ssm`` kernels are CUDA-only.

  - **Stage 2 (eval):** the ``WBM`` Encoder loads that intermediate and pools each
    user's eligible weeks → 256-d; the engine then runs the uniform PCA-50 + probe.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

N_CHANNELS = 19
# WBM_Final_HPO_best architecture (the checkpoint's dims; load is strict=False with
# a missing-keys check, so a wrong dim fails loudly).
_ARCH = dict(in_dim=38, embed_dim=256, hidden_dim=64, num_layers=4, proj_dim=128, dropout=0.223)
DEFAULT_CHECKPOINT = "wandb:MHC_Dataset/mhc-apple-contrastive-transformer/WBM_Final_HPO_best:v1"


# --------------------------------------------------------------------------- #
# Stage-1 helpers: normalization-stat loading and per-example input assembly.
# --------------------------------------------------------------------------- #
def _resolve_norm_stats(paths):
    """Load the canonical hourly normalization constants.

    The pretraining/eval normalization is **not** a naive per-channel z-score: only
    the 7 continuous channels (0–6) are normalized; the binary/sparse channels
    (7–18) are passed through as identity (mean 0, std 1). These constants are built
    once from the daily train split (``build_normalization_stats``) and stored in
    ``normalization_stats_hourly.json`` — a small fixed preprocessing input (like the
    checkpoint), not an embedding cache. Recomputing them naively here would diverge
    (wrong channel set + weekly-vs-daily population), so we load the canonical file.
    """
    import json

    json_path = paths.daily_hourly_hf.parent / "normalization_stats_hourly.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"normalization_stats_hourly.json not found at {json_path}. Build it from "
            "the daily train split (continuous channels 0–6 only; 7–18 identity)."
        )
    stats = json.loads(json_path.read_text())
    means = np.asarray(stats["means"], dtype=np.float32)
    stds = np.asarray(stats["stds"], dtype=np.float32)
    if means.shape != (N_CHANNELS,) or stds.shape != (N_CHANNELS,):
        raise ValueError(f"normalization stats must be ({N_CHANNELS},); got {means.shape}")
    return means, stds


def _process_example(ex, means, stds) -> np.ndarray:
    """Build the (168, 38) input the checkpoint expects: [z-scored values | missing].

    Missing-flag convention (1=missing), matching the WBM pretraining input.
    """
    values = np.asarray(ex["values"], dtype=np.float32)  # (168, 19)
    missing = np.asarray(ex["mask"], dtype=np.float32)  # (168, 19), 1=missing
    present = 1.0 - missing
    norm = (values - means) / stds
    norm = norm * present  # zero out missing
    return np.concatenate([norm, missing], axis=1)  # (168, 38)


def _load_wbm_encoder(checkpoint: str, device):
    """Load + freeze the Mamba2 week encoder from a Lightning checkpoint."""
    import torch

    from downstream_evaluation.models.wbm.week_encoders_mamba2 import (
        Mamba2WeekEncoder,
    )
    from utils.checkpoints import resolve_checkpoint_path

    ckpt_path = str(resolve_checkpoint_path(checkpoint))
    logger.info("loading WBM checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)

    enc_state = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            enc_state[k.replace("model.", "")] = v
        elif k.startswith("encoder."):
            enc_state[k.replace("encoder.", "")] = v
        else:
            enc_state[k] = v

    # Auto-detect proj_head type from the checkpoint (linear vs 3-layer mlp).
    proj_head_type = "mlp"
    w = enc_state.get("proj_head.0.weight")
    if w is not None and w.shape[0] == _ARCH["proj_dim"]:
        proj_head_type = "linear"

    enc = Mamba2WeekEncoder(proj_head_type=proj_head_type, **_ARCH)
    result = enc.load_state_dict(enc_state, strict=False)
    if result.missing_keys:
        raise RuntimeError(
            f"Missing keys loading WBM checkpoint (arch mismatch?): {result.missing_keys}"
        )
    enc.eval().to(device)
    return enc


# --------------------------------------------------------------------------- #
# Stage 1 — extraction (GPU job): raw -> per-(user, week) 256-d embeddings
# --------------------------------------------------------------------------- #
def extract_wbm_embeddings(
    output_dir: str,
    checkpoint: str = DEFAULT_CHECKPOINT,
    data_dir: str | None = None,
    batch_size: int = 64,
    seed: int = 42,
) -> None:
    """Regenerate the WBM embedding intermediate from raw (GPU).

    Writes ``embeddings.npy`` / ``user_ids.npy`` / ``week_starts.npy`` under ``output_dir``.
    """
    import torch
    from tqdm import tqdm

    from data.datasets.indexed_week_dataset import load_indexed_week_dataset
    from openmhc._evaluate import _DatasetPaths

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("WBM extraction device=%s", device)

    paths = _DatasetPaths.resolve(data_dir)
    ds = load_indexed_week_dataset(
        daily_hourly_hf_dir=str(paths.daily_hourly_hf),
        window_index_path=str(paths.window_index),
        window_size=7,
    )
    logger.info("loaded IndexedWeekDataset: %d windows", len(ds))

    user_ids = np.asarray(ds["user_id"], dtype=object)
    week_starts = np.asarray(ds["week_start"], dtype=object)

    # Canonical normalization constants (channels 0–6 z-scored; 7–18 identity).
    means, stds = _resolve_norm_stats(paths)
    logger.info("WBM norm stats: means[:3]=%s stds[:3]=%s", means[:3], stds[:3])

    enc = _load_wbm_encoder(checkpoint, device)
    embeddings = []
    with torch.no_grad():
        for start in tqdm(range(0, len(ds), batch_size), desc="WBM encode"):
            end = min(start + batch_size, len(ds))
            batch = np.stack([_process_example(ds[i], means, stds) for i in range(start, end)])
            x = torch.from_numpy(batch).to(device)
            _h, r = enc(x)  # r: (B, 256) representation
            embeddings.append(r.cpu().numpy())
    emb = np.concatenate(embeddings, axis=0).astype(np.float32)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", emb)
    np.save(out / "user_ids.npy", user_ids)
    np.save(out / "week_starts.npy", week_starts)
    logger.info("wrote %d WBM embeddings (dim=%d) -> %s", len(emb), emb.shape[1], out)


# --------------------------------------------------------------------------- #
# Stage 2 — the WBM Encoder (driven by run_eval; build-on-miss is internal)
# --------------------------------------------------------------------------- #
class WBM:
    """WBM encoder for the engine.

    ``encode_cohort`` returns the cohort's per-user 256-d embeddings; the embeddings
    are produced **on a cache miss** by running the encoder over raw (GPU), saved, and
    reused on a hit — all inside the eval flow.
    """

    name = "wbm"
    input_granularity = "weekly"  # cohort + eligible weeks come from the weekly lookup
    needs_segments = False  # consumes its own build-on-miss embedding cache

    def __init__(
        self,
        data_dir: str | None = None,
        checkpoint: str = DEFAULT_CHECKPOINT,
        cache_dir: str | None = None,
        seed: int = 42,
    ):
        """Store the checkpoint, cache dir, and seed; embeddings load lazily on first use."""
        self._data_dir = data_dir
        self._checkpoint = checkpoint
        self._cache_dir = cache_dir
        self.seed = seed
        self._by_key: dict | None = None
        self._dim = 0

    def _resolve_cache_dir(self) -> Path:
        if self._cache_dir is not None:
            return Path(self._cache_dir)
        return Path("results") / "wbm_embeddings" / "from_raw"

    def _ensure_embeddings(self) -> None:
        if self._by_key is not None:
            return
        cache = self._resolve_cache_dir()
        if not (cache / "embeddings.npy").exists():
            logger.info("WBM embeddings cache miss at %s — extracting from raw (GPU)", cache)
            extract_wbm_embeddings(
                output_dir=str(cache),
                checkpoint=self._checkpoint,
                data_dir=self._data_dir,
                seed=self.seed,
            )
        emb = np.load(cache / "embeddings.npy")
        uids = np.load(cache / "user_ids.npy", allow_pickle=True)
        weeks = np.load(cache / "week_starts.npy", allow_pickle=True)
        self._by_key = {(str(u), str(w)): e for u, w, e in zip(uids, weeks, emb)}
        self._dim = int(emb.shape[1])
        logger.info("WBM embeddings ready: %d segments, dim=%d", len(emb), self._dim)

    def encode_cohort(self, task: str, td) -> np.ndarray:
        """Per-user mean-pool of the WBM embeddings over each user's eligible weeks."""
        self._ensure_embeddings()
        X = np.zeros((len(td.user_ids), self._dim), dtype=np.float32)
        for i, (uid, weeks) in enumerate(zip(td.user_ids, td.dates)):
            vecs = [self._by_key[k] for w in weeks if (k := (str(uid), str(w))) in self._by_key]
            if vecs:
                X[i] = np.mean(vecs, axis=0)
        return X


def _main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Stage-1 WBM embedding extraction (from raw, GPU).")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--data-dir", default=None, help="dataset root (else MHC_DATA_DIR)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    extract_wbm_embeddings(
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    _main()
