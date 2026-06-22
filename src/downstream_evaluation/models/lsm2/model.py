"""LSM2 daily encoder (masked-autoencoder 1D Vision Transformer).

Two stages, both run from raw data:

  - **Stage 1 (extraction, GPU):** ``extract_lsm2_embeddings`` loads the pretrained
    LSM2 checkpoint and runs the *dense* encoder (no artificial masking) over the
    minute-level daily tensors in ``daily_hf``, pooling the non-masked tokens per day
    into one 384-d embedding per (user, day). This regenerates the embeddings from raw
    data rather than loading a prebuilt cache. The ViT forward pass requires a GPU.

  - **Stage 2 (eval):** the ``LSM2`` ``Method`` loads that intermediate, mean-pools
    each user's eligible daily embeddings (the cohort's ``dates``) → 384-d, and runs
    the uniform PCA-50 probe (``openmhc.LinearProbe``) in ``fit`` / ``predict``.

The per-day embedding is small (384-d), so — unlike MultiRocket's per-(task, user)
pooling — we cache one vector per (user, day) and pool per cohort, exactly mirroring
``wbm.py`` (only the granularity differs: daily here, weekly there).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

N_CHANNELS = 19
N_CONTINUOUS = 7  # channels 0–6 are z-scored by the global priors; 7–18 pass through
EMBED_DIM = 384
BATCH_SIZE = 64
# min_wear_fraction=0.5 → keep days with at most (1-0.5)*1440 = 720 nonwear minutes.
MAX_NONWEAR_MINUTES = 720
# Public default: the LSM2 daily checkpoint published on the Hugging Face Hub, so a
# fresh user with no W&B account / no local copy can still fetch it. Override with a
# local .ckpt path (or any wandb:/hf:// ref) via the LSM2_CHECKPOINT env var.
DEFAULT_CHECKPOINT = os.environ.get("LSM2_CHECKPOINT", "hf://MyHeartCounts/openmhc-lsm2-daily")


# --------------------------------------------------------------------------- #
# Stage-1 helpers: model load and input transforms.
# --------------------------------------------------------------------------- #
def _load_lsm2_model(checkpoint: str, device):
    """Load + freeze the LSM2 encoder from a Lightning checkpoint.

    ``LSM2Module.load_from_checkpoint`` rebuilds the architecture from the
    checkpoint's saved hyperparameters and loads the weights; ``.model`` is the
    ``LSM2ViT1D`` encoder. Frozen + eval for inference-only feature extraction.
    """
    from utils.checkpoints import resolve_checkpoint_path

    from openmhc.models.lsm2.modules import LSM2Module

    resolved = resolve_checkpoint_path(checkpoint)
    logger.info("loading LSM2 checkpoint: %s", resolved)
    model = LSM2Module.load_from_checkpoint(str(resolved), map_location=device).model
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _build_transforms(stats_path: Path):
    """ZeroToNaN → hybrid global normalize (channels 0–6 only, pure global prior).

    Priors are channels 0–6 of the **minute-level** ``normalization_stats.json``
    (not the hourly file). ``prior_count=1e12`` selects the pure-global branch, so
    the 7 continuous channels are z-scored by the priors and 7–18 pass through.
    """
    import torch

    from data.normalization import load_global_normalization_stats
    # public data/transforms/__init__.py is empty → import the submodule directly.
    from data.transforms.nan_transforms import HybridNaNAwareNormalize, ZeroToNaNTransform

    channel_stats = load_global_normalization_stats(stats_path)
    mean_prior = torch.zeros(N_CHANNELS)
    std_prior = torch.ones(N_CHANNELS)
    for ch in range(N_CONTINUOUS):
        mean_prior[ch] = float(channel_stats.means[ch])
        std_prior[ch] = float(channel_stats.stds[ch])

    return [
        ZeroToNaNTransform(),
        HybridNaNAwareNormalize(
            mean_prior=mean_prior,
            std_prior=std_prior,
            channels=list(range(N_CONTINUOUS)),
            prior_count=1e12,
        ),
    ]


# --------------------------------------------------------------------------- #
# Stage 1 — extraction (GPU job): raw daily_hf -> per-(user, day) 384-d
# --------------------------------------------------------------------------- #
def extract_lsm2_embeddings(
    output_dir: str,
    checkpoint: str = DEFAULT_CHECKPOINT,
    data_dir: str | None = None,
    batch_size: int = BATCH_SIZE,
    seed: int = 42,
    loader=None,
) -> None:
    """Regenerate the LSM2 day-embedding intermediate from raw (GPU). Writes
    ``embeddings.npy`` (N_days, 384) / ``user_ids.npy`` / ``dates.npy`` under
    ``output_dir`` — one pooled row per eligible (user, day).

    Eligible days come from the provider's daily lookup (the single eligibility source,
    not a re-derived wear/variance filter); the raw minute tensors are fetched through the
    shared minute-resolution :class:`DataLoader` (``loader``; one is built if not injected).
    """
    from collections import defaultdict

    import pandas as pd
    import torch
    from tqdm import tqdm

    from openmhc._evaluate import _DatasetPaths

    from downstream_evaluation.data.provider import lookup_filename
    from downstream_evaluation.data.splits import load_split_file
    from openmhc.models.lsm2.utils import create_inherited_mask

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("LSM2 extraction device=%s", device)

    paths = _DatasetPaths.from_root(data_dir)
    model = _load_lsm2_model(checkpoint, device)
    patch_size = model.patch_size
    transforms = _build_transforms(paths.daily_hf.parent / "normalization_stats.json")

    # Eligible (user, day) = the daily lookup's valid days, restricted to the split-cohort
    # universe (every per-task cohort ⊆ this). Eligibility is the provider's lookup — the
    # single source of truth — not a re-derived wear/variance filter; the raw minute tensors
    # are fetched lazily from daily_hf through the shared DataLoader.
    split_users = load_split_file(paths.splits_file)
    cohort_users: set[str] = set()
    for users in split_users.values():
        cohort_users |= {str(u) for u in users}

    lookup_path = Path(paths.root) / "processed" / lookup_filename("daily", full_history=True)
    lk = pd.read_parquet(lookup_path, columns=["user_id", "date"])
    eligible_by_user: dict[str, list[str]] = defaultdict(list)
    for u, d in zip(lk["user_id"].astype(str), lk["date"].astype(str)):
        if u in cohort_users:
            eligible_by_user[u].append(d[:10])
    logger.info(
        "LSM2 eligible days (daily lookup ∩ cohort): %d users, %d (user,day) cells",
        len(eligible_by_user), sum(len(v) for v in eligible_by_user.values()),
    )

    if loader is None:  # standalone use; run_eval injects the shared minute loader
        from downstream_evaluation.data.loader import DataLoader

        loader = DataLoader(data_dir, resolution="minute")

    emb_list: list[np.ndarray] = []
    uid_list: list[str] = []
    date_list: list[str] = []

    with torch.no_grad():
        for uid in tqdm(sorted(eligible_by_user), desc="LSM2 encode", unit="user"):
            day_values, day_dates = loader.participant_minute(uid, eligible_by_user[uid])
            for start in range(0, len(day_values), batch_size):
                batch_v = day_values[start : start + batch_size]
                batch_d = day_dates[start : start + batch_size]
                batch_x = []
                for v in batch_v:
                    X = torch.as_tensor(v).float()  # (19, 1440)
                    for t in transforms:
                        X = t(X)
                    batch_x.append(X)
                Xb = torch.stack(batch_x).to(device)

                inherited = create_inherited_mask(Xb, patch_size=patch_size)
                latent, mask_out = model.forward_encoder_dense(Xb, inherited)
                latent_np = latent.cpu().float().numpy()  # (B, num_patches, 384)
                mask_np = mask_out.cpu().numpy()  # (B, num_patches), 1=masked

                for j, d in enumerate(batch_d):
                    observed = mask_np[j] == 0
                    if observed.sum() == 0:
                        continue
                    pooled = latent_np[j, observed, :].mean(axis=0).astype(np.float32)
                    emb_list.append(pooled)
                    uid_list.append(uid)
                    date_list.append(d)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    emb = (
        np.stack(emb_list).astype(np.float32)
        if emb_list
        else np.empty((0, EMBED_DIM), dtype=np.float32)
    )
    np.save(out / "embeddings.npy", emb)
    np.save(out / "user_ids.npy", np.array(uid_list, dtype=object))
    np.save(out / "dates.npy", np.array(date_list, dtype=object))
    logger.info("wrote %d LSM2 day-embeddings (dim=%d) -> %s", len(emb), EMBED_DIM, out)


# --------------------------------------------------------------------------- #
# Stage 2 — the LSM2 Encoder (driven by run_eval; build-on-miss is internal)
# --------------------------------------------------------------------------- #
class LSM2:
    """Unified ``Method``: LSM2 daily-encoder embeddings + the uniform linear probe.

    Per-user 384-d embeddings are built **on a cache miss** (dense encoder over raw,
    GPU), saved, and reused on a hit; ``fit`` / ``predict`` mean-pool each cohort
    participant's eligible day-embeddings and run :class:`openmhc.LinearProbe`.
    """

    name = "lsm2"
    input_granularity = "daily"  # per-user eligible days come from the daily lookup
    needs_segments = False  # consumes its own build-on-miss embedding cache
    segment_resolution = "minute"  # extraction reads the minute store (daily_hf) via the loader

    def __init__(
        self,
        data_dir: str | None = None,
        checkpoint: str = DEFAULT_CHECKPOINT,
        cache_dir: str | None = None,
        batch_size: int = BATCH_SIZE,
        seed: int = 42,
    ):
        self._data_dir = data_dir
        self._checkpoint = checkpoint
        self._cache_dir = cache_dir
        self._batch_size = batch_size
        self.seed = seed
        self._by_key: dict | None = None
        self._dim = EMBED_DIM
        self._probe = None
        self._ctx = None  # EvalContext (cohort user_ids + eligible dates), injected per call
        self._loader = None  # shared minute-resolution DataLoader, injected by run_eval

    def set_context(self, ctx) -> None:
        """Receive the per-(task, split) cohort context; the engine injects it before
        ``fit`` / ``predict``. LSM2 pools each user's cached day-embeddings over their
        eligible ``dates`` — neither carried by the clean ``Method`` signature."""
        self._ctx = ctx

    def set_loader(self, loader) -> None:
        """Receive the shared minute-resolution :class:`DataLoader`; cache-miss extraction
        fetches each user's eligible-day minute tensors from it (``run_eval`` injects it;
        standalone use builds its own)."""
        self._loader = loader

    def _resolve_cache_dir(self) -> Path:
        if self._cache_dir is not None:
            return Path(self._cache_dir)
        return Path("results") / "lsm2_embeddings" / "from_raw"

    def _ensure_embeddings(self) -> None:
        if self._by_key is not None:
            return
        cache = self._resolve_cache_dir()
        if not (cache / "embeddings.npy").exists():
            logger.info("LSM2 embeddings cache miss at %s — extracting from raw (GPU)", cache)
            extract_lsm2_embeddings(
                output_dir=str(cache),
                checkpoint=self._checkpoint,
                data_dir=self._data_dir,
                batch_size=self._batch_size,
                seed=self.seed,
                loader=self._loader,
            )
        emb = np.load(cache / "embeddings.npy")
        uids = np.load(cache / "user_ids.npy", allow_pickle=True)
        dates = np.load(cache / "dates.npy", allow_pickle=True)
        self._by_key = {(str(u), str(d)[:10]): e for u, d, e in zip(uids, dates, emb)}
        if emb.size:
            self._dim = int(emb.shape[1])
        logger.info("LSM2 embeddings ready: %d day-embeddings, dim=%d", len(emb), self._dim)

    def _encode(self, user_ids, dates) -> np.ndarray:
        """Per-user mean-pool of the cached LSM2 day-embeddings over each user's eligible days."""
        self._ensure_embeddings()
        X = np.zeros((len(user_ids), self._dim), dtype=np.float32)
        for i, (uid, ds) in enumerate(zip(user_ids, dates)):
            vecs = [self._by_key[k] for d in ds if (k := (str(uid), str(d)[:10])) in self._by_key]
            if vecs:
                X[i] = np.mean(vecs, axis=0)
        return X

    def fit(self, data, labels, task_type) -> None:
        # ``data`` is unused: LSM2 self-serves cached day-embeddings, pooled per user
        # over the cohort's eligible dates from set_context.
        import openmhc

        X = self._encode(self._ctx.user_ids, self._ctx.dates)
        self._probe = openmhc.LinearProbe(task_type, seed=self.seed).fit(X, labels)

    def predict(self, data) -> np.ndarray:
        X = self._encode(self._ctx.user_ids, self._ctx.dates)
        return self._probe.predict(X)


def _main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Stage-1 LSM2 day-embedding extraction (from raw, GPU).")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--data-dir", default=None, help="dataset root (else MHC_DATA_DIR)")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    extract_lsm2_embeddings(
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    _main()
