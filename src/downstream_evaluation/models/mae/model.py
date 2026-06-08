"""MAE / LSM2 daily encoder (masked-autoencoder 1D Vision Transformer).

Two stages, both run from raw data:

  - **Stage 1 (extraction, GPU):** ``extract_mae_embeddings`` loads the pretrained
    LSM2 MAE checkpoint and runs the *dense* encoder (no artificial masking) over the
    minute-level daily tensors in ``daily_hf``, pooling the non-masked tokens per day
    into one 384-d embedding per (user, day). This regenerates the embeddings from raw
    data rather than loading a prebuilt cache. The ViT forward pass requires a GPU.

  - **Stage 2 (eval):** the ``MAE`` Encoder loads that intermediate and mean-pools
    each user's eligible daily embeddings (``td.dates``) → 384-d; the engine then runs
    the uniform PCA-50 + linear probe.

The per-day embedding is small (384-d), so — unlike MultiRocket's per-(task, user)
pooling — we cache one vector per (user, day) and pool per cohort in ``encode_cohort``,
exactly mirroring ``wbm.py`` (only the granularity differs: daily here, weekly there).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

N_CHANNELS = 19
N_CONTINUOUS = 7  # channels 0–6 are z-scored by the global priors; 7–18 pass through
EMBED_DIM = 384
PATCH_SIZE = 10
BATCH_SIZE = 64
# min_wear_fraction=0.5 → keep days with at most (1-0.5)*1440 = 720 nonwear minutes.
MAX_NONWEAR_MINUTES = 720
# Public default: the LSM2 daily checkpoint published on the Hugging Face Hub, so a
# fresh user with no W&B account / no local copy can still fetch it. Override with a
# local .ckpt path (or any wandb:/hf:// ref) via the MAE_CHECKPOINT env var.
DEFAULT_CHECKPOINT = "hf://MyHeartCounts/openmhc-lsm2-daily"


# --------------------------------------------------------------------------- #
# Stage-1 helpers: model load and input transforms.
# --------------------------------------------------------------------------- #
def _load_mae_model(checkpoint: str, device):
    """Load + freeze the LSM2 MAE from a Lightning checkpoint.

    Strips the ``model.`` prefix from the Lightning ``state_dict`` and rebuilds the
    architecture from the checkpoint's ``hyper_parameters`` (falling back to the
    pretrained dims). The dense encoder path used here needs no decoder, but the full
    module is loaded so ``load_state_dict`` is strict (a wrong dim fails loudly).
    """
    import torch

    from utils.checkpoints import resolve_checkpoint_path

    from downstream_evaluation.models.mae.mae_vit1d import MaskedAutoencoderViT1D_LSM2

    resolved = resolve_checkpoint_path(checkpoint)
    logger.info("loading MAE checkpoint: %s", resolved)
    ckpt = torch.load(resolved, map_location="cpu", weights_only=False)

    if "state_dict" in ckpt:
        state_dict = {
            k.removeprefix("model."): v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("model.")
        }
        hp = ckpt.get("hyper_parameters", {})
        model = MaskedAutoencoderViT1D_LSM2(
            seq_length=hp.get("seq_length", 1440),
            patch_size=hp.get("patch_size", PATCH_SIZE),
            in_channels=hp.get("in_channels", N_CHANNELS),
            embed_dim=hp.get("embed_dim", EMBED_DIM),
            depth=hp.get("depth", 12),
            num_heads=hp.get("num_heads", 6),
            decoder_embed_dim=hp.get("decoder_embed_dim", 256),
            decoder_depth=hp.get("decoder_depth", 4),
            decoder_num_heads=hp.get("decoder_num_heads", 4),
            mlp_ratio=hp.get("mlp_ratio", 4.0),
            qkv_bias=hp.get("qkv_bias", True),
            dropout_removal_ratio=hp.get("dropout_removal_ratio", 0.5),
            mask_ratio=hp.get("mask_ratio", 0.5),
        )
        model.load_state_dict(state_dict)
    else:
        model = MaskedAutoencoderViT1D_LSM2()
        model.load_state_dict(ckpt)

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
def extract_mae_embeddings(
    output_dir: str,
    checkpoint: str = DEFAULT_CHECKPOINT,
    data_dir: str | None = None,
    batch_size: int = BATCH_SIZE,
    seed: int = 42,
) -> None:
    """Regenerate the MAE day-embedding intermediate from raw (GPU). Writes
    ``embeddings.npy`` (N_days, 384) / ``user_ids.npy`` / ``dates.npy`` under
    ``output_dir`` — one pooled row per kept (user, day).
    """
    from collections import defaultdict

    import datasets as hf_ds
    import torch
    from tqdm import tqdm

    from openmhc._evaluate import _DatasetPaths

    from data.processing.hf_config import DEFAULT_VARIANCE_THRESHOLDS
    from downstream_evaluation.data.splits import load_split_file
    from downstream_evaluation.models.mae.utils import create_inherited_mask

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("MAE extraction device=%s", device)

    paths = _DatasetPaths.resolve(data_dir)
    model = _load_mae_model(checkpoint, device)
    patch_size = model.patch_size
    transforms = _build_transforms(paths.daily_hf.parent / "normalization_stats.json")

    # daily_hf: (19, 1440) channel-first minute tensors + per-day QC metadata.
    logger.info("loading daily_hf: %s", paths.daily_hf)
    ds = hf_ds.load_from_disk(str(paths.daily_hf))
    n = len(ds)

    # Cohort universe = union of all split users (every per-task cohort ⊆ this set),
    # so extracting over it covers every user encode_cohort can ask for, no more.
    split_users = load_split_file(paths.splits_file)
    cohort_users: set[str] = set()
    for users in split_users.values():
        cohort_users |= {str(u) for u in users}

    user_ids_arr = np.asarray(ds["user_id"], dtype=object).astype(str)
    dates_arr = np.asarray(ds["date"], dtype=object).astype(str)
    nonwear_arr = np.asarray(ds["total_nonwear_minutes"], dtype=np.float64)
    var_arr = np.asarray(ds["channel_variance"], dtype=np.float64)

    # Day filters: split membership, wear time, per-channel variance (NaN = keep).
    split_mask = np.array([u in cohort_users for u in user_ids_arr], dtype=bool)
    wear_mask = nonwear_arr <= MAX_NONWEAR_MINUTES
    var_mask = np.ones(n, dtype=bool)
    for ch, thresh in DEFAULT_VARIANCE_THRESHOLDS.items():
        if ch < var_arr.shape[1]:
            col = var_arr[:, ch]
            var_mask &= np.isnan(col) | (col >= thresh)
    keep = split_mask & wear_mask & var_mask
    kept = np.where(keep)[0]
    logger.info("daily_hf filtered: %d -> %d days (split+wear+variance)", n, len(kept))

    by_user: dict[str, list[int]] = defaultdict(list)
    for idx in kept:
        by_user[user_ids_arr[idx]].append(int(idx))

    # Read only the `values` column per row (avoid decoding the other big arrays).
    ds_vals = ds.select_columns(["values"])

    emb_list: list[np.ndarray] = []
    uid_list: list[str] = []
    date_list: list[str] = []

    with torch.no_grad():
        for uid, indices in tqdm(sorted(by_user.items()), desc="MAE encode", unit="user"):
            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start : start + batch_size]
                batch_x = []
                for idx in batch_idx:
                    X = torch.as_tensor(ds_vals[idx]["values"]).float()  # (19, 1440)
                    for t in transforms:
                        X = t(X)
                    batch_x.append(X)
                Xb = torch.stack(batch_x).to(device)

                inherited = create_inherited_mask(Xb, patch_size=patch_size)
                latent, mask_out = model.forward_encoder_dense(Xb, inherited)
                latent_np = latent.cpu().float().numpy()  # (B, num_patches, 384)
                mask_np = mask_out.cpu().numpy()  # (B, num_patches), 1=masked

                for j, idx in enumerate(batch_idx):
                    observed = mask_np[j] == 0
                    if observed.sum() == 0:
                        continue
                    pooled = latent_np[j, observed, :].mean(axis=0).astype(np.float32)
                    emb_list.append(pooled)
                    uid_list.append(uid)
                    date_list.append(dates_arr[idx][:10])

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
    logger.info("wrote %d MAE day-embeddings (dim=%d) -> %s", len(emb), EMBED_DIM, out)


# --------------------------------------------------------------------------- #
# Stage 2 — the MAE Encoder (driven by run_eval; build-on-miss is internal)
# --------------------------------------------------------------------------- #
class MAE:
    """MAE daily encoder for the engine. ``encode_cohort`` returns the cohort's
    per-user 384-d embeddings — built **on a cache miss** by running the dense
    encoder over raw (GPU), saved, and reused on a hit — all inside the eval flow.
    The engine then runs the uniform PCA-50 + linear probe.
    """

    name = "mae"
    input_granularity = "daily"  # per-user eligible days come from the daily lookup
    needs_segments = False  # consumes its own build-on-miss embedding cache

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

    def _resolve_cache_dir(self) -> Path:
        if self._cache_dir is not None:
            return Path(self._cache_dir)
        return Path("results") / "mae_embeddings" / "from_raw"

    def _ensure_embeddings(self) -> None:
        if self._by_key is not None:
            return
        cache = self._resolve_cache_dir()
        if not (cache / "embeddings.npy").exists():
            logger.info("MAE embeddings cache miss at %s — extracting from raw (GPU)", cache)
            extract_mae_embeddings(
                output_dir=str(cache),
                checkpoint=self._checkpoint,
                data_dir=self._data_dir,
                batch_size=self._batch_size,
                seed=self.seed,
            )
        emb = np.load(cache / "embeddings.npy")
        uids = np.load(cache / "user_ids.npy", allow_pickle=True)
        dates = np.load(cache / "dates.npy", allow_pickle=True)
        self._by_key = {(str(u), str(d)[:10]): e for u, d, e in zip(uids, dates, emb)}
        if emb.size:
            self._dim = int(emb.shape[1])
        logger.info("MAE embeddings ready: %d day-embeddings, dim=%d", len(emb), self._dim)

    def encode_cohort(self, task: str, td) -> np.ndarray:
        """Per-user mean-pool of the MAE day-embeddings over each user's eligible days."""
        self._ensure_embeddings()
        X = np.zeros((len(td.user_ids), self._dim), dtype=np.float32)
        for i, (uid, dates) in enumerate(zip(td.user_ids, td.dates)):
            vecs = [self._by_key[k] for d in dates if (k := (str(uid), str(d)[:10])) in self._by_key]
            if vecs:
                X[i] = np.mean(vecs, axis=0)
        return X


def _main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Stage-1 MAE day-embedding extraction (from raw, GPU).")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--data-dir", default=None, help="dataset root (else MHC_DATA_DIR)")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    extract_mae_embeddings(
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    _main()
