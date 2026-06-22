"""Shared normalization utilities for per-channel z-scoring.

Provides a framework-agnostic ChannelStats dataclass, functions to compute
per-channel mean/std from HuggingFace datasets or DataLoaders, and a loader
for the canonical global normalization stats file.

The canonical stats are pre-computed by ``scripts/build_normalization_stats.py``
and stored at ``data/processed/normalization_stats.json``. All daily-data
consumers (MAE, PyPOTS, imputation eval) load from this file via
``load_global_normalization_stats()`` rather than recomputing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

N_CHANNELS = 19
HR_CHANNEL = 5
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATS_PATH = "data/processed/normalization_stats.json"


@dataclass(frozen=True)
class ChannelStats:
    """Per-channel normalization statistics (mean/std).

    Holds numpy arrays of shape (N_CHANNELS,) with identity defaults
    (mean=0, std=1) for channels not explicitly computed.

    Attributes:
        means: Per-channel means, shape (N_CHANNELS,).
        stds: Per-channel stds, shape (N_CHANNELS,).
        channels: Tuple of channel indices that were computed.
        epsilon: Small constant for numerical stability.
    """

    means: np.ndarray
    stds: np.ndarray
    channels: tuple[int, ...]
    epsilon: float = 1e-8

    def normalize_numpy(self, x: np.ndarray) -> np.ndarray:
        """Normalize array in-place along channel dimension.

        Args:
            x: Array with a channel dimension matching self.channels.
                Supports shapes (..., C, T) where C indexes channels.

        Returns:
            Normalized copy of x (only computed channels are modified).
        """
        out = x.copy()
        for ch in self.channels:
            m = self.means[ch]
            s = self.stds[ch]
            out[..., ch, :] = (out[..., ch, :] - m) / (s + self.epsilon)
        return out

    def denormalize_numpy(self, z: np.ndarray) -> np.ndarray:
        """Reverse normalization (inverse of normalize_numpy).

        Args:
            z: Normalized array with same shape conventions as normalize_numpy.

        Returns:
            Denormalized copy of z.
        """
        out = z.copy()
        for ch in self.channels:
            m = self.means[ch]
            s = self.stds[ch]
            out[..., ch, :] = out[..., ch, :] * (s + self.epsilon) + m
        return out

    def to_torch(self) -> tuple:
        """Convert to torch tensors for HybridNaNAwareNormalize.

        Returns:
            (mean_prior, std_prior) as torch.Tensor of shape (N_CHANNELS,).
        """
        import torch

        return (
            torch.from_numpy(self.means.copy()).float(),
            torch.from_numpy(self.stds.copy()).float(),
        )

    def save(self, path: str | Path) -> None:
        """Save stats to JSON file (human-readable, no torch dependency).

        Args:
            path: Output JSON file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "means": self.means.tolist(),
            "stds": self.stds.tolist(),
            "channels": list(self.channels),
            "epsilon": self.epsilon,
        }
        path.write_text(json.dumps(data, indent=2))
        logger.info(f"Saved normalization stats to {path}")

    @classmethod
    def load(cls, path: str | Path) -> ChannelStats:
        """Load stats from JSON file.

        Args:
            path: Path to JSON file written by save().

        Returns:
            Loaded ChannelStats instance.
        """
        path = Path(path)
        data = json.loads(path.read_text())
        return cls(
            means=np.array(data["means"], dtype=np.float64),
            stds=np.array(data["stds"], dtype=np.float64),
            channels=tuple(data["channels"]),
            epsilon=data.get("epsilon", 1e-8),
        )


def load_global_normalization_stats(
    path: str | Path | None = None,
) -> ChannelStats:
    """Load the canonical global normalization statistics.

    Args:
        path: Override path. Defaults to ``data/processed/normalization_stats.json``
            resolved relative to the repository root.

    Returns:
        Loaded ChannelStats instance.

    Raises:
        FileNotFoundError: If the stats file does not exist.
    """
    if path is None:
        resolved = REPO_ROOT / DEFAULT_STATS_PATH
    else:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = REPO_ROOT / resolved

    if not resolved.exists():
        raise FileNotFoundError(
            f"Global normalization stats not found at {resolved}. "
            "Run 'python scripts/build_normalization_stats.py' to generate them."
        )
    return ChannelStats.load(resolved)


def compute_channel_stats(
    hf_dataset,
    channels: list[int] | None = None,
    max_samples: int | None = 10000,
    hr_channel: int = HR_CHANNEL,
) -> ChannelStats:
    """Compute per-channel mean/std from a HuggingFace daily dataset.

    NaN-aware computation that excludes NaN values. For the heart rate channel,
    zeros are also excluded (physiologically invalid).

    This is the shared implementation previously in _compute_daily_stats.

    Args:
        hf_dataset: HF dataset with 'values' column of shape (19, 1440).
        channels: Channel indices to compute stats for. Defaults to [0..6].
        max_samples: Max samples to use for speed. None = use all.
        hr_channel: Heart rate channel index (zeros excluded).

    Returns:
        ChannelStats with full (N_CHANNELS,) arrays; non-computed channels
        get identity defaults (mean=0, std=1).
    """
    if channels is None:
        channels = list(range(7))

    sums = np.zeros(len(channels), dtype=np.float64)
    sq_sums = np.zeros(len(channels), dtype=np.float64)
    counts = np.zeros(len(channels), dtype=np.float64)

    n = len(hf_dataset)
    if max_samples:
        n = min(n, max_samples)

    for i in tqdm(range(n), desc="Computing normalization stats"):
        ex = hf_dataset[i]
        values = np.asarray(ex["values"], dtype=np.float32)  # (19, 1440)

        for j, ch in enumerate(channels):
            v = values[ch]
            if ch == hr_channel:
                valid = (v != 0) & ~np.isnan(v)
            else:
                valid = ~np.isnan(v)

            if not np.any(valid):
                continue

            v_valid = v[valid]
            sums[j] += np.sum(v_valid)
            sq_sums[j] += np.sum(v_valid**2)
            counts[j] += len(v_valid)

    # Build full arrays with identity defaults
    means = np.zeros(N_CHANNELS, dtype=np.float64)
    stds = np.ones(N_CHANNELS, dtype=np.float64)

    for j, ch in enumerate(channels):
        if counts[j] > 0:
            mu = sums[j] / counts[j]
            var = (sq_sums[j] / counts[j]) - (mu**2)
            var = max(0.0, var)
            sigma = np.sqrt(var)
            means[ch] = float(mu)
            stds[ch] = float(sigma) if sigma > 1e-6 else 1.0
        # else: keeps identity defaults (0, 1)

    return ChannelStats(
        means=means,
        stds=stds,
        channels=tuple(channels),
    )


def compute_channel_stats_hourly(
    hf_dataset,
    channels: list[int] | None = None,
    max_samples: int | None = None,
    hr_channel: int = HR_CHANNEL,
) -> ChannelStats:
    """Compute per-channel mean/std from a daily_hourly_hf dataset.

    Uses the ``mask`` column (1=missing, 0=observed) to exclude missing values.
    For the heart rate channel, observed zeros are also excluded
    (physiologically invalid).

    Args:
        hf_dataset: HF dataset with ``values`` (19, 24) and ``mask`` (19, 24).
        channels: Channel indices to compute stats for. Defaults to [0..6].
        max_samples: Max samples to use. None = use all.
        hr_channel: Heart rate channel index (zeros excluded).

    Returns:
        ChannelStats with full (N_CHANNELS,) arrays; non-computed channels
        get identity defaults (mean=0, std=1).
    """
    if channels is None:
        channels = list(range(7))

    sums = np.zeros(len(channels), dtype=np.float64)
    sq_sums = np.zeros(len(channels), dtype=np.float64)
    counts = np.zeros(len(channels), dtype=np.float64)

    n = len(hf_dataset)
    if max_samples:
        n = min(n, max_samples)

    for i in tqdm(range(n), desc="Computing hourly normalization stats"):
        ex = hf_dataset[i]
        values = np.asarray(ex["values"], dtype=np.float32)  # (19, 24)
        mask = np.asarray(ex["mask"], dtype=np.float32)  # (19, 24), 1=missing

        for j, ch in enumerate(channels):
            v = values[ch]
            m = mask[ch]
            # observed = mask is 0
            valid = m == 0
            if ch == hr_channel:
                valid = valid & (v != 0)

            if not np.any(valid):
                continue

            v_valid = v[valid]
            sums[j] += np.sum(v_valid)
            sq_sums[j] += np.sum(v_valid**2)
            counts[j] += len(v_valid)

    # Build full arrays with identity defaults
    means = np.zeros(N_CHANNELS, dtype=np.float64)
    stds = np.ones(N_CHANNELS, dtype=np.float64)

    for j, ch in enumerate(channels):
        if counts[j] > 0:
            mu = sums[j] / counts[j]
            var = (sq_sums[j] / counts[j]) - (mu**2)
            var = max(0.0, var)
            sigma = np.sqrt(var)
            means[ch] = float(mu)
            stds[ch] = float(sigma) if sigma > 1e-6 else 1.0

    return ChannelStats(
        means=means,
        stds=stds,
        channels=tuple(channels),
    )


def compute_channel_stats_from_loader(
    train_loader,
    channels: list[int] | None = None,
) -> ChannelStats:
    """Compute per-channel mean/std from a DataLoader.

    Alternative entry point for DataLoader-based computation. Expects batches
    of (data, masks) where data is (B, C, T) and masks is (B, C, T).

    Args:
        train_loader: DataLoader yielding (data, mask) batches.
        channels: Channel indices to compute. Defaults to all N_CHANNELS.

    Returns:
        ChannelStats with computed means/stds.
    """
    if channels is None:
        channels = list(range(N_CHANNELS))

    channel_sums = np.zeros(N_CHANNELS, dtype=np.float64)
    channel_sq_sums = np.zeros(N_CHANNELS, dtype=np.float64)
    channel_counts = np.zeros(N_CHANNELS, dtype=np.float64)

    for data, masks in train_loader:
        data_np = data.numpy()
        masks_np = masks.numpy()

        valid = (masks_np == 1) & np.isfinite(data_np)
        data_masked = np.where(valid, data_np, 0.0)

        # Sum over batch and time dims: (B, C, T) -> (C,)
        channel_sums += data_masked.sum(axis=(0, 2))
        channel_sq_sums += (data_masked**2).sum(axis=(0, 2))
        channel_counts += valid.sum(axis=(0, 2))

    means = np.zeros(N_CHANNELS, dtype=np.float64)
    stds = np.ones(N_CHANNELS, dtype=np.float64)

    for ch in channels:
        if channel_counts[ch] > 0:
            mu = channel_sums[ch] / channel_counts[ch]
            var = (channel_sq_sums[ch] / channel_counts[ch]) - mu**2
            var = max(0.0, var)
            sigma = np.sqrt(var)
            means[ch] = float(mu)
            stds[ch] = float(sigma) if sigma > 1e-6 else 1.0

    return ChannelStats(
        means=means,
        stds=stds,
        channels=tuple(channels),
    )
