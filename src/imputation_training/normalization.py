"""Minimal channel-stats helper for training-time normalization.

Matches the schema of ``normalization_stats.json`` files OpenMHC ships
in its dataset cache (``<cache>/processed/normalization_stats.json``) so
the same JSON file can be reused at inference (via
:meth:`openmhc.imputers.pypots._PyPOTSImputerBase._load_stats`).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ChannelStats:
    """Per-channel z-score parameters."""

    means: np.ndarray  # shape (n_channels,)
    stds: np.ndarray  # shape (n_channels,)
    channels: tuple[int, ...]  # channel indices the stats apply to
    epsilon: float

    @classmethod
    def from_path(cls, path: str | Path) -> ChannelStats:
        """Load channel stats from a ``normalization_stats.json`` file.

        Args:
            path: Path to the JSON file holding ``means``, ``stds``, and
                ``channels`` (and an optional ``epsilon``).

        Returns:
            The parsed :class:`ChannelStats`.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"normalization_stats.json not found: {p}")
        raw = json.loads(p.read_text())
        return cls(
            means=np.asarray(raw["means"], dtype=np.float32),
            stds=np.asarray(raw["stds"], dtype=np.float32),
            channels=tuple(int(c) for c in raw["channels"]),
            epsilon=float(raw.get("epsilon", 1e-8)),
        )

    def normalize_numpy(self, x: np.ndarray) -> np.ndarray:
        """Z-score the configured channels in-place; returns the same array.

        Args:
            x: ``(B, C, T)``-shaped batch.

        Returns:
            A new ``(B, C, T)`` float32 array with the configured
            channels z-scored. Other channels are passed through
            unchanged (intentional — binary channels stay binary).
        """
        out = x.copy()
        for ch in self.channels:
            out[..., ch, :] = (out[..., ch, :] - self.means[ch]) / (self.stds[ch] + self.epsilon)
        return out

    def copy_to(self, dst: str | Path) -> Path:
        """Drop a copy of the underlying JSON next to a release bundle.

        Equivalent to writing the dataclass back out as JSON; uses the
        original file contents byte-for-byte when ``source_path`` is set.
        For the training pipeline we always load from a canonical file
        and copy that file, so the on-disk JSON exactly matches what
        inference will load.
        """
        dst = Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Round-trip through JSON so we don't depend on the caller having
        # kept the source path around.
        dst.write_text(
            json.dumps(
                {
                    "means": self.means.tolist(),
                    "stds": self.stds.tolist(),
                    "channels": list(self.channels),
                    "epsilon": self.epsilon,
                },
                indent=2,
            )
        )
        return dst


def derive_stats_path_from_daily_hf(daily_hf_dir: str | Path) -> Path:
    """Convention: ``<cache>/processed/daily_hf`` → ``<cache>/processed/normalization_stats.json``.

    This mirrors the layout :func:`openmhc.download_dataset` materializes.
    """
    p = Path(daily_hf_dir)
    if p.name != "daily_hf":
        raise ValueError(
            f"Expected a path ending in 'daily_hf'; got {p}. "
            "Pass normalization_stats explicitly if your layout differs."
        )
    return p.parent / "normalization_stats.json"


def copy_stats_file(src: str | Path, dst: str | Path) -> Path:
    """Byte-copy a stats JSON without round-tripping through Python."""
    src_p = Path(src)
    dst_p = Path(dst)
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_p, dst_p)
    return dst_p
