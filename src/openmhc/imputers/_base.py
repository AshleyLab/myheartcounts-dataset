"""Optional base class providing reusable bootstrapping for imputers.

Subclassing is not required — the public ``Imputer`` protocol is
duck-typed. Use this base when you want shared helpers for computing
training-set statistics and looking up per-user metadata.

Subclasses implement ``impute`` and typically call helpers in their
``__init__`` to fit themselves on the official train split.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterator, Literal

import numpy as np

from openmhc._data_utils import iter_train_data, load_sample_metadata
from openmhc._dataset import Version

N_CHANNELS = 19
SEQ_LEN = 1440


class BaseImputer:
    """Helpers for the common imputer setup tasks.

    Provides streaming computation of channel statistics and
    metadata-only access to the evaluation splits. Subclasses are free
    to use any combination of these helpers (or none).
    """

    n_channels: int = N_CHANNELS
    seq_len: int = SEQ_LEN

    def __init__(
        self,
        version: Version,
        data_dir: str | Path | None = None,
    ) -> None:
        """Store the dataset root and version used by helper methods.

        Args:
            version: ``"xs"`` or ``"full"``. Required — propagated to
                every internal ``iter_*`` / ``load_sample_metadata`` call
                so the dataset root's ``dataset_version.json`` marker
                can verify the imputer is being fit on the version the
                caller intended.
            data_dir: Override for the dataset root. If omitted,
                ``MHC_DATA_DIR`` must be set.
        """
        self._version: Version = version
        self._data_dir = data_dir

    # ------------------------------------------------------------------
    # Training data helpers
    # ------------------------------------------------------------------

    def iter_train_batches(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(data, mask)`` batches from the official train split.

        Each batch is ``(B, 19, 1440)`` float32. ``data`` has NaN at
        missing positions; ``mask`` is 1 where observed, 0 elsewhere.
        """
        return iter_train_data(version=self._version, data_dir=self._data_dir)

    def compute_channel_means(self) -> np.ndarray:
        """Per-channel mean over all observed positions in the train split.

        Returns:
            Shape ``(n_channels,)`` float32 array.
        """
        sums = np.zeros(self.n_channels, dtype=np.float64)
        counts = np.zeros(self.n_channels, dtype=np.float64)
        for data, mask in self.iter_train_batches():
            obs = (mask > 0.5) & np.isfinite(data)
            data_obs = np.where(obs, data, 0.0)
            sums += data_obs.sum(axis=(0, 2))
            counts += obs.sum(axis=(0, 2))
        means = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
        return means.astype(np.float32)

    def compute_channel_means_stds(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-channel mean and standard deviation in one streaming pass.

        Returns:
            Tuple ``(means, stds)`` of shape ``(n_channels,)`` float32
            arrays. ``stds`` are clipped to a small positive floor to
            avoid divide-by-zero in normalized metrics.
        """
        sums = np.zeros(self.n_channels, dtype=np.float64)
        sq_sums = np.zeros(self.n_channels, dtype=np.float64)
        counts = np.zeros(self.n_channels, dtype=np.float64)
        for data, mask in self.iter_train_batches():
            obs = (mask > 0.5) & np.isfinite(data)
            data_obs = np.where(obs, data, 0.0)
            sums += data_obs.sum(axis=(0, 2))
            sq_sums += (data_obs ** 2).sum(axis=(0, 2))
            counts += obs.sum(axis=(0, 2))
        safe = np.maximum(counts, 1)
        means = np.where(counts > 0, sums / safe, 0.0)
        variance = np.where(counts > 0, (sq_sums / safe) - means ** 2, 0.0)
        stds = np.sqrt(np.maximum(variance, 0.0))
        stds = np.where(counts > 1, stds, 1.0)
        stds = np.maximum(stds, 1e-6)
        return means.astype(np.float32), stds.astype(np.float32)

    def compute_temporal_means(self) -> np.ndarray:
        """Per-(channel, minute-of-day) mean over the train split.

        Folds observations by `t % seq_len`, so it works for both
        single-day and multi-day windows.

        Returns:
            Shape ``(n_channels, seq_len)`` float32 array. Channels
            with no observed values at a given minute are filled with
            the channel's overall mean (or 0 if the channel is empty).
        """
        T = self.seq_len
        sums = np.zeros((self.n_channels, T), dtype=np.float64)
        counts = np.zeros((self.n_channels, T), dtype=np.float64)
        ch_sums = np.zeros(self.n_channels, dtype=np.float64)
        ch_counts = np.zeros(self.n_channels, dtype=np.float64)
        for data, mask in self.iter_train_batches():
            obs = (mask > 0.5) & np.isfinite(data)
            data_obs = np.where(obs, data, 0.0)
            B, C, full_T = data_obs.shape
            n_folds = full_T // T
            for k in range(n_folds):
                s = k * T
                sums += data_obs[:, :, s : s + T].sum(axis=0)
                counts += obs[:, :, s : s + T].sum(axis=0)
            ch_sums += data_obs.sum(axis=(0, 2))
            ch_counts += obs.sum(axis=(0, 2))
        ch_means = np.where(ch_counts > 0, ch_sums / np.maximum(ch_counts, 1), 0.0)
        per_minute = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
        # Fill empty (channel, minute) cells with the channel mean.
        per_minute = np.where(
            np.isnan(per_minute),
            ch_means[:, None],
            per_minute,
        )
        return per_minute.astype(np.float32)

    # ------------------------------------------------------------------
    # Evaluation-split metadata helpers
    # ------------------------------------------------------------------

    def load_metadata(
        self, split: Literal["train", "val", "test"]
    ) -> list[dict]:
        """Return per-sample metadata for the requested split.

        Each entry is ``{"sample_idx": int, "user_id": str, "date": str}``.
        ``sample_idx`` is the split-local position that aligns with the
        ``sample_indices`` kwarg passed to ``impute``.
        """
        return load_sample_metadata(
            split, version=self._version, data_dir=self._data_dir
        )

    def build_user_index(
        self, split: Literal["train", "val", "test"]
    ) -> dict[str, list[int]]:
        """Group sample indices by user for one split.

        Returns:
            A dict mapping ``user_id`` to a list of split-local sample
            indices belonging to that user.
        """
        index: dict[str, list[int]] = defaultdict(list)
        for record in self.load_metadata(split):
            index[record["user_id"]].append(record["sample_idx"])
        return dict(index)

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    def impute(
        self,
        data: np.ndarray,
        observed_mask: np.ndarray,
        target_mask: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        raise NotImplementedError(
            "BaseImputer subclasses must implement `impute`."
        )
