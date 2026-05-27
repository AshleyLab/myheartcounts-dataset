"""Personalized (per-user) imputers."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from openmhc.imputers._personalized_base import PersonalizedImputerBase


class PersonalizedMeanImputer(PersonalizedImputerBase):
    """Per-user, per-channel mean imputation.

    Each user's own per-channel mean (computed from their observed data
    in the eval splits) fills their masked positions. Channels with
    zero observations for a user fall back to the global channel mean
    from training.
    """

    name = "personalized_mean"

    def _compute_global_fallback(self) -> np.ndarray:
        return self.compute_channel_means()

    def _init_user_accumulator(self) -> dict:
        return {
            "sums": np.zeros(self.n_channels, dtype=np.float64),
            "counts": np.zeros(self.n_channels, dtype=np.float64),
        }

    def _update_user_accumulator(
        self, acc: dict, sample_data: np.ndarray, sample_mask: np.ndarray
    ) -> None:
        valid = (sample_mask > 0.5) & np.isfinite(sample_data)
        data_masked = np.where(valid, sample_data, 0.0)
        acc["sums"] += data_masked.sum(axis=1)
        acc["counts"] += valid.sum(axis=1)

    def _finalize_user_fill_values(
        self, acc: dict, global_fallback: np.ndarray
    ) -> np.ndarray:
        has_obs = acc["counts"] > 0
        return np.where(
            has_obs,
            acc["sums"] / np.maximum(acc["counts"], 1),
            global_fallback,
        ).astype(np.float32)

    def _apply_fill(
        self,
        result: np.ndarray,
        target_mask: np.ndarray,
        fill_values: np.ndarray,
        sample_idx: int,
    ) -> None:
        for ch in range(self.n_channels):
            target = target_mask[sample_idx, ch, :] == 1
            result[sample_idx, ch, target] = fill_values[ch]


class PersonalizedModeImputer(PersonalizedImputerBase):
    """Per-user, per-channel mode imputation.

    Each user's most frequent observed value (after rounding to
    ``decimal_precision`` places) per channel fills their masked
    positions. Channels with zero observations for a user fall back to
    the global channel mode from training.
    """

    name = "personalized_mode"

    def __init__(
        self,
        decimal_precision: int = 1,
        data_dir: str | Path | None = None,
    ) -> None:
        self.decimal_precision = decimal_precision
        super().__init__(data_dir=data_dir)

    def _compute_global_fallback(self) -> np.ndarray:
        # Reuse the same Counter-based computation as ModeImputer.
        counters: list[Counter] = [Counter() for _ in range(self.n_channels)]
        for data, mask in self.iter_train_batches():
            valid = (mask > 0.5) & np.isfinite(data)
            for ch in range(self.n_channels):
                ch_valid = valid[:, ch, :]
                if not ch_valid.any():
                    continue
                values = data[:, ch, :][ch_valid]
                rounded = np.round(values, self.decimal_precision)
                counters[ch].update(rounded.tolist())
        modes = np.zeros(self.n_channels, dtype=np.float32)
        for ch, counter in enumerate(counters):
            if counter:
                modes[ch] = counter.most_common(1)[0][0]
        return modes

    def _init_user_accumulator(self) -> list[Counter]:
        return [Counter() for _ in range(self.n_channels)]

    def _update_user_accumulator(
        self,
        acc: list[Counter],
        sample_data: np.ndarray,
        sample_mask: np.ndarray,
    ) -> None:
        valid = (sample_mask > 0.5) & np.isfinite(sample_data)
        for ch in range(self.n_channels):
            ch_valid = valid[ch, :]
            if not ch_valid.any():
                continue
            vals = sample_data[ch, :][ch_valid]
            rounded = np.round(vals, self.decimal_precision)
            acc[ch].update(rounded.tolist())

    def _finalize_user_fill_values(
        self, acc: list[Counter], global_fallback: np.ndarray
    ) -> np.ndarray:
        user_modes = global_fallback.copy()
        for ch in range(self.n_channels):
            if acc[ch]:
                user_modes[ch] = acc[ch].most_common(1)[0][0]
        return user_modes.astype(np.float32)

    def _apply_fill(
        self,
        result: np.ndarray,
        target_mask: np.ndarray,
        fill_values: np.ndarray,
        sample_idx: int,
    ) -> None:
        for ch in range(self.n_channels):
            target = target_mask[sample_idx, ch, :] == 1
            result[sample_idx, ch, target] = fill_values[ch]


class PersonalizedTemporalMeanImputer(PersonalizedImputerBase):
    """Per-user, per-channel, per-minute mean imputation.

    Fallback chain: per-user (channel, minute) mean → user's overall
    channel mean → global per-(channel, minute) mean from training.
    """

    name = "personalized_temporal_mean"

    def _compute_global_fallback(self) -> np.ndarray:
        return self.compute_temporal_means()  # (C, 1440)

    def _init_user_accumulator(self) -> dict:
        T = self.seq_len
        return {
            "minute_sums": np.zeros((self.n_channels, T), dtype=np.float64),
            "minute_counts": np.zeros((self.n_channels, T), dtype=np.float64),
            "ch_sums": np.zeros(self.n_channels, dtype=np.float64),
            "ch_counts": np.zeros(self.n_channels, dtype=np.float64),
        }

    def _update_user_accumulator(
        self, acc: dict, sample_data: np.ndarray, sample_mask: np.ndarray
    ) -> None:
        T = self.seq_len
        valid = (sample_mask > 0.5) & np.isfinite(sample_data)
        data_masked = np.where(valid, sample_data, 0.0)
        acc["ch_sums"] += data_masked.sum(axis=1)
        acc["ch_counts"] += valid.sum(axis=1)
        full_T = sample_data.shape[1]
        for k in range(full_T // T):
            s = k * T
            acc["minute_sums"] += data_masked[:, s : s + T]
            acc["minute_counts"] += valid[:, s : s + T]

    def _finalize_user_fill_values(
        self, acc: dict, global_fallback: np.ndarray
    ) -> np.ndarray:
        has_channel_obs = acc["ch_counts"] > 0
        user_channel_means = np.where(
            has_channel_obs,
            acc["ch_sums"] / np.maximum(acc["ch_counts"], 1),
            0.0,
        )
        has_minute_obs = acc["minute_counts"] > 0
        user_temporal = np.where(
            has_minute_obs,
            acc["minute_sums"] / np.maximum(acc["minute_counts"], 1),
            user_channel_means[:, None],
        )
        for ch in range(self.n_channels):
            if not has_channel_obs[ch]:
                user_temporal[ch, :] = global_fallback[ch, :]
        return user_temporal.astype(np.float32)

    def _apply_fill(
        self,
        result: np.ndarray,
        target_mask: np.ndarray,
        fill_values: np.ndarray,
        sample_idx: int,
    ) -> None:
        T = result.shape[2]
        n_repeats = max(T // self.seq_len, 1)
        tiled = np.tile(fill_values, (1, n_repeats))
        mask = target_mask[sample_idx] == 1
        result[sample_idx][mask] = tiled[mask]
