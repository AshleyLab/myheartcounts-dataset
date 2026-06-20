"""Personalized (per-user) imputers."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from openmhc._dataset import Version
from openmhc.imputers._personalized_base import PersonalizedImputerBase


class PersonalizedMeanImputer(PersonalizedImputerBase):
    """Per-user, per-channel mean imputation with leave-one-sample-out fill.

    Each user's own per-channel mean fills their masked positions. The
    mean is computed leave-one-sample-out across the user's samples in
    the eval split, so the held-out cells of the sample being scored
    cannot inform their own fill. Channels with zero observations for a
    user (after LOSO subtraction) fall back to the global channel mean
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

    def _init_sample_contribution(
        self, sample_data: np.ndarray, sample_mask: np.ndarray
    ) -> dict:
        # Mirror the per-sample addends in `_update_user_accumulator` so the
        # finalize step can subtract them exactly.
        valid = (sample_mask > 0.5) & np.isfinite(sample_data)
        data_masked = np.where(valid, sample_data, 0.0)
        return {
            "sums": data_masked.sum(axis=1).astype(np.float64),
            "counts": valid.sum(axis=1).astype(np.float64),
        }

    def _finalize_user_fill_values(
        self,
        acc: dict,
        sample_contrib: dict | None,
        global_fallback: np.ndarray,
    ) -> np.ndarray:
        if sample_contrib is None:
            sums = acc["sums"]
            counts = acc["counts"]
        else:
            sums = acc["sums"] - sample_contrib["sums"]
            counts = acc["counts"] - sample_contrib["counts"]
        has_obs = counts > 0
        return np.where(
            has_obs,
            sums / np.maximum(counts, 1),
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
    """Per-user, per-channel mode-of-per-sample-modes with LOSO fill.

    Each sample contributes its own per-channel mode — the most-frequent
    rounded value within that sample's observed cells. The user's fill at
    each channel is the mode across the user's per-sample modes (i.e.
    "mode of daily modes"), with the sample being scored excluded under
    LOSO. Channels with no defined per-sample mode for any of the user's
    remaining samples fall back to the global channel mode from training.

    Math change vs. the historical contract (which counted every observed
    rounded value across all of a user's samples): within-day frequency is
    collapsed to a single per-day vote per channel before the user-level
    aggregation. For binary channels this is nearly identical because the
    per-day mode equals the majority class anyway; for continuous channels
    the per-day vote is coarser but still well-defined, and the per-sample
    state shrinks from a list of ``Counter`` objects (up to MB per sample)
    to a ``(C,) float32`` array (≈76 B per sample), making the LOSO
    bookkeeping cheap. With a single-sample user, the LOSO exclusion
    empties the pool and the fill falls back to ``global_fallback``.
    """

    name = "personalized_mode"

    def __init__(
        self,
        version: Version,
        decimal_precision: int = 1,
        data_dir: str | Path | None = None,
    ) -> None:
        self.decimal_precision = decimal_precision
        super().__init__(version=version, data_dir=data_dir)

    def _compute_global_fallback(self) -> np.ndarray:
        # Train-split per-channel mode of all rounded observed values.
        # (This stays cell-pooled because train-split global stats don't
        # leak into eval-split LOSO; the per-user fill is the only place
        # the macro-of-per-sample-modes contract applies.)
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

    def _compute_sample_mode(
        self, sample_data: np.ndarray, sample_mask: np.ndarray
    ) -> np.ndarray:
        """Per-channel mode of this sample's rounded observed values.

        Returns ``(n_channels,) float32``; NaN at channels where the
        sample has no observed cells.
        """
        valid = (sample_mask > 0.5) & np.isfinite(sample_data)
        out = np.full(self.n_channels, np.nan, dtype=np.float32)
        for ch in range(self.n_channels):
            ch_valid = valid[ch, :]
            if not ch_valid.any():
                continue
            vals = sample_data[ch, :][ch_valid]
            rounded = np.round(vals, self.decimal_precision).astype(np.float32)
            # np.unique is sorted ascending; np.argmax on ties returns the
            # first (smallest) value — deterministic tie-break.
            unique, counts = np.unique(rounded, return_counts=True)
            out[ch] = float(unique[int(np.argmax(counts))])
        return out

    def _init_user_accumulator(self) -> list[np.ndarray]:
        # One ``(n_channels,) float32`` per sample, appended in order.
        return []

    def _update_user_accumulator(
        self,
        acc: list[np.ndarray],
        sample_data: np.ndarray,
        sample_mask: np.ndarray,
    ) -> None:
        acc.append(self._compute_sample_mode(sample_data, sample_mask))

    def _init_sample_contribution(
        self,
        sample_data: np.ndarray,
        sample_mask: np.ndarray,
    ) -> np.ndarray:
        # Same content as the row appended to ``acc``; equality matched in
        # ``_finalize_user_fill_values`` for LOSO.
        return self._compute_sample_mode(sample_data, sample_mask)

    def _finalize_user_fill_values(
        self,
        acc: list[np.ndarray],
        sample_contrib: np.ndarray | None,
        global_fallback: np.ndarray,
    ) -> np.ndarray:
        if sample_contrib is None:
            rows = acc
        else:
            # Drop one row matching sample_contrib by value (NaN-aware).
            # If no row matches (shouldn't happen under normal use), keep
            # the full set — better to over-include than spuriously drop.
            rows = []
            excluded = False
            for row in acc:
                if (
                    not excluded
                    and np.array_equal(row, sample_contrib, equal_nan=True)
                ):
                    excluded = True
                    continue
                rows.append(row)
            if not excluded:
                rows = acc

        if not rows:
            # Single-sample user under LOSO → no rows remain; fall back.
            return global_fallback.astype(np.float32, copy=True)

        stacked = np.stack(rows, axis=0)  # (N_remaining, n_channels)
        user_modes = global_fallback.astype(np.float32, copy=True)
        for ch in range(self.n_channels):
            col = stacked[:, ch]
            col = col[np.isfinite(col)]
            if col.size == 0:
                continue
            unique, counts = np.unique(col, return_counts=True)
            user_modes[ch] = float(unique[int(np.argmax(counts))])
        return user_modes

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
    """Per-user, per-channel, per-minute mean imputation with LOSO fill.

    Fallback chain: per-user (channel, minute) mean → user's overall
    channel mean → global per-(channel, minute) mean from training. All
    per-user means are computed leave-one-sample-out across the user's
    samples in the eval split.

    Memory note: this imputer stores a per-sample (channel, minute)
    contribution (`~C * T * 4` bytes per sample, ≈110 KB at C=19,
    T=1440 float32, plus a same-shape uint8 count mask) for every
    val/test sample. For ``version="full"`` this can total a few GB —
    acceptable for a CPU baseline.
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

    def _init_sample_contribution(
        self, sample_data: np.ndarray, sample_mask: np.ndarray
    ) -> dict:
        # Mirror the per-sample addends folded by `_update_user_accumulator`.
        T = self.seq_len
        valid = (sample_mask > 0.5) & np.isfinite(sample_data)
        data_masked = np.where(valid, sample_data, 0.0)
        full_T = sample_data.shape[1]
        minute_sums = np.zeros((self.n_channels, T), dtype=np.float32)
        minute_counts = np.zeros((self.n_channels, T), dtype=np.uint8)
        for k in range(full_T // T):
            s = k * T
            minute_sums += data_masked[:, s : s + T].astype(np.float32)
            minute_counts += valid[:, s : s + T].astype(np.uint8)
        return {
            "minute_sums": minute_sums,
            "minute_counts": minute_counts,
            "ch_sums": data_masked.sum(axis=1).astype(np.float64),
            "ch_counts": valid.sum(axis=1).astype(np.float64),
        }

    def _finalize_user_fill_values(
        self,
        acc: dict,
        sample_contrib: dict | None,
        global_fallback: np.ndarray,
    ) -> np.ndarray:
        if sample_contrib is None:
            minute_sums = acc["minute_sums"]
            minute_counts = acc["minute_counts"]
            ch_sums = acc["ch_sums"]
            ch_counts = acc["ch_counts"]
        else:
            minute_sums = acc["minute_sums"] - sample_contrib["minute_sums"]
            minute_counts = acc["minute_counts"] - sample_contrib["minute_counts"]
            ch_sums = acc["ch_sums"] - sample_contrib["ch_sums"]
            ch_counts = acc["ch_counts"] - sample_contrib["ch_counts"]
        has_channel_obs = ch_counts > 0
        user_channel_means = np.where(
            has_channel_obs,
            ch_sums / np.maximum(ch_counts, 1),
            0.0,
        )
        has_minute_obs = minute_counts > 0
        user_temporal = np.where(
            has_minute_obs,
            minute_sums / np.maximum(minute_counts, 1),
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
