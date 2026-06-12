"""Shared infrastructure for personalized (per-user) imputation.

Subclasses provide five hooks:

- :meth:`_compute_global_fallback` — global fill values from the train
  split (used when a user has no observations for a channel).
- :meth:`_init_user_accumulator` — fresh per-user state.
- :meth:`_update_user_accumulator` — fold one sample into per-user state.
- :meth:`_init_sample_contribution` — capture one sample's contribution
  so it can be subtracted at impute-time for leave-one-sample-out (LOSO).
- :meth:`_finalize_user_fill_values` — turn per-user state into fill
  values, optionally subtracting a single sample's contribution (LOSO).
- :meth:`_apply_fill` — fill masked positions for a single sample.

The base scans the official val + test splits once in ``__init__`` to
build per-user accumulators and per-sample contributions, then dispatches
in :meth:`impute`. When ``user_ids`` *and* ``sample_indices`` are both
provided and the ``(user_id, sample_idx)`` pair was seen during the
build pass, the fill is computed leave-one-sample-out: the sample's own
contribution is subtracted from its user's stats before finalization.
This prevents the held-out cells (which the harness masks *after* the
imputer's init) from informing their own fill values, a leakage source
in the prior implementation. Unknown ``(user_id, sample_idx)`` pairs
fall back to the standard per-user fill; unknown users fall back to the
global fallback.
"""

from __future__ import annotations

import abc
import logging
from pathlib import Path
from typing import Any

import numpy as np

from openmhc._data_utils import iter_split_data
from openmhc._dataset import Version
from openmhc.imputers._base import BaseImputer

logger = logging.getLogger(__name__)


class PersonalizedImputerBase(BaseImputer, abc.ABC):
    """Base class for per-user imputers with leave-one-sample-out fill."""

    def __init__(
        self,
        version: Version,
        data_dir: str | Path | None = None,
    ) -> None:
        super().__init__(version=version, data_dir=data_dir)
        self._global_fallback: Any = self._compute_global_fallback()
        # Raw per-user accumulators (needed for LOSO subtraction at impute time).
        self._user_accumulators: dict[str, Any] = {}
        # Pre-finalized per-user fill values (non-LOSO; fallback when a sample
        # is not in our val/test scan, e.g. direct-API usage with custom data).
        self._user_fill_values: dict[str, Any] = {}
        # LOSO bookkeeping: (user_id, split-local sample_idx) -> per-sample
        # contribution that can be subtracted from the user accumulator.
        # The user-level canonical split (sharable_users_seed42_2026.json)
        # guarantees a user appears in only one split, so this key is unique.
        self._sample_contributions: dict[tuple[str, int], Any] = {}
        for split in ("val", "test"):
            self._build_user_fill_values_for_split(split)

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _compute_global_fallback(self) -> Any:
        """Compute global fill values from the train split."""

    @abc.abstractmethod
    def _init_user_accumulator(self) -> Any:
        """Create a fresh accumulator for a new user."""

    @abc.abstractmethod
    def _update_user_accumulator(
        self, acc: Any, sample_data: np.ndarray, sample_mask: np.ndarray
    ) -> None:
        """Fold one sample into the per-user accumulator (in-place).

        Args:
            acc: Accumulator returned by :meth:`_init_user_accumulator`.
            sample_data: Sensor data of shape ``(C, T)``.
            sample_mask: Binary mask of shape ``(C, T)``, ``1`` = valid.
        """

    @abc.abstractmethod
    def _init_sample_contribution(
        self, sample_data: np.ndarray, sample_mask: np.ndarray
    ) -> Any:
        """Capture the state needed to subtract this sample's contribution.

        Returned object is consumed by :meth:`_finalize_user_fill_values`
        (LOSO branch) and must mirror the per-sample contribution that
        :meth:`_update_user_accumulator` folded into the user accumulator.

        Args:
            sample_data: Sensor data of shape ``(C, T)``.
            sample_mask: Binary mask of shape ``(C, T)``, ``1`` = valid.
        """

    @abc.abstractmethod
    def _finalize_user_fill_values(
        self, acc: Any, sample_contrib: Any | None, global_fallback: Any
    ) -> Any:
        """Turn accumulated state into per-user fill values.

        When ``sample_contrib is None``, returns the standard finalization
        over the full per-user accumulator. When ``sample_contrib`` is
        supplied, returns the LOSO finalization: the per-sample
        contribution is subtracted from ``acc`` before finalizing.
        """

    @abc.abstractmethod
    def _apply_fill(
        self,
        result: np.ndarray,
        target_mask: np.ndarray,
        fill_values: Any,
        sample_idx: int,
    ) -> None:
        """Fill ``target_mask == 1`` positions for one sample (in-place)."""

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_user_fill_values_for_split(self, split: str) -> None:
        """Stream the split once, accumulate per-user state + per-sample contribs."""
        metadata = self.load_metadata(split)
        user_ids = [m["user_id"] for m in metadata]
        sample_idxs = [int(m["sample_idx"]) for m in metadata]
        sample_offset = 0
        for data_batch, mask_batch in iter_split_data(
            split, version=self._version, data_dir=self._data_dir
        ):
            B = data_batch.shape[0]
            for i in range(B):
                pos = sample_offset + i
                uid = user_ids[pos]
                s_idx = sample_idxs[pos]
                acc = self._user_accumulators.get(uid)
                if acc is None:
                    acc = self._init_user_accumulator()
                    self._user_accumulators[uid] = acc
                self._update_user_accumulator(acc, data_batch[i], mask_batch[i])
                # Per-sample contribution for LOSO subtraction.
                key = (uid, s_idx)
                # Defensive: a duplicate would indicate the user-level split
                # invariant was violated (same user in val and test).
                if key in self._sample_contributions:
                    logger.warning(
                        "Duplicate (user_id, sample_idx) %r across splits — "
                        "LOSO subtraction may be incorrect for this sample.",
                        key,
                    )
                self._sample_contributions[key] = self._init_sample_contribution(
                    data_batch[i], mask_batch[i]
                )
            sample_offset += B

        # Pre-compute non-LOSO fill values for users we just observed.
        # Don't overwrite a value already finalized from an earlier split
        # (shouldn't happen under user-level splits, but matches prior behavior).
        for uid, acc in self._user_accumulators.items():
            if uid not in self._user_fill_values:
                self._user_fill_values[uid] = self._finalize_user_fill_values(
                    acc, None, self._global_fallback
                )

    # ------------------------------------------------------------------
    # Imputer protocol
    # ------------------------------------------------------------------

    def impute(
        self,
        data: np.ndarray,
        observed_mask: np.ndarray,
        target_mask: np.ndarray,
        *,
        user_ids: list[str] | None = None,
        sample_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """Per-sample dispatch on ``(user_ids, sample_indices)`` with LOSO.

        Fill-value selection per sample ``i``:

        1. If both ``user_ids[i]`` and ``sample_indices[i]`` are provided
           and the pair was seen during the build pass, fill with the
           leave-one-sample-out finalization (subtract this sample's
           contribution from its user's accumulator).
        2. Else if ``user_ids[i]`` is known, fill with the pre-finalized
           per-user value. This path is non-LOSO; use it only for samples
           outside the official val/test scan (e.g. direct-API usage with
           custom data).
        3. Else, fill with the global fallback.
        """
        result = data.copy()
        N = data.shape[0]
        for i in range(N):
            uid = user_ids[i] if (user_ids is not None and i < len(user_ids)) else None
            s_idx: int | None = None
            if sample_indices is not None and i < len(sample_indices):
                s_idx = int(sample_indices[i])

            fill_values: Any = None
            if uid is not None and s_idx is not None:
                contrib = self._sample_contributions.get((uid, s_idx))
                if contrib is not None:
                    user_acc = self._user_accumulators[uid]
                    fill_values = self._finalize_user_fill_values(
                        user_acc, contrib, self._global_fallback
                    )
            if fill_values is None and uid is not None:
                fill_values = self._user_fill_values.get(uid)
            if fill_values is None:
                fill_values = self._global_fallback

            self._apply_fill(result, target_mask, fill_values, i)
        return result.astype(np.float32, copy=False)
