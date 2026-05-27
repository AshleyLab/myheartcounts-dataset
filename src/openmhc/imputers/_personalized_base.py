"""Shared infrastructure for personalized (per-user) imputation.

Subclasses provide four hooks:

- :meth:`_compute_global_fallback` — global fill values from the train
  split (used when a user has no observations for a channel).
- :meth:`_init_user_accumulator` — fresh per-user state.
- :meth:`_update_user_accumulator` — fold one sample into per-user state.
- :meth:`_finalize_user_fill_values` — turn per-user state into fill values.
- :meth:`_apply_fill` — fill masked positions for a single sample.

The base scans the official val + test splits once in ``__init__`` to
build per-user fill values, then dispatches by ``user_ids`` in
``impute``. Unknown users fall back to the global fallback.
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
    """Base class for per-user imputers."""

    def __init__(
        self,
        version: Version,
        data_dir: str | Path | None = None,
    ) -> None:
        super().__init__(version=version, data_dir=data_dir)
        self._global_fallback: Any = self._compute_global_fallback()
        self._user_fill_values: dict[str, Any] = {}
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
    def _finalize_user_fill_values(self, acc: Any, global_fallback: Any) -> Any:
        """Turn accumulated state into per-user fill values."""

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
        """Stream the split once, accumulate per-user state, finalize."""
        metadata = self.load_metadata(split)
        user_ids = [m["user_id"] for m in metadata]
        accumulators: dict[str, Any] = {}
        sample_offset = 0
        for data_batch, mask_batch in iter_split_data(
            split, version=self._version, data_dir=self._data_dir
        ):
            B = data_batch.shape[0]
            for i in range(B):
                uid = user_ids[sample_offset + i]
                acc = accumulators.get(uid)
                if acc is None:
                    acc = self._init_user_accumulator()
                    accumulators[uid] = acc
                self._update_user_accumulator(acc, data_batch[i], mask_batch[i])
            sample_offset += B

        for uid, acc in accumulators.items():
            # Don't overwrite a finalized value from a previous split.
            if uid not in self._user_fill_values:
                self._user_fill_values[uid] = self._finalize_user_fill_values(
                    acc, self._global_fallback
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
    ) -> np.ndarray:
        """Per-sample dispatch on ``user_ids``, falling back to global."""
        result = data.copy()
        N = data.shape[0]
        for i in range(N):
            uid = None
            if user_ids is not None and i < len(user_ids):
                uid = user_ids[i]
            fill_values = self._user_fill_values.get(uid, self._global_fallback) \
                if uid is not None else self._global_fallback
            self._apply_fill(result, target_mask, fill_values, i)
        return result.astype(np.float32, copy=False)
