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

**Lazy per-user state.** ``__init__`` opens the HF dataset once via
:func:`openmhc._data_utils.open_eval_user_context` and builds only a
per-user row index. No val+test pre-scan happens at init. Instead,
:meth:`impute` ensures the current batch's user is cached, lazily
streaming that one user's ``(data, mask)`` rows on first contact (a
quick "open-once, mmap, slice per user" loop, not a full
DataLoader pass). When ``impute`` sees a batch belonging to a
different user, it evicts the previous user's state before loading
the new one — so each forked worker holds at most one user's
accumulator + per-sample contributions at a time.

This avoids the OOM the eager scan triggered for
``PersonalizedTemporalMeanImputer`` under the 6-way fork pool
(64 GB × 6 = 384 GB before this change). With the lazy state the
peak per-worker RSS is bounded by the heaviest single user (~440 MB
for temporal_mean), comfortably under any worker count.

**Pair this with user-grouped batches.** Personalized imputers expose
``requires_user_grouped_batches = True`` so the eval data loader
delivers one user's samples per batch (multiple users packed up to
``batch_size`` is fine too — :meth:`impute` groups by user
internally). Without user-grouped batches, every batch boundary
forces an eviction-and-reload, which is correct but slow.

**LOSO fill-value selection in :meth:`impute`**:

1. If both ``user_ids[i]`` and ``sample_indices[i]`` are provided and
   the pair was discovered during this user's lazy build, fill with
   the leave-one-sample-out finalization (subtract this sample's
   contribution from its user's accumulator).
2. Else if ``user_ids[i]`` is known, fill with the pre-finalized
   per-user value. Non-LOSO; used for samples outside the official
   val/test scan (e.g. direct-API usage with custom data).
3. Else, fill with the global fallback.
"""

from __future__ import annotations

import abc
import logging
from pathlib import Path
from typing import Any

import numpy as np

from openmhc._data_utils import EvalUserContext, open_eval_user_context
from openmhc._dataset import Version
from openmhc.imputers._base import BaseImputer

logger = logging.getLogger(__name__)


class PersonalizedImputerBase(BaseImputer, abc.ABC):
    """Base class for per-user imputers with leave-one-sample-out fill.

    Class attribute ``requires_user_grouped_batches`` is read by the
    eval harness to flip its data loader into user-grouped batch order.
    Defaults to ``True``; non-personalized subclasses don't inherit
    this class, so it doesn't affect the wider ecosystem.
    """

    requires_user_grouped_batches: bool = True

    def __init__(
        self,
        version: Version,
        data_dir: str | Path | None = None,
    ) -> None:
        super().__init__(version=version, data_dir=data_dir)
        self._global_fallback: Any = self._compute_global_fallback()
        # Open the HF dataset once and build the per-user row index.
        # Memory-mapped Arrow → fork-safe; per-user indices are small dicts.
        self._user_ctx: EvalUserContext = self._open_user_context()
        # Lazy per-user state: cached for at most one user at a time.
        self._cached_user_id: str | None = None
        self._user_accumulators: dict[str, Any] = {}
        self._user_fill_values: dict[str, Any] = {}
        # LOSO bookkeeping: (user_id, split-local sample_idx) -> per-sample
        # contribution that can be subtracted from the user accumulator.
        self._sample_contributions: dict[tuple[str, int], Any] = {}

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
    def _init_sample_contribution(self, sample_data: np.ndarray, sample_mask: np.ndarray) -> Any:
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

    def _open_user_context(self) -> EvalUserContext:
        """Open the HF dataset and build the per-user row index.

        Wrapped in a method so tests can monkeypatch this single seam
        instead of patching out every helper the real path uses.
        """
        return open_eval_user_context(version=self._version, data_dir=self._data_dir)

    def _build_user_state(self, user_id: str) -> None:
        """Stream one user's samples and build their accumulator + contributions."""
        ctx = self._user_ctx
        split = ctx.user_to_split.get(user_id)
        if split is None:
            # Unknown user — leave caches empty; impute() will fall through
            # to the global fallback for this user.
            return
        acc = self._init_user_accumulator()
        # Per-sample contributions are keyed on the *split-local* sample_idx
        # the harness emits in ``sample_indices`` at impute time. Build a
        # hf_idx → split_local_idx lookup from the context's split-wide
        # HF index list — one dict per split per build, amortized cheap.
        split_hf = ctx.split_hf_indices[split]
        hf_to_split_local = {int(hf): i for i, hf in enumerate(split_hf)}

        user_hf_indices = ctx.user_to_hf_indices[user_id]
        for hf_idx, (sample_data, sample_mask) in zip(
            user_hf_indices, ctx.iter_user_samples(user_id), strict=True
        ):
            s_idx = hf_to_split_local[int(hf_idx)]
            self._update_user_accumulator(acc, sample_data, sample_mask)
            self._sample_contributions[(user_id, s_idx)] = self._init_sample_contribution(
                sample_data, sample_mask
            )

        self._user_accumulators[user_id] = acc
        # Pre-compute the non-LOSO fill value for the "user known, sample
        # not seen" fallback path. Cheap relative to the per-sample work.
        self._user_fill_values[user_id] = self._finalize_user_fill_values(
            acc, None, self._global_fallback
        )

    def _ensure_user_cached(self, user_id: str) -> None:
        """Make sure this user's state is loaded; evict the previous one."""
        if self._cached_user_id == user_id:
            return
        if self._cached_user_id is not None:
            self._evict_cached_user()
        self._build_user_state(user_id)
        self._cached_user_id = user_id

    def _evict_cached_user(self) -> None:
        """Drop the currently-cached user's state."""
        uid = self._cached_user_id
        if uid is None:
            return
        self._user_accumulators.pop(uid, None)
        self._user_fill_values.pop(uid, None)
        # Drop all per-sample contributions for this user.
        keys = [k for k in self._sample_contributions if k[0] == uid]
        for k in keys:
            self._sample_contributions.pop(k, None)
        self._cached_user_id = None

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
        """Per-sample dispatch on ``(user_ids, sample_indices)`` with LOSO."""
        result = data.copy()
        N = data.shape[0]

        # Group batch positions by user so we touch each user's cache once.
        user_to_positions: dict[str | None, list[int]] = {}
        for i in range(N):
            uid = user_ids[i] if user_ids is not None and i < len(user_ids) else None
            user_to_positions.setdefault(uid, []).append(i)

        for uid, positions in user_to_positions.items():
            if uid is not None:
                self._ensure_user_cached(uid)
            for i in positions:
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
