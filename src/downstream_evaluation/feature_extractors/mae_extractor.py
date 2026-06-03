"""MAE daily encoder feature extractor for downstream evaluation.

Loads pre-pooled per-day MAE embeddings (384-dim) from per-split HDF5 files and
aggregates them to user-level vectors with time-window-aware day filtering. MAE
features are daily-granularity, so the time window is applied at the day level
before each user's days are mean-pooled into a single vector.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_path(p: str | Path) -> Path:
    """Resolve a path relative to the repo root when it is not absolute."""
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p)


class MAEDailyExtractor:
    """Load pre-pooled daily MAE embeddings and aggregate them to user level.

    Stores per-day embeddings as ``{user_id: {date_str: (D,) ndarray}}``, then
    ``get_user_embeddings()`` filters each user's days by a time window and
    mean-pools them into a single user-level vector.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        pooled_embeddings_dir: str = "data/processed/mae_pooled_embeddings",
    ):
        """Initialize the extractor.

        Args:
            embed_dim: Dimensionality of the pooled per-day vectors.
            pooled_embeddings_dir: Directory holding per-split ``<split>.h5`` files.
        """
        self.embed_dim = embed_dim
        self.pooled_embeddings_dir = pooled_embeddings_dir
        # {user_id: {date_str: (D,) ndarray}} — populated by load_precomputed.
        self.user_day_embeddings: dict[str, dict[str, np.ndarray]] = {}

    def load_precomputed(self, splits: list[str] | None = None) -> None:
        """Load pre-pooled per-day embeddings from per-split HDF5 files.

        Expected layout::

            <pooled_embeddings_dir>/<split>.h5
              embeddings:  (N, D)  float32
              user_ids:    (N,)    string
              dates:       (N,)    string

        Args:
            splits: Which splits to load (default: train, validation, test).
        """
        base_dir = _resolve_path(self.pooled_embeddings_dir)
        if splits is None:
            splits = ["train", "validation", "test"]

        total = 0
        for split in splits:
            h5_path = base_dir / f"{split}.h5"
            if not h5_path.exists():
                logger.warning("Pooled embeddings not found: %s", h5_path)
                continue

            with h5py.File(h5_path, "r") as f:
                embeddings = f["embeddings"][:]
                user_ids = f["user_ids"][:].astype(str)
                dates = f["dates"][:].astype(str)

            for i in range(len(user_ids)):
                uid = user_ids[i]
                date = dates[i]
                if uid not in self.user_day_embeddings:
                    self.user_day_embeddings[uid] = {}
                self.user_day_embeddings[uid][date] = embeddings[i]

            total += len(user_ids)
            logger.info(
                "Loaded %d day-embeddings from %s (%d users)",
                len(user_ids),
                h5_path.name,
                len(set(user_ids)),
            )

        logger.info(
            "Precomputed MAE embeddings: %d total days, %d users",
            total,
            len(self.user_day_embeddings),
        )

    def get_user_embeddings(
        self,
        user_ids: set[str],
        clip_dates: dict[str, str] | None,
        time_window,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Aggregate per-day embeddings to user level with time-window filtering.

        For each user, select the days falling within the time window relative to
        the user's label date, then mean-pool them into a single ``(D,)`` vector.

        Args:
            user_ids: User IDs to include.
            clip_dates: ``{user_id: date_str}`` label dates for temporal filtering.
                Required when ``time_window`` is not full.
            time_window: ``TimeWindow`` controlling which days to include relative
                to the label date.

        Returns:
            X: ``(N_users, D)`` float32 feature matrix.
            uids: ``(N_users,)`` object array of user ID strings.
        """
        import pandas as pd

        X_list = []
        uid_list = []

        for uid in sorted(user_ids):
            day_embs = self.user_day_embeddings.get(uid)
            if not day_embs:
                continue

            if time_window.is_full or clip_dates is None:
                selected = list(day_embs.values())
            else:
                clip_date_str = clip_dates.get(uid)
                if clip_date_str is None:
                    # No clip date → skip user for non-full windows.
                    continue

                clip_ts = pd.Timestamp(clip_date_str)
                selected = []
                for date_str, emb in day_embs.items():
                    delta_days = (pd.Timestamp(str(date_str)) - clip_ts).days
                    delta_weeks = delta_days / 7.0
                    if (
                        time_window.max_weeks_before is not None
                        and delta_weeks < -time_window.max_weeks_before
                    ):
                        continue
                    if (
                        time_window.max_weeks_after is not None
                        and delta_weeks > time_window.max_weeks_after
                    ):
                        continue
                    selected.append(emb)

            if not selected:
                continue

            user_emb = np.mean(selected, axis=0).astype(np.float32)
            X_list.append(user_emb)
            uid_list.append(uid)

        if X_list:
            return np.stack(X_list), np.array(uid_list, dtype=object)
        return (
            np.empty((0, self.embed_dim), dtype=np.float32),
            np.empty(0, dtype=object),
        )
