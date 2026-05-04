"""Segment-level feature store for extract-once, evaluate-many pipeline.

Extracts segment-level features (weekly or daily) for ALL segments in a single
pass, then provides fast time-windowed aggregation to user-level features for
any (task, time_window) combination without re-extracting.

Supports two segment types controlled by DataConfig.segment_type:
  - "weekly": 168-hour segments from weekly_hf (default)
  - "daily": 24-hour segments from daily_hf

Typical usage:
    # Phase 1: build (once per feature_type)
    store = WeekFeatureStore.build(hf_dataset, feature_config, split_users, seed=42)
    store.save("results/stores/stat_simple.npz")

    # Phase 2: evaluate (per task x time_window x classifier)
    splits = store.aggregate_for_task(labels_col, clip_dates, time_window, split_users)
    X_train, y_train, uids_train = splits["train"]
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from downstream_evaluation.config import (
    BaselineFeatureConfig,
    EncoderFeatureConfig,
    FeatureConfig,
    MultiRocketConfig,
    TimeWindow,
)

if TYPE_CHECKING:
    import datasets as hf_ds

logger = logging.getLogger(__name__)

# Sentinel values for missing labels in the parquet
_MISSING_INT = -1
_MISSING_FLOAT = -1.0

# Seconds per segment type
_SECONDS_PER_WEEK = 7 * 24 * 3600
_SECONDS_PER_DAY = 24 * 3600

# Hours per segment type (for coverage weighting)
_HOURS_PER_SEGMENT = {"weekly": 168, "daily": 24}


@dataclass
class _StoreMetadata:
    """Internal metadata saved alongside feature arrays."""

    feature_type: str  # "stat_simple", "stat_full", "ssl_encoder", or "multirocket"
    feature_dim: int
    n_samples: int
    segment_type: str = "weekly"  # "weekly" or "daily"
    # SSL encoder normalization stats (None for statistical features)
    norm_means: np.ndarray | None = None
    norm_stds: np.ndarray | None = None
    # Provenance (for staleness detection)
    dataset_dir: str | None = None  # HF dataset dir used to build the store
    checkpoint_path: str | None = None  # SSL checkpoint (encoder features only)


class WeekFeatureStore:
    """Stores segment-level features (weekly or daily) for the entire dataset.

    After building, provides fast time-windowed, per-task aggregation
    to user-level feature vectors.

    Despite the class name, this store handles both weekly and daily segments.
    The segment_type is tracked in metadata and affects time window filtering
    and coverage weighting.
    """

    def __init__(
        self,
        features: np.ndarray,
        user_ids: np.ndarray,
        segment_starts: np.ndarray,
        segment_starts_ns: np.ndarray,
        metadata: _StoreMetadata,
        n_valid_hours: np.ndarray | None = None,
    ):
        """Initialize from pre-extracted arrays. Use build() or load() instead.

        Args:
            features: (N, D) float32 feature matrix.
            user_ids: (N,) object array of user ID strings.
            segment_starts: (N,) object array of segment start ISO strings
                (week_start for weekly, date for daily).
            segment_starts_ns: (N,) int64 array of segment starts as nanosecond timestamps.
            metadata: Store metadata (includes segment_type).
            n_valid_hours: (N,) int array of valid hours per segment (for coverage weighting).
        """
        self.features = features
        self.user_ids = user_ids
        self.segment_starts = segment_starts
        self.segment_starts_ns = segment_starts_ns
        self.metadata = metadata
        self.n_valid_hours = n_valid_hours

        # Backward-compatible aliases
        self.week_starts = segment_starts
        self.week_starts_ns = segment_starts_ns

        # Build user_id → list of dataset indices for fast lookup
        self._user_to_indices: dict[str, list[int]] = defaultdict(list)
        for i, uid in enumerate(self.user_ids):
            self._user_to_indices[uid].append(i)

        segment_label = self.metadata.segment_type
        logger.info(
            f"WeekFeatureStore: {self.n_samples} {segment_label} samples, "
            f"{self.feature_dim}D features, "
            f"{len(self._user_to_indices)} unique users"
        )

    @property
    def n_samples(self) -> int:
        """Number of week samples in the store."""
        return self.features.shape[0]

    @property
    def feature_dim(self) -> int:
        """Dimensionality of the feature vectors."""
        return self.features.shape[1]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        hf_dataset: hf_ds.Dataset,
        feature_config: FeatureConfig,
        split_users: dict[str, set[str]],
        seed: int = 42,
        dataset_dir: str | None = None,
        segment_type: str = "weekly",
    ) -> WeekFeatureStore:
        """Extract segment-level features for all samples in one pass.

        Args:
            hf_dataset: Full HF dataset (all users, all segments).
            feature_config: Which features to extract (statistical or ssl_encoder).
            split_users: User split mapping (train/validation/test → set of user_ids).
                Used to compute encoder normalization stats from train users only.
            seed: Random seed for reproducibility.
            dataset_dir: Path to the HF dataset directory (stored as provenance
                metadata for staleness detection on subsequent loads).
            segment_type: "weekly" or "daily" — determines which columns to read
                and how time windows are interpreted.

        Returns:
            Populated WeekFeatureStore.
        """
        t0 = time.time()

        # Read user_ids, segment starts, and coverage from the dataset
        # Column names differ between daily and weekly HF datasets
        if segment_type == "daily":
            timestamp_col = "date"
        else:
            timestamp_col = "week_start"

        logger.info(
            "Reading user_ids, %s, and coverage from HF dataset (segment_type=%s)...",
            timestamp_col,
            segment_type,
        )
        user_ids = np.array(hf_dataset["user_id"], dtype=object)
        segment_starts = np.array(hf_dataset[timestamp_col], dtype=object)
        n_valid_hours = (
            np.array(hf_dataset["n_valid_hours"], dtype=np.int32)
            if "n_valid_hours" in hf_dataset.column_names
            else None
        )

        # Pre-compute nanosecond timestamps for fast time window filtering
        segment_starts_ns = (
            pd.to_datetime(segment_starts.tolist()).astype("int64").values
        )  # nanoseconds since epoch

        # Extract features
        if feature_config.type == "statistical":
            features, norm_means, norm_stds, feat_type = cls._build_statistical(
                hf_dataset, feature_config.statistical
            )
        elif feature_config.type == "ssl_encoder":
            features, norm_means, norm_stds, feat_type = cls._build_ssl_encoder(
                hf_dataset, feature_config.ssl_encoder, user_ids, split_users, seed
            )
        elif feature_config.type == "multirocket":
            features, norm_means, norm_stds, feat_type = cls._build_multirocket(
                hf_dataset, feature_config.multirocket, user_ids, split_users, seed
            )
        elif feature_config.type == "fe_handcrafted_weekly":
            raise RuntimeError(
                "fe_handcrafted_weekly has no in-process extractor. "
                "Build offline with scripts/build_weekly_handcrafted_store.py "
                "and load via --store.load_path."
            )
        else:
            raise ValueError(f"Unknown feature type: {feature_config.type}")

        # Extract checkpoint path for encoder features (provenance tracking)
        checkpoint_path = None
        if feature_config.type == "ssl_encoder" and feature_config.ssl_encoder:
            checkpoint_path = feature_config.ssl_encoder.checkpoint_path

        metadata = _StoreMetadata(
            feature_type=feat_type,
            feature_dim=features.shape[1],
            n_samples=features.shape[0],
            segment_type=segment_type,
            norm_means=norm_means,
            norm_stds=norm_stds,
            dataset_dir=dataset_dir,
            checkpoint_path=checkpoint_path,
        )

        elapsed = time.time() - t0
        logger.info(
            f"WeekFeatureStore built in {elapsed:.1f}s: "
            f"{features.shape[0]} {segment_type} samples, {features.shape[1]}D"
        )

        return cls(
            features, user_ids, segment_starts, segment_starts_ns, metadata, n_valid_hours
        )

    @staticmethod
    def _build_statistical(
        hf_dataset: hf_ds.Dataset,
        config: BaselineFeatureConfig,
    ) -> tuple[np.ndarray, None, None, str]:
        """Extract statistical features (no normalization stats needed)."""
        from downstream_evaluation.feature_extractors.baseline_extractor import (
            BaselineFeatureExtractor,
        )

        extractor = BaselineFeatureExtractor(config)
        features = extractor.extract_features_only(hf_dataset)
        feat_type = "stat_full" if config.use_full_features else "stat_simple"
        return features, None, None, feat_type

    @staticmethod
    def _build_ssl_encoder(
        hf_dataset: hf_ds.Dataset,
        config: EncoderFeatureConfig,
        user_ids: np.ndarray,
        split_users: dict[str, set[str]],
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """Extract SSL encoder features with train-split normalization."""
        from downstream_evaluation.feature_extractors.encoder_extractor import (
            EncoderFeatureExtractor,
            _compute_normalization_stats,
        )

        extractor = EncoderFeatureExtractor(config, random_state=seed)

        if config.source == "checkpoint":
            if config.normalization_stats_path:
                # Load pre-computed normalization stats from JSON file
                import json
                from pathlib import Path

                stats_path = Path(config.normalization_stats_path)
                if not stats_path.exists():
                    raise FileNotFoundError(
                        f"normalization_stats_path={stats_path} does not exist"
                    )
                with open(stats_path) as f:
                    stats = json.load(f)
                means = np.array(stats["means"], dtype=np.float32)
                stds = np.array(stats["stds"], dtype=np.float32)
                logger.info(
                    f"Loaded normalization stats from {stats_path}: "
                    f"means={means[:3]}... stds={stds[:3]}..."
                )
            else:
                # Compute normalization stats from train users only
                train_users = split_users.get("train", set())
                train_mask = np.isin(user_ids, list(train_users))
                train_indices = np.where(train_mask)[0].tolist()

                logger.info(
                    f"Computing SSL encoder normalization stats from "
                    f"{len(train_indices)} train-split samples..."
                )
                train_subset = hf_dataset.select(train_indices)
                means, stds = _compute_normalization_stats(train_subset)
                logger.info(
                    f"SSL encoder norm stats (train): means={means[:3]}... stds={stds[:3]}..."
                )
            extractor.set_normalization_stats(means, stds)
        else:
            means, stds = None, None

        features = extractor.extract_features_only(hf_dataset)
        return features, means, stds, "ssl_encoder"

    @staticmethod
    def _build_multirocket(
        hf_dataset: hf_ds.Dataset,
        config: MultiRocketConfig,
        user_ids: np.ndarray,
        split_users: dict[str, set[str]],
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, str]:
        """Extract MultiRocket features with train-only fit.

        Z-score normalization stats are computed from train-split observed
        values only, then applied to all data before kernel fitting.
        MultiRocket kernels are fitted on training samples only to prevent
        information leakage, then applied to transform all samples.
        """
        from downstream_evaluation.feature_extractors.multirocket_extractor import (
            MultiRocketFeatureExtractor,
        )

        extractor = MultiRocketFeatureExtractor(config, random_state=seed)

        # Identify train indices for fit-only
        train_users = split_users.get("train", set())
        train_mask = np.isin(user_ids, list(train_users))
        train_indices = np.where(train_mask)[0].tolist()

        logger.info(
            f"Fitting MultiRocket on {len(train_indices)} train samples, "
            f"transforming all {len(hf_dataset)} samples..."
        )
        features = extractor.extract_with_splits(hf_dataset, train_indices)
        return features, extractor.norm_means, extractor.norm_stds, "multirocket"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the feature store to disk as a .npz file.

        Args:
            path: Output path (should end with .npz).
        """
        # Guard: refuse to save extremely large stores that would OOM during
        # np.savez_compressed (which must read everything into memory).
        feat_bytes = self.features.shape[0] * self.features.shape[1] * self.features.dtype.itemsize
        _MAX_SAVE_BYTES = 50 * 1024**3  # 50 GB
        if feat_bytes > _MAX_SAVE_BYTES:
            raise MemoryError(
                f"Feature store is too large to save as .npz "
                f"({feat_bytes / 1e9:.1f} GB > {_MAX_SAVE_BYTES / 1e9:.0f} GB limit). "
                f"For large daily-segment stores, skip --store.save_path and "
                f"re-extract each run, or use the incremental aggregation pathway."
            )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "features": self.features,
            "user_ids": self.user_ids,
            "week_starts": self.segment_starts,
            "week_starts_ns": self.segment_starts_ns,
            "feature_type": np.array(self.metadata.feature_type),
            "n_samples": np.array(self.metadata.n_samples),
            "segment_type": np.array(self.metadata.segment_type),
        }
        if self.metadata.norm_means is not None:
            save_dict["norm_means"] = self.metadata.norm_means
        if self.metadata.norm_stds is not None:
            save_dict["norm_stds"] = self.metadata.norm_stds
        if self.n_valid_hours is not None:
            save_dict["n_valid_hours"] = self.n_valid_hours
        # Provenance metadata for staleness detection
        if self.metadata.dataset_dir is not None:
            save_dict["dataset_dir"] = np.array(self.metadata.dataset_dir)
        if self.metadata.checkpoint_path is not None:
            save_dict["checkpoint_path"] = np.array(self.metadata.checkpoint_path)

        np.savez_compressed(path, **save_dict)
        file_size_mb = path.stat().st_size / (1024 * 1024)
        logger.info(f"WeekFeatureStore saved to {path} ({file_size_mb:.1f} MB)")

    @classmethod
    def load(
        cls,
        path: str | Path,
        expected_dataset_dir: str | None = None,
        expected_checkpoint_path: str | None = None,
    ) -> WeekFeatureStore:
        """Load a feature store from a .npz file.

        Args:
            path: Path to the .npz file.
            expected_dataset_dir: If provided, warn when the store was built
                from a different dataset directory (indicates staleness).
            expected_checkpoint_path: If provided, warn when the store was
                built from a different checkpoint (encoder features only).

        Returns:
            Loaded WeekFeatureStore.
        """
        path = Path(path)
        if not path.exists():
            # Try with .npz extension
            if not path.suffix:
                path = path.with_suffix(".npz")
            if not path.exists():
                raise FileNotFoundError(f"Feature store not found: {path}")

        logger.info(f"Loading WeekFeatureStore from {path}...")
        data = np.load(path, allow_pickle=True)

        features = data["features"]
        user_ids = data["user_ids"]
        segment_starts = data["week_starts"]  # key kept as "week_starts" for backward compat
        segment_starts_ns = data["week_starts_ns"]
        feature_type = str(data["feature_type"])

        # Segment type (absent in stores built before daily support)
        segment_type = str(data["segment_type"]) if "segment_type" in data else "weekly"

        norm_means = data["norm_means"] if "norm_means" in data else None
        norm_stds = data["norm_stds"] if "norm_stds" in data else None
        n_valid_hours = data["n_valid_hours"] if "n_valid_hours" in data else None

        # Read provenance metadata (may be absent in older stores)
        stored_dataset_dir = str(data["dataset_dir"]) if "dataset_dir" in data else None
        stored_checkpoint = str(data["checkpoint_path"]) if "checkpoint_path" in data else None

        # Staleness warnings
        if expected_dataset_dir and stored_dataset_dir:
            if stored_dataset_dir != expected_dataset_dir:
                logger.warning(
                    f"STALE FEATURE STORE: built from '{stored_dataset_dir}' "
                    f"but current dataset is '{expected_dataset_dir}'. "
                    f"Delete {path} and re-extract features."
                )
        if expected_checkpoint_path and stored_checkpoint:
            if stored_checkpoint != expected_checkpoint_path:
                logger.warning(
                    f"STALE FEATURE STORE: built from checkpoint '{stored_checkpoint}' "
                    f"but current checkpoint is '{expected_checkpoint_path}'. "
                    f"Delete {path} and re-extract features."
                )
        if stored_dataset_dir is None:
            logger.warning(
                f"Feature store {path} has no provenance metadata (built before this check "
                f"was added). Consider re-extracting to ensure consistency."
            )

        metadata = _StoreMetadata(
            feature_type=feature_type,
            feature_dim=features.shape[1],
            n_samples=features.shape[0],
            segment_type=segment_type,
            norm_means=norm_means,
            norm_stds=norm_stds,
            dataset_dir=stored_dataset_dir,
            checkpoint_path=stored_checkpoint,
        )

        return cls(
            features, user_ids, segment_starts, segment_starts_ns, metadata, n_valid_hours
        )

    # ------------------------------------------------------------------
    # Phase 2: Time-windowed aggregation
    # ------------------------------------------------------------------

    def aggregate_for_task(
        self,
        task_labels: np.ndarray | pd.Series,
        task_type: str,
        clip_dates: dict[str, str] | None,
        time_window: TimeWindow,
        split_users: dict[str, set[str]],
        pooling_method: str = "mean",
    ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
        """Aggregate week-level features to user-level for a single task.

        For each user in each split:
        1. Check valid label (not sentinel)
        2. Filter weeks by time window (using clip_dates as reference)
        3. Pool surviving week features (mean or coverage-weighted mean)
        4. Return per-split (X, y, user_ids, n_weeks_total)

        Args:
            task_labels: (N,) array of labels aligned with the store, where
                -1 (int) or -1.0 (float) indicates missing.
            task_type: "binary", "ordinal", or "continuous".
            clip_dates: {user_id: label_date_iso} for this task, or None.
            time_window: Which weeks to include.
            split_users: {"train": set, "validation": set, "test": set}.
            pooling_method: "mean" for simple mean, "cov_weighted_mean" for
                coverage-weighted mean (each week weighted by n_valid_hours/168).

        Returns:
            Dict with keys "train", "val", "test", each mapping to
            (X, y, user_ids, n_weeks) where:
            - X: (n_users, D) float32 aggregated features
            - y: (n_users,) labels
            - user_ids: (n_users,) string user IDs
            - n_weeks: total week count across all users in this split
        """
        if isinstance(task_labels, pd.Series):
            task_labels = task_labels.values

        # Validate pooling method
        use_cov_weights = pooling_method == "cov_weighted_mean"
        if use_cov_weights and self.n_valid_hours is None:
            logger.warning(
                "cov_weighted_mean requested but n_valid_hours not available "
                "in feature store — falling back to simple mean pooling."
            )
            use_cov_weights = False

        # Pre-compute clip date nanoseconds for fast comparison
        clip_ns = None
        if clip_dates and time_window.needs_clip_dates:
            clip_ns = {}
            for uid, date_str in clip_dates.items():
                clip_ns[uid] = pd.Timestamp(date_str).value  # nanoseconds

        # Determine missing sentinel based on dtype
        is_float = np.issubdtype(task_labels.dtype, np.floating)

        # Map split names (split file uses "validation", CSV uses "val")
        split_name_map = {
            "train": "train",
            "val": "validation",
            "test": "test",
        }

        results = {}
        for out_key, split_key in split_name_map.items():
            split_user_set = split_users.get(split_key, set())
            X_list = []
            y_list = []
            uid_list = []
            total_weeks = 0

            for uid in sorted(split_user_set):
                indices = self._user_to_indices.get(uid)
                if indices is None:
                    continue

                # Check label validity (take first non-missing label for this user)
                label = task_labels[indices[0]]
                if is_float and (np.isnan(label) or label == _MISSING_FLOAT):
                    continue
                if not is_float and label == _MISSING_INT:
                    continue

                # Apply time window filter
                if time_window.is_full or clip_ns is None:
                    # "full" condition — use all weeks
                    week_indices = indices
                else:
                    ref_ns = clip_ns.get(uid)
                    if ref_ns is None:
                        # No clip date for this user — skip (can't apply window)
                        continue
                    week_indices = self._filter_weeks_by_window(indices, ref_ns, time_window)

                if len(week_indices) == 0:
                    continue

                # Pool features across surviving segments
                if use_cov_weights:
                    # Coverage-weighted mean: w_i = n_valid_hours_i / hours_per_segment
                    hours_per_seg = _HOURS_PER_SEGMENT.get(self.metadata.segment_type, 168)
                    weights = (
                        self.n_valid_hours[week_indices].astype(np.float32) / hours_per_seg
                    )
                    weight_sum = weights.sum()
                    if weight_sum > 0:
                        user_features = (weights[:, np.newaxis] * self.features[week_indices]).sum(
                            axis=0
                        ) / weight_sum
                    else:
                        # All-zero coverage (shouldn't happen after filtering) — fallback to mean
                        user_features = self.features[week_indices].mean(axis=0)
                else:
                    user_features = self.features[week_indices].mean(axis=0)

                X_list.append(user_features)
                y_list.append(label)
                uid_list.append(uid)
                total_weeks += len(week_indices)

            if X_list:
                X = np.stack(X_list)
                y = np.array(y_list)
                uids = np.array(uid_list, dtype=object)
            else:
                X = np.empty((0, self.feature_dim), dtype=np.float32)
                y = np.empty(0)
                uids = np.empty(0, dtype=object)

            results[out_key] = (X, y, uids, total_weeks)

        n_train = results["train"][0].shape[0]
        n_val = results["val"][0].shape[0]
        n_test = results["test"][0].shape[0]
        logger.debug(f"Aggregated: train={n_train}, val={n_val}, test={n_test} users")

        return results

    def aggregate_for_segment_recovery(
        self,
        task_labels: np.ndarray | pd.Series,
        split_users: dict[str, set[str]],
        person_mean_center: bool = False,
    ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
        """Pair each segment's features with its own label (same-segment recovery).

        For every user-segment with a valid label, emit one
        ``(features, label, user_id)`` example — one prediction per populated
        segment (i.e. one per ISO week for weekly segments). Use this to
        evaluate representation recovery: *"does the week's embedding preserve
        enough information to recover the week's aggregated outcome?"*

        Contrast with ``aggregate_for_task``, which mean-pools all of a user's
        segments into a single cross-sectional feature per user.

        Label validity is assumed pre-applied: segments with missing sentinels
        (``-1`` / ``-1.0``) are skipped. Splits are user-level.

        Args:
            task_labels: ``(N,)`` array aligned with the store. Each segment
                has its own label. Missing = ``-1`` (int) or ``-1.0`` (float).
            split_users: ``{"train", "validation", "test"} -> set[user_id]``.
            person_mean_center: If True, subtract each user's mean feature
                vector and mean label from their samples — isolates within-
                person variation from between-person trait (needed for
                honest representation-recovery metrics when ICC is high).

        Returns:
            ``{"train", "val", "test"} -> (X, y, user_ids, n_samples)``.
            ``n_samples`` equals ``X.shape[0]`` (one sample per populated,
            valid-label segment in the split).
        """
        if isinstance(task_labels, pd.Series):
            task_labels = task_labels.values

        is_float = np.issubdtype(task_labels.dtype, np.floating)
        missing_sentinel = _MISSING_FLOAT if is_float else _MISSING_INT

        def _is_missing(lbl) -> bool:
            if is_float:
                return bool(np.isnan(lbl)) or lbl == missing_sentinel
            return lbl == missing_sentinel

        def _valid_indices(user_indices: list[int]) -> list[int]:
            return [i for i in user_indices if not _is_missing(task_labels[i])]

        # Pre-compute per-user feature/label means (used only if centering)
        user_feat_mean: dict[str, np.ndarray] = {}
        user_label_mean: dict[str, float] = {}
        if person_mean_center:
            for uid, indices in self._user_to_indices.items():
                valid = _valid_indices(indices)
                if not valid:
                    continue
                user_feat_mean[uid] = self.features[valid].mean(axis=0)
                user_label_mean[uid] = float(np.mean([float(task_labels[i]) for i in valid]))

        split_name_map = {"train": "train", "val": "validation", "test": "test"}
        results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}

        for out_key, split_key in split_name_map.items():
            split_user_set = split_users.get(split_key, set())
            X_list: list[np.ndarray] = []
            y_list: list[float] = []
            uid_list: list[str] = []

            for uid in sorted(split_user_set):
                indices = self._user_to_indices.get(uid)
                if not indices:
                    continue

                mu_feat = user_feat_mean.get(uid) if person_mean_center else None
                mu_label = user_label_mean.get(uid, 0.0) if person_mean_center else 0.0

                for idx in indices:
                    label = task_labels[idx]
                    if _is_missing(label):
                        continue

                    feat = self.features[idx]
                    y_val = float(label)
                    if mu_feat is not None:
                        feat = feat - mu_feat
                        y_val -= mu_label

                    X_list.append(feat)
                    y_list.append(y_val)
                    uid_list.append(uid)

            if X_list:
                X = np.stack(X_list)
                y = np.asarray(y_list, dtype=np.float32)
                uids = np.asarray(uid_list, dtype=object)
            else:
                X = np.empty((0, self.feature_dim), dtype=np.float32)
                y = np.empty(0, dtype=np.float32)
                uids = np.empty(0, dtype=object)

            results[out_key] = (X, y, uids, X.shape[0])

        logger.info(
            "Segment-recovery aggregation: train=%d, val=%d, test=%d samples "
            "(person_mean_center=%s)",
            results["train"][0].shape[0],
            results["val"][0].shape[0],
            results["test"][0].shape[0],
            person_mean_center,
        )

        return results

    def _filter_weeks_by_window(
        self,
        indices: list[int],
        ref_ns: int,
        time_window: TimeWindow,
    ) -> list[int]:
        """Filter segment indices by time window relative to reference date.

        The time window's max_weeks_before/after are interpreted in the
        segment's native unit: weeks for weekly segments, weeks for daily
        segments (i.e., the window is always specified in weeks regardless
        of segment type).

        Args:
            indices: Dataset indices for this user's segments.
            ref_ns: Reference date as nanoseconds since epoch.
            time_window: Window parameters (always in weeks).

        Returns:
            Filtered list of indices.
        """
        seg_ns = self.segment_starts_ns[indices]

        # Time window is always specified in weeks, regardless of segment type
        ns_per_week = _SECONDS_PER_WEEK * 1_000_000_000

        if time_window.max_weeks_before is not None:
            lower_bound = ref_ns - time_window.max_weeks_before * ns_per_week
        else:
            lower_bound = -np.inf

        if time_window.max_weeks_after is not None:
            upper_bound = ref_ns + time_window.max_weeks_after * ns_per_week
        else:
            upper_bound = np.inf

        mask = (seg_ns >= lower_bound) & (seg_ns <= upper_bound)
        return [indices[i] for i, keep in enumerate(mask) if keep]
