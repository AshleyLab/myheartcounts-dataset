"""Online PyPOTS datasets for forecasting training.

This module replaces the previous "export everything to H5 first" flow with a
manifest-driven Dataset that slices samples directly from the original
trajectory HuggingFace dataset on demand.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import datasets as hf_ds
import h5py
import numpy as np
import torch
from pygrinder import fill_and_get_mask_torch
from pypots.data.dataset.base import BaseDataset
from torch.utils.data import BatchSampler, Sampler

from data.transforms.nan_transforms import ZeroToNaNTransform
from forecasting_evaluation.config import FeaturesConfig
from forecasting_evaluation.data.standard_scaler import (
    ChannelStandardScalerStats,
)
from forecasting_evaluation.feature_extractors.multivariate_extractor import (
    MultivariateFeatureExtractor,
)

logger = logging.getLogger(__name__)
SCALER_VARIANT = "train_only_standard_scaler_v1"
# v2: the row-group manifest is model-agnostic — it no longer drops windows whose
# history is shorter than a model's fixed context length (`n_steps`). That is now
# a per-model concern. Bumped so any v1 manifests on disk are rebuilt.
MANIFEST_VERSION = 2


@dataclass
class ModelConfig:
    """PyPOTS forecasting model configuration.

    Relocated verbatim from the former
    ``forecasting_evaluation.forecasting_training.config`` module. Only
    ``n_steps``/``n_pred_steps``/``n_features`` feed the content-addressed cache
    digest (:func:`history_cf_cache_subdir`) and the dataset window sizing; the
    remaining fields are retained for back-compat with checkpoints whose
    ``training_config.json`` carries them. The forecasting *training* package
    defines its own training-flavored ``ModelConfig`` separately.

    Reference:
    - PyPOTS DLinear forecasting API
    - PyPOTS MixLinear forecasting API
    - PyPOTS TEFN forecasting API
    - PyPOTS SegRNN forecasting API

    Shared parameters:
    - ``n_steps``: input history length
    - ``n_features``: number of channels/features
    - ``n_pred_steps``: forecast horizon
    - ``loss``: PyPOTS training loss name, resolved from ``pypots.nn.modules.loss``
    - ``validation_metric``: PyPOTS validation metric name

    DLinear-specific parameters:
    - ``moving_avg_window_size``
    - ``individual``
    - ``d_model`` (used in non-individual mode)

    MixLinear-specific parameters:
    - ``period_len``
    - ``lpf``
    - ``alpha``
    - ``rank``

    TEFN-specific parameters:
    - ``n_fod``
    - ``apply_nonstationary_norm``

    SegRNN-specific parameters:
    - ``seg_len``
    - ``d_model``
    - ``dropout``

    ``base_model_name`` is only used by the Chronos-2 fine-tuning path.
    """

    model_name: str = "chronos2"
    base_model_name: str = "amazon/chronos-2"
    n_steps: int = 168  # Shared: input history length in hours.
    n_pred_steps: int = 24  # Shared: forecast horizon in hours.
    n_features: int = 19  # Shared: number of input/output channels.
    loss: str = "mae"
    validation_metric: str = "mae"

    # DLinear-specific parameters.
    d_model: int = 64
    moving_avg_window_size: int = 25
    individual: bool = False

    # MixLinear-specific parameters.
    period_len: int = 24
    lpf: int = 2
    alpha: float = 0.5
    rank: int = 2

    # TEFN-specific parameters.
    n_fod: int = 2
    apply_nonstationary_norm: bool = False

    # SegRNN-specific parameters.
    seg_len: int = 24

    # Shared deep-learning hyperparameters used by several forecasting architectures.
    n_layers: int = 2
    n_heads: int = 4
    d_ffn: int = 128
    dropout: float = 0.1

    # Frequency / decomposition style params.
    top_k: int = 5
    n_kernels: int = 6


@dataclass(frozen=True)
class ForecastingSampleDescriptor:
    """Manifest entry describing one forecasting sample."""

    dataset_row_idx: int
    user_id: str
    current_day: int
    history_end_hour: int
    pred_end_hour: int


@dataclass(frozen=True)
class ForecastingWindowDescriptor:
    """Window metadata for one sample within a row group."""

    current_day: int
    history_end_hour: int
    pred_end_hour: int


@dataclass(frozen=True)
class ForecastingRowGroup:
    """All valid forecasting windows derived from one trajectory row."""

    dataset_row_idx: int
    user_id: str
    windows: tuple[ForecastingWindowDescriptor, ...]


class ForecastingSampleIndexBuilder:
    """Build row-grouped manifests from split trajectories and sample indices."""

    HOURS_PER_DAY = 24

    def __init__(
        self,
        split_ds: hf_ds.Dataset,
        sample_index_file: str | Path,
        n_steps: int,
        n_pred_steps: int,
    ) -> None:
        """Initialize the manifest builder from a split and sample index."""
        self._split_ds = split_ds
        self._sample_index_file = Path(sample_index_file)
        self._n_steps = int(n_steps)
        self._n_pred_steps = int(n_pred_steps)
        self._sample_index = self._load_sample_index()

    def _load_sample_index(self) -> dict[str, list[int]]:
        if not self._sample_index_file.exists():
            raise FileNotFoundError(f"sample_index_file not found: {self._sample_index_file}")

        loaded = json.loads(self._sample_index_file.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(
                f"sample_index_file must contain a user->days mapping: {self._sample_index_file}"
            )

        normalized: dict[str, list[int]] = {}
        for user_id, day_list in loaded.items():
            if not isinstance(day_list, list):
                continue
            normalized[str(user_id)] = sorted({int(day) for day in day_list if int(day) >= 1})
        return normalized

    def build_row_groups(self) -> list[ForecastingRowGroup]:
        """Create row-grouped slice descriptors matching the previous H5 export logic."""
        row_groups: list[ForecastingRowGroup] = []

        for row_idx, row in enumerate(self._split_ds):
            user_id = str(row["user_id"])
            candidate_days = self._sample_index.get(user_id, [])
            if not candidate_days:
                continue

            trajectory_length = len(row["values"])
            windows: list[ForecastingWindowDescriptor] = []
            for current_day in candidate_days:
                history_end = current_day * self.HOURS_PER_DAY
                pred_end = history_end + self._n_pred_steps

                # Data-quality drops only, applied equally to every model: no
                # history before the day boundary, and ground truth must exist for
                # the full horizon. Model-capability filtering (e.g. a fixed
                # context length `n_steps`) is intentionally NOT applied here — the
                # manifest is model-agnostic so every model sees the same window
                # set. Training datasets that need fixed windows re-filter at
                # runtime in `resolve_runtime_row_groups`.
                if history_end <= 0:
                    continue
                if pred_end > trajectory_length:
                    continue

                windows.append(
                    ForecastingWindowDescriptor(
                        current_day=current_day,
                        history_end_hour=history_end,
                        pred_end_hour=pred_end,
                    )
                )

            if windows:
                row_groups.append(
                    ForecastingRowGroup(
                        dataset_row_idx=row_idx,
                        user_id=user_id,
                        windows=tuple(windows),
                    )
                )

        logger.info(
            "Built online forecasting row-group manifest with %d rows and %d samples",
            len(row_groups),
            sum(len(group.windows) for group in row_groups),
        )
        return row_groups

    def build(self) -> list[ForecastingSampleDescriptor]:
        """Backward-compatible flat manifest view."""
        manifest: list[ForecastingSampleDescriptor] = []
        for row_group in self.build_row_groups():
            for window in row_group.windows:
                manifest.append(
                    ForecastingSampleDescriptor(
                        dataset_row_idx=row_group.dataset_row_idx,
                        user_id=row_group.user_id,
                        current_day=window.current_day,
                        history_end_hour=window.history_end_hour,
                        pred_end_hour=window.pred_end_hour,
                    )
                )
        return manifest


def history_cf_cache_subdir(
    base_dir: str | Path,
    data_config,
    model_config: ModelConfig,
    features_config: FeaturesConfig,
) -> Path:
    """Return content-addressed cache path for per-row history_cf storage."""
    key = {
        "trajectory_hf_dir": str(data_config.trajectory_hf_dir),
        "split_file": str(data_config.split_file),
        "sample_index_file": str(data_config.sample_index_file),
        "train_ratio": data_config.train_ratio,
        "val_ratio": data_config.val_ratio,
        "split_seed": data_config.split_seed,
        "max_samples": data_config.max_samples,
        "feature_channel": features_config.channel,
        "n_steps": model_config.n_steps,
        "n_pred_steps": model_config.n_pred_steps,
        "n_features": model_config.n_features,
        "scaler_variant": SCALER_VARIANT,
    }
    digest = hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:8]
    return Path(base_dir) / digest


def resolve_cache_base_dir(data_config) -> Path:
    """Cache base dir under the configured data root: {data_root}/cache/forecasting.

    The data root is the parent of ``trajectory_hf_dir`` (e.g.
    ``${MHC_DATA_DIR}/hourly_trajectory`` -> ``${MHC_DATA_DIR}``), so caches live
    next to the dataset they are derived from instead of the current working dir.
    """
    return Path(data_config.trajectory_hf_dir).parent / "cache" / "forecasting"


def history_cf_manifest_path(cache_dir: str | Path, split_name: str) -> Path:
    """Return the JSON manifest cache path for one split."""
    normalized_split = str(split_name).strip().lower()
    return Path(cache_dir) / f"{normalized_split}_manifest.json"


def build_history_cf_rows(
    split_ds: hf_ds.Dataset,
    features_config: FeaturesConfig,
    model_config: ModelConfig,
) -> list[torch.Tensor]:
    """Extract raw channel-first history tensors for one split."""
    extractor = MultivariateFeatureExtractor(
        config=features_config,
        forecasting_length=model_config.n_pred_steps,
    )
    zero_to_nan_transform = ZeroToNaNTransform()

    rows: list[torch.Tensor] = []
    for row in split_ds:
        features = extractor.extract(row)
        history_cf = torch.as_tensor(features["history"], dtype=torch.float32)
        history_cf = zero_to_nan_transform(history_cf)
        rows.append(history_cf)
    return rows


def write_row_group_manifest(
    row_groups: list[ForecastingRowGroup],
    manifest_path: str | Path,
) -> Path:
    """Persist row-group manifest as JSON."""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest_version": MANIFEST_VERSION,
        "row_groups": [
            {
                "dataset_row_idx": row_group.dataset_row_idx,
                "user_id": row_group.user_id,
                "windows": [
                    {
                        "current_day": window.current_day,
                        "history_end_hour": window.history_end_hour,
                        "pred_end_hour": window.pred_end_hour,
                    }
                    for window in row_group.windows
                ],
            }
            for row_group in row_groups
        ],
    }
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_path


def _manifest_version_matches(manifest_path: str | Path) -> bool:
    """Return whether a persisted manifest matches the current MANIFEST_VERSION."""
    try:
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return int(payload.get("manifest_version", 0)) == MANIFEST_VERSION


def load_row_group_manifest(manifest_path: str | Path) -> list[ForecastingRowGroup]:
    """Load row-group manifest from JSON."""
    manifest_path = Path(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    row_groups_payload = payload.get("row_groups", [])
    row_groups: list[ForecastingRowGroup] = []
    for row_group_payload in row_groups_payload:
        windows = tuple(
            ForecastingWindowDescriptor(
                current_day=int(window_payload["current_day"]),
                history_end_hour=int(window_payload["history_end_hour"]),
                pred_end_hour=int(window_payload["pred_end_hour"]),
            )
            for window_payload in row_group_payload.get("windows", [])
        )
        row_groups.append(
            ForecastingRowGroup(
                dataset_row_idx=int(row_group_payload["dataset_row_idx"]),
                user_id=str(row_group_payload["user_id"]),
                windows=windows,
            )
        )
    return row_groups


def load_or_build_row_group_manifest(
    *,
    split_ds: hf_ds.Dataset,
    sample_index_file: str | Path,
    model_config: ModelConfig,
    manifest_path: str | Path,
    split_name: str,
    overwrite: bool = False,
) -> list[ForecastingRowGroup]:
    """Reuse manifest JSON when present, otherwise rebuild and persist it."""
    manifest_path = Path(manifest_path)
    if manifest_path.exists() and not overwrite and _manifest_version_matches(manifest_path):
        row_groups = load_row_group_manifest(manifest_path)
        logger.info(
            "Manifest cache already exists at %s, reusing it (%d rows, %d samples)",
            manifest_path,
            len(row_groups),
            sum(len(group.windows) for group in row_groups),
        )
        return row_groups
    if manifest_path.exists() and not overwrite:
        logger.info(
            "Manifest cache at %s is stale (version != %d); rebuilding",
            manifest_path,
            MANIFEST_VERSION,
        )

    logger.info("Building %s manifest cache at %s", split_name, manifest_path)
    row_groups = ForecastingSampleIndexBuilder(
        split_ds=split_ds,
        sample_index_file=sample_index_file,
        n_steps=model_config.n_steps,
        n_pred_steps=model_config.n_pred_steps,
    ).build_row_groups()
    write_row_group_manifest(row_groups, manifest_path)
    logger.info(
        "Saved %s manifest cache to %s (%d rows, %d samples)",
        split_name,
        manifest_path,
        len(row_groups),
        sum(len(group.windows) for group in row_groups),
    )
    return row_groups


def write_history_cf_cache(
    history_cf_rows: list[torch.Tensor],
    cache_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Persist per-row history_cf tensors to HDF5."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not overwrite:
        logger.info("history_cf cache already exists at %s, reusing it", cache_path)
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    n_features = int(history_cf_rows[0].shape[0]) if history_cf_rows else 0

    with h5py.File(cache_path, "w") as handle:
        rows_group = handle.create_group("history_cf_rows")
        for row_idx, history_cf in enumerate(history_cf_rows):
            rows_group.create_dataset(
                str(row_idx),
                data=history_cf.cpu().numpy().astype(np.float32),
                compression="gzip",
            )

        handle.attrs["n_features"] = n_features
        handle.attrs["n_rows"] = len(history_cf_rows)

    logger.info("Saved history_cf H5 cache for %d rows to %s", len(history_cf_rows), cache_path)
    return cache_path


def write_history_cf_cache_from_dataset(
    split_ds: hf_ds.Dataset,
    cache_path: str | Path,
    features_config: FeaturesConfig,
    model_config: ModelConfig,
    *,
    overwrite: bool = False,
) -> Path:
    """Stream raw history_cf tensors from a split dataset into HDF5."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not overwrite:
        logger.info("history_cf cache already exists at %s, reusing it", cache_path)
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    extractor = MultivariateFeatureExtractor(
        config=features_config,
        forecasting_length=model_config.n_pred_steps,
    )
    zero_to_nan_transform = ZeroToNaNTransform()

    n_rows = 0
    n_features = int(model_config.n_features)
    with h5py.File(cache_path, "w") as handle:
        rows_group = handle.create_group("history_cf_rows")
        for row_idx, row in enumerate(split_ds):
            features = extractor.extract(row)
            history_cf = torch.as_tensor(features["history"], dtype=torch.float32)
            history_cf = zero_to_nan_transform(history_cf)
            if row_idx == 0:
                n_features = int(history_cf.shape[0])
            rows_group.create_dataset(
                str(row_idx),
                data=history_cf.cpu().numpy().astype(np.float32),
                compression="gzip",
            )
            n_rows += 1

        handle.attrs["n_features"] = n_features
        handle.attrs["n_rows"] = n_rows

    logger.info("Saved streaming history_cf H5 cache for %d rows to %s", n_rows, cache_path)
    return cache_path


def precompute_history_cf_cache(
    split_ds: hf_ds.Dataset,
    cache_path: str | Path,
    features_config: FeaturesConfig,
    model_config: ModelConfig,
    scaler_stats: ChannelStandardScalerStats | None = None,
    overwrite: bool = False,
) -> Path:
    """Materialize one history_cf tensor per split row and store it in HDF5."""
    history_cf_rows = build_history_cf_rows(split_ds, features_config, model_config)
    if scaler_stats is not None:
        history_cf_rows = [scaler_stats.transform_history_cf(row) for row in history_cf_rows]
    return write_history_cf_cache(history_cf_rows, cache_path, overwrite=overwrite)


def _resolve_window_hours(
    *,
    current_day: int,
    horizon_length: int,
    daily_start_hour_offset: int,
) -> tuple[int, int]:
    history_end_hour = current_day * 24 + daily_start_hour_offset
    pred_end_hour = history_end_hour + horizon_length
    return history_end_hour, pred_end_hour


def resolve_runtime_row_groups(
    *,
    row_groups: list[ForecastingRowGroup],
    row_lengths_by_idx: dict[int, int],
    history_length: int,
    horizon_length: int,
    daily_start_hour_offset: int,
    include_short_history: bool = True,
) -> list[ForecastingRowGroup]:
    """Resolve offset-adjusted windows without changing persisted manifest data.

    When ``include_short_history`` is True (the default), windows whose history
    is shorter than ``history_length`` are kept — the dataset NaN-left-pads them
    to the fixed window, matching how the evaluator feeds short prefixes to
    ``BasePyPOTSForecastingModel.predict``. Set False to reproduce the legacy
    drop-short behavior.
    """
    resolved_row_groups: list[ForecastingRowGroup] = []

    for row_group in row_groups:
        trajectory_length = int(row_lengths_by_idx.get(int(row_group.dataset_row_idx), 0))
        resolved_windows: list[ForecastingWindowDescriptor] = []
        for window in row_group.windows:
            history_end_hour, pred_end_hour = _resolve_window_hours(
                current_day=int(window.current_day),
                horizon_length=horizon_length,
                daily_start_hour_offset=daily_start_hour_offset,
            )
            if history_end_hour <= 0:
                continue
            if not include_short_history and history_end_hour < history_length:
                continue
            if pred_end_hour > trajectory_length:
                continue
            resolved_windows.append(
                ForecastingWindowDescriptor(
                    current_day=int(window.current_day),
                    history_end_hour=history_end_hour,
                    pred_end_hour=pred_end_hour,
                )
            )

        if resolved_windows:
            resolved_row_groups.append(
                ForecastingRowGroup(
                    dataset_row_idx=int(row_group.dataset_row_idx),
                    user_id=str(row_group.user_id),
                    windows=tuple(resolved_windows),
                )
            )

    return resolved_row_groups


class PyPOTSForecastingDataset(BaseDataset):
    """Manifest-backed forecasting Dataset aligned with PyPOTS BaseDataset semantics."""

    def __init__(
        self,
        history_cf_source: str | Path | list[torch.Tensor],
        row_groups: list[ForecastingRowGroup],
        model_config: ModelConfig,
        daily_start_hour_offset: int = 0,
        include_short_history: bool = True,
    ) -> None:
        """Initialize a dataset that slices forecasting windows on demand."""
        # We intentionally don't call BaseDataset.__init__ because our data source is
        # a manifest over a precomputed per-row cache rather than an in-memory dict / H5 file.
        self._history_length = int(model_config.n_steps)
        self._horizon_length = int(model_config.n_pred_steps)
        self._daily_start_hour_offset = int(daily_start_hour_offset)
        self._include_short_history = bool(include_short_history)
        if not 0 <= self._daily_start_hour_offset < 24:
            raise ValueError(
                "daily_start_hour_offset must be within [0, 24), "
                f"but received {self._daily_start_hour_offset}"
            )
        self._history_cf_source = history_cf_source
        self._history_cf_rows = self._load_history_cf_rows(history_cf_source)
        self._history_cf_file_handle = None
        self._cached_row_idx: int | None = None
        self._cached_row_tensor: torch.Tensor | None = None
        self._row_groups = resolve_runtime_row_groups(
            row_groups=row_groups,
            row_lengths_by_idx=self._build_row_lengths_by_idx(history_cf_source),
            history_length=self._history_length,
            horizon_length=self._horizon_length,
            daily_start_hour_offset=self._daily_start_hour_offset,
            include_short_history=self._include_short_history,
        )
        self._sample_lookup = self._build_sample_lookup(self._row_groups)
        self.data = None
        self.return_X_ori = False
        self.return_X_pred = True
        self.return_y = False
        self.file_type = "online"
        self.fetch_data = self._fetch_data_from_manifest
        self.n_samples = len(self._sample_lookup)
        self.n_steps = self._history_length
        self.n_features = int(model_config.n_features)
        self.n_pred_steps = self._horizon_length
        self.n_pred_features = int(model_config.n_features)

    @staticmethod
    def _load_history_cf_rows(history_cf_source: str | Path | list[torch.Tensor]) -> list[torch.Tensor] | None:
        if isinstance(history_cf_source, list):
            return history_cf_source

        history_cf_source = Path(history_cf_source)
        if history_cf_source.suffix.lower() != ".h5":
            raise ValueError(f"Unsupported history_cf cache format: {history_cf_source}")
        return None

    @staticmethod
    def _build_row_lengths_by_idx(
        history_cf_source: str | Path | list[torch.Tensor],
    ) -> dict[int, int]:
        if isinstance(history_cf_source, list):
            return {idx: int(row.shape[1]) for idx, row in enumerate(history_cf_source)}

        history_cf_path = Path(history_cf_source)
        with h5py.File(history_cf_path, "r") as handle:
            rows_group = handle["history_cf_rows"]
            return {
                int(row_idx): int(rows_group[row_idx].shape[1])
                for row_idx in rows_group.keys()
            }

    @staticmethod
    def _build_sample_lookup(
        row_groups: list[ForecastingRowGroup],
    ) -> list[tuple[int, int]]:
        sample_lookup: list[tuple[int, int]] = []
        for group_idx, row_group in enumerate(row_groups):
            for window_idx, _window in enumerate(row_group.windows):
                sample_lookup.append((group_idx, window_idx))
        return sample_lookup

    def build_batch_sampler(self, batch_size: int, shuffle: bool) -> BatchSampler:
        """Create a sampler that keeps windows from the same row adjacent."""
        return ForecastingRowGroupedBatchSampler(
            row_groups=self._row_groups,
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def _open_history_cf_file_handle(self):
        if self._history_cf_file_handle is None:
            if isinstance(self._history_cf_source, list):
                return None
            self._history_cf_file_handle = h5py.File(self._history_cf_source, "r")
        return self._history_cf_file_handle

    def _get_history_cf_row(self, dataset_row_idx: int) -> torch.Tensor:
        if self._cached_row_idx == dataset_row_idx and self._cached_row_tensor is not None:
            return self._cached_row_tensor

        if self._history_cf_rows is not None:
            history_cf = self._history_cf_rows[dataset_row_idx]
        else:
            handle = self._open_history_cf_file_handle()
            if handle is None or "history_cf_rows" not in handle:
                raise ValueError(f"Invalid history_cf cache format: {self._history_cf_source}")
            history_cf = torch.from_numpy(handle["history_cf_rows"][str(dataset_row_idx)][...]).to(
                torch.float32
            )

        self._cached_row_idx = dataset_row_idx
        self._cached_row_tensor = history_cf
        return history_cf

    def __len__(self) -> int:
        """Return the number of available forecasting samples."""
        return len(self._sample_lookup)

    def _fetch_data_from_manifest(self, idx: int) -> tuple[torch.Tensor, ...]:
        group_idx, window_idx = self._sample_lookup[idx]
        row_group = self._row_groups[group_idx]
        window = row_group.windows[window_idx]
        history_cf = self._get_history_cf_row(row_group.dataset_row_idx)

        # Match the evaluator's predict(): when the available history is shorter
        # than the fixed window, NaN-left-pad to ``history_length`` (older
        # positions missing) instead of slicing with a negative start. With
        # include_short_history enabled, this is the training-time mirror of
        # BasePyPOTSForecastingModel.predict's padding.
        start = window.history_end_hour - self._history_length
        if start < 0:
            available = history_cf[:, : window.history_end_hour]
            history_window = torch.full(
                (available.shape[0], self._history_length),
                float("nan"),
                dtype=available.dtype,
            )
            history_window[:, -available.shape[1] :] = available
        else:
            history_window = history_cf[:, start : window.history_end_hour]
        target_window = history_cf[:, window.history_end_hour : window.pred_end_hour]

        x = history_window.transpose(0, 1).contiguous()
        x_pred = target_window.transpose(0, 1).contiguous()
        x, missing_mask = fill_and_get_mask_torch(x)
        x_pred, x_pred_missing_mask = fill_and_get_mask_torch(x_pred)

        return (
            torch.tensor(idx, dtype=torch.int64),
            x,
            missing_mask,
            x_pred,
            x_pred_missing_mask,
        )


class ForecastingRowGroupedBatchSampler(Sampler[list[int]]):
    """Yield sample-index batches while keeping samples from the same row adjacent."""

    def __init__(
        self,
        row_groups: list[ForecastingRowGroup],
        batch_size: int,
        shuffle: bool,
    ) -> None:
        """Initialize a row-group-aware batch sampler."""
        self._row_groups = row_groups
        self._batch_size = int(batch_size)
        self._shuffle = shuffle

    def __iter__(self):
        """Yield batches of sample indices with row-local adjacency."""
        row_indices = list(range(len(self._row_groups)))
        if self._shuffle:
            permutation = torch.randperm(len(row_indices)).tolist()
            row_indices = [row_indices[i] for i in permutation]

        batch: list[int] = []
        running_sample_idx = 0
        group_start_indices: list[int] = []
        for row_group in self._row_groups:
            group_start_indices.append(running_sample_idx)
            running_sample_idx += len(row_group.windows)

        for row_group_idx in row_indices:
            row_group = self._row_groups[row_group_idx]
            sample_indices = list(
                range(
                    group_start_indices[row_group_idx],
                    group_start_indices[row_group_idx] + len(row_group.windows),
                )
            )
            if self._shuffle and len(sample_indices) > 1:
                sample_indices = [sample_indices[i] for i in torch.randperm(len(sample_indices)).tolist()]

            for sample_idx in sample_indices:
                batch.append(sample_idx)
                if len(batch) == self._batch_size:
                    yield batch
                    batch = []

        if batch:
            yield batch

    def __len__(self) -> int:
        """Return the number of batches produced by this sampler."""
        total_samples = sum(len(group.windows) for group in self._row_groups)
        return (total_samples + self._batch_size - 1) // self._batch_size


def build_pypots_forecasting_dataset(
    split_ds: hf_ds.Dataset,
    sample_index_file: str | Path | None,
    model_config: ModelConfig,
    features_config: FeaturesConfig,
    daily_start_hour_offset: int = 0,
    history_cf_source: str | Path | list[torch.Tensor] | None = None,
    row_groups: list[ForecastingRowGroup] | None = None,
    include_short_history: bool = True,
) -> PyPOTSForecastingDataset:
    """Construct an online forecasting Dataset for one split."""
    if row_groups is None:
        if sample_index_file is None:
            raise ValueError("sample_index_file is required when row_groups is not provided")
        row_groups = ForecastingSampleIndexBuilder(
            split_ds=split_ds,
            sample_index_file=sample_index_file,
            n_steps=model_config.n_steps,
            n_pred_steps=model_config.n_pred_steps,
        ).build_row_groups()
    if history_cf_source is None:
        history_cf_source = [
            ZeroToNaNTransform()(
                torch.as_tensor(
                    MultivariateFeatureExtractor(
                        config=features_config,
                        forecasting_length=model_config.n_pred_steps,
                    ).extract(row)["history"],
                    dtype=torch.float32,
                )
            )
            for row in split_ds
        ]
    return PyPOTSForecastingDataset(
        history_cf_source=history_cf_source,
        row_groups=row_groups,
        model_config=model_config,
        daily_start_hour_offset=daily_start_hour_offset,
        include_short_history=include_short_history,
    )
