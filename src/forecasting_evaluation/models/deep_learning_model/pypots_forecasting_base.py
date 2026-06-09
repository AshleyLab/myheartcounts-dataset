"""Shared adapter base for loading PyPOTS forecasting checkpoints."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from forecasting_evaluation.config import FeaturesConfig
from forecasting_evaluation.forecasting_training.online_dataset import (
    history_cf_cache_subdir,
    resolve_cache_base_dir,
)
from forecasting_evaluation.forecasting_training.standard_scaler import (
    ChannelStandardScalerStats,
    load_stats_json,
)
from forecasting_evaluation.models.base import BasePredictionModel

logger = logging.getLogger(__name__)

WANDB_PREFIX = "wandb:"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "mhc-benchmark" / "artifacts"


def infer_n_features(features_config: FeaturesConfig) -> int:
    """Infer feature count from the configured forecasting feature selection."""
    if features_config.channel != "all":
        raise ValueError(f"Unknown channel type: {features_config.channel}")
    return 19


def resolve_checkpoint_path(path: str, cache_dir: str | Path | None = None) -> Path:
    """Resolve a local checkpoint path or download a W&B model artifact."""
    if not path.startswith(WANDB_PREFIX):
        local = Path(path)
        if not local.exists():
            raise FileNotFoundError(f"Checkpoint not found: {local}")
        return local

    import wandb

    artifact_ref = path[len(WANDB_PREFIX) :]
    filename_selector = None
    if "#" in artifact_ref:
        artifact_ref, filename_selector = artifact_ref.rsplit("#", 1)

    if artifact_ref.count("/") < 2:
        raise ValueError(
            f"Malformed wandb artifact reference: '{path}'. "
            "Expected format: wandb:ENTITY/PROJECT/ARTIFACT:VERSION[#filename]"
        )

    cache_root = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    safe_name = artifact_ref.replace("/", "_").replace(":", "_")
    artifact_root = cache_root / safe_name
    artifact_root.mkdir(parents=True, exist_ok=True)

    artifact = wandb.Api().artifact(artifact_ref, type="model")
    artifact_dir = Path(artifact.download(root=str(artifact_root)))

    if filename_selector:
        resolved = artifact_dir / filename_selector
        if not resolved.exists():
            raise FileNotFoundError(f"Requested file '{filename_selector}' not found in {artifact_dir}")
        return resolved

    model_files = sorted(
        p
        for p in artifact_dir.rglob("*.pypots")
        if not p.name.startswith("events.out.tfevents")
    )
    if not model_files:
        raise FileNotFoundError(f"No .pypots file found in artifact directory {artifact_dir}")
    return model_files[0]


class BasePyPOTSForecastingModel(BasePredictionModel, ABC):
    """Common adapter for PyPOTS forecasting models used by evaluator."""

    def __init__(self, checkpoint_path: str, model_name: str) -> None:
        """Resolve checkpoint metadata and load the trained PyPOTS model."""
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self._checkpoint_dir = resolve_checkpoint_path(self.checkpoint_path)
        self._training_config = self._load_training_config(self._checkpoint_dir)
        self._scaler_stats = self._load_scaler_stats()
        # The harness feeds raw history and this model standardizes internally, so
        # missing scaler stats can no longer be caught by the eval cache builder.
        # Fail fast here instead of silently predicting in standardized space.
        if self.uses_standard_scaler and self._scaler_stats is None:
            raise FileNotFoundError(
                "PyPOTS checkpoint was trained with standard scaling, but "
                "standard_scaler_stats.json could not be resolved from the saved "
                "training config or co-located with the checkpoint. Predictions "
                "would otherwise be produced in standardized space."
            )
        self._model = self._load_model()

    def _load_model(self):
        """Instantiate the concrete PyPOTS model and load checkpoint weights."""
        model_file = self._resolve_model_file(self._checkpoint_dir)
        model = self.build_model()
        model.load(str(model_file))
        logger.info("Loaded %s checkpoint from %s", self.model_name, model_file)
        return model

    def _resolve_model_file(self, checkpoint_path: Path) -> Path:
        """Resolve a checkpoint path that may point to a directory or a file."""
        if checkpoint_path.is_file():
            return checkpoint_path

        pypots_files = sorted(
            p
            for p in checkpoint_path.rglob("*.pypots")
            if not p.name.startswith("events.out.tfevents")
        )
        if not pypots_files:
            raise FileNotFoundError(f"No .pypots file found under {checkpoint_path}")
        return pypots_files[0]

    def _load_training_config(self, checkpoint_path: Path) -> dict:
        """Load training config saved beside the checkpoint, if available."""
        config_dir = checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent
        json_path = config_dir / "training_config.json"
        yaml_path = config_dir / "training_config.yaml"

        if json_path.exists():
            with json_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)

        if yaml_path.exists():
            try:
                import yaml

                with yaml_path.open("r", encoding="utf-8") as handle:
                    data = yaml.safe_load(handle)
                return data or {}
            except ImportError:
                logger.warning("PyYAML not available; cannot read %s", yaml_path)

        return {}

    def _get_training_config_value(self, section: str, key: str):
        """Fetch a value from the saved training config if present."""
        section_data = self._training_config.get(section, {})
        if isinstance(section_data, dict):
            return section_data.get(key)
        return None

    @property
    def training_daily_start_hour_offset(self) -> int:
        """Return the training-time runtime slicing offset saved with the checkpoint."""
        value = self._get_training_config_value("forecasting", "daily_start_hour_offset")
        if value is None:
            return 0
        return int(value)

    @property
    def uses_standard_scaler(self) -> bool:
        """Whether the checkpoint was trained on standardized history_cf windows."""
        return bool(self._get_training_config_value("training", "whether_standardscaler"))

    @property
    def scaler_stats(self) -> ChannelStandardScalerStats | None:
        """Return loaded training-time scaler stats when available."""
        return self._scaler_stats

    def _load_scaler_stats(self) -> ChannelStandardScalerStats | None:
        """Load training-time StandardScaler stats for inverse-transforming predictions."""
        if not self.uses_standard_scaler:
            return None

        # Self-contained release bundles co-locate the scaler stats with the
        # checkpoint. Prefer that over the content-addressed training cache so a
        # downloaded bundle works without rebuilding the dataset-derived cache.
        colocated = self._colocated_scaler_stats_path()
        if colocated is not None and colocated.exists():
            return load_stats_json(colocated)

        data_config = self._training_config.get("data")
        model_config = self._training_config.get("model")
        features_config = self._training_config.get("features")
        h5_export_config = self._training_config.get("h5_export")
        if not isinstance(data_config, dict) or not isinstance(model_config, dict):
            logger.warning(
                "Checkpoint %s indicates standard scaling, but training config is incomplete; "
                "predictions will remain in standardized space",
                self._checkpoint_dir,
            )
            return None

        if not isinstance(features_config, dict):
            features_config = {"channel": "all"}
        if not isinstance(h5_export_config, dict):
            h5_export_config = {}

        try:
            data_config_ns = SimpleNamespace(**data_config)
            # Honor a baked training output_dir for existing checkpoints; otherwise
            # resolve under the configured data root ({data_root}/cache/forecasting).
            base_output_dir = h5_export_config.get("output_dir") or resolve_cache_base_dir(
                data_config_ns
            )
            cache_dir = history_cf_cache_subdir(
                base_dir=Path(base_output_dir) / "history_cf_cache",
                data_config=data_config_ns,
                model_config=SimpleNamespace(**model_config),
                features_config=SimpleNamespace(**features_config),
            )
            stats_path = cache_dir / "standard_scaler_stats.json"
            if not stats_path.exists():
                logger.warning(
                    "Checkpoint %s expects scaler stats at %s, but the file does not exist",
                    self._checkpoint_dir,
                    stats_path,
                )
                return None
            return load_stats_json(stats_path)
        except Exception as exc:  # pragma: no cover - defensive path
            logger.warning("Failed to load scaler stats for %s: %s", self._checkpoint_dir, exc)
            return None

    def _colocated_scaler_stats_path(self) -> Path | None:
        """Return the bundle-local scaler stats path, if the checkpoint has one.

        Release bundles ship ``standard_scaler_stats.json`` next to the
        ``.pypots`` checkpoint (in the checkpoint directory, or beside the file
        when the checkpoint path points directly at the ``.pypots``).
        """
        base = self._checkpoint_dir if self._checkpoint_dir.is_dir() else self._checkpoint_dir.parent
        return base / "standard_scaler_stats.json"

    def _inverse_transform_point_forecast(self, point_result: np.ndarray) -> np.ndarray:
        """Map standardized point forecasts back to the original value space."""
        if self._scaler_stats is None:
            return point_result
        tensor = torch.as_tensor(point_result, dtype=torch.float32)
        return self._scaler_stats.inverse_transform_history_cf(tensor).cpu().numpy()

    def _inverse_transform_quantile_forecast(self, quantiles_result: np.ndarray) -> np.ndarray:
        """Map standardized quantile forecasts back to the original value space."""
        if self._scaler_stats is None:
            return quantiles_result
        restored_quantiles = []
        for quantile_idx in range(quantiles_result.shape[2]):
            tensor = torch.as_tensor(quantiles_result[:, :, quantile_idx], dtype=torch.float32)
            restored_quantiles.append(
                self._scaler_stats.inverse_transform_history_cf(tensor).cpu().numpy()
            )
        return np.stack(restored_quantiles, axis=2)

    @abstractmethod
    def build_model(self):
        """Build a concrete PyPOTS model instance with matching architecture settings."""

    @property
    @abstractmethod
    def n_steps(self) -> int:
        """Return the fixed history length expected by the concrete model."""

    def predict(
        self,
        history: np.ndarray,
        horizon: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Forecast with the loaded PyPOTS model (PyPOTS missing-value idiom).

        The harness passes the **raw** full-prefix history (shape
        ``(n_features, history_length)``, possibly shorter than ``n_steps`` and
        possibly containing NaN). This model owns its own input handling:
        standardization (when the checkpoint was trained on standardized
        windows), trailing fixed-window selection, and NaN left-padding for
        short histories.

        Short histories are **NaN-left-padded** to ``n_steps`` rather than
        dropped or returned as NaN. This is the PyPOTS idiom for partial
        observation: PyPOTS' loader (``pygrinder.fill_and_get_mask_torch``,
        ``nan=0``) fills ``NaN -> 0`` and builds a ``missing_mask`` the model
        consumes (``SaitsEmbedding`` concatenates it onto ``X``). In
        standardized space ``0`` is the channel mean, so a short history is just
        "the leading positions are missing" fed through the exact mechanism the
        checkpoint was trained on. The forecast is therefore **finite** — these
        models do not return NaN for short history and do not trigger the
        evaluator's Seasonal-Naive fallback.

        Caveat: training dropped sub-``n_steps`` windows, so a very short history
        is an in-mechanism extrapolation (a quality nuance, not a correctness
        issue); no retraining is required.

        Args:
            history: Full-prefix history of shape (n_features, history_length).
            horizon: Number of future hours requested (the model emits its
                trained ``n_pred_steps``; the evaluator asserts they match).
        """
        history_cf = np.asarray(history, dtype=np.float32)
        # The harness feeds raw history to every model; standardize internally
        # to match training-time input space. Channel-wise affine scaling is
        # applied per timestep, so transform-then-slice is equivalent to the
        # previous slice-of-pre-standardized-cache path.
        if self._scaler_stats is not None:
            history_cf = (
                self._scaler_stats.transform_history_cf(
                    torch.as_tensor(history_cf, dtype=torch.float32)
                )
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
        if history_cf.shape[1] < self.n_steps:
            # Match training-time fixed-window shape while preserving all available
            # recent history: left-pad the older missing context with NaNs.
            padded_history = np.full(
                (history_cf.shape[0], self.n_steps),
                np.nan,
                dtype=np.float32,
            )
            padded_history[:, -history_cf.shape[1] :] = history_cf
            history_cf = padded_history

        # Match training-time H5 export: keep only the most recent fixed window.
        model_input = history_cf[:, -self.n_steps :].T
        result = self._model.predict({"X": model_input[None, ...]})
        forecasting = result["forecasting"]

        if hasattr(forecasting, "detach"):
            forecasting = forecasting.detach().cpu().numpy()
        else:
            forecasting = np.asarray(forecasting)

        point_result = forecasting[0].T.astype(np.float32, copy=False)
        point_result = self._inverse_transform_point_forecast(point_result).astype(
            np.float32,
            copy=False,
        )
        quantiles_result = None
        return point_result, quantiles_result
