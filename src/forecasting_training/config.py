"""Configuration dataclasses for forecasting PyPOTS model training.

Reuses :class:`forecasting_evaluation.config.DataConfig`,
:class:`~forecasting_evaluation.config.ForecastingConfig`, and
:class:`~forecasting_evaluation.config.FeaturesConfig` so train and eval share
the same trajectory paths, splits, sample index, and feature selection. The
dependency direction is one-way (``forecasting_training`` → ``forecasting_evaluation``);
eval never imports training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from forecasting_evaluation.config import DataConfig, FeaturesConfig, ForecastingConfig


@dataclass
class ModelConfig:
    """PyPOTS forecasting model architecture (DLinear / MixLinear / SegRNN).

    The factory in :mod:`forecasting_training.model_registry` reads only the
    fields relevant to ``model_name``; the rest are ignored. ``n_steps`` is the
    input history length (hours), ``n_pred_steps`` the forecast horizon, and
    ``n_features`` the channel count — these three also key the shared
    history_cf cache (:func:`forecasting_evaluation.data.online_dataset.history_cf_cache_subdir`).
    """

    model_name: str = "dlinear"  # one of: dlinear, mixlinear, segrnn
    n_steps: int = 168
    n_pred_steps: int = 24
    n_features: int = 19

    # DLinear-specific (``d_model`` is shared with SegRNN).
    moving_avg_window_size: int = 25
    individual: bool = False
    d_model: int = 64

    # MixLinear-specific.
    period_len: int = 24
    lpf: int = 2
    alpha: float = 0.5
    rank: int = 2

    # SegRNN-specific (``dropout`` only used by SegRNN here).
    seg_len: int = 24
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    epochs: int = 50
    batch_size: int = 64
    patience: int = 10
    optimizer_lr: float = 1.0e-3
    device: str = "auto"
    num_workers: int = 4
    clip_grad_norm: float | None = None
    # Standardize history_cf with a train-fit channel StandardScaler. The bundle
    # ships the fitted stats and the eval side inverse-transforms with them, so
    # this must stay consistent with what `release.write_release` packages.
    whether_standardscaler: bool = True
    # Train on the same padded-short-history distribution the evaluator feeds:
    # keep windows whose history is shorter than ``n_steps`` (NaN-left-padded)
    # instead of dropping them. Default ON for fair benchmark comparison.
    include_short_history: bool = True


@dataclass
class OutputConfig:
    """Where to write the trained model + release bundle, and W&B options."""

    # Per-run scratch dir where PyPOTS writes its raw .pypots file mid-training.
    saving_path: str = "models/forecasting_pypots"
    model_saving_strategy: str = "best"  # "best", "better", "all"

    # The packaged openmhc release dir (manifest + scaler stats + training_config).
    # When None, the runner derives it from saving_path.
    release_dir: str | None = None

    # Optional W&B logging (mirrors imputation_training). When enabled, PyPOTS'
    # TensorBoard scalars stream into the run via sync_tensorboard=True.
    wandb_enabled: bool = False
    wandb_project: str = "mhc-forecasting"
    wandb_entity: str | None = None
    # When true (and W&B is enabled), log the release bundle as a W&B artifact.
    upload_wandb_artifact: bool = False


@dataclass
class ForecastingTrainingConfig:
    """Root configuration for forecasting PyPOTS model training."""

    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    forecasting: ForecastingConfig = field(default_factory=ForecastingConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def __post_init__(self) -> None:
        """Suffix the saving path with a non-zero daily start-hour offset.

        Mirrors the former ``ForecastingPyPOTSTrainingConfig`` behavior so
        offset-shifted runs don't overwrite the offset-0 checkpoint. No-op at
        the default offset of 0.
        """
        offset = int(self.forecasting.daily_start_hour_offset)
        if offset == 0:
            return
        output_path = Path(self.output.saving_path)
        suffix = f"_{offset}"
        if not output_path.name.endswith(suffix):
            self.output.saving_path = str(output_path.with_name(f"{output_path.name}{suffix}"))
