"""Configuration dataclasses for forecasting PyPOTS training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from forecasting_evaluation.config import DataConfig, FeaturesConfig, ForecastingConfig


@dataclass
class H5ExportConfig:
    """Deprecated HDF5 export configuration for forecasting samples."""

    output_dir: str = "data/processed/forecasting_pypots_h5"
    overwrite: bool = False
    chunk_size: int = 512


@dataclass
class ModelConfig:
    """PyPOTS forecasting model configuration.

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


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    finetune_mode: Literal["full", "lora"] = "full"
    whether_standardscaler: bool = False
    epochs: int = 50
    batch_size: int = 64
    patience: int = 10
    optimizer_lr: float = 1.0e-3
    device: str = "auto"
    disable_data_parallel: bool = True
    # LoRA-specific (Chronos-2)
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0


@dataclass
class OutputConfig:
    """Model saving and tracking configuration."""

    saving_path: str = "models/forecasting_pypots"
    finetuned_ckpt_name: str = "finetuned-ckpt"
    model_saving_strategy: Literal["best", "better", "all"] = "best"
    upload_wandb_artifact: bool = True
    wandb_project: str = "mhc-forecasting"
    wandb_entity: str = "MHC_Dataset"


@dataclass
class ForecastingPyPOTSTrainingConfig:
    """Root configuration for forecasting PyPOTS training."""

    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    forecasting: ForecastingConfig = field(default_factory=ForecastingConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    h5_export: H5ExportConfig = field(default_factory=H5ExportConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def __post_init__(self) -> None:
        """Normalize output paths derived from runtime-only forecasting settings."""
        offset = int(self.forecasting.daily_start_hour_offset)
        if offset == 0:
            return

        output_path = Path(self.output.saving_path)
        suffix = f"_{offset}"
        if not output_path.name.endswith(suffix):
            self.output.saving_path = str(output_path.with_name(f"{output_path.name}{suffix}"))
