"""Configuration dataclasses for PyPOTS model training.

Reuses :class:`imputation_evaluation.config.DataConfig` so train and eval
share the same split semantics, daily-hf path, and QA filters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from imputation_evaluation.config import DataConfig


@dataclass
class H5ExportConfig:
    """HDF5 export configuration (PyPOTS consumes data via H5).

    Attributes:
        output_dir: Base directory for cached H5 files. The runner
            creates an 8-char content-addressed subdir under this so
            different split/preprocessing combinations never collide.
        overwrite: If False, skip re-export when all H5 files already exist.
        chunk_size: Samples per H5 write chunk.
        val_mask_ratio: Fraction of observed values to artificially mask
            in val/test for PyPOTS's own per-epoch validation metric.
            Independent from the openmhc evaluation masks.
        normalize: If True, z-score continuous channels (0-6) using
            ``normalization_stats.json`` from the dataset cache. The
            stats file is copied alongside the H5 files so inference can
            denormalize consistently.
        stats_max_samples: Deprecated, retained for back-compat.
    """

    output_dir: str = "data/pypots_h5"
    overwrite: bool = False
    chunk_size: int = 1000
    val_mask_ratio: float = 0.2
    normalize: bool = True
    stats_max_samples: int | None = 10000


@dataclass
class ModelConfig:
    """PyPOTS model configuration — superset across all supported models.

    Most fields are model-specific. The factory in
    :mod:`imputation_training.model_registry` only reads the fields
    relevant to ``model_name``; extra fields are ignored. Defaults match
    PyPOTS's own defaults where possible.
    """

    model_name: str = "brits"
    n_steps: int = 1440
    n_features: int = 19

    # BRITS-specific
    rnn_hidden_size: int = 128

    # Shared by Transformer-family models (TimesNet, FEDformer)
    n_layers: int = 2
    d_model: int = 64
    n_heads: int = 4
    d_ffn: int = 64
    dropout: float = 0.1
    apply_nonstationary_norm: bool = False

    # TimesNet-specific
    top_k: int = 5
    n_kernels: int = 6

    # DLinear-specific
    moving_avg_window_size: int = 25
    individual: bool = False
    ORT_weight: float = 1.0
    MIT_weight: float = 1.0

    # FEDformer-specific (note: PyPOTS calls its own field ``version`` —
    # we forward this through unchanged; the imputation-eval CLI exposes
    # it as ``variant`` to avoid colliding with the openmhc dataset
    # version).
    version: str = "Fourier"
    modes: int = 32
    mode_select: str = "random"


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    epochs: int = 100
    batch_size: int = 32
    patience: int = 10
    optimizer_lr: float = 1e-3
    weight_decay: float = 0.0
    clip_grad_norm: float | None = None
    device: str = "auto"
    num_workers: int = 4


@dataclass
class OutputConfig:
    """Where to write the trained model + release bundle."""

    # Per-run scratch dir where PyPOTS saves its raw .pypots file mid-training.
    saving_path: str = "models/pypots"
    model_saving_strategy: str = "best"  # "best", "better", "all"

    # The packaged openmhc release dir (manifest + sidecar). When None,
    # the runner derives it from saving_path.
    release_dir: str | None = None

    # Optional W&B logging.
    wandb_enabled: bool = False
    wandb_project: str = "openmhc-pypots"
    wandb_entity: str | None = None


@dataclass
class PyPOTSTrainingConfig:
    """Root configuration for PyPOTS model training."""

    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    h5_export: H5ExportConfig = field(default_factory=H5ExportConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
