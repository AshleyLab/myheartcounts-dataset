"""Configuration dataclasses for imputation evaluation.

Provides a structured configuration schema for evaluating imputation methods
across different masking scenarios on daily sensor data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration for daily data."""

    zero_to_nan: bool = True  # Apply ZeroToNaNTransform (HR=0→NaN, all-zero steps/dist/energy→NaN)


@dataclass
class FilterConfig:
    """Quality filtering configuration."""

    min_wear_fraction: float = 0.5  # Remove days with < 50% wear-time (0.0 disables)
    variance_filter_enabled: bool = True  # Remove days with near-zero channel variance
    variance_thresholds: dict[int, float] | None = None  # Uses DEFAULT_VARIANCE_THRESHOLDS if None


@dataclass
class DataConfig:
    """Data source and splitting configuration."""

    daily_hf_dir: str = "data/hf_daily"
    split_file: str | None = None
    train_ratio: float = 0.6
    val_ratio: float = 0.1
    split_seed: int = 42
    max_samples_per_split: int | None = (
        None  # Limit samples per split for testing (None = no limit)
    )

    # DataLoader parameters (for data loading, mask generation, method fitting)
    batch_size: int = 5000
    num_workers: int = 4  # Worker processes for DataLoader prefetching
    pin_memory: bool = True

    # Evaluation parallelism (separate from DataLoader workers)
    # Each worker process handles ALL scenarios for its assigned batch (batch-level parallelism)
    num_eval_workers: int = 1  # Parallel processes for batch evaluation (1 = sequential)
    num_eval_dl_workers: int | None = None  # DataLoader workers for eval (None = use num_workers)

    # Multi-day context window
    n_days: int = 1  # Number of days per sample window (1-7)

    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)

    def __post_init__(self):
        """Validate configuration values."""
        if not 1 <= self.n_days <= 7:
            raise ValueError(f"n_days must be 1-7, got {self.n_days}")


@dataclass
class RandomNoiseConfig:
    """Random noise masking configuration."""

    enabled: bool = True
    patch_size: int = 30  # Consecutive minutes per patch
    mask_ratio: float = 0.5  # Fraction of valid data to mask


@dataclass
class TemporalSliceConfig:
    """Temporal slice masking configuration."""

    enabled: bool = True
    mask_ratio: float = 0.25  # Fraction of valid timesteps to mask
    min_block_size: int = 30  # Min contiguous block (minutes)
    max_block_size: int = 60  # Max contiguous block (minutes)


@dataclass
class SignalSliceConfig:
    """Signal slice masking configuration."""

    enabled: bool = True
    mask_ratio: float = 0.5  # Fraction of channels to drop (Mode A)
    device_groups: dict[str, list[int]] = field(
        default_factory=lambda: {
            "iphone": [0, 1, 2],
            "watch": [3, 4, 5, 6],
        }
    )


@dataclass
class SleepGapConfig:
    """Sleep gap masking configuration."""

    enabled: bool = True
    asleep_channel: int = 7
    inbed_channel: int = 8


@dataclass
class WorkoutGapConfig:
    """Workout gap masking configuration."""

    enabled: bool = True
    mask_channels: list[int] = field(default_factory=lambda: [5, 6])  # HR + Active Energy
    workout_channels: list[int] = field(default_factory=lambda: list(range(9, 19)))


@dataclass
class IntensityFailureConfig:
    """Intensity failure masking configuration."""

    enabled: bool = True
    hr_channel: int = 5
    hr_threshold: float = 160.0  # In BPM; auto-converted to Hz if data is in Hz
    hr_unit: str = "auto"  # "auto" | "bpm" | "hz" — auto-detects from data
    mask_channels: list[int] = field(default_factory=lambda: [5, 6])  # HR + Active Energy


@dataclass
class MaskingConfig:
    """Masking scenario configuration."""

    mask_seed: int = 42
    masks_file: str | None = None  # Load pre-computed masks from .npz
    random_noise: RandomNoiseConfig = field(default_factory=RandomNoiseConfig)
    temporal_slice: TemporalSliceConfig = field(default_factory=TemporalSliceConfig)
    signal_slice: SignalSliceConfig = field(default_factory=SignalSliceConfig)
    sleep_gap: SleepGapConfig = field(default_factory=SleepGapConfig)
    workout_gap: WorkoutGapConfig = field(default_factory=WorkoutGapConfig)
    intensity_failure: IntensityFailureConfig = field(default_factory=IntensityFailureConfig)


@dataclass
class LSM2MethodConfig:
    """LSM2 (formerly MAE) method configuration."""

    checkpoint_path: str = ""  # Path to trained LSM2 checkpoint (.ckpt)
    device: str = "cuda"  # Device for inference ("cuda", "cuda:0", "cpu")
    inference_batch_size: int = 64  # Batch size for GPU inference

    # Must match LSM2 training config (configs/lsm2/default.yaml in the private repo).
    # 1e12 = global/population normalization (default for LSM2 training)
    # 0 = instance normalization; >0 = hybrid
    normalization_prior_count: float = 1.0e12
    stats_max_samples: int | None = 10000  # Max samples for computing normalization stats
    inference_dropout_removal_ratio: float | None = None  # Override dropout removal at inference (None = use checkpoint value)


@dataclass
class PyPOTSMethodConfig:
    """PyPOTS-specific method configuration."""

    model_path: str = ""  # Path to saved PyPOTS model directory
    model_name: str = "brits"  # PyPOTS model class name (brits, saits)
    device: str = "cuda"  # Device for inference ("cuda", "cuda:0", "cpu")
    inference_batch_size: int = 64  # Batch size for inference
    normalization_stats_path: str | None = None  # Path to normalization_stats.json from H5 export
    # Architecture params needed to construct model before loading weights
    n_steps: int = 1440
    n_features: int = 19
    rnn_hidden_size: int = 128
    n_layers: int = 2  # TimesNet, FEDformer
    top_k: int = 5  # TimesNet
    d_model: int = 64  # TimesNet, FEDformer
    d_ffn: int = 64  # TimesNet, FEDformer
    n_kernels: int = 6  # TimesNet
    n_heads: int = 4  # FEDformer
    moving_avg_window_size: int = 25  # FEDformer
    dropout: float = 0.1  # TimesNet, FEDformer
    apply_nonstationary_norm: bool = False  # TimesNet
    version: str = "Fourier"  # FEDformer
    modes: int = 32  # FEDformer
    mode_select: str = "random"  # FEDformer
    trmf_lags: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])  # TRMF
    trmf_K: int = 10  # TRMF rank
    trmf_lambda_f: float = 0.1  # TRMF factor regularization
    trmf_lambda_x: float = 0.1  # TRMF coefficient regularization
    trmf_lambda_w: float = 0.1  # TRMF lag weight regularization
    trmf_alpha: float = 0.01  # TRMF temporal update rate
    trmf_eta: float = 1.0  # TRMF temporal regularization strength
    trmf_max_iter: int = 1000  # TRMF max EM iterations


@dataclass
class MethodConfig:
    """Imputation method configuration.

    Future methods can add their own fields here (e.g., encoder_checkpoint for neural methods).
    """

    type: Literal[
        "mean",
        "mode",
        "temporal_mean",
        "temporal_mode",
        "linear",
        "locf",
        "lsm2",
        "lsm2_weekly_sparse",
        "pypots",
        "brits",
        "timesnet",
        "dlinear",
        "fedformer",
        "personalized_mean",
        "personalized_mode",
        "personalized_temporal_mean",
    ] = "mean"
    decimal_precision: int = 1  # Rounding precision for mode computation
    release_dir: str | None = None  # Manifest-bundled release dir (paper checkpoints)
    device: str = "cuda"  # Inference device for neural imputers
    inference_batch_size: int = 64  # Inference batch size for neural imputers
    lsm2: LSM2MethodConfig = field(default_factory=LSM2MethodConfig)  # LSM2-specific config
    pypots: PyPOTSMethodConfig = field(default_factory=PyPOTSMethodConfig)  # PyPOTS-specific config


@dataclass
class OutputConfig:
    """Output configuration for results."""

    results_dir: str = "results/imputation_eval"
    experiment_name: str | None = None
    experiment_name_prefix: str | None = None
    save_config: bool = True


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    compute_metrics: bool = True  # If False, only save pairs (no MetricAccumulator)
    save_pairs: bool = True  # Save raw (gt, pred) pairs to Parquet


@dataclass
class VisualizationConfig:
    """Visualization configuration for imputation plots."""

    enabled: bool = False  # Whether to generate visualization plots
    plots_per_scenario: int = 5  # Number of random samples per masking scenario
    channels: list[int] | None = (
        None  # Channels to plot (None = default continuous 0-6 + sleep 7-8)
    )
    figsize_per_channel: tuple[float, float] = (15.0, 2.5)  # Figure size per channel subplot
    seed: int | None = None  # Seed for sample selection (None = use config.seed)
    dpi: int = 150  # DPI for saved figures
    format: str = "png"  # Output format: png, pdf, svg
    split: str = "test"  # Which split to visualize: "val" or "test"


@dataclass
class SensitivityConfig:
    """Sensitivity analysis configuration for demographic subgroup evaluation."""

    enabled: bool = False
    age_bins: list[int] = field(default_factory=lambda: [18, 30, 40, 50, 60])
    # Derived bin labels: "18-29", "30-39", "40-49", "50-59", "60+"


@dataclass
class BootstrapConfig:
    """Participant-level cluster bootstrap for imputation metric CIs.

    When ``enabled``, the runner forces pair-saving on and computes percentile
    confidence intervals + standard errors by resampling users (clusters), not
    rows. See ``src/imputation_evaluation/evaluation/bootstrap.py``.
    """

    enabled: bool = False
    n_boot: int = 1000
    ci_level: float = 0.95
    seed: int = 42
    include_auc: bool = True
    # Where to write structured bootstrap_metrics.json (None = next to results.json).
    output_path: str | None = None


@dataclass
class WandbConfig:
    """Weights & Biases logging configuration."""

    enabled: bool = False
    project: str = "mhc-imputation-eval"
    entity: str | None = "MHC_Dataset"
    tags: list[str] = field(default_factory=list)
    log_plots: bool = False  # Upload visualization plots as wandb images


@dataclass
class ImputationEvalConfig:
    """Root configuration for imputation evaluation."""

    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    method: MethodConfig = field(default_factory=MethodConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    sensitivity: SensitivityConfig = field(default_factory=SensitivityConfig)
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
