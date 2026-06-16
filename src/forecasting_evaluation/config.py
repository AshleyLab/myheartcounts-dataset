"""Configuration dataclasses for downstream forecasting evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Literal


@dataclass
class DataConfig:
    """Data source and splitting configuration."""
    trajectory_hf_dir: str = "data/hourly_trajectory"
    task_name: str = "hourly_trajectory_forecasting"
    split_file: str | None = "data/splits/sharable_users_seed42_2026.json"
    day_remain_mask: str | None = "data/forecasting_sample_index/day_remain_mask.json"
    # Required for forecasting evaluation; ships in the openmhc.download_dataset()
    # bundle (see docs/manual-dataset-setup.md to assemble a root by hand).
    sample_index_file: str = "data/forecasting_sample_index/sample_index_P_24_M_H_7_3_S_100.json"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    split_seed: int = 42
    num_workers: int = 4
    max_samples: int | None = None

ModelType = Literal[
    "seasonal_naive",
    "autoARIMA",
    "autoETS",
    "chronos2",
    "toto",
    "mixlinear",
    "dlinear",
    "segrnn",
]

@dataclass
class SeasonalNaiveModelConfig:
    """seasonal_naive model hyperparameters."""

    season_length: int = 24
    quantile_levels: list[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )


@dataclass
class AutoARIMAModelConfig:
    """autoARIMA model hyperparameters."""

    start_p: int = 2
    start_q: int = 2
    max_p: int = 5
    max_q: int = 5
    seasonal: bool = True
    start_P: int = 1
    start_Q: int = 1
    max_P: int = 2
    max_Q: int = 2
    max_d: int = 2
    max_D: int = 1
    information_criterion: str = "aic"
    suppress_warnings: bool = True
    trace: bool = False
    error_action: str = "ignore"
    stepwise: bool = False
    n_jobs: int = -1
    max_history_length: int | None = 24 * 14  # Limit to recent 336 hours (14 days)


@dataclass
class AutoETSModelConfig:
    """autoETS model hyperparameters."""

    auto: bool = True
    sp: int = 24
    information_criterion: str = "aic"
    n_jobs: int = -1
    max_history_length: int | None = 24 * 14  # Limit to recent 336 hours (14 days)


@dataclass
class Chronos2ModelConfig:
    """chronos2 model hyperparameters."""

    temp: int = 1
    pretrained_model_name_or_path: str = "amazon/chronos-2"
    checkpoint_path: str | None = None
    training_output_dir: str | None = None
    finetuned_ckpt_name: str | None = None
    device: str = "auto"
    torch_dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"


@dataclass
class TotoModelConfig:
    """Toto model hyperparameters."""

    pretrained_model_name_or_path: str = "Datadog/Toto-Open-Base-1.0"
    checkpoint_path: str | None = None
    lora_alpha: float | None = None
    device: str = "auto"
    context_length: int = 2048
    num_samples: int = 256
    samples_per_batch: int = 256
    use_kv_cache: bool = True
    time_interval_seconds: int = 3600  # 1 hour for hourly wearable data


@dataclass
class MixLinearModelConfig:
    """mixlinear model hyperparameters."""

    checkpoint_path: str | None = None
    device: str = "auto"
    batch_size: int = 64
    n_steps: int | None = None
    n_pred_steps: int | None = None
    n_features: int | None = None
    period_len: int = 24
    lpf: int = 2
    alpha: float = 0.5
    rank: int = 2


@dataclass
class DLinearModelConfig:
    """dlinear model hyperparameters."""

    checkpoint_path: str | None = None
    device: str = "auto"
    batch_size: int = 64
    n_steps: int | None = None
    n_pred_steps: int | None = None
    n_features: int | None = None
    moving_avg_window_size: int = 25
    individual: bool = False
    d_model: int | None = None


@dataclass
class SegRNNModelConfig:
    """segrnn model hyperparameters."""

    checkpoint_path: str | None = None
    device: str = "auto"
    batch_size: int = 64
    n_steps: int | None = None
    n_pred_steps: int | None = None
    n_features: int | None = None
    seg_len: int = 24
    d_model: int = 64
    dropout: float = 0.1


@dataclass
class ForecastingModelConfig:
    """Model configuration using type + per-model nested sub-configs.

    Mirrors the imputation style: choose `type`, then tune fields under the
    corresponding nested config block.
    """

    type: ModelType = "seasonal_naive"
    name: str | None = None
    release_dir: str | None = None

    seasonal_naive: SeasonalNaiveModelConfig = field(default_factory=SeasonalNaiveModelConfig)
    autoARIMA: AutoARIMAModelConfig = field(default_factory=AutoARIMAModelConfig)
    autoETS: AutoETSModelConfig = field(default_factory=AutoETSModelConfig)
    chronos2: Chronos2ModelConfig = field(default_factory=Chronos2ModelConfig)
    toto: TotoModelConfig = field(default_factory=TotoModelConfig)
    mixlinear: MixLinearModelConfig = field(default_factory=MixLinearModelConfig)
    dlinear: DLinearModelConfig = field(default_factory=DLinearModelConfig)
    segrnn: SegRNNModelConfig = field(default_factory=SegRNNModelConfig)

@dataclass
class OutputConfig:
    """Output configuration for results."""

    results_dir: str = "results/forecasting_eval"
    save_config: bool = True
    overwrite_existing_parquet: bool = False


@dataclass
class MetricsConfig:
    """Which offline metrics to persist, and how to group channels.

    All metrics are derived cheaply from the stored predictions in one pass;
    these lists only control what gets persisted to the metrics tree. The skill
    score and ranking need ``mae`` (continuous) + ``auprc`` (binary); the full
    sets are persisted by default so the scoring method can be decided later.
    Set ``binary_metrics=[]`` to skip the binary-metric pass entirely.
    """

    point_metrics: list[str] = field(
        default_factory=lambda: ["mae", "mse", "mase", "mase_all", "ql", "sql"]
    )
    binary_metrics: list[str] = field(default_factory=lambda: ["auprc", "auroc", "f1"])
    # Merge paired phone/watch channels (e.g. step count, distance) before scoring.
    combine_channels: bool = True
    # Threshold to binarize continuous scores for F1.
    f1_threshold: float = 0.5


@dataclass
class EvaluatorConfig:
    """Evaluator execution configuration (sequential-only)."""

    mode: Literal["sequential"] = "sequential"

@dataclass
class ForecastingConfig:
    """Configuration for forecasting-specific parameters."""

    forecasting_length: int = 24  # Number of hours to forecast
    daily_start_hour_offset: int = 0

@dataclass
class FeaturesConfig:
    """Feature extraction configuration."""

    # type: Literal["covariate", "multivariate"] = "multivariate"
    covariate_types: list[Literal["hour_in_day", "day_in_week"]] | None = None
    # Forecasting evaluation currently supports the full 19-channel feature set only.
    channel: Literal["all"] = "all"


@dataclass
class ForecastingEvalConfig:
    """Root configuration for downstream evaluation."""

    seed: int = 42
    experiment_name: str | None = "Default_Test"
    debug_mode: bool = True
    time_granularity: Literal["minutely","hourly","daily"] = "hourly"
    data: DataConfig = field(default_factory=DataConfig)
    forecasting: ForecastingConfig = field(default_factory=ForecastingConfig)
    model: ForecastingModelConfig = field(default_factory=ForecastingModelConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)


def print_config(config: ForecastingEvalConfig) -> None:
    """Print configuration in a clean, automatically structured format.

    This function automatically adapts to changes in the configuration classes,
    so adding/removing fields doesn't require manual updates here.

    Args:
        config: The forecasting evaluation configuration to display.
    """

    def format_value(value, indent: int = 0) -> str:
        """Format a value for display with proper indentation."""
        indent_str = "  " * indent

        if value is None:
            return "None"
        elif isinstance(value, bool):
            return str(value)
        elif isinstance(value, int | float):
            return str(value)
        elif isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, list | tuple):
            if not value:
                return "[]"
            return f"[{', '.join(repr(v) if isinstance(v, str) else str(v) for v in value)}]"
        elif is_dataclass(value):
            # Recursively format nested dataclass
            lines = []
            for fld in fields(value):
                field_name = fld.name.replace("_", " ").title()
                field_value = getattr(value, fld.name)
                formatted = format_value(field_value, indent + 1)
                lines.append(f"{indent_str}  - {field_name}: {formatted}")
            return "\n" + "\n".join(lines)
        else:
            return str(value)

    def format_field_name(name: str) -> str:
        """Convert snake_case to Title Case."""
        return name.replace("_", " ").title()

    def emit(line: str = "") -> None:
        """Print immediately even when stdout is redirected (e.g. SLURM logs)."""
        print(line, flush=True)

    # Print header
    emit("\n" + "=" * 80)
    emit("FORECASTING EVALUATION CONFIGURATION".center(80))
    emit("=" * 80 + "\n")

    # Automatically iterate through all top-level fields
    config_fields = fields(config)
    for i, fld in enumerate(config_fields):
        field_name = format_field_name(fld.name)
        field_value = getattr(config, fld.name)

        # Determine the tree character
        if i == 0:
            tree_char = "┌─"
        elif i == len(config_fields) - 1:
            tree_char = "└─"
        else:
            tree_char = "├─"

        # For nested dataclasses, print section header
        if is_dataclass(field_value):
            emit(f"{tree_char} {field_name}")
            # Print all fields of the nested dataclass
            for nested_field in fields(field_value):
                nested_name = format_field_name(nested_field.name)
                nested_value = getattr(field_value, nested_field.name)
                formatted_value = format_value(nested_value)

                # Handle multi-line formatted values (nested dataclasses)
                if "\n" in formatted_value:
                    emit(f"│  • {nested_name}:{formatted_value}")
                else:
                    # Align values nicely
                    emit(f"│  • {nested_name:<20} {formatted_value}")
        else:
            # Simple field - print directly
            formatted_value = format_value(field_value)
            emit(f"{tree_char} {field_name:<20} {formatted_value}")

        # Add blank line between sections (except for the last one)
        if i < len(config_fields) - 1:
            emit("│")

    emit("\n" + "=" * 80 + "\n")
