"""Configuration dataclasses for downstream evaluation.

Provides a structured configuration schema that mirrors the Lightning CLI
pattern but is designed for sklearn-based evaluation.

Two config hierarchies:
  1. DownstreamEvalConfig — single-task eval (used by DownstreamEvaluator, ResultsWriter, tests)
  2. EvalConfig — unified batch eval pipeline (scripts/downstream_eval/run_downstream_eval.py)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

# Fixed reference timestamp for static enrollment-time label lookups.
# The Labels API returns the nearest-in-time match; this anchors all lookups
# consistently to the enrollment era.
LABEL_REFERENCE_DATE = "2020-06-01"

# ---------------------------------------------------------------------------
# Time Window — parameterized week selection for temporal experiments
# ---------------------------------------------------------------------------


@dataclass
class TimeWindow:
    """Parameterized week selection relative to a reference date (label date).

    Two fields cover all windowing cases:
    - max_weeks_before: max weeks before reference (None=unlimited, 0=exclude)
    - max_weeks_after: max weeks after reference (None=unlimited, 0=exclude)

    Examples:
        TimeWindow.full()              → keep all weeks (ignore label date)
        TimeWindow.before_label()      → all weeks on or before label date
        TimeWindow.before_label_n(52)  → all weeks before label + up to 52w after
        TimeWindow.before_n(10)        → last 10 weeks before label date
        TimeWindow.around_n(10)        → ±10 weeks around label date
    """

    name: str
    max_weeks_before: int | None = None  # None = unlimited
    max_weeks_after: int | None = None  # None = unlimited

    # -- factory classmethods for common presets --

    @classmethod
    def full(cls) -> TimeWindow:
        """All weeks, ignoring label date entirely."""
        return cls(name="full", max_weeks_before=None, max_weeks_after=None)

    @classmethod
    def before_label(cls) -> TimeWindow:
        """All weeks on or before the label date (current 'clipped')."""
        return cls(name="before_label", max_weeks_before=None, max_weeks_after=0)

    @classmethod
    def before_label_n(cls, n: int) -> TimeWindow:
        """All weeks before the label date, plus up to *n* weeks after.

        Useful for including a bounded post-label buffer while keeping the
        full history.  For example, ``before_label_52a`` keeps all data
        before the label date and up to 52 weeks (1 year) after.
        """
        return cls(name=f"before_label_{n}a", max_weeks_before=None, max_weeks_after=n)

    @classmethod
    def before_n(cls, n: int) -> TimeWindow:
        """Last *n* weeks before (and including) the label date."""
        return cls(name=f"before_{n}w", max_weeks_before=n, max_weeks_after=0)

    @classmethod
    def after_n(cls, n: int) -> TimeWindow:
        """First *n* weeks after (and including) the label date."""
        return cls(name=f"after_{n}w", max_weeks_before=0, max_weeks_after=n)

    @classmethod
    def around_n(cls, n: int) -> TimeWindow:
        """±*n* weeks around the label date."""
        return cls(name=f"around_{n}w", max_weeks_before=n, max_weeks_after=n)

    @classmethod
    def around(cls, before: int, after: int) -> TimeWindow:
        """Asymmetric window: *before* weeks before, *after* weeks after."""
        return cls(
            name=f"around_{before}b_{after}a",
            max_weeks_before=before,
            max_weeks_after=after,
        )

    @property
    def is_full(self) -> bool:
        """True if this window ignores the label date (keeps all weeks)."""
        return self.max_weeks_before is None and self.max_weeks_after is None

    @property
    def needs_clip_dates(self) -> bool:
        """True if this window requires clip_dates to filter weeks."""
        return not self.is_full


def parse_time_windows(spec: str) -> list[TimeWindow]:
    """Parse a comma-separated time window specification string.

    Supported formats:
        "full"              → TimeWindow.full()
        "before_label"      → TimeWindow.before_label()
        "before_label_52a"  → TimeWindow.before_label_n(52)  (unlimited before + 52w after)
        "before_10w"        → TimeWindow.before_n(10)
        "after_5w"          → TimeWindow.after_n(5)
        "around_10w"        → TimeWindow.around_n(10)
        "around_10b_5a"     → TimeWindow.around(before=10, after=5)

    Args:
        spec: Comma-separated string, e.g. "full,before_label,before_10w"

    Returns:
        List of TimeWindow objects.

    Raises:
        ValueError: If a token cannot be parsed.
    """
    windows = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue

        if token == "full":
            windows.append(TimeWindow.full())
        elif token == "before_label":
            windows.append(TimeWindow.before_label())
        elif m := re.fullmatch(r"before_label_(\d+)a", token):
            windows.append(TimeWindow.before_label_n(int(m.group(1))))
        elif m := re.fullmatch(r"before_(\d+)w", token):
            windows.append(TimeWindow.before_n(int(m.group(1))))
        elif m := re.fullmatch(r"after_(\d+)w", token):
            windows.append(TimeWindow.after_n(int(m.group(1))))
        elif m := re.fullmatch(r"around_(\d+)w", token):
            windows.append(TimeWindow.around_n(int(m.group(1))))
        elif m := re.fullmatch(r"around_(\d+)b_(\d+)a", token):
            windows.append(TimeWindow.around(int(m.group(1)), int(m.group(2))))
        else:
            raise ValueError(
                f"Cannot parse time window token: {token!r}. "
                "Expected: full, before_label, before_label_Na, before_Nw, "
                "after_Nw, around_Nw, or around_Nb_Na"
            )
    return windows


@dataclass
class DataConfig:
    """Data source and splitting configuration.

    segment_type controls the temporal granularity of input segments:
      - "weekly": 168-hour (7-day) segments from weekly_hf (default, current behavior)
      - "daily": 24-hour segments from daily_hourly_hf (hourly-resolution, filtered,
        ~5 GB).  Each row is one user-day with shape (19, 24) channels-first.

    When segment_type="daily", daily_hourly_hf_dir is used as the data source
    and daily_labels_lookup_path provides the index-aligned labels.

    On-the-fly weekly construction (alternative to weekly_hf_dir):
      Set daily_hourly_hf_dir + window_index_path to build weekly tensors at
      runtime from the compact daily_hourly_hf dataset. This avoids storing a
      large weekly_hf dataset and enables flexible windowing (stride, window_size).

    label_validity_criterion controls which data-quality filter to apply at
    runtime when deciding whether a user has sufficient wearable data for a
    given label.  Two named criteria are supported (see
    ``scripts/labels/build_label_validity.py`` for details):

      - "c1" (default): At least 1 filtered day in the label's time window.
            Maximises the eligible participant pool.
      - "c2": At least one contiguous 7-day subwindow with >=5 filtered days.
            Stricter quality filter; results in ~55 % fewer eligible samples
            but modestly higher downstream performance.
      - null / "none": No validity filtering — all user-label pairs are kept.

    The criterion is resolved to a JSON file at runtime and applied as a
    column-level mask on the labels-lookup parquet (invalid labels are set
    to the missing sentinel -1 / -1.0).  C1 reads the canonical
    ``label_validity.json`` (DVC-tracked, sharable-user filtered upstream);
    C2 still reads the local ``label_validity_c2.json`` built by
    ``scripts/labels/build_label_validity.py --criterion weekly_5of7``.
    """

    segment_type: Literal["weekly", "daily"] = "weekly"
    weekly_hf_dir: str = "data/processed/weekly_hf"
    daily_hf_dir: str = "data/processed/daily_hf"
    # daily_hourly_hf: filtered hourly-resolution daily dataset (19, 24) per sample
    daily_hourly_hf_dir: str = "data/processed/daily_hourly_hf"
    # On-the-fly weekly construction (alternative to pre-built weekly_hf)
    window_index_path: str | None = None
    window_size: int = 7
    weekly_labels_lookup_path: str = "data/processed/weekly_labels_lookup.parquet"
    daily_labels_lookup_path: str = "data/processed/daily_labels_lookup.parquet"
    task_name: str = "Diabetes"
    split_file: str = "data/splits/sharable_users_seed42_2026.json"
    split_seed: int = 42
    num_workers: int = 4
    clip_dates_path: str | None = None  # Path to clip_dates.json for temporal clipping
    min_valid_days_per_week: int = 5  # Filter weeks with < N valid days (0 = no filter)
    # Label validity criterion — see class docstring for details.
    label_validity_criterion: str | None = "c1"
    # Segment-level prediction for longitudinal labels (Watch_*, happiness).
    # When True, each segment's features are paired with that same segment's
    # label (same-segment recovery) — one prediction per (user, ISO week).
    longitudinal: bool = False
    # Subtract per-user mean features and mean label before training.
    # Recommended for longitudinal evaluation to isolate within-person
    # variation from between-person trait.
    person_mean_center: bool = False
    # Directory containing label_validity.json (C1, DVC-tracked) and
    # label_validity_c2.json (C2, locally built).
    label_validity_dir: str = "data/labels"

    @property
    def use_indexed(self) -> bool:
        """Whether to use the indexed (daily_hourly_hf + window index) path.

        Only activates for weekly segments — daily segments load daily_hourly_hf
        directly and ignore the window index.
        """
        return (
            self.segment_type == "weekly"
            and self.daily_hourly_hf_dir is not None
            and self.window_index_path is not None
        )

    @property
    def hf_dir(self) -> str:
        """Return the HF dataset directory for the configured segment type."""
        if self.segment_type == "daily":
            return self.daily_hourly_hf_dir
        return self.weekly_hf_dir

    @property
    def label_validity_path(self) -> str | None:
        """Resolve the label-validity JSON path from the criterion name.

        Returns None when no validity filtering is requested.  C1 maps to
        the canonical (sharable-filtered, DVC-tracked) ``label_validity.json``;
        any other criterion keeps the suffixed ``label_validity_<crit>.json``
        convention so the C2 ablation path still works.
        """
        crit = self.label_validity_criterion
        if crit is None or crit.lower() == "none":
            return None
        if crit.lower() == "c1":
            return f"{self.label_validity_dir}/label_validity.json"
        return f"{self.label_validity_dir}/label_validity_{crit}.json"

    @property
    def active_labels_lookup_path(self) -> str:
        """Return the labels lookup path for the configured segment type."""
        if self.segment_type == "daily":
            return self.daily_labels_lookup_path
        return self.weekly_labels_lookup_path


@dataclass
class BaselineFeatureConfig:
    """Configuration for baseline (mean/std) feature extraction."""

    use_full_features: bool = False  # True = 42 columns, False = 19 sensor channels only


@dataclass
class EncoderFeatureConfig:
    """Configuration for encoder-based feature extraction."""

    source: Literal["checkpoint", "precomputed"] = "checkpoint"
    checkpoint_path: str | None = None
    precomputed_dir: str | None = None
    embedding_type: Literal["h", "z"] = "h"
    # Encoder architecture (must match checkpoint)
    encoder_type: Literal["transformer", "mamba2", "jets"] = "transformer"
    in_dim: int = 42
    embed_dim: int = 256
    hidden_dim: int = 256
    num_layers: int = 4
    n_heads: int = 8  # Only used for transformer
    proj_dim: int = 128
    dropout: float = 0.05  # Only used for mamba2
    # JETS-specific
    encoder_ffn_mult: int = 4  # Only used for jets
    patch_size: int = 4  # Only used for jets
    batch_size: int = 64
    # Pre-computed normalization stats (JSON with "means" and "stds" arrays).
    # When set, these stats are used instead of computing on-the-fly from train split.
    # This ensures eval uses the same normalization as pretraining.
    normalization_stats_path: str | None = None


@dataclass
class FEXGBoostFeatureConfig:
    """Configuration for FE-XGBoost hand-crafted feature extraction.

    Features are pre-computed by run_fe_xgboost_prepare.py and stored as
    3 Parquet files in features_dir (timeseries, curve_analysis, day_dynamics).
    """

    features_dir: str = "data/features/fe_xgboost"


@dataclass
class GRUDConfig:
    """GRU-D supervised sequence model configuration (PyPOTS).

    GRU-D uses temporal decay on hidden states and input features to handle
    missingness-aware classification directly on raw time series.

    When multi_task=True, trains a single GRU-D encoder with per-task
    classification heads on all classification tasks simultaneously.
    """

    rnn_hidden_size: int = 64
    batch_size: int = 32
    epochs: int = 50
    patience: int = 10
    lr: float = 1e-3
    device: str = "auto"  # "auto" = GPU if available, else CPU
    max_weeks_per_user: int | None = None  # Cap weeks per user for training fairness
    multi_task: bool = False  # Train one model for all tasks (shared encoder, per-task heads)


@dataclass
class BRITSConfig:
    """BRITS supervised sequence model configuration (PyPOTS).

    BRITS uses bidirectional RNNs with consistency constraints between forward
    and backward imputations, jointly optimizing imputation + classification.

    When multi_task=True, trains a single BRITS encoder with per-task
    classification heads on all classification tasks simultaneously.
    """

    rnn_hidden_size: int = 64
    batch_size: int = 32
    epochs: int = 50
    patience: int = 10
    lr: float = 1e-3
    device: str = "auto"  # "auto" = GPU if available, else CPU
    max_weeks_per_user: int | None = None  # Cap weeks per user for training fairness
    multi_task: bool = False  # Train one model for all tasks (shared encoder, per-task heads)


@dataclass
class JETSEncoderConfig:
    """Configuration for JETS triplet encoder feature extraction.

    JETS operates on irregular observation triplets (dt, value, metric_id)
    rather than dense weekly tensors. Features are extracted at the user level
    (not week level) by encoding all observation chunks per user and mean-pooling.

    Requires:
    - A pretrained JETS checkpoint
    - The jets_observations.parquet file
    """

    checkpoint_path: str | None = None  # Path to JETS Lightning checkpoint
    observations_path: str = "data/processed/jets_observations.parquet"
    d_model: int = 256
    num_metrics: int = 19
    encoder_depth: int = 8
    encoder_ffn_mult: int = 4
    num_patches: int = 64
    max_obs_per_chunk: int = 5000
    min_obs_per_user: int = 300
    outlier_zscore: float = 8.0
    batch_size: int = 64
    device: str = "auto"


@dataclass
class MAEEncoderConfig:
    """Configuration for MAE daily encoder feature extraction.

    MAE features are dense per-token encoder representations from the LSM2 MAE
    model, extracted per day from daily_hf.  Two source modes:

      - "precomputed": Load pre-pooled (384-dim per day) embeddings from HDF5
        files produced by scripts/pool_mae_features.py.
      - "checkpoint": Extract on-the-fly from an MAE checkpoint (requires GPU).

    Unlike the SSL encoder (which uses weekly segments), MAE features are
    daily-granularity, enabling time-window filtering at the day level before
    user-level mean pooling.
    """

    source: Literal["precomputed", "checkpoint"] = "precomputed"
    pooled_embeddings_dir: str = "data/processed/mae_pooled_embeddings"
    checkpoint_path: str | None = None
    daily_hf_dir: str = "data/processed/daily_hf"
    normalization_stats_path: str = "data/processed/normalization_stats.json"
    embed_dim: int = 384
    patch_size: int = 10
    batch_size: int = 64
    device: str = "auto"
    min_wear_fraction: float = 0.5


@dataclass
class MultiRocketConfig:
    """MultiRocketMultivariate feature extraction configuration (sktime).

    MultiRocketMultivariate applies random convolutional kernels to both raw
    and first-order differenced multivariate time series, using 4 pooling
    operators (PPV, MPV, MIPV, LSPV) per kernel. Output dimensionality is
    2 * n_features_per_kernel * num_kernels (default: 2 * 4 * 6250 = 50,000).

    Missing data handling: z-score normalization using train-split channel
    means/stds (computed from observed values only), then zero-fill missing
    positions. This is equivalent to imputing with the training mean before
    z-scoring — the same approach used by the SSL encoder.

    Reference:
        Tan et al., "MultiRocket: Multiple pooling operators and transformations
        for fast and effective time series classification", 2022.
    """

    num_kernels: int = 6_250  # Must be a multiple of 84; rounded down if not
    max_dilations_per_kernel: int = 32
    n_features_per_kernel: int = 4  # PPV + MPV + MIPV + LSPV
    normalise: bool = False
    n_jobs: int = 1
    random_state: int | None = None
    transform_chunk_size: int = 0  # 0 = transform all at once; >0 = chunk size for incremental aggregation


@dataclass
class VLMConfig:
    """Configuration for VLM zero-shot evaluation (e.g., Gemini).

    Renders weekly sensor tensors as images and queries a VLM for per-week
    predictions, then aggregates per user. Test-split only, no training.

    Per-week predictions are cached as Parquet under ``cache_dir`` keyed by
    ``(task, model_id, ds_idx)`` so swapping the validity criterion, time
    window, or aggregation rule never re-bills the API.
    """

    model_id: str = "gemini-3.1-pro-preview"
    cache_dir: str = "results/vlm_cache"
    max_output_tokens: int = 8192


@dataclass
class HybridConfig:
    """Configuration for hybrid SSL + fallback evaluation.

    The hybrid pathway trains two independent classifiers:
      1. Primary (SSL encoder, weekly segments) — covers users with ≥1 weekly tensor
      2. Fallback — covers ALL users with daily data

    At test time, the SSL prediction is used when available; otherwise the
    fallback prediction is used. Metrics are computed over the full eligible
    test set (SSL + fallback users combined).

    Supported fallback methods:
      - "fe_xgboost": Pre-computed user-level features + XGBoost classifiers
      - "stat_full": Daily feature store (456-dim) + XGBoost classifiers
      - "stat_simple": Daily feature store (38-dim) + linear probes
    """

    fallback_method: Literal["fe_xgboost", "stat_full", "stat_simple"] = "fe_xgboost"
    # Path to pre-built fallback feature store (.npz). Required for stat_full/stat_simple.
    fallback_store_path: str | None = None


@dataclass
class FeatureConfig:
    """Feature extraction configuration.

    Feature type routing:
      - "statistical"   → BaselineFeatureExtractor (stat_simple or stat_full)
      - "ssl_encoder"   → EncoderFeatureExtractor (SSL embeddings)
      - "multirocket"   → MultiRocketFeatureExtractor (convolutional kernel features)
      - "jets_encoder"  → JETSTripletExtractor (JETS triplet observation embeddings)
      - "mae_encoder"   → MAEDailyExtractor (MAE daily encoder embeddings)
      - "gru_d"         → Supervised GRU-D (PyPOTS) — trains per task, no feature store
      - "brits"         → Supervised BRITS (PyPOTS) — trains per task, no feature store
      - "fe_xgboost"    → FEXGBoostExtractor (pre-computed hand-crafted features)
      - "hybrid"        → Hybrid SSL encoder + stat_full fallback for full-cohort eval
      - "fe_handcrafted_weekly" → load-only: per-(user, week) handcrafted feature store
                           pre-built by scripts/build_weekly_handcrafted_store.py;
                           feeds the feature-store pathway with --store.load_path
      - "vlm"           → VLMZeroShotEvaluator (zero-shot VLM, e.g. Gemini); no fitting,
                          test split only, predictions aggregated per user
    """

    type: Literal[
        "statistical", "ssl_encoder", "multirocket", "jets_encoder",
        "mae_encoder", "gru_d", "brits", "fe_xgboost", "hybrid",
        "fe_handcrafted_weekly", "vlm",
    ] = "statistical"
    statistical: BaselineFeatureConfig = field(default_factory=BaselineFeatureConfig)
    ssl_encoder: EncoderFeatureConfig = field(default_factory=EncoderFeatureConfig)
    multirocket: MultiRocketConfig = field(default_factory=MultiRocketConfig)
    jets_encoder: JETSEncoderConfig = field(default_factory=JETSEncoderConfig)
    mae_encoder: MAEEncoderConfig = field(default_factory=MAEEncoderConfig)
    gru_d: GRUDConfig = field(default_factory=GRUDConfig)
    brits: BRITSConfig = field(default_factory=BRITSConfig)
    fe_xgboost: FEXGBoostFeatureConfig = field(default_factory=FEXGBoostFeatureConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)


@dataclass
class LogRegConfig:
    """LogisticRegression hyperparameters."""

    max_iter: int = 4000
    class_weight: str | None = "balanced"
    C: float = 1.0
    solver: str = "liblinear"  # More robust for small datasets than lbfgs
    n_jobs: int = 1  # liblinear doesn't support n_jobs=-1


@dataclass
class OrderedLogitConfig:
    """Ordered Logit (statsmodels) hyperparameters."""

    distr: str = "logit"  # 'logit' or 'probit'
    method: str = "bfgs"  # 'bfgs', 'lbfgs', 'powell'
    max_iter: int = 500  # Maximum number of iterations
    alpha: float = 0.2  # Regularization parameter


@dataclass
class SVMConfig:
    """SVM hyperparameters."""

    kernel: str = "rbf"
    C: float = 1.0
    class_weight: str | None = "balanced"
    probability: bool = True


@dataclass
class RandomForestConfig:
    """RandomForest hyperparameters."""

    n_estimators: int = 100
    max_depth: int | None = None
    class_weight: str | None = "balanced"
    n_jobs: int = -1


@dataclass
class LinearRegressionConfig:
    """Linear Regression hyperparameters."""

    fit_intercept: bool = True
    copy_X: bool = True
    n_jobs: int | None = None
    positive: bool = False


@dataclass
class ElasticNetConfig:
    """ElasticNetCV (Linear Regression with L1/L2 regularization, alpha chosen by CV)."""

    l1_ratio: float = 0.5  # Mixing parameter (0=L2, 1=L1)
    n_alphas: int = 100  # Number of alphas along the regularization path
    cv: int = 5  # Number of cross-validation folds
    max_iter: int = 1000


@dataclass
class SVRConfig:
    """Support Vector Regression hyperparameters."""

    kernel: str = "rbf"
    C: float = 1.0  # Regularization parameter
    epsilon: float = 0.1  # Epsilon-tube


@dataclass
class RandomForestRegressorConfig:
    """Random Forest Regressor hyperparameters."""

    n_estimators: int = 100
    max_depth: int | None = None
    n_jobs: int = -1


@dataclass
class XGBClassifierConfig:
    """XGBoost Classifier hyperparameters."""

    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    min_child_weight: int = 1
    gamma: float = 0.0  # Minimum loss reduction for split
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0  # L1 regularization
    reg_lambda: float = 1.0  # L2 regularization
    scale_pos_weight: float | None = None  # None = auto-compute from training labels for binary tasks
    n_jobs: int = -1
    eval_metric: str = "logloss"


@dataclass
class XGBRegressorConfig:
    """XGBoost Regressor hyperparameters."""

    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0  # L1 regularization
    reg_lambda: float = 1.0  # L2 regularization
    n_jobs: int = -1
    objective: str = "reg:squarederror"


@dataclass
class XGBOrdinalConfig:
    """XGBoost Ordinal hyperparameters."""

    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0  # L1 regularization
    reg_lambda: float = 1.0  # L2 regularization
    n_jobs: int = -1
    use_label_encoder: bool = False
    objective: str = "binary:logistic"


@dataclass
class ClassifierConfig:
    """Classifier selection and hyperparameters."""

    type: Literal[
        "logistic_regression",
        "svm",
        "random_forest_classifier",
        "elastic_net",
        "xgboost_classifier",
        "linear_regression",
        "svr",
        "random_forest_regressor",
        "xgboost_regressor",
        "logreg_ordinal",
        "xgboost_ordinal",
    ] = "logistic_regression"
    use_scaler: bool = True
    scaler_type: Literal["robust", "standard"] = "robust"  # "robust" = z-score + clip ±10σ (prevents overflow), "standard" = z-score only
    pca_n_components: int | None = None  # PCA dim reduction before classifier (None = disabled)
    pca_whiten: bool = False  # If True, whiten PCA output (unit variance per component)
    use_l2_norm: bool = False  # L2-normalize each sample before classifier
    logistic_regression: LogRegConfig = field(default_factory=LogRegConfig)
    svm: SVMConfig = field(default_factory=SVMConfig)
    random_forest_classifier: RandomForestConfig = field(default_factory=RandomForestConfig)
    linear_regression: LinearRegressionConfig = field(default_factory=LinearRegressionConfig)
    elastic_net: ElasticNetConfig = field(default_factory=ElasticNetConfig)
    svr: SVRConfig = field(default_factory=SVRConfig)
    random_forest_regressor: RandomForestRegressorConfig = field(
        default_factory=RandomForestRegressorConfig
    )
    xgboost_classifier: XGBClassifierConfig = field(default_factory=XGBClassifierConfig)
    xgboost_regressor: XGBRegressorConfig = field(default_factory=XGBRegressorConfig)
    xgboost_ordinal: XGBOrdinalConfig = field(default_factory=XGBOrdinalConfig)


@dataclass
class AggregationConfig:
    """Aggregation configuration for segment-to-user pooling.

    Segments are either weekly (168h) or daily (24h), controlled by
    DataConfig.segment_type.

    Supported methods:
        - "mean": Simple mean over segment embeddings.
        - "cov_weighted_mean": Coverage-weighted mean where each segment is
          weighted by its coverage fraction (n_valid_hours / hours_per_segment).
          Segments with more observed data contribute proportionally more.
    """

    level: Literal["user", "week"] = "user"
    method: Literal["mean", "cov_weighted_mean"] = "mean"


@dataclass
class OutputConfig:
    """Output configuration for results."""

    results_dir: str = "results/downstream_eval"
    experiment_name: str | None = None
    save_predictions: bool = True
    save_config: bool = True


@dataclass
class DownstreamEvalConfig:
    """Root configuration for single-task downstream evaluation.

    Used by DownstreamEvaluator, ResultsWriter, and the forecasting evaluation
    pipeline. For batch multi-task evaluation, see EvalConfig below.
    """

    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


# ---------------------------------------------------------------------------
# Unified Eval Pipeline — EvalConfig (run_downstream_eval.py)
# ---------------------------------------------------------------------------

# Default classifier mapping: task_type → list of classifier types to evaluate.
# Can be overridden in YAML via experiment.classifiers_by_task_type.
DEFAULT_CLASSIFIERS_BY_TASK_TYPE: dict[str, list[str]] = {
    "binary": ["logistic_regression", "xgboost_classifier"],
    "multiclass": ["logistic_regression"],
    "ordinal": ["logreg_ordinal"],
    "regression": ["linear_regression"],
}


@dataclass
class StoreConfig:
    """Feature store persistence configuration."""

    save_path: str | None = None  # Save store after extraction (e.g. results/stores/baseline.npz)
    load_path: str | None = None  # Load pre-built store (skip extraction)


@dataclass
class EvalExperimentConfig:
    """Experiment-level settings for the unified eval pipeline."""

    tasks: list[str] | None = None  # None = all tasks from labels API
    time_windows: str = "full,before_label"  # Comma-separated TimeWindow specs
    classifiers_by_task_type: dict[str, list[str]] = field(
        default_factory=lambda: dict(DEFAULT_CLASSIFIERS_BY_TASK_TYPE)
    )
    resume: bool = False  # Skip already-completed (task, features, classifier, condition) rows

    # Demographic covariate augmentation.  When non-empty, the listed label
    # columns (from the labels-lookup parquet) are appended to the feature
    # vector for every task *except* when the task itself is one of the
    # listed covariates.  Example: ["age", "BiologicalSex", "BMI_values"]
    demographic_covariates: list[str] | None = None


@dataclass
class EvalOutputConfig:
    """Output configuration for unified eval pipeline."""

    results_dir: str = "results/eval"
    csv_name: str | None = None  # Auto-named by feature type if None
    save_predictions: bool = False
    baseline_csv: str | None = None  # Path to baseline results CSV for skill score


@dataclass
class EvalConfig:
    r"""Root configuration for the unified eval pipeline (run_downstream_eval.py).

    Composes all sub-configs needed for the evaluation workflow:
      - data: paths to HF dataset, labels, splits, clip dates
      - features: feature extraction / supervised model settings
      - classifier: hyperparameters for all sklearn classifier types
      - store: feature store save/load paths (stat, ssl, multirocket only)
      - experiment: tasks, time windows, classifier mapping, resume
      - output: results directory and CSV naming

    Two evaluation pathways:
      1. Feature store (stat, ssl_encoder, multirocket):
         Extract features once → store → aggregate → sklearn classifier
      2. Supervised sequence (gru_d, brits):
         Train model per (task, time_window) → predict weeks → aggregate probs

    Usage with YAML configs::

        python scripts/downstream_eval/run_downstream_eval.py \\
            --config configs/downstream_eval/base.yaml \\
            --config configs/downstream_eval/stat_simple.yaml

        python scripts/downstream_eval/run_downstream_eval.py \\
            --config configs/downstream_eval/base.yaml \\
            --config configs/downstream_eval/gru_d.yaml
    """

    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    store: StoreConfig = field(default_factory=StoreConfig)
    experiment: EvalExperimentConfig = field(default_factory=EvalExperimentConfig)
    output: EvalOutputConfig = field(default_factory=EvalOutputConfig)
