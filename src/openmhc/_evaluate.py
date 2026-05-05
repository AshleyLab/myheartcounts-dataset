"""Public evaluation functions for OpenMHC.

These functions provide a simple interface to the benchmark's evaluation
pipelines. They accept duck-typed Encoder/Imputer objects and return
structured results.
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from openmhc._protocols import Encoder, Imputer

from openmhc._dataset import data_dir as _resolve_default_data_dir
from openmhc._results import DownstreamResults, ImputationResults

logger = logging.getLogger(__name__)


@dataclass
class _DatasetPaths:
    """Resolved paths into the dataset directory.

    All paths are derived from a single ``root`` so the API stays consistent
    with what :func:`openmhc.download_dataset` produces. The expected layout
    matches DATASET.md.
    """

    root: Path
    daily_hourly_hf: Path
    daily_hf: Path
    window_index: Path
    weekly_labels_lookup: Path
    splits_file: Path
    norm_stats: Path
    clip_dates: Path
    labels_dir: Path

    @classmethod
    def resolve(cls, override: str | Path | None = None) -> "_DatasetPaths":
        """Build the paths bundle from an explicit override or the default.

        Resolution order matches :func:`openmhc.data_dir`:

        1. ``override`` argument (if provided)
        2. ``MHC_DATA_DIR`` env var
        3. ``~/.cache/openmhc/data``
        """
        root = _resolve_default_data_dir(override)
        return cls(
            root=root,
            daily_hourly_hf=root / "processed" / "daily_hourly_hf",
            daily_hf=root / "processed" / "daily_hf",
            window_index=root / "processed" / "window_index_w7_s7_d5.parquet",
            weekly_labels_lookup=root / "processed" / "weekly_labels_lookup_stride7.parquet",
            splits_file=root / "splits" / "sharable_users_seed42_2026.json",
            norm_stats=root / "processed" / "normalization_stats_hourly.json",
            clip_dates=root / "labels" / "clip_dates.json",
            labels_dir=root / "labels",
        )


def _ensure_labels_env(labels_dir: Path) -> None:
    """Point the bundled `labels.api` module at the downloaded labels dir.

    `labels.api` reads `LABELS_DATA_PATH` / `CONTEXT_LABELS_PATH` env vars
    at import time. Set them to the dataset paths if the user hasn't.
    """
    if not os.getenv("LABELS_DATA_PATH"):
        os.environ["LABELS_DATA_PATH"] = str(labels_dir / "last_labels.json")
    if not os.getenv("CONTEXT_LABELS_PATH"):
        os.environ["CONTEXT_LABELS_PATH"] = str(labels_dir / "context_labels.json")


# ---------------------------------------------------------------------------
# Downstream evaluation
# ---------------------------------------------------------------------------


def evaluate_downstream(
    encoder: Encoder,
    tasks: str | list[str] = "all",
    data_dir: str | Path | None = None,
    seed: int = 42,
) -> DownstreamResults:
    """Run downstream health prediction evaluation with a custom encoder.

    Encodes weekly sensor tensors via `encoder.encode()` and evaluates the
    resulting embeddings on up to 33 health prediction tasks using linear
    probes (logistic regression for binary, ordinal logistic for ordinal,
    ridge regression for continuous).

    Args:
        encoder: Object with an `encode(weekly_tensors) -> embeddings` method.
            Input shape is (B, 168, 38), output shape is (B, D).
        tasks: "all" to run all 33 tasks, or a list of task name strings.
        data_dir: Path to the `daily_hourly_hf` dataset directory. None uses
            the default location.
        seed: Random seed for classifiers and splits.

    Returns:
        A DownstreamResults instance with per-task metrics and a global score
        (mean AUROC across binary tasks).
    """
    import pandas as pd

    paths = _DatasetPaths.resolve(data_dir)
    _ensure_labels_env(paths.labels_dir)

    from downstream_evaluation.config import ClassifierConfig, parse_time_windows
    from downstream_evaluation.data.splits import load_split_file
    from downstream_evaluation.evaluation.metrics import (
        compute_binary_metrics,
        compute_multiclass_metrics,
        compute_ordinal_metrics,
        compute_regression_metrics,
        get_task_type,
    )
    from downstream_evaluation.models.registry import create_model
    from labels.api import TARGET_NAMES

    # Resolve tasks.
    if tasks == "all":
        task_list = sorted(TARGET_NAMES)
    elif isinstance(tasks, str):
        task_list = [tasks]
    else:
        task_list = list(tasks)

    # Resolve paths.
    daily_hourly_dir = paths.daily_hourly_hf
    split_file = paths.splits_file
    window_index_path = paths.window_index
    labels_path = paths.weekly_labels_lookup
    clip_dates_path = paths.clip_dates

    # Load user splits.
    split_users = load_split_file(split_file)

    # Load labels.
    labels_df = pd.read_parquet(labels_path)

    # Apply coverage filter (min 5 valid days).
    valid_indices = None
    if "n_valid_days" in labels_df.columns:
        mask = labels_df["n_valid_days"].values >= 5
        valid_indices = np.where(mask)[0]
        labels_df = labels_df.iloc[valid_indices].reset_index(drop=True)

    # Load clip dates.
    all_clip_dates: dict = {}
    if clip_dates_path.exists():
        with open(clip_dates_path) as f:
            all_clip_dates = json.load(f)

    # Load dataset (on-the-fly weekly from daily_hourly_hf + window index).
    from data.datasets.indexed_week_dataset import load_indexed_week_dataset

    hf_dataset = load_indexed_week_dataset(
        daily_hourly_hf_dir=str(daily_hourly_dir),
        window_index_path=str(window_index_path),
        window_size=7,
    )
    logger.info("Loaded IndexedWeekDataset: %d windows", len(hf_dataset))

    if valid_indices is not None:
        hf_dataset = hf_dataset.select(valid_indices)

    # Extract features using the user's encoder.
    features, user_ids, segment_starts = _extract_encoder_features(
        encoder, hf_dataset, split_users, seed
    )

    # Build a lightweight store for aggregation.
    from downstream_evaluation.feature_store import WeekFeatureStore, _StoreMetadata

    segment_starts_ns = (
        pd.to_datetime(segment_starts.tolist()).astype("int64").values
    )

    n_valid_hours = (
        np.array(hf_dataset["n_valid_hours"], dtype=np.int32)
        if "n_valid_hours" in hf_dataset.column_names
        else None
    )

    store_meta = _StoreMetadata(
        feature_type="custom_encoder",
        feature_dim=features.shape[1],
        n_samples=features.shape[0],
        segment_type="weekly",
    )
    store = WeekFeatureStore(
        features, user_ids, segment_starts, segment_starts_ns, store_meta, n_valid_hours
    )

    del hf_dataset

    # Evaluate: iterate over tasks with "full" time window and linear probes.
    time_windows = parse_time_windows("full,before_label")

    clf_mapping = {
        "binary": ["logistic_regression"],
        "ordinal": ["ordinal_logit_at"],
        "regression": ["ridge_cv"],
        "multiclass": ["logistic_regression"],
    }

    records: list[dict] = []
    binary_aurocs: list[float] = []

    for task_name in task_list:
        try:
            task_type = get_task_type(task_name)
        except ValueError:
            logger.warning("Unknown task %s, skipping", task_name)
            continue

        classifiers = clf_mapping.get(task_type, [])
        if not classifiers:
            continue

        for tw in time_windows:
            if tw.needs_clip_dates and task_name not in all_clip_dates:
                continue

            clip = all_clip_dates.get(task_name)

            for clf_type in classifiers:
                try:
                    splits = store.aggregate_for_task(
                        labels_df=labels_df,
                        task_name=task_name,
                        clip_dates=clip,
                        time_window=tw,
                        split_users=split_users,
                        method="mean",
                    )
                except Exception as e:
                    logger.warning(
                        "Task %s/%s failed aggregation: %s", task_name, tw.name, e
                    )
                    continue

                X_train, y_train, _ = splits["train"]
                X_val, y_val, _ = splits["validation"]
                X_test, y_test, _ = splits["test"]

                if len(X_train) == 0 or len(X_test) == 0:
                    logger.warning("Task %s: empty split, skipping", task_name)
                    continue

                clf_config = ClassifierConfig(type=clf_type, use_scaler=True)
                clf = create_model(clf_config, random_state=seed, task_type=task_type)

                # NaN handling for sklearn.
                for X in [X_train, X_val, X_test]:
                    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    clf.fit(X_train, y_train)

                    row = {
                        "task": task_name,
                        "task_type": task_type,
                        "classifier": clf_type,
                        "n_train": len(X_train),
                        "n_test": len(X_test),
                    }

                    if task_type == "binary":
                        test_prob = clf.predict_proba(X_test)[:, 1]
                        m = compute_binary_metrics(y_test, test_prob)
                        row["metric"] = "auroc"
                        row["value"] = m["auroc"]
                        records.append(row)
                        binary_aurocs.append(m["auroc"])
                        records.append({
                            **row,
                            "metric": "auprc",
                            "value": m["auprc"],
                        })

                    elif task_type == "ordinal":
                        test_pred = clf.predict(X_test)
                        m = compute_ordinal_metrics(y_test, test_pred)
                        for metric_name in ("spearman_r", "qwk", "mae_ordinal"):
                            records.append(
                                {**row, "metric": metric_name, "value": m[metric_name]}
                            )

                    elif task_type == "multiclass":
                        test_pred = clf.predict(X_test)
                        m = compute_multiclass_metrics(y_test, test_pred)
                        for metric_name in ("accuracy", "f1_macro"):
                            records.append(
                                {**row, "metric": metric_name, "value": m[metric_name]}
                            )

                    elif task_type == "regression":
                        test_pred = clf.predict(X_test)
                        m = compute_regression_metrics(y_test, test_pred)
                        for metric_name in ("mse", "mae", "pearson_r", "r2"):
                            records.append(
                                {**row, "metric": metric_name, "value": m[metric_name]}
                            )

    global_score = float(np.mean(binary_aurocs)) if binary_aurocs else 0.0

    return DownstreamResults(records=records, global_score=global_score)


def _extract_encoder_features(
    encoder: Encoder,
    hf_dataset,
    split_users: dict[str, set[str]],
    seed: int,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract features from all samples using the user's encoder.

    Normalizes sensor values using training-set statistics, constructs
    (168, 38) tensors, and calls `encoder.encode()` in batches.

    Args:
        encoder: Object implementing the Encoder protocol.
        hf_dataset: HuggingFace dataset with `values`, `mask`, `user_id`,
            and timestamp columns.
        split_users: Mapping of split name to set of user ID strings. The
            "train" split is used to compute normalization statistics.
        seed: Random seed for sampling training statistics.
        batch_size: Number of samples per encoding batch.

    Returns:
        A tuple of (features, user_ids, segment_starts) where:
            - features: float32 array of shape (N, D).
            - user_ids: object array of shape (N,) with user ID strings.
            - segment_starts: object array of shape (N,) with ISO timestamp
              strings.
    """
    from tqdm import tqdm

    N = len(hf_dataset)
    user_ids = np.array(hf_dataset["user_id"], dtype=object)

    # Determine timestamp column.
    if "week_start" in hf_dataset.column_names:
        segment_starts = np.array(hf_dataset["week_start"], dtype=object)
    elif "date" in hf_dataset.column_names:
        segment_starts = np.array(hf_dataset["date"], dtype=object)
    else:
        segment_starts = np.array([""] * N, dtype=object)

    # Compute normalization stats from training set.
    train_users = split_users.get("train", set())
    train_mask = np.array([uid in train_users for uid in user_ids])

    logger.info(
        "Computing normalization stats from %d training samples...", train_mask.sum()
    )
    n_channels = 19
    sums = np.zeros(n_channels, dtype=np.float64)
    sq_sums = np.zeros(n_channels, dtype=np.float64)
    counts = np.zeros(n_channels, dtype=np.float64)

    train_indices = np.where(train_mask)[0]
    stats_n = min(len(train_indices), 10000)
    rng = np.random.RandomState(seed)
    stats_indices = (
        rng.choice(train_indices, stats_n, replace=False) if stats_n > 0 else []
    )

    for idx in stats_indices:
        vals = np.array(hf_dataset[int(idx)]["values"], dtype=np.float32)
        mask = np.array(hf_dataset[int(idx)]["mask"], dtype=np.float32)
        if vals.ndim == 2:
            vals_ch = vals[:, :n_channels]
            mask_ch = mask[:, :n_channels]
        else:
            continue
        observed = mask_ch > 0.5
        for c in range(n_channels):
            obs = vals_ch[:, c][observed[:, c]]
            if len(obs) > 0:
                sums[c] += obs.sum()
                sq_sums[c] += (obs**2).sum()
                counts[c] += len(obs)

    means = np.where(counts > 0, sums / counts, 0.0).astype(np.float32)
    stds = np.where(
        counts > 1,
        np.sqrt((sq_sums / counts) - (sums / counts) ** 2),
        1.0,
    ).astype(np.float32)
    stds = np.maximum(stds, 1e-6)

    # Extract features in batches.
    all_features: list[np.ndarray] = []

    for start in tqdm(range(0, N, batch_size), desc="Encoding", unit="batch"):
        end = min(start + batch_size, N)
        batch_tensors = np.zeros((end - start, 168, 38), dtype=np.float32)

        for i, idx in enumerate(range(start, end)):
            row = hf_dataset[int(idx)]
            vals = np.array(row["values"], dtype=np.float32)
            mask = np.array(row["mask"], dtype=np.float32)

            T = min(vals.shape[0], 168)
            vals_ch = vals[:T, :n_channels]
            mask_ch = mask[:T, :n_channels]

            # Z-score normalize.
            observed = mask_ch > 0.5
            normalized = np.where(observed, (vals_ch - means) / stds, np.nan)

            batch_tensors[i, :T, :n_channels] = normalized
            batch_tensors[i, :T, n_channels : 2 * n_channels] = 1.0 - mask_ch

        embeddings = encoder.encode(batch_tensors)
        all_features.append(np.asarray(embeddings, dtype=np.float32))

    features = np.concatenate(all_features, axis=0)
    logger.info(
        "Extracted %dD features for %d samples", features.shape[1], features.shape[0]
    )

    return features, user_ids, segment_starts


# ---------------------------------------------------------------------------
# Imputation evaluation
# ---------------------------------------------------------------------------


def evaluate_imputation(
    imputer: Imputer,
    masking_scenarios: str | list[str] = "all",
    data_dir: str | Path | None = None,
    seed: int = 42,
) -> ImputationResults:
    """Run imputation evaluation with a custom imputer.

    Fits the imputer on training data, then evaluates on validation and test
    sets using up to 6 masking scenarios.

    Args:
        imputer: Object with `fit(data, masks)` and
            `impute(data, observed_mask, target_mask)` methods.
        masking_scenarios: "all" to run all 6 scenarios, or a list of scenario
            name strings.
        data_dir: Path to the `daily_hf` dataset directory. None uses the
            default location.
        seed: Random seed for mask generation.

    Returns:
        An ImputationResults instance with per-scenario, per-split metrics.

    Raises:
        ValueError: If an unknown masking scenario name is provided.
    """
    from openmhc._constants import MASKING_SCENARIOS

    # Resolve scenarios.
    if masking_scenarios == "all":
        scenario_list = list(MASKING_SCENARIOS)
    elif isinstance(masking_scenarios, str):
        scenario_list = [masking_scenarios]
    else:
        scenario_list = list(masking_scenarios)

    # Validate scenario names.
    for s in scenario_list:
        if s not in MASKING_SCENARIOS:
            raise ValueError(
                f"Unknown masking scenario: {s!r}. "
                f"Valid scenarios: {MASKING_SCENARIOS}"
            )

    paths = _DatasetPaths.resolve(data_dir)
    _ensure_labels_env(paths.labels_dir)
    daily_hf_dir = paths.daily_hf

    # Build a minimal config.
    from imputation_evaluation.config import (
        DataConfig,
        EvalConfig,
        ImputationEvalConfig,
        MaskingConfig,
        MethodConfig,
        OutputConfig,
        SensitivityConfig,
        VisualizationConfig,
        WandbConfig,
    )
    from imputation_evaluation.data.data_loader import ImputationDataLoader

    masking_cfg = MaskingConfig(mask_seed=seed)
    masking_cfg.random_noise.enabled = "random_noise" in scenario_list
    masking_cfg.temporal_slice.enabled = "temporal_slice" in scenario_list
    masking_cfg.signal_slice.enabled = "signal_slice" in scenario_list
    masking_cfg.sleep_gap.enabled = "sleep_gap" in scenario_list
    masking_cfg.workout_gap.enabled = "workout_gap" in scenario_list
    masking_cfg.intensity_failure.enabled = "intensity_failure" in scenario_list

    data_cfg = DataConfig(
        daily_hf_dir=str(daily_hf_dir),
        split_file=str(paths.splits_file),
        split_seed=seed,
        batch_size=5000,
        num_workers=4,
        num_eval_workers=1,
    )

    eval_cfg = EvalConfig(
        include_ks=False,
        include_wasserstein=False,
        compute_metrics=True,
        save_pairs=False,
    )

    cfg = ImputationEvalConfig(
        seed=seed,
        data=data_cfg,
        masking=masking_cfg,
        method=MethodConfig(type="mean"),
        output=OutputConfig(),
        evaluation=eval_cfg,
        visualization=VisualizationConfig(),
        sensitivity=SensitivityConfig(),
        wandb=WandbConfig(),
    )

    # Load data.
    loader = ImputationDataLoader(cfg.data)
    loaded = loader.load()

    # Generate masks.
    from imputation_evaluation.masking import MaskCacheGenerator, create_mask_generators

    generators = create_mask_generators(cfg.masking)
    mask_gen = MaskCacheGenerator(
        generators=generators,
        seed=cfg.masking.mask_seed,
    )
    mask_cache = mask_gen.generate(loaded.val_loader, loaded.test_loader)

    # Wrap the user's imputer in our internal method interface.
    adapter = _ImputerMethodAdapter(imputer)

    logger.info("Fitting imputer on training data...")
    adapter.fit(loaded.train_loader)

    # Run evaluation.
    from imputation_evaluation.evaluation.evaluator import ImputationEvaluator

    evaluator = ImputationEvaluator(
        scenarios=[g.name for g in generators],
        num_eval_workers=cfg.data.num_eval_workers,
        include_ks=cfg.evaluation.include_ks,
        include_wasserstein=cfg.evaluation.include_wasserstein,
        compute_metrics=True,
        save_pairs=False,
    )

    results = evaluator.run(
        val_loader=loaded.val_loader,
        test_loader=loaded.test_loader,
        mask_cache=mask_cache,
        method=adapter,
        channel_stds=loaded.channel_stds,
    )

    return ImputationResults(scenarios=results.get("scenarios", results))


class _ImputerMethodAdapter:
    """Adapt a user's Imputer to the internal ImputationMethod interface.

    This adapter collects batched training data, computes channel standard
    deviations for normalized metrics, and translates argument names between
    the public Imputer protocol and the internal evaluation pipeline.

    Attributes:
        name: Name of the imputation method (defaults to "custom_imputer").
        channel_stds: Per-channel standard deviations computed during fit,
            or None before fit is called.
    """

    def __init__(self, imputer: Imputer) -> None:
        """Initialize the adapter.

        Args:
            imputer: Object implementing the Imputer protocol.
        """
        self._imputer = imputer
        self._channel_stds: np.ndarray | None = None

    @property
    def name(self) -> str:
        """Return the method name."""
        return getattr(self._imputer, "name", "custom_imputer")

    @property
    def channel_stds(self) -> np.ndarray | None:
        """Return per-channel standard deviations from training data."""
        return self._channel_stds

    def fit(self, train_loader) -> None:
        """Collect training data from the loader and call the user's fit.

        Args:
            train_loader: DataLoader yielding batches of training data with
                "values" and "mask" keys.
        """
        all_data = []
        all_masks = []
        for batch in train_loader:
            data = batch["values"] if isinstance(batch, dict) else batch[0]
            mask = batch["mask"] if isinstance(batch, dict) else batch[1]
            all_data.append(np.asarray(data, dtype=np.float32))
            all_masks.append(np.asarray(mask, dtype=np.float32))

        data = np.concatenate(all_data, axis=0)
        masks = np.concatenate(all_masks, axis=0)

        # Compute channel stds for normalized metrics.
        observed = masks > 0.5
        stds = []
        for c in range(data.shape[1]):
            vals = data[:, c, :][observed[:, c, :]]
            stds.append(float(np.std(vals)) if len(vals) > 1 else 1.0)
        self._channel_stds = np.array(stds, dtype=np.float32)

        self._imputer.fit(data, masks)

    def impute(
        self,
        data: np.ndarray,
        original_masks: np.ndarray,
        artificial_masks: np.ndarray,
    ) -> np.ndarray:
        """Delegate to the user's impute, translating argument names.

        Args:
            data: Sensor values of shape (N, 19, 1440) with NaN at masked
                positions.
            original_masks: Binary mask of shape (N, 19, 1440) where
                1 = originally observed.
            artificial_masks: Binary mask of shape (N, 19, 1440) where
                1 = positions to impute.

        Returns:
            Array of shape (N, 19, 1440) with imputed values.
        """
        return self._imputer.impute(
            data=data,
            observed_mask=original_masks,
            target_mask=artificial_masks,
        )

    def prepare_split(self, *args, **kwargs) -> None:
        """No-op; some internal methods use this hook."""
