"""Public evaluation functions for OpenMHC.

These functions provide a simple interface to the benchmark's evaluation
pipelines. They accept duck-typed Encoder/Imputer objects and return
structured results.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from openmhc._protocols import Encoder, Imputer

from openmhc._dataset import (
    EXPECTED_N_USERS,
    Version,
    read_dataset_marker,
)
from openmhc._dataset import (
    data_dir as _resolve_dataset_root,
)
from openmhc._results import ForecastingResults, ImputationResults, PredictionResults

logger = logging.getLogger(__name__)

# Pre-computed max91d masks shipped with this repo (val/ + test/ subdirs, one
# .npz per scenario).  These are required for full-dataset evaluation because
# regenerating masks on the fly produces a different random sample each run,
# which would make leaderboard scores non-reproducible.
_REPO_ROOT = Path(__file__).parent.parent.parent
_MAX91D_MASKS_DIR = (
    _REPO_ROOT / "data" / "imputation" / "masks" / "sharable_users_seed42_2026_max91d"
)
_XS_MASKS_DIR = _REPO_ROOT / "data" / "imputation" / "masks" / "sharable_users_seed42_2026_xs"


_SPLIT_FILENAMES: dict[str, str] = {
    "full": "sharable_users_seed42_2026.json",
    "xs": "sharable_users_seed42_2026_xs.json",
}


@dataclass
class _DatasetPaths:
    """Resolved paths into the dataset directory.

    All paths are derived from a single ``root`` so the API stays consistent
    with what :func:`openmhc.download_dataset` produces. The expected layout
    matches DATASET.md.

    The resolver never falls back. The caller passes ``version`` explicitly,
    the resolver cross-checks it against the root's ``dataset_version.json``
    marker, and any mismatch between the requested version, the marker, and
    the actual user count in the split file is raised. Use
    :meth:`require` to validate the existence of just the sub-paths a given
    track needs.
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
    hourly_trajectory: Path
    forecasting_sample_index_dir: Path
    version: str = "full"  # "xs" or "full"

    @classmethod
    def resolve(
        cls,
        override: str | Path | None = None,
        version: Version | None = None,
    ) -> _DatasetPaths:
        """Build the paths bundle for an explicit version + dataset root.

        Args:
            override: Explicit dataset root. Falls back to ``MHC_DATA_DIR``
                when ``None``. Never falls back to ``~/.cache/openmhc/data``.
            version: ``"xs"`` or ``"full"``. Required — there is no
                filename-based auto-detect. The version is cross-checked
                against the root's ``dataset_version.json`` marker and
                against the user count in the resolved split file.

        Raises:
            ValueError: If ``version`` is not provided, is not one of
                ``"xs"`` / ``"full"``, or disagrees with the marker /
                split-file contents.
            FileNotFoundError: If the dataset root is missing or has no
                ``dataset_version.json`` marker (see
                :func:`openmhc.write_dataset_marker` to backfill it).
        """
        if version is None:
            raise ValueError(
                "_DatasetPaths.resolve(version=...) is required. "
                "Auto-detection by filename has been removed; pass "
                "version='xs' or version='full' explicitly so the resolver "
                "can cross-check it against the dataset_version.json marker."
            )
        if version not in EXPECTED_N_USERS:
            raise ValueError(f"version must be one of {sorted(EXPECTED_N_USERS)}, got {version!r}")

        root = _resolve_dataset_root(override)

        marker = read_dataset_marker(root)
        if marker["version"] != version:
            raise ValueError(
                f"Dataset at {root} is version {marker['version']!r} "
                f"(per dataset_version.json) but the caller requested "
                f"{version!r}. Point data_dir / MHC_DATA_DIR at the correct "
                f"root, or pass version={marker['version']!r}."
            )

        splits_file = root / "splits" / _SPLIT_FILENAMES[version]
        cls._validate_split_file(splits_file, version, marker)

        return cls(
            root=root,
            daily_hourly_hf=root / "processed" / "daily_hourly_hf",
            daily_hf=root / "processed" / "daily_hf",
            window_index=root / "processed" / "window_index_w7_s7_d5.parquet",
            weekly_labels_lookup=root / "processed" / "weekly_labels_lookup_stride7.parquet",
            splits_file=splits_file,
            norm_stats=root / "processed" / "normalization_stats_hourly.json",
            clip_dates=root / "labels" / "clip_dates.json",
            labels_dir=root / "labels",
            hourly_trajectory=root / "hourly_trajectory",
            forecasting_sample_index_dir=root / "forecasting_sample_index",
            version=version,
        )

    @staticmethod
    def _validate_split_file(splits_file: Path, version: str, marker: dict) -> None:
        """Verify the split file exists and its user count matches the marker."""
        if not splits_file.exists():
            raise FileNotFoundError(
                f"Split file for version {version!r} not found:\n  {splits_file}\n\n"
                f"Expected layout under the dataset root: "
                f"splits/{_SPLIT_FILENAMES[version]}"
            )
        try:
            payload = json.loads(splits_file.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed split file {splits_file}: {exc}") from exc
        n_users = sum(len(v) for v in payload.values() if isinstance(v, (list, set, tuple)))
        expected = marker.get("n_users", EXPECTED_N_USERS[version])
        if n_users != expected:
            raise ValueError(
                f"Split file {splits_file} contains {n_users} users, but version "
                f"{version!r} expects {expected}. The split file appears to be "
                f"from a different release. Re-download with "
                f"openmhc.download_dataset(version={version!r}) or replace the "
                f"split file with the correct one."
            )

    def require(self, *attr_names: str) -> None:
        """Assert that the named sub-paths exist; raise a combined error if not.

        Each track only needs a subset of the resolved paths — Track 1 needs
        ``daily_hourly_hf`` + ``window_index`` + ``weekly_labels_lookup``,
        Track 2 needs ``daily_hf``, Track 3 needs ``hourly_trajectory`` +
        ``forecasting_sample_index_dir``. Call this from the corresponding
        ``evaluate_*`` function so a missing artifact is reported up front
        with a single error listing every missing piece.

        Args:
            *attr_names: Names of attributes on ``self`` whose paths must
                exist.

        Raises:
            FileNotFoundError: If any of the named paths is missing. The
                message lists every missing path at once.
        """
        missing: list[tuple[str, Path]] = []
        for name in attr_names:
            path = getattr(self, name)
            if not path.exists():
                missing.append((name, path))
        if missing:
            lines = "\n".join(f"  {n}: {p}" for n, p in missing)
            raise FileNotFoundError(
                f"Dataset at {self.root} (version={self.version!r}) is missing "
                f"required files:\n{lines}\n\n"
                "Re-run openmhc.download_dataset() or restore the dataset "
                "from the upstream release."
            )


_LABELS_PAYLOAD_ENV_FILES = {
    "LABELS_DATA_PATH": "last_labels.json",
    "CONTEXT_LABELS_PATH": "context_labels.json",
    "ENROLLMENT_DATA_PATH": "enrollment_info.json",
    "LABEL_VALIDITY_PATH": "label_validity.json",
    "HEALTHKIT_DAILY_PATH": "healthkit_daily.json",
}


def _ensure_labels_env(labels_dir: Path) -> None:
    """Point `labels.api` at a resolved dataset root for large label payloads.

    `labels.api` reads its data-file paths from env vars at import time and
    caches them in module-level Path constants. We set each var if the user
    hasn't, then reload the module if it was already imported (e.g. via
    ``openmhc.list_tasks()``) so the cached paths reflect the new values.

    Without this, paths fall back to the repo-local ``data/labels/`` (which
    ships only schema files), and `enrollment_info.json` / `label_validity.json`
    silently load as empty — breaking the imputation sensitivity pathway and
    the default ``return_valid_only=True`` behaviour of ``get_labels``.
    """
    changed = False
    for env_var, filename in _LABELS_PAYLOAD_ENV_FILES.items():
        if not os.getenv(env_var):
            os.environ[env_var] = str(labels_dir / filename)
            changed = True

    if changed:
        import importlib
        import sys

        if "labels.api" in sys.modules:
            importlib.reload(sys.modules["labels.api"])


# ---------------------------------------------------------------------------
# Prediction evaluation
# ---------------------------------------------------------------------------


def evaluate_prediction(
    encoder: Encoder,
    version: Version,
    tasks: str | list[str] = "all",
    data_dir: str | Path | None = None,
    seed: int = 42,
) -> PredictionResults:
    """Run health-prediction evaluation with a custom encoder.

    Encodes weekly sensor tensors via `encoder.encode()` and evaluates the
    resulting embeddings on up to 33 health prediction tasks using linear
    probes (logistic regression for binary, ordinal logistic for ordinal,
    ridge regression for continuous).

    Args:
        encoder: Object with an `encode(weekly_tensors) -> embeddings` method.
            Input shape is (B, 168, 38), output shape is (B, D).
        version: ``"xs"`` (593-user reviewer subset) or ``"full"``
            (11,894-user leaderboard split). Required — cross-checked
            against the dataset root's ``dataset_version.json`` marker.
        tasks: "all" to run all 33 tasks, or a list of task name strings.
        data_dir: Override for the dataset root (the same root that
            ``download_dataset`` writes to). If omitted, ``MHC_DATA_DIR`` must
            be set. All sub-paths (`processed/daily_hourly_hf/`, `splits/`,
            `labels/`, etc.) are derived from this root.
        seed: Random seed for classifiers and splits.

    Returns:
        A PredictionResults instance with per-task metrics and a global score
        (mean AUROC across binary tasks).
    """
    import pandas as pd

    paths = _DatasetPaths.resolve(data_dir, version=version)
    paths.require(
        "daily_hourly_hf",
        "window_index",
        "weekly_labels_lookup",
        "labels_dir",
    )
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

    segment_starts_ns = pd.to_datetime(segment_starts.tolist()).astype("int64").values

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
                if task_name not in labels_df.columns:
                    logger.warning("Task %s missing from labels lookup, skipping", task_name)
                    break
                task_labels = labels_df[task_name].values
                try:
                    splits = store.aggregate_for_task(
                        task_labels,
                        task_type,
                        clip,
                        tw,
                        split_users,
                        pooling_method="mean",
                    )
                except Exception as e:
                    logger.warning("Task %s/%s failed aggregation: %s", task_name, tw.name, e)
                    continue

                # Current signature returns 4-tuple (X, y, user_ids, n_weeks)
                # under "train"/"val"/"test" keys (split file used "validation").
                X_train, y_train, *_ = splits["train"]
                X_val, y_val, *_ = splits["val"]
                X_test, y_test, *_ = splits["test"]

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
                        records.append(
                            {
                                **row,
                                "metric": "auprc",
                                "value": m["auprc"],
                            }
                        )

                    elif task_type == "ordinal":
                        test_pred = clf.predict(X_test)
                        m = compute_ordinal_metrics(y_test, test_pred)
                        for metric_name in ("spearman_r", "qwk", "mae_ordinal"):
                            records.append({**row, "metric": metric_name, "value": m[metric_name]})

                    elif task_type == "multiclass":
                        test_pred = clf.predict(X_test)
                        m = compute_multiclass_metrics(y_test, test_pred)
                        for metric_name in ("accuracy", "f1_macro"):
                            records.append({**row, "metric": metric_name, "value": m[metric_name]})

                    elif task_type == "regression":
                        test_pred = clf.predict(X_test)
                        m = compute_regression_metrics(y_test, test_pred)
                        for metric_name in ("mse", "mae", "pearson_r", "r2"):
                            records.append({**row, "metric": metric_name, "value": m[metric_name]})

    global_score = float(np.mean(binary_aurocs)) if binary_aurocs else 0.0

    return PredictionResults(records=records, global_score=global_score)


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

    logger.info("Computing normalization stats from %d training samples...", train_mask.sum())
    n_channels = 19
    sums = np.zeros(n_channels, dtype=np.float64)
    sq_sums = np.zeros(n_channels, dtype=np.float64)
    counts = np.zeros(n_channels, dtype=np.float64)

    train_indices = np.where(train_mask)[0]
    stats_n = min(len(train_indices), 10000)
    rng = np.random.RandomState(seed)
    stats_indices = rng.choice(train_indices, stats_n, replace=False) if stats_n > 0 else []

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
    logger.info("Extracted %dD features for %d samples", features.shape[1], features.shape[0])

    return features, user_ids, segment_starts


# ---------------------------------------------------------------------------
# Imputation evaluation
# ---------------------------------------------------------------------------


def evaluate_imputation(
    imputer: Imputer,
    version: Version,
    masking_scenarios: str | list[str] = "all",
    data_dir: str | Path | None = None,
    seed: int = 42,
    *,
    n_days: int = 1,
    bootstrap: bool | dict = False,
    max_samples: int | None = None,
    num_workers: int = 0,
    num_eval_workers: int = 1,
    pin_memory: bool = False,
) -> ImputationResults:
    """Run imputation evaluation with a custom imputer.

    The imputer is responsible for its own setup (loading checkpoints,
    computing training statistics, building per-user state) — typically
    in ``__init__``. The harness only calls ``impute`` and never asks
    the imputer to train or fit.

    Args:
        imputer: Object implementing ``impute(data, observed_mask,
            target_mask, *, sample_indices=None, user_ids=None,
            dates=None, day_offsets=None)`` per the ``Imputer`` protocol.
            ``day_offsets`` is only forwarded when ``n_days > 1``; it carries
            per-window calendar-day deltas (``-1`` for padded slots) for
            calendar-aware models (e.g. RoPE day embeddings).
        version: ``"full"`` (11,894-user leaderboard split) or ``"xs"``
            (593-user reviewer subset). Required — cross-checked against
            the dataset root's ``dataset_version.json`` marker.
        masking_scenarios: "all" to run all 6 scenarios, or a list of
            scenario name strings.
        data_dir: Override for the dataset root (the same root that
            ``download_dataset`` writes to). If omitted, ``MHC_DATA_DIR`` must
            be set. All sub-paths (`processed/daily_hf/`, `splits/`, `labels/`,
            etc.) are derived from this root.
        seed: Random seed for mask generation (XS only; full always uses
            pre-computed masks).
        n_days: Number of consecutive days per evaluation window (1-7).
            Defaults to ``1`` (single-day windows — matches the historical
            behavior and all daily models). Setting ``n_days=7`` enables
            multi-day windows required by weekly models like
            ``LSM2WeeklySparseImputer`` and any 7-day PyPOTS variant. The
            internal harness assembles non-overlapping per-user windows from
            the daily HF dataset; the imputer receives tensors of shape
            ``(B, 19, n_days * 1440)``.
        bootstrap: Opt-in participant-level cluster bootstrap. ``False``
            (default) skips it. ``True`` enables with defaults
            (``n_boot=1000, ci_level=0.95, seed=42, include_auc=True``).
            Pass a dict to override fields, e.g.
            ``{"n_boot": 500, "include_auc": False}``. When enabled, raw
            (gt, pred) pairs are written to a temporary directory that is
            cleaned up before this function returns; CI/SE fields appear as
            sibling columns in ``ImputationResults.to_dataframe()``.
        max_samples: Limit samples per split for testing/debugging (None =
            no limit). Mirrors ``evaluate_forecasting``. Plumbs into
            ``DataConfig.max_samples_per_split``.
        num_workers: DataLoader worker processes for loading, mask generation,
            and the one train pass that computes metric-normalization stats.
            Defaults to ``0`` (synchronous; notebook-safe). Raise toward your
            CPU count to overlap I/O with compute. Plumbs into
            ``DataConfig.num_workers``.
        num_eval_workers: Parallel processes for the evaluation loop. Defaults
            to ``1`` (sequential). With ``> 1`` the harness evaluates batches
            concurrently via ``ProcessPoolExecutor`` (batch-level, all
            scenarios per worker) — dramatically faster on the full split, with
            results numerically identical to the sequential path. Caveat: the
            imputer is pickled to each worker, so it must be importable; a class
            defined in a notebook cell can fail under the ``spawn`` start method
            (works under Linux ``fork``). Plumbs into
            ``DataConfig.num_eval_workers``.
        pin_memory: DataLoader ``pin_memory`` flag. Defaults to ``False``; set
            ``True`` to speed host→GPU transfer for a GPU imputer. Plumbs into
            ``DataConfig.pin_memory``.

    Returns:
        An ImputationResults instance with per-scenario, per-split metrics.

    Raises:
        TypeError: If ``imputer`` does not implement ``impute``.
        ValueError: If an unknown masking scenario name or version is provided.
        FileNotFoundError: If no dataset is found, or if the full dataset is
            selected but the pre-computed max91d masks are missing from the
            repository.
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
                f"Unknown masking scenario: {s!r}. Valid scenarios: {MASKING_SCENARIOS}"
            )

    paths = _DatasetPaths.resolve(data_dir, version=version)
    paths.require("daily_hf", "splits_file", "labels_dir")
    _ensure_labels_env(paths.labels_dir)

    from imputation_evaluation.config import (
        BootstrapConfig,
        DataConfig,
        EvalConfig,
        ImputationEvalConfig,
        MaskingConfig,
        OutputConfig,
        SensitivityConfig,
        VisualizationConfig,
        WandbConfig,
    )
    from imputation_evaluation.runner import run_eval

    if bootstrap is False or bootstrap is None:
        bootstrap_cfg = BootstrapConfig()
    elif bootstrap is True:
        bootstrap_cfg = BootstrapConfig(enabled=True)
    elif isinstance(bootstrap, dict):
        overrides = dict(bootstrap)
        overrides.setdefault("enabled", True)
        bootstrap_cfg = BootstrapConfig(**overrides)
    else:
        raise TypeError(f"bootstrap must be bool or dict, got {type(bootstrap).__name__}")

    masking_cfg = MaskingConfig(mask_seed=seed)
    masking_cfg.random_noise.enabled = "random_noise" in scenario_list
    masking_cfg.temporal_slice.enabled = "temporal_slice" in scenario_list
    masking_cfg.signal_slice.enabled = "signal_slice" in scenario_list
    masking_cfg.sleep_gap.enabled = "sleep_gap" in scenario_list
    masking_cfg.workout_gap.enabled = "workout_gap" in scenario_list
    masking_cfg.intensity_failure.enabled = "intensity_failure" in scenario_list

    if paths.version == "full":
        if not _MAX91D_MASKS_DIR.exists():
            raise FileNotFoundError(
                f"Pre-computed masks not found at:\n  {_MAX91D_MASKS_DIR}\n\n"
                "Full-dataset evaluation requires the max91d masks to ensure "
                "reproducible leaderboard scores across runs. The masks ship with "
                "this repository — make sure you cloned it with `git clone` and "
                "installed with `pip install -e .`. If the masks directory is "
                "missing, re-clone or check that `data/imputation/masks/` was not "
                "excluded by a sparse-checkout or .gitignore rule."
            )
        masking_cfg.masks_file = str(_MAX91D_MASKS_DIR)
    elif (
        paths.version == "xs"
        and seed == 42
        and n_days == 1
        and max_samples is None
        and _XS_MASKS_DIR.exists()
    ):
        # XS ships precomputed masks too (mirrors `full`), but only for the
        # canonical full-split config they were generated for: seed 42,
        # single-day windows, all XS val+test samples. A scenario subset still
        # loads fine (the cache holds all six). Other seed / n_days, or a
        # max_samples-bounded run, fall through to on-the-fly generation below:
        # the cache spans the full split, so its applicable indices would
        # overrun a max_samples-limited dataset — and bounded generation is
        # cheap anyway (it only masks the small subset). On-the-fly over the
        # full split is the slow case (~20 min) this cache exists to avoid.
        masking_cfg.masks_file = str(_XS_MASKS_DIR)
    # else (xs with a non-canonical config, max_samples set, or cache absent):
    # masks_file stays None → runner generates masks on the fly.

    data_cfg = DataConfig(
        daily_hf_dir=str(paths.daily_hf),
        split_file=str(paths.splits_file),
        split_seed=seed,
        batch_size=5000,
        num_workers=num_workers,
        num_eval_workers=num_eval_workers,
        pin_memory=pin_memory,
        n_days=n_days,
        max_samples_per_split=max_samples,
    )

    eval_cfg = EvalConfig(
        compute_metrics=True,
        save_pairs=False,
    )

    adapter = _ImputerMethodAdapter(imputer)
    logger.info("Running imputation eval with custom imputer...")

    def _build_cfg(output_dir: str) -> ImputationEvalConfig:
        # ``method`` is omitted: ``ImputationEvalConfig`` defaults it to a stock
        # ``MethodConfig``, and ``run_eval`` ignores ``cfg.method`` entirely when
        # we pass our own ``method=adapter`` below.
        return ImputationEvalConfig(
            seed=seed,
            data=data_cfg,
            masking=masking_cfg,
            output=OutputConfig(results_dir=output_dir),
            evaluation=eval_cfg,
            visualization=VisualizationConfig(),
            sensitivity=SensitivityConfig(),
            bootstrap=bootstrap_cfg,
            wandb=WandbConfig(),
        )

    if bootstrap_cfg.enabled:
        # Bootstrap requires pair files on disk; stash them in a tempdir so the
        # user's data root stays clean. The runner writes bootstrap_metrics.json
        # under results_dir too — also lives + dies with the tempdir.
        import tempfile

        with tempfile.TemporaryDirectory(prefix="openmhc_bootstrap_") as td:
            cfg = _build_cfg(td)
            results = run_eval(cfg, method=adapter)
    else:
        cfg = _build_cfg(OutputConfig().results_dir)
        results = run_eval(cfg, method=adapter)

    return ImputationResults(scenarios=results.get("scenarios", results))


_N_CHANNELS = 19


class _ImputerMethodAdapter:
    """Adapt a user's Imputer to the internal ImputationMethod interface.

    Responsibilities:

    1. **Stream the train split once** to compute per-channel standard
       deviations for metric normalization (this is the harness's
       concern, never the user's).
    2. **Cache per-split user_id and date arrays** in
       :meth:`prepare_split` so they can be sliced per batch.
    3. **Filter forwarded kwargs** by inspecting the user's ``impute``
       signature so methods that only declare three positional args
       still work, and personalized methods receive ``user_ids`` /
       ``dates`` / ``sample_indices``.
    4. **Compute a channel-aware global fallback fill** during the same
       train pass: per-channel observed mean for continuous channels
       (0–6) and per-channel majority class for binary channels
       (7–18). Exposed via :attr:`fallback_fill` so the harness can
       substitute NaN cells the user's ``impute`` failed to produce.
    """

    def __init__(self, imputer: Imputer) -> None:
        if not hasattr(imputer, "impute"):
            raise TypeError(
                "Imputer must define an `impute(data, observed_mask, "
                "target_mask, ...)` method. The fit-based API was removed; "
                "see openmhc.imputers for ready-to-use baselines."
            )
        self._imputer = imputer
        self._channel_stds: np.ndarray | None = None
        self._fallback_fill: np.ndarray | None = None
        # Per-split metadata, populated by prepare_split.
        self._current_user_ids: list[str] | None = None
        self._current_dates: list[str] | None = None

        # Inspect the user's impute signature once.
        import inspect as _inspect

        try:
            params = _inspect.signature(imputer.impute).parameters
            accepts_anything = any(
                p.kind == _inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
            self._fwd_sample_indices = accepts_anything or "sample_indices" in params
            self._fwd_user_ids = accepts_anything or "user_ids" in params
            self._fwd_dates = accepts_anything or "dates" in params
            self._fwd_day_offsets = accepts_anything or "day_offsets" in params
        except (TypeError, ValueError):
            # Built-in or C-extension impute: forward everything.
            self._fwd_sample_indices = True
            self._fwd_user_ids = True
            self._fwd_dates = True
            self._fwd_day_offsets = True

    @property
    def name(self) -> str:
        return getattr(self._imputer, "name", "custom_imputer")

    @property
    def channel_stds(self) -> np.ndarray | None:
        return self._channel_stds

    @property
    def fallback_fill(self) -> np.ndarray | None:
        """Per-channel global fill used to substitute NaN cells the imputer fails to produce."""
        return self._fallback_fill

    def fit(self, train_loader) -> None:
        """Stream the train loader once to compute per-channel stds and a fallback fill.

        Does not invoke any user method — all imputer setup happens
        in the user's ``__init__``. The same single pass produces both
        ``channel_stds`` (metric normalization) and ``fallback_fill``
        (channel-aware global substitution for non-finite imputed cells).
        """
        from data.processing.hf_config import (
            BINARY_CHANNEL_INDICES,
            CONTINUOUS_CHANNEL_INDICES,
        )

        sums = np.zeros(_N_CHANNELS, dtype=np.float64)
        sq_sums = np.zeros(_N_CHANNELS, dtype=np.float64)
        counts = np.zeros(_N_CHANNELS, dtype=np.float64)
        for batch in train_loader:
            data = batch["values"] if isinstance(batch, dict) else batch[0]
            mask = batch["mask"] if isinstance(batch, dict) else batch[1]
            data = np.asarray(data, dtype=np.float32)
            mask = np.asarray(mask, dtype=np.float32)
            obs = (mask > 0.5) & np.isfinite(data)
            data_obs = np.where(obs, data, 0.0)
            sums += data_obs.sum(axis=(0, 2))
            sq_sums += (data_obs**2).sum(axis=(0, 2))
            counts += obs.sum(axis=(0, 2))
        safe = np.maximum(counts, 1)
        means = np.where(counts > 0, sums / safe, 0.0)
        variance = np.where(counts > 0, (sq_sums / safe) - means**2, 0.0)
        stds = np.sqrt(np.maximum(variance, 0.0))
        stds = np.where(counts > 1, stds, 1.0)
        self._channel_stds = np.maximum(stds, 1e-6).astype(np.float32)

        # Fallback fill: continuous channels → mean, binary channels → majority class.
        fallback = np.zeros(_N_CHANNELS, dtype=np.float32)
        for ch in CONTINUOUS_CHANNEL_INDICES:
            fallback[ch] = float(means[ch])
        for ch in BINARY_CHANNEL_INDICES:
            fallback[ch] = 1.0 if means[ch] > 0.5 else 0.0
        self._fallback_fill = fallback

    def prepare_split(self, hf_dataset, split_indices, zero_to_nan_transform) -> None:
        """Cache per-split user_id and date arrays.

        Called once per eval split by the internal evaluator before
        batches are processed. Storing both columns up front avoids
        per-batch HF row reads.
        """
        all_user_ids = list(hf_dataset["user_id"])
        all_dates = list(hf_dataset["date"])
        self._current_user_ids = [all_user_ids[i] for i in split_indices]
        self._current_dates = [all_dates[i] for i in split_indices]

    def impute(
        self,
        data: np.ndarray,
        original_masks: np.ndarray,
        artificial_masks: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Forward to the user's impute, filtering kwargs by signature."""
        forward: dict = {
            "data": data,
            "observed_mask": original_masks,
            "target_mask": artificial_masks,
        }
        sample_indices = kwargs.get("sample_indices")
        if self._fwd_sample_indices and sample_indices is not None:
            forward["sample_indices"] = np.asarray(sample_indices)
        if sample_indices is not None and self._current_user_ids is not None:
            si = np.asarray(sample_indices)
            if self._fwd_user_ids:
                forward["user_ids"] = [self._current_user_ids[int(i)] for i in si]
            if self._fwd_dates:
                forward["dates"] = [self._current_dates[int(i)] for i in si]
        day_offsets = kwargs.get("day_offsets")
        if self._fwd_day_offsets and day_offsets is not None:
            forward["day_offsets"] = np.asarray(day_offsets)
        return self._imputer.impute(**forward)


# ---------------------------------------------------------------------------
# Forecasting evaluation (Track 3)
# ---------------------------------------------------------------------------


def evaluate_forecasting(
    forecaster,
    version: Version,
    forecasting_length: int = 24,
    data_dir: str | Path | None = None,
    seed: int = 42,
    max_samples: int | None = None,
    *,
    num_workers: int = 4,
) -> ForecastingResults:
    """Run forecasting evaluation (Track 3) with a custom forecaster.

    Args:
        forecaster: Object satisfying the :class:`Forecaster` protocol — has
            ``predict(history, horizon)`` returning a ``(n_channels, horizon)``
            point forecast (optionally a ``(point, quantiles)`` tuple).
            ``history`` is the full-prefix window; emit ``NaN`` for any cell the
            model cannot predict and the harness substitutes the Seasonal-Naive
            baseline (reported via ``ForecastingResults.overall_fallback_rate``).
        version: ``"xs"`` or ``"full"``. Required — cross-checked against
            the dataset root's ``dataset_version.json`` marker.
        forecasting_length: Forecast horizon in hours. Defaults to 24
            (matching the paper's Track 3 sub-task).
        data_dir: Override for the dataset root. If omitted,
            ``MHC_DATA_DIR`` must be set.
        seed: Random seed.
        max_samples: Limit prediction samples per user (debugging).
        num_workers: DataLoader worker processes for loading trajectories.
            Defaults to ``4``. The forecasting evaluator is sequential-only
            (no parallel-eval mode), so this only affects data loading;
            ``max_samples`` remains the main lever for keeping a run fast.
            Plumbs into ``DataConfig.num_workers``.

    Returns:
        :class:`ForecastingResults` with per-channel metrics.
    """
    paths = _DatasetPaths.resolve(data_dir, version=version)
    paths.require(
        "hourly_trajectory",
        "forecasting_sample_index_dir",
        "splits_file",
        "labels_dir",
    )
    _ensure_labels_env(paths.labels_dir)

    from forecasting_evaluation.config import (
        DataConfig,
        EvaluatorConfig,
        FeaturesConfig,
        ForecastingConfig,
        ForecastingEvalConfig,
        ForecastingModelConfig,
        OutputConfig,
    )
    from forecasting_evaluation.runner import run_eval

    # Pick the paper/Hydra-default sample-index file for the requested horizon:
    # the quality-filtered set (M = target day retained, H_7_3 = >=3 of prior 7
    # days retained, S_100 = <=100 start days/user, seed 42). Matches
    # configs/forecasting/data/default.yaml so the public API is paper-parity.
    sample_index_file = (
        paths.forecasting_sample_index_dir
        / f"sample_index_P_{forecasting_length}_M_H_7_3_S_100.json"
    )

    data_cfg = DataConfig(
        trajectory_hf_dir=str(paths.hourly_trajectory),
        split_file=str(paths.splits_file),
        day_remain_mask=str(paths.forecasting_sample_index_dir / "day_remain_mask.json"),
        sample_index_file=str(sample_index_file),
        split_seed=seed,
        num_workers=num_workers,
        max_samples=max_samples,
    )
    forecasting_cfg = ForecastingConfig(forecasting_length=forecasting_length)

    with tempfile.TemporaryDirectory(prefix="openmhc-fc-") as tmp_results:
        cfg = ForecastingEvalConfig(
            seed=seed,
            experiment_name="openmhc_run",
            debug_mode=False,
            data=data_cfg,
            forecasting=forecasting_cfg,
            model=ForecastingModelConfig(),  # ignored by _CustomModelEvaluator
            features=FeaturesConfig(),
            evaluator=EvaluatorConfig(),
            output=OutputConfig(results_dir=tmp_results),
        )
        # The user's forecaster satisfies the unified Forecaster contract
        # (``predict(history, horizon, *optional kwargs)``); the evaluator
        # invokes it directly through its duck-typed call path — no adapter.
        result = run_eval(cfg, model=forecaster)

    return ForecastingResults(
        per_channel=result.get("per_channel", {}),
        run_dir=str(result.get("run_dir", "")),
        n_samples=int(result.get("n_samples", 0)),
        overall_fallback_rate=float(result.get("overall_fallback_rate", 0.0)),
        fallback_rate=dict(result.get("fallback_rate", {})),
    )
