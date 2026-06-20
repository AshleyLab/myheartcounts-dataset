"""Evaluation metrics for downstream tasks.

Supports binary classification, multiclass classification, and regression tasks.
Regression metrics: MSE, MAE, Pearson correlation, R².

All metric functions return bootstrap standard errors (suffix ``_se``) alongside
point estimates.  The bootstrap resamples the test set 1000 times with
replacement and reports ``std(metric_values)`` as the SE.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from labels.api import LABEL_NAMES, LABEL_TYPES

# ---------------------------------------------------------------------------
# Bootstrap standard error
# ---------------------------------------------------------------------------

_N_BOOTSTRAP = 1000
_BOOTSTRAP_SEED = 42


def bootstrap_se(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = _N_BOOTSTRAP,
    seed: int = _BOOTSTRAP_SEED,
) -> float:
    """Compute bootstrap standard error for a metric.

    Args:
        y_true: Ground truth labels/values (N,).
        y_score: Predicted scores/labels/values (N,).
        metric_fn: ``f(y_true, y_score) -> float``.
        n_boot: Number of bootstrap resamples.
        seed: Random seed for reproducibility.

    Returns:
        Standard error (std of bootstrapped metric values).
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        try:
            scores[i] = metric_fn(y_true[idx], y_score[idx])
        except Exception:
            scores[i] = np.nan
    return float(np.nanstd(scores, ddof=1))


def get_task_type(task_name: str) -> str:
    """Determine task type from label_types.json.

    Args:
        task_name: Name of the label/task.

    Returns:
        One of "binary", "multiclass", or "regression".

    Raises:
        ValueError: If task_name is not found in LABEL_TYPES.
    """
    label_type = LABEL_TYPES.get(task_name)

    if label_type is None:
        available_labels = "\n".join(f"  - {label}" for label in sorted(LABEL_NAMES))
        raise ValueError(
            f"Unknown task name: {task_name}\nAvailable task names:\n{available_labels}"
        )

    if label_type == "binary":
        return "binary"
    elif label_type == "ordinal":
        return "ordinal"
    elif label_type == "categorical":
        return "multiclass"
    elif label_type == "continuous":
        return "regression"
    else:
        raise ValueError(f"Unknown label type '{label_type}' for task '{task_name}'")


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Compute AUROC and AUPRC for binary classification.

    Args:
        y_true: Ground truth binary labels (N,).
        y_prob: Predicted probabilities for positive class (N,).

    Returns:
        Dictionary with 'auroc', 'auprc', and their bootstrap SEs.
    """
    nan_result = {
        "auroc": float("nan"),
        "auprc": float("nan"),
        "auroc_se": float("nan"),
        "auprc_se": float("nan"),
    }

    # Handle edge cases
    if len(np.unique(y_true)) < 2:
        return nan_result

    # Check for NaN/Inf in predictions
    if np.isnan(y_prob).any() or np.isinf(y_prob).any():
        return nan_result

    # Clip probabilities to valid range to handle numerical issues
    y_prob = np.clip(y_prob, 1e-10, 1.0 - 1e-10)

    def _auroc(yt, yp):
        if len(np.unique(yt)) < 2:
            return np.nan
        return roc_auc_score(yt, yp)

    def _auprc(yt, yp):
        if len(np.unique(yt)) < 2:
            return np.nan
        return average_precision_score(yt, yp)

    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "auroc_se": bootstrap_se(y_true, y_prob, _auroc),
        "auprc_se": bootstrap_se(y_true, y_prob, _auprc),
    }


def compute_multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute accuracy and macro F1 for multiclass classification.

    Args:
        y_true: Ground truth class labels (N,).
        y_pred: Predicted class labels (N,).

    Returns:
        Dictionary with 'accuracy' and 'f1_macro' metrics.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def compute_ordinal_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute Spearman's Rho, Quadratic Weighted Kappa, and MAE for ordinal data.

    Args:
        y_true: Ground truth ordinal labels (N,).
        y_pred: Predicted ordinal labels (N,).

    Returns:
        Dictionary with 'spearman_r', 'qwk', 'mae_ordinal', and their bootstrap SEs.
    """

    def _spearman(yt, yp):
        if len(np.unique(yt)) < 2 or len(np.unique(yp)) < 2:
            return np.nan
        return spearmanr(yt, yp)[0]

    def _qwk(yt, yp):
        return cohen_kappa_score(yt, yp, weights="quadratic")

    def _mae(yt, yp):
        return mean_absolute_error(yt, yp)

    # 1. Spearman Rank Correlation
    spearman_val, _ = spearmanr(y_true, y_pred)

    # 2. Quadratic Weighted Kappa (QWK)
    qwk_val = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    # 3. Mean Absolute Error (MAE)
    mae_val = mean_absolute_error(y_true, y_pred)

    return {
        "spearman_r": float(spearman_val),
        "qwk": float(qwk_val),
        "mae_ordinal": float(mae_val),
        "spearman_r_se": bootstrap_se(y_true, y_pred, _spearman),
        "qwk_se": bootstrap_se(y_true, y_pred, _qwk),
        "mae_ordinal_se": bootstrap_se(y_true, y_pred, _mae),
    }


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MSE, MAE, Pearson correlation, and R² for regression.

    Args:
        y_true: Ground truth values (N,).
        y_pred: Predicted values (N,).

    Returns:
        Dictionary with 'mse', 'mae', 'pearson_r', 'r2', and their bootstrap SEs.
    """
    # Epsilon threshold for "zero variance" — guards against float-precision
    # residuals when y_true/y_pred are effectively constant after centering.
    _VAR_EPS = 1e-12

    def _pearson(yt, yp):
        if np.var(yt) < _VAR_EPS or np.var(yp) < _VAR_EPS:
            return np.nan
        return pearsonr(yt, yp)[0]

    def _mse(yt, yp):
        return mean_squared_error(yt, yp)

    def _mae(yt, yp):
        return mean_absolute_error(yt, yp)

    def _r2(yt, yp):
        if np.var(yt) < _VAR_EPS:
            return np.nan
        return r2_score(yt, yp)

    # Handle edge case of constant predictions or true values
    if np.var(y_pred) < _VAR_EPS:
        warnings.warn(
            "Predictions are constant (zero variance). Pearson correlation will be NaN.",
            UserWarning,
            stacklevel=2,
        )
        pearson_r = float("nan")
    elif np.var(y_true) < _VAR_EPS:
        warnings.warn(
            "True values are constant (zero variance). Pearson correlation will be NaN.",
            UserWarning,
            stacklevel=2,
        )
        pearson_r = float("nan")
    else:
        pearson_r, _ = pearsonr(y_true, y_pred)
        pearson_r = float(pearson_r)

    # R² — only undefined when y_true is constant (SS_tot = 0)
    if np.var(y_true) < _VAR_EPS:
        r2 = float("nan")
    else:
        r2 = float(r2_score(y_true, y_pred))

    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "pearson_r": pearson_r,
        "r2": r2,
        "mse_se": bootstrap_se(y_true, y_pred, _mse),
        "mae_se": bootstrap_se(y_true, y_pred, _mae),
        "pearson_r_se": bootstrap_se(y_true, y_pred, _pearson),
        "r2_se": bootstrap_se(y_true, y_pred, _r2),
    }


def compute_per_user_regression_metrics(
    user_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[dict[str, float], list[dict]]:
    """Per-user summaries for segment-level longitudinal regression.

    Used by the same-segment recovery evaluation (one prediction per user-week):
    groups predictions by user, computes R² and Pearson r within each user who
    contributes ≥2 observations, and returns distribution summaries of those
    per-user metrics (median, quartiles, mean).

    Users with a single observation contribute to ``n_users_total`` but are
    excluded from the per-user R² / Pearson r distributions (both are
    undefined with n=1). Users whose ``y_true`` has zero variance (all equal
    after centering) are included in ``per_user_records`` as NaN records (so
    callers can see they existed) but are excluded from the R² / Pearson r
    distribution summaries. ``n_users_zero_var`` surfaces the count of such
    users.

    Args:
        user_ids: ``(N,)`` user IDs, one per observation.
        y_true: ``(N,)`` ground truth values.
        y_pred: ``(N,)`` predicted values.

    Returns:
        ``(summary, per_user_records)`` where ``summary`` is a dict with
        distribution summaries and ``per_user_records`` is a list of dicts
        with keys ``user_id``, ``n_obs``, ``r2``, ``pearson_r``.
    """
    # Variance-equals-zero comparison uses an epsilon rather than exact equality
    # to tolerate float-precision residuals after person-mean centering.
    _VAR_EPS = 1e-12

    def _empty_summary(n_total: int, n_zero_var: int = 0) -> dict[str, float]:
        return {
            "n_users_total": int(n_total),
            "n_users_ge2obs": 0,
            "n_users_zero_var": int(n_zero_var),
            "median_user_r2": float("nan"),
            "q25_user_r2": float("nan"),
            "q75_user_r2": float("nan"),
            "mean_user_r2": float("nan"),
            "median_user_pearson_r": float("nan"),
            "mean_user_pearson_r": float("nan"),
        }

    user_ids = np.asarray(user_ids)
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    unique_users, inverse = np.unique(user_ids, return_inverse=True)
    n_total = len(unique_users)
    if n_total == 0:
        return _empty_summary(0), []

    per_user_records: list[dict] = []
    n_zero_var = 0

    for u_idx in range(n_total):
        mask = inverse == u_idx
        n_obs = int(mask.sum())
        if n_obs < 2:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        if np.var(yt) < _VAR_EPS:
            # R² / Pearson r undefined when labels are constant. Keep a NaN
            # record so callers can see the user existed.
            n_zero_var += 1
            per_user_records.append(
                {
                    "user_id": str(unique_users[u_idx]),
                    "n_obs": n_obs,
                    "r2": float("nan"),
                    "pearson_r": float("nan"),
                }
            )
            continue
        u_r2 = float(r2_score(yt, yp))
        u_r = float(pearsonr(yt, yp)[0]) if np.var(yp) > _VAR_EPS else float("nan")
        per_user_records.append(
            {
                "user_id": str(unique_users[u_idx]),
                "n_obs": n_obs,
                "r2": u_r2,
                "pearson_r": u_r,
            }
        )

    # Aggregates over records with defined R² / Pearson r only
    finite_r2 = [r["r2"] for r in per_user_records if not np.isnan(r["r2"])]
    finite_pr = [r["pearson_r"] for r in per_user_records if not np.isnan(r["pearson_r"])]

    if not finite_r2:
        return _empty_summary(n_total, n_zero_var=n_zero_var), per_user_records

    r2_arr = np.asarray(finite_r2, dtype=np.float64)
    r_arr = np.asarray(finite_pr, dtype=np.float64)

    summary = {
        "n_users_total": int(n_total),
        "n_users_ge2obs": int(len(r2_arr)),
        "n_users_zero_var": int(n_zero_var),
        "median_user_r2": float(np.median(r2_arr)),
        "q25_user_r2": float(np.quantile(r2_arr, 0.25)),
        "q75_user_r2": float(np.quantile(r2_arr, 0.75)),
        "mean_user_r2": float(np.mean(r2_arr)),
        "median_user_pearson_r": float(np.median(r_arr)) if len(r_arr) else float("nan"),
        "mean_user_pearson_r": float(np.mean(r_arr)) if len(r_arr) else float("nan"),
    }
    return summary, per_user_records
