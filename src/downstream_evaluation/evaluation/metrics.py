"""Evaluation metrics for downstream tasks.

One primary metric per task type — the headline reporting metric and nothing else:

  - binary      → AUPRC
  - multiclass  → accuracy
  - ordinal     → Spearman's rho
  - regression  → Pearson r

Each returns the point estimate plus its bootstrap standard error (suffix
``_se``): the bootstrap resamples the test set 1000 times with replacement and
reports ``std(metric_values)``. Secondary metrics (AUROC, QWK, MAE, F1, MSE, R²)
are intentionally not computed — they are not reported and only add cost.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, average_precision_score, r2_score

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
    """AUPRC (primary binary metric) + bootstrap SE.

    Args:
        y_true: Ground truth binary labels (N,).
        y_prob: Predicted probabilities/scores for the positive class (N,).
    """
    if len(np.unique(y_true)) < 2 or np.isnan(y_prob).any() or np.isinf(y_prob).any():
        return {"auprc": float("nan"), "auprc_se": float("nan")}
    y_prob = np.clip(y_prob, 1e-10, 1.0 - 1e-10)

    def _auprc(yt, yp):
        if len(np.unique(yt)) < 2:
            return np.nan
        return average_precision_score(yt, yp)

    return {
        "auprc": float(average_precision_score(y_true, y_prob)),
        "auprc_se": bootstrap_se(y_true, y_prob, _auprc),
    }


def compute_multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Accuracy (primary multiclass metric) + bootstrap SE."""

    def _acc(yt, yp):
        return accuracy_score(yt, yp)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "accuracy_se": bootstrap_se(y_true, y_pred, _acc),
    }


def compute_ordinal_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Spearman's rho (primary ordinal metric) + bootstrap SE."""

    def _spearman(yt, yp):
        if len(np.unique(yt)) < 2 or len(np.unique(yp)) < 2:
            return np.nan
        return spearmanr(yt, yp)[0]

    spearman_val, _ = spearmanr(y_true, y_pred)
    return {
        "spearman_r": float(spearman_val),
        "spearman_r_se": bootstrap_se(y_true, y_pred, _spearman),
    }


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Pearson r (primary regression metric) + bootstrap SE."""
    # Epsilon threshold for "zero variance" — guards float-precision residuals when
    # y_true/y_pred are effectively constant after centering (Pearson r → NaN).
    _VAR_EPS = 1e-12

    def _pearson(yt, yp):
        if np.var(yt) < _VAR_EPS or np.var(yp) < _VAR_EPS:
            return np.nan
        return pearsonr(yt, yp)[0]

    if np.var(y_pred) < _VAR_EPS or np.var(y_true) < _VAR_EPS:
        pearson_r = float("nan")
    else:
        pearson_r = float(pearsonr(y_true, y_pred)[0])
    return {
        "pearson_r": pearson_r,
        "pearson_r_se": bootstrap_se(y_true, y_pred, _pearson),
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
