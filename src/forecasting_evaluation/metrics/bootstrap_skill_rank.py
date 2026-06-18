"""Paired participant-level (user) bootstrap for forecasting skill & rank.

Mirrors the imputation cross-method bootstrap
(``imputation_evaluation/evaluation/bootstrap_skill_rank.py``): one shared
user-resample matrix is applied to **every model + baseline jointly**, the
existing point-flow aggregators are re-run per draw, and draws are reduced to
``mean / se / percentile-CI``. The cluster unit is the **user** (not the
forecast window), so between-user variance is captured.

Tier 1 design: the per-user error tables are built **once** (the only disk IO),
then each bootstrap draw resamples users with replacement, replica-expands the
tables so pandas ``pivot``/``groupby`` keep duplicate users (correct cluster
weighting), and re-runs the *exact* point-flow pure functions
(``_compute_long_skill_scores`` / ``_build_model_summary`` for skill,
``_compute_mean_ranks`` for rank). The identity draw therefore reproduces the
published point estimates — see ``tests/test_forecasting_bootstrap_skill_rank.py``.
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np
import pandas as pd

from forecasting_evaluation.metrics.grouped_metric_rank_summary import (
    _build_binary_user_rows,
    _build_continuous_user_rows,
    _compute_mean_ranks,
    _compute_overall_category_balanced_ranks,
)
from forecasting_evaluation.metrics.skill_score_summary import (
    _build_error_table,
    _build_model_summary,
    _compute_long_skill_scores,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Small helpers — copied verbatim from
# imputation_evaluation.evaluation.bootstrap so the two tracks stay decoupled
# (they are tiny and battle-tested; a cross-track import would couple the
# public forecasting package to the imputation internals).
# --------------------------------------------------------------------------


def _bootstrap_indices(n_users: int, n_boot: int, seed: int) -> np.ndarray:
    """Draw an ``(n_boot, n_users)`` matrix of user indices sampled w/ replacement."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_users, size=(n_boot, n_users), dtype=np.int64)


def _summarize(values: np.ndarray, ci_level: float) -> dict:
    """Reduce a ``(B,)`` array to ``{mean, se, ci_lo, ci_hi, n_boot}``; NaN-dropped."""
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    n_valid = int(finite.size)
    if n_valid == 0:
        return {
            "mean": float("nan"),
            "se": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "n_boot": 0,
        }
    alpha = 1.0 - ci_level
    return {
        "mean": float(np.mean(finite)),
        "se": float(np.std(finite, ddof=1)) if n_valid > 1 else 0.0,
        "ci_lo": float(np.percentile(finite, 100.0 * (alpha / 2.0))),
        "ci_hi": float(np.percentile(finite, 100.0 * (1.0 - alpha / 2.0))),
        "n_boot": n_valid,
    }


def _seed_for(seed: int, tag: str) -> int:
    """Stable derived seed via SHA-256 (Python's ``hash()`` is salted for str)."""
    digest = hashlib.sha256(f"{seed}|{tag}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


# --------------------------------------------------------------------------
# Replica expansion — the multiplicity-preserving resample
# --------------------------------------------------------------------------


def _draw_replica_frame(users: list[str], idx_vec: np.ndarray) -> pd.DataFrame:
    """One drawn multiset -> frame of ``(user_id, _unit)`` with per-user replica ids.

    A user drawn ``m`` times yields ``m`` rows with distinct ``_unit`` suffixes
    (``<user>#r0``..``<user>#r{m-1}``) so downstream ``pivot``/``groupby`` keep
    all ``m`` copies — i.e. weight that user's contribution by its bootstrap
    multiplicity. The same frame is applied to every model's rows so the k-th
    replica of a user lines up across model and baseline.
    """
    drawn = pd.Series(np.asarray(users, dtype=object)[idx_vec], dtype="object").astype(str)
    drawn = drawn.reset_index(drop=True)
    rep = drawn.groupby(drawn).cumcount()
    unit = drawn + "#r" + rep.astype(str)
    return pd.DataFrame({"user_id": drawn.to_numpy(), "_unit": unit.to_numpy()})


def _resample(df: pd.DataFrame, replicas: pd.DataFrame, user_col: str) -> pd.DataFrame:
    """Inner-join ``df`` to the drawn replica frame, rewriting the user key to ``_unit``.

    ``user_col`` is the column in ``df`` that holds the user id (``unit_id`` for
    the skill table, ``user_id`` for the rank table). The rewritten key keeps
    duplicate draws distinct.
    """
    out = replicas.merge(df, left_on="user_id", right_on=user_col, how="inner")
    out[user_col] = out["_unit"]
    return out


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def bootstrap_skill_rank(
    *,
    models: dict[str, dict[str, str]],
    baseline_model: str,
    continuous_metrics: list[str],
    binary_metrics: list[str],
    continuous_channel_indices: tuple[int, ...],
    binary_channel_indices: tuple[int, ...],
    binary_groups: list[tuple[str, tuple[int, ...]]],
    n_boot: int = 1000,
    seed: int = 42,
    ci_level: float = 0.95,
    clip_lower: float = 0.01,
    clip_upper: float = 100.0,
    min_pairs: int = 1,
    within_user_aggregation: str = "micro",
) -> dict[str, pd.DataFrame]:
    """Paired user-bootstrap CIs for forecasting skill scores and mean ranks.

    Args:
        models: ``{name: {"path": metrics_dir, "display_name": ...}}``.
        baseline_model: key in ``models`` used as the skill-score denominator.
        continuous_metrics: metric keys scored on continuous channels (e.g. mae).
        binary_metrics: metric keys scored on binary channels (e.g. auprc).
        continuous_channel_indices: continuous channels to score.
        binary_channel_indices: binary channels to score.
        binary_groups: ``[(group_name, channel_indices), ...]`` for rank scopes
            (e.g. ``[("sleep", (7, 8)), ("workout", tuple(range(9, 19)))]``).
        n_boot: number of bootstrap draws.
        seed: master RNG seed (a per-run seed is derived deterministically).
        ci_level: percentile-CI level (0.95 -> 2.5/97.5).
        clip_lower: lower clip on the per-task error ratio (skill aggregator).
        clip_upper: upper clip on the per-task error ratio (skill aggregator).
        min_pairs: minimum paired units required to score a task.
        within_user_aggregation: 'micro' (default) weights each window by its finite
            horizon-cell count when building per-user errors; 'macro' averages
            per-window means unweighted (legacy). Shared with the point flow.

    Returns:
        ``{"skill_scores": df, "avg_rankings": df}`` where each row carries
        ``mean, se, ci_lo, ci_hi, n_boot``. ``skill_scores`` is keyed by
        ``(model, scope)`` (scope = ``channel_<i>_score`` / ``sleep_score`` /
        ``workout_score``); ``avg_rankings`` by ``(model, scope, metric)``.
    """
    metric_groups = {
        "continuous": {
            "metrics": [m.strip().lower() for m in continuous_metrics if m.strip()],
            "channel_indices": continuous_channel_indices,
        },
        "binary": {
            "metrics": [m.strip().lower() for m in binary_metrics if m.strip()],
            "channel_indices": binary_channel_indices,
        },
    }

    # ---- Phase 0: build per-user tables ONCE (the only disk IO) ----
    error_df = _build_error_table(
        models=models,
        metric_groups=metric_groups,
        aggregation_unit="user",
        within_user_aggregation=within_user_aggregation,
    )
    cont_user = _build_continuous_user_rows(
        models=models,
        metrics=[m.strip().lower() for m in continuous_metrics if m.strip()],
        channel_indices=continuous_channel_indices,
        within_user_aggregation=within_user_aggregation,
    )
    bin_user = _build_binary_user_rows(
        models=models,
        metrics=[m.strip().lower() for m in binary_metrics if m.strip()],
        groups=binary_groups,
        within_user_aggregation=within_user_aggregation,
    )
    rank_frames = [f for f in (cont_user, bin_user) if not f.empty]
    rank_user_df = pd.concat(rank_frames, ignore_index=True) if rank_frames else pd.DataFrame()

    if error_df.empty and rank_user_df.empty:
        logger.warning("Bootstrap: no error/rank rows discovered; returning empty tables.")
        return {"skill_scores": pd.DataFrame(), "avg_rankings": pd.DataFrame()}

    # ---- Phase 1: union users + one shared resample matrix ----
    user_set: set[str] = set()
    if not error_df.empty:
        user_set |= set(error_df["unit_id"].astype(str))
    if not rank_user_df.empty:
        user_set |= set(rank_user_df["user_id"].astype(str))
    users = sorted(user_set)
    n_users = len(users)
    if n_users == 0:
        return {"skill_scores": pd.DataFrame(), "avg_rankings": pd.DataFrame()}
    idx_b = _bootstrap_indices(n_users, n_boot, _seed_for(seed, "forecasting"))
    logger.info("Forecasting bootstrap: U=%d users, B=%d, seed=%d", n_users, n_boot, seed)

    # ---- Phase 2: per-draw recompute via the existing pure aggregators ----
    skill_records: list[dict] = []
    rank_records: list[dict] = []
    for b in range(n_boot):
        replicas = _draw_replica_frame(users, idx_b[b])

        if not error_df.empty:
            err_b = _resample(error_df, replicas, "unit_id")
            long_b = _compute_long_skill_scores(
                error_df=err_b,
                models=models,
                baseline_model=baseline_model,
                clip_lower=clip_lower,
                clip_upper=clip_upper,
                min_pairs=min_pairs,
            )
            summ_b = _build_model_summary(
                long_df=long_b, models=models, baseline_model=baseline_model
            )
            score_cols = [c for c in summ_b.columns if c.endswith("_score")]
            for _, row in summ_b.iterrows():
                for col in score_cols:
                    skill_records.append(
                        {
                            "model": row["model"],
                            "scope": col,
                            "draw": b,
                            "value": float(row[col]),
                        }
                    )

        if not rank_user_df.empty:
            rank_b_input = _resample(rank_user_df, replicas, "user_id")
            ranks_b = _compute_mean_ranks(user_metric_df=rank_b_input)
            overall_b = _compute_overall_category_balanced_ranks(user_metric_df=rank_b_input)
            if not overall_b.empty:
                ranks_b = pd.concat([ranks_b, overall_b], ignore_index=True)
            for _, row in ranks_b.iterrows():
                rank_records.append(
                    {
                        "model": row["model"],
                        "scope": row["scope"],
                        "metric": row["metric"],
                        "draw": b,
                        "value": float(row["rank"]),
                    }
                )

    return {
        "skill_scores": _summary_table(skill_records, ["model", "scope"], ci_level),
        "avg_rankings": _summary_table(rank_records, ["model", "scope", "metric"], ci_level),
    }


def _summary_table(records: list[dict], key_cols: list[str], ci_level: float) -> pd.DataFrame:
    """Reduce per-draw value records to one summarised row per key tuple."""
    out_cols = key_cols + ["mean", "se", "ci_lo", "ci_hi", "n_boot"]
    if not records:
        return pd.DataFrame(columns=out_cols)
    df = pd.DataFrame(records)
    rows: list[dict] = []
    for keys, grp in df.groupby(key_cols, sort=True):
        row = dict(zip(key_cols, keys if isinstance(keys, tuple) else (keys,)))
        row.update(_summarize(grp["value"].to_numpy(), ci_level))
        rows.append(row)
    return pd.DataFrame(rows, columns=out_cols)
