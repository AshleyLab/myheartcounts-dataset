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
``_compute_all_ranks`` for rank). The identity draw therefore reproduces the
published point estimates — see ``tests/test_forecasting_bootstrap_skill_rank.py``.
"""

from __future__ import annotations

import hashlib
import logging
from statistics import NormalDist

import numpy as np
import pandas as pd

from forecasting_evaluation.metrics import metric_spec as _spec
from forecasting_evaluation.metrics.grouped_metric_rank_summary import (
    _build_binary_user_rows,
    _build_continuous_user_rows,
    _compute_all_ranks,
)
from forecasting_evaluation.metrics.per_user_errors import to_error_df, to_rank_user_df
from forecasting_evaluation.metrics.skill_score_summary import (
    _build_error_table,
    _build_model_summary,
    _compute_long_skill_scores,
)

logger = logging.getLogger(__name__)

# Headline scopes that receive a point estimate + BCa interval (the rest keep the
# percentile CI only). Derived from the 4 sensor categories so they track
# metric_spec. Skill columns carry the ``_score`` suffix; rank scopes do not.
_CATEGORY_NAMES: tuple[str, ...] = tuple(name for name, _ in _spec.CATEGORY_SCOPES)
SKILL_HEADLINE_SCOPES: frozenset[str] = frozenset(
    {"overall_score", *(f"{name}_score" for name in _CATEGORY_NAMES)}
)
RANK_HEADLINE_SCOPES: frozenset[str] = frozenset({"overall", *_CATEGORY_NAMES})


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


# --------------------------------------------------------------------------
# BCa (bias-corrected & accelerated) interval — point-anchored alternative to the
# percentile CI for skewed / downward-biased statistics (the fairness disparity
# ratio). Re-anchors the interval near the point estimate and corrects bias + skew
# (second-order accurate). Uses ``statistics.NormalDist`` for Φ / Φ⁻¹ (no scipy).
# --------------------------------------------------------------------------

_NORM = NormalDist()


def _jackknife_acceleration(jack: np.ndarray) -> float:
    """BCa acceleration from leave-one-out jackknife values (nan-aware).

    ``a = Σ d³ / (6 · (Σ d²)^{3/2})`` with ``d = mean_i(θ₍ᵢ₎) − θ₍ᵢ₎``. Returns
    ``0.0`` when fewer than two finite values are present or ``Σ d² == 0``.
    """
    arr = np.asarray(jack, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size < 2:
        return 0.0
    d = finite.mean() - finite
    s2 = float(np.sum(d**2))
    if s2 == 0.0:
        return 0.0
    return float(np.sum(d**3)) / (6.0 * s2**1.5)


def _bca_interval(
    draws: np.ndarray, point: float, jack: np.ndarray, ci_level: float
) -> tuple[float, float]:
    """Bias-corrected & accelerated CI for one statistic.

    Args:
        draws: bootstrap draws ``θ*_b`` (NaN-dropped).
        point: the deterministic point estimate ``θ̂`` (the reported value).
        jack: leave-one-user-out jackknife values ``θ₍ᵢ₎`` (NaN-aware).
        ci_level: e.g. 0.95 -> a 2.5/97.5 percentile-equivalent interval.

    Guards (fall back to the plain percentile interval): empty/non-finite point,
    non-finite ``z0``/``a``, or a zero BCa denominator ``1 − a(z0 + z_q)``. All
    draws equal -> ``[point, point]``. When ``z0 = a = 0`` the adjusted percentiles
    reduce to ``α/2`` and ``1 − α/2``, i.e. the percentile interval exactly.
    """
    arr = np.asarray(draws, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    n = int(finite.size)
    alpha = 1.0 - ci_level

    def _percentile() -> tuple[float, float]:
        if n == 0:
            return float("nan"), float("nan")
        return (
            float(np.percentile(finite, 100.0 * (alpha / 2.0))),
            float(np.percentile(finite, 100.0 * (1.0 - alpha / 2.0))),
        )

    if n == 0 or not np.isfinite(point):
        return _percentile()
    if np.ptp(finite) == 0.0:
        return float(point), float(point)

    # Bias correction z0 from the fraction of draws below the point (clipped so
    # an extreme point still yields a finite z0).
    prop = float(np.count_nonzero(finite < point)) / n
    prop = min(max(prop, 0.5 / n), 1.0 - 0.5 / n)
    z0 = _NORM.inv_cdf(prop)
    a = _jackknife_acceleration(jack)
    if not (np.isfinite(z0) and np.isfinite(a)):
        return _percentile()

    out: list[float] = []
    for z_q in (_NORM.inv_cdf(alpha / 2.0), _NORM.inv_cdf(1.0 - alpha / 2.0)):
        denom = 1.0 - a * (z0 + z_q)
        if denom == 0.0 or not np.isfinite(denom):
            return _percentile()
        adj = z0 + (z0 + z_q) / denom
        if not np.isfinite(adj):
            return _percentile()
        frac = min(max(_NORM.cdf(adj), 0.0), 1.0)
        out.append(float(np.percentile(finite, 100.0 * frac)))
    return out[0], out[1]


def _draws_by_key(records: list[dict], key_cols: list[str]) -> dict[tuple, np.ndarray]:
    """Group per-draw value records into ``{key tuple: draws array}``."""
    out: dict[tuple, list[float]] = {}
    for rec in records:
        out.setdefault(tuple(rec[c] for c in key_cols), []).append(rec["value"])
    return {key: np.asarray(values, dtype=np.float64) for key, values in out.items()}


def _pad_jackknife_maps(per_user_maps: list[dict[tuple, float]]) -> dict[tuple, np.ndarray]:
    """Align a list of per-user ``{key: value}`` maps into ``{key: array}``.

    The k-th array entry is user k's leave-one-out value, NaN where that user's
    recompute lacked the key (so every key spans all users, NaN-aware downstream).
    """
    keys: set[tuple] = set()
    for m in per_user_maps:
        keys |= m.keys()
    return {
        key: np.array([m.get(key, np.nan) for m in per_user_maps], dtype=np.float64) for key in keys
    }


def _augment_with_bca(
    summary_df: pd.DataFrame,
    *,
    draws_by_key: dict[tuple, np.ndarray],
    point_by_key: dict[tuple, float],
    jack_by_key: dict[tuple, np.ndarray],
    scopes: frozenset[str],
    ci_level: float,
    key_cols: list[str],
) -> pd.DataFrame:
    """Add ``point``, ``bca_lo``, ``bca_hi`` columns to a ``_summary_table`` output.

    ``point`` is filled for every row (from ``point_by_key``); ``bca_lo``/``bca_hi``
    only for rows whose ``scope`` is in ``scopes`` (NaN elsewhere). The percentile
    columns are left untouched.
    """
    out = summary_df.copy()
    if out.empty:
        for col in ("point", "bca_lo", "bca_hi"):
            out[col] = pd.Series(dtype=np.float64)
        return out

    points, los, his = [], [], []
    for _, row in out.iterrows():
        key = tuple(row[c] for c in key_cols)
        point = point_by_key.get(key, float("nan"))
        point = float(point) if point is not None and np.isfinite(point) else float("nan")
        points.append(point)
        if row["scope"] in scopes and key in draws_by_key:
            lo, hi = _bca_interval(
                draws_by_key[key], point, jack_by_key.get(key, np.empty(0)), ci_level
            )
            los.append(lo)
            his.append(hi)
        else:
            los.append(float("nan"))
            his.append(float("nan"))
    out["point"] = points
    out["bca_lo"] = los
    out["bca_hi"] = his
    return out


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
# Leave-one-user-out jackknife of the deterministic point flow (BCa acceleration)
# --------------------------------------------------------------------------


def _jackknife_skill_points(
    error_df: pd.DataFrame,
    models: dict[str, dict[str, str]],
    *,
    baseline_model: str,
    clip_lower: float,
    clip_upper: float,
    min_pairs: int,
    scopes: frozenset[str],
) -> dict[tuple, np.ndarray]:
    """Leave-one-user-out jackknife of the skill-score headline scopes.

    Re-runs the point flow (``_compute_long_skill_scores`` + ``_build_model_summary``)
    on ``error_df`` minus each user in turn, returning ``{(model, scope): array}``
    over the dropped users (NaN where a scope is absent for that recompute).
    """
    users = sorted(set(error_df["unit_id"].astype(str)))
    unit_arr = error_df["unit_id"].astype(str).to_numpy()
    per_user_maps: list[dict[tuple, float]] = []
    for user in users:
        summ = _build_model_summary(
            long_df=_compute_long_skill_scores(
                error_df=error_df.loc[unit_arr != user],
                models=models,
                baseline_model=baseline_model,
                clip_lower=clip_lower,
                clip_upper=clip_upper,
                min_pairs=min_pairs,
            ),
            models=models,
            baseline_model=baseline_model,
        )
        cols = [c for c in scopes if c in summ.columns]
        per_user_maps.append(
            {
                (row["model"], col): float(row[col])
                for _, row in summ.iterrows()
                for col in cols
                if pd.notna(row[col])
            }
        )
    return _pad_jackknife_maps(per_user_maps)


def _jackknife_rank_points(
    rank_user_df: pd.DataFrame, *, scopes: frozenset[str]
) -> dict[tuple, np.ndarray]:
    """Leave-one-user-out jackknife of the mean-rank headline scopes.

    Re-runs ``_compute_all_ranks`` on ``rank_user_df`` minus each user, returning
    ``{(model, scope, metric): array}`` over the dropped users.
    """
    users = sorted(set(rank_user_df["user_id"].astype(str)))
    uid_arr = rank_user_df["user_id"].astype(str).to_numpy()
    per_user_maps: list[dict[tuple, float]] = []
    for user in users:
        ranks = _compute_all_ranks(user_metric_df=rank_user_df.loc[uid_arr != user])
        per_user_maps.append(
            {
                (row["model"], row["scope"], row["metric"]): float(row["rank"])
                for _, row in ranks.iterrows()
                if row["scope"] in scopes and pd.notna(row["rank"])
            }
        )
    return _pad_jackknife_maps(per_user_maps)


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
    bca_skill_rank: bool = False,
    per_user_metrics: pd.DataFrame | None = None,
    return_draws: bool = False,
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
        bca_skill_rank: when True, also add ``point``, ``bca_lo``, ``bca_hi`` columns
            (point estimate + bias-corrected & accelerated CI) for the headline
            scopes — skill ``overall_score`` + the 4 ``<category>_score``; rank
            ``overall`` + the 4 categories. Off by default (skill/rank are
            near-unbiased: bootstrap mean ≈ point, so the percentile CI suffices).
        per_user_metrics: optional canonical substrate frame
            (:func:`per_user_errors.build_per_user_metrics`). When given, the
            per-user error/rank tables are reconstructed from it (micro/user only)
            instead of re-scanning the metric trees.
        return_draws: when True, also include the raw per-draw long frames
            ``skill_draws`` (model, scope, draw, value) and ``rank_draws``
            (model, scope, metric, draw, value) in the result — the bootstrap
            reference shipped to the leaderboard dataset.

    Returns:
        ``{"skill_scores": df, "avg_rankings": df}`` where each row carries
        ``mean, se, ci_lo, ci_hi, n_boot`` (plus ``point, bca_lo, bca_hi`` when
        ``bca_skill_rank``). ``skill_scores`` is keyed by ``(model, scope)``
        (scope = ``channel_<i>_score`` / ``sleep_score`` / ``workout_score``);
        ``avg_rankings`` by ``(model, scope, metric)``.
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
    # When the caller passes the canonical substrate, reconstruct both tables from
    # it (micro/user only) instead of re-scanning the metric trees.
    if per_user_metrics is not None:
        if within_user_aggregation != "micro":
            raise ValueError(
                "per_user_metrics substrate supports only within_user_aggregation="
                f"'micro'; got {within_user_aggregation!r}."
            )
        error_df = to_error_df(per_user_metrics, user_col="unit_id")
        rank_user_df = to_rank_user_df(per_user_metrics, binary_groups=binary_groups)
    else:
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
        rank_user_df = (
            pd.concat(rank_frames, ignore_index=True) if rank_frames else pd.DataFrame()
        )

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
            ranks_b = _compute_all_ranks(user_metric_df=rank_b_input)
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

    skill_summary = _summary_table(skill_records, ["model", "scope"], ci_level)
    rank_summary = _summary_table(rank_records, ["model", "scope", "metric"], ci_level)

    if bca_skill_rank:
        if not error_df.empty:
            point_summary = _build_model_summary(
                long_df=_compute_long_skill_scores(
                    error_df=error_df,
                    models=models,
                    baseline_model=baseline_model,
                    clip_lower=clip_lower,
                    clip_upper=clip_upper,
                    min_pairs=min_pairs,
                ),
                models=models,
                baseline_model=baseline_model,
            )
            point_by_key = {
                (row["model"], col): float(row[col])
                for _, row in point_summary.iterrows()
                for col in point_summary.columns
                if col.endswith("_score") and pd.notna(row[col])
            }
            skill_summary = _augment_with_bca(
                skill_summary,
                draws_by_key=_draws_by_key(skill_records, ["model", "scope"]),
                point_by_key=point_by_key,
                jack_by_key=_jackknife_skill_points(
                    error_df,
                    models,
                    baseline_model=baseline_model,
                    clip_lower=clip_lower,
                    clip_upper=clip_upper,
                    min_pairs=min_pairs,
                    scopes=SKILL_HEADLINE_SCOPES,
                ),
                scopes=SKILL_HEADLINE_SCOPES,
                ci_level=ci_level,
                key_cols=["model", "scope"],
            )
        if not rank_user_df.empty:
            point_ranks = _compute_all_ranks(user_metric_df=rank_user_df)
            rank_point_by_key = {
                (row["model"], row["scope"], row["metric"]): float(row["rank"])
                for _, row in point_ranks.iterrows()
                if pd.notna(row["rank"])
            }
            rank_summary = _augment_with_bca(
                rank_summary,
                draws_by_key=_draws_by_key(rank_records, ["model", "scope", "metric"]),
                point_by_key=rank_point_by_key,
                jack_by_key=_jackknife_rank_points(rank_user_df, scopes=RANK_HEADLINE_SCOPES),
                scopes=RANK_HEADLINE_SCOPES,
                ci_level=ci_level,
                key_cols=["model", "scope", "metric"],
            )

    result = {"skill_scores": skill_summary, "avg_rankings": rank_summary}
    if return_draws:
        result["skill_draws"] = pd.DataFrame(
            skill_records, columns=["model", "scope", "draw", "value"]
        )
        result["rank_draws"] = pd.DataFrame(
            rank_records, columns=["model", "scope", "metric", "draw", "value"]
        )
    return result


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
