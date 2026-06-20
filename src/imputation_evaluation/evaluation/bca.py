"""BCa (bias-corrected & accelerated) bootstrap interval — stdlib only.

A point-anchored alternative to the percentile CI for skewed / downward-biased
statistics like the fairness disparity ratio ``D = max_g E − min_g E``. Re-anchors
the interval at a separately computed point estimate and corrects bias + skew
(second-order accurate). Uses ``statistics.NormalDist`` for Φ / Φ⁻¹ — no scipy.

This module is a track-siloed copy of the same helpers in
``forecasting_evaluation.metrics.bootstrap_skill_rank`` (see METRICS.md §S7).
The two tracks intentionally stay decoupled — each track owns its own
``evaluation/`` module and the helpers are tiny and battle-tested.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

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
    """Add ``point``, ``bca_lo``, ``bca_hi`` columns to a summary table.

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
