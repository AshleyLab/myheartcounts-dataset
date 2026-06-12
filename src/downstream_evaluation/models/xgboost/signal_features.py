"""Signal-processing feature extraction for the XGBoost pipeline.

Provides three extraction functions:
1. Statistical features (time domain) — pure Polars expressions (10 per metric)
2. ARIMA features (autoregressive model) — statsmodels
3. Cross-correlation features — statsmodels

FFT features and redundant stat features (MIN, MAX, MEAN) are omitted: an ablation
showed zero impact on AUROC across 6 clinical disease targets, and MIN/MAX/MEAN are
fully redundant with PEAK/P2P and the mean-based stat features.

These operate on user-level daily time series (the sequence of daily values from the
timeseries pipeline's daily checkpoints, ~30-365 points per user).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import polars as pl

# ── Metrics to process ────────────────────────────────────────────────────────
# Daily columns from the timeseries pipeline's checkpoints that serve as time series
SIGNAL_PROCESSING_METRICS = [
    "daily_watch_steps_sum",
    "daily_iphone_steps_sum",
    "daily_iphone_flights_sum",
    "daily_watch_distance_sum",
    "daily_iphone_distance_sum",
    "daily_watch_hr_median",
    "daily_watch_hr_p5",
    "daily_watch_hr_p95",
    "daily_watch_energy_sum",
    "daily_sleep_minutes",
    "daily_inbed_minutes",
    "daily_watch_active_minutes",
    "daily_iphone_active_minutes",
    "daily_workout_minutes",
]

# Subset for cross-correlation (avoid redundant iPhone/Watch pairs)
CC_METRICS = [
    "daily_watch_steps_sum",
    "daily_watch_hr_median",
    "daily_watch_hr_p5",
    "daily_watch_energy_sum",
    "daily_sleep_minutes",
    "daily_watch_active_minutes",
    "daily_workout_minutes",
    "daily_inbed_minutes",
]

MIN_SERIES_LENGTH = 10
MIN_SERIES_LENGTH_ARIMA = 15

# Canonical ARIMA(2,1,0) parameter names — always output these columns.
# ARIMA(2,1,0) with differencing does not estimate a constant, so "const"
# is excluded (always null). statsmodels returns ar.L1, ar.L2, sigma2.
ARIMA_PARAM_NAMES = ["ar.L1", "ar.L2", "sigma2"]

# ── Helpers ───────────────────────────────────────────────────────────────────


def _short_name(metric: str) -> str:
    """Strip 'daily_' prefix for compact feature naming."""
    return metric.removeprefix("daily_")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Statistical features — pure Polars expressions
# ══════════════════════════════════════════════════════════════════════════════


def build_stat_expressions(metrics: list[str] | None = None) -> list[pl.Expr]:
    """Build Polars aggregation expressions for 10 statistical features per metric.

    Returns expressions suitable for use inside group_by("user_id").agg(...).
    Each expression produces one column named sp_stat_{FEATURE}_{metric_short}.

    Features:
        N, RMS, VAR, STD, POWER, PEAK, P2P,
        CREST_FACTOR, SKEW, KURTOSIS

    MIN, MAX, MEAN were removed (redundant with PEAK/P2P and mean-based stats;
    ablation showed zero AUROC impact across 6 clinical targets).
    """
    if metrics is None:
        metrics = SIGNAL_PROCESSING_METRICS

    exprs: list[pl.Expr] = []

    for m in metrics:
        s = _short_name(m)
        # Polars skips null in aggregations but propagates NaN.
        # fill_nan(None) converts NaN → null so aggregations ignore them.
        col = pl.col(m).fill_nan(None)

        exprs.extend(
            [
                col.drop_nulls().count().alias(f"sp_stat_N_{s}"),
                # RMS = sqrt(mean(x^2))
                col.pow(2).mean().sqrt().alias(f"sp_stat_RMS_{s}"),
                # VAR (population)
                col.var(ddof=0).alias(f"sp_stat_VAR_{s}"),
                # STD (population)
                col.std(ddof=0).alias(f"sp_stat_STD_{s}"),
                # POWER = mean(x^2)
                col.pow(2).mean().alias(f"sp_stat_POWER_{s}"),
                # PEAK = max(|x|)
                col.abs().max().alias(f"sp_stat_PEAK_{s}"),
                # P2P = max - min
                (col.max() - col.min()).alias(f"sp_stat_P2P_{s}"),
                # CREST_FACTOR = peak / rms, guard div-by-zero
                pl.when(col.pow(2).mean().sqrt() > 0)
                .then(col.abs().max() / col.pow(2).mean().sqrt())
                .otherwise(None)
                .alias(f"sp_stat_CREST_FACTOR_{s}"),
                # SKEW = E[(x - mean)^3] / std^3, guard zero std
                pl.when(col.std(ddof=0) > 0)
                .then(((col - col.mean()).pow(3)).mean() / col.std(ddof=0).pow(3))
                .otherwise(None)
                .alias(f"sp_stat_SKEW_{s}"),
                # KURTOSIS = E[(x - mean)^4] / var^2 - 3, guard zero var
                pl.when(col.var(ddof=0) > 0)
                .then(((col - col.mean()).pow(4)).mean() / col.var(ddof=0).pow(2) - 3)
                .otherwise(None)
                .alias(f"sp_stat_KURTOSIS_{s}"),
            ]
        )

    return exprs


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: ARIMA + CC — Python functions (numpy/statsmodels)
# ══════════════════════════════════════════════════════════════════════════════


def arima_feature_extraction(
    x: np.ndarray, order: tuple[int, int, int] = (2, 1, 0)
) -> dict[str, float | None]:
    """ARIMA model coefficients as features.

    - Fits ARIMA(2,1,0) to the series
    - Returns canonical parameter dict with keys from ARIMA_PARAM_NAMES

    Always returns all canonical params; missing ones are None.
    Returns all-None dict on failure.
    """
    result = {p: None for p in ARIMA_PARAM_NAMES}
    try:
        from statsmodels.tsa.arima.model import ARIMA

        model = ARIMA(x, order=order)
        fit = model.fit(method_kwargs={"warn_convergence": False})
        for name, val in zip(fit.param_names, fit.params):
            if name in result:
                result[name] = float(val)
    except Exception:
        pass
    return result


def cc_feature_extraction(series_dict: dict[str, np.ndarray], lags: int = 3) -> dict[str, float]:
    """Cross-correlation features for all pairs of metrics.

    - For each pair (a, b) in combinations(metrics, 2): compute ccf at lags 0..2
    - Returns cc_lag{N}_{a}__{b} keys

    Args:
        series_dict: metric_short_name → numpy array (all same length, aligned)
        lags: number of lags to compute (default 3 → lags 0, 1, 2)

    Returns dict of cross-correlation values.
    """
    import statsmodels.api as sm

    result = {}
    metric_names = sorted(series_dict.keys())
    for a, b in combinations(metric_names, 2):
        try:
            cc = sm.tsa.stattools.ccf(series_dict[a], series_dict[b], adjusted=False)
            for lag_i in range(lags):
                result[f"cc_lag{lag_i}_{a}__{b}"] = float(cc[lag_i])
        except Exception:
            for lag_i in range(lags):
                result[f"cc_lag{lag_i}_{a}__{b}"] = float("nan")
    return result


def extract_arima_cc_for_user(user_df: pl.DataFrame) -> pl.DataFrame:
    """Extract ARIMA and CC features for a single user.

    Called via group_by("user_id").map_groups(). Receives one user's daily data
    sorted by date. For each metric: drop nulls, apply ARIMA (if len >= 15).
    Align CC metrics to common valid dates, apply CC.

    Returns single-row DataFrame with user_id + all ARIMA/CC columns.
    """
    user_id = user_df["user_id"][0]
    row: dict[str, object] = {"user_id": user_id}

    # Per-metric ARIMA
    for m in SIGNAL_PROCESSING_METRICS:
        s = _short_name(m)

        if m not in user_df.columns:
            for param in ARIMA_PARAM_NAMES:
                row[f"sp_arima_{param}_{s}"] = None
            continue

        vals = user_df[m].drop_nulls().drop_nans().to_numpy()

        # ARIMA — always returns all ARIMA_PARAM_NAMES keys
        if len(vals) >= MIN_SERIES_LENGTH_ARIMA:
            arima_feats = arima_feature_extraction(vals)
            for param_name in ARIMA_PARAM_NAMES:
                row[f"sp_arima_{param_name}_{s}"] = arima_feats[param_name]
        else:
            for param in ARIMA_PARAM_NAMES:
                row[f"sp_arima_{param}_{s}"] = None

    # Cross-correlation: align CC_METRICS to common valid dates
    cc_metric_shorts = [_short_name(m) for m in CC_METRICS]
    available_cc = [m for m in CC_METRICS if m in user_df.columns]

    if len(available_cc) >= 2:
        # Build mask of rows where all CC metrics are non-null
        mask = pl.lit(True)
        for m in available_cc:
            mask = mask & pl.col(m).is_not_null() & pl.col(m).is_not_nan()
        aligned = user_df.filter(mask)

        if len(aligned) >= MIN_SERIES_LENGTH:
            series_dict = {}
            for m in available_cc:
                series_dict[_short_name(m)] = aligned[m].to_numpy()
            cc_feats = cc_feature_extraction(series_dict, lags=3)
            row.update(cc_feats)
        else:
            # Too short — fill NaN for all CC features
            for a, b in combinations(sorted(cc_metric_shorts), 2):
                for lag_i in range(3):
                    row[f"cc_lag{lag_i}_{a}__{b}"] = None
    else:
        for a, b in combinations(sorted(cc_metric_shorts), 2):
            for lag_i in range(3):
                row[f"cc_lag{lag_i}_{a}__{b}"] = None

    df = pl.DataFrame([row])
    # Ensure all feature columns are Float64, not Null.  When a user has
    # insufficient data in a narrow date window every value is None and Polars
    # infers the column as Null type.  map_groups then fails with SchemaError
    # when concatenating Null-typed columns with Float64 from other groups.
    df = df.cast({c: pl.Float64 for c in df.columns if c != "user_id" and df[c].dtype == pl.Null})
    return df
