"""Pipeline Curve Analysis: FPCA + Cosinor + HROS from averaged daily curves.

Treats each user's averaged daily activity curve (1440 minutes) as a smooth
function and extracts features via three complementary approaches:

Four-phase approach:
  Phase A -- Chunked extraction of per-user averaged daily curves (checkpointed)
  Phase B -- Impute residual missingness with population minute-level means
             (non-sparse channels) or B-spline basis smoothing (sparse channels)
  Phase C -- Fit FPCA per channel and extract per-user scores
  Phase D -- Fit cosinor model to HR curve + compute HROS profile statistics

Output: one row per user with:
  - 40 FPCA score columns (10 per channel x 4 channels)
  - 7 cosinor features (MESOR, amplitude, acrophase, R^2, p-value, amplitude ratio, n_minutes)
  - 5 HROS profile features (mean, std, daytime mean, nighttime mean, day/night ratio)
  = 52 feature columns + user_id
"""

from __future__ import annotations

import math
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pyarrow as pa
import scipy.stats
from skfda.misc.operators import LinearDifferentialOperator
from skfda.misc.regularization import L2Regularization

from data.processing.hf_config import DEFAULT_VARIANCE_THRESHOLDS

if TYPE_CHECKING:
    from skfda.representation import FDataBasis
from skfda.representation import FDataIrregular
from skfda.representation.basis import BSplineBasis

from .constants import IPHONE_STEPS, WATCH_ENERGY, WATCH_HR, WATCH_STEPS

# Channels to extract FPCA features from
# sparse_zeros: True = zeros during wear are sensor gaps (exclude them)
#               False = zeros during wear are valid data (include them)
FPCA_CHANNELS = {
    WATCH_HR: {"name": "watch_hr", "scale": 60.0, "sparse_zeros": True},
    WATCH_STEPS: {"name": "watch_steps", "scale": 1.0, "sparse_zeros": False},
    WATCH_ENERGY: {"name": "watch_energy", "scale": 1.0, "sparse_zeros": False},
    IPHONE_STEPS: {"name": "iphone_steps", "scale": 1.0, "sparse_zeros": False},
}

MINUTES_PER_DAY = 1440

# Minimum valid data points to fit a cosinor model (< 200 ≈ < 14% of day
# is too sparse for reliable circadian parameter estimation).
MIN_COSINOR_DATAPOINTS = 200

# Daytime/nighttime boundaries in minutes from midnight.
DAYTIME_START_MIN = 360  # 6:00 AM
DAYTIME_END_MIN = 1320  # 10:00 PM
NIGHTTIME_LATE_MIN = 1380  # 11:00 PM


# ── Phase A helpers ──────────────────────────────────────────────────────────


def _accumulate_arrow_file(
    arrow_file: Path,
    user_sums: dict[str, dict[int, np.ndarray]],
    user_counts: dict[str, dict[int, np.ndarray]],
    max_nonwear_minutes: int | None = None,
    variance_filter: bool = True,
    cutoff_dates: dict[str, str] | None = None,
    eligible_keys: set[tuple[str, str]] | None = None,
) -> int:
    """Process one Arrow file: accumulate per-user running sums and counts.

    Accumulates per-user running sums and valid-minute counts for each FPCA
    channel. Non-wear minutes are treated as missing.

    Vectorized: extracts whole columns as 2D numpy arrays via PyArrow compute
    instead of iterating row-by-row in Python.

    Returns the number of rows (user-days) processed.
    """
    import pyarrow.compute as pc

    with pa.ipc.open_stream(arrow_file) as reader:
        table = reader.read_all()

    col_names = set(table.column_names)

    # Provider eligibility (single source): keep rows whose (user_id, date) is eligible,
    # replacing the wear/variance/cutoff re-derivation below.
    if eligible_keys is not None:
        uids = table.column("user_id").to_pylist()
        dts = [str(d)[:10] for d in table.column("date").to_pylist()]
        table = table.filter([(u, d) in eligible_keys for u, d in zip(uids, dts)])
        if table.num_rows == 0:
            return 0

    # Filter out high-nonwear rows before processing
    if eligible_keys is None and max_nonwear_minutes is not None and "total_nonwear_minutes" in col_names:
        mask = pc.less_equal(table.column("total_nonwear_minutes"), max_nonwear_minutes)
        table = table.filter(mask)
        if table.num_rows == 0:
            return 0

    # Variance filter: reject rows where a monitored channel has near-zero
    # variance (flat signal = sensor malfunction). Inline PyArrow-level
    # implementation to avoid Polars conversion in this PyArrow-native function.
    if eligible_keys is None and variance_filter and "channel_variance" in col_names:
        var_col = table.column("channel_variance")
        keep_mask = []
        for i in range(table.num_rows):
            variances = var_col[i].as_py()
            ok = True
            for ch_idx, min_var in DEFAULT_VARIANCE_THRESHOLDS.items():
                if ch_idx < len(variances):
                    v = variances[ch_idx]
                    # NaN/None means insufficient data (<2 valid values) — skip.
                    if v is None or (isinstance(v, float) and math.isnan(v)):
                        continue
                    if v < min_var:
                        ok = False
                        break
            keep_mask.append(ok)
        table = table.filter(keep_mask)
        if table.num_rows == 0:
            return 0

    # Future-data cutoff: drop rows after each user's cutoff date
    if eligible_keys is None and cutoff_dates is not None:
        user_ids_col = table.column("user_id").to_pylist()
        dates_col = [str(d) for d in table.column("date").to_pylist()]
        keep_mask = []
        for i in range(table.num_rows):
            uid = user_ids_col[i]
            d = dates_col[i]
            cutoff = cutoff_dates.get(uid)
            keep_mask.append(cutoff is None or d <= cutoff)
        table = table.filter(keep_mask)
        if table.num_rows == 0:
            return 0

    n_rows = table.num_rows
    user_ids = table.column("user_id").to_pylist()

    # Auto-detect schema: MHC-B uses "values" instead of "data"
    data_col = "data" if "data" in col_names else "values"
    data_arr = table.column(data_col).combine_chunks()

    # Build wear mask from nonwear_vector if available, else assume all-wear
    if "nonwear_vector" in col_names:
        nonwear_arr = table.column("nonwear_vector").combine_chunks()
        nw_flat = nonwear_arr.values.to_numpy(zero_copy_only=False).astype(np.float32)
        wear_mask = nw_flat.reshape(n_rows, MINUTES_PER_DAY) == 0  # True where worn
    else:
        wear_mask = np.ones((n_rows, MINUTES_PER_DAY), dtype=bool)

    # Extract only the 4 needed channels as 2D numpy arrays (no Python intermediary)
    # Cast extension types (e.g. HuggingFace Array2DExtensionType) to storage before compute
    if hasattr(data_arr.type, "storage_type"):
        data_arr = data_arr.cast(data_arr.type.storage_type)

    channel_matrices: dict[int, np.ndarray] = {}
    for ch_idx, ch_info in FPCA_CHANNELS.items():
        ch_list = pc.list_element(data_arr, ch_idx)  # ListArray<float64>
        ch_flat = ch_list.values.to_numpy(zero_copy_only=False).astype(np.float64)
        channel_matrices[ch_idx] = ch_flat.reshape(n_rows, MINUTES_PER_DAY) * ch_info["scale"]

    del table, data_arr
    # nonwear_arr only exists when nonwear_vector column was present

    # Group row indices by user_id
    uid_to_rows: dict[str, list[int]] = {}
    for i, uid in enumerate(user_ids):
        uid_to_rows.setdefault(uid, []).append(i)

    # Accumulate per user (vectorized over all of a user's days at once)
    for uid, row_indices in uid_to_rows.items():
        idx = np.array(row_indices)
        user_wear = wear_mask[idx]  # (n_days, 1440)

        if uid not in user_sums:
            user_sums[uid] = {}
            user_counts[uid] = {}

        for ch_idx, ch_info in FPCA_CHANNELS.items():
            ch_vals = channel_matrices[ch_idx][idx]  # (n_days, 1440)

            if ch_info["sparse_zeros"]:
                valid = user_wear & np.isfinite(ch_vals) & (ch_vals > 0)
            else:
                valid = user_wear & np.isfinite(ch_vals)

            if ch_idx not in user_sums[uid]:
                user_sums[uid][ch_idx] = np.zeros(MINUTES_PER_DAY, dtype=np.float64)
                user_counts[uid][ch_idx] = np.zeros(MINUTES_PER_DAY, dtype=np.float64)

            # Sum across all days at once — no Python row loop
            user_sums[uid][ch_idx] += np.where(valid, ch_vals, 0.0).sum(axis=0)
            user_counts[uid][ch_idx] += valid.sum(axis=0).astype(np.float64)

    return n_rows


def _compute_averaged_curves(
    user_sums: dict[str, dict[int, np.ndarray]],
    user_counts: dict[str, dict[int, np.ndarray]],
) -> pl.DataFrame:
    """Convert running sums/counts into averaged daily curves per user per channel."""
    rows = []
    for uid in sorted(user_sums.keys()):
        row: dict = {"user_id": uid}
        for ch_idx, ch_info in FPCA_CHANNELS.items():
            s = user_sums[uid].get(ch_idx, np.zeros(MINUTES_PER_DAY))
            c = user_counts[uid].get(ch_idx, np.zeros(MINUTES_PER_DAY))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                avg = np.where(c > 0, s / c, np.nan)
            row[f"avg_curve_{ch_info['name']}"] = avg.tolist()
        rows.append(row)
    return pl.DataFrame(rows)


# ── Phase B helpers ──────────────────────────────────────────────────────────


def _impute_population_mean(matrix: np.ndarray) -> np.ndarray:
    """Fill NaN values with population mean at each minute position."""
    col_means = np.nanmean(matrix, axis=0)
    # If entire minute column is NaN, fill with 0
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)
    nan_mask = np.isnan(matrix)
    matrix[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    return matrix


def _build_sparse_basis_representation(
    matrix: np.ndarray,
    n_basis: int = 24,
    smoothing_parameter: float = 1e5,
    min_obs: int = 10,
) -> FDataBasis:
    """Convert sparse averaged curves to a smooth B-spline basis representation.

    For channels like HR where zeros during wear are sensor gaps (not real zeros),
    the averaged curve has ~80% NaN. Instead of imputing with population means
    (which destroys between-user variance), we fit a smooth B-spline through
    each user's actual observations using FDataIrregular.to_basis().

    Uses L2 regularization on the 2nd derivative (curvature penalty) to prevent
    wild extrapolation in unobserved regions. The smoothing_parameter must be
    large (~1e5) because the penalty matrix scales with domain range.

    Users with fewer than `min_obs` valid points get the population-mean basis
    representation (fitted from the mean of all valid users' basis coefficients).

    Args:
        matrix: (n_users, 1440) array with NaN for missing minutes.
        n_basis: Number of B-spline basis functions (default 24 = ~1 per hour).
        smoothing_parameter: Regularization strength for curvature penalty (default 1e5).
        min_obs: Minimum non-NaN observations to attempt basis fitting.

    Returns:
        FDataBasis with shape (n_users, n_basis).
    """
    from skfda.representation import FDataBasis

    n_users = matrix.shape[0]
    basis = BSplineBasis(domain_range=(0, MINUTES_PER_DAY), n_basis=n_basis)

    # Separate users into valid (enough observations) and insufficient
    valid_user_indices = []
    insufficient_user_indices = []
    all_start_indices = []
    all_points = []
    all_values = []
    obs_offset = 0

    for i in range(n_users):
        curve = matrix[i]
        finite_mask = np.isfinite(curve)
        n_obs = finite_mask.sum()

        if n_obs >= min_obs:
            valid_user_indices.append(i)
            minutes = np.where(finite_mask)[0].astype(float)
            vals = curve[finite_mask]

            all_start_indices.append(obs_offset)
            all_points.extend(minutes.tolist())
            all_values.extend(vals.tolist())
            obs_offset += len(minutes)
        else:
            insufficient_user_indices.append(i)

    n_valid = len(valid_user_indices)
    n_insufficient = len(insufficient_user_indices)
    print(
        f"    Basis smoothing: {n_valid} users with >={min_obs} obs, "
        f"{n_insufficient} users insufficient (will get mean basis)"
    )

    # Fit basis representation for all valid users at once
    start_indices = np.array(all_start_indices)
    points = np.array(all_points).reshape(-1, 1)
    values = np.array(all_values).reshape(-1, 1)

    fd_irreg = FDataIrregular(
        start_indices=start_indices,
        points=points,
        values=values,
        domain_range=((0, MINUTES_PER_DAY),),
    )
    # Penalize curvature (2nd derivative) to prevent wild coefficients
    # in regions with few observations
    reg = L2Regularization(LinearDifferentialOperator(2))
    fd_valid = fd_irreg.to_basis(
        basis,
        smoothing_parameter=smoothing_parameter,
        regularization=reg,
    )

    # Compute mean coefficients from valid users for fallback
    mean_coefficients = fd_valid.coefficients.mean(axis=0, keepdims=True)

    # Assemble full coefficient matrix (all users, in original order)
    all_coefficients = np.zeros((n_users, n_basis), dtype=np.float64)

    for j, user_idx in enumerate(valid_user_indices):
        all_coefficients[user_idx] = fd_valid.coefficients[j]

    for user_idx in insufficient_user_indices:
        all_coefficients[user_idx] = mean_coefficients[0]

    return FDataBasis(basis=basis, coefficients=all_coefficients)


# ── Phase D helpers (cosinor + HROS) ────────────────────────────────────────


_COSINOR_FEATURES = [
    "hr_cosinor_mesor",
    "hr_cosinor_amplitude",
    "hr_cosinor_acrophase",
    "hr_cosinor_r2",
    "hr_cosinor_p_value",
    "hr_cosinor_amplitude_ratio",
    "hr_cosinor_n_minutes",
]

_HROS_PROFILE_FEATURES = [
    "hros_profile_mean",
    "hros_profile_std",
    "hros_profile_daytime_mean",
    "hros_profile_nighttime_mean",
    "hros_profile_day_night_ratio",
]


def _fit_cosinor(curve: np.ndarray) -> dict | None:
    """Fit single-component cosinor to a 1440-element averaged HR curve.

    Model: y(t) = MESOR + beta_r * cos(2*pi*t/24) + beta_s * sin(2*pi*t/24)

    This is a 3-parameter linear OLS model solved via np.linalg.lstsq.

    Args:
        curve: (1440,) array with NaN where no HR data exists.

    Returns:
        Dict with cosinor features, or None if fewer than MIN_COSINOR_DATAPOINTS valid positions.
    """
    valid = np.isfinite(curve)
    n_valid = int(valid.sum())
    if n_valid < MIN_COSINOR_DATAPOINTS:
        return None

    t_hours = np.where(valid)[0] / 60.0  # minute indices -> decimal hours
    y = curve[valid]
    omega = 2 * np.pi / 24.0

    # Design matrix: [1, cos(wt), sin(wt)]
    X = np.column_stack(
        [
            np.ones(n_valid),
            np.cos(omega * t_hours),
            np.sin(omega * t_hours),
        ]
    )

    beta, _residuals, _rank, _sv = np.linalg.lstsq(X, y, rcond=None)
    mesor, beta_r, beta_s = beta

    amplitude = np.sqrt(beta_r**2 + beta_s**2)
    # Peak time: y = M + A*cos(wt - phi) peaks when wt = phi = arctan2(beta_s, beta_r)
    acrophase_rad = np.arctan2(beta_s, beta_r)
    acrophase_hours = (acrophase_rad * 24 / (2 * np.pi)) % 24

    y_hat = X @ beta
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None

    # F-test: are beta_r, beta_s jointly significant?
    if ss_res > 0 and n_valid > 3:
        f_stat = ((ss_tot - ss_res) / 2) / (ss_res / (n_valid - 3))
        p_value = 1 - scipy.stats.f.cdf(f_stat, 2, n_valid - 3)
    else:
        f_stat = None
        p_value = None

    return {
        "hr_cosinor_mesor": float(mesor),
        "hr_cosinor_amplitude": float(amplitude),
        "hr_cosinor_acrophase": float(acrophase_hours),
        "hr_cosinor_r2": float(r2) if r2 is not None else None,
        "hr_cosinor_p_value": float(p_value) if p_value is not None else None,
        "hr_cosinor_amplitude_ratio": float(amplitude / mesor) if mesor > 0 else None,
        "hr_cosinor_n_minutes": n_valid,
    }


def _compute_hros_profile(avg_hr: np.ndarray, avg_steps: np.ndarray) -> dict:
    """Compute HROS profile statistics from averaged HR and steps curves.

    HROS = HR / (steps + 1) at each minute position (only where HR is valid).
    Captures how much the heart rate is elevated relative to activity level
    across the 24h profile.

    Args:
        avg_hr: (1440,) averaged HR curve (bpm), NaN where no data.
        avg_steps: (1440,) averaged steps curve (steps/min), may include zeros.

    Returns:
        Dict with 5 HROS profile features.
    """
    valid_hr = np.isfinite(avg_hr)
    hros_profile = np.full(MINUTES_PER_DAY, np.nan)
    hros_profile[valid_hr] = avg_hr[valid_hr] / (avg_steps[valid_hr] + 1.0)

    # Time windows
    daytime = slice(DAYTIME_START_MIN, DAYTIME_END_MIN)
    nighttime_mask = np.zeros(MINUTES_PER_DAY, dtype=bool)
    nighttime_mask[0:DAYTIME_START_MIN] = True  # midnight – 6 AM
    nighttime_mask[NIGHTTIME_LATE_MIN:MINUTES_PER_DAY] = True  # 11 PM – midnight

    all_valid = hros_profile[np.isfinite(hros_profile)]
    day_valid = hros_profile[daytime]
    day_valid = day_valid[np.isfinite(day_valid)]
    night_valid = hros_profile[nighttime_mask]
    night_valid = night_valid[np.isfinite(night_valid)]

    result = {
        "hros_profile_mean": float(np.nanmean(all_valid)) if len(all_valid) > 0 else None,
        "hros_profile_std": float(np.nanstd(all_valid)) if len(all_valid) > 0 else None,
        "hros_profile_daytime_mean": (float(np.nanmean(day_valid)) if len(day_valid) > 0 else None),
        "hros_profile_nighttime_mean": (
            float(np.nanmean(night_valid)) if len(night_valid) > 0 else None
        ),
    }

    if (
        result["hros_profile_daytime_mean"] is not None
        and result["hros_profile_nighttime_mean"] is not None
    ):
        night_mean = result["hros_profile_nighttime_mean"]
        result["hros_profile_day_night_ratio"] = (
            result["hros_profile_daytime_mean"] / night_mean if night_mean > 0 else None
        )
    else:
        result["hros_profile_day_night_ratio"] = None

    return result


# ── Main entry point ─────────────────────────────────────────────────────────


def build_curve_analysis_features(
    arrow_dir: Path,
    output_path: Path | None = None,
    n_components: int = 10,
    checkpoint_path: Path | None = None,
    splits: list[str] | None = None,
    n_basis: int = 24,
    smoothing_parameter: float = 1e5,
    min_obs: int = 10,
    max_nonwear_minutes: int | None = None,
    variance_filter: bool = True,
    cutoff_dates: dict[str, str] | None = None,
    eligible_keys: set[tuple[str, str]] | None = None,
) -> pl.DataFrame:
    """Build curve analysis user-level features from Arrow files.

    Phases A-C extract per-user averaged daily curves for 4 channels, fit
    Functional PCA, and produce per-user FPCA scores.  Phase D fits a cosinor
    model to each user's averaged HR curve and computes HROS profile statistics
    from the averaged HR and steps curves (already in memory from Phase A).

    For sparse channels (HR, where zeros during wear are sensor gaps), uses
    B-spline basis smoothing instead of population-mean imputation to preserve
    between-user variance.

    Args:
        arrow_dir: Path to directory containing Arrow files (with train/test/val subdirs).
        output_path: Optional path to write the output Parquet file.
        n_components: Number of FPCA components per channel (default 10 -> 40 total).
        checkpoint_path: Path to cache averaged curves as Parquet (enables resume).
        splits: Optional list of splits to process (default: all).
        n_basis: Number of B-spline basis functions for sparse channels (default 24).
        smoothing_parameter: Curvature penalty strength for basis smoothing (default 1e5).
        min_obs: Minimum non-NaN observations per user for basis fitting (default 10).
        max_nonwear_minutes: If set, drop rows where total_nonwear_minutes exceeds
                             this value before accumulating curves.
        variance_filter: If True (default), drop rows where a monitored channel
                         has near-zero variance (flat signal = sensor malfunction).
        cutoff_dates: Optional ``{user_id: "YYYY-MM-DD"}`` per-user data cutoff.
                      Rows with ``date > cutoff_dates[user_id]`` are excluded.

    Returns:
        DataFrame with user_id + (n_components x 4 channels) FPCA score columns
        + 7 cosinor features + 5 HROS profile features.
    """
    from skfda.preprocessing.dim_reduction import FPCA
    from skfda.representation import FDataGrid

    arrow_dir = Path(arrow_dir)
    if not arrow_dir.exists():
        raise FileNotFoundError(f"Directory not found: {arrow_dir}")

    if splits is None:
        splits = ["train", "test", "val"]

    # ── Phase A: Per-user averaged daily curves ──────────────────────────────
    curves_df = None
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        print(f"Loading cached averaged curves from {checkpoint_path}")
        curves_df = pl.read_parquet(checkpoint_path)
        print(f"  {curves_df.shape[0]} users loaded from checkpoint")

    if curves_df is None:
        print("Phase A: Extracting per-user averaged daily curves (chunked)...")

        arrow_files = []
        for split in splits:
            split_dir = arrow_dir / split
            if split_dir.exists():
                arrow_files.extend(sorted(split_dir.glob("*.arrow")))

        # Flat directory fallback (MHC-B daily_hf has no subdirs)
        if not arrow_files:
            arrow_files = sorted(arrow_dir.glob("data-*.arrow"))

        if not arrow_files:
            raise ValueError(f"No Arrow files found in {arrow_dir}")

        print(f"  Processing {len(arrow_files)} Arrow files across {len(splits)} splits")

        user_sums: dict[str, dict[int, np.ndarray]] = {}
        user_counts: dict[str, dict[int, np.ndarray]] = {}

        t0 = time.time()
        for i, arrow_file in enumerate(arrow_files):
            n_rows = _accumulate_arrow_file(
                arrow_file,
                user_sums,
                user_counts,
                max_nonwear_minutes,
                variance_filter=variance_filter,
                cutoff_dates=cutoff_dates,
                eligible_keys=eligible_keys,
            )
            elapsed = time.time() - t0
            done = i + 1
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(arrow_files) - done) / rate if rate > 0 else 0
            print(
                f"  [{done}/{len(arrow_files)}] {arrow_file.name}: {n_rows} rows, "
                f"{len(user_sums)} users so far ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)",
                flush=True,
            )

        print(f"  Computing averaged curves for {len(user_sums)} users...")
        curves_df = _compute_averaged_curves(user_sums, user_counts)
        del user_sums, user_counts

        if checkpoint_path is not None:
            cp = Path(checkpoint_path)
            cp.parent.mkdir(parents=True, exist_ok=True)
            curves_df.write_parquet(cp)
            print(f"  Saved averaged curves checkpoint to {cp}")

    # ── Phase B: Assemble matrices and impute / smooth ───────────────────────
    print("Phase B: Assembling matrices and processing channels...")
    user_ids = curves_df["user_id"].to_list()
    n_users = len(user_ids)
    channel_matrices: dict[str, np.ndarray] = {}  # non-sparse channels (FDataGrid path)
    channel_basis: dict[str, object] = {}  # sparse channels (FDataBasis path)

    for ch_idx, ch_info in FPCA_CHANNELS.items():
        col_name = f"avg_curve_{ch_info['name']}"
        raw_lists = curves_df[col_name].to_list()
        matrix = np.array(raw_lists, dtype=np.float64)  # (n_users, 1440)

        n_nan_total = np.isnan(matrix).sum()
        nan_frac = n_nan_total / matrix.size * 100

        if ch_info["sparse_zeros"]:
            # Sparse channel (HR): B-spline basis smoothing
            print(
                f"  {ch_info['name']}: {n_users}x{MINUTES_PER_DAY}, "
                f"{n_nan_total} NaN ({nan_frac:.1f}%) -> basis smoothing"
            )
            fd_basis = _build_sparse_basis_representation(
                matrix,
                n_basis=n_basis,
                smoothing_parameter=smoothing_parameter,
                min_obs=min_obs,
            )
            channel_basis[ch_info["name"]] = fd_basis
        else:
            # Non-sparse channel: population-mean imputation (unchanged)
            matrix = _impute_population_mean(matrix)
            n_nan_after = np.isnan(matrix).sum()
            print(
                f"  {ch_info['name']}: {n_users}x{MINUTES_PER_DAY}, "
                f"imputed {n_nan_total - n_nan_after} NaN values"
            )
            channel_matrices[ch_info["name"]] = matrix

    # ── Phase C: Fit FPCA per channel ────────────────────────────────────────
    print(f"Phase C: Fitting FPCA with {n_components} components per channel...")
    grid_points = np.arange(MINUTES_PER_DAY)
    score_columns: dict[str, list] = {"user_id": user_ids}

    # Process all channels in FPCA_CHANNELS order (deterministic)
    for ch_idx, ch_info in FPCA_CHANNELS.items():
        ch_name = ch_info["name"]
        fpca = FPCA(n_components=n_components)

        # skfda's FPCA delegates to sklearn PCA, which on a large matrix (>500x500) selects
        # the RANDOMIZED SVD solver with no random_state — it draws from the global NumPy
        # RNG, so the scores vary run-to-run. Seed that RNG before each fit so the
        # components are reproducible across runs and machines.
        np.random.seed(42)
        if ch_name in channel_basis:
            # Sparse channel: FPCA on FDataBasis (B-spline smoothed)
            fd = channel_basis[ch_name]
            scores = fpca.fit_transform(fd)
        else:
            # Non-sparse channel: FPCA on FDataGrid (population-mean imputed)
            fd = FDataGrid(channel_matrices[ch_name], grid_points=grid_points)
            scores = fpca.fit_transform(fd)

        # Canonicalize component signs. FPCA eigenfunctions are defined only up to sign,
        # and skfda leaves the sign machine/library-version dependent, so the same data can
        # yield ``+component`` on one host and ``-component`` on another. Flip each component
        # (and its scores) so its largest-magnitude loading is positive — a deterministic
        # rule that makes the fpca_* features reproducible across machines.
        loadings = (
            np.asarray(fpca.components_(grid_points)).reshape(n_components, -1)
            if ch_name in channel_basis
            else fpca.components_.data_matrix[:, :, 0]
        )
        for k in range(n_components):
            if loadings[k][np.argmax(np.abs(loadings[k]))] < 0:
                scores[:, k] *= -1

        # Variance explained
        total_var = fpca.explained_variance_.sum()
        cumvar = np.cumsum(fpca.explained_variance_) / total_var * 100
        cumvar_str = ", ".join(f"{v:.1f}%" for v in cumvar)
        label = "(basis)" if ch_name in channel_basis else "(grid)"
        print(f"  {ch_name} {label}: cumulative variance explained = [{cumvar_str}]")

        for k in range(n_components):
            col = f"fpca_{ch_name}_{k + 1}"
            score_columns[col] = scores[:, k].tolist()

    result = pl.DataFrame(score_columns)
    print(f"  FPCA scores: {result.shape[0]} users x {result.shape[1] - 1} columns")

    # ── Phase D: Cosinor + HROS from averaged curves ────────────────────────
    print("Phase D: Fitting cosinor + HROS profile from averaged curves...")
    hr_lists = curves_df["avg_curve_watch_hr"].to_list()
    steps_lists = curves_df["avg_curve_watch_steps"].to_list()

    cosinor_rows = []
    n_cosinor_ok = 0
    for i in range(n_users):
        avg_hr = np.array(hr_lists[i], dtype=np.float64)
        avg_steps = np.array(steps_lists[i], dtype=np.float64)

        row: dict = {}

        # Cosinor fit
        cosinor = _fit_cosinor(avg_hr)
        if cosinor is not None:
            row.update(cosinor)
            n_cosinor_ok += 1
        else:
            for feat in _COSINOR_FEATURES:
                row[feat] = None

        # HROS profile
        row.update(_compute_hros_profile(avg_hr, avg_steps))
        cosinor_rows.append(row)

    print(f"  Cosinor fit successful: {n_cosinor_ok}/{n_users} users")
    print(f"  Cosinor insufficient data: {n_users - n_cosinor_ok} users (< 200 valid minutes)")

    cosinor_df = pl.DataFrame(cosinor_rows)
    result = pl.concat([result, cosinor_df], how="horizontal")

    n_features = result.shape[1] - 1  # exclude user_id
    print(f"Pipeline Curve Analysis: {result.shape[0]} users x {n_features} features")

    # ── Write output ─────────────────────────────────────────────────────────
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.write_parquet(output_path)
        print(f"Wrote curve analysis features to {output_path}")

    return result
