"""Feature extraction functions for MHC wearable data.

This module provides Polars expressions for extracting features from the nested
time-series data. Features are organized into categories:

- Robust Baselines: Simple daily aggregates (sum, median)
- Physical Activity: Activity-specific and workout metrics
- Sleep: Sleep duration and quality metrics
- Circadian: Time-of-day patterns

Each category has two functions:
- `extract_<category>_daily()`: Returns expressions for day-level metrics
- `aggregate_<category>_to_user()`: Returns expressions for user-level aggregations

Usage:
    >>> lf.with_columns(extract_robust_baselines_daily())
    >>> lf.group_by("user_id").agg(aggregate_robust_baselines_to_user())
"""

from __future__ import annotations

import polars as pl

from .constants import (
    CHANNEL_NAMES,
    IPHONE_DISTANCE,
    IPHONE_FLIGHTS,
    IPHONE_STEPS,
    SLEEP_ASLEEP,
    SLEEP_INBED,
    WATCH_DISTANCE,
    WATCH_ENERGY,
    WATCH_HR,
    WATCH_STEPS,
    WORKOUT_CHANNELS,
    WORKOUT_CYCLING,
    WORKOUT_ELLIPTICAL,
    WORKOUT_FUNCTIONAL,
    WORKOUT_HIIT,
    WORKOUT_MIXED_CARDIO,
    WORKOUT_RUNNING,
    WORKOUT_STRENGTH,
    WORKOUT_WALKING,
)

# =============================================================================
# Robust Baselines
# =============================================================================


def extract_robust_baselines_daily() -> list[pl.Expr]:
    """Extract day-level robust baseline metrics for continuous channels.

    Computes daily aggregates for channels 0-6:
    - Steps/Distance/Flights/Energy: daily SUM (cumulative metrics)
    - Heart Rate: daily MEDIAN of non-zero values only (converted to bpm)
      Note: HR is sampled intermittently (~84% of minutes are zero), so zeros
      must be filtered before computing median.

    Returns:
        List of Polars expressions. Output columns:
        - daily_iphone_steps_sum
        - daily_iphone_distance_sum
        - daily_iphone_flights_sum
        - daily_watch_steps_sum
        - daily_watch_distance_sum
        - daily_watch_hr_p5 (5th percentile in bpm)
        - daily_watch_hr_median (in bpm, converted from beats/sec)
        - daily_watch_hr_p95 (95th percentile in bpm)
        - daily_watch_energy_sum

    Example:
        >>> lf.with_columns(extract_robust_baselines_daily())
    """
    # HR: filter zeros and NaN first (only sampled intermittently, ~84% of minutes
    # are zero, and channels are all-NaN for users without a watch).
    # NaN > 0 is True in Polars, so we must gate on is_finite().
    hr_nonzero = (
        pl.col("data")
        .arr.get(WATCH_HR)
        .list.eval(pl.element().filter(pl.element().is_finite() & (pl.element() > 0)))
    )

    return [
        # iPhone metrics (sum for cumulative)
        pl.col("data").arr.get(IPHONE_STEPS).list.sum().alias("daily_iphone_steps_sum"),
        pl.col("data").arr.get(IPHONE_DISTANCE).list.sum().alias("daily_iphone_distance_sum"),
        pl.col("data").arr.get(IPHONE_FLIGHTS).list.sum().alias("daily_iphone_flights_sum"),
        # Watch metrics
        pl.col("data").arr.get(WATCH_STEPS).list.sum().alias("daily_watch_steps_sum"),
        pl.col("data").arr.get(WATCH_DISTANCE).list.sum().alias("daily_watch_distance_sum"),
        # HR percentiles (converted to bpm)
        (hr_nonzero.list.eval(pl.element().quantile(0.05)).list.first() * 60).alias(
            "daily_watch_hr_p5"
        ),
        (hr_nonzero.list.median() * 60).alias("daily_watch_hr_median"),
        (hr_nonzero.list.eval(pl.element().quantile(0.95)).list.first() * 60).alias(
            "daily_watch_hr_p95"
        ),
        pl.col("data").arr.get(WATCH_ENERGY).list.sum().alias("daily_watch_energy_sum"),
    ]


def aggregate_robust_baselines_to_user() -> list[pl.Expr]:
    """Aggregate day-level robust baselines to user-level features.

    Uses percentiles for robust central tendency (resistant to outlier days).

    Requires columns from `extract_robust_baselines_daily()` to be computed first.

    Returns:
        List of Polars expressions. Output columns:

        Metadata:
        - n_days: Number of days of data for this user

        For sum metrics (steps, distance, flights, energy):
        - <metric>_p5: 5th percentile across days
        - <metric>_p50: Median across days
        - <metric>_p95: 95th percentile across days
        - <metric>_iqr: Interquartile range (P75 - P25)

        For HR (already has daily p5/median/p95):
        - watch_hr_p5_p50: Median of daily 5th percentile HR
        - watch_hr_median_p50: Median of daily median HR
        - watch_hr_p95_p50: Median of daily 95th percentile HR
        - watch_hr_iqr: IQR of daily median HR

    Example:
        >>> lf.with_columns(extract_robust_baselines_daily()).group_by("user_id").agg(
        ...     aggregate_robust_baselines_to_user()
        ... )
    """
    # Non-HR metrics: aggregate with p5, p50, p95, iqr
    sum_metrics = [
        "daily_iphone_steps_sum",
        "daily_iphone_distance_sum",
        "daily_iphone_flights_sum",
        "daily_watch_steps_sum",
        "daily_watch_distance_sum",
        "daily_watch_energy_sum",
    ]

    # Start with metadata
    exprs = [pl.len().alias("n_days")]

    for metric in sum_metrics:
        # Remove "daily_" prefix for cleaner user-level names
        base_name = metric.replace("daily_", "").replace("_sum", "")
        exprs.extend(
            [
                pl.col(metric).quantile(0.05).alias(f"{base_name}_p5"),
                pl.col(metric).median().alias(f"{base_name}_p50"),
                pl.col(metric).quantile(0.95).alias(f"{base_name}_p95"),
                (pl.col(metric).quantile(0.75) - pl.col(metric).quantile(0.25)).alias(
                    f"{base_name}_iqr"
                ),
            ]
        )

    # HR metrics: we have daily p5, median, p95 - take median of each across days
    exprs.extend(
        [
            pl.col("daily_watch_hr_p5").median().alias("watch_hr_p5_p50"),
            pl.col("daily_watch_hr_median").median().alias("watch_hr_median_p50"),
            pl.col("daily_watch_hr_p95").median().alias("watch_hr_p95_p50"),
            # Also include IQR of the daily median HR
            (
                pl.col("daily_watch_hr_median").quantile(0.75)
                - pl.col("daily_watch_hr_median").quantile(0.25)
            ).alias("watch_hr_iqr"),
        ]
    )

    return exprs


# =============================================================================
# Physical Activity - Sub-functions
# =============================================================================

# Default HR max threshold (80% of ~200 bpm typical max)
# Can be parameterized per-user if age is available: 0.8 * (220 - age)
DEFAULT_HR_MAX_THRESHOLD_BPM = 120  # 80% of 200 bpm

# Intensity zone thresholds (steps per minute)
SEDENTARY_THRESHOLD = 0  # 0 steps = sedentary
MODERATE_THRESHOLD = 40  # 60+ steps/min = moderate
VIGOROUS_THRESHOLD = 80  # 100+ steps/min = vigorous


def _extract_active_minutes_daily() -> list[pl.Expr]:
    """Extract daily active minutes from both Watch and iPhone.

    Active minute = minute with steps > 0 on the respective device.

    Returns:
        List of Polars expressions:
        - daily_watch_active_minutes: Minutes with watch steps > 0
        - daily_iphone_active_minutes: Minutes with iPhone steps > 0
        - daily_combined_active_minutes: Minutes with steps > 0 on either device
    """
    return [
        # Watch active minutes (fill_nan(None) so NaN minutes are excluded, not counted as active)
        (
            pl.col("data")
            .arr.get(WATCH_STEPS)
            .list.eval(pl.element().fill_nan(None) > 0)
            .list.sum()
            .alias("daily_watch_active_minutes")
        ),
        # iPhone active minutes
        (
            pl.col("data")
            .arr.get(IPHONE_STEPS)
            .list.eval(pl.element().fill_nan(None) > 0)
            .list.sum()
            .alias("daily_iphone_active_minutes")
        ),
    ]


# Minimum steps required for reliable gait efficiency calculation
MIN_STEPS_FOR_GAIT_EFFICIENCY = 500
# Maximum physiological gait efficiency (meters per step)
# Normal walking ~0.7m, running ~1.2m, anything above 1.5m is likely GPS drift
MAX_GAIT_EFFICIENCY = 1.5


def _extract_gait_efficiency_daily() -> list[pl.Expr]:
    """Extract walking/running efficiency: Distance / Steps ratio.

    Captures gait/stride length signatures. Higher values = longer strides.
    Filters:
    - Only computed when steps >= MIN_STEPS_FOR_GAIT_EFFICIENCY
    - Capped at MAX_GAIT_EFFICIENCY to filter GPS drift outliers

    Returns:
        List of Polars expressions:
        - daily_watch_gait_efficiency: Watch distance / watch steps (meters/step)
        - daily_iphone_gait_efficiency: iPhone distance / iPhone steps (meters/step)
    """
    exprs = []
    for device, steps_ch, dist_ch in [
        ("watch", WATCH_STEPS, WATCH_DISTANCE),
        ("iphone", IPHONE_STEPS, IPHONE_DISTANCE),
    ]:
        steps_sum = pl.col("data").arr.get(steps_ch).list.sum()
        dist_sum = pl.col("data").arr.get(dist_ch).list.sum()
        raw_efficiency = dist_sum / steps_sum

        # Only compute efficiency if enough steps, and cap at physiological max
        efficiency = (
            pl.when(steps_sum >= MIN_STEPS_FOR_GAIT_EFFICIENCY)
            .then(
                pl.when(raw_efficiency <= MAX_GAIT_EFFICIENCY)
                .then(raw_efficiency)
                .otherwise(None)  # Treat extreme values as missing
            )
            .otherwise(None)
        ).alias(f"daily_{device}_gait_efficiency")
        exprs.append(efficiency)

    return exprs


def _extract_hr_coupling_daily() -> list[pl.Expr]:
    """Extract HR-Activity coupling metrics using time-based proxies.

    Since Polars list.eval doesn't support struct filtering, we use time-based
    proxies that correlate well with activity states:
    - Daytime HR (6am-10pm): proxy for "active" HR
    - Nighttime HR (11pm-6am): proxy for "resting" HR
    - P95 HR: proxy for "workout/peak" HR

    Returns:
        List of Polars expressions:
        - daily_hr_daytime_median: Median HR during wake hours (6am-10pm)
        - daily_hr_nighttime_median: Median HR during sleep hours (11pm-6am)
        - daily_hr_p95: 95th percentile HR (proxy for workout intensity)
    """
    hr_list = pl.col("data").arr.get(WATCH_HR)

    # Helper: filter HR to valid non-zero readings (NaN > 0 is True in Polars,
    # so we must gate on is_finite() to exclude NaN values from all-NaN channels).
    _hr_valid = pl.element().is_finite() & (pl.element() > 0)

    # Daytime HR: 6am-10pm (minutes 360-1320)
    hr_daytime = (
        hr_list.list.slice(360, 960)  # 6am to 10pm
        .list.eval(pl.element().filter(_hr_valid))
        .list.median()
        * 60  # Convert to bpm
    ).alias("daily_hr_daytime_median")

    # Nighttime HR: 11pm-6am (minutes 1380-1440 + 0-360)
    hr_late_night = hr_list.list.slice(1380, 60)  # 11pm to midnight
    hr_early_morning = hr_list.list.slice(0, 360)  # midnight to 6am

    hr_nighttime = (
        hr_late_night.list.concat(hr_early_morning)
        .list.eval(pl.element().filter(_hr_valid))
        .list.median()
        * 60  # Convert to bpm
    ).alias("daily_hr_nighttime_median")

    # P95 HR as proxy for workout/peak exertion
    hr_p95 = (
        hr_list.list.eval(pl.element().filter(_hr_valid))
        .list.eval(pl.element().quantile(0.95))
        .list.first()
        * 60
    ).alias("daily_hr_p95")

    return [hr_daytime, hr_nighttime, hr_p95]


def _extract_intensity_zones_daily() -> list[pl.Expr]:
    """Extract time spent in different activity intensity zones for BOTH Watch and iPhone.

    Zones based on steps per minute (thresholds from constants):
    - Sedentary: 0 steps/min during WEAR time only (non-wear excluded)
    - Light: 1 to MODERATE_THRESHOLD-1 steps/min
    - Moderate: MODERATE_THRESHOLD to VIGOROUS_THRESHOLD-1 steps/min
    - Vigorous: VIGOROUS_THRESHOLD+ steps/min

    Returns:
        List of Polars expressions for watch and iPhone:
        - daily_watch_sedentary_minutes, daily_iphone_sedentary_minutes
        - daily_watch_light_minutes, daily_iphone_light_minutes
        - daily_watch_moderate_minutes, daily_iphone_moderate_minutes
        - daily_watch_vigorous_minutes, daily_iphone_vigorous_minutes
    """
    exprs = []

    # Non-wear minutes: all activity channels are zeroed during non-wear,
    # so raw sedentary count (steps==0) includes non-wear time.
    # Subtract total_nonwear_minutes to get wear-only sedentary.
    nonwear_total = pl.col("total_nonwear_minutes")

    for device, channel in [("watch", WATCH_STEPS), ("iphone", IPHONE_STEPS)]:
        # Convert NaN to null so NaN minutes are excluded from all zone counts
        # (NaN > 0 is True in Polars, which would wrongly count NaN as active)
        steps = pl.col("data").arr.get(channel).list.eval(pl.element().fill_nan(None))
        raw_sedentary = steps.list.eval(pl.element() == 0).list.sum()
        exprs.extend(
            [
                # Sedentary: 0 steps during wear time (subtract non-wear minutes)
                pl.max_horizontal(raw_sedentary - nonwear_total, 0).alias(
                    f"daily_{device}_sedentary_minutes"
                ),
                # Light: 1 to MODERATE_THRESHOLD-1
                steps.list.eval((pl.element() > 0) & (pl.element() < MODERATE_THRESHOLD))
                .list.sum()
                .alias(f"daily_{device}_light_minutes"),
                # Moderate: MODERATE_THRESHOLD to VIGOROUS_THRESHOLD-1
                steps.list.eval(
                    (pl.element() >= MODERATE_THRESHOLD) & (pl.element() < VIGOROUS_THRESHOLD)
                )
                .list.sum()
                .alias(f"daily_{device}_moderate_minutes"),
                # Vigorous: VIGOROUS_THRESHOLD+
                steps.list.eval(pl.element() >= VIGOROUS_THRESHOLD)
                .list.sum()
                .alias(f"daily_{device}_vigorous_minutes"),
            ]
        )

    return exprs


def _extract_workout_metrics_daily() -> list[pl.Expr]:
    """Extract daily workout presence and metrics.

    Note: Some workout channels contain NaN values for users without that workout type.
    We fill NaN with 0 before summing to handle this properly.

    Returns:
        List of Polars expressions:
        - daily_workout_minutes: Total minutes in any workout
        - daily_has_any_workout: Binary flag if any workout occurred
        - daily_workout_type_count: Number of unique workout types today
        - daily_has_<type>: Binary indicators for each workout type
        - daily_cardio_minutes: Minutes in cardio workouts (walking, running, cycling, elliptical, HIIT, mixed)
        - daily_strength_minutes: Minutes in strength workouts (strength, functional)
    """

    # Helper: sum a workout channel, filling NaN with 0
    def workout_channel_sum(ch: int) -> pl.Expr:
        return pl.col("data").arr.get(ch).list.eval(pl.element().fill_nan(0)).list.sum()

    # Helper: check if workout occurred (any minute > 0, NaN treated as 0)
    def workout_occurred(ch: int) -> pl.Expr:
        return (workout_channel_sum(ch) > 0).cast(pl.Int8)

    # Total workout minutes (sum across all channels, NaN -> 0)
    workout_minutes = pl.sum_horizontal([workout_channel_sum(ch) for ch in WORKOUT_CHANNELS]).alias(
        "daily_workout_minutes"
    )

    # Any workout flag
    has_any_workout = (
        (pl.sum_horizontal([workout_channel_sum(ch) for ch in WORKOUT_CHANNELS]) > 0)
        .cast(pl.Int8)
        .alias("daily_has_any_workout")
    )

    # Count unique workout types (how many different workout types today)
    workout_type_count = pl.sum_horizontal([workout_occurred(ch) for ch in WORKOUT_CHANNELS]).alias(
        "daily_workout_type_count"
    )

    # Binary indicators for each workout type
    workout_indicators = [
        workout_occurred(ch).alias(f"daily_has_{CHANNEL_NAMES[ch].replace('workout_', '')}")
        for ch in WORKOUT_CHANNELS
    ]

    # Cardio vs Strength breakdown
    # Cardio: walking, running, cycling, elliptical, HIIT, mixed cardio
    cardio_channels = [
        WORKOUT_WALKING,
        WORKOUT_RUNNING,
        WORKOUT_CYCLING,
        WORKOUT_ELLIPTICAL,
        WORKOUT_HIIT,
        WORKOUT_MIXED_CARDIO,
    ]
    # Strength: strength training, functional training
    strength_channels = [WORKOUT_STRENGTH, WORKOUT_FUNCTIONAL]

    cardio_minutes = pl.sum_horizontal([workout_channel_sum(ch) for ch in cardio_channels]).alias(
        "daily_cardio_minutes"
    )

    strength_minutes = pl.sum_horizontal(
        [workout_channel_sum(ch) for ch in strength_channels]
    ).alias("daily_strength_minutes")

    return [
        workout_minutes,
        has_any_workout,
        workout_type_count,
        cardio_minutes,
        strength_minutes,
    ] + workout_indicators


def _extract_device_sync_daily() -> list[pl.Expr]:
    """Extract sync factor between iPhone and Watch step counts.

    Measures agreement between devices. Low sync might indicate:
    - Device not worn consistently
    - Potential injury (favoring one device)
    - Different activity types

    Sync ratio = min(watch, iphone) / max(watch, iphone)
    - 1.0 = perfect agreement
    - 0.0 = one device recorded nothing

    Returns:
        List of Polars expressions:
        - daily_step_sync_ratio: Concordance between watch and phone steps (0-1)
    """
    watch_steps = pl.col("data").arr.get(WATCH_STEPS).list.sum()
    iphone_steps = pl.col("data").arr.get(IPHONE_STEPS).list.sum()

    # Sync ratio: min/max (1 = perfect sync, 0 = no sync)
    sync_ratio = (
        pl.min_horizontal(watch_steps, iphone_steps)
        / pl.max_horizontal(watch_steps, iphone_steps).replace(0, None)
    ).alias("daily_step_sync_ratio")

    return [sync_ratio]


def _extract_hr_intensity_daily() -> list[pl.Expr]:
    """Extract heart rate intensity metrics.

    Measures cardiovascular effort:
    - Daily max HR achieved
    - Whether 80% HR max threshold was reached (proxy for intense exercise)

    Returns:
        List of Polars expressions:
        - daily_hr_max: Maximum HR recorded (bpm)
        - daily_hr_80pct_achieved: Binary flag if HR >= 120 bpm was reached
    """
    # Filter to finite non-zero HR first (list.max() propagates NaN, and
    # NaN >= threshold is True in Polars, giving false positives).
    hr_bpm = (
        pl.col("data")
        .arr.get(WATCH_HR)
        .list.eval(pl.element().filter(pl.element().is_finite() & (pl.element() > 0)))
        .list.max()
        * 60
    )

    return [
        hr_bpm.alias("daily_hr_max"),
        (hr_bpm >= DEFAULT_HR_MAX_THRESHOLD_BPM).cast(pl.Int8).alias("daily_hr_80pct_achieved"),
    ]


def _extract_workout_hr_dynamics_daily() -> list[pl.Expr]:
    """Extract HR dynamics around workout boundaries (start and end).

    Uses simplified day-level proxies:
    - Finds the LAST workout minute of the day (end of last workout)
    - Finds the FIRST workout minute of the day (start of first workout)
    - Samples HR at offsets from those boundaries

    **Post-Workout HR Recovery** (Features 1 & 2):
        HR drop after the last workout ends. Captures cardiovascular recovery
        fitness — faster recovery = better cardiorespiratory fitness.
        - daily_hr_at_workout_end: HR (bpm) at the last workout minute
        - daily_hr_recovery_1min: HR drop (bpm) 1 min after workout end
        - daily_hr_recovery_5min: HR drop (bpm) 5 min after workout end
        - daily_hr_recovery_10min: HR drop (bpm) 10 min after workout end
        - daily_hr_recovery_ratio: Normalized recovery (HR_end - HR_5min) / HR_end

    **HR Activation** (Feature 3):
        HR rise after workout starts. Captures cardiovascular responsiveness.
        - daily_hr_at_workout_start: HR (bpm) at the first workout minute
        - daily_hr_activation_5min: HR rise (bpm) 5 min after workout start
        - daily_hr_activation_10min: HR rise (bpm) 10 min after workout start

    All metrics are null on days without workouts or when HR is not sampled
    at the relevant minute (HR is intermittent, ~84% zeros).

    Note: HR is stored as beats/second (channel 5), multiply by 60 for bpm.
    Workout channels (9-18) are binary flags with possible NaN for non-trackers.

    Returns:
        List of Polars expressions for day-level workout HR dynamics.
    """
    hr_list = pl.col("data").arr.get(WATCH_HR)

    # Find first/last workout minute across all 10 workout channels.
    # For each channel, find indices where workout==1, then take min (first)
    # and max (last). Take overall min/max across channels.
    last_workout_per_channel = []
    first_workout_per_channel = []
    for ch in WORKOUT_CHANNELS:
        arr = pl.col("data").arr.get(ch).list.eval(pl.element().fill_nan(0))
        idx_expr = arr.list.eval(
            pl.when(pl.element() > 0).then(pl.int_range(pl.len())).otherwise(None)
        )
        last_workout_per_channel.append(idx_expr.list.max())
        first_workout_per_channel.append(idx_expr.list.min())

    last_workout_min = pl.max_horizontal(last_workout_per_channel)
    first_workout_min = pl.min_horizontal(first_workout_per_channel)
    has_workout = last_workout_min.is_not_null()

    # --- Post-Workout HR Recovery ---
    # Sample HR at workout end and at 1/5/10 min offsets (null_on_oob for safety).
    # HR is sampled intermittently (~84% zeros), so we use a small window median
    # around each target minute to get a more reliable reading.
    # Window: +-2 min (5-minute window) to increase chance of a valid HR sample.

    def _hr_window_median(minute_expr: pl.Expr, window_half: int = 2) -> pl.Expr:
        """Median HR (bpm) in a small window around target minute, ignoring zeros/NaN."""
        # Clamp start to 0
        start = pl.max_horizontal(minute_expr - window_half, pl.lit(0))
        window_size = window_half * 2 + 1
        return (
            hr_list.list.slice(start, window_size)
            .list.eval(pl.element().filter(pl.element().is_finite() & (pl.element() > 0)))
            .list.median()
            * 60
        )

    hr_at_end = _hr_window_median(last_workout_min)
    hr_at_1min = _hr_window_median(last_workout_min + 1)
    hr_at_5min = _hr_window_median(last_workout_min + 5)
    hr_at_10min = _hr_window_median(last_workout_min + 10)

    exprs = [
        pl.when(has_workout).then(hr_at_end).otherwise(None).alias("daily_hr_at_workout_end"),
        # HR recovery (positive = HR dropping = good recovery)
        pl.when(has_workout)
        .then(hr_at_end - hr_at_1min)
        .otherwise(None)
        .alias("daily_hr_recovery_1min"),
        pl.when(has_workout)
        .then(hr_at_end - hr_at_5min)
        .otherwise(None)
        .alias("daily_hr_recovery_5min"),
        pl.when(has_workout)
        .then(hr_at_end - hr_at_10min)
        .otherwise(None)
        .alias("daily_hr_recovery_10min"),
        # Normalized recovery ratio
        (
            pl.when(has_workout & (hr_at_end > 0))
            .then((hr_at_end - hr_at_5min) / hr_at_end)
            .otherwise(None)
            .alias("daily_hr_recovery_ratio")
        ),
    ]

    # --- HR Activation at Workout Start ---
    hr_at_start = _hr_window_median(first_workout_min)
    hr_at_start_5 = _hr_window_median(first_workout_min + 5)
    hr_at_start_10 = _hr_window_median(first_workout_min + 10)

    exprs.extend(
        [
            pl.when(has_workout)
            .then(hr_at_start)
            .otherwise(None)
            .alias("daily_hr_at_workout_start"),
            # HR activation (positive = HR rising = normal response)
            pl.when(has_workout)
            .then(hr_at_start_5 - hr_at_start)
            .otherwise(None)
            .alias("daily_hr_activation_5min"),
            pl.when(has_workout)
            .then(hr_at_start_10 - hr_at_start)
            .otherwise(None)
            .alias("daily_hr_activation_10min"),
            # Normalized activation ratio
            (
                pl.when(has_workout & (hr_at_start > 0))
                .then((hr_at_start_5 - hr_at_start) / hr_at_start)
                .otherwise(None)
                .alias("daily_hr_activation_ratio")
            ),
        ]
    )

    return exprs


def _extract_sedentary_bouts_daily() -> list[pl.Expr]:
    """Extract sedentary time during wake hours for both Watch and iPhone.

    Captures prolonged sitting periods (e.g., office work).
    Wake hours defined as 6am-10pm (minutes 360-1320).

    Non-wear minutes during wake hours are excluded: all activity channels
    are zeroed during non-wear, so raw zero-step counts would be inflated.

    Note: This counts total sedentary minutes during wake hours.
    True bout detection (consecutive zeros) is complex in pure Polars.

    Returns:
        List of Polars expressions:
        - daily_watch_wake_sedentary_minutes: Watch sedentary during wake hours
        - daily_iphone_wake_sedentary_minutes: iPhone sedentary during wake hours
    """
    # Non-wear minutes during wake hours (6am-10pm = minutes 360-1320)
    wake_nonwear = pl.col("nonwear_vector").list.slice(360, 960).list.sum()

    exprs = []

    for device, channel in [("watch", WATCH_STEPS), ("iphone", IPHONE_STEPS)]:
        raw_wake_sedentary = (
            pl.col("data")
            .arr.get(channel)
            .list.slice(360, 960)  # 6am to 10pm (16 hours)
            .list.eval(pl.element().fill_nan(None) == 0)  # NaN → null (excluded)
            .list.sum()
        )
        # Subtract non-wear minutes during wake hours
        exprs.append(
            pl.max_horizontal(raw_wake_sedentary - wake_nonwear, 0).alias(
                f"daily_{device}_wake_sedentary_minutes"
            )
        )

    return exprs


def _extract_hros_daily() -> list[pl.Expr]:
    """Extract daily Heart Rate Over Steps ratio (Mishra et al. 2020 inspired).

    HROS captures how much the heart rate elevates relative to physical
    activity. Elevated HROS (high HR at low activity) is a validated
    marker for infection, illness, and autonomic dysfunction.

    Computed from raw channel data (not from other daily columns):
    - Daytime window: 6am-10pm (minutes 360-1320)
    - HR: median of non-zero readings (bpm)
    - Steps: total steps / active minutes (steps per active minute)
    - HROS = median_hr / (step_rate + 1)

    Returns:
        List of Polars expressions:
        - daily_hros: HR-over-steps ratio for the day (null if no HR or no steps)
    """
    hr_arr = pl.col("data").arr.get(WATCH_HR)
    steps_arr = pl.col("data").arr.get(WATCH_STEPS)

    # Median HR (bpm) during daytime (6am-10pm), valid readings only
    daytime_hr = hr_arr.list.slice(360, 960)
    hr_median = (
        daytime_hr.list.eval(
            pl.element().filter(pl.element().is_finite() & (pl.element() > 0)).median()
        ).list.first()
        * 60.0
    )

    # Total steps and active minutes during same window
    daytime_steps = steps_arr.list.slice(360, 960).list.sum()
    active_mins = (
        steps_arr.list.slice(360, 960)
        .list.eval(pl.element().filter(pl.element() > 0).len())
        .list.first()
    )

    step_rate = daytime_steps / active_mins.replace(0, None)

    return [
        (hr_median / (step_rate + 1.0)).alias("daily_hros"),
    ]


# =============================================================================
# Physical Activity - Main Functions
# =============================================================================


def extract_physical_activity_daily() -> list[pl.Expr]:
    """Extract day-level physical activity metrics.

    Combines all physical activity sub-extractors into a comprehensive feature set.

    Sub-categories:
    - Active minutes (watch + iPhone)
    - Gait efficiency (distance/steps ratio)
    - HR-Activity coupling
    - Intensity zones (sedentary/light/moderate/vigorous)
    - Workout metrics (presence, types, cardio vs strength)
    - Device sync (watch vs phone agreement)
    - HR intensity (max HR, 80% threshold)
    - Sedentary patterns (wake-time sedentary)

    Returns:
        List of Polars expressions for day-level features.

    Example:
        >>> lf.with_columns(extract_physical_activity_daily())
    """
    return [
        *_extract_active_minutes_daily(),
        *_extract_gait_efficiency_daily(),
        *_extract_hr_coupling_daily(),
        *_extract_intensity_zones_daily(),
        *_extract_workout_metrics_daily(),
        *_extract_device_sync_daily(),
        *_extract_hr_intensity_daily(),
        *_extract_sedentary_bouts_daily(),
        *_extract_workout_hr_dynamics_daily(),
        *_extract_hros_daily(),
    ]


def aggregate_physical_activity_to_user() -> list[pl.Expr]:
    """Aggregate day-level physical activity to user-level features.

    Aggregation strategy:
    - Continuous metrics: median (P50) and IQR for robust central tendency
    - Binary flags: proportion of days (frequency)
    - Ratios: median across days

    Requires columns from `extract_physical_activity_daily()` to be computed first.

    Returns:
        List of Polars expressions for user-level aggregation.

    Example:
        >>> lf.with_columns(extract_physical_activity_daily()).group_by("user_id").agg(
        ...     aggregate_physical_activity_to_user()
        ... )
    """
    exprs = []

    # --- Active Minutes ---
    for device in ["watch", "iphone"]:
        col = f"daily_{device}_active_minutes"
        exprs.extend(
            [
                pl.col(col).median().alias(f"{device}_active_minutes_p50"),
                (pl.col(col).quantile(0.75) - pl.col(col).quantile(0.25)).alias(
                    f"{device}_active_minutes_iqr"
                ),
            ]
        )

    # --- Gait Efficiency ---
    for device in ["watch", "iphone"]:
        col = f"daily_{device}_gait_efficiency"
        exprs.append(pl.col(col).median().alias(f"{device}_gait_efficiency_p50"))

    # --- HR Coupling (time-based proxies) ---
    exprs.extend(
        [
            pl.col("daily_hr_daytime_median").median().alias("hr_daytime_p50"),
            pl.col("daily_hr_nighttime_median").median().alias("hr_nighttime_p50"),
            pl.col("daily_hr_p95").median().alias("hr_p95_p50"),
            # HR coupling ratio: daytime / nighttime (active vs rest proxy)
            (
                pl.col("daily_hr_daytime_median").median()
                / pl.col("daily_hr_nighttime_median").median().replace(0, None)
            ).alias("hr_daytime_nighttime_ratio"),
            # HR peak ratio: P95 / nighttime (workout intensity proxy)
            (
                pl.col("daily_hr_p95").median()
                / pl.col("daily_hr_nighttime_median").median().replace(0, None)
            ).alias("hr_peak_rest_ratio"),
        ]
    )

    # --- Intensity Zones (both devices) ---
    for device in ["watch", "iphone"]:
        for zone in ["sedentary", "light", "moderate", "vigorous"]:
            col = f"daily_{device}_{zone}_minutes"
            exprs.extend(
                [
                    pl.col(col).median().alias(f"{device}_{zone}_minutes_p50"),
                    (pl.col(col).quantile(0.75) - pl.col(col).quantile(0.25)).alias(
                        f"{device}_{zone}_minutes_iqr"
                    ),
                ]
            )

    # --- Workout Metrics ---
    exprs.extend(
        [
            # Workout duration
            pl.col("daily_workout_minutes").median().alias("workout_minutes_p50"),
            (
                pl.col("daily_workout_minutes").quantile(0.75)
                - pl.col("daily_workout_minutes").quantile(0.25)
            ).alias("workout_minutes_iqr"),
            # Workout density: proportion of days with any workout
            pl.col("daily_has_any_workout").mean().alias("workout_density"),
            # Workout specialization: median number of workout types per workout day
            pl.col("daily_workout_type_count")
            .filter(pl.col("daily_has_any_workout") == 1)
            .median()
            .alias("workout_specialization_p50"),
            # Total unique workout types ever used by this user
            # (sum of "ever did this workout" across all types)
            pl.sum_horizontal(
                [
                    (
                        pl.col(f"daily_has_{CHANNEL_NAMES[ch].replace('workout_', '')}").sum() > 0
                    ).cast(pl.Int8)
                    for ch in WORKOUT_CHANNELS
                ]
            ).alias("total_workout_types_ever"),
            # Cardio vs Strength ratio
            (
                pl.col("daily_cardio_minutes").sum()
                / (
                    pl.col("daily_cardio_minutes").sum() + pl.col("daily_strength_minutes").sum()
                ).replace(0, None)
            ).alias("cardio_strength_ratio"),
        ]
    )

    # Workout type frequencies
    workout_types = [CHANNEL_NAMES[ch].replace("workout_", "") for ch in WORKOUT_CHANNELS]
    for wtype in workout_types:
        exprs.append(pl.col(f"daily_has_{wtype}").mean().alias(f"workout_freq_{wtype}"))

    # --- Device Sync ---
    exprs.append(pl.col("daily_step_sync_ratio").median().alias("step_sync_ratio_p50"))

    # --- HR Intensity ---
    exprs.extend(
        [
            pl.col("daily_hr_max").median().alias("hr_max_p50"),
            pl.col("daily_hr_max").quantile(0.95).alias("hr_max_p95"),
            # Percent of days achieving 80% HR max
            pl.col("daily_hr_80pct_achieved").mean().alias("pct_days_hr_80pct_achieved"),
        ]
    )

    # --- Sedentary Patterns (both devices) ---
    for device in ["watch", "iphone"]:
        col = f"daily_{device}_wake_sedentary_minutes"
        exprs.extend(
            [
                pl.col(col).median().alias(f"{device}_wake_sedentary_minutes_p50"),
                (pl.col(col).quantile(0.75) - pl.col(col).quantile(0.25)).alias(
                    f"{device}_wake_sedentary_minutes_iqr"
                ),
            ]
        )

    # --- Workout HR Dynamics ---
    # Post-workout HR recovery (only on workout days, nulls are ignored)
    exprs.extend(
        [
            pl.col("daily_hr_at_workout_end").median().alias("hr_at_workout_end_p50"),
            pl.col("daily_hr_recovery_1min").median().alias("hr_recovery_1min_p50"),
            pl.col("daily_hr_recovery_5min").median().alias("hr_recovery_5min_p50"),
            pl.col("daily_hr_recovery_10min").median().alias("hr_recovery_10min_p50"),
            pl.col("daily_hr_recovery_ratio").median().alias("hr_recovery_ratio_p50"),
            # IQR of recovery for consistency
            (
                pl.col("daily_hr_recovery_5min").quantile(0.75)
                - pl.col("daily_hr_recovery_5min").quantile(0.25)
            ).alias("hr_recovery_5min_iqr"),
        ]
    )

    # HR activation at workout start
    exprs.extend(
        [
            pl.col("daily_hr_at_workout_start").median().alias("hr_at_workout_start_p50"),
            pl.col("daily_hr_activation_5min").median().alias("hr_activation_5min_p50"),
            pl.col("daily_hr_activation_10min").median().alias("hr_activation_10min_p50"),
            pl.col("daily_hr_activation_ratio").median().alias("hr_activation_ratio_p50"),
        ]
    )

    # --- HROS (Heart Rate Over Steps) ---
    exprs.extend(
        [
            pl.col("daily_hros").median().alias("hros_p50"),
            (pl.col("daily_hros").quantile(0.75) - pl.col("daily_hros").quantile(0.25)).alias(
                "hros_iqr"
            ),
        ]
    )

    return exprs


def aggregate_rhr_stability_to_user() -> list[pl.Expr]:
    """Aggregate Resting Heart Rate (RHR) stability to user level using RMSSD.

    RMSSD = Root Mean Square of Successive Differences of daily HR values.
    Captures day-to-day volatility in HR — distinct from IQR (overall spread)
    because RMSSD is sensitive to *rapid fluctuations* vs gradual drift.

    Clinical significance: High RMSSD of resting HR may indicate illness,
    overtraining, or inconsistent recovery patterns.

    Requires columns from ``extract_robust_baselines_daily()``:
    - daily_watch_hr_p5 (basal/resting HR)
    - daily_watch_hr_median (overall daily HR)

    Uses ``date`` column for chronological ordering of successive differences.

    Returns:
        List of Polars expressions:
        - rhr_p5_rmssd: RMSSD of daily P5 HR (resting HR volatility, bpm)
        - rhr_median_rmssd: RMSSD of daily median HR (overall HR volatility, bpm)
    """
    return [
        (pl.col("daily_watch_hr_p5").sort_by(pl.col("date")).diff().pow(2).mean().sqrt()).alias(
            "rhr_p5_rmssd"
        ),
        (pl.col("daily_watch_hr_median").sort_by(pl.col("date")).diff().pow(2).mean().sqrt()).alias(
            "rhr_median_rmssd"
        ),
    ]


# =============================================================================
# Sleep - Sub-functions
# =============================================================================

# Time windows for sleep timing detection
MORNING_START = 0  # Midnight
MORNING_END = 720  # Noon (12pm) - look for wake-up in this window
EVENING_START = 1080  # 6pm
EVENING_END = 1440  # Midnight - look for bedtime in this window

# Nighttime window for restless sleep detection
NIGHTTIME_START = 0  # Midnight
NIGHTTIME_END = 300  # 5am

# Pre-sleep activity window (3 hours before typical bedtime ~11pm)
PRESLEEP_START = 1200  # 8pm
PRESLEEP_END = 1380  # 11pm


def _extract_sleep_duration_daily() -> list[pl.Expr]:
    """Extract basic sleep duration metrics.

    Handles NaN values in sleep channels:
    - Sleep data is binary: 0 = awake, 1 = asleep
    - Days are either all-NaN (no tracking) or all-valid (0/1), never mixed
    - If ALL values are NaN (no sleep tracking for this day), returns null

    Returns:
        List of Polars expressions:
        - daily_has_sleep_data: 1 if user has sleep tracking for this day, 0 otherwise
        - daily_sleep_minutes: Total minutes asleep (null if no data)
        - daily_inbed_minutes: Total minutes in bed (null if no data)
        - daily_sleep_efficiency: Ratio of sleep to in-bed time (0-1)
    """
    sleep_arr = pl.col("data").arr.get(SLEEP_ASLEEP)
    inbed_arr = pl.col("data").arr.get(SLEEP_INBED)

    # Check if user has any valid (non-NaN) sleep data for this day
    has_data = (
        (sleep_arr.list.eval(~pl.element().is_nan()).list.sum() > 0)
        .cast(pl.Int8)
        .alias("daily_has_sleep_data")
    )

    # Sum with fill_nan(0), but return null if no data at all
    sleep_sum = (
        pl.when(sleep_arr.list.eval(~pl.element().is_nan()).list.sum() > 0)
        .then(sleep_arr.list.eval(pl.element().fill_nan(0)).list.sum())
        .otherwise(None)
    ).alias("daily_sleep_minutes")

    inbed_sum = (
        pl.when(inbed_arr.list.eval(~pl.element().is_nan()).list.sum() > 0)
        .then(inbed_arr.list.eval(pl.element().fill_nan(0)).list.sum())
        .otherwise(None)
    ).alias("daily_inbed_minutes")

    # Sleep efficiency: sleep / inbed (handle divide by zero and null)
    # Cap at 1.0 since efficiency >1 is impossible (data quality issue when inbed < asleep)
    raw_efficiency = (
        sleep_arr.list.eval(pl.element().fill_nan(0)).list.sum()
        / inbed_arr.list.eval(pl.element().fill_nan(0)).list.sum()
    )
    sleep_efficiency = (
        pl.when(
            (sleep_arr.list.eval(~pl.element().is_nan()).list.sum() > 0)
            & (inbed_arr.list.eval(pl.element().fill_nan(0)).list.sum() > 0)
        )
        .then(pl.when(raw_efficiency > 1.0).then(1.0).otherwise(raw_efficiency))
        .otherwise(None)
    ).alias("daily_sleep_efficiency")

    return [has_data, sleep_sum, inbed_sum, sleep_efficiency]


def _extract_sleep_timing_daily() -> list[pl.Expr]:
    """Extract sleep timing metrics (wake time, bedtime, midpoint).

    Sleep patterns span midnight, so we use heuristics:
    - Wake time: Last sleep→wake transition in morning hours (0-12pm)
    - Bedtime: First wake→sleep transition in evening hours (6pm-midnight)
    - Sleep midpoint: Average of all asleep minute indices

    Note: These are approximations. True sleep detection would require
    linking consecutive days. Returns null if no sleep data or no
    transitions detected in the expected windows.

    Returns:
        List of Polars expressions:
        - daily_wake_minute: Approximate wake time (minute of day, 0-720)
        - daily_bedtime_minute: Approximate bedtime (minute of day, 1080-1440)
        - daily_sleep_midpoint: Center of sleep mass (minute of day)
    """
    # Sleep channel is binary: 0 = awake, 1 = asleep
    # NaN indicates no sleep tracking for that day (entire array is NaN)
    # Days are never mixed (all-NaN or all-valid), so fill_nan(0) is safe
    sleep_arr = pl.col("data").arr.get(SLEEP_ASLEEP).list.eval(pl.element().fill_nan(0))

    # Check if any sleep data exists
    has_any_sleep = sleep_arr.list.sum() > 0

    # Wake time: Find the LAST minute that is asleep in morning hours (0-720)
    # The minute AFTER that is when they woke up
    morning_sleep = sleep_arr.list.slice(MORNING_START, MORNING_END)

    # Find indices where asleep == 1, then get max
    # Only add 1 if we found a valid wake time
    wake_minute = (
        pl.when(has_any_sleep)
        .then(
            morning_sleep.list.eval(
                pl.when(pl.element() == 1).then(pl.int_range(pl.len())).otherwise(None)
            ).list.max()
            + 1  # Wake is the minute after last sleep
        )
        .otherwise(None)
    ).alias("daily_wake_minute")

    # Bedtime: Find the FIRST minute that is asleep in evening hours (1080-1440)
    evening_sleep = sleep_arr.list.slice(EVENING_START, EVENING_END - EVENING_START)

    # Find first asleep minute in evening window
    bedtime_minute = (
        pl.when(has_any_sleep)
        .then(
            evening_sleep.list.eval(
                pl.when(pl.element() == 1).then(pl.int_range(pl.len())).otherwise(None)
            ).list.min()
            + EVENING_START  # Offset to get actual minute of day
        )
        .otherwise(None)
    ).alias("daily_bedtime_minute")

    # Sleep midpoint: (first_asleep + last_asleep) / 2
    # This gives the "center of mass" of sleep
    sleep_midpoint = (
        pl.when(has_any_sleep)
        .then(
            (
                sleep_arr.list.eval(
                    pl.when(pl.element() == 1).then(pl.int_range(pl.len())).otherwise(None)
                ).list.min()
                + sleep_arr.list.eval(
                    pl.when(pl.element() == 1).then(pl.int_range(pl.len())).otherwise(None)
                ).list.max()
            )
            / 2
        )
        .otherwise(None)
    ).alias("daily_sleep_midpoint")

    return [wake_minute, bedtime_minute, sleep_midpoint]


def _extract_nighttime_movement_daily() -> list[pl.Expr]:
    """Extract nighttime movement as a marker of restless sleep.

    Sums activity metrics between midnight and 5am (minutes 0-300).
    Movement during this window may indicate:
    - Restless sleep / sleep fragmentation
    - Nocturia (bathroom visits)
    - Sleep disorders

    Note: NaN values are treated as 0 (no activity recorded).

    Returns:
        List of Polars expressions:
        - daily_nighttime_watch_steps: Watch steps 00:00-05:00
        - daily_nighttime_iphone_steps: iPhone steps 00:00-05:00
        - daily_nighttime_energy: Active energy 00:00-05:00
    """
    return [
        # Watch steps during night (fill NaN with 0)
        (
            pl.col("data")
            .arr.get(WATCH_STEPS)
            .list.slice(NIGHTTIME_START, NIGHTTIME_END)
            .list.eval(pl.element().fill_nan(0))
            .list.sum()
            .alias("daily_nighttime_watch_steps")
        ),
        # iPhone steps during night (fill NaN with 0)
        (
            pl.col("data")
            .arr.get(IPHONE_STEPS)
            .list.slice(NIGHTTIME_START, NIGHTTIME_END)
            .list.eval(pl.element().fill_nan(0))
            .list.sum()
            .alias("daily_nighttime_iphone_steps")
        ),
        # Active energy during night (fill NaN with 0)
        (
            pl.col("data")
            .arr.get(WATCH_ENERGY)
            .list.slice(NIGHTTIME_START, NIGHTTIME_END)
            .list.eval(pl.element().fill_nan(0))
            .list.sum()
            .alias("daily_nighttime_energy")
        ),
    ]


def _extract_presleep_activity_daily() -> list[pl.Expr]:
    """Extract activity in the 3 hours before typical bedtime (8pm-11pm).

    Pre-sleep activity patterns may affect sleep quality:
    - High activity close to bedtime → delayed sleep onset
    - Low evening activity → better sleep initiation

    Note: Uses fixed 8pm-10pm window as proxy since exact bedtime
    detection requires cross-day analysis. NaN values are treated as 0.

    Returns:
        List of Polars expressions:
        - daily_presleep_watch_steps: Watch steps 8pm-10pm
        - daily_presleep_iphone_steps: iPhone steps 8pm-10pm
        - daily_presleep_energy: Active energy 8pm-10pm
        - daily_presleep_watch_steps_median: Median per-minute watch steps (intensity)
        - daily_presleep_iphone_steps_median: Median per-minute iPhone steps (intensity)
        - daily_presleep_energy_median: Median per-minute energy (intensity)
    """
    window_length = PRESLEEP_END - PRESLEEP_START  # 120 minutes

    return [
        # Total steps in pre-sleep window (fill NaN with 0)
        (
            pl.col("data")
            .arr.get(WATCH_STEPS)
            .list.slice(PRESLEEP_START, window_length)
            .list.eval(pl.element().fill_nan(0))
            .list.sum()
            .alias("daily_presleep_watch_steps")
        ),
        (
            pl.col("data")
            .arr.get(IPHONE_STEPS)
            .list.slice(PRESLEEP_START, window_length)
            .list.eval(pl.element().fill_nan(0))
            .list.sum()
            .alias("daily_presleep_iphone_steps")
        ),
        # Total energy in pre-sleep window (fill NaN with 0)
        (
            pl.col("data")
            .arr.get(WATCH_ENERGY)
            .list.slice(PRESLEEP_START, window_length)
            .list.eval(pl.element().fill_nan(0))
            .list.sum()
            .alias("daily_presleep_energy")
        ),
        # Median intensity (per-minute) in pre-sleep window
        # Note: list.median() ignores NaN by default
        (
            pl.col("data")
            .arr.get(WATCH_STEPS)
            .list.slice(PRESLEEP_START, window_length)
            .list.median()
            .alias("daily_presleep_watch_steps_median")
        ),
        (
            pl.col("data")
            .arr.get(IPHONE_STEPS)
            .list.slice(PRESLEEP_START, window_length)
            .list.median()
            .alias("daily_presleep_iphone_steps_median")
        ),
        (
            pl.col("data")
            .arr.get(WATCH_ENERGY)
            .list.slice(PRESLEEP_START, window_length)
            .list.median()
            .alias("daily_presleep_energy_median")
        ),
    ]


def _extract_weekend_flag_daily() -> list[pl.Expr]:
    """Extract weekend indicator from date column.

    Parses the date string to determine if it's a weekend day.
    Weekend = Saturday (6) or Sunday (7) in ISO weekday.

    Requires: 'date' column in YYYY-MM-DD format.

    Returns:
        List of Polars expressions:
        - daily_is_weekend: 1 if Saturday/Sunday, 0 otherwise
    """
    # Parse date string and extract weekday (1=Mon, 7=Sun)
    weekday = pl.col("date").str.to_date("%Y-%m-%d").dt.weekday()

    return [
        # Weekend = Saturday (6) or Sunday (7)
        ((weekday == 6) | (weekday == 7)).cast(pl.Int8).alias("daily_is_weekend")
    ]


# =============================================================================
# Sleep - Main Functions
# =============================================================================


def extract_sleep_daily() -> list[pl.Expr]:
    """Extract day-level sleep metrics.

    Combines all sleep sub-extractors:
    - Sleep duration (total sleep, in-bed, efficiency)
    - Sleep timing (wake time, bedtime, midpoint)
    - Nighttime movement (restless sleep markers)
    - Pre-sleep activity (activity before bed)
    - Weekend flag (for weekday/weekend comparisons)

    Returns:
        List of Polars expressions. Output columns include:
        - daily_sleep_minutes, daily_inbed_minutes, daily_sleep_efficiency
        - daily_wake_minute, daily_bedtime_minute, daily_sleep_midpoint
        - daily_nighttime_watch_steps, daily_nighttime_iphone_steps, daily_nighttime_energy
        - daily_presleep_watch_steps, daily_presleep_iphone_steps, daily_presleep_energy
        - daily_presleep_watch_steps_median, daily_presleep_energy_median
        - daily_is_weekend

    Example:
        >>> lf.with_columns(extract_sleep_daily())
    """
    return [
        *_extract_sleep_duration_daily(),
        *_extract_sleep_timing_daily(),
        *_extract_nighttime_movement_daily(),
        *_extract_presleep_activity_daily(),
        *_extract_weekend_flag_daily(),
    ]


def _aggregate_sleep_duration_to_user() -> list[pl.Expr]:
    """Aggregate sleep duration metrics to user level.

    Only considers days with valid sleep data (daily_has_sleep_data == 1).

    Returns:
        List of Polars expressions for sleep duration aggregation.
    """
    return [
        # Count days with sleep data
        pl.col("daily_has_sleep_data").sum().alias("n_days_with_sleep_data"),
        # Sleep duration - median and IQR (drop_nulls happens implicitly)
        pl.col("daily_sleep_minutes").median().alias("sleep_minutes_p50"),
        (
            pl.col("daily_sleep_minutes").quantile(0.75)
            - pl.col("daily_sleep_minutes").quantile(0.25)
        ).alias("sleep_minutes_iqr"),
        # In-bed duration
        pl.col("daily_inbed_minutes").median().alias("inbed_minutes_p50"),
        (
            pl.col("daily_inbed_minutes").quantile(0.75)
            - pl.col("daily_inbed_minutes").quantile(0.25)
        ).alias("inbed_minutes_iqr"),
        # Sleep efficiency
        pl.col("daily_sleep_efficiency").median().alias("sleep_efficiency_p50"),
        (
            pl.col("daily_sleep_efficiency").quantile(0.75)
            - pl.col("daily_sleep_efficiency").quantile(0.25)
        ).alias("sleep_efficiency_iqr"),
    ]


def _aggregate_sleep_timing_to_user() -> list[pl.Expr]:
    """Aggregate sleep timing metrics to user level with consistency measures.

    Consistency is measured using:
    - Median: typical timing
    - MAD (Median Absolute Deviation): robust variability measure
    - IQR: another robust variability measure

    Returns:
        List of Polars expressions for sleep timing aggregation.
    """
    exprs = []

    for metric, name in [
        ("daily_wake_minute", "wake"),
        ("daily_bedtime_minute", "bedtime"),
        ("daily_sleep_midpoint", "sleep_midpoint"),
    ]:
        # Median timing
        exprs.append(pl.col(metric).median().alias(f"{name}_p50"))

        # IQR for variability
        exprs.append(
            (pl.col(metric).quantile(0.75) - pl.col(metric).quantile(0.25)).alias(f"{name}_iqr")
        )

        # MAD (Median Absolute Deviation) = median(|x - median(x)|)
        # In Polars, we compute this as: median of absolute deviations
        # Note: This requires computing median first, which isn't directly expressible
        # in a single aggregation. We use std as a proxy, or compute MAD differently.
        # For now, we use a scaled IQR as MAD proxy: IQR * 0.7413 ≈ MAD for normal data
        exprs.append(
            ((pl.col(metric).quantile(0.75) - pl.col(metric).quantile(0.25)) * 0.7413).alias(
                f"{name}_mad_proxy"
            )
        )

    return exprs


def _aggregate_nighttime_movement_to_user() -> list[pl.Expr]:
    """Aggregate nighttime movement metrics to user level.

    Returns:
        List of Polars expressions for nighttime movement aggregation.
    """
    exprs = []

    for metric in [
        "daily_nighttime_watch_steps",
        "daily_nighttime_iphone_steps",
        "daily_nighttime_energy",
    ]:
        base_name = metric.replace("daily_", "")
        exprs.extend(
            [
                pl.col(metric).median().alias(f"{base_name}_p50"),
                (pl.col(metric).quantile(0.75) - pl.col(metric).quantile(0.25)).alias(
                    f"{base_name}_iqr"
                ),
            ]
        )

    return exprs


def _aggregate_presleep_activity_to_user() -> list[pl.Expr]:
    """Aggregate pre-sleep activity metrics to user level.

    Returns:
        List of Polars expressions for pre-sleep activity aggregation.
    """
    exprs = []

    # Totals
    for metric in [
        "daily_presleep_watch_steps",
        "daily_presleep_iphone_steps",
        "daily_presleep_energy",
    ]:
        base_name = metric.replace("daily_", "")
        exprs.append(pl.col(metric).median().alias(f"{base_name}_p50"))

    # Intensities (medians of daily medians)
    for metric in [
        "daily_presleep_watch_steps_median",
        "daily_presleep_iphone_steps_median",
        "daily_presleep_energy_median",
    ]:
        base_name = metric.replace("daily_", "")
        exprs.append(pl.col(metric).median().alias(f"{base_name}_p50"))

    return exprs


def _aggregate_weekend_weekday_sleep_to_user() -> list[pl.Expr]:
    """Aggregate weekend vs weekday sleep patterns to user level.

    Computes:
    - Sleep duration ratio (weekend / weekday)
    - Bedtime shift (weekend - weekday)
    - Wake time shift (weekend - weekday)

    Social jet lag = difference in sleep timing between weekdays and weekends.

    Returns:
        List of Polars expressions for weekend/weekday comparison.
    """
    # Filter expressions for weekend vs weekday
    is_weekend = pl.col("daily_is_weekend") == 1
    is_weekday = pl.col("daily_is_weekend") == 0

    return [
        # Weekend sleep duration (median)
        pl.col("daily_sleep_minutes")
        .filter(is_weekend)
        .median()
        .alias("sleep_minutes_weekend_p50"),
        # Weekday sleep duration (median)
        pl.col("daily_sleep_minutes")
        .filter(is_weekday)
        .median()
        .alias("sleep_minutes_weekday_p50"),
        # Sleep duration ratio: weekend / weekday (>1 = more sleep on weekends)
        (
            pl.col("daily_sleep_minutes").filter(is_weekend).median()
            / pl.col("daily_sleep_minutes").filter(is_weekday).median().replace(0, None)
        ).alias("sleep_weekend_weekday_ratio"),
        # Bedtime comparison (social jet lag - bedtime component)
        pl.col("daily_bedtime_minute").filter(is_weekend).median().alias("bedtime_weekend_p50"),
        pl.col("daily_bedtime_minute").filter(is_weekday).median().alias("bedtime_weekday_p50"),
        # Bedtime shift: weekend - weekday (positive = later bedtime on weekends)
        (
            pl.col("daily_bedtime_minute").filter(is_weekend).median()
            - pl.col("daily_bedtime_minute").filter(is_weekday).median()
        ).alias("bedtime_weekend_shift"),
        # Wake time comparison (social jet lag - wake component)
        pl.col("daily_wake_minute").filter(is_weekend).median().alias("wake_weekend_p50"),
        pl.col("daily_wake_minute").filter(is_weekday).median().alias("wake_weekday_p50"),
        # Wake shift: weekend - weekday (positive = later wake on weekends)
        (
            pl.col("daily_wake_minute").filter(is_weekend).median()
            - pl.col("daily_wake_minute").filter(is_weekday).median()
        ).alias("wake_weekend_shift"),
        # Social jet lag: shift in sleep midpoint between weekday and weekend
        # Absolute difference captures magnitude regardless of direction
        (
            (
                pl.col("daily_sleep_midpoint").filter(is_weekend).median()
                - pl.col("daily_sleep_midpoint").filter(is_weekday).median()
            ).abs()
        ).alias("social_jet_lag"),
        # Count of weekend vs weekday days (for data quality check)
        is_weekend.sum().alias("n_weekend_days"),
        is_weekday.sum().alias("n_weekday_days"),
    ]


def aggregate_sleep_to_user() -> list[pl.Expr]:
    """Aggregate day-level sleep to user-level features.

    Combines all sleep aggregation sub-functions:
    - Sleep duration (median, IQR)
    - Sleep timing consistency (median, IQR, MAD proxy)
    - Nighttime movement (restless sleep markers)
    - Pre-sleep activity patterns
    - Weekend vs weekday comparisons (social jet lag)

    Requires columns from `extract_sleep_daily()` to be computed first.

    Returns:
        List of Polars expressions for user-level aggregation.

    Example:
        >>> lf.with_columns(extract_sleep_daily()).group_by("user_id").agg(
        ...     aggregate_sleep_to_user()
        ... )
    """
    return [
        *_aggregate_sleep_duration_to_user(),
        *_aggregate_sleep_timing_to_user(),
        *_aggregate_nighttime_movement_to_user(),
        *_aggregate_presleep_activity_to_user(),
        *_aggregate_weekend_weekday_sleep_to_user(),
    ]


# =============================================================================
# Circadian - Helpers
# =============================================================================

# Signals used for circadian features: (channel_index, signal_name)
_CIRCADIAN_SIGNALS = [
    (WATCH_STEPS, "watch_steps"),
    (WATCH_ENERGY, "watch_energy"),
    (IPHONE_STEPS, "iphone_steps"),
]


def _extract_time_window_sums(channel: int, signal_name: str) -> list[pl.Expr]:
    """Extract 4 time-window activity sums for a single channel.

    Windows:
    - Morning: 6am-12pm (minutes 360-720)
    - Afternoon: 12pm-6pm (minutes 720-1080)
    - Evening: 6pm-12am (minutes 1080-1440)
    - Night: 12am-6am (minutes 0-360)

    Args:
        channel: Channel index (0-18).
        signal_name: Signal name for column naming (e.g., "watch_steps").

    Returns:
        List of 4 Polars expressions for time-window sums.
    """
    arr = pl.col("data").arr.get(channel)
    return [
        arr.list.slice(360, 360).list.sum().alias(f"daily_{signal_name}_morning"),
        arr.list.slice(720, 360).list.sum().alias(f"daily_{signal_name}_afternoon"),
        arr.list.slice(1080, 360).list.sum().alias(f"daily_{signal_name}_evening"),
        arr.list.slice(0, 360).list.sum().alias(f"daily_{signal_name}_night"),
    ]


def _extract_hourly_bins_daily(channel: int, signal_name: str) -> list[pl.Expr]:
    """Extract 24 hourly activity bins for a single channel.

    Bins the 1440-minute array into 24 hourly sums (minutes 0-59 → hour 0,
    minutes 60-119 → hour 1, etc.).

    These bins are used downstream for:
    - Acrophase (argmax of hourly bins)
    - Interdaily Stability (IS) at user level
    - Intradaily Variability (IV)

    Args:
        channel: Channel index (0-18).
        signal_name: Signal name for column naming (e.g., "watch_steps").

    Returns:
        List of 24 Polars expressions, one per hour.
    """
    arr = pl.col("data").arr.get(channel)
    return [
        arr.list.slice(h * 60, 60).list.sum().alias(f"daily_hour_{h:02d}_{signal_name}")
        for h in range(24)
    ]


def _extract_acrophase_daily(channel: int, signal_name: str) -> pl.Expr:
    """Extract daily acrophase (peak activity hour, 0-23).

    Bins minute-level data into 24 hourly sums, then finds the hour index
    with maximum activity using argmax. Returns null for days with zero total
    activity (argmax would spuriously return 0 = midnight).

    Args:
        channel: Channel index (0-18).
        signal_name: Signal name for column naming (e.g., "watch_steps").

    Returns:
        Single Polars expression producing the peak hour (0-23) per day,
        or null when total activity is zero.
    """
    arr = pl.col("data").arr.get(channel)
    hourly_sums = [
        arr.list.slice(h * 60, 60).list.eval(pl.element().fill_nan(None)).list.sum()
        for h in range(24)
    ]
    total = pl.sum_horizontal(hourly_sums)
    return (
        pl.when(total.is_not_null() & (total > 0))
        .then(pl.concat_list(hourly_sums).list.arg_max())
        .otherwise(None)
        .alias(f"daily_acrophase_{signal_name}")
    )


# =============================================================================
# Circadian - Main Daily Extraction
# =============================================================================


def extract_circadian_daily() -> list[pl.Expr]:
    """Extract day-level circadian rhythm metrics.

    For each of three activity signals (watch steps, watch energy, iPhone steps),
    computes:

    **Time-window sums** (12 columns):
        Activity totals for morning (6am-12pm), afternoon (12pm-6pm),
        evening (6pm-12am), and night (12am-6am).

    **Hourly bins** (72 columns):
        Activity sum per clock hour (0-23). Used downstream for Interdaily
        Stability (IS) at user level.

    **Acrophase** (3 columns):
        Peak activity hour (0-23) via argmax of hourly bins.

    **Intradaily Variability** (3 columns):
        Within-day activity fragmentation from hourly bins.

    Returns:
        List of ~90 Polars expressions. Output column naming:
        - ``daily_{signal}_{window}`` for time windows
        - ``daily_hour_{HH}_{signal}`` for hourly bins
        - ``daily_acrophase_{signal}`` for peak hour
        - ``daily_iv_{signal}`` for intradaily variability

    Example:
        >>> lf.with_columns(extract_circadian_daily())
    """
    exprs: list[pl.Expr] = []

    for channel, signal_name in _CIRCADIAN_SIGNALS:
        exprs.extend(_extract_time_window_sums(channel, signal_name))
        exprs.extend(_extract_hourly_bins_daily(channel, signal_name))
        exprs.append(_extract_acrophase_daily(channel, signal_name))

    # IV depends on hourly bin columns, but Polars with_columns computes all
    # expressions in the same call from the original columns.  Hourly bins are
    # derived from "data" (original), so they are available as sibling
    # expressions.  However, IV references the *output* of hourly bins, which
    # won't exist yet within the same with_columns.  We must therefore split
    # this into a second with_columns call OR compute IV inline from raw data.
    #
    # Solution: compute IV directly from raw minute data (same slice pattern)
    # so it doesn't depend on the hourly bin columns.
    for channel, signal_name in _CIRCADIAN_SIGNALS:
        arr = pl.col("data").arr.get(channel)
        # fill_nan(None) so all-NaN channels produce null sums (not NaN),
        # which propagate correctly through arithmetic and the denominator guard.
        hour_vals = [
            arr.list.slice(h * 60, 60).list.eval(pl.element().fill_nan(None)).list.sum()
            for h in range(24)
        ]
        N = 24

        sq_diffs = [(hour_vals[h] - hour_vals[h - 1]).pow(2) for h in range(1, N)]
        numerator = pl.sum_horizontal(sq_diffs)

        hourly_mean = pl.mean_horizontal(hour_vals)
        sq_devs = [(val - hourly_mean).pow(2) for val in hour_vals]
        denominator = pl.sum_horizontal(sq_devs)

        exprs.append(
            pl.when(denominator.is_finite() & (denominator > 0))
            .then((N * numerator) / ((N - 1) * denominator))
            .otherwise(None)
            .alias(f"daily_iv_{signal_name}")
        )

    return exprs


# =============================================================================
# Circadian - User-Level Aggregation Helpers
# =============================================================================


def _aggregate_time_windows_to_user() -> list[pl.Expr]:
    """Aggregate time-window sums to user level.

    For each signal, computes:
    - Median activity per time window (4 × 3 = 12 features)
    - Chronotype morning ratio: morning / (morning + evening) (3 features)
    - Day/night ratio: (morning + afternoon + evening) / night (3 features)

    Returns:
        List of ~18 Polars expressions.
    """
    exprs: list[pl.Expr] = []

    for _channel, signal_name in _CIRCADIAN_SIGNALS:
        morning = pl.col(f"daily_{signal_name}_morning")
        afternoon = pl.col(f"daily_{signal_name}_afternoon")
        evening = pl.col(f"daily_{signal_name}_evening")
        night = pl.col(f"daily_{signal_name}_night")

        # Median activity per time window
        exprs.extend(
            [
                morning.median().alias(f"{signal_name}_morning_p50"),
                afternoon.median().alias(f"{signal_name}_afternoon_p50"),
                evening.median().alias(f"{signal_name}_evening_p50"),
                night.median().alias(f"{signal_name}_night_p50"),
            ]
        )

        # Chronotype ratio: morning / (morning + evening)
        exprs.append(
            (morning.median() / (morning.median() + evening.median()).replace(0, None)).alias(
                f"chronotype_morning_ratio_{signal_name}"
            )
        )

        # Day/night ratio: (morning + afternoon + evening) / night
        exprs.append(
            (
                (morning.median() + afternoon.median() + evening.median())
                / night.median().replace(0, None)
            ).alias(f"activity_day_night_ratio_{signal_name}")
        )

    return exprs


def _aggregate_acrophase_to_user() -> list[pl.Expr]:
    """Aggregate daily acrophase to user level.

    For each signal, computes:
    - Median acrophase (typical peak hour) (3 features)
    - Acrophase spread — consistency of peak timing (3 features)
      Computed as IQR/2 of daily acrophase values (robust dispersion proxy).
    - Social jetlag — |weekend median acrophase - weekday median acrophase| (3 features)

    Requires ``daily_is_weekend`` column from sleep extraction.

    Returns:
        List of ~9 Polars expressions.
    """
    is_weekend = pl.col("daily_is_weekend") == 1
    is_weekday = pl.col("daily_is_weekend") == 0

    exprs: list[pl.Expr] = []

    for _channel, signal_name in _CIRCADIAN_SIGNALS:
        acro = pl.col(f"daily_acrophase_{signal_name}")

        # Median acrophase (typical peak hour)
        exprs.append(acro.median().alias(f"acrophase_{signal_name}_p50"))

        # Acrophase spread — "Circadian Drift" marker
        # IQR/2 as a robust dispersion proxy: low = consistent peak timing,
        # high = erratic peak timing across days.
        exprs.append(
            ((acro.quantile(0.75) - acro.quantile(0.25)) * 0.5).alias(
                f"acrophase_{signal_name}_spread"
            )
        )

        # Social jetlag: |weekend acrophase - weekday acrophase|
        exprs.append(
            (acro.filter(is_weekend).median() - acro.filter(is_weekday).median())
            .abs()
            .alias(f"social_jetlag_{signal_name}")
        )

    return exprs


def _aggregate_is_to_user() -> list[pl.Expr]:
    """Compute Interdaily Stability (IS) at user level.

    IS measures how consistently the 24-hour activity pattern repeats from day
    to day.  Range 0-1; higher = more consistent daily rhythm.

    Formula (Van Someren et al., 1999):
        IS = (D * sum_h((x_h_mean - x_grand)^2))
             / sum_{all d,h}((x_{h,d} - x_grand)^2)

    where:
    - D = number of days for this user
    - x_h_mean = mean activity at hour h across all days
    - x_grand = overall mean across all hours and days
    - Sum in denominator is over every (day, hour) pair

    IS is equivalent to the R-squared of a one-way ANOVA with hour-of-day
    as the factor.  Range: 0 (random) to 1 (identical days).

    Uses the 72 hourly bin columns from ``extract_circadian_daily()``.

    Returns:
        List of 3 Polars expressions (one per signal).
    """
    exprs: list[pl.Expr] = []

    for _channel, signal_name in _CIRCADIAN_SIGNALS:
        hour_cols = [pl.col(f"daily_hour_{h:02d}_{signal_name}") for h in range(24)]

        # Number of days (rows) for this user
        n_days = hour_cols[0].count()

        # Mean activity at each hour across all days (24 scalars)
        hourly_means = [col.mean() for col in hour_cols]

        # Grand mean: mean of the 24 hourly means
        # (valid because every hour has the same number of days)
        grand_mean = pl.mean_horizontal(hourly_means)

        # Numerator: D * sum_h((hourly_mean_h - grand_mean)^2)
        # This is the "between-hours" sum of squares, scaled by D
        numerator = n_days * pl.sum_horizontal([(hm - grand_mean).pow(2) for hm in hourly_means])

        # Denominator: total sum of squares over all (day, hour) pairs
        # In agg context, ((col - grand_mean).pow(2)).sum() sums across days
        # for that hour column.
        denominator = pl.sum_horizontal([((col - grand_mean).pow(2)).sum() for col in hour_cols])

        exprs.append(
            pl.when(denominator > 0)
            .then(numerator / denominator)
            .otherwise(None)
            .alias(f"is_{signal_name}")
        )

    return exprs


def _aggregate_iv_to_user() -> list[pl.Expr]:
    """Aggregate daily Intradaily Variability (IV) to user level.

    Takes the median of per-day IV values for each signal.

    Returns:
        List of 3 Polars expressions.
    """
    return [
        pl.col(f"daily_iv_{signal_name}").median().alias(f"iv_{signal_name}_p50")
        for _channel, signal_name in _CIRCADIAN_SIGNALS
    ]


def _aggregate_npcra_to_user() -> list[pl.Expr]:
    """Compute Non-Parametric Circadian Rhythm Analysis (NPCRA) metrics.

    Van Someren et al. (1999): L5, M10, Relative Amplitude.

    For each signal in ``_CIRCADIAN_SIGNALS``, computes from hourly bin columns:
    - **L5**: Mean activity during the least active 5 consecutive hours
    - **L5 onset**: Start hour of the L5 window (0-23)
    - **M10**: Mean activity during the most active 10 consecutive hours
    - **M10 onset**: Start hour of the M10 window (0-23)
    - **RA**: Relative amplitude = (M10 - L5) / (M10 + L5)

    Uses circular wrap-around (hour 23 wraps to hour 0) to handle
    L5 windows that cross midnight.

    Requires hourly bin columns from ``extract_circadian_daily()``.

    Returns:
        List of 15 Polars expressions (5 per signal × 3 signals).
    """
    exprs = []
    for _channel, signal_name in _CIRCADIAN_SIGNALS:
        hourly_means = [pl.col(f"daily_hour_{h:02d}_{signal_name}").mean() for h in range(24)]

        # 24 rolling sums of 5 consecutive hours (circular wrap-around)
        rolling_5 = [
            pl.sum_horizontal([hourly_means[(start + i) % 24] for i in range(5)])
            for start in range(24)
        ]

        # 24 rolling sums of 10 consecutive hours (circular wrap-around)
        rolling_10 = [
            pl.sum_horizontal([hourly_means[(start + i) % 24] for i in range(10)])
            for start in range(24)
        ]

        l5_list = pl.concat_list(rolling_5)
        m10_list = pl.concat_list(rolling_10)

        l5_value = l5_list.list.min() / 5.0
        m10_value = m10_list.list.max() / 10.0

        exprs.extend(
            [
                l5_value.alias(f"l5_{signal_name}"),
                l5_list.list.arg_min().alias(f"l5_onset_{signal_name}"),
                m10_value.alias(f"m10_{signal_name}"),
                m10_list.list.arg_max().alias(f"m10_onset_{signal_name}"),
                ((m10_value - l5_value) / (m10_value + l5_value).replace(0, None)).alias(
                    f"ra_{signal_name}"
                ),
            ]
        )

    return exprs


# =============================================================================
# Circadian - Main User Aggregation
# =============================================================================


def aggregate_circadian_to_user() -> list[pl.Expr]:
    """Aggregate day-level circadian metrics to user-level features.

    Combines all circadian aggregation sub-functions:

    **Time-window medians & ratios** (~18 features):
        Median activity per time window, chronotype morning ratio,
        day/night activity ratio — for each of 3 signals.

    **Acrophase** (~9 features):
        Median peak hour, MAD (circadian drift), social jetlag
        (weekend vs weekday phase shift) — for each of 3 signals.

    **Interdaily Stability** (3 features):
        Day-to-day consistency of the 24h activity pattern.

    **Intradaily Variability** (3 features):
        Median within-day activity fragmentation.

    Requires columns from ``extract_circadian_daily()`` and
    ``daily_is_weekend`` from ``extract_sleep_daily()``.

    Returns:
        List of ~33 Polars expressions for user-level aggregation.

    Example:
        >>> lf.with_columns(
        ...     extract_sleep_daily() + extract_circadian_daily()
        ... ).group_by("user_id").agg(
        ...     aggregate_circadian_to_user()
        ... )
    """
    return [
        *_aggregate_time_windows_to_user(),
        *_aggregate_acrophase_to_user(),
        *_aggregate_is_to_user(),
        *_aggregate_iv_to_user(),
        *_aggregate_npcra_to_user(),
    ]


# =============================================================================
# Convenience Functions
# =============================================================================


def get_all_daily_extractors() -> list[pl.Expr]:
    """Get all day-level feature extraction expressions.

    Combines all daily extractors into a single list.

    Returns:
        List of all day-level Polars expressions.

    Example:
        >>> lf.with_columns(get_all_daily_extractors())
    """
    return [
        *extract_robust_baselines_daily(),
        *extract_physical_activity_daily(),
        *extract_sleep_daily(),
        *extract_circadian_daily(),
    ]


def get_all_user_aggregators() -> list[pl.Expr]:
    """Get all user-level aggregation expressions.

    Combines all user aggregators into a single list.
    Assumes day-level features have been computed.

    Returns:
        List of all user-level Polars expressions.

    Example:
        >>> lf.with_columns(get_all_daily_extractors()).group_by("user_id").agg(
        ...     get_all_user_aggregators()
        ... )
    """
    return [
        *aggregate_robust_baselines_to_user(),
        *aggregate_physical_activity_to_user(),
        *aggregate_rhr_stability_to_user(),
        *aggregate_sleep_to_user(),
        *aggregate_circadian_to_user(),
    ]
