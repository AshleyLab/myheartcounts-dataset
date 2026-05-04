"""Canonical channel definitions and Hugging Face dataset schema for HealthKit daily data."""

from __future__ import annotations

import datasets as hf_ds

CHANNEL_NAMES = [
    "hk_iphone:HKQuantityTypeIdentifierStepCount",
    "hk_iphone:HKQuantityTypeIdentifierDistanceWalkingRunning",
    "hk_iphone:HKQuantityTypeIdentifierFlightsClimbed",
    "hk_watch:HKQuantityTypeIdentifierStepCount",
    "hk_watch:HKQuantityTypeIdentifierDistanceWalkingRunning",
    "hk_watch:HKQuantityTypeIdentifierHeartRate",
    "hk_watch:HKQuantityTypeIdentifierActiveEnergyBurned",
    "sleep:asleep",
    "sleep:inbed",
    "workout:HKWorkoutActivityTypeWalking",
    "workout:HKWorkoutActivityTypeCycling",
    "workout:HKWorkoutActivityTypeRunning",
    "workout:HKWorkoutActivityTypeOther",
    "workout:HKWorkoutActivityTypeMixedMetabolicCardioTraining",
    "workout:HKWorkoutActivityTypeTraditionalStrengthTraining",
    "workout:HKWorkoutActivityTypeElliptical",
    "workout:HKWorkoutActivityTypeHighIntensityIntervalTraining",
    "workout:HKWorkoutActivityTypeFunctionalStrengthTraining",
    "workout:HKWorkoutActivityTypeYoga",
]

CHANNEL_UNITS = [
    "steps/min",
    "m/min",
    "count/min",
    "steps/min",
    "m/min",
    "bpm",
    "cal/min",
    "binary",
    "binary",
    "binary",
    "binary",
    "binary",
    "binary",
    "binary",
    "binary",
    "binary",
    "binary",
    "binary",
    "binary",
]

N_CHANNELS = len(CHANNEL_NAMES)
MINUTES_PER_DAY = 1440
FEATURE_DTYPE = "float32"

CONTINUOUS_CHANNEL_INDICES = list(range(7))  # Channels 0-6
BINARY_CHANNEL_INDICES = list(range(7, 19))  # Channels 7-18

DEFAULT_VARIANCE_THRESHOLDS: dict[int, float] = {
    0: 1.0,  # iPhone Steps
    1: 1.0,  # iPhone Distance
    3: 1.0,  # Watch Steps
    4: 1.0,  # Watch Distance
    5: 0.0001,  # Watch Heart Rate
    6: 1.0,  # Watch Active Energy
}


def hf_features() -> hf_ds.Features:
    """Return the Arrow schema for one daily example."""
    return hf_ds.Features(
        {
            "values": hf_ds.Array2D(
                shape=(N_CHANNELS, MINUTES_PER_DAY),
                dtype=FEATURE_DTYPE,
            ),
            "user_id": hf_ds.Value("string"),
            "date": hf_ds.Value("string"),
            "has_any_data": hf_ds.Sequence(hf_ds.Value("bool")),
            "has_any_data_shape": hf_ds.Sequence(hf_ds.Value("int32")),
            "minutes_nonzero_or_nan": hf_ds.Sequence(hf_ds.Value("int32")),
            "minutes_nonzero_or_nan_shape": hf_ds.Sequence(hf_ds.Value("int32")),
            "channel_names": hf_ds.Sequence(hf_ds.Value("string")),
            "channel_units": hf_ds.Sequence(hf_ds.Value("string")),
            "total_nonwear_minutes": hf_ds.Value("float32"),
            "channel_variance": hf_ds.Sequence(hf_ds.Value("float32")),
        }
    )
