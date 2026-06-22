"""Channel definitions and configuration for MHC wearable data.

This module defines constants for the 19 time-series channels in the MHC dataset,
along with helper mappings for feature naming and channel groupings.
"""

from __future__ import annotations

# Minutes per day (1440 = 24 hours × 60 minutes)
MINUTES_PER_DAY = 1440

# =============================================================================
# Channel Indices
# =============================================================================

# iPhone Metrics (Channels 0-2)
IPHONE_STEPS = 0
IPHONE_DISTANCE = 1
IPHONE_FLIGHTS = 2

# Apple Watch Metrics (Channels 3-6)
WATCH_STEPS = 3
WATCH_DISTANCE = 4
WATCH_HR = 5  # Stored as beats/second, multiply by 60 for bpm
WATCH_ENERGY = 6  # Stored as cal/min

# Sleep Metrics (Channels 7-8)
SLEEP_ASLEEP = 7
SLEEP_INBED = 8

# Workout Types (Channels 9-18) - All binary
WORKOUT_WALKING = 9
WORKOUT_CYCLING = 10
WORKOUT_RUNNING = 11
WORKOUT_OTHER = 12
WORKOUT_MIXED_CARDIO = 13
WORKOUT_STRENGTH = 14
WORKOUT_ELLIPTICAL = 15
WORKOUT_HIIT = 16
WORKOUT_FUNCTIONAL = 17
WORKOUT_YOGA = 18

# =============================================================================
# Channel Groupings
# =============================================================================

# Continuous channels (activity metrics with numeric values)
CONTINUOUS_CHANNELS = [
    IPHONE_STEPS,
    IPHONE_DISTANCE,
    IPHONE_FLIGHTS,
    WATCH_STEPS,
    WATCH_DISTANCE,
    WATCH_HR,
    WATCH_ENERGY,
]

# Sleep channels (binary)
SLEEP_CHANNELS = [SLEEP_ASLEEP, SLEEP_INBED]

# Workout channels (binary)
WORKOUT_CHANNELS = [
    WORKOUT_WALKING,
    WORKOUT_CYCLING,
    WORKOUT_RUNNING,
    WORKOUT_OTHER,
    WORKOUT_MIXED_CARDIO,
    WORKOUT_STRENGTH,
    WORKOUT_ELLIPTICAL,
    WORKOUT_HIIT,
    WORKOUT_FUNCTIONAL,
    WORKOUT_YOGA,
]

# =============================================================================
# Channel Names (for feature naming)
# =============================================================================

CHANNEL_NAMES = {
    IPHONE_STEPS: "iphone_steps",
    IPHONE_DISTANCE: "iphone_distance",
    IPHONE_FLIGHTS: "iphone_flights",
    WATCH_STEPS: "watch_steps",
    WATCH_DISTANCE: "watch_distance",
    WATCH_HR: "watch_hr",
    WATCH_ENERGY: "watch_energy",
    SLEEP_ASLEEP: "sleep_asleep",
    SLEEP_INBED: "sleep_inbed",
    WORKOUT_WALKING: "workout_walking",
    WORKOUT_CYCLING: "workout_cycling",
    WORKOUT_RUNNING: "workout_running",
    WORKOUT_OTHER: "workout_other",
    WORKOUT_MIXED_CARDIO: "workout_mixed_cardio",
    WORKOUT_STRENGTH: "workout_strength",
    WORKOUT_ELLIPTICAL: "workout_elliptical",
    WORKOUT_HIIT: "workout_hiit",
    WORKOUT_FUNCTIONAL: "workout_functional",
    WORKOUT_YOGA: "workout_yoga",
}

# Full channel info with units (for documentation/validation)
CHANNEL_INFO = {
    IPHONE_STEPS: {"name": "iphone_steps", "unit": "steps/min", "type": "continuous"},
    IPHONE_DISTANCE: {"name": "iphone_distance", "unit": "meters/min", "type": "continuous"},
    IPHONE_FLIGHTS: {"name": "iphone_flights", "unit": "count", "type": "continuous"},
    WATCH_STEPS: {"name": "watch_steps", "unit": "steps/min", "type": "continuous"},
    WATCH_DISTANCE: {"name": "watch_distance", "unit": "meters/min", "type": "continuous"},
    WATCH_HR: {"name": "watch_hr", "unit": "beats/sec (×60 for bpm)", "type": "continuous"},
    WATCH_ENERGY: {"name": "watch_energy", "unit": "cal/min", "type": "continuous"},
    SLEEP_ASLEEP: {"name": "sleep_asleep", "unit": "binary", "type": "binary"},
    SLEEP_INBED: {"name": "sleep_inbed", "unit": "binary", "type": "binary"},
    WORKOUT_WALKING: {"name": "workout_walking", "unit": "binary", "type": "binary"},
    WORKOUT_CYCLING: {"name": "workout_cycling", "unit": "binary", "type": "binary"},
    WORKOUT_RUNNING: {"name": "workout_running", "unit": "binary", "type": "binary"},
    WORKOUT_OTHER: {"name": "workout_other", "unit": "binary", "type": "binary"},
    WORKOUT_MIXED_CARDIO: {"name": "workout_mixed_cardio", "unit": "binary", "type": "binary"},
    WORKOUT_STRENGTH: {"name": "workout_strength", "unit": "binary", "type": "binary"},
    WORKOUT_ELLIPTICAL: {"name": "workout_elliptical", "unit": "binary", "type": "binary"},
    WORKOUT_HIIT: {"name": "workout_hiit", "unit": "binary", "type": "binary"},
    WORKOUT_FUNCTIONAL: {"name": "workout_functional", "unit": "binary", "type": "binary"},
    WORKOUT_YOGA: {"name": "workout_yoga", "unit": "binary", "type": "binary"},
}
