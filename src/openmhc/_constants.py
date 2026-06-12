"""Public constants for the MHC-Benchmark API."""

SENSOR_CHANNELS: list[str] = [
    "iphone_steps",
    "iphone_distance",
    "iphone_flights",
    "watch_steps",
    "watch_distance",
    "watch_hr",
    "watch_energy",
    "sleep_asleep",
    "sleep_inbed",
    "workout_walking",
    "workout_cycling",
    "workout_running",
    "workout_other",
    "workout_mixed_cardio",
    "workout_strength",
    "workout_elliptical",
    "workout_hiit",
    "workout_functional",
    "workout_yoga",
]
"""Ordered list of 19 sensor channel names matching column order in tensors."""

MASKING_SCENARIOS: list[str] = [
    "random_noise",
    "temporal_slice",
    "signal_slice",
    "sleep_gap",
    "workout_gap",
    "intensity_failure",
]
"""All 6 imputation masking scenario names."""

BENCHMARK_TASKS: list[str] = [
    "Atrial fibrillation (Afib)", "BMI_categories", "BMI_values", "BiologicalSex", "CAD",
    "Cerebrovascular Disease", "Congenital Heart", "Diabetes", "GoSleepTime_categories", "Hdl",
    "Heart Failure or CHF", "Hypertension", "Ldl", "PH", "Peripheral/Systemic Vascular Disease",
    "SystolicBloodPressure", "TotalCholesterol", "WakeUpTime_categories", "WeightKilograms", "age",
    "blood_pressure_categories", "cardiovascular_disease", "feel_worthwhile1", "feel_worthwhile2",
    "feel_worthwhile3", "feel_worthwhile4", "framingham_risk", "satisfiedwith_life",
    "sleep_diagnosis1", "sleep_time_categories", "vigorous_act", "work",
]
"""The 32 benchmark prediction tasks (``evaluate_prediction(tasks="all")`` runs these)."""
