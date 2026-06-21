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

# Human-readable display names for the benchmark tasks: the stable internal code (the
# dataset's raw survey identifier — the key used in results CSVs, the API, and the
# label metadata) -> the label used in the paper and leaderboard. Internal codes stay
# the keys; this is a presentation layer applied at render. The well-being labels are
# sourced from the ONS wellbeing survey question text in
# ``data/labels/survey_documentation/wellbeing/`` (feel_worthwhile1 = "things you do
# are worthwhile", 2 = "how about happy?", 3 = "worried?", 4 = "depressed?"). Note
# feel_worthwhile2 is the ONS "happy" item, distinct from the (non-benchmark)
# ``happiness`` daily-slider field. Keys cover exactly ``BENCHMARK_TASKS``.
TASK_DISPLAY_NAMES: dict[str, str] = {
    "Atrial fibrillation (Afib)": "Atrial Fibrillation",
    "BMI_categories": "BMI Categories",
    "BMI_values": "BMI Value",
    "BiologicalSex": "Biological Sex",
    "CAD": "Coronary Artery Disease",
    "Cerebrovascular Disease": "Cerebrovascular Disease",
    "Congenital Heart": "Congenital Heart Disease",
    "Diabetes": "Diabetes",
    "GoSleepTime_categories": "Bedtime",
    "Hdl": "HDL Cholesterol",
    "Heart Failure or CHF": "Heart Failure / CHF",
    "Hypertension": "Hypertension",
    "Ldl": "LDL Cholesterol",
    "PH": "Pulmonary Hypertension",
    "Peripheral/Systemic Vascular Disease": "Vascular Disease",
    "SystolicBloodPressure": "Systolic Blood Pressure",
    "TotalCholesterol": "Total Cholesterol",
    "WakeUpTime_categories": "Wake-up Time",
    "WeightKilograms": "Body Weight",
    "age": "Age",
    "blood_pressure_categories": "Blood Pressure Categories",
    "cardiovascular_disease": "Cardiovascular Disease",
    "feel_worthwhile1": "Things Are Worthwhile",
    "feel_worthwhile2": "Feel Happy",
    "feel_worthwhile3": "Feel Worried",
    "feel_worthwhile4": "Feel Depressed",
    "framingham_risk": "Framingham CVD Risk",
    "satisfiedwith_life": "Life Satisfaction",
    "sleep_diagnosis1": "Sleep Disorder Diagnosis",
    "sleep_time_categories": "Sleep Duration",
    "vigorous_act": "Vigorous Activity Minutes",
    "work": "Currently Employed",
}
"""Internal task code -> human-readable display name (covers all of ``BENCHMARK_TASKS``)."""
