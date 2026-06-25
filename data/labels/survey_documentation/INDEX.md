# MHC Variable Index

169 variables used by the MHC-benchmark project (41 targets + 128 contexts), mapped to their source definitions in the iOS codebase. These are exactly the canonical benchmark labels in `data/labels/label_types.json`: the `survey_documentation/` tree has one variable `.md` per label (169 docs), plus a `summary.md` landing page per category and this index.

- **Filename convention**: `<category>/<raw_identifier>.md`. The `field_` prefix used in the benchmark dataset is stripped from filenames but recorded inside each file.
- **Role**: target = model prediction target; context = model input feature
- **Type**: continuous | binary | ordinal | categorical | multi_categorical

## By category

Variables are grouped into 16 semantic subdirectories. Each subdirectory contains its own `summary.md` landing page. The `Labels` column counts entries in `label_types.json`.

| Category | Labels | Contents |
|----------|--------|----------|
| [demographics](demographics/summary.md) | 7 | age, sex/gender, race/ethnicity, education, Fitzpatrick skin type |
| [geography](geography/summary.md) | 2 | country, zip |
| [anthropometrics](anthropometrics/summary.md) | 4 | weight, BMI_values, BMI_categories, height |
| [cardiometabolic_labs](cardiometabolic_labs/summary.md) | 9 | BP, cholesterol (HDL/LDL/Total), glucose, diabetes, hypertension, framingham_risk |
| [cardiovascular_disease_history](cardiovascular_disease_history/summary.md) | 10 | family history, medications, and derived subtype flags (CAD, Afib, CHF, PH, etc.) |
| [physical_activity](physical_activity/summary.md) | 13 | vigorous/moderate activity, work activity, daily-check activity1/activity2 items |
| [sleep](sleep/summary.md) | 10 | sleep_time variants, WakeUpTime, GoSleepTime, sleep diagnoses, derived *_categories |
| [healthkit_watch_metrics](healthkit_watch_metrics/summary.md) | 7 | Watch_RestingHeartRate, VO2Max, HRV, walking HR, stand time, basal energy, respiratory rate |
| [wellbeing](wellbeing/summary.md) | 7 | ONS wellbeing (satisfaction, worthwhile/happy/worried/depressed), daily happiness |
| [risk_perception](risk_perception/summary.md) | 4 | riskfactors1–4 (self-rated CVD risk on 5-point scales) |
| [diet](diet/summary.md) | 7 | fruit, vegetable, fish, grains, sugar_drinks, sodium, alcohol |
| [tobacco_vaping_cannabis](tobacco_vaping_cannabis/summary.md) | 28 | current / past / onset / quit items across vaping, cigarettes, smokeless, cannabis + product multi-selects |
| [parq_readiness](parq_readiness/summary.md) | 7 | PAR-Q yes/no gating items (chest pain, dizziness, heart condition, etc.) |
| [covid_19](covid_19/summary.md) | 16 | COVID test/diagnosis, symptoms, care level, exposure, behaviours |
| [mindset_measures](mindset_measures/summary.md) | 32 | illness mindset (20) + exercise process mindset (7) + adequacy of activity (5) |
| [study_metadata](study_metadata/summary.md) | 6 | labwork, device ownership flags, phone_on_user |

Each subdir's `summary.md` links every variable in that subdir. The flat tables below are the canonical master reference — all 41 targets and all 128 contexts, one row each.

## Targets (41)

| Variable | Type | Raw identifier | Source file | Line | Notes |
|----------|------|----------------|-------------|------|-------|
| age | continuous | heartAgeDataAge | CardioHealth/TasksAndSteps/HeartAgeControllers/HeartAgeRiskFactorCalculations/APHHeartAgeAndRiskFactors.m | ~62 | Manually entered in Heart Age form |
| Atrial fibrillation (Afib) | binary | heart_disease | CardioHealth/Resources/JSONs/cardiosurveys/cardio_CVhealth_survey.json | ~126 | Derived from enumeration value (option value=8) |
| BiologicalSex | binary | Gender | CardioHealth/TasksAndSteps/HeartAgeControllers/HeartAgeRiskFactorCalculations/APHHeartAgeAndRiskFactors.m | ~74 | Male/Female from Heart Age form |
| BMI_categories | ordinal | (derived) | CardioHealth/Startup/APHAppDelegate.m | ~1337 | Derived from BMI_values (ordinal binning) |
| BMI_values | continuous | (HealthKit) | CardioHealth/Startup/APHAppDelegate.m | ~1337 | HKQuantityTypeIdentifierBodyMass from HealthKit |
| blood_pressure_categories | ordinal | (derived) | CardioHealth/Startup/APHAppDelegate.m | ~533,535 | Derived from SystolicBloodPressure and DiastolicBloodPressure (ordinal binning) |
| CAD | binary | heart_disease | CardioHealth/Resources/JSONs/cardiosurveys/cardio_CVhealth_survey.json | ~101 | Coronary Artery Disease (Coronary Blockage/Stenosis) |
| Cerebrovascular Disease | binary | vascular | CardioHealth/Resources/JSONs/cardiosurveys/cardio_CVhealth_survey.json | ~164,169 | Stroke or TIA from vascular options |
| Congenital Heart | binary | heart_disease | CardioHealth/Resources/JSONs/cardiosurveys/cardio_CVhealth_survey.json | ~131 | Congenital Heart Defect (option value=9) |
| Diabetes | binary | (Heart Age form) | CardioHealth/TasksAndSteps/HeartAgeControllers/HeartAgeRiskFactorCalculations/APHHeartAgeAndRiskFactors.m | ~70 | kHeartAgeTestDataDiabetes |
| framingham_risk | continuous | (computed) | CardioHealth/TasksAndSteps/HeartAgeControllers/HeartAgeRiskFactorCalculations/APHHeartAgeAndRiskFactors.m | ~80-150 | Calculated from risk factors using Framingham methodology |
| GoSleepTime_categories | ordinal | GoSleepTime | (derived) | Unknown | Ordinal binning of GoSleepTime |
| happiness | continuous | feel_worthwhile2 | CardioHealth/Resources/JSONs/cardiosurveys/cardio_wellbeing_survey.json | ~40 | "How about happy?" 0-10 scale |
| happiness_categories | ordinal | (derived) | Unknown | Unknown | Derived from happiness (ordinal binning) |
| Heart Failure or CHF | binary | heart_disease | CardioHealth/Resources/JSONs/cardiosurveys/cardio_CVhealth_survey.json | ~121 | Heart Failure or CHF (option value=7) |
| Hdl | continuous | heartAgeDataHdl | CardioHealth/TasksAndSteps/HeartAgeControllers/HeartAgeRiskFactorCalculations/APHHeartAgeAndRiskFactors.m | ~64 | HDL cholesterol from Heart Age form |
| Hypertension | binary | (Heart Age form) | CardioHealth/TasksAndSteps/HeartAgeControllers/HeartAgeRiskFactorCalculations/APHHeartAgeAndRiskFactors.m | ~78 | kHeartAgeTestDataHypertension |
| Ldl | continuous | heartAgeDataLdl | CardioHealth/TasksAndSteps/HeartAgeControllers/HeartAgeRiskFactorCalculations/APHHeartAgeAndRiskFactors.m | ~65 | LDL cholesterol from Heart Age form |
| Peripheral/Systemic Vascular Disease | binary | vascular | CardioHealth/Resources/JSONs/cardiosurveys/cardio_CVhealth_survey.json | ~184 | Peripheral Vascular Disease (option value=5) |
| PH | binary | heart_disease | CardioHealth/Resources/JSONs/cardiosurveys/cardio_CVhealth_survey.json | ~136 | Pulmonary Hypertension (option value=11) |
| cardiovascular_disease | binary | (derived) | CardioHealth/Resources/JSONs/cardiosurveys/cardio_CVhealth_survey.json | ~83 | Derived from heart_disease options |
| SystolicBloodPressure | continuous | heartAgeDataSystolicBloodPressure | CardioHealth/TasksAndSteps/HeartAgeControllers/HeartAgeRiskFactorCalculations/APHHeartAgeAndRiskFactors.m | ~66 | Systolic BP from Heart Age form |
| TotalCholesterol | continuous | heartAgeDataTotalCholesterol | CardioHealth/TasksAndSteps/HeartAgeControllers/HeartAgeRiskFactorCalculations/APHHeartAgeAndRiskFactors.m | ~63 | Total cholesterol from Heart Age form |
| vigorous_act | continuous | vigorous_act | CardioHealth/Resources/JSONs/cardiosurveys/cardio_activitysleep_survey.json | ~137 | Minutes of vigorous activity per week |
| Watch_BasalEnergyBurned | continuous | (HealthKit) | CardioHealth/Startup/APHAppDelegate.m | ~523 | HKQuantityTypeIdentifierBasalEnergyBurned |
| Watch_HeartRateVariabilitySDNN | continuous | (HealthKit) | CardioHealth/Startup/APHAppDelegate.m | ~529 | HKQuantityTypeIdentifierHeartRateVariabilitySDNN |
| Watch_RespiratoryRate | continuous | (HealthKit) | CardioHealth/Startup/APHAppDelegate.m | ~1362 | HKQuantityTypeIdentifierRespiratoryRate |
| Watch_RestingHeartRate | continuous | (HealthKit) | CardioHealth/Startup/APHAppDelegate.m | ~527 | HKQuantityTypeIdentifierRestingHeartRate |
| Watch_StandTime | continuous | (HealthKit) | CardioHealth/Startup/APHAppDelegate.m | ~557 | HKQuantityTypeIdentifierAppleStandTime (iOS 13.0+) |
| Watch_VO2Max | continuous | (HealthKit) | CardioHealth/Startup/APHAppDelegate.m | ~538 | HKQuantityTypeIdentifierVO2Max |
| Watch_WalkingHeartRateAverage | continuous | (HealthKit) | CardioHealth/Startup/APHAppDelegate.m | ~528 | HKQuantityTypeIdentifierWalkingHeartRateAverage |
| WakeUpTime_categories | ordinal | WakeUpTime | (derived) | Unknown | Ordinal binning of WakeUpTime |
| WeightKilograms | continuous | (HealthKit) | CardioHealth/Startup/APHAppDelegate.m | ~524 | HKQuantityTypeIdentifierBodyMass (kg) |
| feel_worthwhile1 | ordinal | feel_worthwhile1 | CardioHealth/Resources/JSONs/cardiosurveys/cardio_wellbeing_survey.json | ~24 | "Extent things you do are worthwhile" 0-10 scale |
| feel_worthwhile2 | ordinal | feel_worthwhile2 | CardioHealth/Resources/JSONs/cardiosurveys/cardio_wellbeing_survey.json | ~41 | "How about happy?" 0-10 scale |
| feel_worthwhile3 | ordinal | feel_worthwhile3 | CardioHealth/Resources/JSONs/cardiosurveys/cardio_wellbeing_survey.json | ~56 | "How about worried?" 0-10 scale |
| feel_worthwhile4 | ordinal | feel_worthwhile4 | CardioHealth/Resources/JSONs/cardiosurveys/cardio_wellbeing_survey.json | ~72 | "How about depressed?" 0-10 scale |
| satisfiedwith_life | ordinal | satisfiedwith_life | CardioHealth/Resources/JSONs/cardiosurveys/cardio_wellbeing_survey.json | ~8 | Overall life satisfaction 0-10 scale |
| sleep_diagnosis1 | binary | sleep_diagnosis1 | CardioHealth/Resources/JSONs/cardiosurveys/cardio_activitysleep_survey.json | ~180 | Sleep disorder diagnosis (yes/no) |
| sleep_time_categories | ordinal | (derived) | CardioHealth/Resources/JSONs/cardiosurveys/cardio_activitysleep_survey.json | ~167 | Derived from sleep_time (ordinal binning) |
| work | binary | work | CardioHealth/Resources/JSONs/cardiosurveys/cardio_activitysleep_survey.json | ~9 | Regular work (boolean) |

## Contexts (128)

| Variable | Type | Raw identifier | Source file | Line | Notes |
|----------|------|----------------|-------------|------|-------|
| field_BloodGlucose | continuous | BloodGlucose | CardioHealth/TasksAndSteps/HeartAgeControllers/HeartAgeRiskFactorCalculations/APHHeartAgeAndRiskFactors.m | ~68 | Blood glucose value |
| field_GoSleepTime | continuous | GoSleepTime | (derived) | Unknown | Likely from device tracking, not in JSON |
| field_HeightCentimeters | continuous | (HealthKit) | CardioHealth/Startup/APHAppDelegate.m | ~525 | Height in meters from HealthKit, converted to cm |
| field_WakeUpTime | continuous | WakeUpTime | (derived) | Unknown | Likely from device tracking, not in JSON |
| field_atwork | ordinal | atwork | CardioHealth/Resources/JSONs/cardiosurveys/cardio_activitysleep_survey.json | ~28 | Work activity level (5-point scale) |
| field_country | categorical | country | CardioHealth/Resources/JSONs/cardiosurveys/cardio_wellbeing_survey.json | ~90 | Country of residence (UK/US/HK) |
| field_education | ordinal | education | CardioHealth/Resources/JSONs/cardiosurveys/cardio_CVhealth_survey.json | ~332 | Education level (7-point scale) |
| field_ethnicity | ordinal | ethnicity | CardioHealth/Resources/JSONs/cardiosurveys/cardio_CVhealth_survey.json | ~214 | Spanish/Hispanic/Latino (5-point scale) |
| field_fruit | continuous | fruit | CardioHealth/Resources/JSONs/cardiosurveys/cardio_diet_survey.json | ~7 | Cups of fruit per day |
| field_grains | continuous | grains | CardioHealth/Resources/JSONs/cardiosurveys/cardio_diet_survey.json | ~50 | Servings of whole grains per day |
| field_moderate_act | continuous | moderate_act | CardioHealth/Resources/JSONs/cardiosurveys/cardio_activitysleep_survey.json | ~122 | Minutes of moderate activity per week |
| field_phone_on_user | ordinal | phone_on_user | CardioHealth/Resources/JSONs/cardiosurveys/cardio_daily_check.json | ~12 | Phone/device with user (4-point scale) |
| field_phys_activity | ordinal | phys_activity | CardioHealth/Resources/JSONs/cardiosurveys/cardio_activitysleep_survey.json | ~69 | Leisure time activity (6-point scale) |
| field_riskfactors1 | ordinal | riskfactors1 | CardioHealth/Resources/JSONs/cardiosurveys/cardio_wellbeing_survey.json | ~193 | Risk over next 10 years (5-point scale) |
| field_riskfactors2 | ordinal | riskfactors2 | CardioHealth/Resources/JSONs/cardiosurveys/cardio_wellbeing_survey.json | ~233 | Risk compared to age/sex (5-point scale) |
| field_riskfactors3 | ordinal | riskfactors3 | CardioHealth/Resources/JSONs/cardiosurveys/cardio_wellbeing_survey.json | ~273 | Lifetime risk (5-point scale) |
| field_riskfactors4 | ordinal | riskfactors4 | CardioHealth/Resources/JSONs/cardiosurveys/cardio_wellbeing_survey.json | ~313 | Lifetime risk vs. age/sex (5-point scale) |
| field_sodium | ordinal | sodium | CardioHealth/Resources/JSONs/cardiosurveys/cardio_diet_survey.json | ~78 | Sodium reduction strategies (3-point multi-select) |
| field_sugar_drinks | continuous | sugar_drinks | CardioHealth/Resources/JSONs/cardiosurveys/cardio_diet_survey.json | ~64 | Beverages with added sugar per week |
| field_vegetable | continuous | vegetable | CardioHealth/Resources/JSONs/cardiosurveys/cardio_diet_survey.json | ~22 | Cups of vegetables per day |
| field_fish | continuous | fish | CardioHealth/Resources/JSONs/cardiosurveys/cardio_diet_survey.json | ~36 | Servings of fish per week |
| alcohol | ordinal | alcohol | CardioHealth/Resources/JSONs/cardiosurveys/cardio_diet_survey.json | ~113 | Alcohol consumption frequency (5-point) |
| Ethnicity_heartage | categorical | Ethnicity | CardioHealth/TasksAndSteps/HeartAgeControllers/HeartAgeRiskFactorCalculations/APHHeartAgeAndRiskFactors.m | ~73 | Ethnicity from Heart Age (African-American/Other) |
| FitzpatrickSkinType | categorical | (skin type) | HealthKit `HKCharacteristicTypeIdentifierFitzpatrickSkinType` | n/a | Fitzpatrick skin type (raw I-VI categorical; released as bucketed `{light, medium, dark}` ordinal) |
| healthcare_worker | categorical | healthcare_worker | CardioHealth/Resources/JSONs/cardiosurveys/cardio_covid_19_survey.json | ~454 | Healthcare position type |
| activity1_intensity | ordinal | activity1_intensity | CardioHealth/Resources/JSONs/cardiosurveys/cardio_daily_check.json | ~139 | Activity intensity (Light/Moderate/Vigorous) |
| activity1_type | categorical | activity1_type | CardioHealth/Resources/JSONs/cardiosurveys/cardio_daily_check.json | ~74 | Activity type (Walking/Jogging/Cycling/etc) |
| activity2_intensity | ordinal | activity2_intensity | CardioHealth/Resources/JSONs/cardiosurveys/cardio_daily_check.json | ~258 | Activity 2 intensity (Light/Moderate/Vigorous) |
| activity2_type | categorical | activity2_type | CardioHealth/Resources/JSONs/cardiosurveys/cardio_daily_check.json | ~193 | Activity 2 type (Walking/Jogging/Cycling/etc) |
| chestPain | binary | chestPain | CardioHealth/TasksAndSteps/APHDynamicParQQuizTask.m | ~42 | Chest pain during activity (PAR-Q) |
| chestPainInLastMonth | binary | chestPainInLastMonth | CardioHealth/TasksAndSteps/APHDynamicParQQuizTask.m | ~43 | Chest pain in past month (PAR-Q) |
| currentCannabisSmoking | ordinal | currentCannabisSmoking | CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json | ~855 | Cannabis smoking frequency |
| currentSmokeless | ordinal | currentSmokeless | CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json | ~472 | Smokeless tobacco use frequency |
| currentSmoking | ordinal | currentSmoking | CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json | ~240 | Cigarette smoking frequency |
| currentVaping | ordinal | currentVaping | CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json | ~8 | Vaping nicotine frequency |
| dizziness | binary | dizziness | CardioHealth/TasksAndSteps/APHDynamicParQQuizTask.m | ~44 | Dizziness/loss of consciousness (PAR-Q) |
| durationQuitSmokeless | categorical | durationQuitSmokeless | CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json | ~614 | Duration quit smokeless tobacco (codes 1-3 ordered, 4=Never/5=Don't know break ordinality) |
| durationQuitSmoking | categorical | durationQuitSmoking | CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json | ~387 | Duration quit smoking (same encoding as above) |
| durationQuitVaping | categorical | durationQuitVaping | CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json | ~155 | Duration quit vaping (same encoding as above) |
| device_activity_band | binary | device | CardioHealth/Resources/JSONs/cardiosurveys/cardio_day_one.json | ~36 | Activity band/pedometer ownership |
| device_iphone | binary | device | CardioHealth/Resources/JSONs/cardiosurveys/cardio_day_one.json | ~36 | iPhone ownership |
| device_other | binary | device | CardioHealth/Resources/JSONs/cardiosurveys/cardio_day_one.json | ~36 | Other device ownership |
| device_smartwatch | binary | device | CardioHealth/Resources/JSONs/cardiosurveys/cardio_day_one.json | ~36 | Smartwatch/Apple Watch ownership |
| everQuitSmokeless | binary | everQuitSmokeless | CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json | ~595 | Tried to quit smokeless in past 12 months |
| everQuitSmoking | binary | everQuitSmoking | CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json | ~368 | Tried to quit smoking in past 12 months |
| everQuitVaping | binary | everQuitVaping | CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json | ~136 | Tried to quit vaping in past 12 months |
| family_history | categorical | family_history | CardioHealth/Resources/JSONs/cardiosurveys/cardio_CVhealth_survey.json | ~10 | Family history of early heart disease |
| heartCondition | binary | heartCondition | CardioHealth/TasksAndSteps/APHDynamicParQQuizTask.m | ~41 | Heart condition (PAR-Q) |
| jointProblem | binary | jointProblem | CardioHealth/TasksAndSteps/APHDynamicParQQuizTask.m | ~45 | Bone/joint problem (PAR-Q) |
| labwork | binary | labwork | CardioHealth/Resources/JSONs/cardiosurveys/cardio_day_one.json | ~69 | Will have lab work in next 7 days |
| lastCannabisSmoking | ordinal | lastCannabisSmoking | CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json | ~1001 | Last time smoked cannabis |
| medications_to_treat | categorical | medications_to_treat | CardioHealth/Resources/JSONs/cardiosurveys/cardio_CVhealth_survey.json | ~44 | Medications for risk factors |
| onsetSmokeless | continuous | onsetSmokeless | CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json | ~521 | Age first used smokeless tobacco |
| onsetSmoking | continuous | onsetSmoking | CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json | ~354 | Age smoked first cigarette |
| onsetVaping | continuous | onsetVaping | CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json | ~122 | Age first vaped |
| pastCannabisSmoking | ordinal | pastCannabisSmoking | CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json | ~954 | Duration of past cannabis smoking |
| pastSmokeless | ordinal | pastSmokeless | CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json | ~535 | Used smokeless tobacco in past (yes/no) |
| pastVaping | ordinal | pastVaping | CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json | ~78 | Vaped in the past (yes/no) |
| physicallyCapable | binary | physicallyCapable | CardioHealth/TasksAndSteps/APHDynamicParQQuizTask.m | ~47 | Any other reason not to be active (PAR-Q) |
| prescriptionDrugs | binary | prescriptionDrugs | CardioHealth/TasksAndSteps/APHDynamicParQQuizTask.m | ~46 | Prescription drugs for BP/heart (PAR-Q) |
| race | categorical | race | CardioHealth/Resources/JSONs/cardiosurveys/cardio_CVhealth_survey.json | ~255 | Race (White/Black/Asian/etc) |
| readinessQuitSmokeless | continuous | readinessQuitSmokeless | CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json | ~581 | Readiness to quit smokeless (1-10) |
| readinessQuitSmoking | continuous | readinessQuitSmoking | CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json | ~289 | Readiness to quit smoking (1-10) |
| readinessQuitVaping | continuous | readinessQuitVaping | CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json | ~57 | Readiness to quit vaping (1-10) |
| sleep_diagnosis2 | categorical | sleep_diagnosis2 | CardioHealth/Resources/JSONs/cardiosurveys/cardio_activitysleep_survey.json | ~200 | Type of sleep disorder |
| sleep_time1 | continuous | sleep_time1 | CardioHealth/Resources/JSONs/cardiosurveys/cardio_activitysleep_survey.json | ~152 | Hours of sleep on weekdays |
| tobaccoProducts | categorical | tobaccoProducts | CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json | ~699 | Tobacco products used in past week |
| tobaccoProductsEver | categorical | tobaccoProductsEver | CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json | ~755 | Tobacco products ever used |
| days_admitted | continuous | days_admitted | CardioHealth/Resources/JSONs/cardiosurveys/cardio_covid_19_survey.json | ~319 | Days hospitalized for COVID |
| zip | categorical | zip | CardioHealth/Resources/JSONs/cardiosurveys/cardio_wellbeing_survey.json | ~177 | Postcode (UK) / zip (US) |
| cannabisSmoking | ordinal | cannabisSmoking | CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json | ~811 | Cannabis smoking history |
| body_self_healing_in_many_different_circumstances | ordinal | body_self_healing_in_many_different_circumstances | CardioHealth/Resources/JSONs/cardiosurveys/cardio_illness_mindset_measure_inventory_survey.json | ~16 | Body self-healing belief (6-point scale) |
| chronic_illness_impact | ordinal | chronic_illness_impact | CardioHealth/Resources/JSONs/cardiosurveys/cardio_illness_mindset_measure_inventory_survey.json | ~61 | Chronic illness impact (6-point scale) |
| chronic_illness_body_meaning | ordinal | chronic_illness_body_meaning | CardioHealth/Resources/JSONs/cardiosurveys/cardio_illness_mindset_measure_inventory_survey.json | ~106 | Body function meaning (6-point scale) |
| chronic_illness_body_coping | ordinal | chronic_illness_body_coping | CardioHealth/Resources/JSONs/cardiosurveys/cardio_illness_mindset_measure_inventory_survey.json | ~151 | Body coping ability (6-point scale) |
| chronic_illness_positive_opportunity | ordinal | chronic_illness_positive_opportunity | CardioHealth/Resources/JSONs/cardiosurveys/cardio_illness_mindset_measure_inventory_survey.json | ~196 | Positive life changes opportunity (6-point scale) |
| chronic_illness_management | ordinal | chronic_illness_management | CardioHealth/Resources/JSONs/cardiosurveys/cardio_illness_mindset_measure_inventory_survey.json | ~241 | Chronic illness manageability (6-point scale) |
| chronic_illness_body_betrayal | ordinal | chronic_illness_body_betrayal | CardioHealth/Resources/JSONs/cardiosurveys/cardio_illness_mindset_measure_inventory_survey.json | ~286 | Body betrayal feeling (6-point scale) |
| chronic_illness_more_meaning_in_life | ordinal | chronic_illness_more_meaning_in_life | CardioHealth/Resources/JSONs/cardiosurveys/cardio_illness_mindset_measure_inventory_survey.json | ~331 | Life meaning finding (6-point scale) |
| chronic_illness_handling | ordinal | chronic_illness_handling | CardioHealth/Resources/JSONs/cardiosurveys/cardio_illness_mindset_measure_inventory_survey.json | ~376 | Chronic illness handling (6-point scale) |
| body_remarkable_self_healing | ordinal | body_remarkable_self_healing | CardioHealth/Resources/JSONs/cardiosurveys/cardio_illness_mindset_measure_inventory_survey.json | ~421 | Body self-healing properties (6-point scale) |
| chronic_illness_spoil | ordinal | chronic_illness_spoil | CardioHealth/Resources/JSONs/cardiosurveys/cardio_illness_mindset_measure_inventory_survey.json | ~466 | Chronic illness spoils life (6-point scale) |
| chronic_illness_challenge | ordinal | chronic_illness_challenge | CardioHealth/Resources/JSONs/cardiosurveys/cardio_illness_mindset_measure_inventory_survey.json | ~511 | Challenge strengthening (6-point scale) |
| chronic_illness_body_handling | ordinal | chronic_illness_body_handling | CardioHealth/Resources/JSONs/cardiosurveys/cardio_illness_mindset_measure_inventory_survey.json | ~556 | Body handling illness (6-point scale) |
| chronic_illness_runing_life | ordinal | chronic_illness_runing_life | CardioHealth/Resources/JSONs/cardiosurveys/cardio_illness_mindset_measure_inventory_survey.json | ~601 | Chronic illness ruins life (6-point scale) |
| chronic_illness_body_management | ordinal | chronic_illness_body_management | CardioHealth/Resources/JSONs/cardiosurveys/cardio_illness_mindset_measure_inventory_survey.json | ~646 | Body design for illness management (6-point scale) |
| chronic_illness_relatively_normal_life | ordinal | chronic_illness_relatively_normal_life | CardioHealth/Resources/JSONs/cardiosurveys/cardio_illness_mindset_measure_inventory_survey.json | ~691 | Normal life with chronic illness (6-point scale) |
| chronic_illness_body_failure | ordinal | chronic_illness_body_failure | CardioHealth/Resources/JSONs/cardiosurveys/cardio_illness_mindset_measure_inventory_survey.json | ~736 | Body failure feeling (6-point scale) |
| chronic_illness_empowering | ordinal | chronic_illness_empowering | CardioHealth/Resources/JSONs/cardiosurveys/cardio_illness_mindset_measure_inventory_survey.json | ~781 | Fighting illness empowering (6-point scale) |
| body_self_healing_from_most_conditions_and_diseases | ordinal | body_self_healing_from_most_conditions_and_diseases | CardioHealth/Resources/JSONs/cardiosurveys/cardio_illness_mindset_measure_inventory_survey.json | ~826 | Body healing from most conditions (6-point scale) |
| chronic_illness_body_blame | ordinal | chronic_illness_body_blame | CardioHealth/Resources/JSONs/cardiosurveys/cardio_illness_mindset_measure_inventory_survey.json | ~871 | Body blame for illness (6-point scale) |
| field_activity1_option | binary | activity1_option | CardioHealth/Resources/JSONs/cardiosurveys/cardio_daily_check.json | ~51 | Did you perform any physical activities yesterday that you think were not recorded… |
| field_activity1_time | continuous | activity1_time | CardioHealth/Resources/JSONs/cardiosurveys/cardio_daily_check.json | ~123 | How long did you do the activity? |
| field_activity2_option | binary | activity2_option | CardioHealth/Resources/JSONs/cardiosurveys/cardio_daily_check.json | ~170 | Did you perform any additional physical activities yesterday that you think were not… |
| field_activity2_time | continuous | activity2_time | CardioHealth/Resources/JSONs/cardiosurveys/cardio_daily_check.json | ~243 | How long did you do the activity? |
| field_antibiotics | multi_categorical | antibiotics | CardioHealth/Resources/JSONs/cardiosurveys/cardio_covid_19_survey.json | ~633 | Are you taking any of the following antibiotics or immune system modulators… |
| field_beneficial | ordinal | beneficial | CardioHealth/Resources/JSONs/cardiosurveys/cardio_adequacy_of_activity_mindset_measure_survey.json | ~116-162 | How harmful/beneficial is your current level of physical activity for your health? |
| field_building | ordinal | building | CardioHealth/Resources/JSONs/cardiosurveys/cardio_covid_19_survey.json | ~831 | How many people live in your building? |
| field_cannabisVaping | ordinal | cannabisVaping | CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json | ~1048 | Do you vape cannabis or cannabis containing products? |
| field_conditions | multi_categorical | conditions | CardioHealth/Resources/JSONs/cardiosurveys/cardio_covid_19_survey.json | ~393 | What conditions do you have? (Select all that apply) |
| field_convenient | ordinal | convenient | CardioHealth/Resources/JSONs/cardiosurveys/cardio_exercise_process_mindset_measure_survey.json | ~121-152 | EXERCISING is: (convenient) |
| field_covid | ordinal | covid | CardioHealth/Resources/JSONs/cardiosurveys/cardio_covid_19_survey.json | ~15 | Have you ever had RNA testing for current COVID-19? |
| field_covid_serologic | ordinal | covid_serologic | CardioHealth/Resources/JSONs/cardiosurveys/cardio_covid_19_survey.json | ~50 | Have you ever had serologic or antibody testing for COVID-19? |
| field_currentCannabisVaping | ordinal | currentCannabisVaping | CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json | ~1092 | How often do you vape cannabis? |
| field_daily_activities | ordinal | daily_activities | CardioHealth/Resources/JSONs/cardiosurveys/cardio_covid_19_survey.json | ~284 | While symptomatic, could you complete your usual daily activities? |
| field_disease | ordinal | disease | CardioHealth/Resources/JSONs/cardiosurveys/cardio_adequacy_of_activity_mindset_measure_survey.json | ~166-212 | How much does your current level of physical (in-)activity increase or decrease your… |
| field_easy | ordinal | easy | CardioHealth/Resources/JSONs/cardiosurveys/cardio_exercise_process_mindset_measure_survey.json | ~16-47 | EXERCISING is: (easy) |
| field_exposure | ordinal | exposure | CardioHealth/Resources/JSONs/cardiosurveys/cardio_covid_19_survey.json | ~760 | Have you been exposed to anyone that tested positive for COVID-19? |
| field_face_covering | ordinal | face_covering | CardioHealth/Resources/JSONs/cardiosurveys/cardio_covid_19_survey.json | ~902 | Do you wear a face covering when you leave the house? |
| field_fun | ordinal | fun | CardioHealth/Resources/JSONs/cardiosurveys/cardio_exercise_process_mindset_measure_survey.json | ~156-187 | EXERCISING is: (fun) |
| field_indulgent | ordinal | indulgent | CardioHealth/Resources/JSONs/cardiosurveys/cardio_exercise_process_mindset_measure_survey.json | ~226-257 | EXERCISING is: (indulgent) |
| field_lastCannabisVaping | ordinal | lastCannabisVaping | CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json | ~1238 | When was the last time you vaped cannabis? |
| field_most_intense_care | ordinal | most_intense_care | CardioHealth/Resources/JSONs/cardiosurveys/cardio_covid_19_survey.json | ~244 | What was the most intense care you received for your symptoms? |
| field_muscles | ordinal | muscles | CardioHealth/Resources/JSONs/cardiosurveys/cardio_adequacy_of_activity_mindset_measure_survey.json | ~216-262 | How much is your current level of physical (in-)activity strengthening or weakening… |
| field_pastCannabisVaping | ordinal | pastCannabisVaping | CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json | ~1191 | For how long did you vape cannabis? |
| field_pleasurable | ordinal | pleasurable | CardioHealth/Resources/JSONs/cardiosurveys/cardio_exercise_process_mindset_measure_survey.json | ~51-82 | EXERCISING is: (pleasurable) |
| field_relaxing | ordinal | relaxing | CardioHealth/Resources/JSONs/cardiosurveys/cardio_exercise_process_mindset_measure_survey.json | ~86-117 | EXERCISING is: (relaxing) |
| field_self_isolating | ordinal | self_isolating | CardioHealth/Resources/JSONs/cardiosurveys/cardio_covid_19_survey.json | ~866 | To what extent are you currently self-isolating? |
| field_severity | ordinal | severity | CardioHealth/Resources/JSONs/cardiosurveys/cardio_covid_19_survey.json | ~379 | When you felt the worst in the past month, from 0 (most sick) - 10 (perfect health)… |
| field_severity_covid | ordinal | severity-covid | CardioHealth/Resources/JSONs/cardiosurveys/cardio_covid_19_survey.json | ~365 | When you felt the worst with COVID-19, from 0 (most sick) - 10 (perfect health) how… |
| field_sleep_time | continuous | sleep_time | CardioHealth/Resources/JSONs/cardiosurveys/cardio_activitysleep_survey.json | ~167 | How much sleep do you think you need every night to be rested? |
| field_sleep_time_daily | continuous | sleep_time | CardioHealth/Resources/JSONs/cardiosurveys/cardio_daily_check.json | ~301 | How many hours of sleep did you get last night? |
| field_smokingHistory | binary | smokingHistory | cardio_vaping_and_smoking_survey.json | ~n/a | Are you currently smoking cigarettes? |
| field_social | ordinal | social | CardioHealth/Resources/JSONs/cardiosurveys/cardio_exercise_process_mindset_measure_survey.json | ~191-222 | EXERCISING is: (social) |
| field_symptoms_past_week | multi_categorical | symptoms_past_week | CardioHealth/Resources/JSONs/cardiosurveys/cardio_covid_19_survey.json | ~n/a | Did you experience any of the following symptoms in the past week ? (Select all that… |
| field_symptoms_week_preceding | multi_categorical | symptoms_week_preceding | CardioHealth/Resources/JSONs/cardiosurveys/cardio_covid_19_survey.json | ~80 | Did you experience any of the following symptoms in the week preceding your COVID… |
| field_unhealthy | ordinal | unhealthy | CardioHealth/Resources/JSONs/cardiosurveys/cardio_adequacy_of_activity_mindset_measure_survey.json | ~16-62 | My current level of physical activity is unhealthy. |
| field_weight | ordinal | weight | CardioHealth/Resources/JSONs/cardiosurveys/cardio_adequacy_of_activity_mindset_measure_survey.json | ~66-112 | My current level of physical activity is helping me achieve or maintain a healthy… |

## Derived / computed variables (documented, no direct survey field)

These have no single survey field — each is documented in its own `.md` file with derivation details:

- `framingham_risk.md` — Computed in iOS (`APHHeartAgeAndRiskFactors.m` lines ~173-271) using the Framingham ASCVD 10-year risk equation.
- `BMI_values.md` — HealthKit body mass & height, or `HKQuantityTypeIdentifierBodyMassIndex`.
- `BMI_categories.md`, `blood_pressure_categories.md`, `WakeUpTime_categories.md`, `GoSleepTime_categories.md`, `happiness_categories.md`, `sleep_time_categories.md` — Post-hoc binning defined in the **MHC-benchmark** repo (bin edges out of scope for this iOS repo).
- `cardiovascular_disease.md`, `Heart Failure or CHF.md`, `Atrial fibrillation (Afib).md`, `PH.md`, `CAD.md`, `Congenital Heart.md`, `Peripheral-Systemic Vascular Disease.md`, `Cerebrovascular Disease.md` — Binary flags derived from `heart_disease` / `vascular` multi-select options in `cardio_CVhealth_survey.json`. Each file lists the specific option value(s) that trigger a positive flag.
- `WakeUpTime.md`, `GoSleepTime.md` — Not in survey JSONs; sourced from AppCore user-info profile items (`kAPCUserInfoItemTypeWakeUpTime`, `kAPCUserInfoItemTypeSleepTime` in `APHAppDelegate.m` lines 1435-1436).
- `FitzpatrickSkinType.md` — Read via `HKCharacteristicTypeIdentifierFitzpatrickSkinType` (APHAppDelegate.m line 1330); not a survey question. Earlier doc revisions misspelled this as `FrickSkinType.md`.

## Summary

- **Targets**: 41 (canonical, from `label_types.json`).
- **Contexts**: 128 (canonical, from `label_types.json`).
- **Total files in `survey_documentation/`**: 169 variable docs + 16 per-category `summary.md` + this `INDEX.md` = 186 `.md` files. The 169 variable docs are exactly the benchmark labels in `label_types.json` (one per label). Raw survey fields that are not released as labels — e.g. the `heart_disease` / `vascular` multi-selects that derived targets such as CAD and Afib are computed from — are referenced inline where relevant but are not separately documented.

All survey identifiers are exact string matches from JSON survey definitions. Heart Age variables map to Objective-C constants in `APHHeartAgeAndRiskFactors.m` and `APHHeartAgeTaskViewController.m`. HealthKit variables map to `HKQuantityTypeIdentifier`/`HKCharacteristicTypeIdentifier` constants in `APHAppDelegate.m`. PAR-Q variables map to static NSString constants in `APHDynamicParQQuizTask.m`. The `field_` prefix from the benchmark dataset is stripped from all filenames but recorded in each file's frontmatter as the "Benchmark column".

## Redundancy / duplicate analysis

Tiered from most to least redundant.

### Derived-from-parent (same info, lower resolution)

Each `_categories` target is post-hoc binning of a continuous parent — keep either the continuous parent or the categorical version, not both:

- `BMI_categories` ← `BMI_values`
- `blood_pressure_categories` ← `SystolicBloodPressure` + `DiastolicBloodPressure`
- `happiness_categories` ← `happiness`
- `sleep_time_categories` ← `sleep_time`
- `WakeUpTime_categories` ← `WakeUpTime`
- `GoSleepTime_categories` ← `GoSleepTime`

### Union of components

- **cardiovascular_disease** = OR of `CAD`, `Heart Failure or CHF`, `Atrial fibrillation (Afib)`, `PH`, `Congenital Heart`, `Peripheral/Systemic Vascular Disease`, `Cerebrovascular Disease`. Modeling it alongside the subtype flags double-counts.

### Same question, different survey visit (bi-weekly duplication)

Many COVID items appear in both `cardio_covid_19_survey.json` (one-time) and `cardio_covid_19_recurrent_survey.json` (bi-weekly) — same variable, different cadence: `covid`, `covid_serologic`, `symptoms_*`, `severity`, `severity_covid`, `most_intense_care`, `daily_activities`, `days_admitted`. The benchmark probably merges these, so each context variable represents both flavours.

### Overlapping but not identical

- **ethnicity / Ethnicity_heartage / race** — three different codings of overlapping info; `Ethnicity_heartage` is a 2-category Framingham simplification.
- **tobaccoProducts / tobaccoProductsEver** — past-week vs ever-used; same structure, different horizons.
- **symptoms_week_preceding / symptoms_past_week** — same checklist, different reference windows.
- **feel_worthwhile2 vs happiness** — both positive-affect measures, but different surveys and cadences (retrospective ONS yesterday-item vs daily slider).
- **sleep_time / sleep_time1 / sleep_time_daily** — all sleep duration, but distinct: perceived need / weekday self-report / daily log.

### Near-duplicates inside the mindset battery

Within `mindset_measures/`, several chronic-illness items cover very close semantic territory (by design — multi-item psychometric batteries use near-duplicates for scale reliability, but they carry heavily correlated signal as modeling inputs):

- `body_self_healing_in_many_different_circumstances`, `body_remarkable_self_healing`, `body_self_healing_from_most_conditions_and_diseases` — three items on body self-healing belief.
- `chronic_illness_handling`, `chronic_illness_body_handling`, `chronic_illness_management`, `chronic_illness_body_management` — four items on "can I/my body handle/manage this illness".
- `chronic_illness_body_betrayal`, `chronic_illness_body_failure`, `chronic_illness_body_blame` — three "body-has-failed-me" items.
- `chronic_illness_spoil` vs `chronic_illness_runing_life` — both "illness ruins life".
