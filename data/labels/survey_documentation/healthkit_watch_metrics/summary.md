# HealthKit Watch Metrics

Seven continuous physiological streams collected passively from Apple Watch via HealthKit background delivery. No user-facing questions; each variable maps directly to an `HKQuantityTypeIdentifier` registered in `CardioHealth/Startup/APHAppDelegate.m` (lines ~1346-1363).

## Variables (7 files)

| Variable | Role | Type | HK identifier | Summary |
|----------|------|------|---------------|---------|
| [Watch_RestingHeartRate](Watch_RestingHeartRate.md) | target | continuous | `HKQuantityTypeIdentifierRestingHeartRate` | Resting heart rate (bpm) |
| [Watch_VO2Max](Watch_VO2Max.md) | target | continuous | `HKQuantityTypeIdentifierVO2Max` | Cardiorespiratory fitness (ml/kg/min) |
| [Watch_HeartRateVariabilitySDNN](Watch_HeartRateVariabilitySDNN.md) | target | continuous | `HKQuantityTypeIdentifierHeartRateVariabilitySDNN` | HRV SDNN (ms) |
| [Watch_WalkingHeartRateAverage](Watch_WalkingHeartRateAverage.md) | target | continuous | `HKQuantityTypeIdentifierWalkingHeartRateAverage` | Average HR while walking (bpm) |
| [Watch_StandTime](Watch_StandTime.md) | target | continuous | `HKQuantityTypeIdentifierAppleStandTime` (iOS 13+) | Minutes standing per day |
| [Watch_BasalEnergyBurned](Watch_BasalEnergyBurned.md) | target | continuous | `HKQuantityTypeIdentifierBasalEnergyBurned` | Basal metabolic energy (kcal) |
| [Watch_RespiratoryRate](Watch_RespiratoryRate.md) | target | continuous | `HKQuantityTypeIdentifierRespiratoryRate` | Respiratory rate (breaths/min) |

## Notes

- All seven are "DAILY_LABELS" in the MHC-benchmark pipeline — longitudinal streams excluded from the paper's main cross-sectional eval table.
- HealthKit permission is requested at onboarding; missing permission means absent data rather than zeros.
- `Watch_StandTime` requires iOS 13.0+; older devices yield no data.
- Weight, height, and BMI are also HealthKit-backed but live in `anthropometrics/`; Fitzpatrick skin type lives in `demographics/`.
