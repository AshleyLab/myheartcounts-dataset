# HeightCentimeters

**Benchmark column**: `field_HeightCentimeters`
**Raw identifier**: `HKQuantityTypeIdentifierHeight` (meters, converted to centimeters)
**Role**: context
**Type**: continuous

## Source
- File: `CardioHealth/Startup/APHAppDelegate.m`
- Line: 1349 (registered in `healthKitQuantityTypesToRead` method)
- Unit configuration: Line 525 (in `researcherSpecifiedUnits` method)
- Collected via: HealthKit background delivery (no user-facing question)

## Question
Not a survey variable. Collected automatically via HealthKit from user health data.

## Answer options / Units
- HKQuantityTypeIdentifier: `HKQuantityTypeIdentifierHeight`
- Expected unit: centimeters (cm) - stored as meters in HealthKit, converted for analysis
- Formula: `[HKUnit meterUnit]`
- Conversion: 1 meter = 100 centimeters
- Source devices: iPhone (user-entered) or Apple Watch

## Observed values

**Total observations**: 25,437 — **type-enforced**: 25,437 (**unique**: 76) — raw Python types seen: `float` (25,437).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 0 |
| q25 | 165.1 |
| median | 175.3 |
| mean | 161.2 |
| q75 | 180.3 |
| max | 271.8 |
| std | 47.69 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `177.8` | 2,667 |
| `180.3` | 2,405 |
| `175.3` | 2,258 |
| `172.7` | 2,054 |
| `170.2` | 2,009 |
| `182.9` | 2,000 |
| `0` | 1,916 |
| `167.6` | 1,567 |
| `185.4` | 1,402 |
| `162.6` | 1,216 |
| `165.1` | 1,154 |
| `188` | 1,148 |
| `160` | 827 |
| `157.5` | 685 |
| `190.5` | 582 |
| `193` | 387 |
| `154.9` | 335 |
| `152.4` | 233 |
| `195.6` | 146 |
| `149.9` | 95 |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history
- Commits touching source file: 117 total
- Most recent material change: ce6ad36 (2020-09-30) Revert "MHC-790 Add new mobility HealthKit fields"
- Notes: Height identifier stable throughout app history; core metric for health calculations

## Notes
- HealthKit permission requested at onboarding via `requestForPermissionForType:kAPCSignUpPermissionsTypeHealthKit`
- Background delivery enabled at hourly frequency
- Height is typically entered once during user signup but can be updated in Health app
- Used as denominator in BMI calculation (BMI = weight / height²)
- Also used for basal metabolic rate (BMR) estimation and activity calibration
- Context variable supporting primary cardiovascular health metrics in MyHeart Counts research
- Generally stable throughout study period (unlikely to change frequently unlike weight)
