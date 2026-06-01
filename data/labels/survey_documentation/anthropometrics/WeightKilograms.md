# WeightKilograms

**Benchmark column**: `WeightKilograms`
**Raw identifier**: `HKQuantityTypeIdentifierBodyMass`
**Role**: target
**Type**: continuous

## Source
- File: `CardioHealth/Startup/APHAppDelegate.m`
- Line: 1346 (registered in `healthKitQuantityTypesToRead` method)
- Unit configuration: Line 524 (in `researcherSpecifiedUnits` method)
- Collected via: HealthKit background delivery (no user-facing question)

## Question
Not a survey variable. Collected automatically via HealthKit from Apple Watch/iPhone sensors and manually entered by user.

## Answer options / Units
- HKQuantityTypeIdentifier: `HKQuantityTypeIdentifierBodyMass`
- Expected unit: kilograms (kg)
- Formula: `[HKUnit gramUnitWithMetricPrefix:HKMetricPrefixKilo]`
- Source devices: iPhone (user-entered), Apple Watch (synced from health data)

## Observed values

**Total observations**: 25,437 — **type-enforced**: 25,437 (**unique**: 353) — raw Python types seen: `float` (25,437).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 30.00 |
| q25 | 63.05 |
| median | 77.11 |
| mean | 77.04 |
| q75 | 91.17 |
| max | 300 |
| std | 27.00 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `30.00` | 2,782 |
| `74.84` | 486 |
| `77.11` | 483 |
| `81.65` | 460 |
| `83.91` | 458 |
| `68.04` | 448 |
| `72.57` | 442 |
| `79.38` | 426 |
| `90.72` | 414 |
| `86.18` | 407 |
| `70.31` | 394 |
| `63.50` | 347 |
| `88.45` | 336 |
| `95.25` | 307 |
| `99.79` | 294 |
| `65.77` | 289 |
| `58.97` | 273 |
| `92.99` | 271 |
| `78.02` | 250 |
| `61.23` | 246 |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history
- Commits touching source file: 117 total
- Most recent material change: ce6ad36 (2020-09-30) Revert "MHC-790 Add new mobility HealthKit fields"
- Notes: Body mass identifier stable throughout app history; core metric for all health calculations

## Notes
- HealthKit permission requested at onboarding via `requestForPermissionForType:kAPCSignUpPermissionsTypeHealthKit`
- Background delivery enabled at hourly frequency
- Users can input weight through Health app on iPhone or Apple Health integration
- Weight data used for BMI calculation, basal metabolic rate estimation, and activity intensity adjustments
- Critical metric for MyHeart Counts cardiovascular health assessment and personalized recommendations
- Cross-referenced with Height to compute BMI and other health indicators
