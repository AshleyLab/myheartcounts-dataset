# Watch_VO2Max

**Benchmark column**: `Watch_VO2Max`
**Raw identifier**: `HKQuantityTypeIdentifierVO2Max`
**Role**: target
**Type**: continuous

## Source
- File: `CardioHealth/Startup/APHAppDelegate.m`
- Line: 1363 (registered in `healthKitQuantityTypesToRead` method)
- Unit configuration: Line 538 (in `researcherSpecifiedUnits` method)
- Collected via: HealthKit background delivery (no user-facing question)

## Question
Not a survey variable. Collected automatically via HealthKit from Apple Watch/iPhone sensors.

## Answer options / Units
- HKQuantityTypeIdentifier: `HKQuantityTypeIdentifierVO2Max`
- Expected unit: milliliters per kilogram per minute (ml/kg/min)
- Formula: `[[HKUnit literUnitWithMetricPrefix:HKMetricPrefixMilli] unitDividedByUnit:[[HKUnit gramUnitWithMetricPrefix:HKMetricPrefixKilo] unitMultipliedByUnit:[HKUnit minuteUnit]]]`
- Source device: Apple Watch (primary source for VO2 Max estimation)

## Observed values

**Total observations**: 91,622 — **type-enforced**: 91,622 (**unique**: 40,955) — raw Python types seen: `float` (91,622).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 14.00 |
| q25 | 27.86 |
| median | 33.77 |
| mean | 33.60 |
| q75 | 39.09 |
| max | 99.00 |
| std | 8.03 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `14.00` | 246 |
| `30.99` | 33 |
| `27.08` | 32 |
| `26.59` | 31 |
| `32.84` | 31 |
| `26.43` | 30 |
| `31.08` | 30 |
| `31.67` | 30 |
| `37.20` | 30 |
| `25.92` | 29 |
| `26.12` | 29 |
| `26.41` | 29 |
| `27.68` | 29 |
| `29.16` | 29 |
| `29.42` | 29 |
| `29.60` | 29 |
| `37.50` | 29 |
| `25.65` | 28 |
| `26.14` | 28 |
| `27.01` | 28 |

_Daily-resolution variant also available in `data/labels/healthkit_daily.json`; this table reflects `last_labels.json` (nearest-per-user measurement) for API consistency._

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history
- Commits touching source file: 117 total
- Most recent material change: ce6ad36 (2020-09-30) Revert "MHC-790 Add new mobility HealthKit fields"
- Notes: VO2 Max identifier stable; reverted mobility field additions but VO2Max registration unchanged

## Notes
- HealthKit permission requested at onboarding via `requestForPermissionForType:kAPCSignUpPermissionsTypeHealthKit`
- Background delivery enabled at hourly frequency
- VO2 Max is Apple Watch Series 3+ metric calculated from workouts and activity patterns
- Represents maximum oxygen utilization capacity, key cardiovascular fitness indicator
- Part of continuous health monitoring for MyHeart Counts research
