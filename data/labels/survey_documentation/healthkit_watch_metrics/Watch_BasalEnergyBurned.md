# Watch_BasalEnergyBurned

**Benchmark column**: `Watch_BasalEnergyBurned`
**Raw identifier**: `HKQuantityTypeIdentifierBasalEnergyBurned`
**Role**: target
**Type**: continuous

## Source
- File: `CardioHealth/Startup/APHAppDelegate.m`
- Line: 1348 (registered in `healthKitQuantityTypesToRead` method)
- Unit configuration: Line 523 (in `researcherSpecifiedUnits` method)
- Collected via: HealthKit background delivery (no user-facing question)

## Question
Not a survey variable. Collected automatically via HealthKit from Apple Watch/iPhone sensors.

## Answer options / Units
- HKQuantityTypeIdentifier: `HKQuantityTypeIdentifierBasalEnergyBurned`
- Expected unit: kilocalories (kcal)
- Formula: `[HKUnit smallCalorieUnit]`
- Source device: Apple Watch (primary source for basal metabolic rate estimation)

## Observed values

**Total observations**: 381,354 — **type-enforced**: 381,354 (**unique**: 359,156) — raw Python types seen: `float` (381,354).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | -4666 |
| q25 | 1455 |
| median | 1784 |
| mean | 1921 |
| q75 | 2181 |
| max | 9989 |
| std | 992.1 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `1595` | 134 |
| `1697` | 130 |
| `1529` | 106 |
| `1286` | 105 |
| `1600` | 105 |
| `3148` | 102 |
| `3138` | 90 |
| `1753` | 88 |
| `1782` | 88 |
| `1577` | 86 |
| `1524` | 83 |
| `1633` | 78 |
| `1816` | 76 |
| `1340` | 75 |
| `1508` | 75 |
| `1779` | 75 |
| `1814` | 74 |
| `1534` | 72 |
| `1513` | 71 |
| `1281` | 70 |

_Daily-resolution variant also available in `data/labels/healthkit_daily.json`; this table reflects `last_labels.json` (nearest-per-user measurement) for API consistency._

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history
- Commits touching source file: 117 total
- Most recent material change: ce6ad36 (2020-09-30) Revert "MHC-790 Add new mobility HealthKit fields"
- Notes: Basal energy identifier stable throughout app history; reverted mobility changes did not affect registration

## Notes
- HealthKit permission requested at onboarding via `requestForPermissionForType:kAPCSignUpPermissionsTypeHealthKit`
- Background delivery enabled at hourly frequency
- Basal energy burned represents resting metabolic rate (calories burned at rest)
- Calculated based on age, weight, height, and biological sex
- Apple Watch continuously estimates basal energy using motion and heart rate data
- Part of comprehensive energy expenditure tracking for cardiovascular health assessment in MyHeart Counts
