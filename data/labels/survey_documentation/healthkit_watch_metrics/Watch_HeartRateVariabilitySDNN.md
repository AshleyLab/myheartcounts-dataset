# Watch_HeartRateVariabilitySDNN

**Benchmark column**: `Watch_HeartRateVariabilitySDNN`
**Raw identifier**: `HKQuantityTypeIdentifierHeartRateVariabilitySDNN`
**Role**: target
**Type**: continuous

## Source
- File: `CardioHealth/Startup/APHAppDelegate.m`
- Line: 1353 (registered in `healthKitQuantityTypesToRead` method)
- Unit configuration: Line 529 (in `researcherSpecifiedUnits` method)
- Collected via: HealthKit background delivery (no user-facing question)

## Question
Not a survey variable. Collected automatically via HealthKit from Apple Watch/iPhone sensors.

## Answer options / Units
- HKQuantityTypeIdentifier: `HKQuantityTypeIdentifierHeartRateVariabilitySDNN`
- Expected unit: milliseconds (ms)
- Formula: `[HKUnit secondUnitWithMetricPrefix:HKMetricPrefixMilli]`
- Source device: Apple Watch (primary source for HRV measurement)

## Observed values

**Total observations**: 264,783 — **type-enforced**: 264,783 (**unique**: 226,319) — raw Python types seen: `float` (264,783).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 0.002341 |
| q25 | 22.87 |
| median | 29.60 |
| mean | 34.53 |
| q75 | 39.23 |
| max | 444 |
| std | 23.44 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `19.30` | 13 |
| `20.64` | 11 |
| `22.16` | 11 |
| `23.95` | 11 |
| `24.71` | 11 |
| `28.41` | 11 |
| `28.69` | 11 |
| `18.79` | 10 |
| `22.18` | 10 |
| `22.75` | 10 |
| `23.03` | 10 |
| `23.92` | 10 |
| `23.93` | 10 |
| `23.96` | 10 |
| `24.41` | 10 |
| `24.71` | 10 |
| `25.91` | 10 |
| `26.46` | 10 |
| `26.58` | 10 |
| `26.83` | 10 |

_Daily-resolution variant also available in `data/labels/healthkit_daily.json`; this table reflects `last_labels.json` (nearest-per-user measurement) for API consistency._

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history
- Commits touching source file: 117 total
- Most recent material change: ce6ad36 (2020-09-30) Revert "MHC-790 Add new mobility HealthKit fields"
- Notes: Heart Rate Variability SDNN (Standard Deviation of NN intervals) identifier stable throughout app history

## Notes
- HealthKit permission requested at onboarding via `requestForPermissionForType:kAPCSignUpPermissionsTypeHealthKit`
- Background delivery enabled at hourly frequency
- SDNN (Standard Deviation of Normal-to-Normal intervals) measures variability in time between heartbeats
- Indicates autonomic nervous system activity; higher HRV typically associated with better cardiovascular fitness
- Collected from Apple Watch during sleep and rest periods
- Part of MyHeart Counts cardiovascular research monitoring
