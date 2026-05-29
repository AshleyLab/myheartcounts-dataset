# Watch_RestingHeartRate

**Benchmark column**: `Watch_RestingHeartRate`
**Raw identifier**: `HKQuantityTypeIdentifierRestingHeartRate`
**Role**: target
**Type**: continuous

## Source
- File: `CardioHealth/Startup/APHAppDelegate.m`
- Line: 1351 (registered in `healthKitQuantityTypesToRead` method)
- Unit configuration: Line 527 (in `researcherSpecifiedUnits` method)
- Collected via: HealthKit background delivery (no user-facing question)

## Question
Not a survey variable. Collected automatically via HealthKit from Apple Watch/iPhone sensors.

## Answer options / Units
- HKQuantityTypeIdentifier: `HKQuantityTypeIdentifierRestingHeartRate`
- Expected unit: beats per minute (bpm)
- Formula: `[[HKUnit countUnit] unitDividedByUnit:[HKUnit secondUnit]]`
- Source device: Apple Watch (primary source for this metric)

## Observed values

**Total observations**: 275,147 — **type-enforced**: 275,147 (**unique**: 1,356) — raw Python types seen: `float` (275,147).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 14.34 |
| q25 | 56.00 |
| median | 62.00 |
| mean | 63.04 |
| q75 | 69.00 |
| max | 175 |
| std | 9.71 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `60.00` | 11,949 |
| `59.00` | 10,479 |
| `63.00` | 9,453 |
| `58.00` | 9,252 |
| `61.00` | 8,972 |
| `57.00` | 8,750 |
| `62.00` | 8,594 |
| `66.00` | 8,459 |
| `67.00` | 8,333 |
| `64.00` | 8,152 |
| `56.00` | 8,106 |
| `68.00` | 7,577 |
| `65.00` | 7,448 |
| `55.00` | 7,175 |
| `69.00` | 6,901 |
| `54.00` | 6,672 |
| `53.00` | 6,372 |
| `70.00` | 6,077 |
| `52.00` | 5,743 |
| `71.00` | 5,568 |

_Daily-resolution variant also available in `data/labels/healthkit_daily.json`; this table reflects `last_labels.json` (nearest-per-user measurement) for API consistency._

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history
- Commits touching source file: 117 total
- Most recent material change: ce6ad36 (2020-09-30) Revert "MHC-790 Add new mobility HealthKit fields"
- Notes: Resting heart rate identifier stable since early releases; reverted mobility changes in Sept 2020 but resting heart rate registration unchanged

## Notes
- HealthKit permission requested at onboarding via `requestForPermissionForType:kAPCSignUpPermissionsTypeHealthKit`
- Background delivery enabled at hourly frequency during app initialization
- Resting heart rate is Apple Watch-specific metric typically collected during sleep or inactivity periods
- Part of continuous cardiovascular monitoring for MyHeart Counts research study
