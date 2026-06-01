# Watch_WalkingHeartRateAverage

**Benchmark column**: `Watch_WalkingHeartRateAverage`
**Raw identifier**: `HKQuantityTypeIdentifierWalkingHeartRateAverage`
**Role**: target
**Type**: continuous

## Source
- File: `CardioHealth/Startup/APHAppDelegate.m`
- Line: 1352 (registered in `healthKitQuantityTypesToRead` method)
- Unit configuration: Line 528 (in `researcherSpecifiedUnits` method)
- Collected via: HealthKit background delivery (no user-facing question)

## Question
Not a survey variable. Collected automatically via HealthKit from Apple Watch/iPhone sensors.

## Answer options / Units
- HKQuantityTypeIdentifier: `HKQuantityTypeIdentifierWalkingHeartRateAverage`
- Expected unit: beats per minute (bpm)
- Formula: `[[HKUnit countUnit] unitDividedByUnit:[HKUnit secondUnit]]`
- Source device: Apple Watch (primary source for walking heart rate)

## Observed values

**Total observations**: 263,606 — **type-enforced**: 263,606 (**unique**: 31,254) — raw Python types seen: `float` (263,606).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 45.00 |
| q25 | 88.29 |
| median | 96.64 |
| mean | 97.08 |
| q75 | 105.2 |
| max | 202 |
| std | 13.30 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `90.00` | 454 |
| `97.00` | 450 |
| `105` | 421 |
| `98.00` | 406 |
| `97.50` | 375 |
| `104` | 330 |
| `90.50` | 294 |
| `91.00` | 291 |
| `89.50` | 287 |
| `105.5` | 285 |
| `99.00` | 277 |
| `89.00` | 276 |
| `106` | 267 |
| `82.00` | 261 |
| `96.00` | 260 |
| `113` | 257 |
| `104.5` | 253 |
| `96.50` | 234 |
| `112` | 231 |
| `83.00` | 229 |

_Daily-resolution variant also available in `data/labels/healthkit_daily.json`; this table reflects `last_labels.json` (nearest-per-user measurement) for API consistency._

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history
- Commits touching source file: 117 total
- Most recent material change: ce6ad36 (2020-09-30) Revert "MHC-790 Add new mobility HealthKit fields"
- Notes: Walking heart rate average identifier stable; reverted mobility additions but registration maintained

## Notes
- HealthKit permission requested at onboarding via `requestForPermissionForType:kAPCSignUpPermissionsTypeHealthKit`
- Background delivery enabled at hourly frequency
- Walking heart rate average measures cardiovascular response during normal walking activity
- Apple Watch Series 5+ metric; provides insight into aerobic fitness level
- Collected from daily walking patterns and light activity
- Used as baseline indicator for cardiovascular health assessment in MyHeart Counts study
