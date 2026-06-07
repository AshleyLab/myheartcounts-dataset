# Watch_StandTime

**Benchmark column**: `Watch_StandTime`
**Raw identifier**: `HKQuantityTypeIdentifierAppleStandTime`
**Role**: target
**Type**: continuous

## Source
- File: `CardioHealth/Startup/APHAppDelegate.m`
- Line: 1377 (registered in `healthKitQuantityTypesToRead` method, iOS 13.0+ only)
- Unit configuration: Line 557 (in `researcherSpecifiedUnits` method, iOS 13.0+ only)
- Collected via: HealthKit background delivery (no user-facing question)

## Question
Not a survey variable. Collected automatically via HealthKit from Apple Watch/iPhone sensors.

## Answer options / Units
- HKQuantityTypeIdentifier: `HKQuantityTypeIdentifierAppleStandTime`
- Expected unit: minutes
- Source device: Apple Watch (tracks stand hours throughout the day)
- Availability: iOS 13.0+ only (conditional registration in `researcherSpecifiedUnits`)

## Observed values

**Total observations**: 180,442 — **type-enforced**: 180,442 (**unique**: 7,436) — raw Python types seen: `float` (180,442).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 0.02 |
| q25 | 1.13 |
| median | 1.68 |
| mean | 1.79 |
| q75 | 2.31 |
| max | 18.35 |
| std | 0.93 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `1.50` | 268 |
| `1.45` | 247 |
| `1.20` | 244 |
| `1.27` | 244 |
| `1.23` | 243 |
| `1.43` | 242 |
| `1.77` | 239 |
| `1.32` | 236 |
| `1.15` | 235 |
| `1.62` | 235 |
| `1.25` | 232 |
| `1.57` | 232 |
| `1.37` | 229 |
| `1.05` | 228 |
| `1.30` | 228 |
| `1.52` | 228 |
| `1.38` | 226 |
| `1.00` | 223 |
| `1.12` | 223 |
| `1.40` | 223 |

_Daily-resolution variant also available in `data/labels/healthkit_daily.json`; this table reflects `last_labels.json` (nearest-per-user measurement) for API consistency._

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history
- Commits touching source file: 117 total
- Most recent material change: ce6ad36 (2020-09-30) Revert "MHC-790 Add new mobility HealthKit fields"
- Notes: Stand time metric added for iOS 13.0+ support; reverted mobility changes in Sept 2020 maintained this support

## Notes
- HealthKit permission requested at onboarding via `requestForPermissionForType:kAPCSignUpPermissionsTypeHealthKit`
- Background delivery enabled at hourly frequency
- Apple Watch tracks stand time as minutes user spent standing during each hour
- Complements Move Ring and Exercise Ring for overall activity monitoring
- Part of cardiovascular health tracking in MyHeart Counts research
- iOS 13.0+ feature conditional availability ensures backward compatibility
