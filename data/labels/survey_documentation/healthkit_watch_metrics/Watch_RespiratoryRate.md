# Watch_RespiratoryRate

**Benchmark column**: `Watch_RespiratoryRate`
**Raw identifier**: `HKQuantityTypeIdentifierRespiratoryRate`
**Role**: target
**Type**: continuous

## Source
- File: `CardioHealth/Startup/APHAppDelegate.m`
- Line: 1362 (registered in `healthKitQuantityTypesToRead` method)
- Unit configuration: Line 537 (in `researcherSpecifiedUnits` method)
- Collected via: HealthKit background delivery (no user-facing question)

## Question
Not a survey variable. Collected automatically via HealthKit from Apple Watch/iPhone sensors.

## Answer options / Units
- HKQuantityTypeIdentifier: `HKQuantityTypeIdentifierRespiratoryRate`
- Expected unit: breaths per minute
- Formula: `[[HKUnit countUnit] unitDividedByUnit:[HKUnit secondUnit]]`
- Source device: Apple Watch (via motion and heart rate sensors)

## Observed values

**Total observations**: 72,605 — **type-enforced**: 72,605 (**unique**: 3,265) — raw Python types seen: `float` (72,605).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | -1.00 |
| q25 | 14.00 |
| median | 15.75 |
| mean | 16.25 |
| q75 | 18.00 |
| max | 106 |
| std | 3.17 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `15.00` | 4,450 |
| `14.00` | 3,471 |
| `14.50` | 3,366 |
| `17.00` | 3,342 |
| `13.00` | 3,261 |
| `15.50` | 3,102 |
| `13.50` | 2,894 |
| `16.00` | 2,822 |
| `16.50` | 2,638 |
| `17.50` | 2,164 |
| `12.50` | 1,818 |
| `18.00` | 1,625 |
| `19.00` | 1,423 |
| `18.50` | 1,394 |
| `12.00` | 1,162 |
| `11.50` | 1,041 |
| `20.00` | 986 |
| `20.50` | 953 |
| `19.50` | 900 |
| `15.25` | 782 |

_Daily-resolution variant also available in `data/labels/healthkit_daily.json`; this table reflects `last_labels.json` (nearest-per-user measurement) for API consistency._

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history
- Commits touching source file: 117 total
- Most recent material change: ce6ad36 (2020-09-30) Revert "MHC-790 Add new mobility HealthKit fields"
- Notes: Respiratory rate identifier stable; reverted mobility additions maintained this registration

## Notes
- HealthKit permission requested at onboarding via `requestForPermissionForType:kAPCSignUpPermissionsTypeHealthKit`
- Background delivery enabled at hourly frequency
- Respiratory rate measured via Apple Watch motion and optical sensors
- Typically collected during rest periods; Apple Watch Series 6+ provides more accurate readings
- Important cardiovascular health metric; elevated respiratory rate may indicate stress or poor fitness
- Included in MyHeart Counts comprehensive vital signs monitoring for research
