# BMI_values

**Benchmark column**: `BMI_values`
**Raw identifier**: `HKQuantityTypeIdentifierBodyMassIndex` OR computed from `HKQuantityTypeIdentifierBodyMass` + `HKQuantityTypeIdentifierHeight`
**Role**: target
**Type**: continuous

## Source
- File: `CardioHealth/Startup/APHAppDelegate.m`
- Line: N/A (not directly registered; computed from Body Mass and Height)
- Height registered: Line 1349 (in `healthKitQuantityTypesToRead` method)
- Weight registered: Line 1346 (in `healthKitQuantityTypesToRead` method)
- Collected via: HealthKit background delivery (no user-facing question)

## Question
Not a survey variable. Computed from Body Mass (kg) and Height (m) automatically via HealthKit data.

## Answer options / Units
- Computed as: Weight (kg) / Height (m²)
- Expected unit: count per square meter (kg/m²)
- Source variables: `HKQuantityTypeIdentifierBodyMass` and `HKQuantityTypeIdentifierHeight`
- Source devices: iPhone (user-entered or synced) and Apple Watch

## Observed values

**Total observations**: 22,334 — **type-enforced**: 22,334 (**unique**: 3,235) — raw Python types seen: `float` (22,334).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 10.40 |
| q25 | 23.02 |
| median | 25.85 |
| mean | 27.14 |
| q75 | 29.91 |
| max | 125 |
| std | 6.55 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `25.10` | 75 |
| `24.37` | 69 |
| `24.39` | 67 |
| `23.67` | 66 |
| `25.83` | 65 |
| `23.01` | 64 |
| `22.96` | 62 |
| `25.10` | 62 |
| `23.73` | 59 |
| `26.50` | 59 |
| `22.24` | 58 |
| `22.89` | 58 |
| `23.49` | 58 |
| `25.09` | 58 |
| `27.26` | 58 |
| `21.52` | 57 |
| `22.81` | 57 |
| `23.71` | 57 |
| `25.09` | 56 |
| `27.32` | 56 |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history
- Commits touching source file: 117 total
- Most recent material change: ce6ad36 (2020-09-30) Revert "MHC-790 Add new mobility HealthKit fields"
- Notes: BMI is derived metric; no standalone HealthKit identifier for BMI registered. Both Height and Body Mass identifiers are stable throughout app history

## Notes
- HealthKit permission requested at onboarding via `requestForPermissionForType:kAPCSignUpPermissionsTypeHealthKit`
- Background delivery enabled at hourly frequency for both Height and Body Mass
- BMI computed from two fundamental measurements collected via HealthKit
- Used in daily insights to categorize weight status (see APHDailyInsights.m reference to "Optimal: BMI of 18.5 to 24.9")
- Primary indicator for cardiovascular risk assessment in MyHeart Counts research
- Automatically updated whenever either Height or Weight changes in HealthKit
