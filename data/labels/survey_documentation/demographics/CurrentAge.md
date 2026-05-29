# CurrentAge

> **Not in released benchmark.** Reason: redundancy — duplicate of target `age`. See `data/labels/RELEASE_NOTES.md` for the full disposition table.


**Benchmark column**: `CurrentAge` / `field_CurrentAge`
**Raw identifier**: Computed from user's date of birth (`APCUser.dateOfBirth`)
**Obj-C constant**: No direct constant; computed dynamically
**Role**: context
**Type**: continuous

## Source
- Computed in: `APCUser+Age.h` (imported at line 37 of `APHHeartAgeTaskViewController.m`)
- Framingham used value: `age` from `[[APCAppDelegate sharedAppDelegate].dataSubstrate.currentUser age]` (line 684)
- Not a direct UI field; pre-populated when user selects "Are you submitting your own heart risk data?" = YES

## Question
> (Not user-facing — computed from date of birth in HealthKit)

## Answer options
N/A — Computed age, not user input.

## Observed values

**Total observations**: 10,405 — **type-enforced**: 10,405 (**unique**: 77) — raw Python types seen: `float` (10,405).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 12.00 |
| q25 | 29.00 |
| median | 38.00 |
| mean | 41.32 |
| q75 | 51.00 |
| max | 130 |
| std | 15.32 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `30.00` | 396 |
| `31.00` | 305 |
| `32.00` | 295 |
| `35.00` | 294 |
| `33.00` | 293 |
| `29.00` | 283 |
| `28.00` | 281 |
| `25.00` | 269 |
| `34.00` | 269 |
| `26.00` | 265 |
| `37.00` | 264 |
| `36.00` | 258 |
| `27.00` | 257 |
| `39.00` | 256 |
| `42.00` | 248 |
| `24.00` | 236 |
| `38.00` | 233 |
| `43.00` | 226 |
| `22.00` | 222 |
| `44.00` | 220 |

_Generated 2026-04-24 from `data/labels/last_labels.json` (md5 `f280e307…`) and `data/labels/context_labels.json` (md5 `f0ec00c9…`)._

## Git history
- View controller commits: `dbdd5a0` (MHC-508), `a2c3b1e` (MHC-86), `2a31f49` (MHC-327 pre-population logic)
- Recent material change: `dbdd5a0` (MHC-508)

## Notes
- `CurrentAge` is not directly entered by the user; it is **computed from the user's stored date of birth** via the `APCUser.age` property (HealthKit `HKCharacteristicTypeIdentifierDateOfBirth`).
- When the user answers "Are you submitting your own heart risk data?" = YES, the app pre-fills the age field with this computed current age (line 684).
- The distinction between `Age_heartage` (self-reported age in the form) and `CurrentAge` (age derived from DoB) is important for data quality and discrepancy tracking. The form allows the user to override or correct the computed value.
- The `APCUser+Age` category is imported, suggesting it provides a convenience method to compute age from the stored HealthKit date of birth.
- In the final results, the **entered/corrected age** (from the form, stored as `kHeartAgeTestDataAge`) is used for Framingham calculation, not the pre-filled `CurrentAge`.
