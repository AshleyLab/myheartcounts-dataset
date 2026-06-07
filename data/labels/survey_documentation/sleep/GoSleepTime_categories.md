# GoSleepTime_categories

**Benchmark column**: `GoSleepTime_categories`
**Raw identifier**: Derived from `GoSleepTime` (time of day in HH:MM format)
**Role**: target
**Type**: ordinal

## Source
- Derivation: Post-hoc binning of bedtime (go-to-sleep time) into time-of-day categories
- iOS calculation: None — not computed in iOS app
- Post-hoc calculation: MHC-benchmark repo post-processing
- Input variables: `GoSleepTime` (time of day, 24-hour format)

## Question
Not directly asked — derived from the bedtime response in the Daily Check-in or sleep survey.

## Derivation details

Bedtime (go-to-sleep time) is classified into ordinal time-of-day categories. The exact bin boundaries are defined in the MHC-benchmark post-processing step. Likely categories include:

- **Very Early Bedtime** (e.g., before 9:00 PM)
- **Early Bedtime** (e.g., 9:00–10:00 PM)
- **Regular Bedtime** (e.g., 10:00–11:00 PM)
- **Late Bedtime** (e.g., 11:00 PM–12:00 AM)
- **Very Late Bedtime** (e.g., after 12:00 AM)

Or a simpler binning (early/normal/late) depending on the study protocol. See MHC-benchmark repo for the precise cutpoints and category labels.

## Observed values

**Total observations**: 25,349 — **type-enforced**: 24,799 (**unique**: 5) — raw Python types seen: `str` (24,799), `float` (550).
**Type-enforcement rejections**: 550 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `4` (Shift Worker) | 8,018 | 32.3% |
| `1` (Early Sleeper) | 6,633 | 26.7% |
| `0` (Normal Sleeper) | 6,628 | 26.7% |
| `2` (Late Sleeper) | 2,370 | 9.6% |
| `3` (Very Late Sleeper) | 1,150 | 4.6% |

**Raw stored values (top 5)** — what `context_labels.json` actually contains before type enforcement:

| raw value | count |
|-----------|------:|
| `Shift Worker` | 8,018 |
| `Early Sleeper` | 6,633 |
| `Normal Sleeper` | 6,628 |
| `Late Sleeper` | 2,370 |
| `Very Late Sleeper` | 1,150 |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (of source/input data)
- Related input: `GoSleepTime` (documented separately, if available)
- Binning applied in MHC-benchmark post-processing

## Notes
- **Ordinal type**: categories have a natural order (earlier bedtime → later bedtime).
- This is post-hoc binning; the iOS app collects raw bedtime, not categories.
- Likely used as a behavioral/lifestyle marker for sleep and circadian health.
- Exact bin edges are not in the iOS codebase; consult MHC-benchmark repo for details.
- Cross-reference: see `WakeUpTime_categories.md` for the related wake-up time variable.
