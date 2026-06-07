# WakeUpTime_categories

**Benchmark column**: `WakeUpTime_categories`
**Raw identifier**: Derived from `WakeUpTime` (time of day in HH:MM format)
**Role**: target
**Type**: ordinal

## Source
- Derivation: Post-hoc binning of wake-up time into time-of-day categories
- iOS calculation: None — not computed in iOS app
- Post-hoc calculation: MHC-benchmark repo post-processing
- Input variables: `WakeUpTime` (time of day, 24-hour format)

## Question
Not directly asked — derived from the wake-up time response in the Daily Check-in or sleep survey.

## Derivation details

Wake-up time is classified into ordinal time-of-day categories. The exact bin boundaries are defined in the MHC-benchmark post-processing step. Likely categories include:

- **Very Early** (e.g., before 6:00 AM)
- **Early** (e.g., 6:00–7:00 AM)
- **Morning** (e.g., 7:00–8:30 AM)
- **Late Morning** (e.g., 8:30–10:00 AM)
- **Late** (e.g., 10:00 AM or later)

Or a simpler binning (early/normal/late) depending on the study protocol. See MHC-benchmark repo for the precise cutpoints and category labels.

## Observed values

**Total observations**: 25,262 — **type-enforced**: 24,617 (**unique**: 4) — raw Python types seen: `str` (24,617), `float` (645).
**Type-enforcement rejections**: 645 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `1` (Early Riser) | 10,590 | 43.0% |
| `0` (Normal Riser) | 6,011 | 24.4% |
| `3` (Very Late Riser) | 5,636 | 22.9% |
| `2` (Late Riser) | 2,380 | 9.7% |

**Raw stored values (top 4)** — what `context_labels.json` actually contains before type enforcement:

| raw value | count |
|-----------|------:|
| `Early Riser` | 10,590 |
| `Normal Riser` | 6,011 |
| `Very Late Riser` | 5,636 |
| `Late Riser` | 2,380 |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (of source/input data)
- Related input: `WakeUpTime` (documented separately, if available)
- Binning applied in MHC-benchmark post-processing

## Notes
- **Ordinal type**: categories have a natural order (early → late in the day).
- This is post-hoc binning; the iOS app collects raw wake-up time, not categories.
- Likely used as a behavioral/lifestyle marker for sleep and circadian health.
- Exact bin edges are not in the iOS codebase; consult MHC-benchmark repo for details.
