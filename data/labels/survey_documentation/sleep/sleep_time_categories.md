# sleep_time_categories

**Benchmark column**: `sleep_time_categories`
**Raw identifier**: Derived from `sleep_time` (continuous hours)
**Role**: target
**Type**: ordinal

## Source
- Derivation: Post-hoc binning of sleep duration into ordinal categories
- iOS calculation: None — not computed in iOS app
- Post-hoc calculation: MHC-benchmark repo post-processing
- Input variables: `sleep_time` (hours slept, continuous)

## Question
Not directly asked in categorized form — derived from the sleep duration response in the Daily Check-in or sleep survey. Users typically enter hours of sleep (e.g., 6, 7.5, 8, etc.).

## Derivation details

Sleep duration (continuous hours) is binned into ordinal categories. The exact cutpoints are defined in the MHC-benchmark post-processing step. Likely categories include:

- **Insufficient** (e.g., < 5 hours)
- **Low** (e.g., 5–6 hours)
- **Normal** (e.g., 6–8 hours)
- **High** (e.g., 8–9 hours)
- **Excessive** (e.g., > 9 hours)

Or align with public health recommendations (e.g., "Short Sleep" < 7 hours, "Recommended" 7–9 hours, "Long Sleep" > 9 hours). See MHC-benchmark repo for the precise cutpoints and category labels.

## Observed values

**Total observations**: 44,149 — **type-enforced**: 44,133 (**unique**: 4) — raw Python types seen: `str` (44,133), `float` (16).
**Type-enforcement rejections**: 16 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `0` (Normal) | 26,723 | 60.6% |
| `1` (Short) | 10,483 | 23.8% |
| `3` (Insufficient) | 4,646 | 10.5% |
| `2` (Too Long) | 2,281 | 5.2% |

**Raw stored values (top 4)** — what `context_labels.json` actually contains before type enforcement:

| raw value | count |
|-----------|------:|
| `Normal` | 26,723 |
| `Short` | 10,483 |
| `Insufficient` | 4,646 |
| `Too Long` | 2,281 |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (of source/input data)
- Related input: `sleep_time` (documented separately, if available)
- Binning applied in MHC-benchmark post-processing

## Notes
- **Ordinal type**: categories have a natural order (insufficient → excessive sleep duration).
- This is post-hoc binning; the iOS app collects raw hours, not categories.
- Likely used as a health marker; note that both short and long sleep are associated with adverse outcomes.
- Exact bin edges are not in the iOS codebase; consult MHC-benchmark repo for details.
- **Filename note**: The benchmark column name is truncated (`categorie` not `categories`); this is preserved exactly as it appears in the dataset.
