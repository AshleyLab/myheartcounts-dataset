# happiness_categories

**Benchmark column**: `happiness_categories`
**Raw identifier**: Derived from `happiness` (continuous 0–10 scale)
**Role**: target
**Type**: ordinal

## Source
- Derivation: Post-hoc binning of happiness/well-being scores into ordinal categories
- iOS calculation: None — not computed in iOS app
- Post-hoc calculation: MHC-benchmark repo post-processing
- Input variables: `happiness` (continuous scale, typically 0–10)

## Question
Not directly asked in categorized form — derived from the happiness/well-being question in the Daily Check-in or Well-Being Survey. Users answer on a numeric 0–10 scale (often presented as "Not happy at all" to "Very happy").

## Derivation details

The happiness score (continuous 0–10) is binned into ordinal categories. The exact cutpoints are defined in the MHC-benchmark post-processing step. Likely categories include:

- **Very Unhappy** (e.g., score 0–2)
- **Unhappy** (e.g., score 3–4)
- **Neutral** (e.g., score 5)
- **Happy** (e.g., score 6–7)
- **Very Happy** (e.g., score 8–10)

Or a simpler binning (low/medium/high happiness) depending on the study protocol. See MHC-benchmark repo for the precise cutpoints and category labels.

## Observed values

**Total observations**: 4,163 — **type-enforced**: 4,117 (**unique**: 4) — raw Python types seen: `str` (4,117), `float` (46).
**Type-enforcement rejections**: 46 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `1` (High) | 1,779 | 43.2% |
| `0` (Very High) | 1,231 | 29.9% |
| `2` (Medium) | 725 | 17.6% |
| `3` (Low) | 382 | 9.3% |

**Raw stored values (top 4)** — what `context_labels.json` actually contains before type enforcement:

| raw value | count |
|-----------|------:|
| `High` | 1,779 |
| `Very High` | 1,231 |
| `Medium` | 725 |
| `Low` | 382 |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (of source/input data)
- Related input: `happiness` (documented separately, if available)
- Binning applied in MHC-benchmark post-processing

## Notes
- **Ordinal type**: categories have a natural order (very unhappy → very happy).
- This is post-hoc binning; the iOS app collects raw 0–10 scores, not categories.
- Likely used as a psychological/well-being marker.
- Exact bin edges are not in the iOS codebase; consult MHC-benchmark repo for details.
