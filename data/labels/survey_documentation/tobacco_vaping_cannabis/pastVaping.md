# pastVaping

**Benchmark column**: `field_pastVaping`
**Raw identifier**: `pastVaping`
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json`
- Line: ~78
- Survey: `Vaping_and_smoking_survey` (Vaping and Smoking)

## Question
> Have you vaped in the past?

## Answer options
| Value | Label |
|-------|-------|
| 1 | Yes |
| 2 | No |
| 3 | Don't know |

## Observed values

**Total observations**: 1,242 — **type-enforced**: 1,242 (**unique**: 3) — raw Python types seen: `float` (1,242).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `2` | 1,111 | 89.5% |
| `1` | 127 | 10.2% |
| `3` | 4 | 0.3% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 10
- Recent material change: 9f9e14a (2020-05-04) [MHC-780] Change ; to , in smoking survey
- Notes: Stable identifier; part of original survey since MHC-614

## Notes
Gating logic: If value is 2 ("No") or 3 ("Don't know"), skip to `currentSmoking` question. If currentVaping was "Not at all" or "Don't know", participant is routed directly to pastVaping.
