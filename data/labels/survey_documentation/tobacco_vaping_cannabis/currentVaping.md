# currentVaping

**Benchmark column**: `field_currentVaping`
**Raw identifier**: `currentVaping`
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json`
- Line: ~8
- Survey: `Vaping_and_smoking_survey` (Vaping and Smoking)

## Question
> Do you currently vape nicotine (use "e-cigs", "mods", "vape pens", "vapes", "JUULs", etc.)?

## Answer options
| Value | Label |
|-------|-------|
| 1 | Daily |
| 2 | Less than daily |
| 3 | Not at all |
| 4 | Don't know |

## Observed values

**Total observations**: 1,330 — **type-enforced**: 1,330 (**unique**: 4) — raw Python types seen: `float` (1,330).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `3` | 1,231 | 92.6% |
| `1` | 59 | 4.4% |
| `2` | 30 | 2.3% |
| `4` | 10 | 0.8% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 10
- Recent material change: 9f9e14a (2020-05-04) [MHC-780] Change ; to , in smoking survey
- Notes: Last changes to survey structure on 2020-05-04; no targeted changes to currentVaping identifier in recent history

## Notes
Gating logic: If value is 3 ("Not at all") or 4 ("Don't know"), skip to `pastVaping` question.
