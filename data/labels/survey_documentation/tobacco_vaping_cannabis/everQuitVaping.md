# everQuitVaping

**Benchmark column**: `field_everQuitVaping`
**Raw identifier**: `everQuitVaping`
**Role**: context
**Type**: binary

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json`
- Line: ~136
- Survey: `Vaping_and_smoking_survey` (Vaping and Smoking)

## Question
> During the past 12 months, have you tried to stop vaping?

## Answer options
- **Type**: Boolean (checkbox)
- **Encoding**: 0 = No, 1 = Yes

## Observed values

**Total observations**: 216 — **type-enforced**: 216 (**unique**: 2) — raw Python types seen: `bool` (216).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `False` | 137 | 63.4% |
| `True` | 79 | 36.6% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 10
- Recent material change: 9f9e14a (2020-05-04) [MHC-780] Change ; to , in smoking survey
- Notes: Stable identifier; part of original vaping survey

## Notes
Gating logic: If value is 0 (No), skip to `currentSmoking`. If Yes (1), continues to durationQuitVaping.
