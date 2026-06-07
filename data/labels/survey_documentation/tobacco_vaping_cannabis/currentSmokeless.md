# currentSmokeless

**Benchmark column**: `field_currentSmokeless`
**Raw identifier**: `currentSmokeless`
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json`
- Line: ~472
- Survey: `Vaping_and_smoking_survey` (Vaping and Smoking)

## Question
> Do you currently use smokeless tobacco (chewing tobacco, snuff, snus, and dissolvable tobacco products)?

## Answer options
| Value | Label |
|-------|-------|
| 1 | Daily |
| 2 | Less than daily |
| 3 | Not at all |
| 4 | Don't know |

## Observed values

**Total observations**: 1,328 — **type-enforced**: 1,328 (**unique**: 4) — raw Python types seen: `float` (1,328).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `3` | 1,305 | 98.3% |
| `1` | 13 | 1.0% |
| `2` | 9 | 0.7% |
| `4` | 1 | 0.1% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 10
- Recent material change: 9f9e14a (2020-05-04) [MHC-780] Change ; to , in smoking survey
- Notes: Stable identifier; smokeless tobacco module added in MHC-756

## Notes
Gating logic: If value is 3 ("Not at all") or 4 ("Don't know"), skip to `pastSmokeless` question. Central branching point for smokeless tobacco module.
