# pastSmokeless

**Benchmark column**: `field_pastSmokeless`
**Raw identifier**: `pastSmokeless`
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json`
- Line: ~535
- Survey: `Vaping_and_smoking_survey` (Vaping and Smoking)

## Question
> Have you used smokeless tobacco (chewing tobacco, snuff, snus, and dissolvable tobacco products) in the past?

## Answer options
| Value | Label |
|-------|-------|
| 1 | Yes |
| 2 | No |
| 3 | Don't know |

## Observed values

**Total observations**: 1,328 — **type-enforced**: 1,328 (**unique**: 3) — raw Python types seen: `float` (1,328).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `2` | 1,209 | 91.0% |
| `1` | 111 | 8.4% |
| `3` | 8 | 0.6% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 10
- Recent material change: 9f9e14a (2020-05-04) [MHC-780] Change ; to , in smoking survey
- Notes: Stable identifier; smokeless tobacco module added in MHC-756

## Notes
Gating logic: If value is 2 ("No") or 3 ("Don't know"), skip to `tobaccoProducts` question. If currentSmokeless was "Not at all" or "Don't know", participant is routed directly to pastSmokeless.
