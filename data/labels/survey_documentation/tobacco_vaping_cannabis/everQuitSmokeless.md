# everQuitSmokeless

**Benchmark column**: `field_everQuitSmokeless`
**Raw identifier**: `everQuitSmokeless`
**Role**: context
**Type**: binary (source is `string` with values `'true'`/`'false'`/`'nan'`; the build extractor maps `'nan'` → null and the API's `_to_bool` converts `'true'`/`'false'`)

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json`
- Line: ~595
- Survey: `Vaping_and_smoking_survey` (Vaping and Smoking)

## Question
> During the past 12 months, have you tried to stop chewing tobacco (chewing tobacco, snuff, snus, and dissolvable tobacco products)?

## Answer options
- **Type**: Boolean (checkbox)
- **Encoding**: 0 = No, 1 = Yes

## Observed values

**Total observations**: 106 — **type-enforced**: 106 (**unique**: 2) — raw Python types seen: `str` (106).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `False` | 82 | 77.4% |
| `True` | 24 | 22.6% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 10
- Recent material change: 9f9e14a (2020-05-04) [MHC-780] Change ; to , in smoking survey
- Notes: Stable identifier; smokeless tobacco module added in MHC-756

## Notes
Gating logic: If value is 0 (No), skip to `tobaccoProducts`. If Yes (1), continues to durationQuitSmokeless.
