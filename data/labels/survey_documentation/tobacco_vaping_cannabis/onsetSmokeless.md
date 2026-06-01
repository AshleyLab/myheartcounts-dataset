# onsetSmokeless

**Benchmark column**: `field_onsetSmokeless`
**Raw identifier**: `onsetSmokeless`
**Role**: context
**Type**: continuous (age, years)

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json`
- Line: ~521
- Survey: `Vaping_and_smoking_survey` (Vaping and Smoking)

## Question
> How old were you first used smokeless tobacco (chewing tobacco, snuff, snus, and dissolvable tobacco products)?

## Answer options
- **Range**: 0 to 110 years
- **Step**: unrestricted (null)
- **Unit**: years
- **UI Hint**: numberfield

## Observed values

**Total observations**: 23 — **type-enforced**: 23 (**unique**: 18) — raw Python types seen: `float` (23).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 2.00 |
| q25 | 16.50 |
| median | 21.00 |
| mean | 27.83 |
| q75 | 39.00 |
| max | 80.00 |
| std | 20.22 |

**Top 18 most frequent values**:

| value | count |
|------:|------:|
| `18.00` | 3 |
| `5.00` | 2 |
| `14.00` | 2 |
| `19.00` | 2 |
| `2.00` | 1 |
| `15.00` | 1 |
| `21.00` | 1 |
| `22.00` | 1 |
| `23.00` | 1 |
| `24.00` | 1 |
| `32.00` | 1 |
| `38.00` | 1 |
| `40.00` | 1 |
| `43.00` | 1 |
| `46.00` | 1 |
| `50.00` | 1 |
| `74.00` | 1 |
| `80.00` | 1 |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 10
- Recent material change: 9f9e14a (2020-05-04) [MHC-780] Change ; to , in smoking survey
- Notes: Stable identifier; smokeless tobacco module added in MHC-756

## Notes
Free-form numeric entry (0-110 years). Asked if participant has past smokeless tobacco use. Note: prompt has minor grammar issue ("first used" instead of "when you first used").
