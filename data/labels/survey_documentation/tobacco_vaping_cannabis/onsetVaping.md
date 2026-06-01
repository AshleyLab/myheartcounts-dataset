# onsetVaping

**Benchmark column**: `field_onsetVaping`
**Raw identifier**: `onsetVaping`
**Role**: context
**Type**: continuous (age, years)

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json`
- Line: ~122
- Survey: `Vaping_and_smoking_survey` (Vaping and Smoking)

## Question
> How old were you when you first vaped (in years)?

## Answer options
- **Range**: 0 to 110 years
- **Step**: unrestricted (null)
- **Unit**: years
- **UI Hint**: numberfield

## Observed values

**Total observations**: 215 — **type-enforced**: 215 (**unique**: 56) — raw Python types seen: `float` (215).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 1.00 |
| q25 | 20.00 |
| median | 29.00 |
| mean | 30.58 |
| q75 | 38.00 |
| max | 78.00 |
| std | 13.38 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `20.00` | 17 |
| `30.00` | 13 |
| `25.00` | 10 |
| `40.00` | 9 |
| `18.00` | 8 |
| `22.00` | 8 |
| `28.00` | 8 |
| `16.00` | 7 |
| `19.00` | 7 |
| `27.00` | 7 |
| `36.00` | 7 |
| `21.00` | 6 |
| `31.00` | 6 |
| `32.00` | 6 |
| `35.00` | 6 |
| `42.00` | 6 |
| `17.00` | 5 |
| `26.00` | 5 |
| `34.00` | 5 |
| `38.00` | 5 |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 10
- Recent material change: 9f9e14a (2020-05-04) [MHC-780] Change ; to , in smoking survey
- Notes: Stable identifier; part of original vaping survey

## Notes
Free-form numeric entry (0-110 years). Asked if participant has past vaping history.
