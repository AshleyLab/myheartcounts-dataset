# onsetSmoking

**Benchmark column**: `field_onsetSmoking`
**Raw identifier**: `onsetSmoking`
**Role**: context
**Type**: continuous (age, years)

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_vaping_and_smoking_survey.json`
- Line: ~354
- Survey: `Vaping_and_smoking_survey` (Vaping and Smoking)

## Question
> How old were you when you smoked your first tobacco cigarette (in years)?

## Answer options
- **Range**: 0 to 110 years
- **Step**: unrestricted (null)
- **Unit**: years
- **UI Hint**: numberfield

## Observed values

**Total observations**: 437 — **type-enforced**: 437 (**unique**: 26) — raw Python types seen: `float` (437).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 5.00 |
| q25 | 14.00 |
| median | 16.00 |
| mean | 16.41 |
| q75 | 18.00 |
| max | 35.00 |
| std | 3.94 |

**Top 20 most frequent values**:

| value | count |
|------:|------:|
| `16.00` | 70 |
| `18.00` | 61 |
| `15.00` | 50 |
| `14.00` | 45 |
| `17.00` | 37 |
| `12.00` | 29 |
| `13.00` | 28 |
| `20.00` | 23 |
| `19.00` | 22 |
| `21.00` | 19 |
| `10.00` | 9 |
| `22.00` | 9 |
| `11.00` | 6 |
| `25.00` | 5 |
| `9.00` | 4 |
| `30.00` | 4 |
| `5.00` | 3 |
| `24.00` | 3 |
| `6.00` | 2 |
| `35.00` | 2 |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 10
- Recent material change: 9f9e14a (2020-05-04) [MHC-780] Change ; to , in smoking survey
- Notes: Stable identifier; part of original smoking survey

## Notes
Free-form numeric entry (0-110 years). Asked if participant has past smoking history.
