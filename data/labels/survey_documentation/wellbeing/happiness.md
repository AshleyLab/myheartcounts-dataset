# happiness

**Benchmark column**: `happiness`
**Raw identifier**: `happiness`
**Role**: target
**Type**: continuous (0–10)

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_daily_check_coaching.json`
- Line: ~9
- Survey: `daily_check` (coaching variant)

## Question
> Happiness

**Detail**: This question asks about how you felt yesterday on a scale from 0 to 10. Zero means you did not experience the feeling "at all" yesterday while 10 means you experienced the feeling "all of the time" yesterday.

## Answer options
| Value | Label |
|-------|-------|
| 0–10 | Slider (1-point increments) |

## Observed values

**Total observations**: 39,766 — **type-enforced**: 39,766 (**unique**: 11) — raw Python types seen: `float` (39,766).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| stat | value |
|------|------:|
| min | 0 |
| q25 | 7.00 |
| median | 8.00 |
| mean | 7.56 |
| q75 | 9.00 |
| max | 10.00 |
| std | 1.99 |

**Top 11 most frequent values**:

| value | count |
|------:|------:|
| `8.00` | 10,263 |
| `9.00` | 8,275 |
| `7.00` | 6,396 |
| `10.00` | 5,714 |
| `6.00` | 3,370 |
| `5.00` | 2,556 |
| `4.00` | 1,293 |
| `3.00` | 846 |
| `2.00` | 501 |
| `0` | 308 |
| `1.00` | 244 |

_Daily-resolution variant also available in `data/labels/healthkit_daily.json`; this table reflects `last_labels.json` (nearest-per-user measurement) for API consistency._

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 4 (daily_check_coaching.json)
- Recent material change: `50743bd` (MHC-126 - Create Alternative Daily survey 2 - change the happiness prompt on coaching daily survey)
- Notes: Added in coaching variant; the happiness prompt was updated in MHC-126

## Notes
This is the target variable in the coaching variant of the daily check-in survey. It measures subjective happiness/well-being on a 0–10 scale for the prior day, collected daily via a slider. This is distinct from other well-being variables in different surveys (e.g., `feel_worthwhile1`, `feel_worthwhile2` from the well-being survey). The coaching survey captures happiness as a key outcome for evaluating intervention impact.
