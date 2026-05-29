# chronic_illness_positive_opportunity

**Benchmark column**: `field_chronic_illness_positive_opportunity`
**Raw identifier**: `chronic_illness_positive_opportunity`
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_illness_mindset_measure_inventory_survey.json`
- Line: 196
- Survey: `Illness_mindset_inventory` (Illness Mindset Inventory / IMI)

## Question
> A chronic illness can be an opportunity to make positive life changes.

## Answer options
| Value | Label |
|-------|-------|
| 1 | Strongly Agree |
| 2 | Agree |
| 3 | Somewhat Agree |
| 4 | Somewhat Disagree |
| 5 | Disagree |
| 6 | Strongly Disagree |

## Observed values

**Total observations**: 1,007 — **type-enforced**: 1,007 (**unique**: 6) — raw Python types seen: `float` (1,007).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `2` | 396 | 39.3% |
| `1` | 276 | 27.4% |
| `3` | 264 | 26.2% |
| `4` | 46 | 4.6% |
| `5` | 16 | 1.6% |
| `6` | 9 | 0.9% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 3
- Recent material change: 5043960 (2019-01-23) MHC-610 Add Illness Mindset Inventory Survey
- Notes: Initial addition of survey; no targeted changes since

## Notes
Item 5 from the Illness Mindset Inventory (IMI), a 21-item battery assessing beliefs about chronic illness and body self-healing capacity. All items use a 6-point agreement scale (Strongly Agree to Strongly Disagree).
