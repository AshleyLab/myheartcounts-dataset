# chronic_illness_runing_life

**Benchmark column**: `field_chronic_illness_runing_life`
**Raw identifier**: `chronic_illness_runing_life` (NOTE: preserves "runing" typo from source)
**Role**: context
**Type**: ordinal

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_illness_mindset_measure_inventory_survey.json`
- Line: 601
- Survey: `Illness_mindset_inventory` (Illness Mindset Inventory / IMI)

## Question
> A chronic illness ruins most aspects of life.

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

**Total observations**: 999 — **type-enforced**: 999 (**unique**: 6) — raw Python types seen: `float` (999).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `4` | 349 | 34.9% |
| `3` | 266 | 26.6% |
| `5` | 224 | 22.4% |
| `2` | 69 | 6.9% |
| `6` | 63 | 6.3% |
| `1` | 28 | 2.8% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Commits: 3
- Recent material change: 5043960 (2019-01-23) MHC-610 Add Illness Mindset Inventory Survey
- Notes: Initial addition of survey; no targeted changes since

## Notes
Item 14 from the Illness Mindset Inventory (IMI), a 21-item battery assessing beliefs about chronic illness and body self-healing capacity. All items use a 6-point agreement scale (Strongly Agree to Strongly Disagree). The identifier contains a typo ("runing" instead of "running") that is preserved from the source JSON.
