# sleep_diagnosis1

**Benchmark column**: `sleep_diagnosis1`
**Raw identifier**: `sleep_diagnosis1`
**Role**: target
**Type**: binary

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_activitysleep_survey.json`
- Line: ~180
- Survey: `ActivitySleep` (Activity and Sleep Survey)

## Question
> Have you ever been told by a doctor or other health professional that you have a sleep disorder?

## Answer options
Boolean (checkbox).

| Value | Label |
|-------|-------|
| 0 | No |
| 1 | Yes |

| Constraint | Value |
|-----------|-------|
| Data Type | boolean |
| UI Hint | checkbox |

## Observed values

**Total observations**: 44,498 — **type-enforced**: 44,498 (**unique**: 2) — raw Python types seen: `str` (44,498).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `False` | 39,037 | 87.7% |
| `True` | 5,461 | 12.3% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Conditional Logic
- **Rule**: When `sleep_diagnosis1 == 0` (false), skip to `END_OF_SURVEY`.
- **Effect**: If user answers "No" (no sleep disorder diagnosis), survey terminates immediately. If "Yes", user proceeds to `sleep_diagnosis2` to specify which sleep disorder(s).

## Git history (file-level)
- Total commits: 6
- Most recent: `7f52783` (2024, MHC-626 - Fix parsing survey element without createdOn property)
- Earlier notable: `c312938` (MHC-626 Upgrade to ResearchKit 2.0), `581fc6e` (fix for MHC-30)
- Notes: No targeted changes to this specific variable; part of broader ResearchKit 2.0 upgrade and parser fixes.

## Notes
- Gating question: controls whether detailed sleep disorder classification (`sleep_diagnosis2`) is presented.
- When `sleep_diagnosis1 == 1`, respondent proceeds to multi-select `sleep_diagnosis2` to identify specific disorder type(s).
- When `sleep_diagnosis1 == 0`, respondent ends the survey early (skips `sleep_diagnosis2`).
- Important for clinical stratification: sleep disorders have significant cardiovascular implications.
- Paired with `sleep_time` and `sleep_time1` (duration measurements) to assess sleep health comprehensively.
