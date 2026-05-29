# work

**Benchmark column**: `work`
**Raw identifier**: `work`
**Role**: target
**Type**: binary

## Source
- File: `CardioHealth/Resources/JSONs/cardiosurveys/cardio_activitysleep_survey.json`
- Line: ~9
- Survey: `ActivitySleep` (Activity and Sleep Survey)

## Question
> Do you do regular work?

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

**Total observations**: 44,430 — **type-enforced**: 44,430 (**unique**: 2) — raw Python types seen: `str` (44,430).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `True` | 36,838 | 82.9% |
| `False` | 7,592 | 17.1% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Conditional Logic
- **Rule**: When `work == 1` (true), skip to `phys_activity`.
- **Effect**: If user answers "No" to regular work, they proceed directly to leisure-time activity questions, bypassing `atwork` (work-time activity detail question).

## Git history (file-level)
- Total commits: 6
- Most recent: `7f52783` (2024, MHC-626 - Fix parsing survey element without createdOn property)
- Earlier notable: `c312938` (MHC-626 Upgrade to ResearchKit 2.0), `581fc6e` (fix for MHC-30)
- Notes: No targeted changes to this specific variable; part of broader ResearchKit 2.0 upgrade and parser fixes.

## Notes
- Gating question: controls whether `atwork` (work-time activity intensity) is presented to the respondent.
- When `work == 1`, user sees `atwork` question with detailed prompt about work-time physical activity.
- When `work == 0`, user skips `atwork` entirely.
- Used to stratify activity data collection: work-time activity is only captured for employed respondents.
- Informs context for cardiovascular health modeling (e.g., occupational activity level).
