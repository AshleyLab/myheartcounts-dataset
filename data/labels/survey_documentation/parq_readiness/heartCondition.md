# heartCondition

**Benchmark column**: `field_heartCondition`
**Raw identifier**: `heartCondition` (ORKStep identifier)
**Role**: context
**Type**: binary (Yes/No)

## Source
- File: `CardioHealth/TasksAndSteps/APHDynamicParQQuizTask.m`
- Constant declaration: line 41
- Step construction: line 109
- Survey: PAR-Q (Physical Activity Readiness Questionnaire; kPhysicalActivityReadinessSurveyIdentifier)

## Question
> Has your doctor ever said that you have a heart condition and that you should only do physical activity recommended by a doctor?

## Answer options
- Yes (1)
- No (0)

(ORKBooleanAnswerFormat — standard iOS yes/no)

## Observed values

**Total observations**: 45,811 — **type-enforced**: 45,811 (**unique**: 2) — raw Python types seen: `bool` (45,811).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `False` | 42,612 | 93.0% |
| `True` | 3,199 | 7.0% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Recent material change: c312938 (2016) MHC-626 Upgrade to ResearchKit 2.0
- 6 commits total affecting this file

## Notes
- Standard PAR-Q item. A "yes" answer typically gates users out of certain active tasks (like the Fitness Test).
- Question text retrieved from NSLocalizedString key "HeartConditionQuestion"
