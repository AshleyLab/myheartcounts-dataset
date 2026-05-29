# jointProblem

**Benchmark column**: `field_jointProblem`
**Raw identifier**: `jointProblem` (ORKStep identifier)
**Role**: context
**Type**: binary (Yes/No)

## Source
- File: `CardioHealth/TasksAndSteps/APHDynamicParQQuizTask.m`
- Constant declaration: line 45
- Step construction: line 141
- Survey: PAR-Q (Physical Activity Readiness Questionnaire; kPhysicalActivityReadinessSurveyIdentifier)

## Question
> Do you have a bone or joint problem (for example, back, knee or hip) that could be made worse by a change in your physical activity?

## Answer options
- Yes (1)
- No (0)

(ORKBooleanAnswerFormat — standard iOS yes/no)

## Observed values

**Total observations**: 45,815 — **type-enforced**: 45,815 (**unique**: 2) — raw Python types seen: `bool` (45,815).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `False` | 36,502 | 79.7% |
| `True` | 9,313 | 20.3% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history (file-level)
- Recent material change: c312938 (2016) MHC-626 Upgrade to ResearchKit 2.0
- 6 commits total affecting this file

## Notes
- Standard PAR-Q item. A "yes" answer typically gates users out of certain active tasks (like the Fitness Test).
- Question text retrieved from NSLocalizedString key "JointProblemTitle"
