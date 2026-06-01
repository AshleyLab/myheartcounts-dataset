# PAR-Q Readiness

The standard 7-item Physical Activity Readiness Questionnaire. All binary yes/no items; a "yes" on any item typically gates the user out of active fitness tasks (like the Fitness Test). Source: `CardioHealth/TasksAndSteps/APHDynamicParQQuizTask.m`.

## Variables (7 files)

| Variable | Role | Type | Summary |
|----------|------|------|---------|
| [heartCondition](heartCondition.md) | context | binary | Doctor-diagnosed heart condition restricting activity |
| [chestPain](chestPain.md) | context | binary | Chest pain during physical activity |
| [chestPainInLastMonth](chestPainInLastMonth.md) | context | binary | Chest pain in past month while *not* active |
| [dizziness](dizziness.md) | context | binary | Loss of balance / consciousness |
| [jointProblem](jointProblem.md) | context | binary | Bone or joint problem worsened by activity |
| [prescriptionDrugs](prescriptionDrugs.md) | context | binary | Prescription drugs for BP or heart condition |
| [physicallyCapable](physicallyCapable.md) | context | binary | Any other reason not to be active |

## Notes

- All seven items constructed as `ORKQuestionStep` with `ORKBooleanAnswerFormat` — see constant declarations in `APHDynamicParQQuizTask.m` lines 36-48.
- A positive response gates users out of the Fitness Test.
