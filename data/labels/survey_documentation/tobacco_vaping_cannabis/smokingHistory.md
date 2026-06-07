# smokingHistory

**Benchmark column**: `field_smokingHistory`
**Raw identifier**: `smokingHistory` (ORKQuestionStep identifier; Obj-C constant `kHeartAgeFormStepSmokingHistory`)
**Role**: context
**Type**: binary

## Source
- Constant declaration: `CardioHealth/TasksAndSteps/HeartAgeControllers/APHHeartAgeTaskViewController.m:54`
- Also: `CardioHealth/Data/APHHeartAgeTask.m:38` (same constant)
- Step construction: `APHHeartAgeTaskViewController.m:218`
- Result read in Framingham calculation: `APHHeartAgeTask.m:49` (`[self resultForIdentifier:kHeartAgeFormStepSmokingHistory]`) and stored under the Framingham input key `kHeartAgeTestDataSmoke` (`APHHeartAgeAndRiskFactors.m:69`)
- Survey: Heart Age / Framingham Risk form (`kHeartStrokeRiskSurveyIdentifier`)

## Question
> Are you currently smoking cigarettes?

(Constructed via `NSLocalizedString(@"Are you currently smoking cigarettes?",nil)` at line 218 with `[ORKBooleanAnswerFormat new]`.)

## Answer options
- Yes (1)
- No (0)

`ORKBooleanAnswerFormat` — standard iOS Yes/No.

## Observed values

**Total observations**: 10,167 — **type-enforced**: 10,167 (**unique**: 2) — raw Python types seen: `bool` (10,167).
**Type-enforcement rejections**: 0 missing (`LabelValueError`), 0 unconvertible (`LabelTypeError`), 0 dictionary-miss (`KeyError`).

| value | count | pct |
|-------|------:|----:|
| `False` | 9,665 | 95.1% |
| `True` | 502 | 4.9% |

_Generated 2026-04-28 from `data/labels/last_labels.json` (md5 `0f65e8fe…`) and `data/labels/context_labels.json` (md5 `560ed125…`)._

## Git history
- `APHHeartAgeTaskViewController.m`: recent material changes — `dbdd5a0` MHC-508 (UK units), `eaf8632` MHC-709 (UI refresh), `c312938` MHC-626 (ResearchKit 2.0 upgrade).
- The `smokingHistory` step identifier appears to have been stable across these revisions (renamings would have surfaced in git log on the constant).

## Notes
- Feeds the Framingham 10-year risk calculation as the "smoker" binary input (`kHeartAgeTestDataSmoke`) — see `framingham_risk.md` in `cardiometabolic_labs/`.
- This is a *separate* question from the detailed `cardio_vaping_and_smoking_survey.json` items (`currentSmoking`, `onsetSmoking`, etc.). The Heart Age form asks a simple "currently smoking?" Y/N as a fast Framingham input; the vaping/smoking survey collects fine-grained detail.
- Was missed in the initial documentation pass because earlier agents were scoped to the user-provided context-variable list which did not mention `smokingHistory`.
