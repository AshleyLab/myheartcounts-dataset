# Mindset Measures

Psychological mindset batteries capturing beliefs about chronic illness, body self-healing, and physical activity. Three distinct sub-batteries (20 + 7 + 5 items) that all use agreement-scale response formats. All items are context variables, ordinal.

## Variables (32 files)

## Sub-battery 1: Illness Mindset Inventory (20 items)

Beliefs about chronic illness and the body's healing capacity. 6-point agreement scale (Strongly agree → Strongly disagree). Source: `cardio_illness_mindset_measure_inventory_survey.json`.

| Variable | Summary |
|----------|---------|
| [body_self_healing_in_many_different_circumstances](body_self_healing_in_many_different_circumstances.md) | Body heals itself across many conditions |
| [body_remarkable_self_healing](body_remarkable_self_healing.md) | Body has remarkable self-healing properties |
| [body_self_healing_from_most_conditions_and_diseases](body_self_healing_from_most_conditions_and_diseases.md) | Body heals from most diseases |
| [chronic_illness_impact](chronic_illness_impact.md) | Chronic illness affects every aspect of life |
| [chronic_illness_body_meaning](chronic_illness_body_meaning.md) | Body function defines life meaning |
| [chronic_illness_body_coping](chronic_illness_body_coping.md) | Body can cope with illness challenges |
| [chronic_illness_positive_opportunity](chronic_illness_positive_opportunity.md) | Illness is an opportunity for positive change |
| [chronic_illness_management](chronic_illness_management.md) | Chronic illness is manageable |
| [chronic_illness_body_betrayal](chronic_illness_body_betrayal.md) | Body has betrayed me |
| [chronic_illness_more_meaning_in_life](chronic_illness_more_meaning_in_life.md) | Illness gives more meaning to life |
| [chronic_illness_handling](chronic_illness_handling.md) | I can handle chronic illness |
| [chronic_illness_spoil](chronic_illness_spoil.md) | Chronic illness spoils life |
| [chronic_illness_challenge](chronic_illness_challenge.md) | Illness challenge makes me stronger |
| [chronic_illness_body_handling](chronic_illness_body_handling.md) | Body is handling illness well |
| [chronic_illness_runing_life](chronic_illness_runing_life.md) | Illness is ruining life (note preserved "runing" typo) |
| [chronic_illness_body_management](chronic_illness_body_management.md) | Body is designed to manage illness |
| [chronic_illness_relatively_normal_life](chronic_illness_relatively_normal_life.md) | Can live relatively normal life with illness |
| [chronic_illness_body_failure](chronic_illness_body_failure.md) | My body has failed me |
| [chronic_illness_empowering](chronic_illness_empowering.md) | Fighting illness is empowering |
| [chronic_illness_body_blame](chronic_illness_body_blame.md) | I blame my body for illness |

## Sub-battery 2: Exercise Process Mindset (7 items)

Perceptions of physical activity. Source: `cardio_exercise_process_mindset_measure_survey.json`.

| Variable | Summary |
|----------|---------|
| [easy](easy.md) | Exercise feels easy vs. difficult |
| [pleasurable](pleasurable.md) | Exercise is pleasurable vs. unpleasant |
| [relaxing](relaxing.md) | Exercise is relaxing vs. stressful |
| [convenient](convenient.md) | Exercise is convenient vs. inconvenient |
| [fun](fun.md) | Exercise is fun vs. boring |
| [social](social.md) | Exercise is social vs. solitary |
| [indulgent](indulgent.md) | Exercise is indulgent vs. depriving |

## Sub-battery 3: Adequacy of Activity Mindset (5 items)

Beliefs about whether one's current activity level is adequate. Source: `cardio_adequacy_of_activity_mindset_measure_survey.json`.

| Variable | Summary |
|----------|---------|
| [unhealthy](unhealthy.md) | Current activity is unhealthy |
| [weight](weight.md) | Activity helps achieve healthy weight |
| [beneficial](beneficial.md) | Activity is beneficial to health |
| [disease](disease.md) | Activity reduces disease risk |
| [muscles](muscles.md) | Activity strengthens muscles |

## Notes

- The benchmark dataset sometimes labels the combined sub-batteries 2 + 3 as an "eating-reasons battery" — this is a misnomer. The surveys are about physical activity, not eating.
- All three sub-batteries were added together in commit MHC-610 (Feb 2025).
- Illness mindset battery has 20 items in the JSON; the benchmark spec mentions 21 — the additional item appears to be absent from the iOS source.
