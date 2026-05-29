# Wellbeing

ONS (UK Office for National Statistics) personal wellbeing framework — life satisfaction plus four daily-emotion items (worthwhile, happy, worried, depressed) — plus a separate daily happiness check-in and its derived ordinal category. All measured on 0–10 scales.

## Variables (7 files)

| Variable | Role | Type | Source | Summary |
|----------|------|------|--------|---------|
| [satisfiedwith_life](satisfiedwith_life.md) | target | ordinal | cardio_wellbeing_survey.json | Overall life satisfaction (0–10) |
| [feel_worthwhile1](feel_worthwhile1.md) | target | ordinal | cardio_wellbeing_survey.json | ONS "things you do are worthwhile" (0–10) |
| [feel_worthwhile2](feel_worthwhile2.md) | target | ordinal | cardio_wellbeing_survey.json | ONS happy yesterday (0–10) |
| [feel_worthwhile3](feel_worthwhile3.md) | target | ordinal | cardio_wellbeing_survey.json | ONS worried yesterday (0–10) |
| [feel_worthwhile4](feel_worthwhile4.md) | target | ordinal | cardio_wellbeing_survey.json | ONS depressed yesterday (0–10) |
| [happiness](happiness.md) | target | continuous | cardio_daily_check.json | Daily happiness check-in (0–10) |
| [happiness_categories](happiness_categories.md) | target | ordinal | Derived | Bins of daily happiness score |

## Notes

- `feel_worthwhile2` and `happiness` are *different* variables: the former is a retrospective yesterday-focused ONS item on the wellbeing survey; the latter is a short same-day 0–10 slider on the daily check-in.
- `feel_worthwhile1–4` are named after question #1 of the ONS battery but each targets a different emotion — preserve the prompts in the per-variable files to disambiguate.
