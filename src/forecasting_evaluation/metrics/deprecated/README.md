# Deprecated metric-aggregation scripts

These scripts are **not** part of the canonical forecasting skill-score /
ranking pipeline. They predate the consolidation and are kept only for
reproducing older paper/QC tables. Prefer the live Layer-2 entry points.

## Live pipeline (use these instead)

```
mhc-forecast-eval                     # Layer 1: predictions + point metrics
                                      #   + binary metrics (auprc/auroc/f1),
                                      #   co-located in <model>_metrics/<LABEL>/
   │
   ├─ metrics/skill_score_summary.py          # skill score vs baseline
   └─ metrics/grouped_metric_rank_summary.py  # grouped mean-rank
```

Both are driven in one shot by `jobs/simurgh/forecasting_eval/skill_rank.sbatch`.
Channel groups and metric directionality come from `metrics/metric_spec.py`
(single source of truth). The skill score and ranking both consume
`mae` (continuous, channels 0-6) + `auprc` (binary, sleep 7-8 / workout 9-18).

## What lives here and why it moved

| script | did | not needed for skill/rank because |
|---|---|---|
| `summary_metrics_result.py` | MAE-by-model-channel-hour CSV + run-stats CSV | a raw per-hour MAE table; skill/rank read the metric parquets directly |
| `binary_summary_metrics_result.py` | per-channel binary metric value ± CI | a raw value view, not a skill/rank input |
| `binary_group_summary_metrics_result.py` | sleep/workout grouped binary value summary | superseded by the binary handling inside skill/rank |
| `mase_validity_summary.py` | % of valid (non-NaN) MASE per model/channel | a data-coverage QC report, not a score |
| `paper_result_generator_one_channel.py` | paper table, one channel (3-hour buckets + ranks) | publication formatting |
| `paper_result_generator_all_channels.py` | paper table, all channels/groups | publication formatting |

`jobs/simurgh/forecasting_eval/deprecated/aggregate_results.sbatch` (which ran
`summary_metrics_result.py`) is likewise retired in favor of `skill_rank.sbatch`.
