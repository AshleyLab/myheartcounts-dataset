"""Deprecated forecasting metric-aggregation.

``summary_metrics_result.py`` produces the legacy MAE-by-channel-hour table
(``mae_by_model_channel_hour.csv`` / ``statistical_result.csv``). It is **not**
part of the canonical skill-score / ranking pipeline and is kept only to
reproduce prior tables; it is still invoked by the deprecated
``jobs/*/forecasting_eval/aggregate_results.sbatch``.

The live Layer-2 path is:

    mhc-forecast-eval  →  <model>_metrics/<LABEL>/{mae,...,auprc,...}
                       →  metrics/skill_score_summary.py   (skill score)
                       →  metrics/grouped_metric_rank_summary.py  (mean rank)
"""
