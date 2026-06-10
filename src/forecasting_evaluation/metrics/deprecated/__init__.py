"""Deprecated forecasting metric-aggregation scripts.

These produce paper/QC artifacts and are **not** part of the canonical
skill-score / ranking pipeline. They are kept (importable, runnable) for
reproducing prior tables, but are not maintained as part of the Layer-2 flow.

The live Layer-2 path is:

    mhc-forecast-eval  →  <model>_metrics/<LABEL>/{mae,...,auprc,...}
                       →  metrics/skill_score_summary.py   (skill score)
                       →  metrics/grouped_metric_rank_summary.py  (mean rank)

See ``deprecated/README.md`` for what each script did and its replacement.
"""
