"""XGBoost downstream method: hand-crafted features + gradient-boosted trees.

Self-contained package. ``model.py`` holds the ``XGBoost`` predictor, which trains
shallow regularized trees on ~495 hand-crafted per-participant features built by the
feature pipelines in this package (``pipeline_timeseries`` → ``pipeline_day_dynamics``
→ ``pipeline_curve_analysis``, over raw ``daily_hf``).
"""

from __future__ import annotations

from downstream_evaluation.models.xgboost.model import XGBoost

__all__ = ["XGBoost"]
