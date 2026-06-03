"""Surface-2 downstream methods — reproduce the paper table from baked features.

Each Method loads pre-baked, ready-to-train feature tables (IC + temporal scope
+ labels already applied private-side) and produces per-task predictions. The
shared ``runner.run_eval`` turns those into metrics.

This is distinct from Surface 1 (``openmhc``), which extracts features live from
a submitted encoder via ``feature_store``.
"""

from downstream_evaluation.methods.base import Method, TaskPrediction
from downstream_evaluation.methods.linear_probe import LinearProbeMethod
from downstream_evaluation.methods.xgboost import XGBoostMethod

__all__ = ["Method", "TaskPrediction", "LinearProbeMethod", "XGBoostMethod"]
