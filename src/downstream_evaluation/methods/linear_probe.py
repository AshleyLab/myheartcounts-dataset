"""LinearProbeMethod — load a baked feature table and fit a linear probe per task.

Covers the linear-probe Contract-A methods: the encoders (mae/ssl/toto/chronos2,
PCA-50 baked private-side, no scaler) and stat_simple (38-d raw, scaler on). The
probe per task type follows the paper headline: LogisticRegression (binary),
K-1 LogReg-Ordinal (ordinal), OLS (regression).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from downstream_evaluation.config import ClassifierConfig
from downstream_evaluation.methods._probe import fit_predict_tables
from downstream_evaluation.methods.base import TaskPrediction

# Paper headline linear probes by task type.
_DEFAULT_CLASSIFIERS: dict[str, str] = {
    "binary": "logistic_regression",
    "multiclass": "logistic_regression",
    "ordinal": "logreg_ordinal",
    "continuous": "linear_regression",
    "regression": "linear_regression",
}


@dataclass
class LinearProbeMethod:
    """Load ``features/<name>/{train,test}.parquet`` → linear probe per task.

    Args:
        name: method/folder name under ``features_dir``.
        features_dir: root holding ``<name>/<split>.parquet``.
        use_scaler: scale before the probe (stat_simple: True; encoders: False —
            they ship PCA-50 with no scaler).
        classifiers: task-type → classifier registry key.
        seed: random_state for the probe.
    """

    name: str
    features_dir: str
    use_scaler: bool = False
    classifiers: dict[str, str] = field(default_factory=lambda: dict(_DEFAULT_CLASSIFIERS))
    seed: int = 42

    def predict(self, tasks: list[str]) -> dict[str, TaskPrediction]:
        base = ClassifierConfig(use_scaler=self.use_scaler, pca_n_components=None)
        return fit_predict_tables(
            self.name, self.features_dir, tasks, self.classifiers, base, self.seed
        )
