"""XGBoostMethod — fe_xgboost: tree probe on raw hand-crafted features.

fe_xgboost ships raw features *with NaN* (XGBoost handles NaN natively) — no PCA,
no scaler. The probe per task type follows the paper headline: XGBoost classifier
(binary/multiclass), XGB-Ordinal (ordinal), XGBoost regressor (regression).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from downstream_evaluation.config import (
    ClassifierConfig,
    XGBClassifierConfig,
    XGBOrdinalConfig,
    XGBRegressorConfig,
)
from downstream_evaluation.methods._probe import fit_predict_tables
from downstream_evaluation.methods.base import TaskPrediction

_XGB_CLASSIFIERS: dict[str, str] = {
    "binary": "xgboost_classifier",
    "multiclass": "xgboost_classifier",
    "ordinal": "xgboost_ordinal",
    "continuous": "xgboost_regressor",
    "regression": "xgboost_regressor",
}

# fe_xgboost hyperparameters from configs/downstream_eval/fe_xgboost.yaml — the
# eval overrides the registry defaults heavily (1000 shallow regularized trees),
# so the public method must carry them to reproduce the table.
_XGB_PARAMS = dict(
    n_estimators=1000,
    max_depth=2,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    n_jobs=-1,
)


@dataclass
class XGBoostMethod:
    """Load ``features/<name>/{train,test}.parquet`` → XGBoost probe per task.

    Args:
        name: method/folder name under ``features_dir`` (e.g. ``fe_xgboost``).
        features_dir: root holding ``<name>/<split>.parquet``.
        classifiers: task-type → classifier registry key.
        seed: random_state for the probe.
    """

    name: str
    features_dir: str
    classifiers: dict[str, str] = field(default_factory=lambda: dict(_XGB_CLASSIFIERS))
    seed: int = 42
    use_scaler: bool = False  # tree model: no scaler, keeps NaN

    def predict(self, tasks: list[str]) -> dict[str, TaskPrediction]:
        base = ClassifierConfig(
            use_scaler=self.use_scaler,
            pca_n_components=None,
            xgboost_classifier=XGBClassifierConfig(**_XGB_PARAMS),
            xgboost_regressor=XGBRegressorConfig(**_XGB_PARAMS),
            xgboost_ordinal=XGBOrdinalConfig(**_XGB_PARAMS),
        )
        return fit_predict_tables(
            self.name, self.features_dir, tasks, self.classifiers, base, self.seed
        )
