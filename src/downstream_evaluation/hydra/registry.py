"""Downstream model registry used by the ``mhc-downstream-eval`` Hydra CLI.

Maps ``MethodConfig.type`` to a builder that constructs the bundled model —
the structured replacement for the ``build_model`` if/elif in the old
``scripts/run_eval.py``. Each builder returns ``(model, manifest_or_none)``;
downstream models have no checkpoint-release manifest, so it is always ``None``.

Imports are deferred into each builder (mirroring the old ``build_model``) so a
CLI run only pulls in the one model's heavy deps (torch / xgboost / …).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from eval_hydra.registry import MethodRegistry

if TYPE_CHECKING:
    from downstream_evaluation.config import DataConfig, MethodConfig


def _linear(method: MethodConfig, data: DataConfig):
    from downstream_evaluation.models.linear import Linear

    return Linear(data_dir=data.data_dir), None


def _xgboost(method: MethodConfig, data: DataConfig):
    from downstream_evaluation.models.xgboost import XGBoost

    return XGBoost(
        data_dir=data.data_dir,
        features_dir=method.features_dir,
        max_future_days=method.max_future_days,
    ), None


def _lsm2(method: MethodConfig, data: DataConfig):
    from downstream_evaluation.models.lsm2 import LSM2

    if method.checkpoint:
        return LSM2(data_dir=data.data_dir, checkpoint=method.checkpoint), None
    return LSM2(data_dir=data.data_dir), None


def _wbm(method: MethodConfig, data: DataConfig):
    from downstream_evaluation.models.hybrid_wbm import Hybrid

    if method.checkpoint:
        return Hybrid(data.data_dir, checkpoint=method.checkpoint), None
    return Hybrid(data.data_dir), None


def _toto(method: MethodConfig, data: DataConfig):
    from downstream_evaluation.models.toto import Toto

    return Toto(data_dir=data.data_dir), None


def _chronos2(method: MethodConfig, data: DataConfig):
    from downstream_evaluation.models.chronos2 import Chronos2

    return Chronos2(data_dir=data.data_dir), None


def _multirocket(method: MethodConfig, data: DataConfig):
    from downstream_evaluation.models.multirocket import MultiRocket
    from openmhc._constants import BENCHMARK_TASKS

    return MultiRocket(data_dir=data.data_dir, tasks=BENCHMARK_TASKS), None


def _gru_d(method: MethodConfig, data: DataConfig):
    from downstream_evaluation.models.grud import GRUD
    from openmhc._constants import BENCHMARK_TASKS

    return GRUD(data_dir=data.data_dir, tasks=BENCHMARK_TASKS), None


METHOD_REGISTRY = MethodRegistry(
    name="downstream method",
    builders={
        "linear": _linear,
        "xgboost": _xgboost,
        "lsm2": _lsm2,
        "wbm": _wbm,
        "toto": _toto,
        "chronos2": _chronos2,
        "multirocket": _multirocket,
        "gru_d": _gru_d,
    },
)
