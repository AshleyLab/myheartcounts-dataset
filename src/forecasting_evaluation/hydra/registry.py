"""Forecasting model registry shim for the Hydra CLI.

The heavy lifting is already done by
:func:`forecasting_evaluation.models.registry.create_forecasting_model` — this
module just wraps the result in the public
:class:`openmhc._evaluate._ForecasterAdapter` and returns the
``(model, manifest_or_none)`` tuple expected by ``eval_hydra.MethodRegistry``.

Forecasting checkpoints don't have a manifest analog yet (only imputation
does), so ``manifest`` is always ``None`` here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from eval_hydra.registry import MethodRegistry
from forecasting_evaluation.models.registry import create_forecasting_model

if TYPE_CHECKING:
    from forecasting_evaluation.config import (
        DataConfig,
        FeaturesConfig,
        ForecastingConfig,
        ForecastingModelConfig,
    )


def _build(
    model_cfg: "ForecastingModelConfig",
    forecasting_cfg: "ForecastingConfig",
    features_cfg: "FeaturesConfig",
    seed: int,
):
    model = create_forecasting_model(
        model_cfg,
        seed=seed,
        forecasting_config=forecasting_cfg,
        features_config=features_cfg,
    )
    return model, None


_KNOWN_TYPES = (
    "seasonal_naive",
    "seasonal_naive_average_history",
    "autoARIMA",
    "autoETS",
    "chronos2",
    "toto",
    "mixlinear",
    "dlinear",
    "segrnn",
)


MODEL_REGISTRY = MethodRegistry(
    name="forecasting model",
    builders={kind: _build for kind in _KNOWN_TYPES},
)
