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

from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING

from eval_hydra.registry import MethodRegistry
from forecasting_evaluation.hydra.release import load_forecasting_manifest
from forecasting_evaluation.models.registry import create_forecasting_model

if TYPE_CHECKING:
    from forecasting_evaluation.config import (
        FeaturesConfig,
        ForecastingConfig,
        ForecastingModelConfig,
    )
    from forecasting_evaluation.hydra.release import ForecastingManifest


def _build(
    model_cfg: ForecastingModelConfig,
    forecasting_cfg: ForecastingConfig,
    features_cfg: FeaturesConfig,
    seed: int,
):
    manifest = _apply_release_manifest(model_cfg)
    model = create_forecasting_model(
        model_cfg,
        seed=seed,
        forecasting_config=forecasting_cfg,
        features_config=features_cfg,
    )
    return model, manifest


def _apply_release_manifest(
    model_cfg: ForecastingModelConfig,
) -> ForecastingManifest | None:
    """Apply a release manifest to the selected nested model config."""
    if not model_cfg.release_dir:
        return None

    manifest = load_forecasting_manifest(model_cfg.release_dir)
    if manifest.kind != model_cfg.type:
        raise ValueError(
            f"Forecasting release kind {manifest.kind!r} does not match "
            f"selected model.type {model_cfg.type!r}."
        )

    nested_cfg = getattr(model_cfg, model_cfg.type)
    if not is_dataclass(nested_cfg):
        raise TypeError(f"Expected dataclass config for model type {model_cfg.type!r}")

    field_names = {field.name for field in fields(nested_cfg)}
    if "checkpoint_path" in field_names:
        setattr(nested_cfg, "checkpoint_path", str(manifest.checkpoint_path))

    for key, value in manifest.arch.items():
        if key in field_names and key != "checkpoint_path":
            setattr(nested_cfg, key, value)

    return manifest


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
