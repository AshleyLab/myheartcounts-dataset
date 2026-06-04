"""Imputation method registry used by the Hydra CLI.

Builds the appropriate :class:`openmhc.imputers.BaseImputer` subclass for a
given ``MethodConfig.type``, wraps it in
:class:`openmhc._evaluate._ImputerMethodAdapter` (the same adapter the public
``evaluate_imputation`` API uses), and surfaces the loaded
:class:`openmhc.imputers.Manifest` when a release directory was used. The CLI
copies that manifest into the run dir so each result is traceable back to its
exact checkpoint + arch params.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from eval_hydra.registry import MethodRegistry
from openmhc._evaluate import _ImputerMethodAdapter
from openmhc.imputers import (
    BRITSImputer,
    DLinearImputer,
    FEDformerImputer,
    LinearImputer,
    LOCFImputer,
    LSM2Imputer,
    LSM2WeeklySparseImputer,
    MeanImputer,
    ModeImputer,
    PersonalizedMeanImputer,
    PersonalizedModeImputer,
    PersonalizedTemporalMeanImputer,
    TemporalMeanImputer,
    TemporalModeImputer,
    TimesNetImputer,
)
from openmhc.imputers._release import load_manifest

if TYPE_CHECKING:
    from imputation_evaluation.config import DataConfig, MethodConfig
    from openmhc.imputers._release import Manifest


# Reference imputers that fit on the train split and take only ``data_dir``.
_REFERENCE_CLASSES: dict[str, type] = {
    "mean": MeanImputer,
    "mode": ModeImputer,
    "linear": LinearImputer,
    "locf": LOCFImputer,
    "temporal_mean": TemporalMeanImputer,
    "temporal_mode": TemporalModeImputer,
    "personalized_mean": PersonalizedMeanImputer,
    "personalized_mode": PersonalizedModeImputer,
    "personalized_temporal_mean": PersonalizedTemporalMeanImputer,
}


# Paper-checkpoint wrappers that subclass ``ReleaseLoadableMixin``.
_PAPER_CHECKPOINT_CLASSES: dict[str, type] = {
    "brits": BRITSImputer,
    "timesnet": TimesNetImputer,
    "dlinear": DLinearImputer,
    # ``dlinear_weekly`` is a 7-day-window variant of DLinear. It reuses the
    # same ``DLinearImputer`` class — the difference lives in the release
    # manifest's ``arch.n_steps`` (10080 vs 1440) and the harness running with
    # ``data.n_days=7``. Manifest ``kind`` stays ``"dlinear"`` so
    # ``from_release`` validation passes against ``DLinearImputer.model_name``.
    "dlinear_weekly": DLinearImputer,
    "fedformer": FEDformerImputer,
    "lsm2": LSM2Imputer,
    "lsm2_weekly_sparse": LSM2WeeklySparseImputer,
}


def _build_reference(
    cls: type, method_cfg: "MethodConfig", data_cfg: "DataConfig"
) -> tuple[Any, None]:
    # Reference imputers take the dataset *root* (looking for
    # ``splits/`` and ``processed/daily_hf`` underneath). The runner's
    # ``data.daily_hf_dir`` is a different concept (the HF disk path
    # consumed by the evaluator), so we let the imputer resolve its root
    # from explicit ``data_dir=`` configuration or ``MHC_DATA_DIR``.
    extra: dict[str, Any] = {}
    # Mode-family imputers (mode, temporal_mode, personalized_mode) accept a
    # ``decimal_precision`` knob that controls value-rounding before tallying.
    # Thread it through so a user override on the CLI actually takes effect.
    if cls in {ModeImputer, TemporalModeImputer, PersonalizedModeImputer}:
        extra["decimal_precision"] = method_cfg.decimal_precision
    imputer = cls(version=data_cfg.version, **extra)
    return _ImputerMethodAdapter(imputer), None


def _build_paper_checkpoint(
    cls: type, method_cfg: "MethodConfig", data_cfg: "DataConfig"
) -> tuple[Any, "Manifest | None"]:
    """Construct a paper-checkpoint imputer from a release dir or inline arch.

    Preferred: ``method.release_dir`` points at a manifest-bundled release. We
    load the manifest, validate ``kind == cls.model_name``, and let
    ``from_release`` reconstruct arch + checkpoint paths automatically.

    Fallback: build the wrapper directly from inline ``method.pypots`` /
    ``method.lsm2`` blocks. Users must keep arch params in sync with the
    training run themselves — the manifest path is the recommended route.
    """
    runtime_kwargs: dict[str, Any] = {
        "version": data_cfg.version,
        "device": method_cfg.device,
        "inference_batch_size": method_cfg.inference_batch_size,
    }
    # LSM2 family accepts an ``inference_dropout_removal_ratio`` runtime knob
    # that overrides the value baked into the checkpoint. Forward it whenever
    # the user supplied a non-None value so CLI overrides (e.g.
    # ``method.lsm2.inference_dropout_removal_ratio=0.0``) actually take
    # effect through both the release-dir and inline-arch code paths.
    if cls in {LSM2Imputer, LSM2WeeklySparseImputer}:
        ratio = method_cfg.lsm2.inference_dropout_removal_ratio
        if ratio is not None:
            runtime_kwargs["inference_dropout_removal_ratio"] = ratio
    if method_cfg.release_dir:
        manifest = load_manifest(method_cfg.release_dir)
        imputer = cls.from_release(method_cfg.release_dir, **runtime_kwargs)
        return _ImputerMethodAdapter(imputer), manifest

    if cls in {LSM2Imputer, LSM2WeeklySparseImputer}:
        lsm2_cfg = method_cfg.lsm2
        if not lsm2_cfg.model_path:
            raise ValueError(
                f"{cls.__name__} requires either ``method.release_dir`` "
                f"or ``method.lsm2.model_path``."
            )
        imputer = cls(
            model_path=lsm2_cfg.model_path,
            normalization_stats_path=lsm2_cfg.normalization_stats_path,
            **runtime_kwargs,
        )
    else:
        pypots_cfg = method_cfg.pypots
        if not pypots_cfg.model_path:
            raise ValueError(
                f"{cls.__name__} requires either ``method.release_dir`` "
                f"or ``method.pypots.model_path``."
            )
        # Inline-arch fallback: pass everything from PyPOTSMethodConfig and let
        # the wrapper sort out which fields it actually consumes.
        from dataclasses import asdict

        inline = asdict(pypots_cfg)
        inline.pop("model_path", None)
        inline.pop("model_name", None)
        # PyPOTSMethodConfig.variant is FEDformer-specific and only that wrapper
        # consumes it; strip it for the other wrappers so they don't reject it.
        if cls is not FEDformerImputer:
            inline.pop("variant", None)
        imputer = cls(model_path=pypots_cfg.model_path, **inline, **runtime_kwargs)

    return _ImputerMethodAdapter(imputer), None


def _make_builder(kind: str):
    if kind in _REFERENCE_CLASSES:
        cls = _REFERENCE_CLASSES[kind]
        return lambda m, d, _cls=cls: _build_reference(_cls, m, d)
    if kind in _PAPER_CHECKPOINT_CLASSES:
        cls = _PAPER_CHECKPOINT_CLASSES[kind]
        return lambda m, d, _cls=cls: _build_paper_checkpoint(_cls, m, d)
    raise KeyError(kind)


METHOD_REGISTRY = MethodRegistry(
    name="imputation method",
    builders={
        kind: _make_builder(kind)
        for kind in (*_REFERENCE_CLASSES, *_PAPER_CHECKPOINT_CLASSES)
    },
)
