"""Package a trained PyPOTS forecasting model into an openmhc release bundle.

A forecasting release bundle is the directory layout consumed by both the
public ``openmhc.forecasters`` API (``DLinearForecaster.from_release(...)``) and
the eval harness (``mhc-forecast-eval model.release_dir=...``):

    release_dir/
    ├── model.pypots                # trained checkpoint
    ├── standard_scaler_stats.json  # train-fit channel StandardScaler (when used)
    ├── training_config.json        # full training config — the arch contract the
    │                               #   eval adapter reads (model/training/forecasting)
    └── openmhc_manifest.json       # spec_version=1 (written via openmhc.forecasters._release)

The manifest writer is reused from :mod:`openmhc.forecasters._release` (no
forecasting-specific writer is duplicated), mirroring how
:mod:`imputation_training.release` reuses :mod:`openmhc.imputers._release`.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from openmhc.forecasters._release import write_manifest

logger = logging.getLogger(__name__)


def write_release(
    *,
    model_name: str,
    arch: dict[str, Any],
    training_config_json: dict[str, Any],
    release_dir: str | Path,
    pypots_checkpoint: str | Path,
    scaler_stats_path: str | Path | None = None,
    provenance: dict[str, Any] | None = None,
) -> Path:
    """Build a complete forecasting release bundle.

    Args:
        model_name: Model kind — one of ``{"dlinear", "mixlinear", "segrnn"}``.
        arch: Architecture kwargs recorded in the manifest (``n_steps``,
            ``n_pred_steps``, ``n_features``). The eval adapter reads most
            architecture from ``training_config.json``; this is a secondary
            fallback for manifest-only consumers.
        training_config_json: ``asdict(ForecastingTrainingConfig)`` — the real
            architecture contract the eval adapter reads (sections ``model``,
            ``training``, ``forecasting``). Written as ``training_config.json``.
        release_dir: Where to write the bundle. Created if missing.
        pypots_checkpoint: Path to the ``.pypots`` file PyPOTS produced. Copied
            to ``release_dir / model.pypots``.
        scaler_stats_path: Path to the train-fit ``standard_scaler_stats.json``.
            Required iff ``training.whether_standardscaler`` is True; copied into
            the bundle and referenced as the manifest's ``normalization_stats``.
        provenance: Optional metadata dict — stored as-is in the manifest.

    Returns:
        Path to the release directory.
    """
    release_dir = Path(release_dir)
    release_dir.mkdir(parents=True, exist_ok=True)

    whether_standardscaler = bool(
        training_config_json.get("training", {}).get("whether_standardscaler", False)
    )
    # Hard invariant: standardized training must ship its scaler stats, and a
    # bundle that ships stats must declare it standardized — otherwise the eval
    # adapter predicts in the wrong value space.
    if whether_standardscaler and scaler_stats_path is None:
        raise ValueError(
            "training.whether_standardscaler=True but no scaler_stats_path was provided; "
            "the release bundle would be unusable (predictions stay standardized)."
        )
    if not whether_standardscaler and scaler_stats_path is not None:
        raise ValueError(
            "scaler_stats_path was provided but training.whether_standardscaler=False; "
            "refusing to ship inconsistent normalization metadata."
        )

    # Copy checkpoint into the bundle under a stable name.
    src_ckpt = Path(pypots_checkpoint)
    if not src_ckpt.exists():
        raise FileNotFoundError(f"PyPOTS checkpoint not found: {src_ckpt}")
    dst_ckpt = release_dir / "model.pypots"
    if src_ckpt.resolve() != dst_ckpt.resolve():
        shutil.copy2(src_ckpt, dst_ckpt)
        logger.info("Copied checkpoint %s -> %s", src_ckpt, dst_ckpt)

    # Copy scaler stats when standardized.
    stats_filename: str | None = None
    if scaler_stats_path is not None:
        src_stats = Path(scaler_stats_path)
        if not src_stats.exists():
            raise FileNotFoundError(f"Scaler stats not found: {src_stats}")
        dst_stats = release_dir / "standard_scaler_stats.json"
        if src_stats.resolve() != dst_stats.resolve():
            shutil.copy2(src_stats, dst_stats)
        stats_filename = "standard_scaler_stats.json"

    # Write the full training config — the arch contract the eval adapter reads.
    (release_dir / "training_config.json").write_text(
        json.dumps(training_config_json, indent=2), encoding="utf-8"
    )

    write_manifest(
        release_dir,
        kind=model_name.lower(),
        checkpoint="model.pypots",
        arch=dict(arch),
        normalization_stats=stats_filename,
        provenance=provenance or {},
    )
    logger.info("Forecasting release bundle written to %s", release_dir)
    return release_dir
