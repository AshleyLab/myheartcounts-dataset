"""Resolve checkpoint paths that may reference W&B artifacts.

Supports two path formats:
- Local paths: ``"/path/to/checkpoint.ckpt"`` — returned as-is (with existence check).
- W&B artifact references: ``"wandb:ENTITY/PROJECT/ARTIFACT:VERSION"`` — downloaded
  via the W&B public API and cached locally.

To select a specific file from a multi-file artifact, append ``#filename``::

    "wandb:ENTITY/PROJECT/ARTIFACT:VERSION#MyModel_epoch21_MAE0.1416.pypots"

Examples::

    from utils.wandb_artifact import resolve_checkpoint_path

    # Local path (unchanged)
    path = resolve_checkpoint_path("results/mae/best.ckpt")

    # W&B artifact (downloaded + cached)
    path = resolve_checkpoint_path("wandb:MHC_Dataset/mhc-mae-ssl/mae:latest")

    # W&B artifact with explicit file selection
    path = resolve_checkpoint_path(
        "wandb:MHC_Dataset/mhc-pypots-dlinear/dlinear:v45#DLinear_epoch21_MAE0.1416.pypots"
    )
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

WANDB_PREFIX = "wandb:"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "mhc-benchmark" / "artifacts"


def is_wandb_reference(path: str) -> bool:
    """Check whether *path* uses the ``wandb:`` prefix."""
    return path.startswith(WANDB_PREFIX)


def resolve_checkpoint_path(path: str, cache_dir: str | Path | None = None) -> Path:
    """Resolve a checkpoint path, downloading from W&B if needed.

    Args:
        path: Local file path **or** ``"wandb:ENTITY/PROJECT/ARTIFACT:VERSION"``.
        cache_dir: Directory for caching downloaded artifacts.
            Defaults to ``~/.cache/mhc-benchmark/artifacts/``.

    Returns:
        Resolved local ``Path`` to the checkpoint file.

    Raises:
        FileNotFoundError: If a local path does not exist.
        ValueError: If a ``wandb:`` reference is malformed.
    """
    if not is_wandb_reference(path):
        local = Path(path)
        if not local.exists():
            raise FileNotFoundError(f"Checkpoint not found: {local}")
        return local

    return _download_wandb_artifact(path, cache_dir)


def _download_wandb_artifact(path: str, cache_dir: str | Path | None = None) -> Path:
    """Download a W&B artifact and return the path to the checkpoint file."""
    import wandb

    artifact_ref = path[len(WANDB_PREFIX) :]  # strip prefix

    # Split off optional #filename selector
    filename_selector = None
    if "#" in artifact_ref:
        artifact_ref, filename_selector = artifact_ref.rsplit("#", 1)

    if artifact_ref.count("/") < 2:
        raise ValueError(
            f"Malformed wandb artifact reference: '{path}'. "
            "Expected format: wandb:ENTITY/PROJECT/ARTIFACT:VERSION[#filename]"
        )

    cache_root = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    # Use a per-artifact subdirectory to prevent file collisions between artifacts
    safe_name = artifact_ref.replace("/", "_").replace(":", "_")
    artifact_root = cache_root / safe_name
    artifact_root.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading W&B artifact: {artifact_ref}")
    api = wandb.Api()
    artifact = api.artifact(artifact_ref, type="model")
    artifact_dir = Path(artifact.download(root=str(artifact_root)))

    # If caller specified an exact filename, use it directly
    if filename_selector:
        resolved = artifact_dir / filename_selector
        if not resolved.exists():
            raise FileNotFoundError(
                f"Requested file '{filename_selector}' not found in artifact. "
                f"Available: {[f.name for f in artifact_dir.iterdir()]}"
            )
        logger.info(f"Resolved artifact to: {resolved}")
        return resolved

    # Find model files inside the artifact directory (.ckpt or .pypots),
    # excluding TensorBoard event files which PyPOTS also saves with .pypots extension.
    def _is_model_file(p: Path) -> bool:
        return p.suffix in (".ckpt", ".pypots") and not p.name.startswith("events.out.tfevents")

    model_files = [p for p in artifact_dir.glob("*.ckpt") if _is_model_file(p)]
    model_files += [p for p in artifact_dir.glob("*.pypots") if _is_model_file(p)]

    # Fallback: artifact.add_dir() may create nested structure
    if not model_files:
        model_files = [p for p in artifact_dir.rglob("*.ckpt") if _is_model_file(p)]
        model_files += [p for p in artifact_dir.rglob("*.pypots") if _is_model_file(p)]

    if len(model_files) == 1:
        resolved = model_files[0]
    elif len(model_files) > 1:
        # Prefer first alphabetically
        resolved = sorted(model_files)[0]
        logger.warning(
            f"Multiple model files in artifact, using {resolved.name}. "
            f"All files: {[f.name for f in model_files]}"
        )
    else:
        # No known model file — try single-file fallback
        all_files = list(artifact_dir.iterdir())
        if len(all_files) == 1:
            resolved = all_files[0]
        else:
            raise FileNotFoundError(
                f"No .ckpt or .pypots file found in artifact directory {artifact_dir}. "
                f"Contents: {[f.name for f in all_files]}"
            )

    logger.info(f"Resolved artifact to: {resolved}")
    return resolved
