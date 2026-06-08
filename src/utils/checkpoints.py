"""Resolve checkpoint paths that may reference remote artifacts.

Supports three path formats:
- Local paths: ``"/path/to/checkpoint.ckpt"`` — returned as-is (with existence check).
- Hugging Face Hub references: ``"hf://ORG/REPO"`` — downloaded from the public Hub
  and cached locally. This is the public default for shipped checkpoints (a fresh
  user with no W&B account / no local copy can still fetch them).
- W&B artifact references: ``"wandb:ENTITY/PROJECT/ARTIFACT:VERSION"`` — downloaded
  via the W&B public API and cached locally.

To select a specific file from a multi-file artifact, append ``#filename``; to pin a
HF revision, append ``@revision`` (before any ``#filename``)::

    "wandb:ENTITY/PROJECT/ARTIFACT:VERSION#MyModel_epoch21_MAE0.1416.pypots"
    "hf://MyHeartCounts/openmhc-lsm2-daily@v1.0#loss=0.2706.ckpt"

Examples::

    from utils.checkpoints import resolve_checkpoint_path

    # Local path (unchanged)
    path = resolve_checkpoint_path("results/mae/best.ckpt")

    # Hugging Face Hub (downloaded + cached; single .ckpt auto-selected)
    path = resolve_checkpoint_path("hf://MyHeartCounts/openmhc-lsm2-daily")

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
HF_PREFIX = "hf://"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "mhc-benchmark" / "artifacts"


def is_wandb_reference(path: str) -> bool:
    """Check whether *path* uses the ``wandb:`` prefix."""
    return path.startswith(WANDB_PREFIX)


def is_hf_reference(path: str) -> bool:
    """Check whether *path* uses the ``hf://`` prefix."""
    return path.startswith(HF_PREFIX)


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
    if is_hf_reference(path):
        return _download_hf_artifact(path, cache_dir)

    if is_wandb_reference(path):
        return _download_wandb_artifact(path, cache_dir)

    local = Path(path)
    if not local.exists():
        raise FileNotFoundError(f"Checkpoint not found: {local}")
    return local


def _download_hf_artifact(path: str, cache_dir: str | Path | None = None) -> Path:
    """Download a checkpoint from a Hugging Face Hub model repo.

    Reference format: ``hf://ORG/REPO[@REVISION][#FILENAME]``. When no ``#FILENAME``
    is given, the single ``.ckpt`` in the repo is auto-selected (``.pypots`` is used
    as a fallback, mirroring the W&B path).
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    ref = path[len(HF_PREFIX) :]  # strip prefix

    # Split off optional #filename selector, then optional @revision pin.
    filename = None
    if "#" in ref:
        ref, filename = ref.rsplit("#", 1)
    revision = None
    if "@" in ref:
        ref, revision = ref.rsplit("@", 1)

    repo_id = ref
    if repo_id.count("/") != 1:
        raise ValueError(
            f"Malformed hf reference: '{path}'. "
            "Expected format: hf://ORG/REPO[@revision][#filename]"
        )

    if filename is None:
        files = list_repo_files(repo_id, revision=revision)
        candidates = [f for f in files if f.endswith(".ckpt")]
        if not candidates:
            candidates = [
                f
                for f in files
                if f.endswith(".pypots") and not Path(f).name.startswith("events.out.tfevents")
            ]
        if not candidates:
            raise FileNotFoundError(
                f"No .ckpt or .pypots file in HF repo '{repo_id}'. Files: {files}"
            )
        if len(candidates) > 1:
            candidates = sorted(candidates)
            logger.warning(
                "Multiple checkpoints in %s, using %s. All: %s",
                repo_id,
                candidates[0],
                candidates,
            )
        filename = candidates[0]

    cache_root = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    cache_root.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Downloading HF checkpoint: %s/%s%s",
        repo_id,
        filename,
        f"@{revision}" if revision else "",
    )
    resolved = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=str(cache_root),
        )
    )
    logger.info(f"Resolved HF checkpoint to: {resolved}")
    return resolved


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
