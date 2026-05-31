"""Package a trained PyPOTS model into an openmhc release bundle.

A release bundle is the directory layout that
:meth:`openmhc.imputers.ReleaseLoadableMixin.from_release` consumes:

    release_dir/
    ├── model.pypots
    ├── normalization_stats.json
    ├── fourier_modes.json        (FEDformer only)
    └── openmhc_manifest.json

The bundle is movable — paths inside the manifest are relative to the
manifest file itself.

For FEDformer, this module additionally captures each
``FourierBlock.index`` from the in-memory trained model and writes it to
``fourier_modes.json``. Without that sidecar, PyPOTS re-draws the
indices on load and the trained weights operate on the wrong frequency
bins (the upstream PyPOTS bug). The corresponding restore logic lives
in :meth:`openmhc.imputers.pypots.FEDformerImputer._post_load`.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from openmhc.imputers._release import write_manifest

logger = logging.getLogger(__name__)


# Per-model whitelist of arch fields that the openmhc inference wrapper
# accepts. Other ModelConfig fields are training/factory bookkeeping and
# must NOT be written to the manifest (they would be splatted into the
# inference wrapper's __init__ as unexpected kwargs).
_ARCH_FIELDS: dict[str, tuple[str, ...]] = {
    "brits": (
        "n_steps", "n_features", "rnn_hidden_size",
    ),
    "dlinear": (
        "n_steps", "n_features", "d_model", "moving_avg_window_size",
    ),
    "timesnet": (
        "n_steps", "n_features", "n_layers", "top_k", "d_model", "d_ffn",
        "n_kernels", "dropout", "apply_nonstationary_norm",
    ),
    "fedformer": (
        # Note: openmhc's FEDformerImputer renames PyPOTS's "version" →
        # "variant" to avoid collision with the dataset version. We do
        # that translation below in build_arch().
        "n_steps", "n_features", "n_layers", "d_model", "n_heads", "d_ffn",
        "moving_avg_window_size", "dropout", "modes", "mode_select",
    ),
}


def build_arch(model_config) -> dict[str, Any]:
    """Slice a ModelConfig down to the fields the inference wrapper accepts.

    Also performs the ``version → variant`` rename for FEDformer
    (PyPOTS's ``version`` collides with the openmhc dataset version arg).
    """
    name = model_config.model_name.lower()
    if name not in _ARCH_FIELDS:
        raise ValueError(
            f"No arch whitelist registered for model {name!r}. "
            "Add an entry to imputation_training.release._ARCH_FIELDS."
        )
    arch = {k: getattr(model_config, k) for k in _ARCH_FIELDS[name]}
    if name == "fedformer":
        arch["variant"] = model_config.version  # PyPOTS field → openmhc field
    return arch


def extract_fourier_indices(model: Any) -> dict[str, list[int]]:
    """Return ``{module_dotted_path: index_list}`` for every FourierBlock.

    ``model`` is a PyPOTS imputer object; ``model.model`` is the inner
    ``nn.Module``. We walk ``named_modules()`` (matching the same dotted
    paths the inference-side restoration uses) and pull ``self.index``
    from each ``FourierBlock``.

    Returns an empty dict if the model has no FourierBlocks (e.g.,
    BRITS) — callers should skip writing the sidecar in that case.
    """
    out: dict[str, list[int]] = {}
    inner = model.model
    for name, module in inner.named_modules():
        if type(module).__name__ != "FourierBlock":
            continue
        idx = list(module.index)
        if not all(isinstance(i, int) for i in idx):
            # PyPOTS stores plain Python ints; defensively coerce.
            idx = [int(i) for i in idx]
        out[name] = idx
    return out


def write_release(
    *,
    model: Any,
    model_config,
    release_dir: str | Path,
    pypots_checkpoint: str | Path,
    normalization_stats: str | Path | None = None,
    provenance: dict[str, Any] | None = None,
) -> Path:
    """Build a complete openmhc release bundle.

    Args:
        model: The trained PyPOTS imputer object (in-memory).
            Required so that ``FourierBlock.index`` can be captured.
        model_config: The :class:`ModelConfig` used to construct
            ``model``. Used to derive the manifest's ``arch`` dict.
        release_dir: Where to write the bundle. Created if missing.
            Files inside are overwritten on each call (idempotent).
        pypots_checkpoint: Path to the ``.pypots`` file PyPOTS produced.
            Copied to ``release_dir / model.pypots``.
        normalization_stats: Optional path to a ``normalization_stats.json``.
            Copied to ``release_dir / normalization_stats.json``. Pass
            ``None`` only for models trained on un-normalized inputs.
        provenance: Optional metadata dict — stored as-is under the
            manifest's ``provenance`` field.

    Returns:
        Path to the release directory.
    """
    release_dir = Path(release_dir)
    release_dir.mkdir(parents=True, exist_ok=True)

    # Copy checkpoint into the bundle under a stable name.
    src_ckpt = Path(pypots_checkpoint)
    if not src_ckpt.exists():
        raise FileNotFoundError(f"PyPOTS checkpoint not found: {src_ckpt}")
    dst_ckpt = release_dir / "model.pypots"
    if src_ckpt.resolve() != dst_ckpt.resolve():
        shutil.copy2(src_ckpt, dst_ckpt)
        logger.info("Copied checkpoint %s -> %s", src_ckpt, dst_ckpt)

    # Copy normalization stats if provided.
    stats_filename: str | None = None
    if normalization_stats is not None:
        src_stats = Path(normalization_stats)
        if not src_stats.exists():
            raise FileNotFoundError(f"Normalization stats not found: {src_stats}")
        dst_stats = release_dir / "normalization_stats.json"
        if src_stats.resolve() != dst_stats.resolve():
            shutil.copy2(src_stats, dst_stats)
        stats_filename = "normalization_stats.json"

    # FEDformer-only: capture FourierBlock.index values to a sidecar.
    # Skip silently for other models (extract_fourier_indices returns {}).
    fourier_filename: str | None = None
    indices = extract_fourier_indices(model)
    if indices:
        if model_config.model_name.lower() != "fedformer":
            raise RuntimeError(
                f"Found FourierBlock modules in {model_config.model_name!r} model "
                "but only FEDformer is registered to carry a fourier_modes sidecar. "
                "Add the model_name to openmhc.imputers._release._FOURIER_MODES_KINDS."
            )
        sidecar_path = release_dir / "fourier_modes.json"
        sidecar_path.write_text(json.dumps(indices, indent=2, sort_keys=True))
        fourier_filename = "fourier_modes.json"
        logger.info(
            "Wrote fourier_modes sidecar (%d FourierBlock entries) to %s",
            len(indices),
            sidecar_path,
        )

    arch = build_arch(model_config)
    write_manifest(
        release_dir,
        kind=model_config.model_name.lower(),
        arch=arch,
        checkpoint="model.pypots",
        normalization_stats=stats_filename,
        fourier_modes=fourier_filename,
        provenance=provenance or {},
    )
    logger.info("Release bundle written to %s", release_dir)
    return release_dir


def model_config_to_dict(model_config) -> dict[str, Any]:
    """Convenience helper for inclusion in provenance dicts."""
    return asdict(model_config)
