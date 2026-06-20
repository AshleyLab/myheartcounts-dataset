"""Manifest format for distributing imputer checkpoints.

A "release" is a directory with three files::

    my-release/
    ├── model.{pypots,ckpt,...}
    ├── normalization_stats.json
    └── openmhc_manifest.json

The manifest captures the training-time invariants — which model class,
which architecture hyperparameters, which stats file — so loading at
inference time is a single call::

    imp = BRITSImputer.from_release("my-release/")

The same call also accepts an ``hf://org/repo[@revision]`` URI to load a
bundle published on the Hugging Face Hub::

    imp = BRITSImputer.from_release("hf://MyHeartCounts/openmhc-brits-imp")
    imp = BRITSImputer.from_release("hf://MyHeartCounts/openmhc-brits-imp@v1.0")

Loading from HF requires the optional ``[hf]`` extra
(``pip install 'openmhc[hf]'``).

Paths inside the manifest are stored *relative to the manifest file* so
the whole directory is movable. ``normalization_stats`` may be ``null``
for checkpoints trained on raw (un-normalized) inputs.

The :class:`ReleaseLoadableMixin` is shared across imputer families
(PyPOTS, LSM2, ...) — subclasses set ``model_name`` to match the
manifest's ``kind`` field and accept ``manifest.arch`` as constructor
kwargs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MANIFEST_FILENAME = "openmhc_manifest.json"
SPEC_VERSION = 2
# Spec versions we can load. v1 lacked the optional ``fourier_modes`` sidecar
# field; v2 adds it. v1 manifests still load via the same code path — the new
# field is parsed only when present.
_SUPPORTED_SPEC_VERSIONS = frozenset({1, 2})
HF_URI_PREFIX = "hf://"

_KNOWN_KINDS = frozenset(
    {
        "brits",
        "timesnet",
        "dlinear",
        "fedformer",
        "lsm2",
        "lsm2_weekly_sparse",
    }
)

# Kinds that may carry a Fourier-modes sidecar. The sidecar exists to work
# around an upstream PyPOTS bug: ``FourierBlock.__init__`` calls
# ``np.random.shuffle(index)`` and stores the result on ``self.index`` as a
# plain Python attribute, so it isn't saved to ``state_dict``. We capture
# those indices at training time and restore them post-load.
_FOURIER_MODES_KINDS = frozenset({"fedformer"})

# Only the bundle's payload files are pulled from HF — model cards and any other
# repo metadata are skipped.
_HF_ALLOW_PATTERNS = (
    MANIFEST_FILENAME,
    "normalization_stats.json",
    "fourier_modes.json",
    "*.pypots",
    "*.ckpt",
    "*.pt",
    "*.pth",
)


@dataclass(frozen=True)
class Manifest:
    """Parsed, path-resolved release manifest.

    ``checkpoint_path``, ``normalization_stats_path``, and
    ``fourier_modes_path`` are absolute paths resolved against the
    manifest file's directory. ``arch`` is the dict of training-time
    architecture kwargs, ready to splat into the wrapper's constructor.

    ``fourier_modes_path`` is only set for kinds that need it
    (currently ``"fedformer"``) and only for manifests written by a
    trainer that knows about the upstream PyPOTS index-not-in-state-dict
    bug. Older manifests (spec_version 1) always have it as ``None``;
    consumers should fall back to the legacy "re-draw on construct"
    behaviour in that case.
    """

    spec_version: int
    kind: str
    arch: dict[str, Any]
    checkpoint_path: Path
    normalization_stats_path: Path | None
    provenance: dict[str, Any]
    manifest_path: Path
    fourier_modes_path: Path | None = None


def _resolve_hf_manifest(uri: str) -> Path:
    """Snapshot-download an ``hf://org/repo[@revision]`` bundle.

    The remote bundle layout is the same as a local release directory: a
    manifest plus its referenced checkpoint and optional normalization
    stats. Files outside that allowlist (e.g. the model card) are skipped.
    Caches via ``huggingface_hub``'s own default location
    (``~/.cache/huggingface/hub``, controllable with ``HF_HOME``).

    Args:
        uri: An ``hf://org/repo`` or ``hf://org/repo@revision`` URI.
            ``revision`` may be any git ref (tag, branch, commit) accepted
            by ``huggingface_hub.snapshot_download``.

    Returns:
        Absolute path to the downloaded manifest file.

    Raises:
        ImportError: If ``huggingface_hub`` is not installed.
        FileNotFoundError: If the downloaded snapshot does not contain a
            manifest file.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "Loading hf:// release bundles requires huggingface_hub. "
            "Install it with: pip install 'openmhc[hf]'"
        ) from exc
    rest = uri[len(HF_URI_PREFIX) :]
    if "@" in rest:
        repo_id, revision = rest.split("@", 1)
    else:
        repo_id, revision = rest, None
    local_dir = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=list(_HF_ALLOW_PATTERNS),
    )
    manifest = Path(local_dir) / MANIFEST_FILENAME
    if not manifest.exists():
        suffix = f" (revision={revision!r})" if revision else ""
        raise FileNotFoundError(f"HF repo {repo_id!r}{suffix} contains no {MANIFEST_FILENAME}")
    return manifest


def _resolve_manifest_path(path: str | Path) -> Path:
    """Accept the manifest file, a directory containing it, or an ``hf://`` URI."""
    if isinstance(path, str) and path.startswith(HF_URI_PREFIX):
        return _resolve_hf_manifest(path)
    p = Path(path)
    if p.is_dir():
        candidate = p / MANIFEST_FILENAME
        if not candidate.exists():
            raise FileNotFoundError(f"No {MANIFEST_FILENAME} in directory {p}")
        return candidate
    if not p.exists():
        raise FileNotFoundError(f"Manifest path does not exist: {p}")
    return p


def load_manifest(path: str | Path) -> Manifest:
    """Read and validate an openmhc release manifest.

    Args:
        path: A manifest file (``openmhc_manifest.json``), a directory
            containing one, or an ``hf://org/repo[@revision]`` URI
            pointing at a Hugging Face Hub repo.

    Returns:
        A :class:`Manifest` with ``checkpoint_path`` and
        ``normalization_stats_path`` resolved against the manifest's
        directory.

    Raises:
        FileNotFoundError: If the manifest, checkpoint, or stats file is
            missing.
        ValueError: If the manifest schema is invalid.
    """
    manifest_file = _resolve_manifest_path(path)
    raw = json.loads(manifest_file.read_text())
    base = manifest_file.parent

    spec_version = raw.get("spec_version")
    if spec_version not in _SUPPORTED_SPEC_VERSIONS:
        raise ValueError(
            f"Unsupported manifest spec_version {spec_version!r}; "
            f"this build understands {sorted(_SUPPORTED_SPEC_VERSIONS)}"
        )

    kind = raw.get("kind")
    if kind not in _KNOWN_KINDS:
        raise ValueError(f"Unknown manifest kind {kind!r}; expected one of {sorted(_KNOWN_KINDS)}")

    checkpoint_rel = raw.get("checkpoint")
    if not checkpoint_rel:
        raise ValueError("Manifest missing required field 'checkpoint'")
    checkpoint_path = (base / checkpoint_rel).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Manifest references missing checkpoint: {checkpoint_path}")

    stats_rel = raw.get("normalization_stats")
    if stats_rel is None:
        stats_path: Path | None = None
    else:
        stats_path = (base / stats_rel).resolve()
        if not stats_path.exists():
            raise FileNotFoundError(f"Manifest references missing stats file: {stats_path}")

    # Spec v2 added an optional ``fourier_modes`` sidecar (FEDformer only).
    # In v1 manifests the field is simply absent.
    fourier_rel = raw.get("fourier_modes")
    if fourier_rel is None:
        fourier_path: Path | None = None
    else:
        if kind not in _FOURIER_MODES_KINDS:
            raise ValueError(
                f"Manifest kind {kind!r} cannot carry a 'fourier_modes' sidecar; "
                f"allowed kinds: {sorted(_FOURIER_MODES_KINDS)}"
            )
        fourier_path = (base / fourier_rel).resolve()
        if not fourier_path.exists():
            raise FileNotFoundError(
                f"Manifest references missing fourier_modes sidecar: {fourier_path}"
            )

    arch = raw.get("arch")
    if not isinstance(arch, dict):
        raise ValueError("Manifest field 'arch' must be a dict")

    provenance = raw.get("provenance") or {}
    if not isinstance(provenance, dict):
        raise ValueError("Manifest field 'provenance' must be a dict if present")

    return Manifest(
        spec_version=spec_version,
        kind=kind,
        arch=dict(arch),
        checkpoint_path=checkpoint_path,
        normalization_stats_path=stats_path,
        provenance=dict(provenance),
        manifest_path=manifest_file,
        fourier_modes_path=fourier_path,
    )


def write_manifest(
    directory: str | Path,
    *,
    kind: str,
    arch: dict[str, Any],
    checkpoint: str,
    normalization_stats: str | None = None,
    fourier_modes: str | None = None,
    provenance: dict[str, Any] | None = None,
    filename: str = MANIFEST_FILENAME,
) -> Path:
    """Write a release manifest into ``directory``.

    Paths in ``checkpoint``, ``normalization_stats`` and ``fourier_modes``
    are stored as-is and interpreted at load time relative to the
    manifest's directory — typically just filenames pointing at siblings.

    Args:
        directory: Release directory; will be created if missing.
        kind: Model kind, one of ``{"brits", "timesnet", "dlinear", "fedformer"}``.
        arch: Training-time architecture kwargs (e.g. ``{"n_steps": 1440,
            "n_features": 19, "rnn_hidden_size": 128}`` for BRITS).
        checkpoint: Path to the ``.pypots`` file, relative to ``directory``.
        normalization_stats: Path to the stats JSON, relative to
            ``directory``. ``None`` if the model expects raw inputs.
        fourier_modes: Path to a Fourier-modes sidecar JSON, relative to
            ``directory``. Only valid when ``kind == "fedformer"`` — captures
            the training-time ``FourierBlock.index`` values that PyPOTS
            does not persist in ``state_dict``. ``None`` to omit
            (manifest stays spec v1 compatible for non-FEDformer kinds).
        provenance: Optional metadata (training run id, dataset version,
            paper table, etc.) — not interpreted, just stored.
        filename: Manifest filename (defaults to ``openmhc_manifest.json``).

    Returns:
        Path to the written manifest file.
    """
    if kind not in _KNOWN_KINDS:
        raise ValueError(f"Unknown manifest kind {kind!r}; expected one of {sorted(_KNOWN_KINDS)}")
    if fourier_modes is not None and kind not in _FOURIER_MODES_KINDS:
        raise ValueError(
            f"Manifest kind {kind!r} cannot carry a 'fourier_modes' sidecar; "
            f"allowed kinds: {sorted(_FOURIER_MODES_KINDS)}"
        )
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "spec_version": SPEC_VERSION,
        "kind": kind,
        "checkpoint": checkpoint,
        "normalization_stats": normalization_stats,
        "arch": dict(arch),
        "provenance": dict(provenance) if provenance else {},
    }
    if fourier_modes is not None:
        payload["fourier_modes"] = fourier_modes
    out = out_dir / filename
    out.write_text(json.dumps(payload, indent=2))
    return out


class ReleaseLoadableMixin:
    """Adds :meth:`from_release` to any imputer wrapper class.

    Subclasses must set the class-level attribute ``model_name`` to a value
    that matches the manifest's ``kind`` field (e.g. ``"brits"``, ``"lsm2"``).
    The constructor must accept ``model_path``, ``normalization_stats_path``,
    and every key in the manifest's ``arch`` dict as keyword arguments.
    """

    model_name: str = ""

    @classmethod
    def from_release(cls, path: str | Path, **runtime_kwargs):
        """Construct from a release directory containing an ``openmhc_manifest.json``.

        The manifest captures training-time invariants (model kind,
        architecture hyperparameters, paths to the checkpoint and
        normalization stats). Runtime knobs like ``device``,
        ``inference_batch_size``, and ``data_dir`` can be passed as
        keyword arguments.

        Args:
            path: A release directory, a direct path to a manifest file,
                or an ``hf://org/repo[@revision]`` URI for a bundle on the
                Hugging Face Hub.
            **runtime_kwargs: Forwarded to the constructor (e.g.
                ``device="cuda:0"``). Must not duplicate any key in the
                manifest's ``arch`` dict.

        Returns:
            An instance of the calling class.

        Raises:
            ValueError: If the manifest's ``kind`` does not match
                ``cls.model_name``.
        """
        manifest = load_manifest(path)
        if manifest.kind != cls.model_name:
            raise ValueError(
                f"Manifest is for kind {manifest.kind!r}, but "
                f"{cls.__name__} expects kind {cls.model_name!r}. "
                f"Use the matching wrapper class."
            )
        stats_path = (
            str(manifest.normalization_stats_path)
            if manifest.normalization_stats_path is not None
            else None
        )
        # Spec v2 may carry a Fourier-modes sidecar for FEDformer; thread it
        # through as a kwarg so :meth:`FEDformerImputer._post_load` can
        # restore ``module.index`` after PyPOTS' state_dict load. Other
        # kinds never see this kwarg (validated by ``load_manifest``).
        extra_kwargs: dict[str, Any] = {}
        if manifest.fourier_modes_path is not None:
            extra_kwargs["fourier_modes_path"] = str(manifest.fourier_modes_path)
        return cls(
            model_path=str(manifest.checkpoint_path),
            normalization_stats_path=stats_path,
            **manifest.arch,
            **extra_kwargs,
            **runtime_kwargs,
        )
