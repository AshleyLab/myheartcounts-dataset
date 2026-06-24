"""Manifest format for distributing Track-1 (outcome-prediction) encoder checkpoints.

A "release" is a directory with a manifest plus the checkpoint payload::

    my-release/
    ├── model.ckpt                 # a Lightning .ckpt (encoder weights)
    ├── normalization_stats.json   # the canonical hourly z-score constants
    └── openmhc_manifest.json

Loading at inference time is a single call::

    enc = WBM.from_release("my-release/")
    enc = WBM.from_release("hf://MyHeartCounts/openmhc-wbm-dp")
    enc = WBM.from_release("hf://MyHeartCounts/openmhc-wbm-dp@v1.0")

The manifest schema is intentionally identical to the forecasting / imputation
release manifests (``spec_version == 1``) so the same tooling stages and
publishes all three tracks' bundles.

Paths inside the manifest are stored *relative to the manifest file* so the
whole directory is movable. ``normalization_stats`` may be ``null`` for
checkpoints that normalize internally.

Loading from the Hugging Face Hub requires the optional ``[hf]`` extra
(``pip install 'openmhc[hf]'``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MANIFEST_FILENAME = "openmhc_manifest.json"
SPEC_VERSION = 1
HF_URI_PREFIX = "hf://"

_KNOWN_KINDS = frozenset({"wbm"})

# Only the bundle's payload files are pulled from HF — model cards and other
# repo metadata are skipped. WBM ships a Lightning ``.ckpt`` plus the canonical
# hourly normalization stats.
_HF_ALLOW_PATTERNS = (
    MANIFEST_FILENAME,
    "normalization_stats.json",
    "*.ckpt",
    "*.pt",
    "*.pth",
)


@dataclass(frozen=True)
class Manifest:
    """Parsed, path-resolved encoder release manifest.

    ``checkpoint_path`` and ``normalization_stats_path`` are absolute paths
    resolved against the manifest file's directory. ``arch`` is the dict of
    training-time architecture kwargs, ready to splat into the wrapper
    constructor.
    """

    spec_version: int
    kind: str
    arch: dict[str, Any]
    checkpoint_path: Path
    normalization_stats_path: Path | None
    provenance: dict[str, Any]
    manifest_path: Path


def _resolve_hf_manifest(uri: str) -> Path:
    """Snapshot-download an ``hf://org/repo[@revision]`` bundle.

    Caches via ``huggingface_hub``'s default location
    (``~/.cache/huggingface/hub``, controllable with ``HF_HOME``).

    Raises:
        ImportError: If ``huggingface_hub`` is not installed.
        FileNotFoundError: If the snapshot contains no manifest file.
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
    p = Path(path).expanduser()
    if p.is_dir():
        candidate = p / MANIFEST_FILENAME
        if not candidate.exists():
            raise FileNotFoundError(f"No {MANIFEST_FILENAME} in directory {p}")
        return candidate
    if not p.exists():
        raise FileNotFoundError(f"Manifest path does not exist: {p}")
    return p


def load_manifest(path: str | Path) -> Manifest:
    """Read and validate an openmhc encoder release manifest.

    Args:
        path: A manifest file (``openmhc_manifest.json``), a directory
            containing one, or an ``hf://org/repo[@revision]`` URI.

    Returns:
        A :class:`Manifest` with ``checkpoint_path`` and
        ``normalization_stats_path`` resolved against the manifest's directory.

    Raises:
        FileNotFoundError: If the manifest, checkpoint, or stats file is missing.
        ValueError: If the manifest schema is invalid.
    """
    manifest_file = _resolve_manifest_path(path)
    raw = json.loads(manifest_file.read_text(encoding="utf-8"))
    base = manifest_file.parent

    spec_version = raw.get("spec_version")
    if spec_version != SPEC_VERSION:
        raise ValueError(
            f"Unsupported encoder manifest spec_version {spec_version!r}; "
            f"this build understands {SPEC_VERSION}"
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

    arch = raw.get("arch") or {}
    if not isinstance(arch, dict):
        raise ValueError("Manifest field 'arch' must be a dict if present")

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
    )


def write_manifest(
    directory: str | Path,
    *,
    kind: str,
    checkpoint: str,
    arch: dict[str, Any] | None = None,
    normalization_stats: str | None = None,
    provenance: dict[str, Any] | None = None,
    filename: str = MANIFEST_FILENAME,
) -> Path:
    """Write an encoder release manifest into ``directory``.

    Paths in ``checkpoint`` and ``normalization_stats`` are stored as-is and
    interpreted at load time relative to the manifest's directory — typically a
    sibling filename.

    Args:
        directory: Release directory; created if missing.
        kind: Model kind, one of ``{"wbm"}``.
        checkpoint: Path to the checkpoint file, relative to ``directory``
            (e.g. ``"model.ckpt"``).
        arch: Optional training-time architecture kwargs splatted into the
            wrapper constructor.
        normalization_stats: Path to the normalization stats JSON, relative to
            ``directory``. ``None`` for models that normalize internally.
        provenance: Optional metadata (W&B artifact, paper table, etc.) —
            stored, not interpreted.
        filename: Manifest filename (defaults to ``openmhc_manifest.json``).

    Returns:
        Path to the written manifest file.
    """
    if kind not in _KNOWN_KINDS:
        raise ValueError(f"Unknown manifest kind {kind!r}; expected one of {sorted(_KNOWN_KINDS)}")
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "spec_version": SPEC_VERSION,
        "kind": kind,
        "checkpoint": checkpoint,
        "normalization_stats": normalization_stats,
        "arch": dict(arch) if arch else {},
        "provenance": dict(provenance) if provenance else {},
    }
    out = out_dir / filename
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


class ReleaseLoadableMixin:
    """Adds :meth:`from_release` to an encoder wrapper class.

    Subclasses must set the class-level attribute ``model_name`` to a value
    that matches the manifest's ``kind`` field (e.g. ``"wbm"``). The constructor
    must accept ``model_path``, ``normalization_stats_path``, and every key in
    the manifest's ``arch`` dict as keyword arguments.
    """

    model_name: str = ""

    @classmethod
    def from_release(cls, path: str | Path, **runtime_kwargs):
        """Construct from a release directory, manifest file, or ``hf://`` URI.

        Args:
            path: A release directory, a direct path to a manifest file, or an
                ``hf://org/repo[@revision]`` URI for a bundle on the Hugging
                Face Hub.
            **runtime_kwargs: Forwarded to the constructor (e.g. ``data_dir=...``,
                ``seed=...``). Must not duplicate any key in the manifest's
                ``arch`` dict.

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
        return cls(
            model_path=str(manifest.checkpoint_path),
            normalization_stats_path=stats_path,
            **manifest.arch,
            **runtime_kwargs,
        )
