#!/usr/bin/env python
r"""Build openmhc release manifests from private-repo eval configs.

Each ``results/imputation_eval/*/config.yaml`` from the private repo
records exactly which checkpoint + stats file + architecture
hyperparameters produced a paper number. This script reads one or more
such configs and stages a release directory per config containing:

    <output_dir>/<release_name>/
    ├── model.pypots                  (copied from the original checkpoint)
    ├── normalization_stats.json      (copied if present and not W&B-only)
    └── openmhc_manifest.json         (written from the config)

Usage::

    python tools/build_manifest.py \\
        --source-repo /path/to/MHC-benchmark \\
        --output-dir releases/ \\
        --config results/imputation_eval/brits_max91d_*/config.yaml

W&B model references (``wandb:...``) are skipped with a warning — they
require ``wandb artifact get`` first; rerun the script pointing
``--config`` at a local copy once the artifact has been pulled.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Private-repo configs were dumped with ``yaml.dump`` and contain
# ``!!python/tuple`` tags (e.g. for ``figsize_per_channel``). Teach the
# safe loader to read tuples as lists rather than refusing.
yaml.SafeLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    lambda loader, node: list(loader.construct_sequence(node)),
)

# Ensure ``openmhc.imputers._release`` is importable when running from a
# fresh checkout: prefer the in-repo src layout over any installed copy.
_REPO_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from openmhc.imputers._release import write_manifest  # noqa: E402

logger = logging.getLogger("build_manifest")

# Per-kind arch field whitelists. Keys not in this list are dropped from the
# manifest's ``arch`` block even when present in the source config (the source
# YAML carries every PyPOTSMethodConfig field regardless of model).
ARCH_FIELDS_BY_KIND: dict[str, tuple[str, ...]] = {
    "brits": ("n_steps", "n_features", "rnn_hidden_size"),
    "timesnet": (
        "n_steps",
        "n_features",
        "n_layers",
        "top_k",
        "d_model",
        "d_ffn",
        "n_kernels",
        "dropout",
        "apply_nonstationary_norm",
    ),
    "dlinear": (
        "n_steps",
        "n_features",
        "moving_avg_window_size",
        "d_model",
    ),
    "fedformer": (
        "n_steps",
        "n_features",
        "n_layers",
        "d_model",
        "n_heads",
        "d_ffn",
        "moving_avg_window_size",
        "dropout",
        "version",
        "modes",
        "mode_select",
    ),
    "lsm2": (
        "seq_length",
        "patch_size",
        "in_channels",
        "embed_dim",
        "depth",
        "num_heads",
        "decoder_embed_dim",
        "decoder_depth",
        "decoder_num_heads",
        "mlp_ratio",
        "mask_ratio",
    ),
    "lsm2_weekly_sparse": (
        # shared with daily/weekly
        "seq_length",
        "patch_size",
        "in_channels",
        "embed_dim",
        "depth",
        "num_heads",
        "decoder_embed_dim",
        "decoder_depth",
        "decoder_num_heads",
        "mlp_ratio",
        "mask_ratio",
        # weekly-sparse-specific
        "num_days",
        "window_minutes",
        "use_rope_day_embed",
        "freeze_encoder",
    ),
}

# Per-kind YAML→manifest field renames. The PyPOTS YAML for FEDformer uses
# `version` (matching PyPOTS's own kwarg) for the frequency basis, but the
# OpenMHC wrapper exposes that as `variant` (its `version` kwarg is reserved
# for the dataset version). Storing `variant` in the manifest is what makes
# `from_release` load cleanly.
_ARCH_FIELD_RENAMES: dict[str, dict[str, str]] = {
    "fedformer": {"version": "variant"},
}

# Kinds whose `method.<kind>` YAML block doesn't carry architecture
# fields — for those, architecture and normalization stats must be
# extracted from the Lightning checkpoint instead.
_LSM2_KINDS = frozenset({"lsm2", "lsm2_weekly_sparse"})

# Source-repo `method.type` values that map to LSM2 kinds. The private
# repo still uses the old `mae` / `mae_weekly_sparse` names; we accept
# both and normalize on the way in.
_LSM2_TYPE_ALIASES: dict[str, str] = {
    "mae": "lsm2",
    "mae_weekly_sparse": "lsm2_weekly_sparse",
    "lsm2": "lsm2",
    "lsm2_weekly_sparse": "lsm2_weekly_sparse",
}


def _load_ckpt(path: Path) -> dict[str, Any]:
    """Load a Lightning ``.ckpt`` (or any torch-saved dict) for inspection."""
    import torch  # heavy dep — local import

    return torch.load(str(path), map_location="cpu", weights_only=False)


def _extract_lsm2_arch_from_ckpt(ckpt: dict[str, Any], kind: str) -> dict[str, Any]:
    """Pull architecture hparams from Lightning's saved ``hyper_parameters`` block.

    Lightning stores constructor kwargs of the ``LightningModule`` here when
    ``self.save_hyperparameters()`` was called at training time. The MAE
    Lightning modules accept the model arch as constructor args directly,
    so we filter the saved hparams by the per-kind whitelist.
    """
    hparams = ckpt.get("hyper_parameters") or ckpt.get("hparams") or {}
    if hasattr(hparams, "items") and not isinstance(hparams, dict):
        hparams = dict(hparams)  # Namespace-like objects
    fields = ARCH_FIELDS_BY_KIND[kind]
    arch: dict[str, Any] = {}
    missing: list[str] = []
    for field_name in fields:
        if field_name in hparams:
            value = hparams[field_name]
            # JSON manifest doesn't tolerate non-primitive types — coerce.
            if isinstance(value, (list, tuple)):
                value = list(value)
            arch[field_name] = value
        else:
            missing.append(field_name)
    if missing:
        logger.warning(
            "Checkpoint hparams are missing %s for kind=%s; manifest arch "
            "will be incomplete and the wrapper will fall back to its "
            "default values (Lightning will still rebuild the model from "
            "saved hparams at load time).",
            missing,
            kind,
        )
    return arch


def _extract_lsm2_stats_from_ckpt(ckpt: dict[str, Any]) -> dict[str, Any] | None:
    """Extract normalization stats stored by the Lightning DataModule.

    Lightning saves DataModule ``state_dict`` under a top-level key matching
    the DataModule's class name (e.g. ``MAEDailyDataModule``), or — for older
    PL versions — under the generic ``LightningDataModule`` key. Scan all
    top-level dict values for one carrying ``normalization_stats``.

    Returns a dict matching the public ``normalization_stats.json`` schema
    (``means``, ``stds``, ``channels``, ``epsilon``) or ``None`` if the
    checkpoint doesn't carry stats (older training runs).
    """
    norm = None
    # Direct keys to try first (cheap), then a broader scan.
    for key in ("LightningDataModule", "datamodule_state_dict"):
        block = ckpt.get(key)
        if isinstance(block, dict) and "normalization_stats" in block:
            norm = block["normalization_stats"]
            break
    if norm is None:
        for key, value in ckpt.items():
            if isinstance(value, dict) and "normalization_stats" in value:
                norm = value["normalization_stats"]
                break
    if norm is None:
        return None
    means = list(norm["mean_prior"])
    stds = list(norm["std_prior"])
    channels = norm.get("channels")
    # The trained-stats dict lists the channels that were *normalized* (the 7
    # continuous ones). Binary channels carry identity values in the prior.
    if channels is None:
        channels = list(range(7))
    return {
        "means": [float(m) for m in means],
        "stds": [float(s) for s in stds],
        "channels": [int(c) for c in channels],
        "epsilon": 1e-8,
    }


@dataclass
class BuildResult:
    """What :func:`build_release` produced (or skipped)."""

    config_path: Path
    status: str  # "built" | "skipped"
    release_dir: Path | None
    reason: str | None = None


def _pick_checkpoint(repo_root: Path, model_path: str) -> Path:
    """Resolve a config's ``model_path`` to a single ``.pypots`` file.

    Accepts a file path, a directory containing one or more ``.pypots``
    files (first sorted match wins, matching the wrapper's own
    resolution), or a ``wandb:`` URI which is rejected with a clear
    message — pull the artifact locally and rerun.
    """
    if model_path.startswith("wandb:"):
        raise ValueError(
            f"W&B artifact reference {model_path!r}; "
            "run `wandb artifact get` and rerun with --config pointing at a "
            "local copy of the config (or override model_path)."
        )
    candidate = Path(model_path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    if not candidate.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {candidate}")
    if candidate.is_file():
        return candidate
    matches = sorted(candidate.glob("*.pypots"))
    if not matches:
        raise FileNotFoundError(f"No .pypots file under directory {candidate}")
    return matches[0]


def _resolve_stats(repo_root: Path, stats_path: str | None) -> Path | None:
    """Resolve the optional stats path against ``repo_root``; ``None`` is fine."""
    if not stats_path:
        return None
    p = Path(stats_path)
    if not p.is_absolute():
        p = repo_root / p
    if p.exists():
        return p
    logger.warning(
        "Stats file referenced by config does not exist locally: %s "
        "(manifest will be written without normalization_stats)",
        p,
    )
    return None


def _extract_arch(pypots_block: dict[str, Any], kind: str) -> dict[str, Any]:
    """Filter the YAML's pypots block down to fields relevant to ``kind``.

    Applies the per-kind YAML→manifest rename in :data:`_ARCH_FIELD_RENAMES`
    so the emitted manifest matches the wrapper's constructor kwargs.
    """
    fields = ARCH_FIELDS_BY_KIND[kind]
    renames = _ARCH_FIELD_RENAMES.get(kind, {})
    arch: dict[str, Any] = {}
    for field_name in fields:
        if field_name not in pypots_block:
            raise KeyError(
                f"Source config is missing required arch field {field_name!r} for kind={kind!r}"
            )
        arch[renames.get(field_name, field_name)] = pypots_block[field_name]
    return arch


def _derive_provenance(
    config_path: Path,
    pypots_block: dict[str, Any],
    output_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a non-empty provenance block from config breadcrumbs."""
    provenance: dict[str, Any] = {
        "source_config": str(config_path),
    }
    eval_run = config_path.parent.name
    if eval_run:
        provenance["eval_run"] = eval_run

    # Pull the timestamp out of the checkpoint path when possible. Paths look
    # like ``models/pypots/brits/20260409_T005730/X.pypots`` so the parent dir
    # name is a recognisable training-run id.
    raw_model_path = pypots_block.get("model_path", "")
    if raw_model_path and not raw_model_path.startswith("wandb:"):
        parent = Path(raw_model_path).parent.name
        if parent:
            provenance["training_run"] = parent
    elif raw_model_path.startswith("wandb:"):
        provenance["wandb_artifact"] = raw_model_path

    if output_config:
        exp = output_config.get("experiment_name")
        if exp:
            provenance["experiment_name"] = exp

    return provenance


def build_release(
    config_path: Path,
    *,
    source_repo: Path,
    output_dir: Path,
    release_name: str | None = None,
    overwrite: bool = False,
    copy_checkpoint: bool = True,
    extra_provenance: dict[str, Any] | None = None,
    model_path_override: str | Path | None = None,
    stats_path_override: str | Path | None = None,
) -> BuildResult:
    """Stage one release directory from one eval config.

    Args:
        config_path: Path to a ``results/imputation_eval/*/config.yaml``.
        source_repo: Repo root used to resolve relative paths in the config.
        output_dir: Parent dir that will receive a per-config subdir.
        release_name: Subdir name (defaults to the eval run name with
            ``pypots_`` stripped).
        overwrite: Replace an existing release dir if present.
        copy_checkpoint: When ``False``, only the manifest + stats are
            staged and the manifest's ``checkpoint`` field points at the
            original file with a relative path. Useful for testing
            without duplicating large weight files.
        extra_provenance: Merged into the derived provenance block.
        model_path_override: When given, use this checkpoint path instead of
            the one recorded in the config.
        stats_path_override: When given, use this normalization-stats path
            instead of the one recorded in the config.
    """
    config = yaml.safe_load(config_path.read_text())
    method = config.get("method") or {}
    method_type = method.get("type", "")

    # Dispatch on family.
    is_lsm2 = method_type in _LSM2_TYPE_ALIASES
    is_pypots = method_type == "pypots"
    if not (is_lsm2 or is_pypots):
        return BuildResult(
            config_path,
            "skipped",
            None,
            f"unsupported method.type {method_type!r}",
        )

    # ------------------------------------------------------------------
    # Pull out a per-family block + resolve the source checkpoint path.
    # ------------------------------------------------------------------
    extracted_stats: dict[str, Any] | None = None

    if is_pypots:
        method_block: dict[str, Any] = method.get("pypots") or {}
        kind = method_block.get("model_name", "").lower()
        if kind not in ARCH_FIELDS_BY_KIND:
            return BuildResult(
                config_path,
                "skipped",
                None,
                f"unsupported model_name {kind!r}; expected one of {sorted(ARCH_FIELDS_BY_KIND)}",
            )
        raw_model_path = (
            str(model_path_override)
            if model_path_override is not None
            else method_block.get("model_path", "")
        )
        stats_yaml_field = "normalization_stats_path"
        path_field = "model_path"
    else:  # is_lsm2
        method_block = method.get("lsm2") or method.get("mae") or {}
        kind = _LSM2_TYPE_ALIASES[method_type]
        raw_model_path = (
            str(model_path_override)
            if model_path_override is not None
            else method_block.get("checkpoint_path", "")
        )
        stats_yaml_field = "normalization_stats_path"
        path_field = "checkpoint_path"

    try:
        checkpoint_src = _pick_checkpoint(source_repo, raw_model_path)
    except (ValueError, FileNotFoundError) as exc:
        return BuildResult(config_path, "skipped", None, str(exc))

    # ------------------------------------------------------------------
    # Resolve stats + arch (per-family logic).
    # ------------------------------------------------------------------
    if is_pypots:
        stats_yaml = (
            str(stats_path_override)
            if stats_path_override is not None
            else method_block.get(stats_yaml_field)
        )
        stats_src = _resolve_stats(source_repo, stats_yaml)
        arch = _extract_arch(method_block, kind)
    else:
        # LSM2: stats come from the checkpoint (extracted, written sibling).
        # Arch comes from the checkpoint's saved Lightning hparams.
        ckpt = _load_ckpt(checkpoint_src)
        if stats_path_override is not None:
            stats_src = _resolve_stats(source_repo, str(stats_path_override))
        else:
            extracted_stats = _extract_lsm2_stats_from_ckpt(ckpt)
            if extracted_stats is None:
                # Fall back to a YAML-pointed stats file if the checkpoint
                # doesn't embed them.
                stats_src = _resolve_stats(source_repo, method_block.get(stats_yaml_field))
            else:
                stats_src = None  # we'll write the extracted dict directly
        arch = _extract_lsm2_arch_from_ckpt(ckpt, kind)

    # ------------------------------------------------------------------
    # Provenance.
    # ------------------------------------------------------------------
    effective_block = dict(method_block)
    if model_path_override is not None:
        effective_block[path_field] = str(model_path_override)
    # _derive_provenance reads "model_path" — adapt for LSM2's "checkpoint_path".
    if is_lsm2 and "model_path" not in effective_block:
        effective_block["model_path"] = effective_block.get("checkpoint_path", "")
    provenance = _derive_provenance(config_path, effective_block, config.get("output"))
    if model_path_override is not None:
        provenance["model_path_override"] = str(model_path_override)
    if extra_provenance:
        provenance.update(extra_provenance)

    # ------------------------------------------------------------------
    # Stage the release directory.
    # ------------------------------------------------------------------
    if release_name is None:
        eval_run = config_path.parent.name
        release_name = eval_run.removeprefix("pypots_").removeprefix("mae_").split("_max")[0]
    release_dir = output_dir / release_name
    if release_dir.exists():
        if not overwrite:
            return BuildResult(
                config_path,
                "skipped",
                release_dir,
                f"release dir already exists (rerun with --overwrite to replace): {release_dir}",
            )
        shutil.rmtree(release_dir)
    release_dir.mkdir(parents=True)

    ckpt_basename = "model.ckpt" if is_lsm2 else "model.pypots"
    if copy_checkpoint:
        ckpt_dst = release_dir / ckpt_basename
        shutil.copy2(checkpoint_src, ckpt_dst)
        checkpoint_rel = ckpt_basename
    else:
        # Reference the original file by relative path from the release dir.
        checkpoint_rel = str(Path("..") / checkpoint_src.relative_to(source_repo))

    stats_rel: str | None = None
    if extracted_stats is not None:
        # LSM2 stats lifted out of the checkpoint — write directly.
        stats_dst = release_dir / "normalization_stats.json"
        stats_dst.write_text(json.dumps(extracted_stats, indent=2))
        stats_rel = "normalization_stats.json"
    elif stats_src is not None:
        stats_dst = release_dir / "normalization_stats.json"
        shutil.copy2(stats_src, stats_dst)
        stats_rel = "normalization_stats.json"

    write_manifest(
        release_dir,
        kind=kind,
        arch=arch,
        checkpoint=checkpoint_rel,
        normalization_stats=stats_rel,
        provenance=provenance,
    )
    return BuildResult(config_path, "built", release_dir)


def _cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--source-repo",
        type=Path,
        required=True,
        help="Path to the private MHC-benchmark repo root.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Parent directory that will receive one subdir per release.",
    )
    p.add_argument(
        "--config",
        type=Path,
        action="append",
        required=True,
        help="Path to a results/imputation_eval/<run>/config.yaml. "
        "Pass multiple times to stage several releases in one invocation.",
    )
    p.add_argument(
        "--release-name",
        type=str,
        default=None,
        help="Override the release subdir name (only valid with a single --config).",
    )
    p.add_argument(
        "--model-path-override",
        type=str,
        default=None,
        help="Use this checkpoint path instead of the one in the config "
        "(only valid with a single --config). Useful for resolving W&B "
        "refs after `wandb artifact get`, or pinning a specific epoch.",
    )
    p.add_argument(
        "--stats-path-override",
        type=str,
        default=None,
        help="Use this normalization_stats.json instead of the config's "
        "(only valid with a single --config).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace any existing release directory.",
    )
    p.add_argument(
        "--no-copy-checkpoint",
        action="store_true",
        help="Skip copying the .pypots file; reference the source by relative path.",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Stage release directories from the given eval configs.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv``).

    Returns:
        Process exit code (``0`` on success, non-zero on a usage error).
    """
    args = _cli().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.release_name and len(args.config) != 1:
        logger.error("--release-name only valid with a single --config")
        return 2
    if (args.model_path_override or args.stats_path_override) and len(args.config) != 1:
        logger.error(
            "--model-path-override / --stats-path-override only valid with a single --config"
        )
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    exit_code = 0
    for cfg in args.config:
        if not cfg.exists():
            logger.error("Config not found: %s", cfg)
            exit_code = 1
            continue
        result = build_release(
            cfg,
            source_repo=args.source_repo.resolve(),
            output_dir=args.output_dir.resolve(),
            release_name=args.release_name,
            overwrite=args.overwrite,
            copy_checkpoint=not args.no_copy_checkpoint,
            model_path_override=args.model_path_override,
            stats_path_override=args.stats_path_override,
        )
        if result.status == "built":
            logger.info("Built %s from %s", result.release_dir, result.config_path)
        else:
            logger.warning("Skipped %s: %s", result.config_path, result.reason or "unknown")
            exit_code = max(exit_code, 1)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
