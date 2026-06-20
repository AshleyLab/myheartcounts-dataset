"""Shared Hydra plumbing for the three OpenMHC evaluation tracks.

This package contains only cross-cutting infrastructure that is identical
across the imputation, forecasting, and downstream prediction tracks:

- ``store``: helpers to register dataclass trees with Hydra's ``ConfigStore``.
- ``artifacts``: per-run-dir artifact writer (resolved config, manifest, wandb id).
- ``wandb_shim``: convert dataclass or DictConfig into a wandb-friendly dict.
- ``registry``: small ``MethodRegistry`` helper for track-specific factories.
- ``launchers/``: shared launcher YAMLs (Submitit/SLURM).

The package depends on the ``[hydra]`` optional extra. Importing it without
hydra/omegaconf installed raises a friendly error on first use.
"""

from __future__ import annotations


def _require_hydra() -> None:
    try:
        import hydra  # noqa: F401
        import omegaconf  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "eval_hydra requires the [hydra] optional extra. "
            "Install with: pip install 'openmhc[hydra]'"
        ) from e


_require_hydra()

from eval_hydra.artifacts import write_run_artifacts  # noqa: E402
from eval_hydra.registry import MethodRegistry  # noqa: E402
from eval_hydra.store import dict_to_dataclass, register_dataclass_tree  # noqa: E402
from eval_hydra.wandb_shim import to_wandb_config  # noqa: E402

__all__ = [
    "MethodRegistry",
    "dict_to_dataclass",
    "register_dataclass_tree",
    "to_wandb_config",
    "write_run_artifacts",
]
