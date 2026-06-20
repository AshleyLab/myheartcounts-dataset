"""Convert dataclass or DictConfig objects into wandb-friendly dicts.

W&B's ``wandb.init(config=...)`` accepts plain dicts. Today the imputation
W&B logger uses ``dataclasses.asdict(config)``, which fails on ``DictConfig``
values produced by Hydra. This shim accepts either input and always returns a
plain dict — small enough to drop in everywhere a config-snapshot is logged.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from omegaconf import DictConfig, OmegaConf


def to_wandb_config(cfg: Any) -> dict[str, Any]:
    """Return a plain dict snapshot of ``cfg`` suitable for ``wandb.init``.

    Accepts:
    - An ``omegaconf.DictConfig`` (from Hydra) — resolved via ``OmegaConf.to_container``.
    - A standard ``@dataclass`` instance — converted via ``dataclasses.asdict``.
    - A plain dict — passed through unchanged.

    Any other type is wrapped under ``{"value": cfg}`` so wandb still accepts it.
    """
    if isinstance(cfg, DictConfig):
        out = OmegaConf.to_container(cfg, resolve=True)
        assert isinstance(out, dict)
        return out
    if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
        return dataclasses.asdict(cfg)
    if isinstance(cfg, dict):
        return cfg
    return {"value": cfg}
