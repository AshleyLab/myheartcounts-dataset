"""Small ``MethodRegistry`` helper used by the per-track Hydra CLIs.

Each track has its own dispatch shape (the imputation registry calls
``Cls.from_release`` for paper checkpoints; the forecasting and downstream
registries wrap existing ``create_*`` factories). This helper just enforces a
consistent ``build(cfg) -> (method, manifest_or_none)`` signature so the shared
artifact writer can record the loaded manifest when one is available.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openmhc.imputers._release import Manifest


Builder = Callable[..., tuple[Any, "Manifest | None"]]


@dataclass
class MethodRegistry:
    """A typed wrapper around ``dict[type_string, builder_callable]``.

    Each builder takes whatever positional/keyword args the track needs
    (typically the method or model sub-config plus a track-level data config)
    and returns ``(method_or_model_instance, manifest_or_none)``.

    The ``manifest`` slot is non-``None`` only when the builder loaded a
    checkpoint via :class:`openmhc.imputers._release.ReleaseLoadableMixin`.
    """

    name: str
    builders: dict[str, Builder]

    def build(self, type_key: str, *args: Any, **kwargs: Any) -> tuple[Any, "Manifest | None"]:
        if type_key not in self.builders:
            known = ", ".join(sorted(self.builders.keys()))
            raise KeyError(
                f"Unknown {self.name} type {type_key!r}. Known: {known}"
            )
        return self.builders[type_key](*args, **kwargs)
