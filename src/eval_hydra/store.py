"""Helpers for populating Hydra's ConfigStore from existing dataclass trees.

The three evaluation tracks each have a root ``@dataclass`` config containing
sub-dataclasses (data, masking/model, output, ...). Where the dataclasses are
clean of ``typing.Literal`` annotations we register them as structured-config
schemas; otherwise we skip silently (omegaconf <2.4 doesn't accept ``Literal``
field types). YAML composition + ``OmegaConf.to_object`` still produce a
correctly-typed dataclass either way; the only thing lost when skipping is
compile-time field-name/type validation.
"""

from __future__ import annotations

import dataclasses
import logging
import typing
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

logger = logging.getLogger(__name__)


def dict_to_dataclass(cls: type, data: Any) -> Any:
    """Recursively build an instance of ``cls`` from nested dicts.

    Works around the omegaconf<2.4 limitation that ``OmegaConf.to_object``
    requires a structured schema to be registered, which in turn rejects
    ``typing.Literal`` field annotations used throughout the eval configs.

    For each field on ``cls``: if the field's type is itself a dataclass and
    the value is a plain dict, recurse; otherwise pass the value through.
    Unknown keys are silently dropped (Hydra may add extras like
    ``_target_`` for launcher configs).
    """
    if isinstance(data, DictConfig):
        data = OmegaConf.to_container(data, resolve=True)
    if not dataclasses.is_dataclass(cls):
        return data
    if not isinstance(data, dict):
        return data

    # Resolve PEP 563 string annotations to actual types. ``get_type_hints``
    # walks the module globals, so nested dataclasses defined in the same
    # module resolve correctly.
    try:
        hints = typing.get_type_hints(cls)
    except Exception:
        hints = {f.name: f.type for f in dataclasses.fields(cls)}

    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        if name not in hints:
            continue
        target_type = hints[name]
        if dataclasses.is_dataclass(target_type) and isinstance(value, dict):
            kwargs[name] = dict_to_dataclass(target_type, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _try_store(cs: ConfigStore, *, name: str, node: type, group: str | None = None) -> bool:
    """Attempt to register ``node`` and report whether it succeeded.

    Returns ``True`` on success, ``False`` if the dataclass uses an annotation
    OmegaConf can't structure (typically ``typing.Literal`` on this codebase).
    """
    try:
        OmegaConf.structured(node)  # probe; raises before we mutate the store
    except (OmegaConfBaseException, ValueError, TypeError) as exc:
        logger.debug("Skipping ConfigStore registration of %s: %s", node.__name__, exc)
        return False
    cs.store(name=name, node=node, group=group)
    return True


def register_dataclass_tree(
    cs: ConfigStore,
    root_cls: type,
    root_name: str,
    group_map: dict[str, type] | None = None,
) -> None:
    """Register ``root_cls`` and any sub-dataclasses with the ConfigStore.

    Args:
        cs: The ``ConfigStore`` instance (typically ``ConfigStore.instance()``).
        root_cls: The root ``@dataclass`` config (e.g. ``ImputationEvalConfig``).
        root_name: The schema name to register the root under. By convention this
            matches the root entry in each track's ``eval.yaml`` defaults list
            (e.g. ``"imputation_eval_schema"``).
        group_map: Optional explicit mapping of ``{group_name: dataclass}``.
            When provided, each entry is registered under the group with the
            schema name ``"<group>_schema"``. When omitted, sub-dataclasses are
            discovered by introspection: each field on ``root_cls`` whose type
            is itself a dataclass becomes a group keyed by the field name.

    The schema-suffix convention exists so user-authored group YAMLs (e.g.
    ``method/brits.yaml``) don't collide with the schema entry the dataclass
    is registered under.
    """
    _try_store(cs, name=root_name, node=root_cls)

    if group_map is None:
        group_map = {
            f.name: f.type if dataclasses.is_dataclass(f.type) else _unwrap(f)
            for f in dataclasses.fields(root_cls)
            if _is_dataclass_field(f)
        }

    for group_name, cls in group_map.items():
        if cls is None or not dataclasses.is_dataclass(cls):
            continue
        _try_store(cs, name=f"{group_name}_schema", node=cls, group=group_name)


def _is_dataclass_field(field: dataclasses.Field[Any]) -> bool:
    if dataclasses.is_dataclass(field.type):
        return True
    return dataclasses.is_dataclass(_unwrap(field))


def _unwrap(field: dataclasses.Field[Any]) -> type | None:
    """Return the resolved dataclass type for ``field``, or None if non-dataclass.

    ``dataclasses.fields(cls).type`` may be a string (PEP 563 forward ref) when
    ``from __future__ import annotations`` is in effect. Resolve via the
    default factory when present, falling back to ``None``.
    """
    factory = field.default_factory  # type: ignore[misc]
    if factory is dataclasses.MISSING:
        return None
    try:
        instance = factory()
    except Exception:
        return None
    return type(instance) if dataclasses.is_dataclass(instance) else None
