"""Input materialization subsystem: declarative specs → per-participant arrays.

The engine, given a model's declared ``input`` spec, calls :func:`input_builder_for` to get
the matching :class:`~.base.InputBuilder`, then ``build(td)`` to fill ``td.inputs``. The IC/TC
cohort comes from the lookup (the provider) regardless; the builder only materializes data.

Two specs — independent of whether the model is an Encoder or a Predictor:
  - ``Raw(resolution)`` — the cohort's eligible raw days at hourly/minute resolution; the model
    windows / featurizes / encodes them itself (the universal escape hatch).
  - ``Window(hours, anchor)`` — one anchored window the framework builds (the TSFM convenience).
"""

from __future__ import annotations

from .base import InputBuilder, ParticipantData
from .raw import RawBuilder
from .spec import InputSpec, Raw, Window
from .window import WindowBuilder


def input_builder_for(spec: InputSpec, data_dir: str | None, temporal=None) -> InputBuilder:
    """Dispatch a declarative ``spec`` to its builder (open/closed: add a spec + builder here)."""
    if isinstance(spec, Raw):
        return RawBuilder(data_dir, spec)
    if isinstance(spec, Window):
        return WindowBuilder(data_dir, spec, temporal)
    raise ValueError(f"no input builder for spec {spec!r}")


__all__ = [
    "InputSpec",
    "Raw",
    "Window",
    "InputBuilder",
    "ParticipantData",
    "RawBuilder",
    "WindowBuilder",
    "input_builder_for",
]
