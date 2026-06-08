"""Declarative input specs — *what shape* of per-participant data a model receives.

A model declares one of these as a class attribute ``input``; the engine materializes
the cohort's data in that shape. The IC/TC cohort always comes from the lookup (see
:class:`~downstream_evaluation.data.provider.TaskDataProvider`); these specs only choose
the *materialization*, on two orthogonal axes:

  - :class:`Raw` — the cohort's eligible raw days at a resolution; the model shapes them.
  - :class:`Window` — one anchored ``hours``-long window the framework builds for you.

Resolution is ``"hourly"`` (``daily_hourly_hf``, 24 bins/day) or ``"minute"``
(``daily_hf``, 1440 bins/day). ``cohort`` is the lookup granularity for *who* is in (daily).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InputSpec:
    """Base class for input specs."""

    @property
    def cohort(self) -> str:
        """Lookup granularity for the cohort (``"daily"`` / ``"weekly"``)."""
        return "daily"


@dataclass(frozen=True)
class Raw(InputSpec):
    """Cohort user's IC/TC-bounded raw days at the chosen resolution.

    The model windows / featurizes / encodes it itself (the universal escape hatch).

    Delivered as ``(n_eligible_days, T, 19)`` per participant, ``T`` = 24 (hourly) /
    1440 (minute). The framework still applies the cohort (IC) and the in-window days
    (TC) from the lookup; only the *shape* is the model's job — so a minute feature-builder,
    a custom-window TSFM, anything, is covered without a bespoke materializer.
    """

    resolution: str = "hourly"

    @property
    def cohort(self) -> str:
        """Return the lookup granularity for the cohort (``"daily"``)."""
        return "daily"


@dataclass(frozen=True)
class Window(InputSpec):
    """One anchored window per participant — ``(1, hours, 19)``.

    ``anchor`` is the window's endpoint:
      - ``"window_end"`` — ends at ``label_date + forward_window`` (reproduces the
        Toto/Chronos-2 baseline; consistent with how the cohort region is defined).
      - ``"label"`` — ends at the label measurement date.
    (``"last_observed"`` is reserved for a future mode.)
    """

    hours: int
    anchor: str = "window_end"
    resolution: str = "hourly"

    @property
    def cohort(self) -> str:
        """Return the lookup granularity for the cohort (``"daily"``)."""
        return "daily"
