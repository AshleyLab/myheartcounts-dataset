"""Declarative input specs — *what shape* of per-participant data a model receives.

A model declares one of these as a class attribute ``input``; the engine materializes
the cohort's data in that shape. The IC/TC cohort always comes from the lookup (see
:class:`~downstream_evaluation.data.provider.TaskDataProvider`); these specs only choose
the *materialization*, on two orthogonal axes:

  - **shape** — :class:`Daily` / :class:`Weekly` eligible segments, or a :class:`Window`
    of ``hours`` anchored to the label.
  - **resolution** — ``"hourly"`` (``daily_hourly_hf``, 24 bins/day) or ``"minute"``
    (``daily_hf``, 1440 bins/day).

``cohort`` is the lookup granularity used to pick *who* is in (daily/weekly); it is
inferred from the spec (overridable later if a window ever needs a weekly cohort).
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
class Daily(InputSpec):
    """Eligible daily segments — ``(n_days, T, 19)`` (T = 24 hourly / 1440 minute)."""

    resolution: str = "hourly"

    @property
    def cohort(self) -> str:
        return "daily"


@dataclass(frozen=True)
class Weekly(InputSpec):
    """Eligible weekly segments — ``(n_weeks, 168, 19)``."""

    resolution: str = "hourly"

    @property
    def cohort(self) -> str:
        return "weekly"


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
        return "daily"
