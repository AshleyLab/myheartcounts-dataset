"""TASK_DISPLAY_NAMES must stay a complete, 1:1 presentation layer over BENCHMARK_TASKS.

The internal codes are the canonical keys (dataset survey identifiers); these are the
paper/leaderboard labels. If a task is ever added to ``BENCHMARK_TASKS`` without a
display name (or vice versa), this fails so the two never silently drift.
"""

from __future__ import annotations

import openmhc
from openmhc._constants import BENCHMARK_TASKS, TASK_DISPLAY_NAMES


def test_covers_exactly_the_benchmark_tasks():
    """Every benchmark task has a display name, and there are no orphans."""
    assert set(TASK_DISPLAY_NAMES) == set(BENCHMARK_TASKS)


def test_display_names_are_nonempty_and_unique():
    """Labels are non-empty and distinct (no two tasks render to the same name)."""
    values = list(TASK_DISPLAY_NAMES.values())
    assert all(isinstance(v, str) and v.strip() for v in values)
    assert len(set(values)) == len(values)


def test_exported_from_public_api():
    """Submitters can reach the map off the top-level package."""
    assert openmhc.TASK_DISPLAY_NAMES is TASK_DISPLAY_NAMES


def test_wellbeing_labels_match_survey_questions():
    """The ONS well-being ordering is the easy one to get wrong — pin it.

    Sourced from data/labels/survey_documentation/wellbeing/ (feel_worthwhile2 is the
    ONS "how about happy?" item, not the separate daily-slider ``happiness`` field).
    """
    assert TASK_DISPLAY_NAMES["feel_worthwhile1"] == "Things Are Worthwhile"
    assert TASK_DISPLAY_NAMES["feel_worthwhile2"] == "Feel Happy"
    assert TASK_DISPLAY_NAMES["feel_worthwhile3"] == "Feel Worried"
    assert TASK_DISPLAY_NAMES["feel_worthwhile4"] == "Feel Depressed"
    assert "happiness" not in TASK_DISPLAY_NAMES  # not a benchmark task
