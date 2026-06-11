"""Pluggable disparity and fairness-combine functions.

.. deprecated::
    This registry backs the legacy ``S − λ·D`` fairness-adjusted skill
    score (Family B) consumed by
    :func:`imputation_evaluation.evaluation.bootstrap_skill_rank.aggregate_skill_rank_fairness`.
    The leaderboard now uses the disparity-ratio "Fairness Skill Score"
    in :mod:`scripts.paper_results.aggregate_fairness_skill_score`, which
    does **not** go through this module. Kept callable for back-compat
    only; do not add new disparities here.

Used by the imputation paper-metrics pipeline. Two registries:

* ``DISPARITY_FUNCTIONS`` — maps a name to a callable that takes a
  ``{subgroup_value: skill_score}`` dict and returns a scalar disparity.
* ``FAIRNESS_COMBINE`` — maps a name to a callable that takes
  ``(S_overall, disparity, lambda_)`` and returns a fairness-adjusted score.

New disparities can be added with a single ``register_disparity`` call (or
by editing this module). The bootstrap and the point-estimate flow both
look up callables by name, so the CLI surface stays the same regardless
of which disparity is selected.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

DisparityFn = Callable[[dict[str, float]], float]
FairnessCombineFn = Callable[[float, float, float], float]


@dataclass(frozen=True)
class DisparitySpec:
    """A registered disparity function plus metadata for downstream consumers."""

    fn: DisparityFn
    higher_is_better: bool   # does a higher value mean *fairer*?
    description: str


DISPARITY_FUNCTIONS: dict[str, DisparitySpec] = {}
FAIRNESS_COMBINE: dict[str, FairnessCombineFn] = {}


def register_disparity(
    name: str,
    fn: DisparityFn,
    *,
    higher_is_better: bool,
    description: str = "",
) -> None:
    """Register (or overwrite) a named disparity function."""
    DISPARITY_FUNCTIONS[name] = DisparitySpec(
        fn=fn, higher_is_better=higher_is_better, description=description,
    )


def register_fairness_combine(name: str, fn: FairnessCombineFn) -> None:
    """Register (or overwrite) a named fairness-combine function."""
    FAIRNESS_COMBINE[name] = fn


def disparity_higher_is_better(name: str) -> bool:
    """Return whether higher values of disparity ``name`` indicate *fairer* outcomes.

    Falls back to ``False`` for unregistered names so legacy callers passing
    raw callables (without a :class:`DisparitySpec`) keep the previous
    ``S − λ·D`` convention.
    """
    spec = DISPARITY_FUNCTIONS.get(name)
    return bool(spec.higher_is_better) if spec is not None else False


def _max_minus_min(g: dict[str, float]) -> float:
    vals = [v for v in g.values() if v is not None and np.isfinite(v)]
    if len(vals) < 2:
        return float("nan")
    return float(max(vals) - min(vals))


def _worst_group(g: dict[str, float]) -> float:
    vals = [v for v in g.values() if v is not None and np.isfinite(v)]
    if not vals:
        return float("nan")
    return float(min(vals))


def _std_across(g: dict[str, float]) -> float:
    vals = [v for v in g.values() if v is not None and np.isfinite(v)]
    if len(vals) < 2:
        return float("nan")
    return float(np.std(vals, ddof=0))


def _relative_drop(g: dict[str, float]) -> float:
    """``(max − min) / max`` — disparity as a fraction of the best subgroup."""
    vals = [v for v in g.values() if v is not None and np.isfinite(v)]
    if len(vals) < 2:
        return float("nan")
    hi = max(vals)
    if hi == 0:
        return float("nan")
    return float((hi - min(vals)) / hi)


register_disparity(
    "max_minus_min", _max_minus_min,
    higher_is_better=False,
    description="max(S_g) − min(S_g) across subgroups (lower is fairer).",
)
register_disparity(
    "worst_group", _worst_group,
    higher_is_better=True,
    description="min(S_g) — worst-subgroup skill score (higher is fairer).",
)
register_disparity(
    "std", _std_across,
    higher_is_better=False,
    description="Population std-dev of S_g across subgroups (lower is fairer).",
)
register_disparity(
    "relative_drop", _relative_drop,
    higher_is_better=False,
    description="(max − min) / max — disparity normalised by the best subgroup.",
)


def _linear_penalty(s: float, d: float, lam: float) -> float:
    """``S − λ·D`` — the original Fair_S in the imputation paper."""
    return float(s - lam * d)


register_fairness_combine("linear_penalty", _linear_penalty)
