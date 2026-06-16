"""The ε-floor on 1−AUROC keeps perfect-baseline users in the paired skill set.

Without the floor, a user whose *baseline* AUROC is exactly 1.0 has baseline error
e^B = 1−AUROC = 0, so the paired ratio e^M/e^B divides by zero and the
``baseline > 0`` guard drops that user. With the floor (e = max(1−AUROC, ε)) the
user's baseline error is ε > 0, so the user is kept with a finite, clipped ratio.
The floor is applied symmetrically to model and baseline by ``metric_to_error``.
"""

from __future__ import annotations

import numpy as np

from forecasting_evaluation.metrics import metric_spec as spec
from forecasting_evaluation.metrics.skill_score_summary import compute_skill_from_errors

# Per-user AUROCs for one binary channel; user u3 is the perfect-baseline case.
MODEL_AUROC = [0.90, 0.80, 0.70, 0.90]
BASE_AUROC = [0.60, 0.70, 0.50, 1.00]  # u3: baseline perfect -> 1−AUROC = 0


def _errors(aurocs):
    return np.array([spec.metric_to_error("auroc", a) for a in aurocs], dtype=float)


def test_perfect_baseline_user_is_kept_with_floor():
    model_e = _errors(MODEL_AUROC)
    base_e = _errors(BASE_AUROC)

    # The perfect-baseline user's error is floored to ε (not 0), so baseline > 0 holds.
    assert base_e[3] == spec.BINARY_ERROR_FLOOR
    assert (base_e > 0).all()

    skill, gm_ratio, n_pairs = compute_skill_from_errors(model_e, base_e)

    # All 4 users contribute (the perfect-baseline user is no longer dropped).
    assert n_pairs == 4
    assert np.isfinite(skill) and np.isfinite(gm_ratio)


def test_floor_changes_result_vs_old_drop_guard():
    model_e = _errors(MODEL_AUROC)
    floored = _errors(BASE_AUROC)

    # Emulate the OLD behaviour: perfect baseline -> exact 0 error -> dropped.
    old_base = floored.copy()
    old_base[3] = 0.0

    _, _, n_floor = compute_skill_from_errors(model_e, floored)
    _, _, n_old = compute_skill_from_errors(model_e, old_base)

    # The floor admits exactly the one previously-dropped user.
    assert n_floor == n_old + 1 == 4
