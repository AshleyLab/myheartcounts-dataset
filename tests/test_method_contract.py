"""The public prediction contract: the ``Method`` boundary guard + the ``CohortStream`` type.

Two API-surface guarantees a submitter relies on, both fast + data-free:

1. ``evaluate_prediction`` validates the model against :class:`openmhc.Method` at the
   front door — a model missing ``predict`` fails immediately with a clear ``TypeError``,
   not deep inside the engine after the dataset is loaded. The guard runs before any path
   resolution, so this test needs no dataset.
2. The cohort handed to ``fit`` / ``predict`` is described by the public
   :class:`openmhc.CohortStream` protocol — the engine's concrete streaming class is
   internal and only ever *satisfies* it structurally, so no internal type is exported.
   (The old leaky ``openmhc.CohortView`` symbol must be gone.)
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

import openmhc


class _HasPredict:
    """Minimal valid Method — predict is the only required method."""

    def predict(self, data):
        return np.zeros(len(list(data)))


class _NoPredict:
    """Defines fit but not predict — does NOT satisfy Method."""

    def fit(self, data, labels, task_type):
        pass


class _StreamLike:
    """Has the CohortStream surface (iterate / len / load) without being a list."""

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0

    def load(self, user_id):
        return np.zeros((1, 24, 38))


class TestMethodBoundaryGuard:
    """evaluate_prediction rejects a non-Method at the front door, data-free."""

    def test_missing_predict_raises_typeerror(self):
        with pytest.raises(TypeError, match="does not satisfy openmhc.Method"):
            openmhc.evaluate_prediction(_NoPredict())

    def test_error_names_predict(self):
        with pytest.raises(TypeError, match="predict"):
            openmhc.evaluate_prediction(_NoPredict())

    def test_has_predict_satisfies_method(self):
        assert isinstance(_HasPredict(), openmhc.Method)

    def test_no_predict_does_not_satisfy_method(self):
        assert not isinstance(_NoPredict(), openmhc.Method)


class TestCohortStreamType:
    """CohortStream is the public description of the cohort handed to fit/predict."""

    def test_cohortstream_is_public(self):
        assert "CohortStream" in openmhc.__all__
        assert openmhc.CohortStream is not None

    def test_internal_cohortview_is_not_exported(self):
        # The leaky concrete engine symbol must not be reachable from the public package.
        assert "CohortView" not in openmhc.__all__
        assert not hasattr(openmhc, "CohortView")

    def test_plain_list_is_not_a_cohortstream(self):
        # A list is a valid `data` (small specs) but is NOT a CohortStream — the two
        # forms differ; submitters must program against iteration, not a concrete type.
        assert not isinstance([1, 2, 3], openmhc.CohortStream)

    def test_streaming_surface_satisfies_cohortstream(self):
        assert isinstance(_StreamLike(), openmhc.CohortStream)

    def test_engine_cohortview_satisfies_cohortstream(self):
        # The concrete internal class structurally satisfies the public protocol.
        from downstream_evaluation.data.cohort import CohortView

        assert issubclass(CohortView, openmhc.CohortStream)


class TestImportIsLight:
    """``import openmhc`` must not drag in the evaluation engine (the public/internal
    boundary). Runs in a clean subprocess so the in-session engine import by other
    tests can't mask a regression. The engine should load only when a probe / an
    ``evaluate_*`` call actually needs it."""

    def test_import_openmhc_does_not_load_engine(self):
        script = (
            "import sys, openmhc\n"
            "cs = openmhc.CohortStream\n"  # touching public types must also stay light
            "leaked = sorted(m for m in sys.modules if 'downstream_evaluation' in m)\n"
            "assert not leaked, leaked\n"
            "print('OK')\n"
        )
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout
