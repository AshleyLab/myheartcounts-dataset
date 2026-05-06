"""Stub for the PyPOTS deep-forecasting base class.

The full implementation lives in the private repo and pulls in pypots /
pygrinder / torch-Lightning. The public OpenMHC API only runs naive and
statistical forecasters (no PyPOTS dependency), so the only thing the
evaluator actually needs from this module is a class symbol to use in
``isinstance(model, BasePyPOTSForecastingModel)`` checks — those checks
return False for naive forecasters and the PyPOTS-specific branches are
skipped.

If you want to evaluate a real PyPOTS-based forecaster against this
benchmark, install the private package (``pip install mhc-benchmark``)
and use that pipeline.
"""

from __future__ import annotations


class BasePyPOTSForecastingModel:
    """No-op stub. See module docstring."""

    uses_standard_scaler = False
    scaler_stats = None
