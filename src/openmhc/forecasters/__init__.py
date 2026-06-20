"""Reference forecasting models that satisfy the public ``Forecaster`` protocol.

Each class loads a released checkpoint bundle (local directory or
``hf://`` URI) via :meth:`from_release` and implements ``predict(history,
horizon)``, delegating to the corresponding internal evaluation engine.

Example:
    >>> import openmhc
    >>> from openmhc.forecasters import Chronos2Forecaster
    >>> fc = Chronos2Forecaster.from_release("hf://MyHeartCounts/openmhc-chronos2-fc")
    >>> results = openmhc.evaluate_forecasting(fc, version="full")

Available reference models (all fine-tuned/trained on the MHC training split):

- :class:`DLinearForecaster`, :class:`SegRNNForecaster`,
  :class:`MixLinearForecaster` — from-scratch neural models. Requires
  ``pip install openmhc[pypots]``.
- :class:`Chronos2Forecaster` — fine-tuned Amazon Chronos-2. Requires
  ``pip install openmhc[chronos]``.
- :class:`TotoForecaster` — fine-tuned Datadog Toto. Requires
  ``pip install openmhc[toto]``.

Released bundles live under ``MyHeartCounts/openmhc-<model>-fc`` on the
Hugging Face Hub.
"""

from openmhc.forecasters._release import Manifest, load_manifest, write_manifest
from openmhc.forecasters.chronos2 import Chronos2Forecaster
from openmhc.forecasters.neural import (
    DLinearForecaster,
    MixLinearForecaster,
    SegRNNForecaster,
)
from openmhc.forecasters.toto import TotoForecaster

__all__ = [
    "Chronos2Forecaster",
    "TotoForecaster",
    "DLinearForecaster",
    "SegRNNForecaster",
    "MixLinearForecaster",
    "Manifest",
    "load_manifest",
    "write_manifest",
]
