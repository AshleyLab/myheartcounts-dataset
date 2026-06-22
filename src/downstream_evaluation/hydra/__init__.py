"""Hydra entry point for the ``mhc-downstream-eval`` console script.

Mirrors ``imputation_evaluation.hydra`` and ``forecasting_evaluation.hydra``:
the CLI composes a :class:`~downstream_evaluation.config.DownstreamEvalConfig`
from ``configs/downstream/*``, builds the bundled model via ``registry``, and
runs it through the public ``openmhc.evaluate_prediction`` API. Importing this
package requires the ``[hydra]`` optional extra.
"""
