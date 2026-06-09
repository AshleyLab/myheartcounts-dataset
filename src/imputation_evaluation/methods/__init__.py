"""Imputation method protocol module.

Intentionally empty at module load to preserve the imputation_evaluation
lazy-import pattern: heavy method implementations live elsewhere and are
imported on demand. ``base.py`` defines the duck-typed ``ImputationMethod``
Protocol referenced under ``TYPE_CHECKING`` in ``runner.py`` and
``evaluation/evaluator.py``; it is the type-only contract the
``_ImputerMethodAdapter`` in ``openmhc._evaluate`` satisfies.
"""
