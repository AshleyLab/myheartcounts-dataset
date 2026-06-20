"""Hydra-driven CLI for the imputation evaluation track.

Entry point: ``mhc-impute-eval`` (declared in ``pyproject.toml``).

The CLI reads YAML configs from ``configs/imputation/`` (resolved relative to
the repo root), validates them against :mod:`imputation_evaluation.config`'s
dataclass schema, dispatches to a method registry, and forwards to
:func:`imputation_evaluation.runner.run_eval`. Public-API users
(:func:`openmhc.evaluate_imputation`) are untouched and never need Hydra.
"""

from __future__ import annotations


def _lazy_main():
    from imputation_evaluation.hydra.cli import main as _main

    return _main


def main(*args, **kwargs):
    """Console-script wrapper that imports lazily.

    This indirection lets ``import imputation_evaluation.hydra`` succeed even
    when Hydra/OmegaConf aren't installed (the friendly error from
    :mod:`eval_hydra` only triggers when the CLI is actually invoked).
    """
    return _lazy_main()(*args, **kwargs)


__all__ = ["main"]
