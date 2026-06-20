"""Hydra-driven CLI for the forecasting evaluation track.

Entry point: ``mhc-forecast-eval`` (declared in ``pyproject.toml``).
"""

from __future__ import annotations


def _lazy_main():
    from forecasting_evaluation.hydra.cli import main as _main

    return _main


def main(*args, **kwargs):
    """Lazily import and invoke the Hydra CLI entry point."""
    return _lazy_main()(*args, **kwargs)


__all__ = ["main"]
