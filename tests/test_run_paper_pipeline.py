"""Tests for scripts/paper_results/run_paper_pipeline.py.

Pins the Phase 0 → Phase 1 contract: the per-method Hydra command must set
``output.results_dir`` to the same per-method ``run_dir`` it pins
``hydra.run.dir`` to, because the runner writes pairs to
``config.output.results_dir/pairs`` (see ``src/imputation_evaluation/runner.py``),
and Phase 1's manifest builder then looks for pairs at ``runs_root/<method>/pairs``.

Without the ``output.results_dir`` override, every method writes to the shared
default ``results/imputation_eval/pairs`` and clobbers each other, and Phase 1
finds no pairs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts" / "paper_results"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import run_paper_pipeline  # noqa: E402


@pytest.fixture
def captured_cmds(monkeypatch):
    """Patch ``_run`` so we can inspect the constructed commands without exec."""
    calls: list[list[str]] = []

    def _fake_run(cmd, dry_run):  # noqa: ARG001 - signature must match
        calls.append(list(cmd))

    monkeypatch.setattr(run_paper_pipeline, "_run", _fake_run)
    return calls


def test_phase0_sets_output_results_dir_per_method(tmp_path, captured_cmds):
    """Each per-method command must pin BOTH hydra.run.dir AND output.results_dir
    to the same per-method run directory."""
    runs_root = tmp_path / "runs"
    cfg = {
        "runs_root": str(runs_root),
        "common_overrides": ["evaluation.save_pairs=true"],
    }
    methods = [{"name": "mean"}, {"name": "locf"}]

    run_paper_pipeline._phase0_run_methods(cfg, methods, dry_run=True)

    assert len(captured_cmds) == 2
    seen_run_dirs = []
    for cmd, m in zip(captured_cmds, methods, strict=True):
        expected_dir = runs_root / m["name"]
        assert f"hydra.run.dir={expected_dir}" in cmd
        assert f"output.results_dir={expected_dir}" in cmd
        # Sanity: the two overrides reference the same dir (Phase 1 expectation).
        seen_run_dirs.append(expected_dir)
        # save_pairs override from common_overrides is still forwarded.
        assert "evaluation.save_pairs=true" in cmd

    # Per-method dirs differ — no shared-pairs clobber.
    assert len(set(seen_run_dirs)) == len(seen_run_dirs)


def test_phase0_returned_pairs_dirs_match_overrides(tmp_path, captured_cmds):
    """The {method: pairs_dir} returned by _phase0_run_methods must point at
    the same location the runner would write pairs to, given the new override:
    ``output.results_dir/pairs == runs_root/<method>/pairs``.

    This is the round-trip Phase 1's manifest builder depends on."""
    runs_root = tmp_path / "runs"
    cfg = {"runs_root": str(runs_root), "common_overrides": []}
    methods = [{"name": "mean"}, {"name": "locf"}]

    method_dirs = run_paper_pipeline._phase0_run_methods(cfg, methods, dry_run=True)

    for cmd, m in zip(captured_cmds, methods, strict=True):
        # Pull the output.results_dir override from the command.
        results_dir_override = next(
            tok for tok in cmd if tok.startswith("output.results_dir=")
        )
        results_dir = Path(results_dir_override.split("=", 1)[1])
        # The pairs_dir returned to the caller must equal the runner's
        # ``config.output.results_dir / "pairs"``.
        assert method_dirs[m["name"]] == results_dir / "pairs"
        # And it must equal runs_root/<method>/pairs — Phase 1's expectation
        # (see run_paper_pipeline.py: --skip-eval branch and _write_manifest).
        assert method_dirs[m["name"]] == runs_root / m["name"] / "pairs"
