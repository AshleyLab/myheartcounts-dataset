"""Tests for scripts/paper_results/imputation/run_paper_pipeline.py.

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

_SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts" / "paper_results" / "imputation"
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
    """Each per-method command pins hydra.run.dir and output.results_dir together.

    Both must point at the same per-method run directory.
    """
    runs_root = tmp_path / "runs"
    # Phase A: pairs are always written; no override needed. Pass a different
    # dummy override (seed) to verify common_overrides plumbing still works.
    cfg = {
        "runs_root": str(runs_root),
        "common_overrides": ["seed=123"],
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
        # common_overrides entries are forwarded verbatim.
        assert "seed=123" in cmd

    # Per-method dirs differ — no shared-pairs clobber.
    assert len(set(seen_run_dirs)) == len(seen_run_dirs)


def test_phase0_returned_pairs_dirs_match_overrides(tmp_path, captured_cmds):
    """The returned {method: pairs_dir} matches where the runner writes pairs.

    Given the new override, ``output.results_dir/pairs == runs_root/<method>/pairs``.
    This is the round-trip Phase 1's manifest builder depends on.
    """
    runs_root = tmp_path / "runs"
    cfg = {"runs_root": str(runs_root), "common_overrides": []}
    methods = [{"name": "mean"}, {"name": "locf"}]

    method_dirs = run_paper_pipeline._phase0_run_methods(cfg, methods, dry_run=True)

    for cmd, m in zip(captured_cmds, methods, strict=True):
        # Pull the output.results_dir override from the command.
        results_dir_override = next(tok for tok in cmd if tok.startswith("output.results_dir="))
        results_dir = Path(results_dir_override.split("=", 1)[1])
        # The pairs_dir returned to the caller must equal the runner's
        # ``config.output.results_dir / "pairs"``.
        assert method_dirs[m["name"]] == results_dir / "pairs"
        # And it must equal runs_root/<method>/pairs — Phase 1's expectation
        # (see run_paper_pipeline.py: --skip-eval branch and _write_manifest).
        assert method_dirs[m["name"]] == runs_root / m["name"] / "pairs"


# ---------------------------------------------------------------------------
# Phase 2 method_filter threading
# ---------------------------------------------------------------------------


def _phase2_cfg(method_filter: list[str] | None = None) -> dict:
    """Minimal cfg dict the Phase 2 helpers actually read."""
    cfg = {
        "draws_path": "/tmp/draws.parquet",
        "output_root": "/tmp/out",
        "baseline_method": "locf",
        "clip_lower": 0.01,
        "clip_upper": 100.0,
        "lambda_fairness": 0.5,
        "fairness_combine": "linear_penalty",
        "ci_level": 0.95,
        "disparity_fns": [],
    }
    if method_filter is not None:
        cfg["method_filter"] = method_filter
    return cfg


def test_phase2_aggregate_passes_method_filter(captured_cmds):
    """A configured ``method_filter`` is forwarded by _phase2_aggregate.

    It passes ``--method-filter`` plus the listed names, in order, to
    aggregate_imputation_paper_metrics.py.
    """
    cfg = _phase2_cfg(method_filter=["locf", "lsm2", "linear"])
    run_paper_pipeline._phase2_aggregate(cfg, dry_run=True)

    assert len(captured_cmds) == 1
    cmd = captured_cmds[0]
    # The flag and its three argument names are present in order.
    i = cmd.index("--method-filter")
    assert cmd[i + 1 : i + 4] == ["locf", "lsm2", "linear"]


def test_phase2_fairness_passes_method_filter(captured_cmds):
    """The fairness sidecar receives the same ``method_filter`` as the aggregator.

    Both aggregators must get the same filter so the two CSVs stay in lockstep.
    """
    cfg = _phase2_cfg(method_filter=["locf", "lsm2", "linear"])
    run_paper_pipeline._phase2_fairness_skill_score(cfg, dry_run=True)

    assert len(captured_cmds) == 1
    cmd = captured_cmds[0]
    i = cmd.index("--method-filter")
    assert cmd[i + 1 : i + 4] == ["locf", "lsm2", "linear"]


def test_phase2_no_method_filter_when_unset(captured_cmds):
    """An absent / null / empty ``method_filter`` makes neither helper emit the flag.

    Phase 2 then sees every method in the parquet (the aggregator's pre-existing
    default behaviour).
    """
    for cfg in (_phase2_cfg(), _phase2_cfg(method_filter=None), _phase2_cfg(method_filter=[])):
        captured_cmds.clear()
        run_paper_pipeline._phase2_aggregate(cfg, dry_run=True)
        run_paper_pipeline._phase2_fairness_skill_score(cfg, dry_run=True)
        for cmd in captured_cmds:
            assert "--method-filter" not in cmd


def test_phase2_raises_if_baseline_not_in_filter(captured_cmds):
    """A ``method_filter`` that excludes the baseline raises before any subprocess launch.

    The driver enforces that the baseline method stays in the comparison pool;
    without it, skill / fairness rows would be empty (the per-task paired ratio
    joins against the baseline). The direct CLI doesn't enforce this; the
    YAML-driven path does, because the cfg dict is in hand and can fail fast.
    """
    cfg = _phase2_cfg(method_filter=["lsm2", "linear"])  # no 'locf'!
    with pytest.raises(ValueError, match="excludes the baseline"):
        run_paper_pipeline._phase2_aggregate(cfg, dry_run=True)
    with pytest.raises(ValueError, match="excludes the baseline"):
        run_paper_pipeline._phase2_fairness_skill_score(cfg, dry_run=True)
    # Nothing was dispatched — the guardrail fires before _run is called.
    assert captured_cmds == []
