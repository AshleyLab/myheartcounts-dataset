"""Run-config loading tolerates stale/removed schema keys (post-hoc recompute)."""

from __future__ import annotations

import textwrap
from pathlib import Path

from forecasting_evaluation.metrics.offline.config_io import load_run_config


def test_load_run_config_drops_unknown_keys(tmp_path: Path):
    """A config written by an older schema (removed `seasonal_naive_average_history`
    field, plus an unknown output key) must still load instead of raising TypeError.
    """
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        textwrap.dedent(
            """
            seed: 42
            experiment_name: legacy_run
            forecasting:
              forecasting_length: 24
            model:
              type: seasonal_naive
              name: seasonal_naive
              seasonal_naive_average_history: {window: 7}
            output:
              results_dir: /tmp/whatever
              some_removed_flag: true
            """
        ).strip(),
        encoding="utf-8",
    )

    loaded = load_run_config(tmp_path)

    assert loaded.experiment_name == "legacy_run"
    assert loaded.forecasting.forecasting_length == 24
    assert loaded.model.type == "seasonal_naive"
    # The removed/unknown keys are dropped, not carried onto the dataclass.
    assert not hasattr(loaded.model, "seasonal_naive_average_history")
    assert loaded.output.results_dir == "/tmp/whatever"
