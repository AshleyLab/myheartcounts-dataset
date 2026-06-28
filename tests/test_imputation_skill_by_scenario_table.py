"""Regression tests for the imputation skill-by-scenario table renderer."""

import math

import pytest
from scripts.paper_results.imputation import make_imputation_skill_by_scenario_table as table


def test_fmt_cell_rejects_nan_se():
    """NaN SEs must not render as zero-width uncertainty."""
    with pytest.raises(ValueError, match="Non-finite SE"):
        table.fmt_cell(
            "mean",
            center=0.2,
            se=math.nan,
            scale100=True,
            ref_zero=False,
            n=0,
            is_best=False,
        )


def test_build_body_rejects_missing_cells(monkeypatch):
    """Missing cells must fail instead of defaulting to ``nan +/- 0``."""
    monkeypatch.setattr(table, "METHODS", {"mean": (r"Mean", "single", "stat")})
    monkeypatch.setattr(
        table,
        "COLUMNS",
        [(r"$S\uparrow$", "skill", "overall", "mean", True, False, True)],
    )
    monkeypatch.setattr(table, "NCOL", 2)

    with pytest.raises(ValueError, match="Missing table cell"):
        table.build_body({"mean": {}})
