"""Tests for the cross-method bootstrap skill / rank aggregator.

Phase-1 (``compute_per_draw_errors``) needs a full pairs/ fixture which
this module does not yet provide. These tests target the pure-pandas
phase-2 aggregator and the Parquet round-trip.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from imputation_evaluation.evaluation.bootstrap_skill_rank import (
    DRAWS_PARQUET_COLUMNS,
    aggregate_skill_rank_fairness,
    read_draws_parquet,
    write_draws_parquet,
)


def _synthetic_draws(
    *,
    methods=("locf", "mean", "linear"),
    scenarios=("random_noise", "block_random"),
    channels=("ch_0", "ch_1"),
    n_boot: int = 20,
    base_error: dict[str, float] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a hand-crafted long-format draws DataFrame.

    Each (scenario, channel) cell gets per-method errors drawn around a
    method-specific mean so that the ranking is deterministic on average
    (locf=1.0, mean=2.0, linear=0.5 → linear < locf < mean → linear ranks 1,
    locf 2, mean 3).
    """
    if base_error is None:
        base_error = {"locf": 1.0, "mean": 2.0, "linear": 0.5}
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for method in methods:
        for scenario in scenarios:
            for ch in channels:
                mu = base_error[method]
                noise = rng.normal(0, 0.05, size=n_boot)
                for b, eps in enumerate(noise):
                    rows.append(
                        {
                            "method": method,
                            "scenario": scenario,
                            "split": "test",
                            "channel": ch,
                            "channel_type": "continuous",
                            "subgroup_attr": "all",
                            "subgroup_value": "all",
                            "draw": int(b),
                            "E": float(max(1e-6, mu + eps)),
                        }
                    )
    return pd.DataFrame(rows)


def test_skill_score_for_baseline_is_zero():
    """``locf`` against itself should have skill_score mean ≈ 0.

    Skill score formula in compute_skill_scores is ``1 - exp(mean(log(ratio)))``
    so ratio=1 → skill=0.
    """
    draws = _synthetic_draws()
    tables = aggregate_skill_rank_fairness(draws, baseline_method="locf")

    skill = tables["skill_scores"]
    locf_overall = skill[(skill["method"] == "locf") & (skill["scope"] == "overall")]
    assert not locf_overall.empty
    mean = float(locf_overall["mean"].iloc[0])
    assert abs(mean) < 5e-2, f"locf overall skill mean expected ≈ 0, got {mean}"


def test_skill_score_columns_match_documented_schema():
    """``skill_scores_bootstrap.csv`` schema is part of the public contract."""
    draws = _synthetic_draws()
    tables = aggregate_skill_rank_fairness(draws, baseline_method="locf")
    cols = set(tables["skill_scores"].columns)
    expected = {"method", "scope", "split", "n_tasks", "mean", "se", "ci_lo", "ci_hi", "n_boot"}
    assert expected.issubset(cols), f"missing columns: {expected - cols}"


def test_avg_rank_order_matches_synthetic_means():
    """Lowest-E method ranks 1, highest-E method ranks 3 (3 methods total)."""
    draws = _synthetic_draws()
    tables = aggregate_skill_rank_fairness(draws, baseline_method="locf")
    rank = tables["avg_rankings"]
    overall = rank[rank["scope"] == "overall"].set_index("method")["mean"].to_dict()
    assert overall["linear"] < overall["locf"] < overall["mean"]
    assert abs(overall["linear"] - 1.0) < 0.1
    assert abs(overall["locf"] - 2.0) < 0.1
    assert abs(overall["mean"] - 3.0) < 0.1


def test_write_then_read_parquet_round_trip(tmp_path):
    """Round-trip preserves rows and the meta sidecar JSON."""
    df = _synthetic_draws(n_boot=5)
    out = tmp_path / "draws.parquet"
    meta = {"n_boot": 5, "seed": 42, "note": "round-trip test"}
    write_draws_parquet(df, out, meta=meta)

    df_read, meta_read = read_draws_parquet(out)
    assert len(df_read) == len(df)
    assert list(df_read.columns) == DRAWS_PARQUET_COLUMNS
    assert meta_read == meta


def test_aggregate_returns_four_tables_even_when_empty():
    """Phase-2 must be defensive against an empty draws DataFrame."""
    df = pd.DataFrame(columns=DRAWS_PARQUET_COLUMNS)
    tables = aggregate_skill_rank_fairness(df)
    assert set(tables.keys()) == {
        "skill_scores",
        "avg_rankings",
        "fairness_subgroups",
        "fairness_summary",
    }
