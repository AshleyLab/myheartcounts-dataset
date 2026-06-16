"""Tests for the cross-method bootstrap skill / rank aggregator.

Most tests target the pure-pandas phase-2 aggregator and the Parquet
round-trip. The phase-1 manifest-equality checks at the bottom of this
file synthesize the minimal pairs tree (just per-method manifests; no
``pairs_chXX.parquet`` is needed because the check fires before any
scenario directory is read).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data.processing.hf_config import N_CHANNELS
from imputation_evaluation.evaluation.bootstrap_skill_rank import (
    DRAWS_PARQUET_COLUMNS,
    _assert_manifests_agree,
    aggregate_skill_rank_fairness,
    compute_per_draw_errors,
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
    baseline_mu = float(base_error.get("locf", 1.0))
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for method in methods:
        for scenario in scenarios:
            for ch in channels:
                mu = base_error[method]
                noise = rng.normal(0, 0.05, size=n_boot)
                for b, eps in enumerate(noise):
                    e_val = float(max(1e-6, mu + eps))
                    # R is the paired ratio against the (idealised) baseline
                    # error ``base_error["locf"]``; locf-vs-self → R ≈ 1, so
                    # locf overall skill ≈ 0. ``compute_skill_scores`` consumes
                    # this directly under the R-mode path.
                    r_val = e_val / baseline_mu
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
                            "E": e_val,
                            "R": r_val,
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


# --------------------------------------------------------------------------
# Phase-1 manifest-equality checks
# --------------------------------------------------------------------------

# These tests verify ``_assert_manifests_agree`` directly and end-to-end via
# ``compute_per_draw_errors``. The validation fires before any pairs file is
# read, so the test fixtures only need to write ``manifest_<split>.parquet``
# per method dir — no ``pairs_chXX.parquet`` needed.

_REF_SAMPLES = [
    (0, "u0", "2024-01-01"),
    (1, "u1", "2024-01-02"),
    (2, "u2", "2024-01-03"),
    (3, "u3", "2024-01-04"),
]


def _write_manifest(
    pairs_dir,
    *,
    split: str,
    rows: list[tuple[int, str, str]],
    drop_column: str | None = None,
) -> None:
    """Write ``manifest_<split>.parquet`` under ``pairs_dir`` from raw rows."""
    pairs_dir.mkdir(parents=True, exist_ok=True)
    cols = {
        "sample_idx": pa.array([r[0] for r in rows], type=pa.int32()),
        "user_id": pa.array([r[1] for r in rows], type=pa.utf8()),
        "date": pa.array([r[2] for r in rows], type=pa.utf8()),
    }
    if drop_column is not None:
        cols.pop(drop_column)
    pq.write_table(pa.table(cols), pairs_dir / f"manifest_{split}.parquet")


def _build_two_method_dirs(
    tmp_path,
    *,
    mismatch_kind: str,
    split: str = "test",
):
    """Materialize two method dirs and return ``(method_manifests, method_dirs)``.

    ``mismatch_kind`` controls what goes wrong in method B's manifest:

    - ``"none"``       : identical manifests (happy path)
    - ``"swap"``       : same sample_idx set but (user_id, date) for sidx=2
                        and sidx=3 swapped
    - ``"missing"``    : method B drops sample_idx=3
    - ``"extra"``      : method B has an extra sample_idx=99
    - ``"dup"``        : method B duplicates sample_idx=2
    - ``"missing_col"``: method B's manifest lacks the ``user_id`` column
    """
    ref_rows = list(_REF_SAMPLES)

    if mismatch_kind == "none":
        b_rows = list(_REF_SAMPLES)
    elif mismatch_kind == "swap":
        b_rows = [
            (0, "u0", "2024-01-01"),
            (1, "u1", "2024-01-02"),
            (2, "u3", "2024-01-04"),  # was u2/2024-01-03
            (3, "u2", "2024-01-03"),  # was u3/2024-01-04
        ]
    elif mismatch_kind == "missing":
        b_rows = ref_rows[:-1]  # drops sample_idx=3
    elif mismatch_kind == "extra":
        b_rows = ref_rows + [(99, "u99", "2024-12-31")]
    elif mismatch_kind == "dup":
        b_rows = ref_rows + [(2, "u2", "2024-01-03")]  # duplicate sidx=2
    elif mismatch_kind == "missing_col":
        b_rows = list(_REF_SAMPLES)
    else:
        raise ValueError(f"unknown mismatch_kind: {mismatch_kind}")

    dir_a = tmp_path / "A"
    dir_b = tmp_path / "B"
    _write_manifest(dir_a, split=split, rows=ref_rows)
    _write_manifest(
        dir_b,
        split=split,
        rows=b_rows,
        drop_column="user_id" if mismatch_kind == "missing_col" else None,
    )

    method_manifests = {
        "A": pq.read_table(dir_a / f"manifest_{split}.parquet"),
        "B": pq.read_table(dir_b / f"manifest_{split}.parquet"),
    }
    method_dirs = {"A": dir_a, "B": dir_b}
    return method_manifests, method_dirs


def _phase1_kwargs(method_dirs, *, with_subgroups: bool):
    """Common kwargs for ``compute_per_draw_errors`` integration tests."""
    sg = None
    if with_subgroups:
        sg = {
            "test": {
                0: {"age_group": "a", "sex": "M"},
                1: {"age_group": "b", "sex": "F"},
                2: {"age_group": "a", "sex": "F"},
                3: {"age_group": "b", "sex": "M"},
            }
        }
    return dict(
        method_dirs=method_dirs,
        scenarios=["scenarioA"],
        splits=["test"],
        n_boot=2,
        seed=0,
        subgroup_mappings=sg,
        channel_stds=np.ones(N_CHANNELS, dtype=np.float64),
        include_auc=False,
    )


def test_assert_manifests_agree_accepts_identical_manifests(tmp_path):
    """Happy path: identical manifests should not raise."""
    manifests, _ = _build_two_method_dirs(tmp_path, mismatch_kind="none")
    # Returns None on success — just shouldn't raise.
    _assert_manifests_agree(manifests, split="test")


def test_phase1_raises_on_swapped_user_date(tmp_path):
    """sample_idx set matches but (user_id, date) is swapped → mismatch."""
    _, method_dirs = _build_two_method_dirs(tmp_path, mismatch_kind="swap")
    with pytest.raises(ValueError, match="Manifest mismatch"):
        compute_per_draw_errors(**_phase1_kwargs(method_dirs, with_subgroups=True))


def test_phase1_raises_on_missing_sample_idx(tmp_path):
    """Method B drops one sample_idx → error mentions 'missing'."""
    _, method_dirs = _build_two_method_dirs(tmp_path, mismatch_kind="missing")
    with pytest.raises(ValueError, match="missing sample_idx"):
        compute_per_draw_errors(**_phase1_kwargs(method_dirs, with_subgroups=True))


def test_phase1_raises_on_extra_sample_idx(tmp_path):
    """Method B has an extra sample_idx → error mentions 'extra'."""
    _, method_dirs = _build_two_method_dirs(tmp_path, mismatch_kind="extra")
    with pytest.raises(ValueError, match="extra sample_idx"):
        compute_per_draw_errors(**_phase1_kwargs(method_dirs, with_subgroups=True))


def test_phase1_raises_on_missing_column(tmp_path):
    """Method B's manifest lacks user_id → error mentions the column."""
    _, method_dirs = _build_two_method_dirs(tmp_path, mismatch_kind="missing_col")
    with pytest.raises(ValueError, match="missing required columns.*user_id"):
        compute_per_draw_errors(**_phase1_kwargs(method_dirs, with_subgroups=True))


def test_phase1_raises_on_duplicate_sample_idx(tmp_path):
    """Method B duplicates sample_idx=2 → error mentions duplicates."""
    _, method_dirs = _build_two_method_dirs(tmp_path, mismatch_kind="dup")
    with pytest.raises(ValueError, match="duplicate sample_idx"):
        compute_per_draw_errors(**_phase1_kwargs(method_dirs, with_subgroups=True))


def test_phase1_skips_check_without_subgroup_mappings(tmp_path):
    """No fairness → mismatched manifests should NOT raise.

    Guards the documented contract that non-fairness bootstrap behavior
    is unchanged. The function may still produce zero rows here because
    the synthetic tree has no pairs_chXX.parquet, but it must not raise
    a ValueError from the manifest check.
    """
    _, method_dirs = _build_two_method_dirs(tmp_path, mismatch_kind="swap")
    # subgroup_mappings=None → check should be skipped entirely.
    df = compute_per_draw_errors(**_phase1_kwargs(method_dirs, with_subgroups=False))
    # Empty is fine — the point is "did not raise". Schema must still match.
    assert list(df.columns) == DRAWS_PARQUET_COLUMNS
