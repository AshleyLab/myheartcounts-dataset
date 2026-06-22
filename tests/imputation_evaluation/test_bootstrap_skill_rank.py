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
    compute_per_draw_errors_from_per_user_errors,
    read_draws_parquet,
    write_draws_parquet,
)
from imputation_evaluation.evaluation.paper_metrics_core import (
    aggregate_task_ranks_to_scopes,
    compute_skill_scores,
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
    # Per-task ranks the new Phase 1 would emit: methods ordered by base E,
    # lowest → rank 1. The synthetic fixture's per-method base E is large
    # relative to the noise, so the per-task rank is the same for every
    # draw (no variance across draws). Real Phase 1 emits the per-draw
    # nanmean over resampled-user ranks — but for an order test that
    # constant per-method value is the right contract.
    sorted_methods = sorted(base_error, key=lambda m: base_error[m])
    rank_lookup = {m: float(i + 1) for i, m in enumerate(sorted_methods)}
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
                            "rank": rank_lookup[method],
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
# task:<scenario>:<channel> — leaf scope emission (skill + rank)
# --------------------------------------------------------------------------
#
# These tests target the kernel functions directly with hand-built inputs so
# the assertions stay local to the new emission code. The synthetic-draws +
# Phase-2 pass-through test below exercises the same scopes through the
# bootstrap.


def _build_paired_R_frame(rows):
    """Build a long-format paired-R frame for compute_skill_scores."""
    return pd.DataFrame(rows)


class TestTaskGrainSkillEmission:
    """The per-(method, scenario, channel) leaf scope ``task:<sc>:<ch>``."""

    def test_task_scope_present_for_every_input_row(self):
        """One ``task:<sc>:<ch>`` row per (method, scenario, channel) input."""
        rows = [
            {"method": "A", "scenario": "random_noise", "channel": "ch_0", "R": 0.4},
            {"method": "A", "scenario": "random_noise", "channel": "ch_1", "R": 0.6},
            {"method": "A", "scenario": "temporal_slice", "channel": "ch_0", "R": 0.8},
            {"method": "B", "scenario": "random_noise", "channel": "ch_0", "R": 1.2},
            {"method": "B", "scenario": "random_noise", "channel": "ch_1", "R": 0.9},
            {"method": "B", "scenario": "temporal_slice", "channel": "ch_0", "R": 1.0},
        ]
        result = compute_skill_scores(_build_paired_R_frame(rows), mode="paired")
        task_scopes = result[result["scope"].str.startswith("task:")]
        # 2 methods × 3 unique (scenario, channel) keys = 6 task rows.
        assert len(task_scopes) == 6
        for _, r in task_scopes.iterrows():
            assert int(r["n_tasks"]) == 1

    def test_task_skill_equals_one_minus_clipped_ratio(self):
        """task:* skill is the degenerate single-task case: ``1 − clip(R)``."""
        rows = [
            {"method": "A", "scenario": "random_noise", "channel": "ch_0", "R": 0.4},
            {"method": "A", "scenario": "random_noise", "channel": "ch_1", "R": 50.0},
            # Out-of-clip extremes: R=0.001 should clip to 0.01, R=500 to 100.
            {"method": "A", "scenario": "random_noise", "channel": "ch_2", "R": 0.001},
            {"method": "A", "scenario": "random_noise", "channel": "ch_3", "R": 500.0},
        ]
        result = compute_skill_scores(_build_paired_R_frame(rows), mode="paired")
        skill_by_ch = (
            result[result["scope"].str.startswith("task:")]
            .set_index("scope")["skill_score"]
            .to_dict()
        )
        # ch_0: R=0.4 inside clip bounds → skill = 1 − 0.4 = 0.6.
        np.testing.assert_allclose(skill_by_ch["task:random_noise:ch_0"], 0.6, rtol=1e-9)
        # ch_1: R=50 inside clip → skill = 1 − 50 = −49.
        np.testing.assert_allclose(skill_by_ch["task:random_noise:ch_1"], -49.0, rtol=1e-9)
        # ch_2: R=0.001 clipped up to 0.01 → skill = 1 − 0.01 = 0.99.
        np.testing.assert_allclose(skill_by_ch["task:random_noise:ch_2"], 0.99, rtol=1e-9)
        # ch_3: R=500 clipped down to 100 → skill = 1 − 100 = −99.
        np.testing.assert_allclose(skill_by_ch["task:random_noise:ch_3"], -99.0, rtol=1e-9)

    def test_scenario_scope_is_geomean_of_task_ratios(self):
        """Per-scenario scope skill reconstructs from its constituent task scopes.

        ::

            S_scenario = 1 − exp(mean(log(1 − S_task)))
                       = 1 − geomean(clipped_R)

        Locks in the consistency property between the new leaf scope and the
        existing aggregated scope.
        """
        rows = [
            {"method": "A", "scenario": "random_noise", "channel": "ch_0", "R": 0.5},
            {"method": "A", "scenario": "random_noise", "channel": "ch_1", "R": 0.8},
            {"method": "A", "scenario": "random_noise", "channel": "ch_2", "R": 1.5},
        ]
        result = compute_skill_scores(_build_paired_R_frame(rows), mode="paired")
        task_skills = result[result["scope"].str.startswith("task:")]["skill_score"].values
        scenario_skill = float(result[result["scope"] == "random_noise"]["skill_score"].iloc[0])
        expected = 1.0 - float(np.exp(np.mean(np.log(1.0 - task_skills))))
        np.testing.assert_allclose(scenario_skill, expected, rtol=1e-9)


class TestTaskGrainRankEmission:
    """``task:<sc>:<ch>`` rows on the rank side mirror the skill side."""

    def test_task_rank_passed_through_unchanged(self):
        """avg_rank for each ``task:*`` row equals the input ``task_rank``."""
        per_task = pd.DataFrame(
            [
                {
                    "method": "A",
                    "scenario": "random_noise",
                    "channel": "ch_0",
                    "task_rank": 1.5,
                    "n_users": 100,
                },
                {
                    "method": "A",
                    "scenario": "random_noise",
                    "channel": "ch_1",
                    "task_rank": 2.0,
                    "n_users": 95,
                },
                {
                    "method": "B",
                    "scenario": "temporal_slice",
                    "channel": "ch_5",
                    "task_rank": 1.0,
                    "n_users": 80,
                },
            ]
        )
        result = aggregate_task_ranks_to_scopes(per_task)
        task_rows = result[result["scope"].str.startswith("task:")]
        by_scope = task_rows.set_index("scope")
        assert by_scope.loc["task:random_noise:ch_0", "avg_rank"] == pytest.approx(1.5)
        assert by_scope.loc["task:random_noise:ch_1", "avg_rank"] == pytest.approx(2.0)
        assert by_scope.loc["task:temporal_slice:ch_5", "avg_rank"] == pytest.approx(1.0)
        # n_users column carries through when present in input.
        assert int(by_scope.loc["task:random_noise:ch_0", "n_users"]) == 100
        # All task rows have n_tasks=1.
        assert (task_rows["n_tasks"] == 1).all()

    def test_scenario_rank_is_mean_of_task_ranks(self):
        """The ``<scenario>`` scope mean equals the mean of its ``task:*`` rows.

        The Stage-2 ``<scenario>`` scope mean equals the mean of its
        ``task:*`` constituents — the leaf rows can be re-aggregated by the
        consumer without round-tripping the per-task fixture.
        """
        per_task = pd.DataFrame(
            [
                {
                    "method": "A",
                    "scenario": "random_noise",
                    "channel": "ch_0",
                    "task_rank": 1.5,
                    "n_users": 100,
                },
                {
                    "method": "A",
                    "scenario": "random_noise",
                    "channel": "ch_1",
                    "task_rank": 2.5,
                    "n_users": 100,
                },
                {
                    "method": "A",
                    "scenario": "random_noise",
                    "channel": "ch_2",
                    "task_rank": 3.0,
                    "n_users": 100,
                },
            ]
        )
        result = aggregate_task_ranks_to_scopes(per_task)
        scenario_rank = float(result[result["scope"] == "random_noise"]["avg_rank"].iloc[0])
        task_ranks = result[result["scope"].str.startswith("task:")]["avg_rank"].values
        np.testing.assert_allclose(scenario_rank, float(np.mean(task_ranks)), rtol=1e-9)
        np.testing.assert_allclose(scenario_rank, (1.5 + 2.5 + 3.0) / 3, rtol=1e-9)

    def test_n_users_optional(self):
        """Rank input without ``n_users`` still emits task scopes; column omitted.

        Rank input without ``n_users`` still emits task scopes; the output
        ``n_users`` column is omitted on every row consistently.
        """
        per_task = pd.DataFrame(
            [
                {"method": "A", "scenario": "random_noise", "channel": "ch_0", "task_rank": 1.0},
                {"method": "A", "scenario": "random_noise", "channel": "ch_1", "task_rank": 2.0},
            ]
        )
        result = aggregate_task_ranks_to_scopes(per_task)
        task_rows = result[result["scope"].str.startswith("task:")]
        assert "n_users" not in result.columns
        assert len(task_rows) == 2


def test_task_scopes_flow_through_bootstrap_phase2():
    """End-to-end: ``task:*`` scopes appear in Phase-2 skill/rank tables.

    The deterministic synthetic-draws fixture uses base errors locf=1.0,
    mean=2.0, linear=0.5 so R is constant per method, and the per-task rank
    is fixed at 1 (linear), 2 (locf), 3 (mean). The bootstrap mean for each
    ``task:<scenario>:<channel>`` cell should equal the deterministic single
    -task value.
    """
    draws = _synthetic_draws()
    tables = aggregate_skill_rank_fairness(draws, baseline_method="locf")

    skill = tables["skill_scores"]
    rank = tables["avg_rankings"]
    skill_scopes = set(skill["scope"])
    rank_scopes = set(rank["scope"])

    # Synthetic fixture uses 2 scenarios × 2 channels → 4 task scopes each.
    expected_task_scopes = {
        f"task:{sc}:{ch}" for sc in ("random_noise", "block_random") for ch in ("ch_0", "ch_1")
    }
    assert expected_task_scopes.issubset(skill_scopes)
    assert expected_task_scopes.issubset(rank_scopes)

    # Each task:* row should report n_tasks=1.
    task_skill = skill[skill["scope"].str.startswith("task:")]
    assert (task_skill["n_tasks"] == 1).all()
    task_rank = rank[rank["scope"].str.startswith("task:")]
    assert (task_rank["n_tasks"] == 1).all()

    # locf-vs-self ⇒ R ≈ 1 ⇒ task skill ≈ 0; linear (E=0.5/locf E=1.0=R=0.5)
    # ⇒ task skill ≈ 0.5. Use the same ~5e-2 tolerance as the existing
    # test_skill_score_for_baseline_is_zero.
    locf_task = task_skill[task_skill["method"] == "locf"]
    assert abs(float(locf_task["mean"].mean())) < 5e-2
    linear_task = task_skill[task_skill["method"] == "linear"]
    np.testing.assert_allclose(float(linear_task["mean"].mean()), 0.5, atol=5e-2)

    # Task rank == task_rank, which is constant per method in the fixture:
    # linear=1, locf=2, mean=3.
    rank_by_method = task_rank.groupby("method", observed=True)["mean"].mean().to_dict()
    np.testing.assert_allclose(rank_by_method["linear"], 1.0, atol=1e-6)
    np.testing.assert_allclose(rank_by_method["locf"], 2.0, atol=1e-6)
    np.testing.assert_allclose(rank_by_method["mean"], 3.0, atol=1e-6)


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


# --------------------------------------------------------------------------
# Per-user errors emission contract (BCa LOO substrate; METRICS.md §S7)
# --------------------------------------------------------------------------


def test_compute_per_draw_errors_returns_tuple_with_emit_per_user(tmp_path):
    """``emit_per_user_errors=True`` returns a 2-tuple with the per-user schema.

    With no pairs files in the synthetic tree the frames are empty, but the
    schema contract must still hold.
    """
    from imputation_evaluation.evaluation.bootstrap_skill_rank import (
        PER_USER_ERRORS_PARQUET_COLUMNS,
    )

    _, method_dirs = _build_two_method_dirs(tmp_path, mismatch_kind="none")
    result = compute_per_draw_errors(
        **_phase1_kwargs(method_dirs, with_subgroups=False),
        emit_per_user_errors=True,
    )
    assert isinstance(result, tuple) and len(result) == 2
    draws_df, per_user_df = result
    assert list(draws_df.columns) == DRAWS_PARQUET_COLUMNS
    assert list(per_user_df.columns) == PER_USER_ERRORS_PARQUET_COLUMNS


def test_compute_per_draw_errors_default_returns_dataframe(tmp_path):
    """The default (``emit_per_user_errors=False``) still returns one DataFrame.

    Backward-compat anchor.
    """
    _, method_dirs = _build_two_method_dirs(tmp_path, mismatch_kind="none")
    out = compute_per_draw_errors(**_phase1_kwargs(method_dirs, with_subgroups=False))
    # Single DataFrame, not a tuple.
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == DRAWS_PARQUET_COLUMNS


def test_per_user_fast_path_uses_split_specific_manifests():
    """Multi-split fast path must rebuild canonical user ordering per split."""
    test_manifest = pa.table(
        {
            "sample_idx": [0, 1],
            "user_id": ["test_u0", "test_u1"],
            "date": ["2024-01-01", "2024-01-02"],
        }
    )
    val_manifest = pa.table(
        {
            "sample_idx": [0],
            "user_id": ["val_u0"],
            "date": ["2024-02-01"],
        }
    )
    per_user_df = pd.DataFrame(
        [
            {
                "method": "locf",
                "scenario": "scenarioA",
                "split": "test",
                "channel": "ch_0",
                "channel_type": "continuous",
                "subgroup_attr": "all",
                "subgroup_value": "all",
                "user_id": "test_u0",
                "E_per_user": 1.0,
            },
            {
                "method": "locf",
                "scenario": "scenarioA",
                "split": "val",
                "channel": "ch_0",
                "channel_type": "continuous",
                "subgroup_attr": "all",
                "subgroup_value": "all",
                "user_id": "val_u0",
                "E_per_user": 2.0,
            },
        ]
    )

    draws = compute_per_draw_errors_from_per_user_errors(
        per_user_df,
        {
            "test": {"locf": test_manifest},
            "val": {"locf": val_manifest},
        },
        n_boot=1,
        seed=0,
    )

    by_split = draws.set_index("split")["E"].to_dict()
    assert by_split["test"] == pytest.approx(1.0)
    assert by_split["val"] == pytest.approx(2.0)
