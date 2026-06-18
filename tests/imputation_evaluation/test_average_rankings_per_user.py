"""Tests for the per-user average-rank refactor.

Pins the post-refactor contract:

1. Point-flow ↔ bootstrap identity-draw parity for average rank
   (`compute_average_rankings` ≡ bootstrap rank kernel
   with `boot_idx = arange(U)`).
2. Point-flow ↔ bootstrap identity-draw parity for the paired skill score
   (`compute_per_task_paired_R` + `compute_skill_scores(mode="paired")` ≡
   bootstrap `_per_method_cell_paired_ratios` + `compute_skill_scores`).
3. Cross-track parity Stage 1: the per-(user, task) rank step matches
   forecasting's `_compute_mean_ranks` on a single-channel fixture.
4. Cross-track parity Stage 2: cross-channel mean of per-channel ranks
   matches forecasting's `AVG[0-K]` construction.
5. Explicit mode is required — the silent switch on column presence has
   been removed.
6. Tie handling uses `method="average"` on both sides.
7. Ragged coverage (a method missing for some (user, task) cells) is
   tolerated via `nanmean`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.processing.hf_config import N_CHANNELS
from imputation_evaluation.evaluation.bootstrap_skill_rank import (
    BinaryRows,
    CellStats,
    _per_method_cell_paired_ratios,
    _per_user_errors_for_cell,
    _rank_per_draw_for_cell,
    compute_per_task_paired_R,
)
from imputation_evaluation.evaluation.paper_metrics_core import (
    aggregate_task_ranks_to_scopes,
    compute_average_rankings,
    compute_skill_scores,
)


def _embed(arr: np.ndarray, n_channels: int = N_CHANNELS) -> np.ndarray:
    """Zero-pad a ``(n_users, k)`` array to ``(n_users, n_channels)``.

    Mirrors the helper in ``test_skill_score_parity.py`` — duplicated
    inline so this file is self-contained (the tests package has no
    ``__init__.py`` and cross-file imports are fragile).
    """
    if arr.shape[1] == n_channels:
        return arr
    out = np.zeros((arr.shape[0], n_channels), dtype=arr.dtype)
    out[:, : arr.shape[1]] = arr
    return out


def _make_cell_stats(
    *,
    n_users: int,
    sae: np.ndarray,
    sse: np.ndarray,
    n: np.ndarray,
    binary_rows: dict[int, BinaryRows] | None = None,
    has_data: np.ndarray | None = None,
    n_channels: int = N_CHANNELS,
) -> CellStats:
    """Minimal CellStats builder for unit tests (mirror of the helper in
    ``test_skill_score_parity.py``)."""
    sae = _embed(sae, n_channels)
    sse = _embed(sse, n_channels)
    n = _embed(n, n_channels)
    if has_data is None:
        has_data = (n.sum(axis=0) > 0) | bool(binary_rows)
    return CellStats(
        user_ids=[f"u{i}" for i in range(n_users)],
        n=n.astype(np.int64),
        sse=sse.astype(np.float64),
        sae=sae.astype(np.float64),
        tp=np.zeros((n_users, n_channels), dtype=np.int64),
        tn=np.zeros((n_users, n_channels), dtype=np.int64),
        fp=np.zeros((n_users, n_channels), dtype=np.int64),
        fn=np.zeros((n_users, n_channels), dtype=np.int64),
        has_data=has_data.astype(bool),
        binary_rows=binary_rows or {},
    )


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _make_per_user_long(
    *,
    methods: tuple[str, ...] = ("locf", "mean", "linear"),
    channels: tuple[str, ...] = ("ch_0", "ch_1"),
    n_users: int = 6,
    base_error: dict[str, float] | None = None,
    seed: int = 0,
    scenario: str = "scenarioA",
) -> pd.DataFrame:
    """Build a long-format per-user errors frame the rank reducers consume.

    Each (method, channel, user) gets ``E = base_error[method] + noise``
    so that the per-(user, channel) rank of methods is consistent across
    users (low ε relative to the per-method gap).
    """
    if base_error is None:
        base_error = {"locf": 1.0, "mean": 2.0, "linear": 0.5}
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for method in methods:
        mu = base_error[method]
        for ch in channels:
            for u in range(n_users):
                e = float(max(1e-3, mu + rng.normal(0, 0.02)))
                rows.append({
                    "method": method,
                    "scenario": scenario,
                    "channel": ch,
                    "channel_type": "continuous",
                    "user_id": f"u{u}",
                    "E": e,
                })
    return pd.DataFrame(rows)


def _make_per_user_E_matrix(
    long_frame: pd.DataFrame,
    *,
    methods: tuple[str, ...],
    channels: tuple[str, ...],
    user_ids: list[str],
) -> dict[str, np.ndarray]:
    """Reshape the long per-user frame into ``{method: (U, n_channels)}``."""
    out: dict[str, np.ndarray] = {}
    ch_to_col = {ch: i for i, ch in enumerate(channels)}
    user_to_row = {u: i for i, u in enumerate(user_ids)}
    for method in methods:
        mat = np.full((len(user_ids), len(channels)), np.nan, dtype=np.float64)
        sub = long_frame[long_frame["method"] == method]
        for _, row in sub.iterrows():
            i = user_to_row.get(row["user_id"])
            j = ch_to_col.get(row["channel"])
            if i is None or j is None:
                continue
            mat[i, j] = row["E"]
        out[method] = mat
    return out


# ---------------------------------------------------------------------------
# 1. Point-flow ↔ bootstrap identity-draw parity (rank)
# ---------------------------------------------------------------------------


def test_per_user_rank_point_flow_matches_bootstrap_identity_draw():
    """`compute_average_rankings` ≡ bootstrap rank kernel
    with `boot_idx = arange(U)`.

    Multi-method, multi-task fixture so the two-stage form is non-trivial
    (per-task mean, then scope-mean). Single subgroup cell.
    """
    methods = ("locf", "mean", "linear")
    channels = ("ch_0", "ch_1", "ch_2")
    n_users = 7
    eu = _make_per_user_long(
        methods=methods, channels=channels, n_users=n_users, seed=11,
    )
    user_ids = [f"u{i}" for i in range(n_users)]

    # Point-flow.
    point = compute_average_rankings(eu)
    point_overall = (
        point[point["scope"] == "overall"].set_index("method")["avg_rank"]
    )

    # Bootstrap identity draw — manual.
    per_user_E_by_method = _make_per_user_E_matrix(
        eu, methods=methods, channels=channels, user_ids=user_ids,
    )
    idx_b = np.arange(n_users).reshape(1, n_users)
    rank_by_method = _rank_per_draw_for_cell(
        per_user_E_by_method, idx_b, channel_indices=range(len(channels)),
    )
    # Mirror Phase 1's emit + Phase 2's aggregation.
    rows: list[dict] = []
    for method, rank_mat in rank_by_method.items():
        for ch_idx, ch in enumerate(channels):
            val = rank_mat[0, ch_idx]
            if not np.isfinite(val):
                continue
            rows.append({
                "method": method,
                "scenario": "scenarioA",
                "channel": ch,
                "task_rank": float(val),
            })
    boot_per_task = pd.DataFrame(rows)
    boot = aggregate_task_ranks_to_scopes(boot_per_task)
    boot_overall = boot[boot["scope"] == "overall"].set_index("method")["avg_rank"]

    # Same methods on both sides, same avg_rank up to fp precision.
    assert set(point_overall.index) == set(boot_overall.index)
    for method in point_overall.index:
        assert point_overall[method] == pytest.approx(boot_overall[method], rel=1e-9)


# ---------------------------------------------------------------------------
# 2. Point-flow ↔ bootstrap identity-draw parity (paired skill)
# ---------------------------------------------------------------------------


def test_point_flow_paired_skill_matches_bootstrap_identity_draw():
    """`compute_per_task_paired_R` + `compute_skill_scores(mode="paired")`
    point-flow matches the bootstrap `_per_method_cell_paired_ratios` +
    `compute_skill_scores(mode="paired")` path at the identity draw.
    """
    methods = ("locf", "mean", "linear")
    channels = ("ch_0", "ch_1")
    n_users = 5
    base = {"locf": 1.0, "mean": 1.5, "linear": 0.7}
    eu = _make_per_user_long(
        methods=methods, channels=channels, n_users=n_users,
        base_error=base, seed=7,
    )

    point_R = compute_per_task_paired_R(eu, baseline_method="locf")
    point_skill = compute_skill_scores(point_R, mode="paired")
    point_overall = (
        point_skill[point_skill["scope"] == "overall"]
        .set_index("method")["skill_score"]
    )

    # Bootstrap path — only need (method, baseline) per-channel R via
    # _per_method_cell_paired_ratios on identity-draw idx_b.
    user_ids = [f"u{i}" for i in range(n_users)]
    boot_overall: dict[str, float] = {}
    boot_R_rows: list[dict] = []
    for method in methods:
        if method == "locf":
            continue
        # Build per-(user, channel) E matrices for method and baseline.
        sae_m = np.zeros((n_users, len(channels)), dtype=np.float64)
        sae_b = np.zeros((n_users, len(channels)), dtype=np.float64)
        n_arr = np.full((n_users, len(channels)), 1, dtype=np.int64)
        for j, ch in enumerate(channels):
            sub_m = eu[(eu["method"] == method) & (eu["channel"] == ch)]
            sub_b = eu[(eu["method"] == "locf") & (eu["channel"] == ch)]
            for u_idx, uid in enumerate(user_ids):
                em = float(sub_m.loc[sub_m["user_id"] == uid, "E"].iloc[0])
                eb = float(sub_b.loc[sub_b["user_id"] == uid, "E"].iloc[0])
                sae_m[u_idx, j] = em
                sae_b[u_idx, j] = eb
        cs_m = _make_cell_stats(
            n_users=n_users, sae=sae_m, sse=np.zeros_like(sae_m), n=n_arr,
        )
        cs_b = _make_cell_stats(
            n_users=n_users, sae=sae_b, sse=np.zeros_like(sae_b), n=n_arr,
        )
        idx_b = np.arange(n_users).reshape(1, n_users)
        R = _per_method_cell_paired_ratios(
            cs_m, cs_b, idx_b, include_auc=False,
        )
        for j, ch in enumerate(channels):
            boot_R_rows.append({
                "method": method,
                "scenario": "scenarioA",
                "channel": ch,
                "channel_type": "continuous",
                "R": float(R[0, j]),
            })
    boot_R = pd.DataFrame(boot_R_rows)
    boot_skill = compute_skill_scores(boot_R, mode="paired")
    boot_overall = (
        boot_skill[boot_skill["scope"] == "overall"]
        .set_index("method")["skill_score"]
    )

    # Compare overall skill for methods present in both.
    common = set(point_overall.index) & set(boot_overall.index)
    assert common, "no methods in common between point-flow and bootstrap skill"
    for method in common:
        assert point_overall[method] == pytest.approx(
            boot_overall[method], rel=1e-9,
        )


# ---------------------------------------------------------------------------
# 3. Cross-track parity, Stage 1: per-channel rank == forecasting reducer
# ---------------------------------------------------------------------------


def _forecasting_mean_ranks_per_user(
    df: pd.DataFrame,
    *,
    metric_value_col: str = "E",
) -> pd.DataFrame:
    """Mirror ``forecasting._compute_mean_ranks`` on (user_id, model, scope, metric).

    Reimplemented inline so the parity test doesn't depend on which
    forecasting branch is on PYTHONPATH. ``ascending=True`` because the
    imputation E is lower-is-better.
    """
    finite = df.loc[np.isfinite(df[metric_value_col])]
    rows: list[pd.DataFrame] = []
    for (scope, metric_name), group in finite.groupby(["scope", "metric"], sort=True):
        pivot = group.pivot(index="user_id", columns="model", values=metric_value_col)
        if pivot.empty:
            continue
        rank_df = pivot.rank(axis=1, method="average", ascending=True)
        long_rank = rank_df.stack(future_stack=True).reset_index()
        long_rank.columns = ["user_id", "model", "rank"]
        long_rank["scope"] = scope
        long_rank["metric"] = metric_name
        rows.append(long_rank)
    if not rows:
        return pd.DataFrame(columns=["scope", "metric", "model", "rank", "rank_n_users"])
    rank_all = pd.concat(rows, ignore_index=True)
    return (
        rank_all.groupby(["scope", "metric", "model"], as_index=False)
        .agg(rank=("rank", "mean"), rank_n_users=("user_id", "nunique"))
    )


def test_per_user_rank_single_task_matches_forecasting():
    """Stage 1 parity: with a single (scenario, channel) task, the imputation
    per-task rank equals forecasting's `_compute_mean_ranks` reducer.

    Map ``(scenario, channel) ↔ (scope, metric)`` on the forecasting side.
    """
    methods = ("locf", "mean", "linear")
    eu = _make_per_user_long(
        methods=methods, channels=("ch_0",), n_users=8, seed=3,
    )

    # Imputation per-user task_rank (mean over users) per (method, scenario, channel).
    imp_full = compute_average_rankings(eu)
    imp_scenario = (
        imp_full[imp_full["scope"] == "scenarioA"]
        .set_index("method")["avg_rank"]
    )

    # Forecasting analog — relabel.
    forecast_df = eu.rename(
        columns={"method": "model", "scenario": "scope", "channel": "metric"},
    )
    forecast_df = forecast_df[["user_id", "model", "scope", "metric", "E"]]
    fc = _forecasting_mean_ranks_per_user(forecast_df)
    # Single channel → single (scope, metric) cell.
    fc_scenario = fc.set_index("model")["rank"]

    common = set(imp_scenario.index) & set(fc_scenario.index)
    assert common == set(methods)
    for method in common:
        assert imp_scenario[method] == pytest.approx(fc_scenario[method], rel=1e-12)


def test_per_user_rank_multitask_matches_forecasting_cross_channel_mean():
    """Stage 2 parity: the imputation overall avg_rank equals the
    cross-channel mean of forecasting per-channel ranks.

    This is the hard contract that the two tracks aggregate ranks across
    tasks/channels in the same way.
    """
    methods = ("locf", "mean", "linear")
    channels = ("ch_0", "ch_1", "ch_2")
    eu = _make_per_user_long(
        methods=methods, channels=channels, n_users=10, seed=5,
    )

    imp = compute_average_rankings(eu)
    imp_overall = (
        imp[imp["scope"] == "overall"].set_index("method")["avg_rank"]
    )

    # Forecasting: per-channel rank, then arithmetic mean across channels.
    forecast_df = eu.rename(
        columns={"method": "model", "scenario": "scope", "channel": "metric"},
    )[["user_id", "model", "scope", "metric", "E"]]
    fc = _forecasting_mean_ranks_per_user(forecast_df)
    # Average per-channel ranks across channels per (scope=scenarioA, model).
    fc_avg = (
        fc.groupby("model", observed=True)["rank"]
        .mean()
    )

    common = set(imp_overall.index) & set(fc_avg.index)
    assert common == set(methods)
    for method in common:
        assert imp_overall[method] == pytest.approx(fc_avg[method], rel=1e-12)


# ---------------------------------------------------------------------------
# 5. Explicit mode required
# ---------------------------------------------------------------------------


def test_compute_average_rankings_raises_without_user_id():
    """compute_average_rankings requires a user_id column; raises if absent."""
    no_uid = pd.DataFrame({
        "method": ["a", "b"],
        "scenario": ["s", "s"],
        "channel": ["ch_0", "ch_0"],
        "channel_type": ["continuous", "continuous"],
        "E": [1.0, 0.5],
    })
    with pytest.raises(ValueError, match="user_id"):
        compute_average_rankings(no_uid)


def test_compute_skill_scores_paired_raises_without_R():
    """mode='paired' requires R; raises if absent."""
    no_R = pd.DataFrame({
        "method": ["a"],
        "scenario": ["s"],
        "channel": ["ch_0"],
        "channel_type": ["continuous"],
        "E": [0.5],
    })
    with pytest.raises(ValueError, match="'R' column"):
        compute_skill_scores(no_R, mode="paired")


def test_compute_skill_scores_pooled_still_works_explicitly():
    """The legacy pooled path still works when explicitly selected."""
    errors = pd.DataFrame({
        "method": ["m", "m"],
        "scenario": ["s", "s"],
        "channel": ["ch_0", "ch_1"],
        "channel_type": ["continuous", "continuous"],
        "E": [0.5, 0.4],
    })
    baseline = pd.DataFrame({
        "method": ["b", "b"],
        "scenario": ["s", "s"],
        "channel": ["ch_0", "ch_1"],
        "channel_type": ["continuous", "continuous"],
        "E": [1.0, 1.0],
    })
    out = compute_skill_scores(errors, baseline, mode="pooled")
    assert not out.empty
    # m beats baseline → positive skill.
    overall = out[(out["method"] == "m") & (out["scope"] == "overall")]
    assert float(overall["skill_score"].iloc[0]) > 0.0


# ---------------------------------------------------------------------------
# 6. Tie handling
# ---------------------------------------------------------------------------


def test_tie_handling_method_average():
    """Two methods with identical per-user E on every user/task → both
    get rank == 1.5 at the task grain and at the scope grain.
    """
    # Two methods tied (E=0.5), one method strictly worse (E=2.0). With
    # method='average', the two ties share ranks 1 and 2 → both 1.5.
    rows = []
    for u in range(4):
        for ch in ("ch_0", "ch_1"):
            rows.append({"method": "A", "scenario": "s", "channel": ch,
                         "channel_type": "continuous", "user_id": f"u{u}", "E": 0.5})
            rows.append({"method": "B", "scenario": "s", "channel": ch,
                         "channel_type": "continuous", "user_id": f"u{u}", "E": 0.5})
            rows.append({"method": "C", "scenario": "s", "channel": ch,
                         "channel_type": "continuous", "user_id": f"u{u}", "E": 2.0})
    out = compute_average_rankings(pd.DataFrame(rows))
    overall = out[out["scope"] == "overall"].set_index("method")["avg_rank"]
    assert overall["A"] == pytest.approx(1.5, rel=1e-12)
    assert overall["B"] == pytest.approx(1.5, rel=1e-12)
    assert overall["C"] == pytest.approx(3.0, rel=1e-12)


# ---------------------------------------------------------------------------
# 7. Ragged coverage
# ---------------------------------------------------------------------------


def test_ragged_user_coverage_tolerated():
    """Method missing for some (user, task) cells: pivot.rank ignores
    those cells (NaN passthrough); nanmean over users computes task_rank
    only on users with finite E. The scope mean drops methods with zero
    finite tasks gracefully.
    """
    rows = []
    # Method A present everywhere. Method B missing for user u3 on ch_1.
    for u in range(4):
        rows.append({"method": "A", "scenario": "s", "channel": "ch_0",
                     "channel_type": "continuous", "user_id": f"u{u}", "E": 0.5 + 0.1 * u})
        rows.append({"method": "B", "scenario": "s", "channel": "ch_0",
                     "channel_type": "continuous", "user_id": f"u{u}", "E": 1.0 + 0.1 * u})
    for u in range(4):
        rows.append({"method": "A", "scenario": "s", "channel": "ch_1",
                     "channel_type": "continuous", "user_id": f"u{u}", "E": 0.5})
        if u != 3:
            rows.append({"method": "B", "scenario": "s", "channel": "ch_1",
                         "channel_type": "continuous", "user_id": f"u{u}", "E": 1.0})
    out = compute_average_rankings(pd.DataFrame(rows))
    overall = out[out["scope"] == "overall"].set_index("method")["avg_rank"]
    # A < B on both tasks for every overlapping user → A rank 1, B rank 2
    # on every (user, task) with both methods present. Average == constant.
    assert overall["A"] == pytest.approx(1.0, rel=1e-12)
    assert overall["B"] == pytest.approx(2.0, rel=1e-12)


# ---------------------------------------------------------------------------
# 8. Per-user errors helper round-trip
# ---------------------------------------------------------------------------


def test_per_user_errors_for_cell_continuous_round_trip():
    """`_per_user_errors_for_cell` continuous channels reproduce sae/n."""
    n_users, n_channels = 3, 2
    sae = np.array([[2.0, 4.0], [3.0, 6.0], [1.0, 2.0]], dtype=np.float64)
    sse = np.zeros_like(sae)
    n_arr = np.full((n_users, n_channels), 4, dtype=np.int64)
    cs = _make_cell_stats(n_users=n_users, sae=sae, sse=sse, n=n_arr)
    per_user_E = _per_user_errors_for_cell(
        cs, n_channels=N_CHANNELS, include_auc=False,
    )
    np.testing.assert_allclose(per_user_E[:, 0], [0.5, 0.75, 0.25])
    np.testing.assert_allclose(per_user_E[:, 1], [1.0, 1.5, 0.5])
