"""Unit tests for Part D — binary-collapsed scopes.

The leaderboard's geomean of clipped error ratios across ``(scenario, channel)``
tasks weights the 10 individually-sparse workout channels 10x sleep's 2
channels. Part D adds three new scopes that collapse the two binary
categories (sleep, workouts) into one task per scenario each:

- ``cat_collapsed:sleep`` / ``cat_collapsed:workouts``
- ``overall_binary_collapsed`` (7 continuous per-channel tasks + 2 binary
  category tasks × scenarios)

These tests pin the math:

  (a) Per-(user, category) E = ``nanmean(1 - AUC)`` over the category's
      channels for that user.
  (b) Per-task E = ``nanmean`` over resampled users of (a) (preserving
      bootstrap multiplicity).
  (c) ``compute_skill_scores`` partitions per-channel and collapsed rows
      cleanly: legacy scopes (``overall``, ``cat:*``, etc.) consume only
      per-channel rows; new scopes only consume the matching mix.
  (d) Rare workout channels with single-class users drop cleanly from the
      collapsed mean without polluting the legacy per-channel paths.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from imputation_evaluation.evaluation.bootstrap_skill_rank import (
    _per_method_cell_collapsed_errors,
)
from imputation_evaluation.evaluation.paper_metrics_core import (
    BINARY_CATEGORIES_ORDERED,
    compute_skill_scores,
    is_collapsed_channel,
)


N_CHANNELS = 19


# ---------------------------------------------------------------------------
# Reducer math: per-(user, category) → per-task E
# ---------------------------------------------------------------------------


class TestCollapsedReducer:
    def test_sleep_two_channel_mean(self):
        """E_cat[u, sleep] = mean(1 - AUC[u, ch_7], 1 - AUC[u, ch_8]).

        Two users; both have well-defined AUC on ch_7 and ch_8; one
        bootstrap draw with no resampling.
        """
        n_users = 2
        per_user_auc = np.full((n_users, N_CHANNELS), np.nan, dtype=np.float64)
        # User 0: AUC(ch_7) = 0.8, AUC(ch_8) = 0.6
        per_user_auc[0, 7] = 0.8
        per_user_auc[0, 8] = 0.6
        # User 1: AUC(ch_7) = 0.5, AUC(ch_8) = 1.0
        per_user_auc[1, 7] = 0.5
        per_user_auc[1, 8] = 1.0

        boot_idx = np.array([[0, 1]])  # one draw, both users
        E = _per_method_cell_collapsed_errors(per_user_auc, boot_idx, BINARY_CATEGORIES_ORDERED)
        # Per-user E for sleep:
        #   u0 = mean(1 - 0.8, 1 - 0.6) = mean(0.2, 0.4) = 0.3
        #   u1 = mean(1 - 0.5, 1 - 1.0) = mean(0.5, 0.0) = 0.25
        # Per-task E: mean(0.3, 0.25) = 0.275
        sleep_idx = next(i for i, (n, _) in enumerate(BINARY_CATEGORIES_ORDERED) if n == "sleep")
        np.testing.assert_allclose(E[0, sleep_idx], 0.275, rtol=1e-9)

    def test_workouts_drops_undefined_channels_per_user(self):
        """A user with no defined AUC on ch_11/ch_12 still gets a workouts E
        from their defined ch_9 and ch_10 — nanmean across the user's own
        defined channels.
        """
        n_users = 2
        per_user_auc = np.full((n_users, N_CHANNELS), np.nan, dtype=np.float64)
        # User 0: only ch_9 = 0.7, the other 9 workout channels are NaN
        per_user_auc[0, 9] = 0.7
        # User 1: ch_9 = 0.5, ch_10 = 0.9 — two defined workout channels
        per_user_auc[1, 9] = 0.5
        per_user_auc[1, 10] = 0.9

        boot_idx = np.array([[0, 1]])
        E = _per_method_cell_collapsed_errors(per_user_auc, boot_idx, BINARY_CATEGORIES_ORDERED)
        # Per-user E for workouts:
        #   u0 = nanmean(1 - 0.7) = 0.3 (only ch_9 defined)
        #   u1 = nanmean(1 - 0.5, 1 - 0.9) = mean(0.5, 0.1) = 0.3
        # Per-task E: mean(0.3, 0.3) = 0.3
        workouts_idx = next(i for i, (n, _) in enumerate(BINARY_CATEGORIES_ORDERED) if n == "workouts")
        np.testing.assert_allclose(E[0, workouts_idx], 0.3, rtol=1e-9)

    def test_user_with_zero_defined_channels_in_category_drops(self):
        """A user with NaN AUC across all workouts channels contributes NaN
        and drops from the per-task macro.
        """
        n_users = 3
        per_user_auc = np.full((n_users, N_CHANNELS), np.nan, dtype=np.float64)
        per_user_auc[0, 9] = 0.8
        per_user_auc[1, 9] = 0.6
        # user 2 has nothing in workouts (all NaN) → per-user E NaN → dropped.

        boot_idx = np.array([[0, 1, 2]])
        E = _per_method_cell_collapsed_errors(per_user_auc, boot_idx, BINARY_CATEGORIES_ORDERED)
        # per-task workouts E = mean(0.2, 0.4) = 0.3
        workouts_idx = next(i for i, (n, _) in enumerate(BINARY_CATEGORIES_ORDERED) if n == "workouts")
        np.testing.assert_allclose(E[0, workouts_idx], 0.3, rtol=1e-9)

    def test_bootstrap_multiplicity_replicates_per_user_E(self):
        """A user resampled twice contributes twice to the per-task mean."""
        n_users = 2
        per_user_auc = np.full((n_users, N_CHANNELS), np.nan, dtype=np.float64)
        per_user_auc[0, 7] = 0.9
        per_user_auc[0, 8] = 0.9
        per_user_auc[1, 7] = 0.5
        per_user_auc[1, 8] = 0.5

        # Draw 0: equal weights. Draw 1: u0 twice.
        boot_idx = np.array([[0, 1], [0, 0]])
        E = _per_method_cell_collapsed_errors(per_user_auc, boot_idx, BINARY_CATEGORIES_ORDERED)
        sleep_idx = next(i for i, (n, _) in enumerate(BINARY_CATEGORIES_ORDERED) if n == "sleep")
        # u0 per-user E = 0.1, u1 = 0.5
        np.testing.assert_allclose(E[0, sleep_idx], 0.3, rtol=1e-9)  # mean(0.1, 0.5)
        np.testing.assert_allclose(E[1, sleep_idx], 0.1, rtol=1e-9)  # mean(0.1, 0.1)


# ---------------------------------------------------------------------------
# compute_skill_scores: scope partitioning, n_tasks counts
# ---------------------------------------------------------------------------


def _build_errors_df(rows):
    """Helper: build a (method, scenario, channel, ..., E) DataFrame."""
    return pd.DataFrame(rows)


class TestCollapsedSkillScopes:
    def test_legacy_overall_excludes_collapsed_rows(self):
        """``overall`` scope should only count per-channel tasks (114 max
        per scenario × method × channel), never the cat_collapsed:* rows.
        """
        scenarios = ["random_noise"]
        rows = []
        # Method A: per-channel rows for two continuous channels and
        # one sleep channel, plus collapsed rows for sleep.
        for sc in scenarios:
            for ch in ("ch_0", "ch_1", "ch_7"):
                rows.append({
                    "method": "A", "scenario": sc, "channel": ch,
                    "channel_type": "continuous" if ch.startswith("ch_") and int(ch[3:]) < 7 else "binary",
                    "E": 0.5,
                })
            rows.append({
                "method": "A", "scenario": sc, "channel": "cat_collapsed:sleep",
                "channel_type": "binary_collapsed", "E": 0.4,
            })
            rows.append({
                "method": "A", "scenario": sc, "channel": "cat_collapsed:workouts",
                "channel_type": "binary_collapsed", "E": 0.45,
            })
        # Baseline (E=1.0 everywhere → ratios = the method's E directly).
        bl_rows = [
            dict(method="LOCF", scenario=sc, channel=ch, channel_type="continuous", E=1.0)
            for sc in scenarios
            for ch in ("ch_0", "ch_1", "ch_7")
        ] + [
            dict(method="LOCF", scenario=sc, channel="cat_collapsed:sleep",
                 channel_type="binary_collapsed", E=1.0)
            for sc in scenarios
        ] + [
            dict(method="LOCF", scenario=sc, channel="cat_collapsed:workouts",
                 channel_type="binary_collapsed", E=1.0)
            for sc in scenarios
        ]

        errors = _build_errors_df(rows)
        bl = _build_errors_df(bl_rows)
        result = compute_skill_scores(errors, bl, mode="pooled")

        overall_row = result[(result["method"] == "A") & (result["scope"] == "overall")]
        assert len(overall_row) == 1
        # ``overall`` includes only the 3 per-channel rows: ch_0, ch_1, ch_7.
        assert int(overall_row["n_tasks"].iloc[0]) == 3

        # cat_collapsed:sleep present and has 1 task (one scenario).
        sleep_row = result[(result["method"] == "A") & (result["scope"] == "cat_collapsed:sleep")]
        assert len(sleep_row) == 1
        assert int(sleep_row["n_tasks"].iloc[0]) == 1

        # overall_binary_collapsed: B.2 two-stage form — n_tasks now counts
        # **buckets**, not constituent tasks. The fixture has data for:
        #   activity (from ch_0, ch_1) + sleep (collapsed) + workouts (collapsed)
        # — 3 buckets, no physiology data. Per-channel ch_7 is binary and is
        # NOT consumed by overall_binary_collapsed (binary side uses the
        # collapsed scope).
        collapsed_overall = result[
            (result["method"] == "A") & (result["scope"] == "overall_binary_collapsed")
        ]
        assert len(collapsed_overall) == 1
        assert int(collapsed_overall["n_tasks"].iloc[0]) == 3

    def test_no_collapsed_rows_means_no_collapsed_scopes(self):
        """Backwards compat: when caller passes only per-channel rows, no
        ``cat_collapsed:*`` / ``overall_binary_collapsed`` scopes appear.
        """
        rows = [
            {"method": "A", "scenario": "random_noise", "channel": "ch_0",
             "channel_type": "continuous", "E": 0.5}
        ]
        bl_rows = [
            {"method": "LOCF", "scenario": "random_noise", "channel": "ch_0",
             "channel_type": "continuous", "E": 1.0}
        ]
        result = compute_skill_scores(
            _build_errors_df(rows), _build_errors_df(bl_rows), mode="pooled",
        )
        scopes = set(result["scope"])
        assert "overall" in scopes
        assert not any(s.startswith("cat_collapsed:") for s in scopes)
        assert "overall_binary_collapsed" not in scopes


class TestCollapsedChannelHelper:
    def test_is_collapsed_channel(self):
        assert is_collapsed_channel("cat_collapsed:sleep")
        assert is_collapsed_channel("cat_collapsed:workouts")
        assert not is_collapsed_channel("ch_7")
        assert not is_collapsed_channel("ch_0")
        assert not is_collapsed_channel("activity")


# ---------------------------------------------------------------------------
# Part D + task-grain emission interaction
# ---------------------------------------------------------------------------


class TestOverallBinaryCollapsedTwoStage:
    """``overall_binary_collapsed`` is the category-balanced two-stage form.

    Stage 1 (per-bucket): mean of log(R) over the bucket's constituent tasks
    — exactly what ``cat:activity`` / ``cat:physiology`` /
    ``cat_collapsed:*`` already report.

    Stage 2 (per method): arithmetic mean over the 4 buckets {activity,
    physiology, sleep, workouts}, so each category contributes equally
    regardless of how many constituent tasks it has.
    """

    def _build_balanced_fixture(self):
        """Two methods, all 4 buckets populated, two scenarios for the
        continuous side (so activity / physiology have multiple tasks)."""
        scenarios = ["random_noise", "temporal_slice"]
        rows = []
        # Method A: per-channel continuous (ch_0..ch_4 activity, ch_5..ch_6
        # physiology). Use known E values to make the bucket geomeans
        # closed-form.
        for sc in scenarios:
            # activity (5 channels, E=0.4 → R=0.4 vs baseline E=1.0)
            for c in range(5):
                rows.append(dict(method="A", scenario=sc, channel=f"ch_{c}",
                                 channel_type="continuous", E=0.4))
            # physiology (2 channels, E=0.5 → R=0.5)
            for c in (5, 6):
                rows.append(dict(method="A", scenario=sc, channel=f"ch_{c}",
                                 channel_type="continuous", E=0.5))
            # cat_collapsed:sleep (E=0.6 → R=0.6)
            rows.append(dict(method="A", scenario=sc,
                             channel="cat_collapsed:sleep",
                             channel_type="binary_collapsed", E=0.6))
            # cat_collapsed:workouts (E=0.7 → R=0.7)
            rows.append(dict(method="A", scenario=sc,
                             channel="cat_collapsed:workouts",
                             channel_type="binary_collapsed", E=0.7))
        # Method B: same shape but E=0.5 for every task so its overall = 0.5.
        for sc in scenarios:
            for c in range(7):
                rows.append(dict(method="B", scenario=sc, channel=f"ch_{c}",
                                 channel_type="continuous", E=0.5))
            for cat in ("sleep", "workouts"):
                rows.append(dict(method="B", scenario=sc,
                                 channel=f"cat_collapsed:{cat}",
                                 channel_type="binary_collapsed", E=0.5))
        # Baseline E=1.0 everywhere → ratios = method's E (in-bounds).
        bl_rows = []
        for sc in scenarios:
            for c in range(7):
                bl_rows.append(dict(method="LOCF", scenario=sc, channel=f"ch_{c}",
                                    channel_type="continuous", E=1.0))
            for cat in ("sleep", "workouts"):
                bl_rows.append(dict(method="LOCF", scenario=sc,
                                    channel=f"cat_collapsed:{cat}",
                                    channel_type="binary_collapsed", E=1.0))
        return _build_errors_df(rows), _build_errors_df(bl_rows)

    def test_n_tasks_equals_number_of_buckets(self):
        """n_tasks counts buckets, not constituent tasks. All 4 present → 4."""
        errors, bl = self._build_balanced_fixture()
        result = compute_skill_scores(errors, bl, mode="pooled")
        for method in ("A", "B"):
            row = result[(result["method"] == method) &
                         (result["scope"] == "overall_binary_collapsed")]
            assert len(row) == 1
            assert int(row["n_tasks"].iloc[0]) == 4

    def test_skill_equals_mean_of_bucket_log_ratios(self):
        """Closed-form check: log ratios cleanly average to a known value.

        Method A bucket log(R)s:
          activity   = log(0.4)
          physiology = log(0.5)
          sleep      = log(0.6)
          workouts   = log(0.7)
        mean = (log(0.4) + log(0.5) + log(0.6) + log(0.7)) / 4
        S    = 1 - exp(mean)
        """
        errors, bl = self._build_balanced_fixture()
        result = compute_skill_scores(errors, bl, mode="pooled")
        mean_log = (np.log(0.4) + np.log(0.5) + np.log(0.6) + np.log(0.7)) / 4
        expected_S = 1.0 - np.exp(mean_log)
        got = float(
            result[(result["method"] == "A") &
                   (result["scope"] == "overall_binary_collapsed")]
            ["skill_score"].iloc[0]
        )
        np.testing.assert_allclose(got, expected_S, rtol=1e-9)

    def test_per_channel_binary_scope_not_consumed(self):
        """Adding ``cat:sleep`` / ``cat:workouts`` per-channel rows must NOT
        change ``overall_binary_collapsed`` — the bucket priority rule is
        "collapsed wins for sleep / workouts".
        """
        errors, bl = self._build_balanced_fixture()
        baseline = float(
            compute_skill_scores(errors, bl, mode="pooled")
            .pipe(lambda d: d[(d["method"] == "A") &
                              (d["scope"] == "overall_binary_collapsed")])
            ["skill_score"].iloc[0]
        )
        # Now sneak in catastrophic per-channel binary rows; they belong to
        # ``cat:sleep`` / ``cat:workouts`` (ch_7..ch_18), NOT the collapsed
        # scopes. The new overall must be unchanged because it consumes the
        # collapsed values, not the per-channel binary ones.
        extra = []
        for sc in ("random_noise", "temporal_slice"):
            for c in (7, 8):                       # sleep per-channel
                extra.append(dict(method="A", scenario=sc, channel=f"ch_{c}",
                                  channel_type="binary", E=99.0))
            for c in range(9, 19):                 # workouts per-channel
                extra.append(dict(method="A", scenario=sc, channel=f"ch_{c}",
                                  channel_type="binary", E=99.0))
        bl_extra = []
        for sc in ("random_noise", "temporal_slice"):
            for c in (7, 8, *range(9, 19)):
                bl_extra.append(dict(method="LOCF", scenario=sc, channel=f"ch_{c}",
                                     channel_type="binary", E=1.0))
        errors_aug = pd.concat([errors, _build_errors_df(extra)], ignore_index=True)
        bl_aug     = pd.concat([bl,     _build_errors_df(bl_extra)], ignore_index=True)
        augmented = float(
            compute_skill_scores(errors_aug, bl_aug, mode="pooled")
            .pipe(lambda d: d[(d["method"] == "A") &
                              (d["scope"] == "overall_binary_collapsed")])
            ["skill_score"].iloc[0]
        )
        np.testing.assert_allclose(augmented, baseline, rtol=1e-9)


class TestTaskScopeWithCollapsedRows:
    """Task-grain ``task:<scenario>:<channel>`` rows must coexist with the
    Part D collapsed scopes — every per-channel input AND every
    ``cat_collapsed:*`` input gets its own leaf scope, without polluting the
    aggregated scopes (``overall``, ``cat:*``, ``overall_binary_collapsed``).
    """

    def test_collapsed_rows_emit_task_scopes(self):
        """``task:<sc>:cat_collapsed:<cat>`` rows are emitted per input."""
        scenarios = ["random_noise", "temporal_slice"]
        rows = []
        for sc in scenarios:
            rows.append({
                "method": "A", "scenario": sc, "channel": "ch_0",
                "channel_type": "continuous", "E": 0.4,
            })
            rows.append({
                "method": "A", "scenario": sc, "channel": "cat_collapsed:sleep",
                "channel_type": "binary_collapsed", "E": 0.5,
            })
            rows.append({
                "method": "A", "scenario": sc, "channel": "cat_collapsed:workouts",
                "channel_type": "binary_collapsed", "E": 0.6,
            })
        bl_rows = [
            {"method": "LOCF", "scenario": sc, "channel": ch,
             "channel_type": ("binary_collapsed" if ch.startswith("cat_collapsed:")
                              else "continuous"),
             "E": 1.0}
            for sc in scenarios
            for ch in ("ch_0", "cat_collapsed:sleep", "cat_collapsed:workouts")
        ]
        result = compute_skill_scores(
            _build_errors_df(rows), _build_errors_df(bl_rows), mode="pooled",
        )

        scopes = set(result[result["method"] == "A"]["scope"])
        for sc in scenarios:
            assert f"task:{sc}:ch_0" in scopes
            assert f"task:{sc}:cat_collapsed:sleep" in scopes
            assert f"task:{sc}:cat_collapsed:workouts" in scopes

        # Skill value at task-grain == 1 − clipped(E/E_baseline). With
        # baseline E=1.0 the ratio is the method's E directly (in-bounds).
        skill_by_scope = (
            result[result["method"] == "A"]
            .set_index("scope")["skill_score"].to_dict()
        )
        np.testing.assert_allclose(
            skill_by_scope["task:random_noise:cat_collapsed:sleep"], 0.5, rtol=1e-9,
        )
        np.testing.assert_allclose(
            skill_by_scope["task:random_noise:cat_collapsed:workouts"], 0.4, rtol=1e-9,
        )
        np.testing.assert_allclose(
            skill_by_scope["task:random_noise:ch_0"], 0.6, rtol=1e-9,
        )

    def test_task_scope_count_per_scenario(self):
        """One ``task:*`` row per (method, scenario, channel) row in the
        input — count is exact, not a lower bound.
        """
        rows = [
            {"method": "A", "scenario": "random_noise", "channel": "ch_0",
             "channel_type": "continuous", "E": 0.4},
            {"method": "A", "scenario": "random_noise", "channel": "ch_1",
             "channel_type": "continuous", "E": 0.5},
            {"method": "A", "scenario": "random_noise",
             "channel": "cat_collapsed:sleep",
             "channel_type": "binary_collapsed", "E": 0.6},
        ]
        bl_rows = [
            {"method": "LOCF", "scenario": "random_noise", "channel": ch,
             "channel_type": ("binary_collapsed" if ch.startswith("cat_collapsed:")
                              else "continuous"),
             "E": 1.0}
            for ch in ("ch_0", "ch_1", "cat_collapsed:sleep")
        ]
        result = compute_skill_scores(
            _build_errors_df(rows), _build_errors_df(bl_rows), mode="pooled",
        )
        task_rows = result[result["scope"].str.startswith("task:")]
        assert len(task_rows) == 3
        assert (task_rows["n_tasks"] == 1).all()
