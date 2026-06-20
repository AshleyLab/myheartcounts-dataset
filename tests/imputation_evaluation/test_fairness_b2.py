"""Tests for the fairness B.2 two-stage form.

The disparity-ratio fairness skill score uses the same category-balanced
two-stage aggregation as ``overall_binary_collapsed`` on the skill / rank
side. Each row pins one property of that estimand:

  - the bucket classifier maps channels to {activity, physiology, sleep,
    workouts} and drops per-channel binary rows
  - the per-attribute skill geomeans within each bucket first, then
    averages over buckets, so a method's score doesn't depend on how many
    individual channels live in each bucket
  - ``aggregate_fairness_skill_score`` strips the per-channel binary rows
    upstream (so the join with the baseline stays cheap) but the kernel
    is correct even if they sneak through (defense in depth)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from imputation_evaluation.evaluation.paper_metrics_core import (
    _per_attribute_skill_keyed,
    b2_bucket_for_channel,
    compute_fair_skill_scores,
)


class TestBucketClassifier:
    """``b2_bucket_for_channel`` channel→bucket assignment."""

    def test_continuous_channels_map_to_activity_or_physiology(self):
        """ch_0..ch_4 map to activity; ch_5/ch_6 map to physiology."""
        for ch in ("ch_0", "ch_1", "ch_2", "ch_3", "ch_4"):
            assert b2_bucket_for_channel(ch, "continuous") == "activity"
        for ch in ("ch_5", "ch_6"):
            assert b2_bucket_for_channel(ch, "continuous") == "physiology"

    def test_collapsed_binary_maps_to_sleep_workouts(self):
        """``cat_collapsed:sleep`` / ``cat_collapsed:workouts`` map to their bucket."""
        assert b2_bucket_for_channel("cat_collapsed:sleep", "binary_collapsed") == "sleep"
        assert b2_bucket_for_channel("cat_collapsed:workouts", "binary_collapsed") == "workouts"

    def test_per_channel_binary_maps_to_none(self):
        """Per-channel binary rows (ch_7..ch_18) map to None (not consumed).

        The sleep / workouts buckets are populated from cat_collapsed:* only.
        """
        for ch_idx in range(7, 19):
            assert b2_bucket_for_channel(f"ch_{ch_idx}", "binary") is None


# ---------------------------------------------------------------------------
# Per-attribute skill — synthetic 2-subgroup × 4-bucket fixture
# ---------------------------------------------------------------------------


def _make_fairness_fixture():
    """Build a two-method × two-subgroup × four-bucket fairness fixture.

    Designed so the per-task log_ratios per bucket are constant inside each
    bucket (so the bucket geomean is closed-form) and the four
    bucket-log-ratios are deliberately different (so cross-bucket averaging
    is visible).
    """
    rows = []

    # The fairness ratio = D_method / D_baseline. We control D_baseline = 1
    # for every task by setting baseline E to (0.5, 1.5) across subgroups
    # — max − min = 1.0. Then for each method-task we set E values so that
    # D_method is the target ratio directly.
    def baseline_pair():
        # subgroup_value -> baseline E so D_baseline = 1
        return {"f": 0.5, "m": 1.5}

    # method "A" disparity targets per bucket — chosen so log(R) values
    # are −log 2, −log 4, −log 8, −log 16 (very disparity-free in
    # activity, hyperfair in workouts):
    targets_A = {
        "activity": 0.5,
        "physiology": 0.25,
        "sleep": 0.125,
        "workouts": 0.0625,
    }
    # method "B" — uniform disparity ratio of 1.0 (matches baseline, S = 0)
    targets_B = {"activity": 1.0, "physiology": 1.0, "sleep": 1.0, "workouts": 1.0}

    def emit(method: str, targets: dict[str, float]):
        # For every task: D_method = target (set E="f"=0, E="m"=target).
        # The baseline rows (added later) use E="f"=0.5, E="m"=1.5 so
        # D_baseline = 1.0 — the per-task ratio is exactly ``target``.
        def E_for(sg: str, target: float) -> float:
            return 0.0 if sg == "f" else target

        # activity: ch_0..ch_4
        for c in range(5):
            for sg in ("f", "m"):
                rows.append(
                    dict(
                        method=method,
                        scenario="rn",
                        channel=f"ch_{c}",
                        channel_type="continuous",
                        subgroup_attr="sex",
                        subgroup_value=sg,
                        E=E_for(sg, targets["activity"]),
                    )
                )
        # physiology: ch_5..ch_6
        for c in (5, 6):
            for sg in ("f", "m"):
                rows.append(
                    dict(
                        method=method,
                        scenario="rn",
                        channel=f"ch_{c}",
                        channel_type="continuous",
                        subgroup_attr="sex",
                        subgroup_value=sg,
                        E=E_for(sg, targets["physiology"]),
                    )
                )
        # sleep: cat_collapsed:sleep
        for sg in ("f", "m"):
            rows.append(
                dict(
                    method=method,
                    scenario="rn",
                    channel="cat_collapsed:sleep",
                    channel_type="binary_collapsed",
                    subgroup_attr="sex",
                    subgroup_value=sg,
                    E=E_for(sg, targets["sleep"]),
                )
            )
        # workouts: cat_collapsed:workouts
        for sg in ("f", "m"):
            rows.append(
                dict(
                    method=method,
                    scenario="rn",
                    channel="cat_collapsed:workouts",
                    channel_type="binary_collapsed",
                    subgroup_attr="sex",
                    subgroup_value=sg,
                    E=E_for(sg, targets["workouts"]),
                )
            )

    emit("A", targets_A)
    emit("B", targets_B)
    # baseline LOCF: E = (0.5, 1.5) for every task
    for c in range(5):
        for sg, eb in baseline_pair().items():
            rows.append(
                dict(
                    method="LOCF",
                    scenario="rn",
                    channel=f"ch_{c}",
                    channel_type="continuous",
                    subgroup_attr="sex",
                    subgroup_value=sg,
                    E=eb,
                )
            )
    for c in (5, 6):
        for sg, eb in baseline_pair().items():
            rows.append(
                dict(
                    method="LOCF",
                    scenario="rn",
                    channel=f"ch_{c}",
                    channel_type="continuous",
                    subgroup_attr="sex",
                    subgroup_value=sg,
                    E=eb,
                )
            )
    for sg, eb in baseline_pair().items():
        rows.append(
            dict(
                method="LOCF",
                scenario="rn",
                channel="cat_collapsed:sleep",
                channel_type="binary_collapsed",
                subgroup_attr="sex",
                subgroup_value=sg,
                E=eb,
            )
        )
    for sg, eb in baseline_pair().items():
        rows.append(
            dict(
                method="LOCF",
                scenario="rn",
                channel="cat_collapsed:workouts",
                channel_type="binary_collapsed",
                subgroup_attr="sex",
                subgroup_value=sg,
                E=eb,
            )
        )
    return pd.DataFrame(rows)


class TestPerAttributeSkillTwoStage:
    """The two-stage per-attribute skill kernel ``_per_attribute_skill_keyed``."""

    def test_n_tasks_counts_buckets_not_tasks(self):
        """n_tasks counts buckets (4), not the 5 + 2 + 1 + 1 = 9 per-channel tasks.

        Under B.2 the per-attribute kernel reports n_tasks = 4 (bucket count).
        """
        df = _make_fairness_fixture()
        out = _per_attribute_skill_keyed(
            df,
            extra_keys=[],
            baseline_method="LOCF",
            clip_lower=1e-2,
            clip_upper=100.0,
        )
        row_A = out[out["method"] == "A"]
        assert len(row_A) == 1
        assert int(row_A["n_tasks"].iloc[0]) == 4

    def test_S_attr_equals_mean_of_bucket_log_ratios(self):
        """Closed-form: S_attr = 1 − exp(mean of the 4 bucket log ratios).

        Each bucket's per-task log ratios are identical inside the bucket
        (all 5 activity channels share the same target=0.5), so the
        bucket-level log ratio is just log(targets[bucket]). The
        per-attribute skill is then 1 - exp(mean of those 4 log values).
        """
        df = _make_fairness_fixture()
        out = _per_attribute_skill_keyed(
            df,
            extra_keys=[],
            baseline_method="LOCF",
            clip_lower=1e-2,
            clip_upper=100.0,
        )
        log_ratios = np.array([np.log(0.5), np.log(0.25), np.log(0.125), np.log(0.0625)])
        expected_S = 1.0 - np.exp(log_ratios.mean())
        got_S = float(out[out["method"] == "A"]["S_attr"].iloc[0])
        np.testing.assert_allclose(got_S, expected_S, rtol=1e-9)

    def test_per_channel_binary_rows_ignored(self):
        """Sneaking catastrophic ch_7..ch_18 rows in must NOT change S_attr.

        The bucket classifier maps per-channel binary rows to None, so they
        never enter the per-attribute skill.
        """
        df = _make_fairness_fixture()
        # 100x worse disparity on the per-channel binary tasks — would
        # crater the score if they leaked through.
        extra = []
        for ch_idx in range(7, 19):
            for sg in ("f", "m"):
                e_m = 0.0 if sg == "f" else 50.0
                e_b = 0.0 if sg == "f" else 1.0
                extra.append(
                    dict(
                        method="A",
                        scenario="rn",
                        channel=f"ch_{ch_idx}",
                        channel_type="binary",
                        subgroup_attr="sex",
                        subgroup_value=sg,
                        E=e_m,
                    )
                )
                extra.append(
                    dict(
                        method="LOCF",
                        scenario="rn",
                        channel=f"ch_{ch_idx}",
                        channel_type="binary",
                        subgroup_attr="sex",
                        subgroup_value=sg,
                        E=e_b,
                    )
                )
        df_aug = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
        baseline = float(
            _per_attribute_skill_keyed(
                df,
                extra_keys=[],
                baseline_method="LOCF",
                clip_lower=1e-2,
                clip_upper=100.0,
            )[lambda d: d["method"] == "A"]["S_attr"].iloc[0]
        )
        with_leak = float(
            _per_attribute_skill_keyed(
                df_aug,
                extra_keys=[],
                baseline_method="LOCF",
                clip_lower=1e-2,
                clip_upper=100.0,
            )[lambda d: d["method"] == "A"]["S_attr"].iloc[0]
        )
        np.testing.assert_allclose(with_leak, baseline, rtol=1e-9)


class TestComputeFairSkillScoresEndToEnd:
    """End-to-end check on ``compute_fair_skill_scores``.

    This is the deterministic point-flow entry that
    ``compute_imputation_paper_metrics`` calls.
    """

    def test_overall_macro_averages_attributes(self):
        """The ``overall`` scope is the macro-mean of the per-attribute scores."""
        df = _make_fairness_fixture()
        # Duplicate the fixture as another attribute so the overall macro
        # averages something non-trivial.
        df_age = df.copy()
        df_age["subgroup_attr"] = "age_group"
        df_age["subgroup_value"] = df_age["subgroup_value"].map({"f": "30-39", "m": "40-49"})
        big = pd.concat([df, df_age], ignore_index=True)
        out = compute_fair_skill_scores(
            big,
            attrs=["sex", "age_group"],
            baseline_method="LOCF",
        )
        a_row = out[(out["method"] == "A") & (out["scope"] == "sex")]
        b_row = out[(out["method"] == "A") & (out["scope"] == "age_group")]
        o_row = out[(out["method"] == "A") & (out["scope"] == "overall")]
        assert len(a_row) == len(b_row) == len(o_row) == 1
        # Macro: mean(S_sex, S_age_group). With the symmetric fixture they
        # should be equal — sanity check.
        np.testing.assert_allclose(
            float(o_row["fair_skill_score"].iloc[0]),
            (float(a_row["fair_skill_score"].iloc[0]) + float(b_row["fair_skill_score"].iloc[0]))
            / 2,
            rtol=1e-9,
        )
