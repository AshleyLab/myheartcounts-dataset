"""Track-1 leaderboard substrate — per-user prediction pairs (supervisor ask + #39).

Builds the substrate from synthetic per-(method, task) prediction files, so it
needs no dataset. Verifies the schema, subgroup expansion, and parquet round-trip.
"""

from __future__ import annotations

import json

import pandas as pd

from downstream_evaluation.evaluation.per_user_pairs import (
    PER_USER_PAIRS_PARQUET_COLUMNS,
    build_per_user_pairs,
    read_per_user_pairs_parquet,
    write_per_user_pairs_parquet,
)
from downstream_evaluation.evaluation.predictions_io import _safe_task


def _write_task(pred_dir, method, task, df):
    d = pred_dir / method / _safe_task(task)
    d.mkdir(parents=True, exist_ok=True)
    df.to_parquet(d / "test.parquet", index=False)


def test_build_per_user_pairs_schema_and_subgroups(tmp_path):
    method = "linear"
    _write_task(
        tmp_path,
        method,
        "Diabetes",
        pd.DataFrame(
            {"uid": ["u0", "u1"], "y_true": [0, 1], "y_pred": [0, 1], "y_proba": [0.2, 0.8]}
        ),
    )
    _write_task(
        tmp_path,
        method,
        "age",
        pd.DataFrame(
            {
                "uid": ["u0", "u1"],
                "y_true": [30.0, 55.0],
                "y_pred": [31.0, 50.0],
                "y_proba": [31.0, 50.0],
            }
        ),
    )
    (tmp_path / "_subgroups.json").write_text(
        json.dumps(
            {
                "u0": {"age_group": "18-29", "sex": "male"},
                "u1": {"age_group": "50-59", "sex": "female"},
            }
        )
    )

    df = build_per_user_pairs(tmp_path, method, ["Diabetes", "age"])

    assert list(df.columns) == PER_USER_PAIRS_PARQUET_COLUMNS
    assert set(df["method"]) == {"linear"}
    assert set(df["subgroup_attr"]) == {"all", "age_group", "sex"}
    # 2 tasks × 2 users × 3 subgroup axes = 12 rows.
    assert len(df) == 12
    # The global cell is labelled all/all.
    assert set(df[df.subgroup_attr == "all"]["subgroup_value"]) == {"all"}
    # Subgroup values are looked up per user.
    assert set(df[(df.subgroup_attr == "age_group") & (df.user_id == "u0")]["subgroup_value"]) == {
        "18-29"
    }
    assert set(df[(df.subgroup_attr == "sex") & (df.user_id == "u1")]["subgroup_value"]) == {
        "female"
    }
    # task_type is derived from the task name.
    assert set(df[df.task == "Diabetes"]["task_type"]) == {"binary"}
    assert set(df[df.task == "age"]["task_type"]) == {"regression"}


def test_build_per_user_pairs_unknown_subgroup(tmp_path):
    _write_task(
        tmp_path,
        "m",
        "Diabetes",
        pd.DataFrame({"uid": ["uX"], "y_true": [1], "y_pred": [1], "y_proba": [0.9]}),
    )
    (tmp_path / "_subgroups.json").write_text(json.dumps({}))

    df = build_per_user_pairs(tmp_path, "m", ["Diabetes"])

    # A user absent from the subgroup map falls into 'unknown' on every axis.
    assert set(df[df.subgroup_attr == "sex"]["subgroup_value"]) == {"unknown"}
    assert set(df[df.subgroup_attr == "age_group"]["subgroup_value"]) == {"unknown"}


def test_missing_task_files_are_skipped(tmp_path):
    _write_task(
        tmp_path,
        "m",
        "Diabetes",
        pd.DataFrame({"uid": ["u0"], "y_true": [1], "y_pred": [1], "y_proba": [0.9]}),
    )
    (tmp_path / "_subgroups.json").write_text(
        json.dumps({"u0": {"age_group": "18-29", "sex": "male"}})
    )

    # "age" has no file → silently skipped, only Diabetes rows emitted.
    df = build_per_user_pairs(tmp_path, "m", ["Diabetes", "age"])
    assert set(df["task"]) == {"Diabetes"}


def test_write_read_roundtrip_and_meta(tmp_path):
    _write_task(
        tmp_path,
        "m",
        "Diabetes",
        pd.DataFrame({"uid": ["u0"], "y_true": [1], "y_pred": [1], "y_proba": [0.9]}),
    )
    (tmp_path / "_subgroups.json").write_text(
        json.dumps({"u0": {"age_group": "18-29", "sex": "male"}})
    )
    df = build_per_user_pairs(tmp_path, "m", ["Diabetes"])

    p = tmp_path / "m.parquet"
    write_per_user_pairs_parquet(df, p, meta={"method": "m", "overall_fallback_rate": 0.0})

    back, meta = read_per_user_pairs_parquet(p)
    assert list(back.columns) == PER_USER_PAIRS_PARQUET_COLUMNS
    assert meta["method"] == "m"
    assert meta["overall_fallback_rate"] == 0.0
    assert str(back["y_proba"].dtype) == "float32"
    assert str(back["method"].dtype) == "category"
