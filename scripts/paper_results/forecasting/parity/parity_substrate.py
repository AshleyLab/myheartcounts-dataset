"""Forecasting substrate parity gate (Gate B + Gate C-pre).

Proves the canonical per-user-metrics substrate refactor is numerically inert:
the substrate-driven reducers reproduce the committed ``forecasting_bca_20260618``
CSVs (skill + rank + their bootstraps + fairness — the latter still under the
*current* max-min disparity). Run this BEFORE switching fairness to MAPD; a green
run means any later fairness delta is attributable to MAPD alone.

Checks
------
* **Gate B** — ``to_error_df``/``to_rank_user_df`` over the substrate equal the
  direct from-trees builds (``skill._build_error_table``,
  ``fair._build_error_table``, ``rank._build_{continuous,binary}_user_rows``),
  float64-exact.
* **Inert** — the substrate-path point reducers equal the from-trees point
  reducers (the refactor changes nothing), AND the from-trees reducers equal the
  committed CSVs (sanity pre-check: no code drift independent of the substrate).
* **Gate C-pre** — substrate-path reducers vs the committed CSVs:
  skill point (``compute_skill_score_tables``), rank point
  (``build_grouped_metric_rank_tables``), skill/rank bootstrap
  (``bootstrap_skill_rank``), fairness point + bootstrap
  (``bootstrap_fair_skill_score``).

The skill/rank bootstrap is a deterministic function of the per-user tables +
seed, so Gate B (identical tables) already implies identical draws; we still
recompute via the substrate and diff against the committed CSVs as the headline
bar.

Heavy: scans the 10 models' metric trees several times + a 1000-draw bootstrap.
Run via ``parity_substrate.sbatch`` on SLURM.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from forecasting_evaluation.metrics import fairness_skill_score_summary as fair  # noqa: E402
from forecasting_evaluation.metrics import grouped_metric_rank_summary as rank  # noqa: E402
from forecasting_evaluation.metrics import metric_spec as _spec  # noqa: E402
from forecasting_evaluation.metrics import skill_score_summary as skill  # noqa: E402
from forecasting_evaluation.metrics.bootstrap_fair_skill_score import (  # noqa: E402
    bootstrap_fair_skill_score,
)
from forecasting_evaluation.metrics.bootstrap_skill_rank import bootstrap_skill_rank  # noqa: E402
from forecasting_evaluation.metrics.per_user_errors import (  # noqa: E402
    build_per_user_metrics,
    read_per_user_metrics_parquet,
    to_error_df,
    to_rank_user_df,
    write_per_user_metrics_parquet,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("parity_substrate")

# Canonical run params (configs/paper/sweep_forecasting.yaml + the committed CSVs).
CONTINUOUS_METRICS = ["mae"]
BINARY_METRICS = ["auroc"]
BASELINE = "seasonal_naive"
CLIP_LOWER, CLIP_UPPER, MIN_PAIRS = 0.01, 100.0, 1
N_BOOT, SEED, CI_LEVEL = 1000, 42, 0.95
AGE_BINS = (18, 30, 40, 50, 60)
BINARY_GROUPS = [(name, tuple(idx)) for name, idx in _spec.BINARY_GROUPS]


class Report:
    """Accumulates per-check pass/fail with the worst observed diff."""

    def __init__(self) -> None:
        """Initialize an empty check log."""
        self.rows: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        """Record one check's outcome and log it."""
        self.rows.append((name, ok, detail))
        logger.info("[%s] %s %s", "PASS" if ok else "FAIL", name, detail)

    def ok(self) -> bool:
        """True iff every recorded check passed."""
        return all(ok for _, ok, _ in self.rows)

    def summary(self) -> str:
        """Render a multi-line PASS/FAIL summary block."""
        lines = ["", "=" * 72, "PARITY SUMMARY", "=" * 72]
        for name, ok, detail in self.rows:
            lines.append(f"  {'PASS' if ok else 'FAIL':4}  {name}  {detail}")
        lines.append("=" * 72)
        lines.append("ALL PASS" if self.ok() else "FAILURES PRESENT")
        return "\n".join(lines)


def _sort(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in keys:
        out[col] = out[col].astype(str)
    return out.sort_values(keys, kind="mergesort").reset_index(drop=True)


def _frames_equal(
    got: pd.DataFrame,
    exp: pd.DataFrame,
    *,
    keys: list[str],
    value_cols: list[str],
    atol: float,
    rtol: float,
) -> tuple[bool, str]:
    """Sorted row-set + value comparison; returns (ok, detail-with-max-abs-diff)."""
    g, e = _sort(got, keys), _sort(exp, keys)
    if len(g) != len(e):
        return False, f"row count {len(g)} != {len(e)}"
    key_mismatch = (g[keys].astype(str).to_numpy() != e[keys].astype(str).to_numpy()).any()
    if key_mismatch:
        return False, "key columns differ after sort (row-set mismatch)"
    worst = 0.0
    worst_col = ""
    for col in value_cols:
        gv = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
        ev = pd.to_numeric(e[col], errors="coerce").to_numpy(dtype=float)
        both_nan = np.isnan(gv) & np.isnan(ev)
        diff = np.abs(gv - ev)
        diff[both_nan] = 0.0
        if not np.allclose(gv, ev, atol=atol, rtol=rtol, equal_nan=True):
            bad = int(np.sum(~(np.isclose(gv, ev, atol=atol, rtol=rtol, equal_nan=True))))
            return False, f"col {col!r}: {bad} cells exceed tol, max|d|={np.nanmax(diff):.3e}"
        col_worst = float(np.nanmax(diff)) if diff.size else 0.0
        if col_worst > worst:
            worst, worst_col = col_worst, col
    return True, f"n={len(g)} max|d|={worst:.2e} ({worst_col})"


def _metric_groups() -> dict:
    return {
        "continuous": {"metrics": CONTINUOUS_METRICS, "channel_indices": _spec.CONTINUOUS_CHANNELS},
        "binary": {"metrics": BINARY_METRICS, "channel_indices": _spec.BINARY_CHANNELS},
    }


def run(summary_dir: Path, substrate_out: Path, *, atol, rtol, do_bootstrap, do_fairness) -> Report:
    """Execute all parity checks; return the Report."""
    import json

    rep = Report()
    models = json.loads((summary_dir / "skill_rank_models.json").read_text())
    models = models.get("models", models)
    models = {name: {"path": path, "display_name": name} for name, path in models.items()}
    logger.info("Loaded %d models from %s", len(models), summary_dir)

    # ---- Build + round-trip the substrate ----
    substrate = build_per_user_metrics(
        models=models,
        continuous_metrics=CONTINUOUS_METRICS,
        binary_metrics=BINARY_METRICS,
        continuous_channel_indices=_spec.CONTINUOUS_CHANNELS,
        binary_channel_indices=_spec.BINARY_CHANNELS,
    )
    logger.info("Built substrate: %d rows", len(substrate))
    write_per_user_metrics_parquet(
        substrate, substrate_out, meta={"run": summary_dir.name, "within_user_aggregation": "micro"}
    )
    substrate, _ = read_per_user_metrics_parquet(substrate_out)
    logger.info("Round-tripped substrate via %s", substrate_out)

    # ---- Gate B: substrate adapters == direct from-trees builds ----
    err_sub = to_error_df(substrate, user_col="unit_id")
    err_trees = skill._build_error_table(
        models=models, metric_groups=_metric_groups(),
        aggregation_unit="user", within_user_aggregation="micro",
    )
    rep.check("GateB.skill_error_df", *_frames_equal(
        err_sub, err_trees, keys=["model", "group", "metric", "channel_idx", "unit_id"],
        value_cols=["error", "n_values"], atol=atol, rtol=rtol,
    ))
    fair_sub = to_error_df(substrate, user_col="user_id")
    fair_trees = fair._build_error_table(
        models=models, continuous_metrics=CONTINUOUS_METRICS, binary_metrics=BINARY_METRICS,
        continuous_channel_indices=_spec.CONTINUOUS_CHANNELS,
        binary_channel_indices=_spec.BINARY_CHANNELS, within_user_aggregation="micro",
    )
    rep.check("GateB.fairness_error_df", *_frames_equal(
        fair_sub, fair_trees, keys=["model", "group", "metric", "channel_idx", "user_id"],
        value_cols=["error", "n_values"], atol=atol, rtol=rtol,
    ))
    rank_sub = to_rank_user_df(substrate, binary_groups=BINARY_GROUPS)
    rank_trees = pd.concat(
        [
            rank._build_continuous_user_rows(
                models=models, metrics=CONTINUOUS_METRICS,
                channel_indices=_spec.CONTINUOUS_CHANNELS, within_user_aggregation="micro",
            ),
            rank._build_binary_user_rows(
                models=models, metrics=BINARY_METRICS, groups=BINARY_GROUPS,
                within_user_aggregation="micro",
            ),
        ],
        ignore_index=True,
    )
    rep.check("GateB.rank_user_df", *_frames_equal(
        rank_sub, rank_trees,
        keys=["model", "scope_type", "scope", "metric", "channel_idx", "user_id"],
        value_cols=["metric_value", "n_values"], atol=atol, rtol=rtol,
    ))

    # ---- Gate C-pre point: skill ----
    common_skill = dict(
        models=models, baseline_model=BASELINE,
        continuous_metrics=CONTINUOUS_METRICS, binary_metrics=BINARY_METRICS,
        continuous_channel_indices=_spec.CONTINUOUS_CHANNELS,
        binary_channel_indices=_spec.BINARY_CHANNELS,
        clip_lower=CLIP_LOWER, clip_upper=CLIP_UPPER, min_pairs=MIN_PAIRS,
        aggregation_unit="user", within_user_aggregation="micro",
    )
    long_s, summ_s, _ = skill.compute_skill_score_tables(**common_skill, per_user_metrics=substrate)
    long_t, summ_t, _ = skill.compute_skill_score_tables(**common_skill)
    skill_long_vals = [
        "skill_score", "geometric_mean_ratio", "n_users", "n_pairs",
        "model_error_mean", "baseline_error_mean",
    ]
    rep.check("Inert.skill_long(sub==trees)", *_frames_equal(
        long_s, long_t, keys=["model", "group", "metric", "channel_idx"],
        value_cols=skill_long_vals, atol=atol, rtol=rtol,
    ))
    exp_long = pd.read_csv(summary_dir / "forecasting_skill_score_long.csv")
    rep.check("GateC.skill_long(sub==committed)", *_frames_equal(
        long_s, exp_long, keys=["model", "group", "metric", "channel_idx"],
        value_cols=skill_long_vals, atol=atol, rtol=rtol,
    ))
    exp_summ = pd.read_csv(summary_dir / "forecasting_skill_score_model_summary.csv")
    summ_vals = [c for c in exp_summ.columns if c not in ("model", "baseline_model")]
    rep.check("GateC.skill_model_summary(sub==committed)", *_frames_equal(
        summ_s, exp_summ, keys=["model"], value_cols=summ_vals, atol=atol, rtol=rtol,
    ))

    # ---- Gate C-pre point: rank ----
    common_rank = dict(
        models=models, continuous_metrics=CONTINUOUS_METRICS, binary_metrics=BINARY_METRICS,
        continuous_channel_indices=_spec.CONTINUOUS_CHANNELS, binary_groups=BINARY_GROUPS,
        within_user_aggregation="micro",
    )
    user_s, rlong_s, _ = rank.build_grouped_metric_rank_tables(
        **common_rank, per_user_metrics=substrate
    )
    exp_rlong = pd.read_csv(summary_dir / "forecasting_grouped_metric_rank_long.csv")
    rep.check("GateC.rank_long(sub==committed)", *_frames_equal(
        rlong_s, exp_rlong, keys=["model", "scope_type", "scope", "metric"],
        value_cols=["metric_mean", "rank", "n_users", "n_values", "rank_n_users"],
        atol=atol, rtol=rtol,
    ))
    exp_user = pd.read_csv(summary_dir / "forecasting_grouped_metric_rank_user_level_long.csv")
    rep.check("GateC.rank_user_level(sub==committed)", *_frames_equal(
        user_s, exp_user,
        keys=["model", "scope_type", "scope", "metric", "channel_idx", "user_id"],
        value_cols=["metric_value", "n_values"], atol=atol, rtol=rtol,
    ))

    # ---- Gate C-pre bootstrap: skill + rank ----
    if do_bootstrap:
        bt = bootstrap_skill_rank(
            models=models, baseline_model=BASELINE,
            continuous_metrics=CONTINUOUS_METRICS, binary_metrics=BINARY_METRICS,
            continuous_channel_indices=_spec.CONTINUOUS_CHANNELS,
            binary_channel_indices=_spec.BINARY_CHANNELS, binary_groups=BINARY_GROUPS,
            n_boot=N_BOOT, seed=SEED, ci_level=CI_LEVEL, within_user_aggregation="micro",
            bca_skill_rank=False, per_user_metrics=substrate,
        )
        exp_sb = pd.read_csv(summary_dir / "forecasting_skill_score_bootstrap.csv")
        rep.check("GateC.skill_bootstrap(sub==committed)", *_frames_equal(
            bt["skill_scores"], exp_sb, keys=["model", "scope"],
            value_cols=["mean", "se", "ci_lo", "ci_hi", "n_boot"], atol=atol, rtol=rtol,
        ))
        exp_rb = pd.read_csv(summary_dir / "forecasting_grouped_metric_rank_bootstrap.csv")
        rep.check("GateC.rank_bootstrap(sub==committed)", *_frames_equal(
            bt["avg_rankings"], exp_rb, keys=["model", "scope", "metric"],
            value_cols=["mean", "se", "ci_lo", "ci_hi", "n_boot"], atol=atol, rtol=rtol,
        ))

    # ---- Gate C-pre fairness: point + bootstrap (still max-min) ----
    if do_fairness:
        from labels.api import ENROLLMENT_PATH, LABELS_PATH

        if not LABELS_PATH or not ENROLLMENT_PATH:
            rep.check("GateC.fairness", False, "labels/enrollment unresolved (set MHC_DATA_DIR)")
        else:
            ft = bootstrap_fair_skill_score(
                models=models, baseline_model=BASELINE,
                continuous_metrics=CONTINUOUS_METRICS, binary_metrics=BINARY_METRICS,
                continuous_channel_indices=_spec.CONTINUOUS_CHANNELS,
                binary_channel_indices=_spec.BINARY_CHANNELS,
                labels_path=str(LABELS_PATH), enrollment_path=str(ENROLLMENT_PATH),
                age_bins=AGE_BINS, n_boot=N_BOOT, seed=SEED, ci_level=CI_LEVEL,
                within_user_aggregation="micro", bca=True, per_user_metrics=substrate,
            )
            exp_fp = pd.read_csv(summary_dir / "forecasting_fairness_skill_score.csv")
            rep.check("GateC.fairness_point(sub==committed)", *_frames_equal(
                ft["fairness_skill_scores_point"], exp_fp, keys=["model", "scope"],
                value_cols=["fair_skill_score", "n_tasks"], atol=atol, rtol=rtol,
            ))
            exp_fb = pd.read_csv(summary_dir / "forecasting_fairness_skill_score_bootstrap.csv")
            rep.check("GateC.fairness_bootstrap(sub==committed)", *_frames_equal(
                ft["fairness_skill_scores"], exp_fb, keys=["model", "scope"],
                value_cols=["mean", "se", "ci_lo", "ci_hi", "n_boot", "point", "bca_lo", "bca_hi"],
                atol=atol, rtol=rtol,
            ))

    return rep


def main() -> int:
    """CLI entry — see module docstring."""
    p = argparse.ArgumentParser(description="Forecasting substrate parity gate.")
    p.add_argument(
        "--summary-dir",
        type=Path,
        default=REPO_ROOT
        / "results/forecasting_eval/simurgh/summary/forecasting_bca_20260618",
    )
    p.add_argument("--substrate-out", type=Path, default=None)
    p.add_argument("--atol", type=float, default=1e-9)
    p.add_argument("--rtol", type=float, default=1e-9)
    p.add_argument("--skip-bootstrap", action="store_true")
    p.add_argument("--skip-fairness", action="store_true")
    args = p.parse_args()

    substrate_out = args.substrate_out or (args.summary_dir / "forecasting_per_user_errors.parquet")
    rep = run(
        args.summary_dir, substrate_out,
        atol=args.atol, rtol=args.rtol,
        do_bootstrap=not args.skip_bootstrap, do_fairness=not args.skip_fairness,
    )
    print(rep.summary())
    return 0 if rep.ok() else 1


if __name__ == "__main__":
    sys.exit(main())
