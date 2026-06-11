r"""Paired user-level bootstrap of the headline downstream benchmark stats (library).

Two-phase, mirroring the imputation paper-metrics pipeline:

* Phase 1 — :func:`compute_per_draw_errors` draws B paired bootstrap resamples on
  the per-(method, task) prediction parquets (for each task, sample N test users
  with replacement, the **same** indices reused across methods → paired
  comparisons), recomputes every method's per-task primary metric, and records the
  error ``E = 1 − metric`` globally and per demographic subgroup as a long-format
  draws frame (:func:`write_draws_parquet` persists it).

* Phase 2 — :func:`aggregate_skill_rank_fairness` reconstructs each draw's per-task
  metric (``1 − E``) and reduces the draws to ``mean / SE / 95 % CI`` for the macro
  (domain-balanced) **skill score** vs the baseline, the **average rank** (both
  Overall + per-domain), and the **fairness** tables (per-subgroup skill, disparity,
  fairness-adjusted skill).

The runnable CLIs are ``scripts/paper_results/bootstrap_downstream_draws.py`` (phase
1) and ``scripts/paper_results/aggregate_downstream_paper_metrics.py`` (phase 2).
Fairness rows require ``predictions_dir/_subgroups.json`` (per-user {age_group, sex});
without it only the global (``subgroup_attr="all"``) rows are produced.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score

from downstream_evaluation.evaluation.skill_score import (
    DEFAULT_CLIP_LOWER,
    DEFAULT_CLIP_UPPER,
    TASK_DOMAIN_MAP,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def _auprc(y_true: np.ndarray, _pred: np.ndarray, y_proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_proba))


def _spearman(y_true: np.ndarray, y_pred: np.ndarray, _proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return float("nan")
    r, _ = spearmanr(y_true, y_pred)
    return float(r) if np.isfinite(r) else float("nan")


def _pearson(y_true: np.ndarray, y_pred: np.ndarray, _proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return float("nan")
    r, _ = pearsonr(y_true, y_pred)
    return float(r) if np.isfinite(r) else float("nan")


# Point metrics only — bypasses compute_*_metrics' internal 1000-iter SE
# bootstrap, which would otherwise turn each outer resample into 1000 nested
# resamples. Defaults are higher-is-better → error = 1 - metric.
PRIMARY_METRIC_FN = {
    "binary": _auprc,
    "ordinal": _spearman,
    "regression": _pearson,
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_method_predictions(
    predictions_dir: Path,
    method: str,
    csvs_dir: Path,
) -> dict[str, dict]:
    """Return {task: {uids, y_true, y_pred, y_proba, task_type}} for one method."""
    # Prefer `eval_*<method>*.csv` (the production naming); fall back to any CSV
    # containing the method name (used by the smoke test CSV `smoke.csv` etc.).
    csv_candidates = sorted(csvs_dir.glob(f"eval_*{method}*.csv"))
    if not csv_candidates:
        csv_candidates = sorted(csvs_dir.glob(f"*{method}*.csv"))
    if not csv_candidates:
        # Last resort: any CSV in the dir whose `features` column contains
        # the method label (e.g. `smoke.csv` was written without the method
        # name in the file name).
        for path in sorted(csvs_dir.glob("*.csv")):
            try:
                head = pd.read_csv(path, nrows=5)
            except Exception:
                continue
            if "features" in head.columns and (head["features"] == method).any():
                csv_candidates = [path]
                break
    if not csv_candidates:
        raise FileNotFoundError(f"No eval CSV matching '{method}' in {csvs_dir}")
    if len(csv_candidates) > 1:
        log.warning(
            "Multiple eval CSVs match '%s': %s — using %s",
            method,
            [p.name for p in csv_candidates],
            csv_candidates[0].name,
        )
    csv = pd.read_csv(csv_candidates[0])
    if "subgroup_variable" in csv.columns:
        csv = csv[csv["subgroup_variable"].fillna("") == ""]
    if "error" in csv.columns:
        csv = csv[csv["error"].fillna("") == ""]
    task_type_map = dict(zip(csv["task"], csv["task_type"]))

    method_dir = predictions_dir / method
    if not method_dir.exists():
        raise FileNotFoundError(f"No predictions dir for method '{method}' at {method_dir}")

    tasks: dict[str, dict] = {}
    for task_dir in sorted(method_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        parquet = task_dir / "test.parquet"
        if not parquet.exists():
            continue
        df = pd.read_parquet(parquet)
        candidates = [
            t for t in task_type_map if t.replace("/", "_").replace(" ", "_") == task_dir.name
        ]
        if not candidates:
            log.warning("No CSV task_type for parquet dir '%s' — skipping", task_dir.name)
            continue
        task_name = candidates[0]
        tasks[task_name] = {
            "uids": df["uid"].to_numpy(),
            "y_true": df["y_true"].to_numpy(),
            "y_pred": df["y_pred"].to_numpy(),
            "y_proba": df["y_proba"].to_numpy(),
            "task_type": task_type_map[task_name],
        }
    log.info("Loaded %d tasks for method '%s'", len(tasks), method)
    return tasks


def align_across_methods(
    method_tasks: dict[str, dict[str, dict]],
) -> dict[str, dict[str, dict]]:
    """Per task, intersect uids across methods and reorder to a shared sequence."""
    methods = list(method_tasks.keys())
    common_tasks = set(method_tasks[methods[0]].keys())
    for m in methods[1:]:
        common_tasks &= set(method_tasks[m].keys())

    aligned: dict[str, dict[str, dict]] = {m: {} for m in methods}
    for task in sorted(common_tasks):
        uid_sets = [set(method_tasks[m][task]["uids"]) for m in methods]
        common_uids = set.intersection(*uid_sets)
        if not common_uids:
            log.warning("Task '%s': empty uid intersection across methods — skipping", task)
            continue
        uid_order = np.array(sorted(common_uids))
        for m in methods:
            payload = method_tasks[m][task]
            uid_to_idx = {uid: i for i, uid in enumerate(payload["uids"])}
            idx = np.array([uid_to_idx[u] for u in uid_order])
            aligned[m][task] = {
                "uids": uid_order,
                "y_true": payload["y_true"][idx],
                "y_pred": payload["y_pred"][idx],
                "y_proba": payload["y_proba"][idx],
                "task_type": payload["task_type"],
            }
        ref_y = aligned[methods[0]][task]["y_true"]
        for m in methods[1:]:
            if not np.array_equal(aligned[m][task]["y_true"], ref_y):
                log.warning(
                    "Task '%s': y_true mismatch between '%s' and '%s' on shared uids",
                    task,
                    methods[0],
                    m,
                )
    return aligned


def load_subgroup_map(predictions_dir: Path) -> dict[str, dict[str, str]] | None:
    """Load ``predictions_dir/_subgroups.json`` ({user_id: {attr: value}}), or None if absent."""
    path = predictions_dir / "_subgroups.json"
    if not path.exists():
        log.warning("No subgroup map at %s — fairness stats will be NaN", path)
        return None
    with path.open() as f:
        sg = json.load(f)
    log.info("Loaded subgroup map for %d users from %s", len(sg), path)
    return sg


# ---------------------------------------------------------------------------
# Aggregation helpers (operate on per-resample dicts)
# ---------------------------------------------------------------------------


def _per_domain_skill_from_ratios(
    ratios: dict[str, float],
    domain_map: dict[str, str],
    clip_lower: float,
    clip_upper: float,
) -> dict[str, float]:
    """Skill score per domain ``+`` Overall.

    Per-domain S = 1 − geomean(clipped ratios for tasks in that domain).
    Overall S = mean of per-domain S (domain-balanced macro aggregate).
    """
    by_domain: dict[str, list[float]] = {}
    for task, r in ratios.items():
        if not np.isfinite(r):
            continue
        domain = domain_map.get(task)
        if domain is None:
            continue
        by_domain.setdefault(domain, []).append(np.clip(r, clip_lower, clip_upper))
    out: dict[str, float] = {}
    for domain, rs in by_domain.items():
        out[domain] = 1.0 - float(np.exp(np.mean(np.log(rs))))
    out["Overall"] = float(np.mean(list(out.values()))) if out else float("nan")
    return out


def _macro_skill_from_ratios(
    ratios: dict[str, float],
    domain_map: dict[str, str],
    clip_lower: float,
    clip_upper: float,
) -> float:
    """Macro (domain-balanced) skill score = ``Overall`` from the per-domain dict."""
    return _per_domain_skill_from_ratios(ratios, domain_map, clip_lower, clip_upper)["Overall"]


def _ratios_for_method(
    per_task_metric: dict[str, dict[str, float]],
    method: str,
    baseline: str,
) -> dict[str, float]:
    """E_method / E_baseline per task (errors = 1 − metric, both higher-is-better)."""
    out: dict[str, float] = {}
    for task, vals in per_task_metric.items():
        m_val = vals.get(method)
        b_val = vals.get(baseline)
        if m_val is None or b_val is None:
            continue
        m_err = 1.0 - m_val
        b_err = 1.0 - b_val
        if not (np.isfinite(m_err) and np.isfinite(b_err) and b_err > 0):
            continue
        out[task] = m_err / b_err
    return out


def _ratios_subgroup_vs_global_baseline(
    sub_per_task: dict[str, dict[str, float]],
    global_per_task: dict[str, dict[str, float]],
    method: str,
    baseline: str,
) -> dict[str, float]:
    """Per-subgroup E_method / *global* E_baseline per task.

    Mirrors compute_fairness_adjusted_score in skill_score.py: the subgroup-S
    numerator uses the method's per-subgroup error, but the denominator stays
    on the baseline's *global* error so every subgroup's S is on the same
    yardstick. This is what makes the baseline's per-subgroup S non-zero
    (proportional to how much the baseline's subgroup performance differs
    from its overall performance), and therefore Linear's disparity / FairS
    are non-trivial: FairS_baseline = 0 − λ · D̄_baseline.
    """
    out: dict[str, float] = {}
    for task, vals in sub_per_task.items():
        m_val = vals.get(method)
        b_val = global_per_task.get(task, {}).get(baseline)
        if m_val is None or b_val is None:
            continue
        m_err = 1.0 - m_val
        b_err = 1.0 - b_val
        if not (np.isfinite(m_err) and np.isfinite(b_err) and b_err > 0):
            continue
        out[task] = m_err / b_err
    return out


def _per_domain_avg_rank(
    per_task_metric: dict[str, dict[str, float]],
    methods: list[str],
    domain_map: dict[str, str],
) -> dict[str, dict[str, float]]:
    """Per-domain + Overall average rank.

    Returns ``{domain_or_'Overall': {method: avg_rank}}``. Per-domain rank =
    mean of per-task ranks within that domain. Overall = mean of per-domain
    ranks (domain-balanced macro aggregate).
    """
    by_domain: dict[str, dict[str, list[float]]] = {}
    for task, method_vals in per_task_metric.items():
        domain = domain_map.get(task)
        if domain is None:
            continue
        # Series.rank handles ties via "average"; negate for descending
        # (higher metric = rank 1).
        ranks = pd.Series({m: -method_vals.get(m, np.nan) for m in methods}).rank(
            method="average",
            ascending=True,
        )
        for m, r in ranks.items():
            if pd.notna(r):
                by_domain.setdefault(domain, {}).setdefault(m, []).append(float(r))

    out: dict[str, dict[str, float]] = {}
    for domain, per_method in by_domain.items():
        out[domain] = {
            m: float(np.mean(per_method[m])) if m in per_method else float("nan") for m in methods
        }
    if out:
        out["Overall"] = {
            m: float(np.mean([out[d][m] for d in out if not np.isnan(out[d][m])]))
            if any(not np.isnan(out[d][m]) for d in out)
            else float("nan")
            for m in methods
        }
    else:
        out["Overall"] = {m: float("nan") for m in methods}
    return out


# ---------------------------------------------------------------------------
# Bootstrap loop
# ---------------------------------------------------------------------------


def _per_task_metric_on_indices(
    aligned: dict[str, dict[str, dict]],
    methods: list[str],
    tasks: list[str],
    task_indices_b: dict[str, np.ndarray],
    mask_per_task: dict[str, np.ndarray] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute per-task per-method metric on the resampled rows.

    If ``mask_per_task[t]`` is provided (a boolean mask indexing into the
    resampled rows), only those rows are used — used for per-subgroup metrics.
    Tasks with fewer than 2 samples after masking are skipped silently.
    """
    out: dict[str, dict[str, float]] = {}
    for t in tasks:
        idx = task_indices_b[t]
        if mask_per_task is not None:
            mask = mask_per_task.get(t)
            if mask is None or mask.sum() < 2:
                continue
            idx = idx[mask]
        tt = aligned[methods[0]][t]["task_type"]
        fn = PRIMARY_METRIC_FN.get(tt)
        if fn is None:
            continue
        out[t] = {}
        for m in methods:
            p = aligned[m][t]
            y = p["y_true"][idx]
            pred = p["y_pred"][idx]
            proba = p["y_proba"][idx]
            try:
                v = float(fn(y, pred, proba))
            except Exception:
                v = float("nan")
            out[t][m] = v
    return out


def _build_subgroup_masks(
    aligned: dict[str, dict[str, dict]],
    tasks: list[str],
    methods: list[str],
    subgroup_map: dict[str, dict[str, str]],
    attributes: list[str],
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Pre-compute {task: {attribute: {value: mask_into_uid_order}}}.

    Mask is over the canonical uid order (same for all methods after alignment).
    Reused inside the bootstrap loop with the resampled indices.
    """
    out: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for t in tasks:
        uids = aligned[methods[0]][t]["uids"]
        out[t] = {}
        for attr in attributes:
            attr_per_uid = np.array([subgroup_map.get(u, {}).get(attr, "unknown") for u in uids])
            values = sorted(set(attr_per_uid.tolist()))
            out[t][attr] = {v: (attr_per_uid == v) for v in values}
    return out


# ---------------------------------------------------------------------------
# Two-phase pipeline: per-draw errors (phase 1) -> skill/rank/fairness (phase 2)
# ---------------------------------------------------------------------------

DRAW_COLS = [
    "method",
    "task",
    "task_type",
    "domain",
    "subgroup_attr",
    "subgroup_value",
    "draw",
    "E",
]

# Sentinel ``draw`` id for the point estimate — the full cohort, no resampling. The
# reported value is this point estimate; the bootstrap draws (``draw >= 0``) give only
# the standard error / CI around it (we never report the bootstrap mean as the value).
POINT_DRAW = -1


def compute_per_draw_errors(
    aligned: dict[str, dict[str, dict]],
    n_bootstrap: int,
    seed: int,
    subgroup_map: dict[str, dict[str, str]] | None = None,
    subgroup_attributes: list[str] | None = None,
    min_subgroup_size: int = 10,
    domain_map: dict[str, str] = TASK_DOMAIN_MAP,
) -> pd.DataFrame:
    """Phase 1: per-(method, task, subgroup, draw) error ``E = 1 - metric``.

    Draws B paired bootstrap resamples (same per-task user indices reused across
    methods) and records each method's per-task primary-metric error, globally
    (``subgroup_attr="all"``) and per demographic subgroup. The long-format frame
    (columns :data:`DRAW_COLS`) is the phase-2 input.
    """
    methods = list(aligned.keys())
    tasks = sorted(aligned[methods[0]].keys())
    task_type = {t: aligned[methods[0]][t]["task_type"] for t in tasks}
    rng = np.random.default_rng(seed)
    task_n = {t: len(aligned[methods[0]][t]["uids"]) for t in tasks}
    task_indices = {t: rng.integers(0, task_n[t], size=(n_bootstrap, task_n[t])) for t in tasks}

    do_fairness = subgroup_map is not None and subgroup_attributes
    attribute_masks = (
        _build_subgroup_masks(aligned, tasks, methods, subgroup_map, subgroup_attributes)
        if do_fairness
        else {}
    )

    rows: list[dict] = []

    def _emit(per_task: dict[str, dict[str, float]], attr: str, value: str, b: int) -> None:
        for t, method_vals in per_task.items():
            dom = domain_map.get(t)
            for m, metric in method_vals.items():
                if metric is None or not np.isfinite(metric):
                    continue
                rows.append(
                    {
                        "method": m,
                        "task": t,
                        "task_type": task_type[t],
                        "domain": dom,
                        "subgroup_attr": attr,
                        "subgroup_value": value,
                        "draw": b,
                        "E": 1.0 - float(metric),
                    }
                )

    def _emit_for_indices(idx_b: dict[str, np.ndarray], b: int) -> None:
        """Emit global + per-subgroup errors for one set of row indices (draw ``b``)."""
        per_task_global = _per_task_metric_on_indices(aligned, methods, tasks, idx_b)
        _emit(per_task_global, "all", "all", b)
        if do_fairness:
            for attr in subgroup_attributes:
                values = sorted({v for t in tasks for v in attribute_masks[t][attr]})
                for value in values:
                    masks_b: dict[str, np.ndarray] = {}
                    for t in tasks:
                        cm = attribute_masks[t][attr].get(value)
                        if cm is None:
                            continue
                        m_b = cm[idx_b[t]]
                        if m_b.sum() < min_subgroup_size:
                            continue
                        masks_b[t] = m_b
                    if not masks_b:
                        continue
                    sub_per_task = _per_task_metric_on_indices(
                        aligned, methods, tasks, idx_b, mask_per_task=masks_b
                    )
                    _emit(sub_per_task, attr, value, b)

    # Point estimate first — the full cohort in order, no resampling — then the B
    # paired bootstrap resamples that quantify its standard error.
    _emit_for_indices({t: np.arange(task_n[t]) for t in tasks}, POINT_DRAW)
    for b in range(n_bootstrap):
        _emit_for_indices({t: task_indices[t][b] for t in tasks}, b)

    return pd.DataFrame(rows, columns=DRAW_COLS)


def write_draws_parquet(df: pd.DataFrame, path: Path, meta: dict | None = None) -> None:
    """Write the phase-1 draws frame to parquet (zstd) + optional sidecar meta JSON."""
    df = df.copy()
    df["E"] = df["E"].astype("float32")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="zstd")
    if meta is not None:
        path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))


def read_draws_parquet(path: Path) -> tuple[pd.DataFrame, dict | None]:
    """Read a phase-1 draws parquet and its sidecar meta JSON (if present)."""
    df = pd.read_parquet(path)
    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    return df, meta


def _summarise(values: list[float], ci_level: float, point: float | None = None) -> dict[str, float]:
    """Point estimate + SE / percentile-CI from the bootstrap draws of one quantity.

    ``values`` are the bootstrap draws (the point draw excluded). ``point`` is the
    full-cohort estimate that is reported as the value; SE and the percentile CI come
    from the bootstrap draws. When ``point`` is omitted the bootstrap mean is the
    fallback value (used only where no point estimate is available).
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    center = (
        float(point)
        if point is not None and np.isfinite(point)
        else (float(np.mean(arr)) if len(arr) else float("nan"))
    )
    if len(arr) == 0:
        return {"point": center, "se": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    alpha = (1.0 - ci_level) / 2.0
    return {
        "point": center,
        "se": float(np.std(arr, ddof=1)) if len(arr) > 1 else float("nan"),
        "ci_lo": float(np.percentile(arr, 100 * alpha)),
        "ci_hi": float(np.percentile(arr, 100 * (1 - alpha))),
    }


def aggregate_skill_rank_fairness(
    draws: pd.DataFrame,
    baseline: str,
    *,
    clip_lower: float = DEFAULT_CLIP_LOWER,
    clip_upper: float = DEFAULT_CLIP_UPPER,
    lambda_fairness: float = 0.5,
    disparity_fns: dict | None = None,
    fairness_combine_name: str = "linear_penalty",
    domain_map: dict[str, str] = TASK_DOMAIN_MAP,
    ci_level: float = 0.95,
) -> dict[str, pd.DataFrame]:
    """Phase 2: summarise per-draw errors into skill / rank / fairness tables.

    Reconstructs each draw's per-task metric (``1 - E``) and reuses the point
    helpers, so per-draw skill/rank/fairness match the one-pass bootstrap exactly.
    Returns ``{skill_scores, avg_rankings, fairness_subgroup_scores,
    fairness_summary}`` (mean / se / ci at ``ci_level``).
    """
    from downstream_evaluation.evaluation.disparity_metrics import (
        DISPARITY_FUNCTIONS,
        FAIRNESS_COMBINE,
    )

    if disparity_fns is None:
        disparity_fns = {"max_minus_min": DISPARITY_FUNCTIONS["max_minus_min"].fn}
    combine_fn = FAIRNESS_COMBINE[fairness_combine_name]

    methods = sorted(draws["method"].unique())
    glob = draws[draws["subgroup_attr"] == "all"]
    sub = draws[draws["subgroup_attr"] != "all"]
    n_boot = int(draws.loc[draws["draw"] != POINT_DRAW, "draw"].nunique())

    task_dom = dict(zip(glob["task"], glob["domain"]))
    domains = sorted({d for d in task_dom.values() if d is not None})
    n_tasks = {"Overall": len({t for t, d in task_dom.items() if d is not None})}
    for dom in domains:
        n_tasks[dom] = len({t for t, d in task_dom.items() if d == dom})

    skill_draws: dict[str, dict[str, list[float]]] = {m: {} for m in methods}
    rank_draws: dict[str, dict[str, list[float]]] = {m: {} for m in methods}
    subS_draws: dict[str, dict[tuple[str, str], list[float]]] = {m: {} for m in methods}
    fair_draws: dict[str, dict[str, list[float]]] = {
        m: {
            "S_overall": [],
            **{f"disparity_{n}": [] for n in disparity_fns},
            **{f"fairness_adjusted_{n}": [] for n in disparity_fns},
        }
        for m in methods
    }
    # Point estimates (full cohort, draw == POINT_DRAW) — the reported values.
    skill_point: dict[str, dict[str, float]] = {m: {} for m in methods}
    rank_point: dict[str, dict[str, float]] = {m: {} for m in methods}
    subS_point: dict[str, dict[tuple[str, str], float]] = {m: {} for m in methods}
    fair_point: dict[str, dict[str, float]] = {m: {} for m in methods}

    for b, g in glob.groupby("draw"):
        is_point = b == POINT_DRAW
        per_task_metric: dict[str, dict[str, float]] = {}
        for t, m_, e_ in zip(g["task"], g["method"], g["E"]):
            per_task_metric.setdefault(t, {})[m_] = 1.0 - e_
        ranks_b = _per_domain_avg_rank(per_task_metric, methods, domain_map)
        cur_overall: dict[str, float] = {}
        for m in methods:
            skill = _per_domain_skill_from_ratios(
                _ratios_for_method(per_task_metric, m, baseline),
                domain_map,
                clip_lower,
                clip_upper,
            )
            cur_overall[m] = skill.get("Overall", float("nan"))
            for scope, val in skill.items():
                if is_point:
                    skill_point[m][scope] = val
                else:
                    skill_draws[m].setdefault(scope, []).append(val)
            for scope, per_m in ranks_b.items():
                if is_point:
                    rank_point[m][scope] = per_m[m]
                else:
                    rank_draws[m].setdefault(scope, []).append(per_m[m])

        if sub.empty:
            continue
        sub_b = sub[sub["draw"] == b]
        subgroup_S: dict[str, dict[str, dict[str, float]]] = {m: {} for m in methods}
        for (attr, value), gv in sub_b.groupby(["subgroup_attr", "subgroup_value"]):
            sub_per_task: dict[str, dict[str, float]] = {}
            for t, m_, e_ in zip(gv["task"], gv["method"], gv["E"]):
                sub_per_task.setdefault(t, {})[m_] = 1.0 - e_
            for m in methods:
                s_sub = _macro_skill_from_ratios(
                    _ratios_subgroup_vs_global_baseline(sub_per_task, per_task_metric, m, baseline),
                    domain_map,
                    clip_lower,
                    clip_upper,
                )
                subgroup_S[m].setdefault(attr, {})[value] = s_sub
                if is_point:
                    subS_point[m][(attr, value)] = s_sub
                else:
                    subS_draws[m].setdefault((attr, value), []).append(s_sub)
        for m in methods:
            s_overall = cur_overall[m]
            if is_point:
                fair_point[m]["S_overall"] = s_overall
            else:
                fair_draws[m]["S_overall"].append(s_overall)
            for dname, dfn in disparity_fns.items():
                per_attr = [dfn(vals) for vals in subgroup_S[m].values() if vals]
                per_attr = [d for d in per_attr if np.isfinite(d)]
                mean_disp = float(np.mean(per_attr)) if per_attr else 0.0
                fa = combine_fn(s_overall, mean_disp, lambda_fairness)
                if is_point:
                    fair_point[m][f"disparity_{dname}"] = mean_disp
                    fair_point[m][f"fairness_adjusted_{dname}"] = fa
                else:
                    fair_draws[m][f"disparity_{dname}"].append(mean_disp)
                    fair_draws[m][f"fairness_adjusted_{dname}"].append(fa)

    skill_rows, rank_rows = [], []
    for m in methods:
        for scope in ["Overall", *domains]:
            if scope in skill_draws[m]:
                skill_rows.append(
                    {
                        "method": m,
                        "scope": scope,
                        **_summarise(skill_draws[m][scope], ci_level, skill_point[m].get(scope)),
                        "n_boot": n_boot,
                        "n_tasks": n_tasks.get(scope, 0),
                    }
                )
            if scope in rank_draws[m]:
                rank_rows.append(
                    {
                        "method": m,
                        "scope": scope,
                        **_summarise(rank_draws[m][scope], ci_level, rank_point[m].get(scope)),
                        "n_boot": n_boot,
                    }
                )

    subgroup_rows = []
    for m in methods:
        for (attr, value), vals in subS_draws[m].items():
            subgroup_rows.append(
                {
                    "method": m,
                    "demographic_attr": attr,
                    "subgroup": value,
                    **_summarise(vals, ci_level, subS_point[m].get((attr, value))),
                    "n_boot": n_boot,
                }
            )

    summary_rows = []
    for m in methods:
        row = {"method": m}
        row.update(
            {
                f"S_overall_{k}": v
                for k, v in _summarise(
                    fair_draws[m]["S_overall"], ci_level, fair_point[m].get("S_overall")
                ).items()
            }
        )
        for dname in disparity_fns:
            row.update(
                {
                    f"disparity_{dname}_{k}": v
                    for k, v in _summarise(
                        fair_draws[m][f"disparity_{dname}"],
                        ci_level,
                        fair_point[m].get(f"disparity_{dname}"),
                    ).items()
                }
            )
            row.update(
                {
                    f"fairness_adjusted_{dname}_{k}": v
                    for k, v in _summarise(
                        fair_draws[m][f"fairness_adjusted_{dname}"],
                        ci_level,
                        fair_point[m].get(f"fairness_adjusted_{dname}"),
                    ).items()
                }
            )
        row["lambda"] = lambda_fairness
        row["fairness_combine"] = fairness_combine_name
        row["n_boot"] = n_boot
        summary_rows.append(row)

    return {
        "skill_scores": pd.DataFrame(skill_rows),
        "avg_rankings": pd.DataFrame(rank_rows),
        "fairness_subgroup_scores": pd.DataFrame(subgroup_rows),
        "fairness_summary": pd.DataFrame(summary_rows),
    }
