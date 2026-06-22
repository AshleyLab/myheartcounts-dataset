r"""Paired user-level bootstrap of the headline downstream benchmark stats (library).

Two-phase, mirroring the imputation paper-metrics pipeline:

* Phase 1 — :func:`compute_per_draw_errors` draws B paired bootstrap resamples on
  the per-(method, task) prediction parquets (for each task, sample N test users
  with replacement, the **same** indices reused across methods → paired
  comparisons), recomputes every method's per-task primary metric, and records the
  error ``E = 1 − metric`` globally and per demographic subgroup as a long-format
  draws frame (:func:`write_draws_parquet` persists it).

* Phase 2 — :func:`aggregate_skill_rank_fairness` reconstructs each draw's per-task
  metric (``1 − E``) and reduces the draws to ``point / SE / 95 % CI`` for the macro
  (domain-balanced) **skill score** vs the baseline, the **average rank** (both
  Overall + per-domain), and the per-subgroup **fairness** skill table. The headline
  disparity-ratio Fairness Skill Score (domain-balanced, BCa) is produced separately
  by ``aggregate_fairness_skill_score.py``.

The runnable CLIs are ``scripts/paper_results/downstream/bootstrap_downstream_draws.py`` (phase
1) and ``scripts/paper_results/downstream/aggregate_downstream_paper_metrics.py`` (phase 2).
Fairness rows require ``predictions_dir/_subgroups.json`` (per-user {age_group, sex});
without it only the global (``subgroup_attr="all"``) rows are produced.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from statistics import NormalDist

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

    The subgroup-S numerator uses the method's per-subgroup error, but the
    denominator stays
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
                        # Skip only an empty cell — a metric on zero users is undefined.
                        # Every nonempty subgroup counts; no minimum-size floor (its noise
                        # is already carried in the bootstrap CI, and non-finite metrics are
                        # dropped at emit). Matches the imputation track.
                        if not m_b.any():
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
    """Deterministic point + bootstrap mean + SE / percentile-CI for one quantity.

    ``values`` are the bootstrap draws (the point draw excluded). ``point`` is the
    full-cohort deterministic estimate; ``mean`` is the bootstrap mean of the draws;
    SE and the percentile CI come from the draws. Both centres are returned so each
    table can report the convention that matches imputation/forecasting — the
    deterministic ``point`` for the BCa-corrected fairness skill score, the bootstrap
    ``mean`` for the (near-unbiased) skill score and average rank. When ``point`` is
    omitted it falls back to the bootstrap mean.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    boot_mean = float(np.mean(arr)) if len(arr) else float("nan")
    center = float(point) if point is not None and np.isfinite(point) else boot_mean
    if len(arr) == 0:
        return {
            "point": center,
            "mean": boot_mean,
            "se": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
        }
    alpha = (1.0 - ci_level) / 2.0
    return {
        "point": center,
        "mean": boot_mean,
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
    domain_map: dict[str, str] = TASK_DOMAIN_MAP,
    ci_level: float = 0.95,
) -> dict[str, pd.DataFrame]:
    """Phase 2: summarise per-draw errors into skill / rank / subgroup tables.

    Reconstructs each draw's per-task metric (``1 - E``) and reuses the point
    helpers, so per-draw skill/rank match the one-pass bootstrap exactly. Returns
    ``{skill_scores, avg_rankings, fairness_subgroup_scores}`` (mean / se /
    percentile ci at ``ci_level``). The headline fairness metric — the
    disparity-ratio Fairness Skill Score — is produced separately by
    ``aggregate_fairness_skill_score.py``.
    """
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
    # Point estimates (full cohort, draw == POINT_DRAW) — the reported values.
    skill_point: dict[str, dict[str, float]] = {m: {} for m in methods}
    rank_point: dict[str, dict[str, float]] = {m: {} for m in methods}
    subS_point: dict[str, dict[tuple[str, str], float]] = {m: {} for m in methods}

    for b, g in glob.groupby("draw"):
        is_point = b == POINT_DRAW
        per_task_metric: dict[str, dict[str, float]] = {}
        for t, m_, e_ in zip(g["task"], g["method"], g["E"]):
            per_task_metric.setdefault(t, {})[m_] = 1.0 - e_
        ranks_b = _per_domain_avg_rank(per_task_metric, methods, domain_map)
        for m in methods:
            skill = _per_domain_skill_from_ratios(
                _ratios_for_method(per_task_metric, m, baseline),
                domain_map,
                clip_lower,
                clip_upper,
            )
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
                if is_point:
                    subS_point[m][(attr, value)] = s_sub
                else:
                    subS_draws[m].setdefault((attr, value), []).append(s_sub)

    skill_rows, rank_rows = [], []
    for m in methods:
        for scope in ["Overall", *domains]:
            if scope in skill_draws[m]:
                pt = skill_point[m].get(scope)
                row = {"method": m, "scope": scope, **_summarise(skill_draws[m][scope], ci_level, pt)}
                row["n_boot"] = n_boot
                row["n_tasks"] = n_tasks.get(scope, 0)
                skill_rows.append(row)
            if scope in rank_draws[m]:
                pt = rank_point[m].get(scope)
                row = {"method": m, "scope": scope, **_summarise(rank_draws[m][scope], ci_level, pt)}
                row["n_boot"] = n_boot
                rank_rows.append(row)

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

    return {
        "skill_scores": pd.DataFrame(skill_rows),
        "avg_rankings": pd.DataFrame(rank_rows),
        "fairness_subgroup_scores": pd.DataFrame(subgroup_rows),
    }


# ---------------------------------------------------------------------------
# BCa (bias-corrected & accelerated) interval — a point-anchored alternative to
# the percentile CI for the fairness disparity-ratio skill score, which is skewed
# and downward-biased (its bootstrap mean sits below the point, so the percentile
# CI is biased low). BCa re-anchors the interval near the reported point estimate
# and corrects for bias (z0) and skew/acceleration (a), second-order accurate. The
# acceleration is estimated from the leave-one-user-out jackknife. Φ / Φ⁻¹ come
# from ``statistics.NormalDist`` (no scipy dependency).
# ---------------------------------------------------------------------------

_NORM = NormalDist()


def _jackknife_acceleration(jack: np.ndarray) -> float:
    """BCa acceleration from leave-one-out jackknife values (nan-aware).

    ``a = Σ d³ / (6 · (Σ d²)^{3/2})`` with ``d = mean_i(θ₍ᵢ₎) − θ₍ᵢ₎``. Returns
    ``0.0`` when fewer than two finite values are present or ``Σ d² == 0``.
    """
    arr = np.asarray(jack, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size < 2:
        return 0.0
    d = finite.mean() - finite
    s2 = float(np.sum(d**2))
    if s2 == 0.0:
        return 0.0
    return float(np.sum(d**3)) / (6.0 * s2**1.5)


def _bca_interval(
    draws: np.ndarray, point: float, jack: np.ndarray, ci_level: float
) -> tuple[float, float]:
    """Bias-corrected & accelerated CI for one statistic.

    Args:
        draws: bootstrap draws ``θ*_b`` (NaN-dropped).
        point: the deterministic point estimate ``θ̂`` (the reported value).
        jack: leave-one-user-out jackknife values ``θ₍ᵢ₎`` (NaN-aware).
        ci_level: e.g. 0.95 -> a 2.5/97.5 percentile-equivalent interval.

    Guards (fall back to the plain percentile interval): empty/non-finite point,
    non-finite ``z0``/``a``, or a zero BCa denominator ``1 − a(z0 + z_q)``. All
    draws equal -> ``[point, point]``. When ``z0 = a = 0`` the adjusted percentiles
    reduce to ``α/2`` and ``1 − α/2``, i.e. the percentile interval exactly.
    """
    arr = np.asarray(draws, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    n = int(finite.size)
    alpha = 1.0 - ci_level

    def _percentile() -> tuple[float, float]:
        if n == 0:
            return float("nan"), float("nan")
        return (
            float(np.percentile(finite, 100.0 * (alpha / 2.0))),
            float(np.percentile(finite, 100.0 * (1.0 - alpha / 2.0))),
        )

    if n == 0 or not np.isfinite(point):
        return _percentile()
    if np.ptp(finite) == 0.0:
        return float(point), float(point)

    # Bias correction z0 from the fraction of draws below the point (clipped so
    # an extreme point still yields a finite z0).
    prop = float(np.count_nonzero(finite < point)) / n
    prop = min(max(prop, 0.5 / n), 1.0 - 0.5 / n)
    z0 = _NORM.inv_cdf(prop)
    a = _jackknife_acceleration(jack)
    if not (np.isfinite(z0) and np.isfinite(a)):
        return _percentile()

    out: list[float] = []
    for z_q in (_NORM.inv_cdf(alpha / 2.0), _NORM.inv_cdf(1.0 - alpha / 2.0)):
        denom = 1.0 - a * (z0 + z_q)
        if denom == 0.0 or not np.isfinite(denom):
            return _percentile()
        adj = z0 + (z0 + z_q) / denom
        if not np.isfinite(adj):
            return _percentile()
        frac = min(max(_NORM.cdf(adj), 0.0), 1.0)
        out.append(float(np.percentile(finite, 100.0 * frac)))
    return out[0], out[1]


def _pad_jackknife_maps(per_user_maps: list[dict[tuple, float]]) -> dict[tuple, np.ndarray]:
    """Align a list of per-user ``{key: value}`` maps into ``{key: array}``.

    The k-th array entry is user k's leave-one-out value, NaN where that user's
    recompute lacked the key (so every key spans all users, NaN-aware downstream).
    """
    keys: set[tuple] = set()
    for m in per_user_maps:
        keys |= m.keys()
    return {
        key: np.array([m.get(key, np.nan) for m in per_user_maps], dtype=np.float64) for key in keys
    }


# ---------------------------------------------------------------------------
# Leave-one-user-out jackknife of the disparity-ratio fairness skill score.
# Feeds the BCa acceleration term. Re-runs the *exact* point flow on the cohort
# minus each user, so ``jackknife_fairness_skill(...)[1]`` (the full-cohort point)
# reproduces the draws POINT_DRAW value by construction.
# ---------------------------------------------------------------------------


def mean_pairwise_abs_diff(values) -> float:
    """Subgroup disparity ``D``: mean ``|E_a − E_c|`` over all unordered subgroup pairs.

    The **mean absolute pairwise difference** (MAPD) of the per-subgroup errors,
    using every subgroup relationship rather than only the two extremes. For
    ``|G| = 2`` it collapses to ``|E_a − E_b|``; for ``|G| ≥ 3`` it smooths over
    every pair. NaN if fewer than 2 finite values. It is sensitive to duplicate
    subgroup values (a duplicate adds a zero-diff pair), so callers must pass one
    error per subgroup value. Mirrors the Track-2/3 ``_mapd``.
    """
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return float("nan")
    d = np.abs(vals[:, None] - vals[None, :])
    return float(np.mean(d[np.triu_indices(vals.size, k=1)]))


def _attr_disparity_ratio_skill(
    g: pd.DataFrame,
    methods: list[str],
    baseline: str,
    clip_lower: float,
    clip_upper: float,
    domain_map: dict[str, str] = TASK_DOMAIN_MAP,
) -> dict[str, float]:
    """Disparity-ratio fairness skill ``{method: S}`` for one (attribute, draw) slice.

    ``g`` carries columns ``task, method, subgroup_value, E``. Per task the disparity
    ``D`` is the mean unordered-pairwise ``|ΔE|`` over subgroup values (MAPD,
    :func:`mean_pairwise_abs_diff`), computed for the model and the baseline over the
    subgroup values **common to both** (inner join). A task is dropped when fewer than
    two common subgroups exist (MAPD is undefined, and a lone subgroup gives
    ``D_model = 0`` → a free near-perfect score) or when ``D_baseline ≤ 0`` /
    non-finite. ``ratio = clip(D_model / D_baseline)``; the per-method score is the
    **domain-balanced** macro of those ratios (:func:`_macro_skill_from_ratios`, the
    same reducer the headline skill score uses): per-domain ``1 − geomean``, then the
    mean over domains — so the fairness skill score aggregates identically to the skill
    score and average rank (each health domain weighted equally, not by its task
    count). Shared by the draws path (``aggregate_fairness_skill_score``) and the
    leave-one-user-out jackknife so the two are identical by construction.
    """
    out: dict[str, float] = {}
    # Per (task, method): the subgroup_value → E map. The disparity for a task is taken
    # over the subgroup set each method shares with the baseline (inner join), so
    # D_model and D_baseline always span the same subgroups; a task with < 2 common
    # subgroups is dropped.
    sub_e: dict[tuple[str, str], dict[str, float]] = {
        (t, m_): dict(zip(gg["subgroup_value"], gg["E"]))
        for (t, m_), gg in g.groupby(["task", "method"])
    }
    tasks = {t for (t, _) in sub_e}
    for m in methods:
        ratios: dict[str, float] = {}
        for t in tasks:
            base_map = sub_e.get((t, baseline))
            model_map = sub_e.get((t, m))
            if base_map is None or model_map is None:
                continue
            common = set(base_map) & set(model_map)
            if len(common) < 2:
                continue
            d_base = mean_pairwise_abs_diff([base_map[s] for s in common])
            d_model = mean_pairwise_abs_diff([model_map[s] for s in common])
            if not (np.isfinite(d_base) and d_base > 0 and np.isfinite(d_model)):
                continue
            ratios[t] = d_model / d_base
        if ratios:
            out[m] = _macro_skill_from_ratios(ratios, domain_map, clip_lower, clip_upper)
    return out


def _fairness_skill_from_indices(
    aligned: dict[str, dict[str, dict]],
    methods: list[str],
    tasks: list[str],
    attributes: tuple[str, ...],
    attribute_masks: dict[str, dict[str, dict[str, np.ndarray]]],
    idx_per_task: dict[str, np.ndarray],
    baseline: str,
    clip_lower: float,
    clip_upper: float,
    domain_map: dict[str, str] = TASK_DOMAIN_MAP,
) -> dict[tuple[str, str], float]:
    """``{(method, scope): S}`` (scopes = each attribute + ``overall``) on given row indices.

    Recomputes the per-(task, method, subgroup) error ``E = 1 − metric`` on the
    supplied per-task row indices (mirrors phase-1's per-subgroup emit: skip a
    subgroup cell that is empty, has < 2 users, or yields a non-finite metric),
    then reduces with :func:`_attr_disparity_ratio_skill`. ``overall`` is the mean
    over attributes that produced a score (mirrors the draws-path macro).
    """
    per_attr: dict[str, dict[str, float]] = {}
    out: dict[tuple[str, str], float] = {}
    for attr in attributes:
        rows: list[dict] = []
        values = sorted({v for t in tasks for v in attribute_masks[t][attr]})
        for value in values:
            masks_b: dict[str, np.ndarray] = {}
            for t in tasks:
                cm = attribute_masks[t][attr].get(value)
                if cm is None:
                    continue
                m_b = cm[idx_per_task[t]]
                if not m_b.any():
                    continue
                masks_b[t] = m_b
            if not masks_b:
                continue
            per_task = _per_task_metric_on_indices(
                aligned, methods, tasks, idx_per_task, mask_per_task=masks_b
            )
            for t, method_vals in per_task.items():
                for m, metric in method_vals.items():
                    if metric is None or not np.isfinite(metric):
                        continue
                    rows.append(
                        {"task": t, "method": m, "subgroup_value": value, "E": 1.0 - float(metric)}
                    )
        if not rows:
            continue
        g = pd.DataFrame(rows, columns=["task", "method", "subgroup_value", "E"])
        for m, s in _attr_disparity_ratio_skill(
            g, methods, baseline, clip_lower, clip_upper, domain_map
        ).items():
            out[(m, attr)] = s
            per_attr.setdefault(m, {})[attr] = s
    for m in methods:
        vals = [per_attr[m][a] for a in attributes if m in per_attr and a in per_attr[m]]
        if vals:
            out[(m, "overall")] = float(np.mean(vals))
    return out


def jackknife_fairness_skill(
    aligned: dict[str, dict[str, dict]],
    subgroup_map: dict[str, dict[str, str]],
    attributes: tuple[str, ...],
    baseline: str,
    *,
    clip_lower: float,
    clip_upper: float,
    domain_map: dict[str, str] = TASK_DOMAIN_MAP,
) -> tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, str], float]]:
    """Exact leave-one-user-out jackknife of the disparity-ratio fairness skill score.

    Re-runs the deterministic point flow (:func:`_fairness_skill_from_indices`) on
    the aligned per-user predictions minus each user. Returns
    ``(jack_by_key, point_by_key)`` where ``jack_by_key[(method, scope)]`` is the
    array of leave-one-user-out S values over the cohort (NaN where a scope is
    absent for that recompute) and ``point_by_key`` is the full-cohort S (== the
    draws POINT_DRAW value). Scopes = each attribute in ``attributes`` + ``overall``.
    ~U deterministic recomputes (U = distinct users across tasks).
    """
    methods = list(aligned.keys())
    tasks = sorted(aligned[methods[0]].keys())
    attribute_masks = _build_subgroup_masks(aligned, tasks, methods, subgroup_map, list(attributes))
    full_idx = {t: np.arange(len(aligned[methods[0]][t]["uids"])) for t in tasks}
    point = _fairness_skill_from_indices(
        aligned, methods, tasks, attributes, attribute_masks, full_idx, baseline, clip_lower, clip_upper,
        domain_map,
    )
    pos = {t: {u: i for i, u in enumerate(aligned[methods[0]][t]["uids"])} for t in tasks}
    users = sorted({u for t in tasks for u in aligned[methods[0]][t]["uids"]})
    per_user_maps: list[dict[tuple, float]] = []
    for u in users:
        idx_u = {}
        for t in tasks:
            p = pos[t].get(u)
            idx_u[t] = full_idx[t] if p is None else np.delete(full_idx[t], p)
        per_user_maps.append(
            _fairness_skill_from_indices(
                aligned, methods, tasks, attributes, attribute_masks, idx_u, baseline, clip_lower, clip_upper,
                domain_map,
            )
        )
    return _pad_jackknife_maps(per_user_maps), point
