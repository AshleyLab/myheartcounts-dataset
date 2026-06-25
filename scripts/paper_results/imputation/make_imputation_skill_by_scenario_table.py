#!/usr/bin/env python3
"""Generate the arXiv imputation *skill-by-scenario* appendix table from HF.

Source of truth: the OpenMHC leaderboard HF dataset
(``MyHeartCounts/OpenMHC-leaderboard-data``). This table includes the dense
``LSM-2 (7-day)`` baseline, so it reduces the **dense-weekly** bootstrap
reference ``imputation/bootstrap_with_dense_weekly/draws.parquet`` (the main
16-method draws + the dense ``lsm2_weekly``) plus the matching per-method
substrate parquets — so all rows are ranked within one consistent 17-method pool.

Columns: Aggregate Skill Score $S$, Average Rank $R$, Fairness Skill Score
$S_{\\text{fair}}$ (the **disparity-ratio** score used by the main table — the
deprecated $S-\\lambda\\bar D$ score and its $\\bar D$ column are dropped), and
per-scenario Skill Scores for the six masking scenarios. Values are bootstrap
mean $\\pm$ SE ($B{=}1000$); $S_{\\text{fair}}$ is the deterministic point
estimate $\\pm$ bootstrap SE.

Reductions are the canonical ones (``aggregate_skill_rank_fairness`` for
skill/rank, ``compute_fairness_skill_scores`` for the disparity-ratio fairness),
identical to ``make_imputation_latex_tables.py`` but over the dense-weekly
superset and with ``bca=False`` (the table renders $\\pm$SE, so the BCa jackknife
is unnecessary).

Usage:
    python scripts/paper_results/imputation/make_imputation_skill_by_scenario_table.py \
        --out ~/MHC-benchmark/paper/sections_arxiv/appendix/imputation_skill_by_scenario_table.tex
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_REPO_ID = "MyHeartCounts/OpenMHC-leaderboard-data"
DRAWS_PATH = "imputation/bootstrap_with_dense_weekly/draws.parquet"
REFERENCE = "locf"

# key -> (latex_label, context, model_group). Labels mirror the existing appendix
# table (plain \cite). lsm2_weekly is the dense 7-day baseline.
METHODS: dict[str, tuple[str, str, str]] = {
    "linear": (r"Linear", "single", "stat"),
    "temporal_mean": (r"Temporal mean", "single", "stat"),
    "locf": (r"LOCF \textit{(reference)}", "single", "stat"),
    "temporal_mode": (r"Temporal mode", "single", "stat"),
    "mode": (r"Mode", "single", "stat"),
    "mean": (r"Mean", "single", "stat"),
    "lsm2": (r"LSM-2~\cite{xu2025lsm}", "single", "neural"),
    "dlinear": (r"DLinear~\cite{zeng2023dlinear}", "single", "neural"),
    "brits": (r"BRITS~\cite{cao2018brits}", "single", "neural"),
    "timesnet": (r"TimesNet~\cite{wu2023timesnet}", "single", "neural"),
    "fedformer": (r"FEDformer~\cite{zhou2022fedformer}", "single", "neural"),
    "personalized_temporal_mean": (r"Pers.\ temp.\ mean", "long", "stat"),
    "personalized_mean": (r"Pers.\ mean", "long", "stat"),
    "personalized_mode": (r"Pers.\ mode", "long", "stat"),
    "lsm2_weekly": (r"LSM-2 (7-day)", "long", "neural"),
    "lsm2_weekly_sparse": (r"LSM-2-Sparse (7-day)", "long", "neural"),
    "dlinear_weekly": (r"DLinear (7-day)~\cite{zeng2023dlinear}", "long", "neural"),
}

# (header, source, scope, center, scale100, lower_better, ref_zero)
#   source: "skill" | "rank" | "fair"
COLUMNS = [
    (r"$S\uparrow$", "skill", "overall", "mean", True, False, True),
    (r"$R\downarrow$", "rank", "overall", "mean", False, True, False),
    (r"$S_{\text{fair}}\uparrow$", "fair", "overall", "point", True, False, True),
    (r"Random noise\,$\uparrow$", "skill", "random_noise", "mean", True, False, True),
    (r"Temporal slice\,$\uparrow$", "skill", "temporal_slice", "mean", True, False, True),
    (r"Signal slice\,$\uparrow$", "skill", "signal_slice", "mean", True, False, True),
    (r"Sleep gap\,$\uparrow$", "skill", "sleep_gap", "mean", True, False, True),
    (r"Workout gap\,$\uparrow$", "skill", "workout_gap", "mean", True, False, True),
    (r"Intensity failure\,$\uparrow$", "skill", "intensity_failure", "mean", True, False, True),
]
NCOL = len(COLUMNS) + 1

SECTION_TITLE = {
    "single": r"\textbf{\emph{Single-day imputation}}",
    "long": r"\textbf{\emph{Long-context imputation ($\geq 7 \times 1440$ time steps)}}",
}
GROUP_TITLE = {
    "stat": r"\cellcolor[HTML]{EFEFEF}\textit{Statistical Models}",
    "neural": r"\cellcolor[HTML]{EFEFEF}\textit{Neural Models}",
}

HEADER_TMPL = r"""\begin{table}[t!]
    \renewcommand{\arraystretch}{1.05}
    \centering
    \captionsetup{width=\textwidth}
    \caption{\textbf{Imputation Results by Masking Scenario.} Aggregate Skill Score $S$ (in \%; $0=$LOCF reference), Average Rank $R$, Fairness Skill Score $S_{\text{fair}}$ (disparity-ratio; see Appendix~\ref{sec:fairness_adjusted_score}), and per-scenario Skill Scores across all six masking scenarios (lower is better for $R$; higher otherwise). Single-day methods above; long-context methods ($\geq 7\times 1440$ time steps) below. Gradients computed within each track. Values are bootstrap means $\pm$ SE ($B{=}1000$); $S_{\text{fair}}$ is the point estimate $\pm$ bootstrap SE.}
    \label{tab:imputation_appendix_skill_by_scenario}
    \small
    \setlength{\tabcolsep}{1.5pt}
    \resizebox{\linewidth}{!}{%
    \begin{tabular}{l ccccccccc}
    \toprule[1.5pt]
    \textbf{Method} & $S\uparrow$ & $R\downarrow$ & $S_{\text{fair}}\uparrow$ & Random noise\,$\uparrow$ & Temporal slice\,$\uparrow$ & Signal slice\,$\uparrow$ & Sleep gap\,$\uparrow$ & Workout gap\,$\uparrow$ & Intensity failure\,$\uparrow$ \\
    \midrule
"""
FOOTER = r"""    \bottomrule[1.5pt]
    \end{tabular}%
    }
\end{table}
"""


def reduce_from_hf(repo_id: str, revision: str | None) -> dict[str, dict[tuple[str, str], tuple[float, float]]]:
    """Return {method: {(source, scope): (center, se)}} from the dense-weekly HF substrate."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    from imputation_evaluation.evaluation.bootstrap_skill_rank import (
        aggregate_skill_rank_fairness,
        read_draws_parquet,
    )

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from aggregate_fairness_skill_score import (  # noqa: E402
        BCA_HEADLINE_SCOPES,
        SENSITIVE_ATTRS,
        _fair_points_by_key,
        _per_user_to_per_cell_E,
        compute_fairness_skill_scores,
    )

    draws_path = hf_hub_download(
        repo_id=repo_id, filename=DRAWS_PATH, repo_type="dataset", revision=revision
    )
    draws_df, _ = read_draws_parquet(Path(draws_path))
    all_rows = draws_df[draws_df["subgroup_attr"] == "all"]
    tables = aggregate_skill_rank_fairness(all_rows)

    per_method = [
        pd.read_parquet(
            hf_hub_download(
                repo_id=repo_id, filename=f"imputation/{m}.parquet",
                repo_type="dataset", revision=revision,
            )
        )
        for m in METHODS
    ]
    per_user_df = pd.concat(per_method, ignore_index=True)
    # Fairness SE from the (cheap) percentile bootstrap; the CENTER is the
    # deterministic point estimate (matches the main table, which uses point).
    # We compute the point directly via _fair_points_by_key — the BCa jackknife
    # (the slow part) is unnecessary since this table renders point +/- SE.
    fairness = compute_fairness_skill_scores(
        draws_df, attrs=list(SENSITIVE_ATTRS), baseline_method=REFERENCE, bca=False,
    )
    # B.2: drop per-channel binary ch_7..ch_18 (sleep/workouts reach fairness only
    # via cat_collapsed:*), matching compute_fairness_skill_scores' point flow.
    pu = per_user_df
    drop = pu["channel"].astype(str).str.match(r"^ch_(?:[7-9]|1[0-8])$") & (
        pu["channel_type"].astype(str) == "binary"
    )
    per_cell = _per_user_to_per_cell_E(pu[~drop & (pu["split"] == "test")])
    points = _fair_points_by_key(
        per_cell, attrs=list(SENSITIVE_ATTRS), baseline_method=REFERENCE,
        clip_lower=1e-2, clip_upper=100.0, scopes=BCA_HEADLINE_SCOPES,
    )
    fair_se = {
        r["method"]: (float(r["se"]) if r["se"] == r["se"] else 0.0)
        for _, r in fairness[(fairness["scope"] == "overall") & (fairness["split"] == "test")].iterrows()
    }

    out: dict[str, dict[tuple[str, str], tuple[float, float]]] = {m: {} for m in METHODS}

    def _ingest(df, source, center_col):
        sub = df[df["split"] == "test"]
        for _, r in sub.iterrows():
            m = r["method"]
            if m not in out:
                continue
            c = r.get(center_col)
            if c is None or (isinstance(c, float) and c != c):
                continue
            se = r.get("se")
            out[m][(source, str(r["scope"]))] = (float(c), float(se) if se == se else 0.0)

    _ingest(tables["skill_scores"], "skill", "mean")
    _ingest(tables["avg_rankings"], "rank", "mean")
    for m in METHODS:
        pt = points.get((m, "overall"))
        if pt is not None:
            out[m][("fair", "overall")] = (float(pt), fair_se.get(m, 0.0))
    return out


def intensity(value: float, vmin: float, vmax: float, lower_better: bool) -> int:
    if vmax == vmin:
        return 0
    frac = (vmax - value) / (vmax - vmin) if lower_better else (value - vmin) / (vmax - vmin)
    return round(frac * 100)


def fmt_cell(method, center, se, scale100, ref_zero, n, is_best) -> str:
    if ref_zero and method == REFERENCE:
        return r"$0.0$"
    s = 100.0 if scale100 else 1.0
    num = f"{center * s:+.1f}" if scale100 else f"{center * s:.1f}"
    se_s = f"{se * s:.1f}"
    body = rf"\mathbf{{{num}}}" if is_best else num
    color = rf"\cellcolor{{customblue!{n}}}" if n > 0 else ""
    return rf"{color}${body}{{\scriptstyle \pm {se_s}}}$"


def build_body(data) -> str:
    lines: list[str] = []
    for ctx in ("single", "long"):
        if ctx == "long":
            lines.append(r"    \midrule")
        lines.append(rf"    \multicolumn{{{NCOL}}}{{l}}{{{SECTION_TITLE[ctx]}}} \\")
        lines.append(r"    \hline")
        section = [m for m, (_, c, _) in METHODS.items() if c == ctx]
        bounds = []
        for _h, src, scope, _ctr, _s, _lb, _rz in COLUMNS:
            vals = [data[m][(src, scope)][0] for m in section if (src, scope) in data[m]]
            bounds.append((min(vals), max(vals)) if vals else (0.0, 0.0))
        for gi, grp in enumerate(("stat", "neural")):
            if gi > 0:
                lines.append(r"    \hline")
            lines.append(rf"    \multicolumn{{{NCOL}}}{{l}}{{{GROUP_TITLE[grp]}}} \\")
            members = [m for m in section if METHODS[m][2] == grp]
            members.sort(key=lambda m: -data[m].get(("skill", "overall"), (0.0, 0.0))[0])
            for m in members:
                cells = []
                for ci, (_h, src, scope, _ctr, scale100, lower, ref_zero) in enumerate(COLUMNS):
                    center, se = data[m].get((src, scope), (float("nan"), 0.0))
                    vmin, vmax = bounds[ci]
                    n = intensity(center, vmin, vmax, lower)
                    best_val = vmin if lower else vmax
                    is_best = (center == best_val) and (m != REFERENCE)
                    cells.append(fmt_cell(m, center, se, scale100, ref_zero, n, is_best))
                lines.append(rf"    {METHODS[m][0]} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    p.add_argument("--revision", default=None)
    p.add_argument(
        "--out", type=Path,
        default=Path.home()
        / "MHC-benchmark/paper/sections_arxiv/appendix/imputation_skill_by_scenario_table.tex",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    data = reduce_from_hf(args.repo_id, args.revision)
    missing = [m for m in METHODS if not data[m]]
    if missing:
        raise SystemExit(f"Methods missing from reduction: {missing}")

    table = HEADER_TMPL + build_body(data) + FOOTER
    if args.dry_run:
        print(table)
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(table)
    print(f"Wrote {args.out} ({len(table)} bytes)")


if __name__ == "__main__":
    main()
