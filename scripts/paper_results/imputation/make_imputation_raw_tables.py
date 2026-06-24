#!/usr/bin/env python3
"""Generate the arXiv imputation *raw-metric* appendix tables from HF.

Source of truth: the per-method substrate on the OpenMHC leaderboard HF dataset
(``MyHeartCounts/OpenMHC-leaderboard-data``, ``imputation/<method>.parquet``).
Each row is a per-(method, scenario, split, channel, channel_type,
subgroup_attr, subgroup_value, user) error ``E_per_user`` where ``continuous``
= per-user MAE and ``binary`` = ``1 - AUC`` (un-floored).

The substrate stores **MAE** (not RMSE — RMSE needs squared errors, which are
not retained), so these tables report MAE for continuous channels and ROC AUC
for binary channels:

    MAE(method, scenario)  = mean over the scenario's continuous channels of
                             [ mean over users of E_per_user ]
    AUC(method, scenario)  = mean over the scenario's binary channels of
                             [ 1 - mean over users of E_per_user ]

Only ``random_noise``, ``temporal_slice``, ``signal_slice`` mask binary
channels, so AUC is reported only for those three; ``sleep_gap``,
``workout_gap``, ``intensity_failure`` are continuous-only (MAE only).

Emits two ``\\begin{table}...\\end{table}`` blocks (single-day and long-context),
matching the existing hand-written appendix tables (booktabs rules, per-column
``customblue!N`` gradient, bold best cell), as point estimates over the test
cohort.

Usage:
    python scripts/paper_results/imputation/make_imputation_raw_tables.py \
        --out-single ~/MHC-benchmark/paper/sections_arxiv/appendix/imputation_raw_single_day_table.tex \
        --out-long   ~/MHC-benchmark/paper/sections_arxiv/appendix/imputation_raw_long_context_table.tex
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_REPO_ID = "MyHeartCounts/OpenMHC-leaderboard-data"

# key -> (latex_label, context, model_group). Labels mirror the existing tables
# (\cite, not \citep). Order within each (context, group) is fixed.
METHODS: dict[str, tuple[str, str, str]] = {
    # single-day -- statistical
    "linear": (r"Linear", "single", "stat"),
    "temporal_mean": (r"Temporal mean", "single", "stat"),
    "locf": (r"LOCF \textit{(reference)}", "single", "stat"),
    "temporal_mode": (r"Temporal mode", "single", "stat"),
    "mode": (r"Mode", "single", "stat"),
    "mean": (r"Mean", "single", "stat"),
    # single-day -- neural
    "lsm2": (r"LSM-2~\cite{xu2025lsm}", "single", "neural"),
    "dlinear": (r"DLinear~\cite{zeng2023dlinear}", "single", "neural"),
    "brits": (r"BRITS~\cite{cao2018brits}", "single", "neural"),
    "timesnet": (r"TimesNet~\cite{wu2023timesnet}", "single", "neural"),
    "fedformer": (r"FEDformer~\cite{zhou2022fedformer}", "single", "neural"),
    # long-context -- statistical
    "personalized_temporal_mean": (r"Pers.\ temp.\ mean", "long", "stat"),
    "personalized_mean": (r"Pers.\ mean", "long", "stat"),
    "personalized_mode": (r"Pers.\ mode", "long", "stat"),
    # long-context -- neural
    "lsm2_weekly": (r"LSM-2 (7-day)", "long", "neural"),
    "lsm2_weekly_sparse": (r"LSM-2-Sparse (7-day)", "long", "neural"),
    "dlinear_weekly": (r"DLinear (7-day)~\cite{zeng2023dlinear}", "long", "neural"),
}

# (column prefix, scenario key, has_auc) in table-column order.
SCENARIOS: list[tuple[str, str, bool]] = [
    ("Random", "random_noise", True),
    ("Temporal", "temporal_slice", True),
    ("Signal", "signal_slice", True),
    ("Sleep", "sleep_gap", False),
    ("Workout", "workout_gap", False),
    ("Intensity", "intensity_failure", False),
]

GROUP_TITLE = {
    "stat": r"\cellcolor[HTML]{EFEFEF}\textit{Statistical Models}",
    "neural": r"\cellcolor[HTML]{EFEFEF}\textit{Neural Models}",
}

CAPTION = {
    "single": (
        r"\textbf{Single-Day Imputation Raw Metrics.} Scenario-level raw metrics on the "
        r"test split. Each MAE entry is the mean per-channel MAE across applicable "
        r"continuous channels; each ROC AUC entry is the macro-average across applicable "
        r"binary channels. Sleep gap, workout gap, and intensity failure mask only "
        r"continuous channels, so ROC AUC is not reported there. Values are point "
        r"estimates over the test cohort."
    ),
    "long": (
        r"\textbf{Long-Context Imputation Raw Metrics.} Scenario-level raw metrics on the "
        r"test split. Each MAE entry is the mean per-channel MAE across applicable "
        r"continuous channels; each ROC AUC entry is the macro-average across applicable "
        r"binary channels. Sleep gap, workout gap, and intensity failure mask only "
        r"continuous channels, so ROC AUC is not reported there. Values are point "
        r"estimates over the test cohort."
    ),
}
LABEL = {
    "single": "tab:imputation_appendix_raw_single_day",
    "long": "tab:imputation_appendix_raw_long_context",
}


def build_columns() -> list[tuple[str, str, str, bool]]:
    """Flatten SCENARIOS into (header, scenario, metric, lower_better) columns."""
    cols: list[tuple[str, str, str, bool]] = []
    for prefix, scen, has_auc in SCENARIOS:
        cols.append((rf"{prefix} MAE$\downarrow$", scen, "mae", True))
        if has_auc:
            cols.append((rf"{prefix} AUC$\uparrow$", scen, "auc", False))
    return cols


def compute_metrics(repo_id: str, revision: str | None) -> dict[str, dict[tuple[str, str], float]]:
    """Return {method: {(scenario, 'mae'|'auc'): value}} point estimates from HF."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    out: dict[str, dict[tuple[str, str], float]] = {}
    for method in METHODS:
        path = hf_hub_download(
            repo_id=repo_id, filename=f"imputation/{method}.parquet",
            repo_type="dataset", revision=revision,
        )
        df = pd.read_parquet(path)
        df = df[(df["subgroup_attr"] == "all") & (df["split"] == "test")]
        vals: dict[tuple[str, str], float] = {}
        for _prefix, scen, has_auc in SCENARIOS:
            s = df[df["scenario"] == scen]
            cont = s[s["channel_type"] == "continuous"]
            # per-channel macro-mean over users, then mean across channels.
            per_ch_mae = cont.groupby("channel")["E_per_user"].mean()
            if len(per_ch_mae):
                vals[(scen, "mae")] = float(per_ch_mae.mean())
            if has_auc:
                binr = s[s["channel_type"] == "binary"]
                per_ch_err = binr.groupby("channel")["E_per_user"].mean()
                if len(per_ch_err):
                    vals[(scen, "auc")] = float((1.0 - per_ch_err).mean())
        out[method] = vals
    return out


def intensity(value: float, vmin: float, vmax: float, lower_better: bool) -> int:
    """Per-column min-max intensity in [0, 100]."""
    if vmax == vmin:
        return 0
    frac = (vmax - value) / (vmax - vmin) if lower_better else (value - vmin) / (vmax - vmin)
    return round(frac * 100)


def fmt_cell(value: float | None, metric: str, n: int, is_best: bool) -> str:
    """One LaTeX cell: optional color + ``$value$`` (MAE 1 dp, AUC 3 dp)."""
    if value is None:
        return ""  # scenario has no channels of this type (e.g. AUC for continuous-only)
    num = f"{value:.1f}" if metric == "mae" else f"{value:.3f}"
    body = rf"\mathbf{{{num}}}" if is_best else num
    color = rf"\cellcolor{{customblue!{n}}}" if n > 0 else ""
    return rf"{color}${body}$"


def build_table(
    ctx: str, metrics: dict[str, dict[tuple[str, str], float]]
) -> str:
    columns = build_columns()
    ndata = len(columns)
    ncol = ndata + 1
    members = [m for m, (_, c, _) in METHODS.items() if c == ctx]

    # Per-column min/max over the methods present in this table.
    bounds = []
    for _h, scen, metric, _lb in columns:
        vals = [metrics[m].get((scen, metric)) for m in members]
        vals = [v for v in vals if v is not None]
        bounds.append((min(vals), max(vals)) if vals else (0.0, 0.0))

    header_cells = " & ".join(h for h, _s, _m, _lb in columns)
    lines = [
        r"\begin{table}[t!]",
        r"    \renewcommand{\arraystretch}{1.05}",
        r"    \centering",
        r"    \captionsetup{width=\textwidth}",
        rf"    \caption{{{CAPTION[ctx]}}}",
        rf"    \label{{{LABEL[ctx]}}}",
        r"    \small",
        r"    \setlength{\tabcolsep}{1.5pt}",
        r"    \resizebox{\linewidth}{!}{%",
        rf"    \begin{{tabular}}{{l {'c' * ndata}}}",
        r"    \toprule[1.5pt]",
        rf"    \textbf{{Method}} & {header_cells} \\",
        r"    \hline",
    ]

    for gi, grp in enumerate(("stat", "neural")):
        if gi > 0:
            lines.append(r"    \hline")
        lines.append(rf"    \multicolumn{{{ncol}}}{{l}}{{{GROUP_TITLE[grp]}}} \\")
        for m in [mm for mm in members if METHODS[mm][2] == grp]:
            cells = []
            for ci, (_h, scen, metric, lower) in enumerate(columns):
                v = metrics[m].get((scen, metric))
                vmin, vmax = bounds[ci]
                if v is None:
                    cells.append("")
                    continue
                n = intensity(v, vmin, vmax, lower)
                best_val = vmin if lower else vmax
                is_best = v == best_val
                cells.append(fmt_cell(v, metric, n, is_best))
            lines.append(rf"    {METHODS[m][0]} & " + " & ".join(cells) + r" \\")

    lines += [r"    \bottomrule[1.5pt]", r"    \end{tabular}%", r"    }", r"\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    p.add_argument("--revision", default=None)
    base = Path.home() / "MHC-benchmark/paper/sections_arxiv/appendix"
    p.add_argument("--out-single", type=Path, default=base / "imputation_raw_single_day_table.tex")
    p.add_argument("--out-long", type=Path, default=base / "imputation_raw_long_context_table.tex")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    metrics = compute_metrics(args.repo_id, args.revision)
    for ctx, out in (("single", args.out_single), ("long", args.out_long)):
        table = build_table(ctx, metrics)
        if args.dry_run:
            print(table)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(table)
        print(f"Wrote {out} ({len(table)} bytes)")


if __name__ == "__main__":
    main()
