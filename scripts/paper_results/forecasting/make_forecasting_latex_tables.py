#!/usr/bin/env python3
"""Generate the arXiv *main* forecasting table from the forecasting bootstrap CSVs.

Reads the bootstrap CSVs in ``forecasting_bca_20260618/`` and emits the full
``\\begin{table}...\\end{table}`` block for ``tab:forecasting_grouped_model_summary``,
matching the existing hand-written arXiv table (booktabs rules, per-column
``customblue!N`` blue gradient, bold best cell).

Uncertainty is reported as a 95% bootstrap confidence interval rendered as an
asymmetric ``value^{+upper}_{-lower}`` super/subscript (not the SE), mirroring
the imputation table. For every column except ``S_fair`` the center is the
bootstrap ``mean`` and the interval is the percentile CI (``ci_lo``/``ci_hi``).
``S_fair`` (from ``forecasting_fairness_skill_score_bootstrap.csv``, scope
``overall``) uses the deterministic ``point`` estimate and the BCa interval
(``bca_lo``/``bca_hi``).

Seasonal Naive is the reference: in every skill column it renders as a plain
``$0.0$`` (no CI, no color); it still gets a real value+CI in the rank column.

Only the 10 bootstrapped models are emitted; the current table's
``LSM-2-Sparse`` row is dropped (it is not in this results set). The model order
is fixed (baseline first, zeroshot/fine-tuned pairs kept together) rather than
skill-sorted, matching the existing table.

Category-column -> scope mapping. The four sensor categories match the
imputation table (no Semantic column: forecasting has no semantic masking):
    Activity     activity_score    (continuous channels 0-4)
    Physiology   physiology_score  (continuous channels 5-6: heart rate + energy)
    Sleep        sleep_score
    Workout      workout_score

Usage:
    python scripts/paper_results/forecasting/make_forecasting_latex_tables.py \
        --results-dir forecasting_bca_20260618 \
        --out ~/MHC-benchmark/paper/sections_arxiv/forecasting_main_results_table.tex
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

# ---------------------------------------------------------------------------
# Model registry: key -> (latex_label, model_group). Order within each group is
# fixed (NOT skill-sorted) to keep model families together, matching the table.
#   model_group : "stat" | "neural" | "foundation"
# ---------------------------------------------------------------------------
REFERENCE = "seasonal_naive"

MODELS: dict[str, tuple[str, str]] = {
    # statistical
    "seasonal_naive": (r"\textsc{Seasonal Naive}", "stat"),
    "autoARIMA": (r"\textsc{AutoARIMA}", "stat"),
    "autoETS": (r"\textsc{AutoETS}", "stat"),
    # neural (trained from scratch)
    "mixlinear": (r"\textsc{MixLinear}~\citep{ma2024mixlinear}", "neural"),
    "dlinear": (r"\textsc{DLinear}~\citep{zeng2023dlinear}", "neural"),
    "segrnn": (r"\textsc{SegRNN}~\citep{lin2025segrnn}", "neural"),
    # time-series foundation models
    "toto_zeroshot_ctx4096": (r"\textsc{Toto}~\citep{cohen2024toto}", "foundation"),
    "toto_finetuned_ctx4096": (r"\textsc{Toto} (FT)", "foundation"),
    "chronos2_zeroshot": (r"\textsc{Chronos-2}~\citep{ansari2025chronos}", "foundation"),
    "chronos2_finetuned": (r"\textsc{Chronos-2} (FT)", "foundation"),
}

GROUP_ORDER = ("stat", "neural", "foundation")
GROUP_TITLE = {
    "stat": "Statistical Methods",
    "neural": "Neural Models",
    "foundation": "Time-Series Foundation Models",
}

# Column spec: (header, csv, scope, center, lo, hi, scale100, lower_better, ref_zero, metric)
#   metric : value of the ``metric`` column to filter on (rank CSV only), else None
SKILL = "forecasting_skill_score_bootstrap.csv"
RANK = "forecasting_grouped_metric_rank_bootstrap.csv"
FAIR = "forecasting_fairness_skill_score_bootstrap.csv"

COLUMNS: list[tuple[str, str, str, str, str, str, bool, bool, bool, str | None]] = [
    (r"$R \downarrow$",            RANK,  "overall",          "mean",  "ci_lo",  "ci_hi",  False, True,  False, "overall"),
    (r"$S \uparrow$",              SKILL, "overall_score",    "mean",  "ci_lo",  "ci_hi",  True,  False, True,  None),
    (r"$S_{\text{fair}} \uparrow$",FAIR,  "overall",          "point", "bca_lo", "bca_hi", True,  False, True,  None),
    (r"Activity\,$\uparrow$",      SKILL, "activity_score",   "mean",  "ci_lo",  "ci_hi",  True,  False, True,  None),
    (r"Physio.\,$\uparrow$",       SKILL, "physiology_score", "mean",  "ci_lo",  "ci_hi",  True,  False, True,  None),
    (r"Sleep\,$\uparrow$",         SKILL, "sleep_score",      "mean",  "ci_lo",  "ci_hi",  True,  False, True,  None),
    (r"Workout\,$\uparrow$",       SKILL, "workout_score",    "mean",  "ci_lo",  "ci_hi",  True,  False, True,  None),
]

NCOL = len(COLUMNS) + 1  # + method column

HEADER = r"""\begin{table*}[t]
\centering
\captionsetup{width=0.98\textwidth}
\caption{
\textbf{Forecasting Results.}
We report Average Rank $R$, Aggregate Skill Score $S$
(in \%; $0=\textsc{Seasonal Naive}$ reference),
Fairness-adjusted Skill Score $S_{\mathrm{fair}}$, and category-specific
Skill Scores for \textit{Activity}, \textit{Physiology}, \textit{Sleep},
and \textit{Workout}. FT denotes fine-tuned. Subscripts and superscripts
indicate the $95\%$ bootstrap confidence interval based on $1000$ resamples.
}
\label{tab:forecasting_grouped_model_summary}

\newcommand{\est}[3]{%
  \ensuremath{#1^{\scriptscriptstyle +#2}_{\scriptscriptstyle -#3}}%
}

\small
\renewcommand{\arraystretch}{1.16}
\setlength{\tabcolsep}{2.2pt}

\begin{tabularx}{\textwidth}{
    >{\raggedright\arraybackslash}X
    *{7}{>{\centering\arraybackslash}m{1.35cm}}
}
\toprule[1.4pt]

\textbf{Method}
& \mbox{$R\,\downarrow$}
& \mbox{$S\,\uparrow$}
& \mbox{$S_{\mathrm{fair}}\,\uparrow$}
& \mbox{Activity~$\uparrow$}
& \mbox{Physio.~$\uparrow$}
& \mbox{Sleep~$\uparrow$}
& \mbox{Workout~$\uparrow$} \\

\midrule
"""

FOOTER = r"""\bottomrule[1.4pt]
\end{tabularx}
\end{table*}
"""


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


def load_metric(
    path: Path,
    scope: str,
    center_col: str,
    lo_col: str,
    hi_col: str,
    metric: str | None,
) -> dict[str, tuple[float, float, float]]:
    """Return {model: (center, lo, hi)} for one scope (optionally a metric filter)."""
    out: dict[str, tuple[float, float, float]] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        missing = [c for c in (center_col, lo_col, hi_col) if c not in fields]
        if missing:
            raise SystemExit(f"{path.name}: columns {missing} not in header {fields!r}")
        for r in reader:
            if r["scope"] != scope:
                continue
            if metric is not None and r.get("metric") != metric:
                continue
            c = r[center_col]
            if c in ("", None):
                continue
            center = float(c)
            lo = float(r[lo_col]) if r[lo_col] not in ("", None) else center
            hi = float(r[hi_col]) if r[hi_col] not in ("", None) else center
            out[r["model"]] = (center, lo, hi)
    return out


def load_columns(results_dir: Path) -> list[dict[str, tuple[float, float, float]]]:
    """Load each column's {model: (center, lo, hi)} map, in COLUMNS order."""
    return [
        load_metric(results_dir / fname, scope, center, lo, hi, metric)
        for _h, fname, scope, center, lo, hi, _s100, _lower, _ref, metric in COLUMNS
    ]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def intensity(value: float, vmin: float, vmax: float, lower_better: bool) -> int:
    """Per-column min-max intensity in [0, 100] (global over all models)."""
    if vmax == vmin:
        return 0
    frac = (vmax - value) / (vmax - vmin) if lower_better else (value - vmin) / (vmax - vmin)
    return round(frac * 100)


def fmt_cell(
    model: str,
    center: float,
    lo: float,
    hi: float,
    scale100: bool,
    ref_zero: bool,
    n: int,
    is_best: bool,
) -> str:
    """One LaTeX cell: optional color + ``\\est{value}{upper}{lower}``."""
    if ref_zero and model == REFERENCE:
        return r"$0.0$"  # baseline reference: plain, no CI, no color
    if scale100:
        s, num = 100.0, f"{center * 100:+.1f}"
    else:
        s, num = 1.0, f"{center:.2f}"  # rank: 2 decimals, no sign
    up = f"{(hi - center) * s:.1f}" if scale100 else f"{(hi - center):.2f}"
    down = f"{(center - lo) * s:.1f}" if scale100 else f"{(center - lo):.2f}"
    body = rf"\mathbf{{{num}}}" if is_best else num
    color = rf"\cellcolor{{customblue!{n}}}" if n > 0 else ""
    return rf"{color}\est{{{body}}}{{{up}}}{{{down}}}"


def build_body(cols: list[dict[str, tuple[float, float, float]]]) -> str:
    """Render the grouped data rows in the arXiv ``\\est``/``tabularx`` layout.

    Each cell sits on its own line (leading ``& ``) and rows are blank-line
    separated, matching the hand-written table so the diff is numbers/colors
    only. Groups are separated by ``\\specialrule`` + a ``\\rowcolor`` title row.
    """
    lines: list[str] = []

    # Per-column min/max over ALL models (single section, global gradient).
    bounds = []
    for ci in range(len(COLUMNS)):
        vals = [cols[ci][m][0] for m in MODELS if m in cols[ci]]
        bounds.append((min(vals), max(vals)) if vals else (0.0, 0.0))

    for gi, grp in enumerate(GROUP_ORDER):
        if gi > 0:
            lines.append(r"\specialrule{\lightrulewidth}{0pt}{0pt}")
        lines.append(r"\rowcolor[HTML]{EFEFEF}")
        lines.append(rf"\multicolumn{{{NCOL}}}{{l}}{{\textit{{{GROUP_TITLE[grp]}}}}} \\")
        lines.append("")
        members = [m for m, (_, g) in MODELS.items() if g == grp]  # fixed dict order
        for m in members:
            label = MODELS[m][0]
            row = [label]
            for ci, col in enumerate(COLUMNS):
                _h, _f, _sc, _ctr, _lo, _hi, scale100, lower, ref_zero, _metric = col
                center, lo, hi = cols[ci][m]
                vmin, vmax = bounds[ci]
                n = intensity(center, vmin, vmax, lower)
                best_val = vmin if lower else vmax
                is_best = (center == best_val) and (m != REFERENCE)
                row.append("& " + fmt_cell(m, center, lo, hi, scale100, ref_zero, n, is_best))
            row[-1] = row[-1] + r" \\"
            lines.append("\n".join(row))
            lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results-dir", type=Path, default=Path("forecasting_bca_20260618"))
    p.add_argument(
        "--out",
        type=Path,
        default=Path.home() / "MHC-benchmark/paper/sections_arxiv/forecasting_main_results_table.tex",
    )
    p.add_argument("--dry-run", action="store_true", help="Print to stdout instead of writing.")
    args = p.parse_args()

    cols = load_columns(args.results_dir)
    missing = [m for m in MODELS if m not in cols[0]]
    if missing:
        raise SystemExit(f"Models missing from CSVs: {missing}")

    table = HEADER + build_body(cols) + FOOTER
    if args.dry_run:
        print(table)
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(table)
    print(f"Wrote {args.out} ({len(table)} bytes)")


if __name__ == "__main__":
    main()
