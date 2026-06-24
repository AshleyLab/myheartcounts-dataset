#!/usr/bin/env python3
"""Generate the arXiv *main* imputation results table — sourced from HF.

The OpenMHC leaderboard HF dataset
(``MyHeartCounts/OpenMHC-leaderboard-data``) is the single source of truth.
This script pulls the bootstrap **substrate** from there —
``imputation/bootstrap/draws.parquet`` (per-draw E/R/rank) plus the 16
per-method ``imputation/<method>.parquet`` files (the BCa leave-one-user-out
jackknife substrate) — runs the canonical Phase-2 reducers, and renders the
table. The local ``imputation_results_paper/`` directory is no longer read.

Because the Phase-2 reduction is expensive (~25-30 min, dominated by the
fairness BCa jackknife over ~1.8k users), the three reduced CSVs are cached
locally keyed by an **md5 fingerprint of the HF source files**. Each cache
entry records the per-file ``sha256`` it was built from; when the live HF
files differ from the cached fingerprint, the cache is rebuilt automatically.

Emits the full ``\\begin{table}...\\end{table}`` block for
``tab:imputation_main_results``, matching the existing hand-written arXiv
table (booktabs rules, per-section ``customblue!N`` blue gradient, bold best
cell).

Uncertainty is reported as a 95% bootstrap confidence interval rendered as an
asymmetric ``value^{+upper}_{-lower}`` super/subscript (not the SE). For every
column except ``S_fair`` the center is the bootstrap ``mean`` and the interval is
the percentile CI (``ci_lo``/``ci_hi``). ``S_fair`` (from
``fairness_skill_score_bootstrap.csv``, scope ``overall``) uses the
deterministic ``point`` estimate and the BCa interval (``bca_lo``/``bca_hi``),
matching ``build_leaderboard_json.py`` / the leaderboard.

Only the main table is generated; the skill-by-scenario appendix table is out of
scope (it includes a dense "LSM2 (7-day)" row that is not in the bootstrap CSVs).

LOCF is the reference: in every skill column it renders as a plain ``$0.0$`` (no
CI, no color); it still gets a real value+CI in the rank column.

Usage:
    # builds (or reuses) the cache from HF, then writes the table
    python scripts/paper_results/imputation/make_imputation_latex_tables.py \
        --out ~/MHC-benchmark/paper/sections_arxiv/imputation_main_results_table.tex

    # force a rebuild of the cached reduction even if the fingerprint matches
    python scripts/paper_results/imputation/make_imputation_latex_tables.py \
        --force --dry-run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("make_imputation_latex_tables")

# HF dataset (universal source of truth) and the substrate paths within it.
DEFAULT_REPO_ID = "MyHeartCounts/OpenMHC-leaderboard-data"
DRAWS_PATH = "imputation/bootstrap/draws.parquet"
DRAWS_META_PATH = "imputation/bootstrap/draws.meta.json"
DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "openmhc" / "imputation_paper_tables"

# The three reduced CSVs the renderer consumes (filenames match the canonical
# Phase-2 sidecars so the cache is drop-in compatible with load_columns()).
REDUCED_CSVS = (
    "skill_scores_bootstrap.csv",
    "avg_rankings_bootstrap.csv",
    "fairness_skill_score_bootstrap.csv",
)
CACHE_MANIFEST = "manifest.json"

# ---------------------------------------------------------------------------
# Method registry: key -> (latex_label, context, model_group)
#   context     : "single" (single-day) | "long" (>= 7-day context)
#   model_group : "stat" (Statistical Models) | "neural" (Neural Models)
# Labels mirror the current arXiv table verbatim (incl. \textsc + \citep).
# ---------------------------------------------------------------------------
REFERENCE = "locf"

METHODS: dict[str, tuple[str, str, str]] = {
    # single-day -- statistical
    "linear": (r"Linear", "single", "stat"),
    "temporal_mean": (r"Temporal mean", "single", "stat"),
    "locf": (r"LOCF \textit{(reference)}", "single", "stat"),
    "temporal_mode": (r"Temporal mode", "single", "stat"),
    "mode": (r"Mode", "single", "stat"),
    "mean": (r"Mean", "single", "stat"),
    # single-day -- neural
    "lsm2": (r"\textsc{LSM-2}~\citep{xu2025lsm}", "single", "neural"),
    "dlinear": (r"DLinear~\citep{zeng2023dlinear}", "single", "neural"),
    "brits": (r"BRITS~\citep{cao2018brits}", "single", "neural"),
    "timesnet": (r"TimesNet~\citep{wu2023timesnet}", "single", "neural"),
    "fedformer": (r"FEDformer~\citep{zhou2022fedformer}", "single", "neural"),
    # long-context -- statistical
    "personalized_temporal_mean": (r"Personalized\ temp.\ mean", "long", "stat"),
    "personalized_mean": (r"Personalized\ mean", "long", "stat"),
    "personalized_mode": (r"Personalized\ mode", "long", "stat"),
    # long-context -- neural
    "lsm2_weekly_sparse": (r"\textsc{LSM-2-Sparse} (7-day)", "long", "neural"),
    "dlinear_weekly": (r"DLinear (7-day)~\citep{zeng2023dlinear}", "long", "neural"),
}

# Column spec: (header, csv, scope, center, lo, hi, scale100, lower_better, ref_zero)
#   scale100  : skill ratios rendered x100 with a sign; rank rendered as-is
#   lower_better: invert the color gradient (best = lowest)
#   ref_zero  : LOCF renders as a plain "$0.0$" (skill columns; baseline == 0)
COLUMNS: list[tuple[str, str, str, str, str, str, bool, bool, bool]] = [
    (r"$R\downarrow$",            "avg_rankings_bootstrap.csv",          "overall",        "mean",  "ci_lo",  "ci_hi",  False, True,  False),
    (r"$S\uparrow$",              "skill_scores_bootstrap.csv",          "overall",        "mean",  "ci_lo",  "ci_hi",  True,  False, True),
    (r"$S_{\text{fair}}\uparrow$","fairness_skill_score_bootstrap.csv",  "overall",        "point", "bca_lo", "bca_hi", True,  False, True),
    (r"Activity\,$\uparrow$",     "skill_scores_bootstrap.csv",          "cat:activity",   "mean",  "ci_lo",  "ci_hi",  True,  False, True),
    (r"Physio.\,$\uparrow$",      "skill_scores_bootstrap.csv",          "cat:physiology", "mean",  "ci_lo",  "ci_hi",  True,  False, True),
    (r"Sleep\,$\uparrow$",        "skill_scores_bootstrap.csv",          "cat:sleep",      "mean",  "ci_lo",  "ci_hi",  True,  False, True),
    (r"Workout\,$\uparrow$",      "skill_scores_bootstrap.csv",          "cat:workouts",   "mean",  "ci_lo",  "ci_hi",  True,  False, True),
    (r"Semantic\,$\uparrow$",     "skill_scores_bootstrap.csv",          "semantic",       "mean",  "ci_lo",  "ci_hi",  True,  False, True),
]

NCOL = len(COLUMNS) + 1  # + method column

SECTION_TITLE = {
    "single": r"\textbf{\emph{Single-day imputation}}",
    "long": r"\textbf{\emph{Long-context imputation ($\geq 7 \times 1440$ time steps)}}",
}
GROUP_TITLE = {
    "stat": r"\cellcolor[HTML]{EFEFEF}\textit{Statistical Models}",
    "neural": r"\cellcolor[HTML]{EFEFEF}\textit{Neural Models}",
}

HEADER = r"""\begin{table}[b!]
    \vspace{-2mm}
    \renewcommand{\arraystretch}{1.05}
    \centering
    \captionsetup{width=\textwidth}
    \caption{\textbf{Imputation Results.} We report Average Rank $R$, Aggregate Skill Score $S$ (in \%; $0=\TN{LOCF}$ reference), Fairness-Adjusted Skill Score $S_{\text{fair}}$, and Channel-Specific Skill Scores for the following channels: \textit{Activity, Physiology, Sleep, Workout}. Finally, we also report performance on all \textit{Semantic} masking approaches (see Appendix \ref{sec:imputation}). Single-day imputation method results are in the upper section of the table; long-context imputation method results ($\geq 7\times 1440$ time steps) are below. %
    Sub/superscripts give the $95\%$ bootstrap confidence interval ($1000$ resamples); $S_{\text{fair}}$ uses the bias-corrected and accelerated (BCa) interval about its point estimate, all other columns the percentile interval about the bootstrap mean.
    }
    \label{tab:imputation_main_results}
    \small
    \setlength{\tabcolsep}{1.5pt}
    \resizebox{\linewidth}{!}{%
    \begin{tabular}{l cccccccc}
    \toprule[1.5pt]
    \textbf{Method} & $R\downarrow$ & $S\uparrow$ & $S_{\text{fair}}\uparrow$ & Activity\,$\uparrow$ & Physio.\,$\uparrow$ & Sleep\,$\uparrow$ & Workout\,$\uparrow$ & Semantic\,$\uparrow$ \\
    \midrule
"""

FOOTER = r"""    \bottomrule[1.5pt]
    \end{tabular}%
    }
    \vspace{-2mm}
\end{table}
"""


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


def load_metric(
    path: Path, scope: str, center_col: str, lo_col: str, hi_col: str
) -> dict[str, tuple[float, float, float]]:
    """Return {method: (center, lo, hi)} for one scope on the test split."""
    out: dict[str, tuple[float, float, float]] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        missing = [c for c in (center_col, lo_col, hi_col) if c not in fields]
        if missing:
            raise SystemExit(f"{path.name}: columns {missing} not in header {fields!r}")
        for r in reader:
            if r.get("split") != "test" or r["scope"] != scope:
                continue
            c = r[center_col]
            if c in ("", None):
                continue
            center = float(c)
            lo = float(r[lo_col]) if r[lo_col] not in ("", None) else center
            hi = float(r[hi_col]) if r[hi_col] not in ("", None) else center
            out[r["method"]] = (center, lo, hi)
    return out


def load_columns(results_dir: Path) -> list[dict[str, tuple[float, float, float]]]:
    """Load each column's {method: (center, lo, hi)} map, in COLUMNS order."""
    return [
        load_metric(results_dir / fname, scope, center, lo, hi)
        for _h, fname, scope, center, lo, hi, _s100, _lower, _ref in COLUMNS
    ]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def intensity(value: float, vmin: float, vmax: float, lower_better: bool) -> int:
    """Per-section, per-column min-max intensity in [0, 100]."""
    if vmax == vmin:
        return 0
    frac = (vmax - value) / (vmax - vmin) if lower_better else (value - vmin) / (vmax - vmin)
    return round(frac * 100)


def fmt_cell(
    method: str,
    center: float,
    lo: float,
    hi: float,
    scale100: bool,
    ref_zero: bool,
    n: int,
    is_best: bool,
) -> str:
    """One LaTeX cell: optional color + ``$value^{+upper}_{-lower}$``."""
    if ref_zero and method == REFERENCE:
        return r"$0.0$"  # baseline reference: plain, no CI, no color
    s = 100.0 if scale100 else 1.0
    num = f"{center * s:+.1f}" if scale100 else f"{center * s:.1f}"
    up = f"{(hi - center) * s:.1f}"
    down = f"{(center - lo) * s:.1f}"
    body = rf"\mathbf{{{num}}}" if is_best else num
    color = rf"\cellcolor{{customblue!{n}}}" if n > 0 else ""
    return rf"{color}${body}^{{+{up}}}_{{-{down}}}$"


def build_body(cols: list[dict[str, tuple[float, float, float]]]) -> str:
    lines: list[str] = []

    for ctx in ("single", "long"):
        if ctx == "long":
            lines.append(r"    \midrule")
        lines.append(rf"    \multicolumn{{{NCOL}}}{{l}}{{{SECTION_TITLE[ctx]}}} \\")
        lines.append(r"    \hline")

        # Per-section, per-column min/max over the methods present in this section.
        section_methods = [m for m, (_, c, _) in METHODS.items() if c == ctx]
        bounds = []  # (vmin, vmax) of the center value per column
        for ci in range(len(COLUMNS)):
            vals = [cols[ci][m][0] for m in section_methods if m in cols[ci]]
            bounds.append((min(vals), max(vals)) if vals else (0.0, 0.0))

        for gi, grp in enumerate(("stat", "neural")):
            if gi > 0:
                lines.append(r"    \hline")
            lines.append(rf"    \multicolumn{{{NCOL}}}{{l}}{{{GROUP_TITLE[grp]}}} \\")

            members = [m for m, (_, c, g) in METHODS.items() if c == ctx and g == grp]
            # order by overall skill (column index 1) descending
            members.sort(key=lambda m: -cols[1][m][0])

            for m in members:
                label = METHODS[m][0]
                cells = []
                for ci, col in enumerate(COLUMNS):
                    _h, _f, _sc, _ctr, _lo, _hi, scale100, lower, ref_zero = col
                    center, lo, hi = cols[ci][m]
                    vmin, vmax = bounds[ci]
                    n = intensity(center, vmin, vmax, lower)
                    best_val = vmin if lower else vmax
                    is_best = (center == best_val) and (m != REFERENCE)
                    cells.append(fmt_cell(m, center, lo, hi, scale100, ref_zero, n, is_best))
                lines.append(rf"    {label} & " + " & ".join(cells) + r" \\")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# HF substrate -> reduced CSVs (md5-fingerprinted local cache)
# ---------------------------------------------------------------------------


def _remote_fingerprint(repo_id: str, revision: str | None) -> dict[str, str]:
    """Map each substrate parquet to its content hash on HF (no full download).

    Returns ``{repo_path: sha256}`` for ``draws.parquet`` and every per-method
    ``imputation/<method>.parquet`` (the BCa jackknife substrate). Parquet
    files are LFS-tracked, so ``lfs.sha256`` is the content identity; the git
    ``blob_id`` is a fallback for any non-LFS entry.
    """
    from huggingface_hub import HfApi

    tree = HfApi().list_repo_tree(
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo="imputation",
        revision=revision,
        recursive=True,
        expand=True,
    )
    fp: dict[str, str] = {}
    for item in tree:
        path = getattr(item, "path", None)
        if path is None or not path.endswith(".parquet"):
            continue  # ignore .meta.json / .md sidecars
        # per-method substrate = imputation/<method>.parquet (one slash);
        # plus the bootstrap draws table.
        is_per_method = path.count("/") == 1
        if not (is_per_method or path == DRAWS_PATH):
            continue
        lfs = getattr(item, "lfs", None)
        sha = lfs.sha256 if lfs is not None else getattr(item, "blob_id", None)
        if sha is None:
            raise SystemExit(f"No content hash available for {path!r} on {repo_id}")
        fp[path] = sha
    if DRAWS_PATH not in fp:
        raise SystemExit(f"{DRAWS_PATH} not found on {repo_id} (revision={revision})")
    return fp


def _cache_key(fingerprint: dict[str, str]) -> str:
    """md5 over the sorted (path, sha256) pairs — the cache identity."""
    payload = json.dumps(fingerprint, sort_keys=True).encode()
    return hashlib.md5(payload).hexdigest()


def _cache_is_valid(cache_dir: Path) -> bool:
    """A cache entry is usable iff the manifest and all reduced CSVs exist."""
    if not (cache_dir / CACHE_MANIFEST).is_file():
        return False
    return all((cache_dir / name).is_file() for name in REDUCED_CSVS)


def _build_reduced_csvs(
    repo_id: str,
    revision: str | None,
    fingerprint: dict[str, str],
    cache_dir: Path,
) -> None:
    """Download the HF substrate, run the Phase-2 reducers, write the cache.

    Heavy imports (pandas, huggingface_hub, the reducer stack) are deferred to
    here so ``--help`` and the render-from-cache path stay light.
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download

    from imputation_evaluation.evaluation.bootstrap_skill_rank import (
        aggregate_skill_rank_fairness,
        read_draws_parquet,
    )

    # compute_fairness_skill_scores lives in scripts/paper_results/, which is
    # not an importable package — add it to the path the same way the pipeline
    # driver shells out to it.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from aggregate_fairness_skill_score import (  # noqa: E402
        SENSITIVE_ATTRS,
        compute_fairness_skill_scores,
    )

    def _dl(path: str) -> str:
        return hf_hub_download(
            repo_id=repo_id, filename=path, repo_type="dataset", revision=revision
        )

    logger.info("Downloading substrate from %s (revision=%s) …", repo_id, revision or "main")
    draws_path = _dl(DRAWS_PATH)
    per_method_paths = sorted(p for p in fingerprint if p != DRAWS_PATH)
    local_per_method = [_dl(p) for p in per_method_paths]
    try:
        draws_meta = json.loads(Path(_dl(DRAWS_META_PATH)).read_text())
    except Exception:  # sidecar is optional provenance only
        draws_meta = {}

    draws_df, _ = read_draws_parquet(Path(draws_path))
    logger.info("Loaded %d draw rows; reducing skill + rank …", len(draws_df))
    # Skill + rank only ever use the ``subgroup_attr == "all"`` rows. Handing
    # the full frame to aggregate_skill_rank_fairness would additionally fire
    # its deprecated (and here unused) S-λD fairness subgroup loop — the
    # dominant cost. Restricting to the "all" rows yields identical skill /
    # rank tables and skips that loop entirely. The leaderboard fairness score
    # is produced by compute_fairness_skill_scores below (which does need the
    # subgroup rows).
    all_rows = draws_df[draws_df["subgroup_attr"] == "all"]
    tables = aggregate_skill_rank_fairness(all_rows)

    logger.info(
        "Concatenating %d per-method parquets for the BCa jackknife …",
        len(local_per_method),
    )
    per_user_df = pd.concat(
        [pd.read_parquet(p) for p in local_per_method], ignore_index=True
    )
    logger.info(
        "Reducing fairness skill score with BCa over %d users "
        "(this is the slow step, ~25 min) …",
        per_user_df["user_id"].nunique(),
    )
    fairness = compute_fairness_skill_scores(
        draws_df,
        attrs=list(SENSITIVE_ATTRS),
        baseline_method="locf",
        bca=True,
        per_user_df=per_user_df,
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    tables["skill_scores"].to_csv(
        cache_dir / "skill_scores_bootstrap.csv", index=False, float_format="%.6f"
    )
    tables["avg_rankings"].to_csv(
        cache_dir / "avg_rankings_bootstrap.csv", index=False, float_format="%.6f"
    )
    fairness.to_csv(
        cache_dir / "fairness_skill_score_bootstrap.csv", index=False, float_format="%.6f"
    )
    # Manifest written LAST — its presence marks the cache entry complete.
    (cache_dir / CACHE_MANIFEST).write_text(
        json.dumps(
            {
                "repo_id": repo_id,
                "revision": revision,
                "cache_key_md5": _cache_key(fingerprint),
                "source_files": fingerprint,
                "source_git_commit": draws_meta.get("git_commit"),
                "n_boot": draws_meta.get("n_boot"),
                "seed": draws_meta.get("seed"),
                "built_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
    )
    logger.info("Wrote reduced CSVs + manifest to %s", cache_dir)


def ensure_reduced_csvs(
    repo_id: str, revision: str | None, cache_root: Path, force: bool
) -> Path:
    """Return a dir holding the three reduced CSVs, rebuilding if HF changed.

    The cache dir is keyed by the md5 of the live HF source-file fingerprint,
    so a changed substrate naturally resolves to a fresh (empty) dir and
    triggers a rebuild; an unchanged substrate hits the existing entry.
    """
    fingerprint = _remote_fingerprint(repo_id, revision)
    key = _cache_key(fingerprint)
    cache_dir = cache_root / key
    if not force and _cache_is_valid(cache_dir):
        logger.info("Cache hit (%s) — using reduced CSVs at %s", key, cache_dir)
        return cache_dir
    if force:
        logger.info("--force: rebuilding cache %s", cache_dir)
    else:
        logger.info("Cache miss (%s) — building from HF substrate", key)
    _build_reduced_csvs(repo_id, revision, fingerprint, cache_dir)
    return cache_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="HF dataset (source of truth).")
    p.add_argument("--revision", default=None, help="HF revision/branch/commit (default: main).")
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
        help=f"Root for the fingerprinted reduction cache (default: {DEFAULT_CACHE_ROOT}).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Rebuild the reduction cache even if the HF fingerprint matches.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path.home() / "MHC-benchmark/paper/sections_arxiv/imputation_main_results_table.tex",
    )
    p.add_argument("--dry-run", action="store_true", help="Print to stdout instead of writing.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    results_dir = ensure_reduced_csvs(args.repo_id, args.revision, args.cache_dir, args.force)

    cols = load_columns(results_dir)
    missing = [m for m in METHODS if m not in cols[0]]
    if missing:
        raise SystemExit(f"Methods missing from CSVs: {missing}")

    table = HEADER + build_body(cols) + FOOTER
    if args.dry_run:
        print(table)
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(table)
    print(f"Wrote {args.out} ({len(table)} bytes)")


if __name__ == "__main__":
    main()
