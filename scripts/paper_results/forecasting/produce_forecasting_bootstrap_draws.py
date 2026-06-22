"""Build the forecasting bootstrap-draws reference for the HF leaderboard dataset.

Mirrors imputation's ``bootstrap/draws.parquet``, adapted to forecasting. Because
forecasting aggregates tasks -> scopes WITHIN each bootstrap draw, its draws are
per-``(reduction, model, scope[, metric], draw)`` for the three reductions
(skill / rank / fairness), unified in one long frame. The CIs reduce from it the
same way (mean / SE / percentile / BCa over draws). Writes:

    <out>/bootstrap_draws.parquet         # per-draw long frame
    <out>/bootstrap_draws.parquet.meta.json

Upload with:
    python tools/upload_leaderboard_bootstrap.py --dir <out> --track forecasting

Heavy (substrate build + 1000-draw skill/rank + fairness bootstraps) — run via
``produce_forecasting_bootstrap_draws.sbatch`` on SLURM.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd  # noqa: E402

from forecasting_evaluation.metrics import metric_spec as _spec  # noqa: E402
from forecasting_evaluation.metrics.bootstrap_fair_skill_score import (  # noqa: E402
    bootstrap_fair_skill_score,
)
from forecasting_evaluation.metrics.bootstrap_skill_rank import bootstrap_skill_rank  # noqa: E402
from forecasting_evaluation.metrics.per_user_errors import build_per_user_metrics  # noqa: E402

N_BOOT, SEED, CI_LEVEL = 1000, 42, 0.95
AGE_BINS = (18, 30, 40, 50, 60)
BINARY_GROUPS = [(name, tuple(idx)) for name, idx in _spec.BINARY_GROUPS]
DRAWS_COLUMNS = ["reduction", "model", "scope", "metric", "draw", "value"]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def main() -> int:
    """Build + write the unified forecasting bootstrap-draws parquet + meta."""
    p = argparse.ArgumentParser(description="Build forecasting bootstrap draws reference.")
    p.add_argument(
        "--summary-dir",
        type=Path,
        default=REPO_ROOT / "results/forecasting_eval/simurgh/summary/forecasting_bca_20260618",
    )
    p.add_argument("--out-dir", type=Path, default=None, help="default: --summary-dir")
    args = p.parse_args()
    out_dir = args.out_dir or args.summary_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    models = json.loads((args.summary_dir / "skill_rank_models.json").read_text())
    models = models.get("models", models)
    models = {
        name: {
            "path": str(path if Path(path).is_absolute() else REPO_ROOT / path),
            "display_name": name,
        }
        for name, path in models.items()
    }
    t0 = time.time()
    substrate = build_per_user_metrics(
        models=models,
        continuous_metrics=list(_spec.PAPER_CONTINUOUS_METRICS),
        binary_metrics=list(_spec.PAPER_BINARY_METRICS),
        continuous_channel_indices=_spec.CONTINUOUS_CHANNELS,
        binary_channel_indices=_spec.BINARY_CHANNELS,
    )
    print(f"substrate: {len(substrate)} rows ({time.time() - t0:.0f}s)")

    sr = bootstrap_skill_rank(
        models=models,
        baseline_model=_spec.PAPER_BASELINE,
        continuous_metrics=list(_spec.PAPER_CONTINUOUS_METRICS),
        binary_metrics=list(_spec.PAPER_BINARY_METRICS),
        continuous_channel_indices=_spec.CONTINUOUS_CHANNELS,
        binary_channel_indices=_spec.BINARY_CHANNELS,
        binary_groups=BINARY_GROUPS,
        n_boot=N_BOOT,
        seed=SEED,
        ci_level=CI_LEVEL,
        within_user_aggregation="micro",
        per_user_metrics=substrate,
        return_draws=True,
    )
    print(f"skill/rank bootstrap done ({time.time() - t0:.0f}s)")

    from labels.api import ENROLLMENT_PATH, LABELS_PATH

    if not LABELS_PATH or not ENROLLMENT_PATH:
        raise SystemExit("labels/enrollment unresolved — set MHC_DATA_DIR.")
    fr = bootstrap_fair_skill_score(
        models=models,
        baseline_model=_spec.PAPER_BASELINE,
        continuous_metrics=list(_spec.PAPER_CONTINUOUS_METRICS),
        binary_metrics=list(_spec.PAPER_BINARY_METRICS),
        continuous_channel_indices=_spec.CONTINUOUS_CHANNELS,
        binary_channel_indices=_spec.BINARY_CHANNELS,
        labels_path=str(LABELS_PATH),
        enrollment_path=str(ENROLLMENT_PATH),
        age_bins=AGE_BINS,
        n_boot=N_BOOT,
        seed=SEED,
        ci_level=CI_LEVEL,
        within_user_aggregation="micro",
        bca=True,
        per_user_metrics=substrate,
        return_draws=True,
    )
    print(f"fairness bootstrap done ({time.time() - t0:.0f}s)")

    frames: list[pd.DataFrame] = []
    sd = sr.get("skill_draws")
    if sd is not None and not sd.empty:
        frames.append(sd.assign(reduction="skill", metric="")[DRAWS_COLUMNS])
    rd = sr.get("rank_draws")
    if rd is not None and not rd.empty:
        frames.append(rd.assign(reduction="rank")[DRAWS_COLUMNS])
    fd = fr.get("fairness_draws")
    if fd is not None and not fd.empty:
        frames.append(fd.assign(reduction="fairness", metric="")[DRAWS_COLUMNS])
    if not frames:
        raise SystemExit("no draws produced — check inputs")
    draws = pd.concat(frames, ignore_index=True)
    draws["draw"] = draws["draw"].astype("int32")
    draws["value"] = draws["value"].astype("float32")
    for col in ("reduction", "model", "scope", "metric"):
        draws[col] = draws[col].astype("category")

    pq = out_dir / "bootstrap_draws.parquet"
    draws.to_parquet(pq, compression="zstd")
    meta = {
        "n_boot": N_BOOT,
        "seed": SEED,
        "ci_level": CI_LEVEL,
        "splits": ["test"],
        "baseline": _spec.PAPER_BASELINE,
        "methods": sorted(models),
        "continuous_metrics": list(_spec.PAPER_CONTINUOUS_METRICS),
        "binary_metrics": list(_spec.PAPER_BINARY_METRICS),
        "age_bins": list(AGE_BINS),
        "reductions": ["skill", "rank", "fairness"],
        "within_user_aggregation": "micro",
        "aggregation_unit": "user",
        "format": "long: (reduction, model, scope, metric, draw, value); "
        "metric is '' for skill/fairness",
        "n_rows": int(len(draws)),
        "elapsed_seconds": round(time.time() - t0, 1),
        "git_commit": _git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
    }
    (out_dir / "bootstrap_draws.parquet.meta.json").write_text(json.dumps(meta, indent=2, default=str))
    counts = draws.groupby("reduction", observed=True).size().to_dict()
    print(f"wrote {len(draws)} rows -> {pq}\n  per reduction: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
