"""Stage the forecasting per-method substrates for the HF leaderboard dataset.

Splits the canonical 10-model per-user substrate into one parquet per method
(``<out>/<method>.parquet``, schema unchanged) — the leaderboard reference set
the maintainers use to score new submissions — and prints the
``tools/upload_leaderboard_substrate.py`` commands to push each to
``MyHeartCounts/OpenMHC-leaderboard-data`` at ``forecasting/<method>.parquet``
(with a display sidecar).

HF auth (``HF_TOKEN`` or ``huggingface-cli login``) + the ``[hf]`` extra are
required for the actual push; this script only stages the files + emits the
commands, so it runs anywhere.

Usage::

    python scripts/paper_results/forecasting/stage_leaderboard_substrates.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from forecasting_evaluation.metrics.per_user_errors import (  # noqa: E402
    read_per_user_metrics_parquet,
    write_per_user_metrics_parquet,
)

# Display metadata (name, type) for the leaderboard sidecars.
METHOD_META: dict[str, tuple[str, str]] = {
    "seasonal_naive": ("Seasonal Naive (baseline)", "Statistical"),
    "autoARIMA": ("AutoARIMA", "Statistical"),
    "autoETS": ("AutoETS", "Statistical"),
    "chronos2_zeroshot": ("Chronos-2 (zero-shot)", "Foundation Model"),
    "chronos2_finetuned": ("Chronos-2 (fine-tuned)", "Foundation Model"),
    "toto_zeroshot_ctx4096": ("Toto (zero-shot, ctx4096)", "Foundation Model"),
    "toto_finetuned_ctx4096": ("Toto (fine-tuned, ctx4096)", "Foundation Model"),
    "dlinear": ("DLinear", "Deep Learning"),
    "mixlinear": ("MixLinear", "Deep Learning"),
    "segrnn": ("SegRNN", "Deep Learning"),
}
DEFAULT_SUBSTRATE = (
    REPO_ROOT
    / "results/forecasting_eval/simurgh/summary/forecasting_full_20260622"
    / "forecasting_per_user_errors.parquet"
)
DEFAULT_RUNS_ROOT = REPO_ROOT / "results/forecasting_eval/simurgh"


def main() -> int:
    """Split the canonical substrate per method and print the upload commands."""
    p = argparse.ArgumentParser(description="Stage forecasting per-method leaderboard substrates.")
    p.add_argument("--substrate", type=Path, default=DEFAULT_SUBSTRATE)
    p.add_argument("--out", type=Path, default=DEFAULT_SUBSTRATE.parent / "leaderboard_substrates")
    p.add_argument("--submitter", default="OpenMHC team")
    p.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help="Per-model runs root; results.json at <runs-root>/<method>/hydra/results.json is "
        "passed as --results-json so the uploader auto-fills the fallback_rate sidecar key.",
    )
    args = p.parse_args()

    df, _ = read_per_user_metrics_parquet(args.substrate)
    args.out.mkdir(parents=True, exist_ok=True)
    methods = sorted(df["model"].astype(str).unique())
    upload = REPO_ROOT / "tools" / "upload_leaderboard_substrate.py"

    print(f"# Staged {len(methods)} per-method substrates from {args.substrate}\n")
    cmds: list[str] = []
    for method in methods:
        sub = df[df["model"].astype(str) == method].reset_index(drop=True)
        dest = args.out / f"{method}.parquet"
        write_per_user_metrics_parquet(sub, dest)
        name, mtype = METHOD_META.get(method, (method, "—"))
        results_json = args.runs_root / method / "hydra" / "results.json"
        cmds.append(
            f'python {upload} --dir {args.out} --method {method} --track forecasting '
            f'--name "{name}" --type "{mtype}" --submitter "{args.submitter}" '
            f"--results-json {results_json}"
        )
        print(f"  {method}: {len(sub)} rows -> {dest}")

    print("\n# Upload (run where HF auth is available — HF_TOKEN or `huggingface-cli login`):")
    print("\n".join(cmds))
    return 0


if __name__ == "__main__":
    sys.exit(main())
