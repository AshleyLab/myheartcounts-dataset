#!/usr/bin/env python3
"""Watchdog for the imputation-eval SLURM batch.

Reads ``$MANIFEST`` (TSV: jobid, label, script), keeps the *latest* row per
label as the live jobid, queries ``sacct`` for state, and:
  - leaves PENDING / RUNNING alone,
  - declares COMPLETED done iff the expected output exists,
  - resubmits FAILED / TIMEOUT / OOM / CANCELLED / NODE_FAIL,
    rewriting the paper_bootstrap dependency against the new live imputer ids,
  - hard-stops after MAX_RETRIES per label.

Idempotent. Print one screen of status + write a one-line summary to
``$OUT_BASE/babysit.log`` whenever it acts.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ----------------------------- configuration --------------------------------

JOBS_DIR = Path(__file__).resolve().parent
OUT_BASE = Path("/scratch/users/schuetzn/openmhc-imputation-eval")
MANIFEST = OUT_BASE / "job_manifest.tsv"
LOG_FILE = OUT_BASE / "babysit.log"

RUNS_ROOT = OUT_BASE / "runs"
PAPER_OUT = OUT_BASE / "paper"

MAX_RETRIES = 3
TERMINAL_FAIL = {
    "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "CANCELLED", "PREEMPTED",
    "BOOT_FAIL", "DEADLINE", "REVOKED",
}
ALIVE = {"PENDING", "RUNNING", "REQUEUED", "RESIZING", "SUSPENDED", "CONFIGURING"}
DONE = {"COMPLETED"}

IMPUTER_LABELS = [
    "baselines", "brits", "dlinear", "fedformer", "timesnet",
    "lsm2", "lsm2_weekly_sparse",
]

# Per-label expected output. Files that, if present, mean the job *really*
# finished (sacct=COMPLETED alone isn't enough; we still want to see artifacts).
def expected_outputs(label: str) -> list[Path]:
    if label == "baselines":
        return [RUNS_ROOT / m / "results.json" for m in (
            "mean", "mode", "linear", "locf", "temporal_mean", "temporal_mode",
            "personalized_mean", "personalized_mode", "personalized_temporal_mean",
        )]
    if label == "paper_bootstrap":
        return [
            PAPER_OUT / "bootstrap_draws.parquet",
            PAPER_OUT / "skill_scores_bootstrap.csv",
            PAPER_OUT / "avg_rankings_bootstrap.csv",
        ]
    return [RUNS_ROOT / label / "results.json"]


# ----------------------------- helpers --------------------------------------

def run(cmd: list[str]) -> str:
    res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(f"[babysit] {' '.join(cmd)} -> exit {res.returncode}\n")
        sys.stderr.write(res.stderr)
    return res.stdout


def sacct_state(jobid: str) -> str:
    """Return the canonical state string for the parent step of jobid.

    sacct reports each step on its own line (e.g. 12345, 12345.batch). We only
    want the parent's state.
    """
    out = run([
        "sacct", "-j", str(jobid), "--format=JobID,State", "--parsable2", "--noheader",
    ])
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 2:
            continue
        jid = parts[0].split(".")[0]
        if jid == str(jobid).split(".")[0]:
            return parts[1].strip().split()[0]   # strip "CANCELLED by 12345" suffix
    return "UNKNOWN"


def read_manifest() -> list[tuple[str, str, str]]:
    """Return rows as (jobid, label, script). Skip comments + blanks."""
    rows: list[tuple[str, str, str]] = []
    if not MANIFEST.exists():
        return rows
    for line in MANIFEST.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        rows.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))
    return rows


def append_manifest(jobid: str, label: str, script: str) -> None:
    with MANIFEST.open("a") as f:
        f.write(f"{jobid}\t{label}\t{script}\n")


def log(msg: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')}  {msg}\n")
    print(msg)


def submit(script: str, dependency: str | None = None) -> str:
    cmd = ["sbatch", "--parsable"]
    if dependency:
        cmd.append(f"--dependency={dependency}")
    cmd.append(str(JOBS_DIR / script))
    out = run(cmd).strip()
    return out.split(";")[0]  # sbatch may return "12345;cluster"


def outputs_present(label: str) -> bool:
    return all(p.exists() for p in expected_outputs(label))


# ----------------------------- main -----------------------------------------

def main() -> int:
    if not MANIFEST.exists():
        print(f"[babysit] no manifest at {MANIFEST}; nothing to do.")
        return 0

    rows = read_manifest()
    # latest row per label -> live jobid
    latest: dict[str, tuple[str, str]] = {}  # label -> (jobid, script)
    retries: dict[str, int] = {}             # label -> count (resubmissions)
    for jobid, label, script in rows:
        retries[label] = retries.get(label, -1) + 1   # first occurrence = 0
        latest[label] = (jobid, script)

    # Snapshot state for every live label.
    statuses: dict[str, str] = {}
    for label, (jobid, _) in latest.items():
        statuses[label] = sacct_state(jobid)

    # Resubmit imputer jobs first so paper_bootstrap can depend on fresh ids.
    actions: list[str] = []
    for label in IMPUTER_LABELS:
        if label not in latest:
            continue
        st = statuses[label]
        jobid, script = latest[label]
        if st in ALIVE:
            continue
        if st in DONE:
            if outputs_present(label):
                continue
            # sacct says done but artifacts missing - treat as failure.
            log(f"[anomaly] {label} jid={jobid} state=COMPLETED but outputs missing")
            st = "FAILED"
        if st in TERMINAL_FAIL:
            if retries[label] >= MAX_RETRIES:
                log(f"[STUCK]   {label} jid={jobid} state={st} retries={retries[label]} >= {MAX_RETRIES}")
                continue
            new_jid = submit(script)
            append_manifest(new_jid, label, script)
            latest[label] = (new_jid, script)
            statuses[label] = "PENDING"
            log(f"[resub]   {label} {jobid}({st}) -> {new_jid}  (retry {retries[label]+1}/{MAX_RETRIES})")
            actions.append(label)

    # Paper-bootstrap is special: dependency must point at the *current* live
    # imputer jobids whenever we resubmit it.
    if "paper_bootstrap" in latest:
        jobid, script = latest["paper_bootstrap"]
        st = statuses["paper_bootstrap"]
        if st in DONE and outputs_present("paper_bootstrap"):
            pass
        elif st in ALIVE and not actions:
            pass  # still pending/running; dependency still valid
        elif st in ALIVE and actions:
            # An imputer was resubmitted under us — the existing dep references
            # a dead jobid. scontrol-update the dependency in place.
            live_ids = [latest[l][0] for l in IMPUTER_LABELS if l in latest]
            new_dep = "afterok:" + ":".join(live_ids)
            run(["scontrol", "update", f"jobid={jobid}", f"Dependency={new_dep}"])
            log(f"[deprw]   paper_bootstrap jid={jobid} dep={new_dep}")
        elif st in TERMINAL_FAIL or (st in DONE and not outputs_present("paper_bootstrap")):
            if retries["paper_bootstrap"] >= MAX_RETRIES:
                log(f"[STUCK]   paper_bootstrap jid={jobid} state={st}")
            else:
                live_ids = [latest[l][0] for l in IMPUTER_LABELS if l in latest]
                dep = "afterok:" + ":".join(live_ids) if live_ids else None
                new_jid = submit(script, dependency=dep)
                append_manifest(new_jid, "paper_bootstrap", script)
                latest["paper_bootstrap"] = (new_jid, script)
                statuses["paper_bootstrap"] = "PENDING"
                log(f"[resub]   paper_bootstrap {jobid}({st}) -> {new_jid}  dep={dep}")

    # ---- status table ----
    width = max(len(l) for l in latest) + 2
    print()
    print(f"{'label':<{width}} {'jobid':>10} {'state':<14} {'retries':>7}  outputs")
    print("-" * (width + 50))
    for label in IMPUTER_LABELS + ["paper_bootstrap"]:
        if label not in latest:
            continue
        jid, _ = latest[label]
        st = statuses[label]
        n = retries[label]
        present = "OK" if outputs_present(label) else "missing"
        print(f"{label:<{width}} {jid:>10} {st:<14} {n:>7}  {present}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
