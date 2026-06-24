# OpenMHC leaderboard — maintainer runbook

Operational notes for the people who run the leaderboard. Submitters don't need
this — see the README "Submit to the Leaderboard" section and
`tools/leaderboard_docs/` instead.

## Two Hugging Face repos

The live leaderboard is two HF repos, separate from this code repo. Auth uses
standard `huggingface_hub` token discovery (`HF_TOKEN` env or
`huggingface-cli login`) with an org-write token on `MyHeartCounts`.

- **Space** `MyHeartCounts/OpenMHC` (`repo_type="space"`, Docker/FastAPI) — the
  live leaderboard *website* (https://myheartcounts-openmhc.hf.space). It
  self-renders HTML and computes the table in-process at startup. There is no
  static `leaderboard.json` and no `/api/data` endpoint.
- **Dataset** `MyHeartCounts/OpenMHC-leaderboard-data` (`repo_type="dataset"`,
  public) — the per-method substrate: `<track>/<method>.parquet` plus display
  sidecars `<track>/<method>.meta.json`
  (`{display_name, type, submitter, subtrack[, overall_fallback_rate]}`). Track 2
  ships `imputation/<method>.parquet` (per-user errors); Track 1 ships
  `downstream/<method>.parquet` (per-user **prediction pairs** — its
  ranking/correlation metrics don't decompose per user, see
  `tools/leaderboard_docs/downstream/SCHEMA.md`). There is also an
  `imputation/bootstrap/` subdir (the per-draw CI reference frame) that is **not**
  used by the live point compute.

## How the Space computes

`leaderboard_compute.py` (in the Space) `snapshot_download`s
`imputation/*.parquet` + `*.meta.json`, then calls the canonical `openmhc`
reducers:

- **Skill** — `compute_per_task_paired_R(baseline_method="locf")` →
  `compute_skill_scores(mode="paired")`.
- **Rank** — `compute_average_rankings` → `aggregate_task_ranks_to_scopes`.
- **Fair skill** — `compute_fair_skill_scores` (disparity-ratio MAPD) at
  `scope="overall"`.

`HEADLINE_SCOPE="overall"`. Skill and rank filter to `subgroup_attr == "all"`;
fairness uses the full frame (it needs the `age_group` / `sex` subgroup rows).

### Fairness gotcha (load-bearing)

`compute_fair_skill_scores` consumes **per-cell MEAN** errors, not raw per-user
rows. Collapse first:

```python
per_cell = df.groupby(
    ["method", "scenario", "channel", "channel_type",
     "subgroup_attr", "subgroup_value"],
    observed=True,
)["E"].mean().reset_index()
```

Feeding raw per-user rows makes the inner merge blow up cartesian-style and
OOM-kill the Space. Pass the full frame (including the `age_group` / `sex` rows);
the reducer ignores `subgroup_attr == "all"` on its own.

## Adding / updating a method (no Space rebuild)

1. Upload the substrate to the **dataset** (`--track imputation` or `downstream`):

   ```bash
   python tools/upload_leaderboard_substrate.py \
       --dir <dir> --method <m> --track imputation \
       --name "<Display>" --type "<Type>" --submitter "<Team>" \
       --fallback-rate <Results.overall_fallback_rate>
   ```

   This writes both `<m>.parquet` and the `<m>.meta.json` sidecar.
   `--fallback-rate` records `overall_fallback_rate` in the sidecar (issue #39);
   omit it and the value is read from the substrate's `<m>.parquet.meta.json`
   provenance sidecar if present, else the row shows "n/a". The `method` column is
   validated against `<m>` for `imputation` and `downstream`.
2. The Space recomputes only on **restart** (in-process cache, no auto-refresh).
   Restart it to pick up the new/changed method. Uploading to the dataset does
   *not* trigger a Space rebuild; only pushing to the Space does.

   > **Track 1 (downstream) Space-side work is not yet wired** (separate Space
   > repo). The substrate is per-user *pairs*, so the Space's Track-1 recompute
   > must run the downstream bootstrap reducers
   > (`downstream_evaluation/evaluation/bootstrap_skill_rank.py`) on the pairs
   > vs. the `linear` baseline, and surface the `overall_fallback_rate` column —
   > it can't reuse the imputation per-cell-mean path verbatim.

Dataset visibility, if needed:
`HfApi().update_repo_settings(repo_id, private=False, repo_type="dataset")`
(the method is `update_repo_settings` — `update_repo_visibility` was renamed in
current `huggingface_hub`).

## Editing & deploying the Space (use HfApi, not git push)

1. Clone (ephemeral — re-clone each session):
   `git clone https://huggingface.co/spaces/MyHeartCounts/OpenMHC`.
   Files: `app.py`, `leaderboard_compute.py`, `Dockerfile`, `requirements.txt`,
   `logo.png`.
2. Edit locally and smoke-test in a slim venv that mirrors the Space (slim deps +
   `pip install -e . --no-deps`), e.g.
   `PYTHONPATH=<clone> python -c "import app; app.index()"`.
3. Deploy:

   ```python
   HfApi().upload_folder(
       folder_path=...,
       repo_id="MyHeartCounts/OpenMHC",
       repo_type="space",
       allow_patterns=[...changed files...],
       ignore_patterns=[".git/**", "**/__pycache__/**", "*.pyc"],
       commit_message=...,
   )
   ```

   This rebuilds the Docker Space.
4. Verify: poll `HfApi().space_info("MyHeartCounts/OpenMHC").runtime.stage`
   (`BUILDING` / `RUNNING_BUILDING` → `RUNNING`). On `BUILD_ERROR`, read the
   build logs via the Space's `logs/build` API endpoint. Then `curl` `/health`
   and `/`.

## Build / runtime gotchas

- `python:*-slim` has **no git** → the `Dockerfile` must `apt-get install git`
  before the `openmhc` VCS pip install.
- Install `openmhc` **`--no-deps`** plus a slim `requirements.txt` (no torch).
  The point-flow reducers are torch-free but still pull `scikit-learn`, `scipy`,
  and `datasets` (via `data.processing.hf_config`).
- Pin `pandas>=2.1,<3`: `compute_average_rankings` needs `future_stack` (≥2.1);
  pandas 3.0 is untested.
- Pin `openmhc` to a **commit SHA, not `@main`**: HF caches the Docker pip
  layer, so a bare `@main` silently serves stale code on rebuilds that don't
  touch the `Dockerfile`. Bump the pinned commit whenever the reducers change.
