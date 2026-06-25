# Imputation bootstrap reference — 17-method (with dense weekly)

Sibling of [`imputation/bootstrap/`](../bootstrap/). Same Phase-1 reducer,
same code path, **but the method pool includes `lsm2_weekly` (the dense
7-day LSM-2 variant)** for 17 methods total instead of 16.

Use this variant when comparing LSM-2 dense-weekly against the rest of the
pool. The canonical 16-method bootstrap at `imputation/bootstrap/` keeps
parity with the published paper's method set.

## Layout

```
imputation/bootstrap_with_dense_weekly/
├── draws.parquet      # per-(method, scenario, channel, subgroup, draw) E/R/rank
└── draws.meta.json    # provenance: seed, n_boot, methods, scenarios, git commit
```

## Differences vs `imputation/bootstrap/`

| | `bootstrap/` (canonical) | `bootstrap_with_dense_weekly/` (this dir) |
|---|---|---|
| Method count | 16 | **17** |
| Includes `lsm2_weekly` (dense 7-day) | no | **yes** |
| Skill / fairness scores | identical to this variant per method (pairwise vs `locf`) | identical to canonical per method |
| Average-rank values | computed across 16-method pool | **shifted** because the comparison pool grew |

**Skill** and **fairness skill score** are computed pairwise against the
LOCF baseline, so removing or adding a non-baseline method doesn't change
any other method's score. **Average rank** depends on the comparison pool
and will differ between the two variants — that's expected and the whole
reason this sibling exists.

## Schema

Identical to the canonical bootstrap — see [`SCHEMA.md`](SCHEMA.md).

## Loading

```python
from huggingface_hub import hf_hub_download
import pandas as pd, json

draws_path = hf_hub_download(
    "MyHeartCounts/OpenMHC-leaderboard-data",
    "imputation/bootstrap_with_dense_weekly/draws.parquet",
    repo_type="dataset",
)
meta_path = hf_hub_download(
    "MyHeartCounts/OpenMHC-leaderboard-data",
    "imputation/bootstrap_with_dense_weekly/draws.meta.json",
    repo_type="dataset",
)
draws = pd.read_parquet(draws_path)
meta = json.loads(open(meta_path).read())
print(meta["seed"], meta["n_boot"], len(meta["methods"]))  # -> 42 1000 17
```

## Uploaded with

`tools/upload_leaderboard_bootstrap.py` (canonical pool) +
`jobs/sherlock/imputation_eval/upload_leaderboard.sbatch` (this sibling) in
the [code repo](https://github.com/AshleyLab/myheartcounts-dataset).
