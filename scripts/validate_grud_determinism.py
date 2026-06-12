"""GRU-D training-stack determinism gate — run before accepting any gru_d refactor.

Two same-seed fits of the real ``_Trainer`` on a subsample of real segments with
deterministic synthetic labels; byte-compares final weights + predictions. On CPU
(deterministic kernels) the two runs MUST be byte-identical — any difference is a
real semantic regression. GPU runs are only reproducible to within training
variance (CUDA index_add_), so this CPU gate is gru_d's no-op check.

Usage: MHC_DATA_DIR=... PYTHONPATH=src python scripts/validate_grud_determinism.py  (~15 min)
"""

import numpy as np
import torch

import downstream_evaluation.models.grud as grud

torch.set_num_threads(8)
grud.EPOCHS = 2
grud.PATIENCE = 99

from downstream_evaluation.data.loader import DataLoader as MHCLoader

dl = MHCLoader(None)
values, _mask, users = dl.segment_store()

# subsample: first 120 distinct users in store order
seen: dict = {}
for i, u in enumerate(users):
    seen.setdefault(u, []).append(i)
sub_users = list(seen)[:120]
idx = np.concatenate([np.asarray(seen[u]) for u in sub_users])
X, uids = values[idx], users[idx]
print(f"subsample: {len(sub_users)} users, {len(idx)} segments", flush=True)


def lab(u: str, mod: int) -> int:  # stable across runs/processes
    return int(hashlib.md5(u.encode()).hexdigest(), 16) % mod


TASKS = ["t_bin", "t_reg", "t_ord"]
y = {
    "t_bin": np.asarray([lab(u, 2) for u in uids], dtype=float),
    "t_reg": np.asarray([float(lab(u, 1000)) / 100.0 for u in uids]),
    "t_ord": np.asarray([lab(u, 4) for u in uids], dtype=float),
}
tr_users = set(sub_users[:90])
m_tr = np.asarray([u in tr_users for u in uids])


def make_ds(m):
    return grud._UserDataset(X[m], {t: v[m] for t, v in y.items()}, uids[m], TASKS)


def run_once():
    train_ds, val_ds = make_ds(m_tr), make_ds(~m_tr)
    tr = grud._Trainer({"t_bin": 2}, ["t_reg"], {"t_ord": 4}, "cpu", seed=42)
    tr.fit(train_ds, val_ds, n_steps=24, n_features=19)
    sd = tr.model.state_dict()
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].cpu().numpy().tobytes())
    preds = np.concatenate(
        [tr.predict(val_ds, t, tt) for t, tt in
         [("t_bin", "binary"), ("t_reg", "regression"), ("t_ord", "ordinal")]]
    )
    return h.hexdigest()[:16], hashlib.sha256(preds.tobytes()).hexdigest()[:16]


a = run_once()
print("run A: weights", a[0], "preds", a[1], flush=True)
b = run_once()
print("run B: weights", b[0], "preds", b[1], flush=True)
print(
    "VERDICT:",
    "IDENTICAL — training stack is seed-deterministic on CPU"
    if a == b
    else "DIFFER — non-determinism inside the training stack even on CPU (real bug)",
)
