"""Multi-task GRU-D — end-to-end supervised baseline (unified Method).

One shared GRU-D backbone (PyPOTS ``BackboneGRUD``) encodes each of a user's daily
``(24, 19)`` segments to a hidden state; the hidden states are mean-pooled across the
user's segments to a single per-user representation, which feeds per-task heads
(softmax for binary/multiclass, linear for regression, K−1 cumulative-link logits for
ordinal). Trained jointly over all tasks with a masked multi-task loss (a user without
a label for task *t* is excluded from task *t*'s term).

This is **trained from scratch** — there is no checkpoint and no determinism beyond a
fixed seed, so results are reproducible only to within training variance, unlike the
deterministic / cached methods.

Engine fit: the per-task ``fit(train_td)`` loop would retrain 32×, so the adapter
trains the one multi-task model on the **first** ``fit`` (assembling every task's
labels from the provider) and serves each task's head on ``predict``.
"""

from __future__ import annotations

import copy
import logging

import numpy as np

logger = logging.getLogger(__name__)

N_SENSOR_CHANNELS = 19
MISSING_LABEL = -1
RNN_HIDDEN_SIZE = 64
BATCH_SIZE = 32  # users per batch
EPOCHS = 50
PATIENCE = 5
LR = 1e-3


def _scatter_mean(src, index, n_groups):
    """Mean-pool ``src`` (M, D) into ``n_groups`` rows by ``index`` (M,)."""
    import torch

    out = torch.zeros(n_groups, src.shape[1], device=src.device, dtype=src.dtype)
    counts = torch.zeros(n_groups, device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    counts.index_add_(0, index, torch.ones(len(index), device=src.device, dtype=src.dtype))
    return out / counts.clamp(min=1.0).unsqueeze(-1)


# --------------------------------------------------------------------------- #
# Per-user dataset: a user's segments + deltas + per-task label
# --------------------------------------------------------------------------- #
class _UserDataset:
    """Torch Dataset of per-user segment groups (one item = one user)."""

    def __init__(self, X, y_by_task, uids, task_names):
        import torch
        from pypots.data.utils import _parse_delta_torch
        from pypots.imputation.locf import locf_torch

        self.task_names = task_names
        X_t = torch.from_numpy(X.astype(np.float32))  # (N, 24, 19), NaN at missing
        self.missing_mask = (~torch.isnan(X_t)).to(torch.float32)  # 1=observed
        self.X_filledLOCF = locf_torch(X_t)
        self.values = torch.nan_to_num(X_t, nan=0.0)
        self.deltas = _parse_delta_torch(self.missing_mask)
        obs = self.missing_mask
        self.empirical_mean = torch.nan_to_num(
            (self.values * obs).reshape(-1, X.shape[2]).sum(0)
            / obs.reshape(-1, X.shape[2]).sum(0),
            nan=0.0,
        )
        # group segment indices by user (preserve first-seen order)
        order, seen = [], {}
        for i, u in enumerate(uids):
            seen.setdefault(u, []).append(i)
        self.user_ids = list(seen.keys())
        self.user_segs = [np.asarray(seen[u], dtype=np.int64) for u in self.user_ids]
        self.y_by_task = {t: y_by_task[t] for t in task_names}
        # per-user label = the (broadcast) label of the user's first segment
        self.user_label = {
            t: np.array([self.y_by_task[t][segs[0]] for segs in self.user_segs])
            for t in task_names
        }

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        segs = self.user_segs[idx]
        sample = {
            "values": self.values[segs],
            "missing_mask": self.missing_mask[segs],
            "deltas": self.deltas[segs],
            "X_filledLOCF": self.X_filledLOCF[segs],
            "empirical_mean": self.empirical_mean,
            "n_segs": len(segs),
        }
        for t in self.task_names:
            sample[f"y_{t}"] = self.user_label[t][idx]
        return sample


def _collate(batch, task_names):
    import torch

    seg_to_user = torch.cat(
        [torch.full((b["n_segs"],), i, dtype=torch.long) for i, b in enumerate(batch)]
    )
    out = {
        "values": torch.cat([b["values"] for b in batch], dim=0),
        "missing_mask": torch.cat([b["missing_mask"] for b in batch], dim=0),
        "deltas": torch.cat([b["deltas"] for b in batch], dim=0),
        "X_filledLOCF": torch.cat([b["X_filledLOCF"] for b in batch], dim=0),
        "empirical_mean": batch[0]["empirical_mean"],
        "segment_to_user": seg_to_user,
        "n_users": len(batch),
    }
    for t in task_names:
        out[f"y_{t}"] = torch.as_tensor([b[f"y_{t}"] for b in batch])
    return out


# --------------------------------------------------------------------------- #
# Module: backbone + per-user mean-pool + per-task heads
# --------------------------------------------------------------------------- #
def _build_module(n_steps, n_features, cls_n, reg_tasks, ord_n):
    import torch.nn as nn
    from pypots.nn.modules.grud import BackboneGRUD

    class _Module(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = BackboneGRUD(n_steps, n_features, RNN_HIDDEN_SIZE)
            self.cls_heads = nn.ModuleDict(
                {t: nn.Linear(RNN_HIDDEN_SIZE, k) for t, k in cls_n.items()}
            )
            self.reg_heads = nn.ModuleDict({t: nn.Linear(RNN_HIDDEN_SIZE, 1) for t in reg_tasks})
            self.ord_heads = nn.ModuleDict(
                {t: nn.Linear(RNN_HIDDEN_SIZE, k - 1) for t, k in ord_n.items()}
            )

        def forward(self, inp):
            _, hidden = self.backbone(
                inp["values"], inp["missing_mask"], inp["deltas"],
                inp["empirical_mean"], inp["X_filledLOCF"],
            )
            rep = _scatter_mean(hidden, inp["segment_to_user"], inp["n_users"])
            out = {}
            for t, head in self.cls_heads.items():
                out[t] = head(rep)
            for t, head in self.reg_heads.items():
                out[t] = head(rep)
            for t, head in self.ord_heads.items():
                out[t] = head(rep)
            return out

    return _Module()


# --------------------------------------------------------------------------- #
# Trainer (multi-task masked loss + early stopping)
# --------------------------------------------------------------------------- #
class _Trainer:
    def __init__(self, cls_n, reg_tasks, ord_n, device, seed=42):
        self.cls_n, self.reg_tasks, self.ord_n = cls_n, reg_tasks, ord_n
        self.task_names = sorted(cls_n) + sorted(reg_tasks) + sorted(ord_n)
        self.device, self.seed = device, seed
        self.model = None
        self._cls_w: dict = {}
        self._ord_w: dict = {}

    def _class_weights(self, ds):
        import torch

        for t, k in self.cls_n.items():
            y = ds.user_label[t]
            y = y[y != MISSING_LABEL]
            if len(y) == 0:
                continue
            counts = np.bincount(y.astype(int), minlength=k).astype(np.float64)
            w = len(y) / (k * np.maximum(counts, 1))
            self._cls_w[t] = torch.tensor(w, dtype=torch.float32, device=self.device)
        for t, k in self.ord_n.items():
            y = ds.user_label[t]
            y = y[y != MISSING_LABEL]
            if len(y) == 0:
                continue
            ks = np.arange(k - 1)
            pos = (y[:, None] > ks[None, :]).sum(0).astype(np.float64)
            neg = len(y) - pos
            self._ord_w[t] = torch.tensor(
                neg / np.maximum(pos, 1), dtype=torch.float32, device=self.device
            )

    def _loss(self, logits, batch):
        import torch
        import torch.nn.functional as F

        total = torch.tensor(0.0, device=self.device)
        nvalid = 0
        for t in self.cls_n:
            y = batch[f"y_{t}"].to(self.device).long()
            m = y != MISSING_LABEL
            if m.sum() == 0:
                continue
            total = total + F.cross_entropy(logits[t][m], y[m], weight=self._cls_w.get(t))
            nvalid += 1
        for t in self.reg_tasks:
            y = batch[f"y_{t}"].to(self.device).float()
            m = ~torch.isnan(y)
            if m.sum() == 0:
                continue
            total = total + F.mse_loss(logits[t][m].squeeze(-1), y[m])
            nvalid += 1
        for t in self.ord_n:
            y = batch[f"y_{t}"].to(self.device).long()
            m = y != MISSING_LABEL
            if m.sum() == 0:
                continue
            lg = logits[t][m]
            ks = torch.arange(lg.shape[1], device=self.device).unsqueeze(0)
            cum = (y[m].unsqueeze(1) > ks).float()
            total = total + F.binary_cross_entropy_with_logits(lg, cum, pos_weight=self._ord_w.get(t))
            nvalid += 1
        return total / max(nvalid, 1)

    def fit(self, train_ds, val_ds, n_steps, n_features):
        import torch
        from torch.utils.data import DataLoader

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.model = _build_module(n_steps, n_features, self.cls_n, self.reg_tasks, self.ord_n).to(
            self.device
        )
        self._class_weights(train_ds)
        opt = torch.optim.Adam(self.model.parameters(), lr=LR)
        coll = lambda b: _collate(b, self.task_names)  # noqa: E731
        tl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=coll)
        vl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=coll) if val_ds else None

        best, best_state, bad = float("inf"), None, 0
        for epoch in range(EPOCHS):
            self.model.train()
            for batch in tl:
                batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                opt.zero_grad()
                loss = self._loss(self.model(batch), batch)
                if loss.item() == 0.0:
                    continue
                loss.backward()
                opt.step()
            if vl is not None:
                vloss = self._val_loss(vl)
                logger.info("GRU-D epoch %d/%d val_loss=%.4f", epoch + 1, EPOCHS, vloss)
                if vloss < best:
                    best, best_state, bad = vloss, copy.deepcopy(self.model.state_dict()), 0
                else:
                    bad += 1
                    if bad >= PATIENCE:
                        logger.info("GRU-D early stop at epoch %d", epoch + 1)
                        break
        if best_state is not None:
            self.model.load_state_dict(best_state)

    @property
    def _no_grad(self):
        import torch

        return torch.no_grad

    def _val_loss(self, loader):
        import torch

        self.model.eval()
        s, n = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                loss = self._loss(self.model(batch), batch)
                s += loss.item() * batch["n_users"]
                n += batch["n_users"]
        return s / max(n, 1)

    def predict(self, ds, task, task_type):
        import torch
        from torch.utils.data import DataLoader

        self.model.eval()
        coll = lambda b: _collate(b, self.task_names)  # noqa: E731
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=coll)
        preds = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                out = self.model(batch)[task]
                if task_type == "binary":
                    preds.append(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
                elif task_type == "ordinal":
                    # expected level = sum of P(y > k) over the K-1 cumulative logits
                    preds.append(torch.sigmoid(out).sum(dim=1).cpu().numpy())
                elif task_type == "multiclass":
                    preds.append(out.argmax(dim=1).cpu().numpy())
                else:  # regression
                    preds.append(out.squeeze(-1).cpu().numpy())
        return np.concatenate(preds)


# --------------------------------------------------------------------------- #
# Engine adapter: trains the one multi-task model on the first fit, serves
# each task's head on predict.
# --------------------------------------------------------------------------- #
class GRUD:
    """Multi-task GRU-D — unified ``Method`` (trained from raw, GPU)."""

    name = "gru_d"
    input_granularity = "daily"  # per-user cohort from the daily lookup
    needs_segments = False  # builds its own per-user segments from raw

    def __init__(self, data_dir=None, tasks=None, seed=42):
        self._data_dir = data_dir
        self._tasks = list(tasks) if tasks is not None else None
        self.seed = seed
        self._trainer = None
        self._segments = None  # {uid: (n_segs, 24, 19)} with NaN at missing
        self._task_types: dict[str, str] = {}
        self._reg_stats: dict[str, tuple[float, float]] = {}  # task -> (mean, std)
        self._ctx = None  # EvalContext (active task + cohort user_ids), injected per call
        self._loader = None  # shared DataLoader, injected by run_eval (whole-store access)

    def set_context(self, ctx) -> None:
        """Receive the per-(task, split) cohort context; the engine injects it before
        ``fit`` / ``predict``. The shared multi-task model trains once (the per-call
        args are ignored); ``predict`` reads the active ``task`` and cohort ``user_ids``
        from here, which the clean ``Method`` signatures do not carry."""
        self._ctx = ctx

    def set_loader(self, loader) -> None:
        """Receive the shared :class:`DataLoader`; GRU-D builds its per-user segment
        store from the loader's whole-history store rather than re-reading the dataset."""
        self._loader = loader

    def _load_segments(self):
        if self._segments is not None:
            return
        # Whole-history store from the shared loader (one daily_hourly_hf read across the
        # run); ``values`` are already (N, 24, 19) time-first with NaN at missing.
        values, _mask, users = self._loader.segment_store()
        by_user: dict[str, list] = {}
        for i, u in enumerate(users):
            by_user.setdefault(u, []).append(i)
        self._segments = {u: values[np.asarray(idx)] for u, idx in by_user.items()}

    def _provider(self):
        from downstream_evaluation.data.provider import LOOKUP_BY_GRANULARITY, TaskDataProvider
        from downstream_evaluation.data.splits import load_split_file
        from openmhc._evaluate import _DatasetPaths

        paths = _DatasetPaths.from_root(self._data_dir)
        lookup = str(paths.root / "processed" / LOOKUP_BY_GRANULARITY["daily"])
        return TaskDataProvider(lookup, load_split_file(paths.splits_file), granularity="daily")

    def _build_split(self, provider, split, cls_n, reg_tasks, ord_n):
        """Flatten the cohort's per-user segments + per-(task) labels for one split."""
        labels: dict[str, dict[str, float]] = {}
        cohort: set[str] = set()
        for t in self._tasks:
            td = provider.task_data(t, split)
            labels[t] = {str(u): lab for u, lab in zip(td.user_ids, td.labels)}
            cohort.update(labels[t])
        users = [u for u in sorted(cohort) if u in self._segments]
        Xs, uids = [], []
        y_by_task = {t: [] for t in self._tasks}
        for u in users:
            segs = self._segments[u]
            Xs.append(segs)
            uids.extend([u] * len(segs))
            for t in self._tasks:
                if t in reg_tasks:
                    val = labels[t].get(u, np.nan)
                    if not np.isnan(val) and t in self._reg_stats:
                        mu, sd = self._reg_stats[t]
                        val = (float(val) - mu) / sd  # train-split z-score
                    y_by_task[t].extend([float(val)] * len(segs))
                else:
                    val = labels[t].get(u, MISSING_LABEL)
                    y_by_task[t].extend([int(val)] * len(segs))
        X = np.concatenate(Xs, axis=0)
        y_by_task = {t: np.asarray(v) for t, v in y_by_task.items()}
        return _UserDataset(X, y_by_task, np.asarray(uids, dtype=object), self._tasks)

    def fit(self, data, labels, task_type) -> None:
        """Train the shared multi-task model once on the first call.

        The per-call ``data`` / ``labels`` / ``task_type`` are ignored: GRU-D assembles
        every task's cohort + labels from its own provider and trains all heads jointly.
        """
        if self._trainer is not None:
            return
        import torch

        from downstream_evaluation.evaluation.metrics import get_task_type

        if self._tasks is None:
            raise ValueError("GRUD requires the task list (multi-task training)")
        self._load_segments()
        provider = self._provider()

        # Build the task → head-type spec from a probe of the train labels.
        cls_n, reg_tasks, ord_n = {}, [], {}
        train_td_labels = {}
        for t in self._tasks:
            tt = get_task_type(t)
            self._task_types[t] = tt
            td = provider.task_data(t, "train")
            train_td_labels[t] = td.labels
            if tt == "regression":
                reg_tasks.append(t)
            else:
                k = int(np.nanmax(td.labels.astype(float))) + 1
                if tt == "ordinal":
                    ord_n[t] = max(k, 2)
                elif tt == "multiclass":
                    cls_n[t] = max(k, 2)
                else:  # binary
                    cls_n[t] = 2

        # Train-split z-score stats for regression targets. Raw-scale MSE (age² ≈
        # 2500 vs CE ≈ 0.7) would dominate the averaged multi-task loss and starve
        # every head, so regression targets are normalized for training and the
        # transform reversed at predict. Pearson r is linear-invariant, so the
        # reverse is cosmetic.
        for t in reg_tasks:
            y = train_td_labels[t].astype(float)
            y = y[~np.isnan(y)]
            mu = float(y.mean()) if len(y) else 0.0
            sd = float(y.std()) if len(y) and y.std() > 1e-8 else 1.0
            self._reg_stats[t] = (mu, sd)

        train_ds = self._build_split(provider, "train", cls_n, reg_tasks, ord_n)
        val_ds = self._build_split(provider, "validation", cls_n, reg_tasks, ord_n)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("GRU-D training: %d train users, device=%s", len(train_ds), device)
        self._trainer = _Trainer(cls_n, reg_tasks, ord_n, device, seed=self.seed)
        self._trainer.fit(train_ds, val_ds, n_steps=24, n_features=N_SENSOR_CHANNELS)

    def predict(self, data) -> np.ndarray:
        """Per-user predictions for the active task, aligned to the cohort ``user_ids``."""
        task = self._ctx.task
        users = [str(u) for u in self._ctx.user_ids if str(u) in self._segments]
        # one-user-per-item dataset, in user_ids order, dummy labels.
        Xs, uids = [], []
        for u in users:
            segs = self._segments[u]
            Xs.append(segs)
            uids.extend([u] * len(segs))
        dummy = {
            t: (np.full(sum(len(self._segments[u]) for u in users), MISSING_LABEL)
                if self._task_types[t] != "regression"
                else np.full(sum(len(self._segments[u]) for u in users), np.nan))
            for t in self._tasks
        }
        ds = _UserDataset(
            np.concatenate(Xs, axis=0), dummy, np.asarray(uids, dtype=object), self._tasks
        )
        preds = self._trainer.predict(ds, task, self._task_types[task])
        if self._task_types[task] == "regression" and task in self._reg_stats:
            mu, sd = self._reg_stats[task]
            preds = preds * sd + mu  # reverse the train-split z-score
        pred_by_user = dict(zip(ds.user_ids, preds))
        return np.array([pred_by_user.get(str(u), 0.0) for u in self._ctx.user_ids], dtype=np.float64)

