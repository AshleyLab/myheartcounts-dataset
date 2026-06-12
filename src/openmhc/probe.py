"""The benchmark's standard linear probe.

Turns a model's per-participant embeddings into predictions using the *same* fixed
head every submission shares — PCA-50 followed by logistic regression (binary /
multiclass), a K-1 ordinal logistic decomposition (ordinal), or ordinary least
squares (regression). Because the head is fixed, a model's score reflects its
representation rather than its choice of classifier.

An encoder-style submission calls this inside ``predict`` so it returns predictions
(not embeddings) while staying comparable to every other encoder::

    class MyEncoder:
        def fit(self, data, labels, task_type):
            emb = np.stack([self._encode(x) for x in data])
            self._probe = openmhc.LinearProbe(task_type).fit(emb, labels)

        def predict(self, data):
            return self._probe.predict(np.stack([self._encode(x) for x in data]))
"""

from __future__ import annotations

import numpy as np

# Task type → the fixed linear head used for it (shared by every probe; config.py
# is dataclasses-only, so this import keeps module load cheap).
from downstream_evaluation.config import PROBE_BY_TASK_TYPE


class LinearProbe:
    """PCA-50 + a fixed linear head, selected by task type."""

    def __init__(self, task_type: str, n_components: int | None = 50, seed: int = 42) -> None:
        """Build the probe for ``task_type``.

        Args:
            task_type: one of ``"binary"``, ``"multiclass"``, ``"ordinal"``,
                ``"regression"``.
            n_components: PCA dimensionality applied before the head (``None``
                disables PCA). Defaults to 50.
            seed: random_state pinning PCA's randomized SVD solver and the
                classifier, so the probe is reproducible.
        """
        if task_type not in PROBE_BY_TASK_TYPE:
            raise ValueError(
                f"task_type must be one of {sorted(PROBE_BY_TASK_TYPE)}, got {task_type!r}"
            )
        # Imported lazily so importing the class stays cheap (sklearn/xgboost load
        # only when a probe is actually built).
        from downstream_evaluation.config import ClassifierConfig
        from downstream_evaluation.models.registry import create_model

        self.task_type = task_type
        config = ClassifierConfig(
            type=PROBE_BY_TASK_TYPE[task_type],
            use_scaler=False,  # encoders are probed on PCA features only; no scaler
            pca_n_components=n_components,
        )
        self._clf = create_model(config, random_state=seed, task_type=task_type)

    @staticmethod
    def _as_features(emb: np.ndarray) -> np.ndarray:
        """Float32 feature matrix with non-finite values zero-filled (NaN/±inf → 0)."""
        x = np.asarray(emb, dtype=np.float32)
        np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return x

    def fit(self, embeddings: np.ndarray, labels: np.ndarray) -> LinearProbe:
        """Fit the probe on training embeddings ``(n, D)`` and their labels ``(n,)``."""
        self._clf.fit(self._as_features(embeddings), labels)
        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict for ``(n, D)`` embeddings.

        Returns the class-1 probability for binary tasks and the point prediction
        otherwise (ordinal labels are integer-valued; regression is continuous).
        """
        x = self._as_features(embeddings)
        if self.task_type == "binary":
            return self._clf.predict_proba(x)[:, 1]
        return self._clf.predict(x)
