"""Sklearn classifier registry and factory.

Creates sklearn Pipelines with StandardScaler + classifier.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

if TYPE_CHECKING:
    from downstream_evaluation.config import ClassifierConfig


class XGBOrdinalWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper for XGBoost to perform ordinal regression using the cumulative link model approach.

    This wrapper trains K-1 binary classifiers for K ordinal levels, where each classifier predicts the probability of the target
    being above a certain threshold. The predict_proba method then converts these probabilities into class probabilities for each ordinal level.
    """

    def __init__(self, params):
        """Initialize the XGBOrdinalWrapper with the given parameters for the XGBoost classifiers.

        Args:
            params: A dictionary of hyperparameters to be passed to each XGBoost classifier. These should be the same for all K-1 classifiers, as they are trained on the same data with different binary targets.
        """
        self.params = params
        self.clfs = []
        self.levels = None

    def fit(self, X, y):
        """Fit K-1 binary XGBoost classifiers for K ordinal levels.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target vector of shape (n_samples,) containing ordinal labels.

        Returns:
            self: Fitted XGBOrdinalWrapper instance with trained classifiers.
        """
        # Identify unique ordinal levels (e.g., [0, 1, 2])
        self.levels = np.sort(np.unique(y))
        self.clfs = []

        # We need K-1 classifiers for K levels
        for i in range(len(self.levels) - 1):
            # Binary target: Is the current label > current level?
            binary_y = (y > self.levels[i]).astype(int)

            clf = XGBClassifier(**self.params)
            clf.fit(X, binary_y)
            self.clfs.append(clf)
        return self

    def predict_proba(self, X):
        """Predict class probabilities for each ordinal level.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            probs: Array of shape (n_samples, K) containing class probabilities for each ordinal level, where K is the number of unique ordinal levels.
        """
        # Get the probability for each binary threshold
        # [P(y > 0), P(y > 1), ...]
        thresh_probs = np.column_stack([c.predict_proba(X)[:, 1] for c in self.clfs])

        # Convert threshold probabilities into class probabilities
        # P(y=0) = 1 - P(y>0)
        # P(y=1) = P(y>0) - P(y>1)
        probs = np.zeros((X.shape[0], len(self.levels)))
        probs[:, 0] = 1 - thresh_probs[:, 0]
        for i in range(1, len(self.levels) - 1):
            probs[:, i] = thresh_probs[:, i - 1] - thresh_probs[:, i]
        probs[:, -1] = thresh_probs[:, -1]

        return probs

    def predict(self, X):
        """Predict ordinal class labels by argmax over the class probabilities
        reconstructed from threshold outputs via differencing, as in
        Frank & Hall (2001).
        """
        return self.predict_proba(X).argmax(axis=1).astype(int)


class LogRegOrdinalWrapper(BaseEstimator, ClassifierMixin):
    """LogisticRegression K-1 binary decomposition for ordinal targets.

    Mirrors ``XGBOrdinalWrapper`` but uses LogisticRegression as the base
    estimator, so linear-probe methods use the same ordinal meta-strategy as the
    tree-based methods — isolating the linear-vs-tree comparison from the
    ordinal-strategy comparison. Trains K-1 binary classifiers for K ordinal
    levels (Frank & Hall cumulative-link decomposition).
    """

    def __init__(self, params, random_state=None):
        """Initialize the wrapper.

        Args:
            params: Hyperparameters passed to each LogisticRegression classifier.
            random_state: Random seed shared by all K-1 classifiers.
        """
        self.params = params
        self.random_state = random_state
        self.clfs = []
        self.levels = None

    def fit(self, X, y):
        """Fit K-1 binary LogisticRegression classifiers for K ordinal levels.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Ordinal target vector of shape (n_samples,).

        Returns:
            self: The fitted wrapper.
        """
        self.levels = np.sort(np.unique(y))
        self.clfs = []
        for i in range(len(self.levels) - 1):
            binary_y = (y > self.levels[i]).astype(int)
            clf = LogisticRegression(**self.params, random_state=self.random_state)
            clf.fit(X, binary_y)
            self.clfs.append(clf)
        return self

    def predict_proba(self, X):
        """Convert the K-1 threshold probabilities into K class probabilities.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples, K) of per-level class probabilities.
        """
        thresh_probs = np.column_stack([c.predict_proba(X)[:, 1] for c in self.clfs])
        probs = np.zeros((X.shape[0], len(self.levels)))
        probs[:, 0] = 1 - thresh_probs[:, 0]
        for i in range(1, len(self.levels) - 1):
            probs[:, i] = thresh_probs[:, i - 1] - thresh_probs[:, i]
        probs[:, -1] = thresh_probs[:, -1]
        return probs

    def predict(self, X):
        """Predict ordinal labels as the argmax of the class probabilities.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Array of shape (n_samples,) of predicted ordinal labels.
        """
        return self.predict_proba(X).argmax(axis=1).astype(int)


class RobustStandardScaler(StandardScaler):
    """StandardScaler that clips extreme values after transformation.

    This prevents numerical issues (overflow/underflow) in downstream classifiers
    when features have outliers or very different scales.
    """

    def __init__(self, clip_value: float = 10.0, **kwargs):
        """Initialize RobustStandardScaler with clipping value.

        Args:
            clip_value: Maximum absolute value to clip transformed features to.
            **kwargs: Additional arguments passed to StandardScaler.
        """
        super().__init__(**kwargs)
        self.clip_value = clip_value

    def transform(self, X, copy=None):
        """Transform features and clip extreme values to prevent numerical issues.

        Args:
            X: Input features to transform.
            copy: Whether to copy the input data.

        Returns:
            Transformed and clipped features.
        """
        X_scaled = super().transform(X, copy=copy)
        # Clip extreme values to prevent numerical issues
        return np.clip(X_scaled, -self.clip_value, self.clip_value)


def create_model(
    config: ClassifierConfig,
    random_state: int | None = None,
    task_type: str | None = None,
) -> Pipeline | BaseEstimator:
    """Create sklearn model, optionally wrapped in a pipeline with scaler.

    Args:
        config: Model configuration specifying type and hyperparameters.
        random_state: Random seed for deterministic behavior. If None, classifiers
            may use non-deterministic random number generation.
        task_type: Task type ("binary", "multiclass", "ordinal", or "regression"). Used to
            select appropriate solver for LogisticRegression on multiclass tasks.

    Returns:
        Sklearn Pipeline with RobustStandardScaler and model if use_scaler=True,
        otherwise returns the model directly.
    """
    clf_type = config.type

    if clf_type == "logistic_regression":
        params = config.logistic_regression
        # Use lbfgs for multiclass (native softmax) instead of liblinear (OvR)
        solver = "lbfgs" if task_type == "multiclass" else params.solver
        # lbfgs doesn't support n_jobs, only use it for liblinear
        n_jobs = params.n_jobs if solver == "liblinear" else 1
        clf = LogisticRegression(
            max_iter=params.max_iter,
            class_weight=params.class_weight,
            C=params.C,
            solver=solver,
            n_jobs=n_jobs,
            random_state=random_state,
        )
    elif clf_type == "linear_regression":
        params = config.linear_regression
        clf = LinearRegression(
            fit_intercept=params.fit_intercept,
            copy_X=params.copy_X,
            n_jobs=params.n_jobs,
            positive=params.positive,
        )
    elif clf_type == "logreg_ordinal":
        # K-1 binary decomposition using LogisticRegression — mirrors xgboost_ordinal
        # so linear-probe methods use the same ordinal meta-strategy as tree methods.
        # Inherits hyperparameters from the configured linear probe.
        params = config.logistic_regression
        solver = params.solver
        n_jobs = params.n_jobs if solver == "liblinear" else 1
        lr_kwargs = dict(
            max_iter=params.max_iter,
            class_weight=params.class_weight,
            C=params.C,
            solver=solver,
            n_jobs=n_jobs,
        )
        clf = LogRegOrdinalWrapper(params=lr_kwargs, random_state=random_state)

    else:
        raise ValueError(f"Unknown model type: {clf_type}")

    # Build pipeline: [scaler] → [L2 norm] → [PCA] → classifier.
    steps: list = []

    if config.use_scaler:
        scaler_type = getattr(config, "scaler_type", "robust")
        if scaler_type == "standard":
            steps.append(StandardScaler())
        else:
            steps.append(RobustStandardScaler(clip_value=10.0))

    if getattr(config, "use_l2_norm", False):
        from sklearn.preprocessing import Normalizer

        steps.append(Normalizer(norm="l2"))

    pca_n = getattr(config, "pca_n_components", None)
    if pca_n is not None:
        from sklearn.decomposition import PCA

        pca_whiten = getattr(config, "pca_whiten", False)
        # random_state pins the randomized SVD solver that sklearn auto-selects for
        # wide feature matrices (e.g. MultiRocket's ~50k dims). Without it the probe
        # is non-reproducible: the rotated subspace varies run-to-run, swinging
        # rare-positive AUPRC by ~0.03. Seeded here so results are deterministic.
        steps.append(PCA(n_components=pca_n, whiten=pca_whiten, random_state=random_state))

    if steps:
        steps.append(clf)
        return make_pipeline(*steps)
    else:
        return clf
