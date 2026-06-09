"""autoARIMA baseline forecasting model."""

import random
import warnings

import numpy as np
import pandas as pd
from sktime.forecasting.arima import AutoARIMA as sktimeAutoARIMA

try:
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
except ImportError:  # pragma: no cover - optional dependency in some envs
    ConvergenceWarning = Warning

from forecasting_evaluation.models.base import BasePredictionModel


class AutoARIMAModel(BasePredictionModel):
    """AutoARIMA forecasting model using sktime.
    
    This model automatically searches for the best ARIMA parameters
    using information criteria (AIC/BIC). It fits a separate model
    for each feature in multivariate time series.

    Note: This model refits from scratch on each predict call.
    """
    
    def __init__(
            self, 
            seed: int = 42,
            start_p: int = 2,
            start_q: int = 2,
            max_p: int = 5,
            max_q: int = 5,
            seasonal: bool = True,
            start_P: int = 1,
            start_Q: int = 1,
            max_P: int = 2,
            max_Q: int = 2,
            max_d: int = 2,
            max_D: int = 1,
            information_criterion: str = "aic",
            suppress_warnings: bool = True,
            trace: bool = False,
            error_action: str = "ignore",
            stepwise: bool = False,
            n_jobs: int = -1,
            max_history_length: int | None = 24 * 14,  # Limit to recent 336 hours (14 days)
            quantile_levels: tuple[float, ...] = (0.1, 0.5, 0.9),
        ):
        """Initialize AutoARIMA model.
        
        Args:
            seed: Random seed for reproducibility.
            start_p: Starting value of p in stepwise procedure.
            start_q: Starting value of q in stepwise procedure.
            max_p: Maximum value of p.
            max_q: Maximum value of q.
            seasonal: Whether to fit a seasonal ARIMA model.
            m: The period for seasonal differencing (24 for hourly data with daily seasonality).
            start_P: Starting value of P in stepwise procedure.
            start_Q: Starting value of Q in stepwise procedure.
            max_P: Maximum value of P.
            max_Q: Maximum value of Q.
            max_d: Maximum value of d (order of first-differencing).
            max_D: Maximum value of D (order of seasonal differencing).
            information_criterion: Information criterion for model selection ('aic', 'bic', 'aicc').
            suppress_warnings: Whether to suppress warnings during fitting.
            trace: Whether to print status on the fits.
            error_action: Action to take if a model fails to fit ('ignore', 'raise', 'warn').
            stepwise: Whether to use stepwise search. Must be False to enable true parallel search.
            n_jobs: Number of parallel jobs (-1 uses all processors).
            max_history_length: Maximum number of recent data points to use for fitting.
                If None, uses all available history. If specified, only the most recent
                max_history_length points are used for model fitting.
            quantile_levels: Quantile levels returned alongside point forecasts.
        """
        self.seed = seed
        self.start_p = start_p
        self.start_q = start_q
        self.max_p = max_p
        self.max_q = max_q
        self.seasonal = seasonal
        self.start_P = start_P
        self.start_Q = start_Q
        self.max_P = max_P
        self.max_Q = max_Q
        self.max_d = max_d
        self.max_D = max_D
        self.information_criterion = information_criterion
        self.suppress_warnings = suppress_warnings
        self.trace = trace
        self.error_action = error_action
        self.stepwise = stepwise
        self.n_jobs = n_jobs
        self.max_history_length = max_history_length
        self.quantile_levels = self._validate_quantile_levels(quantile_levels)
        
        # Set random seeds
        np.random.seed(seed)
        random.seed(seed)

        self.reset()

    def predict(
            self,
            history: np.ndarray,
            horizon: int,
        ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Predict future values using AutoARIMA.

        Fits a new model on all available historical data for each prediction.

        Args:
            history: Full-prefix history of shape (n_features, history_length),
                may contain NaN.
            horizon: Number of future hours to forecast.

        Returns:
            Tuple containing (point_result, quantiles_result):
            - point_result: (n_features, prediction_length) array of point predictions.
            - quantiles_result: (n_features, prediction_length, n_quantiles) quantile forecasts.
        """
        target = history
        prediction_length = horizon

        n_features, _ = target.shape
        n_quantiles = 0 if self.quantile_levels is None else len(self.quantile_levels)

        predictions = np.zeros((n_features, prediction_length))
        quantiles = None
        if n_quantiles > 0:
            quantiles = np.zeros((n_features, prediction_length, n_quantiles))

        for i in range(n_features):
            y = target[i, :]
            y = np.where(np.isnan(y), 0.0, y)

            # Truncate to most recent data if max_history_length is specified
            if self.max_history_length is not None and len(y) > self.max_history_length:
                y = y[-self.max_history_length:]

            # Skip if all values are zero or constant
            if np.all(y == 0) or np.std(y) < 1e-10:
                predictions[i, :] = y[-1]
                if quantiles is not None:
                    quantiles[i, :, :] = y[-1]
                continue

            try:
                model = self._fit_new_model(y)
                fh = np.arange(1, prediction_length + 1)
                y_pred = model.predict(fh=fh)
                predictions[i, :] = y_pred.ravel()
                if quantiles is not None:
                    y_quantiles = model.predict_quantiles(
                        fh=fh,
                        alpha=self.quantile_levels.tolist(),
                    )
                    quantiles[i, :, :] = self._coerce_quantiles_array(
                        y_quantiles,
                        prediction_length=prediction_length,
                    )
            except Exception as e:
                if not self.suppress_warnings:
                    print(f"ARIMA fit/predict failed for feature {i}: {e}")
                predictions[i, :] = y[-1]
                if quantiles is not None:
                    quantiles[i, :, :] = y[-1]

        point_result = predictions
        quantiles_result = quantiles

        return point_result, quantiles_result

    @staticmethod
    def _validate_quantile_levels(quantile_levels: tuple[float, ...] | list[float] | np.ndarray) -> np.ndarray:
        """Validate and normalize configured quantile levels."""
        quantile_array = np.asarray(quantile_levels, dtype=float)
        if quantile_array.ndim != 1 or quantile_array.size == 0:
            raise ValueError("quantile_levels must be a non-empty 1D sequence")
        if np.any((quantile_array <= 0.0) | (quantile_array >= 1.0)):
            raise ValueError("quantile_levels must be strictly between 0 and 1")
        if np.unique(quantile_array).size != quantile_array.size:
            raise ValueError("quantile_levels must not contain duplicates")
        return np.sort(quantile_array)

    def _coerce_quantiles_array(
        self,
        quantile_frame: pd.DataFrame,
        *,
        prediction_length: int,
    ) -> np.ndarray:
        """Convert sktime quantile output to (prediction_length, n_quantiles)."""
        quantile_values = quantile_frame.to_numpy(dtype=float)
        if quantile_values.shape == (prediction_length, len(self.quantile_levels)):
            return quantile_values

        if isinstance(quantile_frame.columns, pd.MultiIndex):
            quantile_columns = []
            for alpha in self.quantile_levels:
                alpha_slice = quantile_frame.xs(alpha, axis=1, level=-1)
                if alpha_slice.shape[1] != 1:
                    raise ValueError(
                        "Expected one quantile column per alpha for univariate AutoARIMA output"
                    )
                quantile_columns.append(alpha_slice.iloc[:, 0].to_numpy(dtype=float))
            return np.stack(quantile_columns, axis=-1)

        raise ValueError(
            "Unexpected quantile prediction shape "
            f"{quantile_values.shape}; expected ({prediction_length}, {len(self.quantile_levels)})"
        )

    def _fit_new_model(self, y: np.ndarray):
        """Fit a new AutoARIMA model for a single univariate series."""
        model = sktimeAutoARIMA(
            start_p=self.start_p,
            start_q=self.start_q,
            max_p=self.max_p,
            max_q=self.max_q,
            stepwise=self.stepwise,
            seasonal=self.seasonal,
            start_P=self.start_P,
            start_Q=self.start_Q,
            max_P=self.max_P,
            max_Q=self.max_Q,
            max_d=self.max_d,
            max_D=self.max_D,
            information_criterion=self.information_criterion,
            suppress_warnings=self.suppress_warnings,
            trace=self.trace,
            error_action=self.error_action,
            random_state=self.seed,
            n_jobs=self.n_jobs
        )

        # Suppress pmdarima constant series warning
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*completely constant.*",
                category=UserWarning
            )
            warnings.filterwarnings(
                "ignore",
                message=".*stepwise model cannot be fit in parallel.*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                category=ConvergenceWarning,
            )
            model.fit(y)

        return model

    def reset(self):
        """Reset model state (no-op for stateless model)."""
        pass
