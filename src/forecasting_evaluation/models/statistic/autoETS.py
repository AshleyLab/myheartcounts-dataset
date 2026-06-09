"""AutoETS baseline forecasting model."""

import random
import warnings

import numpy as np
import pandas as pd
from sktime.forecasting.ets import AutoETS as sktimeAutoETS

try:
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
except ImportError:  # pragma: no cover - optional dependency in some envs
    ConvergenceWarning = Warning

from forecasting_evaluation.models.base import BasePredictionModel

SEASONAL_CYCLE_ERROR_MSG = (
    "Cannot compute initial seasonals using heuristic method with less than two full seasonal cycles in the data."
)
MLE_CONVERGENCE_MSG = "Maximum Likelihood optimization failed to converge"


class AutoETSModel(BasePredictionModel):
    """AutoETS forecasting model using sktime.
    
    This model uses Exponential Smoothing (ETS) framework to automatically
    select the best combination of Error, Trend, and Seasonality components
    via information criteria (AIC/BIC/AICc). For multivariate time series,
    it fits a separate model for each feature.
    
    Note: Since sktime's AutoETS doesn't have true incremental update,
    this model refits on all historical data each time.
    """
    
    def __init__(
            self, 
            seed: int = 42,
            auto: bool = True,
            sp: int = 24,
            information_criterion: str = "aic",
            n_jobs: int = -1,
            max_history_length: int | None = None,
            quantile_levels: tuple[float, ...] = (0.1, 0.5, 0.9),
        ):
        """Initialize AutoETS model.
        
        Args:
            seed: Random seed for reproducibility.
            auto: Set True to enable automatic model selection.
            sp: The number of periods in a complete seasonal cycle for seasonal
                (Holt-Winters) models. For example, 4 for quarterly data with an
                annual cycle or 24 for hourly data with daily seasonality.
            information_criterion: Information criterion for model selection
                ('aic', 'aicc', 'bic').
            n_jobs: Number of parallel jobs (-1 uses all processors).
            max_history_length: Maximum number of recent data points to use for fitting.
                If None, uses all available history. If specified, only the most recent
                max_history_length points are used for model fitting.
            quantile_levels: Quantile levels returned alongside point forecasts.
        """
        self.seed = seed
        self.auto = auto
        self.sp = sp
        self.information_criterion = information_criterion
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
        """Predict future values using AutoETS.

        Fits a new model on all available historical data for each prediction.

        Args:
            history: Full-prefix history of shape (n_features, history_length),
                may contain NaN.
            horizon: Number of future hours to forecast.
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
            y = self._forward_fill_nan(y)
            
            # Truncate to most recent data if max_history_length is specified
            if self.max_history_length is not None and len(y) > self.max_history_length:
                y = y[-self.max_history_length:]
            
            # Preprocessing: skip constant or all-zero series
            if np.all(y == 0) or np.std(y) < 1e-10:
                predictions[i, :] = y[-1]
                if quantiles is not None:
                    quantiles[i, :, :] = y[-1]
                continue
            
            # Fit model on (truncated) historical data
            try:
                # ETS seasonal initialization needs at least two full cycles.
                # Fall back to non-seasonal model when history is too short.
                effective_sp = self.sp if self.sp <= 1 or len(y) >= 2 * self.sp else 1

                model = sktimeAutoETS(
                    auto=self.auto,
                    sp=effective_sp,
                    information_criterion=self.information_criterion,
                    n_jobs=self.n_jobs,
                    random_state=self.seed
                )
                
                # Suppress warnings about non-positive time series
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=".*time series is not strictly positive.*",
                        category=UserWarning
                    )
                    warnings.filterwarnings(
                        "ignore",
                        category=ConvergenceWarning,
                    )
                    warnings.filterwarnings(
                        "ignore",
                        message=f".*{MLE_CONVERGENCE_MSG}.*",
                        category=Warning,
                    )
                    model.fit(y)
                
                # Perform prediction
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
                # Quietly ignore known short-history seasonal initialization failures.
                if SEASONAL_CYCLE_ERROR_MSG not in str(e):
                    print(f"ETS fit/predict failed for feature {i}: {e}")
                predictions[i, :] = y[-1]
                if quantiles is not None:
                    quantiles[i, :, :] = y[-1]
        
        return predictions, quantiles

    @staticmethod
    def _forward_fill_nan(y: np.ndarray) -> np.ndarray:
        """Fill missing values from the previous timestamp."""
        y_filled = np.asarray(y, dtype=float).copy()
        nan_mask = np.isnan(y_filled)
        if not np.any(nan_mask):
            return y_filled

        valid_indices = np.flatnonzero(~nan_mask)
        if valid_indices.size == 0:
            return np.zeros_like(y_filled, dtype=float)

        first_valid = valid_indices[0]
        if first_valid > 0:
            y_filled[:first_valid] = y_filled[first_valid]

        valid_mask = ~np.isnan(y_filled)
        previous_valid_indices = np.maximum.accumulate(
            np.where(valid_mask, np.arange(y_filled.size), 0)
        )
        return y_filled[previous_valid_indices]

    @staticmethod
    def _validate_quantile_levels(
        quantile_levels: tuple[float, ...] | list[float] | np.ndarray,
    ) -> np.ndarray:
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
                        "Expected one quantile column per alpha for univariate AutoETS output"
                    )
                quantile_columns.append(alpha_slice.iloc[:, 0].to_numpy(dtype=float))
            return np.stack(quantile_columns, axis=-1)

        raise ValueError(
            "Unexpected quantile prediction shape "
            f"{quantile_values.shape}; expected ({prediction_length}, {len(self.quantile_levels)})"
        )
    
    def reset(self):
        """Reset model state (no-op for stateless model)."""
        pass
