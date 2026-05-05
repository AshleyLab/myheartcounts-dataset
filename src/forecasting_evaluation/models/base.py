"""Base protocol for forecasting models."""

from __future__ import annotations

import time
import tracemalloc
from abc import ABC, abstractmethod

import numpy as np

from forecasting_evaluation.data.types import SubTrajectoryInput


class BasePredictionModel(ABC):
    """Protocol for Base Prediction Models."""
    model_name: str  # Optional name attribute for model instance (e.g. "ARIMA")
    quantile_levels: np.ndarray | None = None
    
    @abstractmethod
    def predict(
        self,
        inputs: SubTrajectoryInput,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Predict future values for given time series data.

        Args:
            inputs: Typed forecasting sub-trajectory input containing target,
                covariates, horizon, and metadata.

        Returns:
            Tuple of:
                - point forecast of shape (n_features, prediction_length), or None.
                - quantile forecast of shape (n_features, prediction_length, n_quantiles), or None.
        """
        pass

    def reset(self) -> None:
        """Reset model state if applicable (e.g. for stateful models)."""
        pass

    def predict_wrapper(
        self,
        inputs: SubTrajectoryInput
    ) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
        """Wrapper around predict() that handles performance tracking.
        
        This method:
        1. Starts performance tracking (time and memory)
        2. Calls the model's predict() method
        3. Collects performance metrics
        4. Returns predictions along with performance metadata
        
        Args:
            inputs: Typed forecasting sub-trajectory input.
            
        Returns:
            Tuple of:
                - point forecast of shape (n_features, prediction_length), or None.
                - quantile forecast of shape (n_features, prediction_length, n_quantiles), or None.
                - base_result: dict with total performance metrics for this prediction call:
                    - 'prediction_time_seconds': total prediction wall time in seconds
                    - 'memory_usage_mb': total peak traced memory in MB
        """
        # Start performance tracking
        tracemalloc.start()
        start_time = time.time()
        
        # Call model's predict method
        point_result, quantiles_result = self.predict(inputs)
        
        # Collect performance metrics
        prediction_time = time.time() - start_time
        _current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        base_result = {
            "prediction_time_seconds": float(prediction_time),
            "memory_usage_mb": float(peak_mem / 1024 / 1024),
        }

        return point_result, quantiles_result, base_result