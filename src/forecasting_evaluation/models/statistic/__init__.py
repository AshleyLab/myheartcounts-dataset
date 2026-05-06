"""Classic time series forecasting models."""

from forecasting_evaluation.models.statistic.autoARIMA import AutoARIMAModel
from forecasting_evaluation.models.statistic.autoETS import AutoETSModel

__all__ = ["AutoARIMAModel", "AutoETSModel"]
