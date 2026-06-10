"""Forecasting model/function registry and factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forecasting_evaluation.config import (
        FeaturesConfig,
        ForecastingConfig,
        ForecastingModelConfig,
    )
    from forecasting_evaluation.models.base import BasePredictionModel


def create_forecasting_model(
    config: ForecastingModelConfig,
    seed: int | None = None,
    force_single_core_stats: bool = False,
    forecasting_config: ForecastingConfig | None = None,
    features_config: FeaturesConfig | None = None,
) -> BasePredictionModel:
    """Create a prediction model from config.

    Args:
        config: Forecasting model configuration specifying type and hyperparameters.
        seed: Random seed for deterministic behavior. If None, forecasting may
            use non-deterministic random number generation.
        force_single_core_stats: If True, force statistical models to n_jobs=1
            to avoid nested parallelism when evaluator already runs in parallel.
        forecasting_config: Forecasting window configuration required by
            learned forecasting models that depend on fixed history/horizon.
        features_config: Feature extraction configuration required by learned
            forecasting models that infer feature dimensionality from channels.

    Returns:
        Forecasting model with predict method.
    """
    return _create_model(
        config,
        seed,
        force_single_core_stats,
        forecasting_config,
        features_config,
    )


def _create_model(
    config: ForecastingModelConfig,
    seed: int | None,
    force_single_core_stats: bool,
    forecasting_config: ForecastingConfig | None,
    features_config: FeaturesConfig | None,
) -> BasePredictionModel:
    """Create a single model from single-model config.
    
    Args:
        config: Single-model configuration.
        seed: Random seed.
        force_single_core_stats: If True, override statistical models to run
            with a single core (`n_jobs=1`) to avoid nested parallelism.
        forecasting_config: Forecasting window configuration passed through to
            models that need training-time history and horizon settings.
        features_config: Feature extraction configuration passed through to
            models that need channel-dependent feature counts.
    
    Returns:
        Instantiated model with custom name attribute.
    """
    model_name = config.name if config.name else config.type
    
    # Type-safe model creation based on instance type
    if config.type == "naive":
        raise NotImplementedError("Naive model not yet implemented")

    elif config.type == "seasonal_naive":
        seasonal_cfg = config.seasonal_naive
        from forecasting_evaluation.models.naive.seasonal_naive import SeasonalNaiveModel
        model = SeasonalNaiveModel(
            seed=seed,
            seasonal=seasonal_cfg.season_length,
            quantile_levels=seasonal_cfg.quantile_levels,
        )
        model.model_name = model_name

    elif config.type == "autoARIMA":
        arima_cfg = config.autoARIMA
        from forecasting_evaluation.models.statistic.autoARIMA import AutoARIMAModel
        n_jobs = 1 if force_single_core_stats else arima_cfg.n_jobs
        model = AutoARIMAModel(
            seed=seed,
            start_p=arima_cfg.start_p,
            start_q=arima_cfg.start_q,
            max_p=arima_cfg.max_p,
            max_q=arima_cfg.max_q,
            seasonal=arima_cfg.seasonal,
            start_P=arima_cfg.start_P,
            start_Q=arima_cfg.start_Q,
            max_P=arima_cfg.max_P,
            max_Q=arima_cfg.max_Q,
            max_d=arima_cfg.max_d,
            max_D=arima_cfg.max_D,
            information_criterion=arima_cfg.information_criterion,
            suppress_warnings=arima_cfg.suppress_warnings,
            trace=arima_cfg.trace,
            error_action=arima_cfg.error_action,
            stepwise=arima_cfg.stepwise,
            n_jobs=n_jobs,
            max_history_length=arima_cfg.max_history_length,
        )
        model.model_name = model_name

    elif config.type == "autoETS":
        ets_cfg = config.autoETS
        from forecasting_evaluation.models.statistic.autoETS import AutoETSModel
        n_jobs = 1 if force_single_core_stats else ets_cfg.n_jobs
        model = AutoETSModel(
            seed=seed,
            auto=ets_cfg.auto,
            sp=ets_cfg.sp,
            information_criterion=ets_cfg.information_criterion,
            n_jobs=n_jobs,
            max_history_length=ets_cfg.max_history_length,
        )
        model.model_name = model_name

    elif config.type == "chronos2":
        from forecasting_evaluation.models.foundational_model.chronos2 import Chronos2Model
        model = Chronos2Model(
            config=config.chronos2,
            seed=seed,
        )
        model.model_name = model_name

    elif config.type == "toto":
        from forecasting_evaluation.models.foundational_model.toto import TotoModel
        model = TotoModel(
            config=config.toto,
            seed=seed,
        )
        model.model_name = model_name

    elif config.type == "mixlinear":
        if forecasting_config is None or features_config is None:
            raise ValueError(
                "forecasting_config and features_config are required for "
                "mixlinear model creation"
            )
        from forecasting_evaluation.models.deep_learning_model.mixlinear import MixLinearModel
        model = MixLinearModel(
            config=config.mixlinear,
            forecasting_config=forecasting_config,
            features_config=features_config,
        )
        model.model_name = model_name

    elif config.type == "dlinear":
        if forecasting_config is None or features_config is None:
            raise ValueError(
                "forecasting_config and features_config are required for "
                "dlinear model creation"
            )
        from forecasting_evaluation.models.deep_learning_model.dlinear import DLinearModel
        model = DLinearModel(
            config=config.dlinear,
            forecasting_config=forecasting_config,
            features_config=features_config,
        )
        model.model_name = model_name

    elif config.type == "segrnn":
        if forecasting_config is None or features_config is None:
            raise ValueError(
                "forecasting_config and features_config are required for "
                "segrnn model creation"
            )
        from forecasting_evaluation.models.deep_learning_model.segrnn import SegRNNModel
        model = SegRNNModel(
            config=config.segrnn,
            forecasting_config=forecasting_config,
            features_config=features_config,
        )
        model.model_name = model_name

    else:
        raise ValueError(f"Unknown forecasting model type: {config.type}")

    return model
