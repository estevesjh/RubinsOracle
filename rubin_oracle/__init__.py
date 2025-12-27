"""Rubin's Oracle - Unified Weather Forecasting Package.

A unified interface for time series forecasting using Prophet and NeuralProphet
models for weather forecasting at Cerro Pachon (Rubin Observatory).

Example:
    >>> from rubin_oracle import ProphetForecaster, NeuralProphetForecaster
    >>> from rubin_oracle.config import ProphetConfig, NeuralProphetConfig
    >>>
    >>> # Using Prophet
    >>> config = ProphetConfig.from_yaml("configs/prophet_default.yaml")
    >>> model = ProphetForecaster(config)
    >>> model.fit(train_df)
    >>> predictions = model.predict(periods=24)
    >>> standardized = model.standardize_output(predictions)
    >>>
    >>> # Using NeuralProphet
    >>> config = NeuralProphetConfig(lag_days=48, n_forecast=24, epochs=20)
    >>> model = NeuralProphetForecaster(config)
    >>> model.fit(train_df)
    >>> predictions = model.predict(recent_df)
    >>> standardized = model.standardize_output(predictions)
"""

from rubin_oracle.base import (
    BiWeeklyRetraining,
    DailyRetraining,
    Forecaster,
    MonthlyRetraining,
    NoRetraining,
    RetrainingStrategy,
    ValidationMixin,
    WeeklyRetraining,
)
from rubin_oracle.config import (
    BaseForecasterConfig,
    DecomposerConfig,
    NeuralProphetConfig,
    ProphetConfig,
)
from rubin_oracle.models import EnsembleForecaster, ProphetForecaster
from rubin_oracle.preprocessing import (
    BandpassDecomposer,
    HighLowFreqDecomposer,
    RubinVMDDecomposer,
    SignalDecomposer,
    fill_nan_periodic,
    preprocess_for_forecast,
)

__version__ = "0.1.0"

# Try to import NeuralProphet (optional dependency)
try:
    from rubin_oracle.models import NeuralProphetForecaster

    _has_neural = True
except ImportError:
    _has_neural = False

# Define public API
__all__ = [
    # Base classes and protocols
    "Forecaster",
    "ValidationMixin",
    "RetrainingStrategy",
    "NoRetraining",
    "MonthlyRetraining",
    "WeeklyRetraining",
    "BiWeeklyRetraining",
    "DailyRetraining",
    # Configs
    "BaseForecasterConfig",
    "DecomposerConfig",
    "ProphetConfig",
    "NeuralProphetConfig",
    # Preprocessing
    "SignalDecomposer",
    "HighLowFreqDecomposer",
    "BandpassDecomposer",
    "RubinVMDDecomposer",
    "fill_nan_periodic",
    "preprocess_for_forecast",
    # Forecasters
    "ProphetForecaster",
    "EnsembleForecaster",
]

if _has_neural:
    __all__.append("NeuralProphetForecaster")
