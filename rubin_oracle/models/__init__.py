"""Forecasting model implementations for Rubin's Oracle."""

from rubin_oracle.models.prophet import ProphetForecaster
from rubin_oracle.models.ensemble import EnsembleForecaster

# NeuralProphet is optional
try:
    from rubin_oracle.models.neural_prophet import NeuralProphetForecaster
    __all__ = ['ProphetForecaster', 'NeuralProphetForecaster', 'EnsembleForecaster']
except ImportError:
    __all__ = ['ProphetForecaster', 'EnsembleForecaster']
