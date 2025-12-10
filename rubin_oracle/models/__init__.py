"""Forecasting model implementations for Rubin's Oracle."""

from rubin_oracle.models.prophet import ProphetForecaster

# NeuralProphet is optional
try:
    from rubin_oracle.models.neural_prophet import NeuralProphetForecaster
    __all__ = ['ProphetForecaster', 'NeuralProphetForecaster']
except ImportError:
    __all__ = ['ProphetForecaster']
