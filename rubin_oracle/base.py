"""Base protocol and interfaces for Rubin's Oracle forecasters.

This module defines the Forecaster protocol that all forecasting models must implement,
ensuring a consistent interface across different model types.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

from rubin_oracle.config import BaseForecasterConfig


@runtime_checkable
class Forecaster(Protocol):
    """Protocol defining the interface for all forecasting models.

    All forecaster implementations must provide these attributes and methods
    to ensure a consistent API across different model types.

    Attributes:
        name: Human-readable name of the forecaster
        config: Configuration object containing model hyperparameters
        model_: The underlying fitted model (available after fit() is called)
    """

    name: str
    config: BaseForecasterConfig
    model_: object  # The actual fitted model instance

    def fit(self, df: pd.DataFrame) -> Forecaster:
        """Fit the forecaster to training data.

        Args:
            df: Training data with columns:
                - ds (datetime): Timestamps
                - y (float): Target values

        Returns:
            Self for method chaining

        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        ...

    def predict(
        self,
        df: pd.DataFrame | None = None,
        periods: int | None = None,
    ) -> pd.DataFrame:
        """Generate forecasts.

        Args:
            df: Recent data for autoregressive models (required for NeuralProphet).
                Must contain at least lag_days of historical data.
                If None, uses data from fit() for Prophet.
            periods: Number of periods to forecast. If None, uses config.n_forecast.

        Returns:
            DataFrame with model-specific forecast columns.
            For Prophet: ds, yhat, yhat_lower, yhat_upper
            For NeuralProphet: ds, yhat1, yhat2, ..., yhat_lower1, yhat_upper1, ...

        Raises:
            RuntimeError: If model hasn't been fitted yet
            ValueError: If df is required but not provided, or has insufficient data
        """
        ...

    def standardize_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert model-specific output to standardized format.

        Transforms forecast output to a consistent schema across all models.

        Args:
            df: Raw forecast output from predict()

        Returns:
            DataFrame with standardized columns:
                - ds (datetime): Target timestamp (what the forecast is FOR)
                - yhat (float): Point forecast
                - yhat_lower (float): Lower uncertainty bound
                - yhat_upper (float): Upper uncertainty bound
                - step (int): Forecast horizon (1, 2, 3, ...)

        Example:
            >>> predictions = model.predict(recent_df)
            >>> standardized = model.standardize_output(predictions)
            >>> standardized.head()
                       ds      yhat  yhat_lower  yhat_upper  step
            0  2024-01-01  15.2        14.8        15.6        1
            1  2024-01-02  15.5        15.0        16.0        2
        """
        ...

    def save(self, path: str | Path) -> None:
        """Save the fitted model to disk.

        Args:
            path: Directory path where model will be saved

        Raises:
            RuntimeError: If model hasn't been fitted yet
        """
        ...

    @classmethod
    def load(cls, path: str | Path) -> Forecaster:
        """Load a previously saved model from disk.

        Args:
            path: Directory path where model was saved

        Returns:
            Loaded forecaster instance with fitted model

        Raises:
            FileNotFoundError: If model files don't exist
        """
        ...
