"""Configuration classes for Rubin's Oracle forecasting models.

This module defines Pydantic-based configuration classes for Prophet and NeuralProphet
forecasters, providing type-safe configuration with YAML serialization support.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class BaseForecasterConfig(BaseModel):
    """Base configuration for all forecaster models.

    Attributes:
        name: Model name identifier
        lag_days: Number of historical days to use (NeuralProphet: n_lags, Prophet: training window)
        n_forecast: Number of steps to forecast ahead
        freq: Data frequency for training and forecasting (e.g., 'h', '15min', 'D')
        yearly_seasonality: Enable yearly seasonality (bool or Fourier order)
        weekly_seasonality: Enable weekly seasonality (bool or Fourier order)
        daily_seasonality: Enable daily seasonality (bool or Fourier order)
        seasonality_mode: Type of seasonality decomposition
        growth: Trend growth type
        n_changepoints: Number of potential changepoints for trend
        changepoints_range: Proportion of history for changepoint detection
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    lag_days: int = Field(default=48, ge=1)
    n_forecast: int = Field(default=48, ge=1)

    # Data frequency
    freq: str = Field(default='h', description="Pandas frequency string (e.g., 'h', '15min', 'D')")

    # Seasonality
    yearly_seasonality: bool | int = False
    weekly_seasonality: bool | int = False
    daily_seasonality: bool | int = True
    seasonality_mode: Literal["additive", "multiplicative"] = "additive"

    # Trend
    growth: Literal["linear", "logistic", "flat"] = "linear"
    n_changepoints: int = Field(default=12, ge=0)
    changepoints_range: float = Field(default=0.85, gt=0.0, le=1.0)

    @classmethod
    def from_yaml(cls, path: str | Path) -> BaseForecasterConfig:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            Configuration instance

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            ValueError: If YAML content is invalid
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path where YAML file will be saved
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)


class ProphetConfig(BaseForecasterConfig):
    """Configuration for Prophet forecaster.

    Extends BaseForecasterConfig with Prophet-specific hyperparameters.

    Attributes:
        name: Fixed to "prophet"
        changepoint_prior_scale: Flexibility of trend changepoints
        seasonality_prior_scale: Strength of seasonality
        holidays_prior_scale: Strength of holiday effects
        interval_width: Width of uncertainty intervals (0.68 ≈ 1 sigma)
    """

    name: Literal["prophet"] = "prophet"
    changepoint_prior_scale: float = Field(default=0.05, gt=0.0)
    seasonality_prior_scale: float = Field(default=10.0, gt=0.0)
    holidays_prior_scale: float = Field(default=0.0, gt=0.0)
    interval_width: float = Field(default=0.68, gt=0.0, lt=1.0)


class NeuralProphetConfig(BaseForecasterConfig):
    """Configuration for NeuralProphet forecaster.

    Extends BaseForecasterConfig with NeuralProphet-specific hyperparameters.

    Attributes:
        name: Fixed to "neural_prophet"
        ar_layers: Hidden layer sizes for autoregressive component (empty = linear AR)
        ar_reg: Regularization strength for AR weights
        trend_reg: Regularization strength for trend
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Optimizer learning rate
        loss_func: Loss function name
        optimizer: Optimizer name
        quantiles: Quantile levels for uncertainty estimation
        drop_missing: Whether to drop rows with missing values
        impute_missing: Whether to impute missing values
    """

    name: Literal["neural_prophet"] = "neural_prophet"

    # AR configuration
    ar_layers: list[int] = Field(default_factory=list)
    ar_reg: float = Field(default=1.0, ge=0.0)

    # Trend
    trend_reg: float = Field(default=1.0, ge=0.0)

    # Training
    epochs: int = Field(default=15, ge=1)
    batch_size: int = Field(default=128, ge=1)
    learning_rate: float = Field(default=0.003, gt=0.0)
    loss_func: str = "SmoothL1Loss"
    optimizer: str = "AdamW"

    # Uncertainty
    quantiles: list[float] = Field(default_factory=lambda: [0.16, 0.84])

    # Data handling
    drop_missing: bool = True
    impute_missing: bool = True
