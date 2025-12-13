"""Configuration classes for Rubin's Oracle forecasting models.

This module defines Pydantic-based configuration classes for Prophet and NeuralProphet
forecasters, providing type-safe configuration with YAML serialization support.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class DecomposerConfig(BaseModel):
    """Configuration for signal decomposition.

    Supports two decomposition methods:
    - 'none': No decomposition (default)
    - 'bandpass': BandpassDecomposer with period_pairs
    - 'vmd': RubinVMDDecomposer (requires vmdpy)

    Attributes:
        method: Decomposition method ('none', 'bandpass', 'vmd')
        freq: Observations per day (e.g., 96 for 15-min, 24 for hourly)
        verbose: Whether to print decomposition info

        BandpassDecomposer parameters:
        period_pairs: List of (low, high) period pairs in days
        filter_type: Filter type ('savgol' or 'butterworth')
        edge_method: Edge handling method
        edge_pad_periods: Padding for edge mitigation
        nan_fill: NaN filling method
        nan_fill_period: Period for periodic NaN filling
        use_edge_weighting: Apply edge weighting
        butter_order: Butterworth filter order

        RubinVMDDecomposer parameters:
        alpha: VMD bandwidth constraint
        K_stage1: VMD modes for stage 1
        K_stage2: VMD modes for stage 2
    """

    model_config = ConfigDict(extra="forbid")

    method: Literal['none', 'bandpass', 'vmd'] = Field(
        default='none',
        description="Decomposition method"
    )

    # Common parameters
    freq: int = Field(default=96, ge=1, description="Observations per day")
    verbose: bool = False
    include_residual: bool = Field(
        default=False,
        description="Include residual (original - reconstructed) as a feature"
    )

    # BandpassDecomposer parameters
    period_pairs: list[tuple[float, float]] | None = Field(
        default=None,
        description="List of (low, high) period pairs in days for bandpass"
    )
    filter_type: Literal['savgol', 'butterworth'] = Field(
        default='butterworth',
        description="Filter type for bandpass decomposition"
    )
    edge_method: Literal['none', 'reflect', 'symmetric', 'constant', 'extrapolate'] = Field(
        default='reflect',
        description="Edge handling method for bandpass"
    )
    edge_pad_periods: float = Field(
        default=2.0,
        ge=0.0,
        description="Padding periods for edge mitigation"
    )
    nan_fill: Literal['periodic', 'linear', 'ffill'] = Field(
        default='periodic',
        description="NaN filling method"
    )
    nan_fill_period: float = Field(
        default=1.0,
        gt=0.0,
        description="Period in days for periodic NaN filling"
    )
    use_edge_weighting: bool = Field(
        default=False,
        description="Apply edge weighting after filtering"
    )
    butter_order: int = Field(
        default=4,
        ge=1,
        description="Butterworth filter order"
    )

    # RubinVMDDecomposer parameters
    alpha: float = Field(
        default=2000,
        gt=0,
        description="VMD bandwidth constraint"
    )
    K_stage1: int = Field(
        default=5,
        ge=1,
        description="Number of VMD modes for stage 1"
    )
    K_stage2: int = Field(
        default=3,
        ge=1,
        description="Number of VMD modes for stage 2"
    )


class BaseForecasterConfig(BaseModel):
    """Base configuration for all forecaster models.

    Attributes:
        name: Model name identifier
        lag_days: Number of historical days to use (NeuralProphet: n_lags, Prophet: training window)
        n_forecast: Number of steps to forecast ahead
        freq: Data frequency for training and forecasting (e.g., 'h', '15min', 'D')
        freq_per_day: Observations per day (4=15min, 24=hourly) - deprecated, use decomposer.freq
        yearly_seasonality: Enable yearly seasonality (bool or Fourier order)
        weekly_seasonality: Enable weekly seasonality (bool or Fourier order)
        daily_seasonality: Enable daily seasonality (bool or Fourier order)
        seasonality_mode: Type of seasonality decomposition
        n_changepoints: Number of potential changepoints for trend
        changepoints_range: Proportion of history for changepoint detection
        decomposer: Signal decomposition configuration
        train_start_date: Start date for training data (None = use all available)
        train_end_date: End date for training data (None = use all available)
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    lag_days: int = Field(default=48, ge=1)
    n_forecast: int = Field(default=48, ge=1)

    # Data frequency
    freq: str = Field(default='h', description="Pandas frequency string (e.g., 'h', '15min', 'D')")
    freq_per_day: int = Field(default=4, ge=1, description="Observations per day (deprecated, use decomposer.freq)")

    # Seasonality
    yearly_seasonality: bool | int = False
    weekly_seasonality: bool | int = False
    daily_seasonality: bool | int = True
    seasonality_mode: Literal["additive", "multiplicative"] = "additive"

    # Trend
    n_changepoints: int = Field(default=12, ge=0)
    changepoints_range: float = Field(default=0.85, gt=0.0, le=1.0)

    # Preprocessing - new nested config
    decomposer: DecomposerConfig = Field(
        default_factory=DecomposerConfig,
        description="Signal decomposition configuration"
    )
    use_time_features: bool = Field(
        default=False,
        description="Add cyclic time features (hour_sin/cos, doy_sin/cos)"
    )
    train_start_date: str | None = Field(default=None, description="Start date for training (YYYY-MM-DD)")
    train_end_date: str | None = Field(default=None, description="End date for training (YYYY-MM-DD)")
    model_dir: str | None = Field(default=None, description="Directory for saving/loading model checkpoints")

    @property
    def use_decomposition(self) -> bool:
        """Backward compatibility property."""
        return self.decomposer.method != 'none'

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
            # Use mode='json' to convert tuples to lists for YAML compatibility
            yaml.dump(self.model_dump(mode='json'), f, default_flow_style=False, sort_keys=False)


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
    epochs: int = Field(default=100, ge=1)
    batch_size: int = Field(default=128, ge=1)
    learning_rate: float = Field(default=0.003, gt=0.0)
    loss_func: str = "SmoothL1Loss"
    optimizer: str = "AdamW"

    # Early stopping
    early_stopping: bool = Field(default=True, description="Enable early stopping")
    valid_pct: float = Field(default=0.1, ge=0.0, le=0.5, description="Validation set percentage")
    patience: int = Field(default=10, ge=1, description="Reserved for future use (NeuralProphet doesn't expose this)")

    # Uncertainty
    quantiles: list[float] = Field(default_factory=lambda: [0.16, 0.84])

    # Data handling
    drop_missing: bool = True
    impute_missing: bool = True
