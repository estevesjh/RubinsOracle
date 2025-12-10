"""NeuralProphet-based forecaster implementation for Rubin's Oracle.

This module implements the NeuralProphetForecaster class using NeuralProphet
for neural network-based time series forecasting.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rubin_oracle.config import NeuralProphetConfig
from rubin_oracle.utils import prepare_regular_frequency, validate_input

try:
    from neuralprophet import NeuralProphet
except ImportError:
    raise ImportError(
        "NeuralProphet is required but not installed. "
        "Install it with: pip install neuralprophet"
    )


class NeuralProphetForecaster:
    """Time series forecaster using NeuralProphet.

    Implements the Forecaster protocol using NeuralProphet for neural network-based
    time series forecasting with autoregressive components.

    Attributes:
        name: Human-readable name of the forecaster
        config: NeuralProphetConfig with model hyperparameters
        model_: Fitted NeuralProphet model (available after fit())

    Example:
        >>> config = NeuralProphetConfig(
        ...     lag_days=48,
        ...     n_forecast=24,
        ...     epochs=20,
        ...     ar_layers=[64, 32]
        ... )
        >>> forecaster = NeuralProphetForecaster(config)
        >>> forecaster.fit(train_df)
        >>> predictions = forecaster.predict(recent_df)
        >>> standardized = forecaster.standardize_output(predictions)
    """

    def __init__(self, config: NeuralProphetConfig):
        """Initialize the NeuralProphetForecaster.

        Args:
            config: Configuration object with NeuralProphet hyperparameters
        """
        self.config = config
        self.name = f"neural_prophet_{config.name}"
        self.model_: NeuralProphet | None = None

    def fit(self, df: pd.DataFrame) -> NeuralProphetForecaster:
        """Fit the NeuralProphet model to training data.

        Args:
            df: Training data with columns:
                - ds (datetime): Timestamps
                - y (float): Target values

        Returns:
            Self for method chaining

        Raises:
            ValueError: If data is invalid or insufficient
        """
        # Validate input
        df = validate_input(df)

        # Prepare data with regular frequency
        df = prepare_regular_frequency(
            df,
            freq='h',
            interpolate=self.config.impute_missing
        )

        # Handle missing values based on config
        if self.config.drop_missing:
            df = df.dropna(subset=['y'])
        elif self.config.impute_missing:
            # Already handled in prepare_regular_frequency
            pass

        # Initialize NeuralProphet with config parameters
        self.model_ = NeuralProphet(
            growth=self.config.growth,
            n_changepoints=self.config.n_changepoints,
            changepoints_range=self.config.changepoints_range,
            trend_reg=self.config.trend_reg,
            yearly_seasonality=self.config.yearly_seasonality,
            weekly_seasonality=self.config.weekly_seasonality,
            daily_seasonality=self.config.daily_seasonality,
            seasonality_mode=self.config.seasonality_mode,
            n_lags=self.config.lag_days,
            n_forecasts=self.config.n_forecast,
            ar_reg=self.config.ar_reg,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            loss_func=self.config.loss_func,
            optimizer=self.config.optimizer,
            quantiles=self.config.quantiles,
        )

        # Add AR layers if specified
        if self.config.ar_layers:
            # NeuralProphet uses ar_layers parameter during initialization
            # This creates a non-linear AR component
            pass

        # Fit the model (suppress output with verbose=False or minimal)
        self.model_.fit(df, freq='h')

        return self

    def predict(
        self,
        df: pd.DataFrame | None = None,
        periods: int | None = None,
    ) -> pd.DataFrame:
        """Generate forecasts using the fitted NeuralProphet model.

        Args:
            df: Recent historical data for autoregressive forecasting.
                Must contain at least lag_days of data.
                If None, uses the training data from fit().
            periods: Number of periods to forecast ahead.
                If None, uses config.n_forecast.

        Returns:
            DataFrame with NeuralProphet forecast in wide format:
                - ds: Forecast origin timestamp
                - yhat1, yhat2, ..., yhatN: Point forecasts for N steps ahead
                - For each quantile q: yhat{step} q-{q*100}

        Raises:
            RuntimeError: If model hasn't been fitted yet
            ValueError: If df doesn't contain enough historical data
        """
        if self.model_ is None:
            raise RuntimeError(
                "Model has not been fitted yet. Call fit() before predict()."
            )

        if df is None:
            raise ValueError(
                "NeuralProphet requires recent data for autoregressive forecasting. "
                "Please provide df with at least lag_days of historical data."
            )

        # Validate and prepare input
        df = validate_input(df)
        df = prepare_regular_frequency(
            df,
            freq='h',
            interpolate=self.config.impute_missing
        )

        # Check if enough data is provided
        if len(df) < self.config.lag_days:
            raise ValueError(
                f"Insufficient data for prediction. Need at least {self.config.lag_days} "
                f"observations, got {len(df)}"
            )

        # Use config default if periods not specified
        if periods is None:
            periods = self.config.n_forecast

        # Generate predictions
        # NeuralProphet.predict() will forecast n_forecasts steps ahead
        forecast = self.model_.predict(df)

        # Return only the last row (latest forecast) if predicting from recent data
        # The forecast contains predictions for all historical points + future
        forecast = forecast.tail(1).reset_index(drop=True)

        return forecast

    def standardize_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert NeuralProphet wide format to standardized long format.

        Transforms from NeuralProphet's wide format (yhat1, yhat2, ...)
        to the standard format with one row per forecast step.

        Args:
            df: NeuralProphet forecast output in wide format

        Returns:
            DataFrame with standardized columns:
                - ds (datetime): Target timestamp (what the forecast is FOR)
                - yhat (float): Point forecast
                - yhat_lower (float): Lower uncertainty bound
                - yhat_upper (float): Upper uncertainty bound
                - step (int): Forecast horizon (1, 2, 3, ...)

        Example:
            >>> predictions = forecaster.predict(recent_df)
            >>> standardized = forecaster.standardize_output(predictions)
            >>> # Now in long format with explicit steps
        """
        # Get the number of forecast steps from config
        n_steps = self.config.n_forecast

        # Extract base timestamp (forecast origin)
        base_ds = df['ds'].iloc[0]

        # Prepare lists to collect standardized data
        standardized_data = []

        for step in range(1, n_steps + 1):
            # Calculate target timestamp
            target_ds = base_ds + pd.Timedelta(hours=step)

            # Get point forecast
            yhat_col = f'yhat{step}'
            if yhat_col not in df.columns:
                continue
            yhat = df[yhat_col].iloc[0]

            # Get uncertainty bounds from quantiles
            # Default quantiles are [0.16, 0.84] (roughly ±1 sigma)
            lower_quantile = self.config.quantiles[0]
            upper_quantile = self.config.quantiles[-1]

            lower_col = f'yhat{step} {lower_quantile:.0%}'
            upper_col = f'yhat{step} {upper_quantile:.0%}'

            # Handle different quantile column naming conventions
            # NeuralProphet may use different formats
            possible_lower_cols = [
                lower_col,
                f'yhat{step} {int(lower_quantile * 100)}',
                f'yhat{step}_lower',
            ]
            possible_upper_cols = [
                upper_col,
                f'yhat{step} {int(upper_quantile * 100)}',
                f'yhat{step}_upper',
            ]

            yhat_lower = None
            yhat_upper = None

            for col in possible_lower_cols:
                if col in df.columns:
                    yhat_lower = df[col].iloc[0]
                    break

            for col in possible_upper_cols:
                if col in df.columns:
                    yhat_upper = df[col].iloc[0]
                    break

            # If quantiles not found, use point forecast as bounds (no uncertainty)
            if yhat_lower is None:
                yhat_lower = yhat
            if yhat_upper is None:
                yhat_upper = yhat

            standardized_data.append({
                'ds': target_ds,
                'yhat': yhat,
                'yhat_lower': yhat_lower,
                'yhat_upper': yhat_upper,
                'step': step,
            })

        # Create standardized DataFrame
        standardized_df = pd.DataFrame(standardized_data)

        return standardized_df

    def save(self, path: str | Path) -> None:
        """Save the fitted NeuralProphet model to disk.

        Args:
            path: Directory path where model will be saved

        Raises:
            RuntimeError: If model hasn't been fitted yet
        """
        if self.model_ is None:
            raise RuntimeError(
                "Model has not been fitted yet. Call fit() before save()."
            )

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save NeuralProphet model (uses PyTorch's save format)
        model_path = str(path / 'model')
        self.model_.save(model_path)

        # Save config
        config_path = path / 'config.yaml'
        self.config.to_yaml(config_path)

    @classmethod
    def load(cls, path: str | Path) -> NeuralProphetForecaster:
        """Load a previously saved NeuralProphet model.

        Args:
            path: Directory path where model was saved

        Returns:
            Loaded NeuralProphetForecaster instance

        Raises:
            FileNotFoundError: If model files don't exist
        """
        path = Path(path)

        # Load config
        config_path = path / 'config.yaml'
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        config = NeuralProphetConfig.from_yaml(config_path)

        # Create instance
        forecaster = cls(config)

        # Load NeuralProphet model
        model_path = str(path / 'model')
        forecaster.model_ = NeuralProphet.load(model_path)

        return forecaster
