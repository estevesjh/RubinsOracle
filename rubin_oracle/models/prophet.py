"""Prophet-based forecaster implementation for Rubin's Oracle.

This module implements the ProphetForecaster class using Facebook Prophet
for time series forecasting.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from prophet import Prophet
from prophet.serialize import model_from_json, model_to_json

from rubin_oracle.base import ValidationMixin
from rubin_oracle.config import ProphetConfig
from rubin_oracle.preprocessing import preprocess_for_forecast
from rubin_oracle.utils import validate_input


class ProphetForecaster(ValidationMixin):
    """Time series forecaster using Facebook Prophet.

    Implements the Forecaster protocol using Prophet for univariate
    time series forecasting with trend and seasonality components.

    Attributes:
        name: Human-readable name of the forecaster
        config: ProphetConfig with model hyperparameters
        model_: Fitted Prophet model (available after fit())

    Example:
        >>> config = ProphetConfig(
        ...     lag_days=48,
        ...     n_forecast=24,
        ...     daily_seasonality=True
        ... )
        >>> forecaster = ProphetForecaster(config)
        >>> forecaster.fit(train_df)
        >>> predictions = forecaster.predict(periods=24)
        >>> standardized = forecaster.standardize_output(predictions)
    """

    def __init__(self, config: ProphetConfig):
        """Initialize the ProphetForecaster.

        Args:
            config: Configuration object with Prophet hyperparameters
        """
        self.config = config
        self.name = f"prophet_{config.name}"
        self.model_: Prophet | None = None
        self._fit_df: pd.DataFrame | None = None

    def fit(self, df: pd.DataFrame) -> ProphetForecaster:
        """Fit the Prophet model to training data.

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

        # Apply preprocessing (decomposition and date filtering)
        df = preprocess_for_forecast(
            df,
            decompose=self.config.use_decomposition,
            freq=self.config.freq_per_day,
            savgol_mode=self.config.savgol_mode,
            train_start_date=self.config.train_start_date,
            train_end_date=self.config.train_end_date,
            lag_days=self.config.lag_days,
        )

        # Store training data for predict() when df is not provided
        self._fit_df = df.copy()

        # Initialize Prophet with config parameters
        self.model_ = Prophet(
            changepoints=None,
            n_changepoints=self.config.n_changepoints,
            changepoint_range=self.config.changepoints_range,
            yearly_seasonality=self.config.yearly_seasonality,
            weekly_seasonality=self.config.weekly_seasonality,
            daily_seasonality=self.config.daily_seasonality,
            seasonality_mode=self.config.seasonality_mode,
            changepoint_prior_scale=self.config.changepoint_prior_scale,
            seasonality_prior_scale=self.config.seasonality_prior_scale,
            interval_width=self.config.interval_width,
        )

        # Add decomposed components as regressors if decomposition was applied
        if self.config.use_decomposition:
            decomposed_components = [
                'y_high', 'y_p0', 'y_p1', 'y_p2', 'y_p3', 'y_p4', 'y_p5', 'y_56day_trend'
            ]
            for component in decomposed_components:
                if component in df.columns:
                    self.model_.add_regressor(component)

        # Fit the model
        self.model_.fit(df)

        return self

    def predict(
        self,
        df: pd.DataFrame | None = None,
        periods: int | None = None,
    ) -> pd.DataFrame:
        """Generate forecasts using the fitted Prophet model.

        Args:
            df: Historical data (not used by Prophet, kept for API consistency).
                If None, uses the training data from fit().
            periods: Number of periods to forecast ahead.
                If None, uses config.n_forecast.

        Returns:
            DataFrame with Prophet forecast columns:
                - ds: Forecast timestamps
                - yhat: Point forecast
                - yhat_lower: Lower uncertainty bound
                - yhat_upper: Upper uncertainty bound

        Raises:
            RuntimeError: If model hasn't been fitted yet
        """
        if self.model_ is None:
            raise RuntimeError(
                "Model has not been fitted yet. Call fit() before predict()."
            )

        # Use config default if periods not specified
        if periods is None:
            periods = self.config.n_forecast

        # Create future dataframe with configured frequency
        future = self.model_.make_future_dataframe(periods=periods, freq=self.config.freq)

        # Generate predictions
        forecast = self.model_.predict(future)

        # Return only the forecasted future period (not the historical fitted values)
        forecast = forecast.tail(periods)

        # Select relevant columns
        forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].reset_index(drop=True)

        return forecast

    def standardize_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert Prophet output to standardized format.

        Adds a 'step' column indicating the forecast horizon (1, 2, 3, ...).

        Args:
            df: Prophet forecast output with columns: ds, yhat, yhat_lower, yhat_upper

        Returns:
            DataFrame with added 'step' column

        Example:
            >>> predictions = forecaster.predict(periods=3)
            >>> standardized = forecaster.standardize_output(predictions)
            >>> standardized['step'].tolist()
            [1, 2, 3]
        """
        df = df.copy()

        # Add step column (1-indexed forecast horizon)
        df['step'] = range(1, len(df) + 1)

        # Ensure column order
        df = df[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'step']]

        return df

    def _fit_and_predict(
        self,
        df_history: pd.DataFrame,
        forecast_time: pd.Timestamp,
    ) -> pd.DataFrame:
        """Fit on history and generate forecast for validation.

        Prophet requires refitting for each validation step since it doesn't
        have autoregressive components that depend on recent observations.

        Args:
            df_history: Historical data up to forecast_time
            forecast_time: The time at which the forecast is issued

        Returns:
            DataFrame with columns: ds, yhat1, yhat2, ..., yhat_lower1, yhat_upper1, ...
        """
        # Refit the model on historical data
        self.fit(df_history)

        # Generate forecast
        forecast = self.predict(periods=self.config.n_forecast)

        # Convert Prophet's single-step format to multi-step format
        # Prophet returns: ds, yhat, yhat_lower, yhat_upper
        # We need: ds, yhat1, yhat2, ..., yhat_lower1, yhat_upper1, ...
        result_rows = []
        for i, row in forecast.iterrows():
            step = i + 1
            result_rows.append({
                'ds': row['ds'],
                f'yhat{step}': row['yhat'],
                f'yhat_lower{step}': row['yhat_lower'],
                f'yhat_upper{step}': row['yhat_upper'],
            })

        # For Prophet, each row is a separate forecast step
        # We return one row per target timestamp
        df_result = forecast[['ds']].copy()
        for step in range(1, self.config.n_forecast + 1):
            if step <= len(forecast):
                df_result[f'yhat{step}'] = forecast.iloc[step - 1]['yhat']
                df_result[f'yhat_lower{step}'] = forecast.iloc[step - 1]['yhat_lower']
                df_result[f'yhat_upper{step}'] = forecast.iloc[step - 1]['yhat_upper']

        return df_result

    def save(self, path: str | Path) -> None:
        """Save the fitted Prophet model to disk.

        Saves both the model and configuration as JSON files.

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

        # Save Prophet model
        model_path = path / 'model.json'
        with open(model_path, 'w') as f:
            json.dump(model_to_json(self.model_), f)

        # Save config
        config_path = path / 'config.yaml'
        self.config.to_yaml(config_path)

    @classmethod
    def load(cls, path: str | Path) -> ProphetForecaster:
        """Load a previously saved Prophet model.

        Args:
            path: Directory path where model was saved

        Returns:
            Loaded ProphetForecaster instance

        Raises:
            FileNotFoundError: If model files don't exist
        """
        path = Path(path)

        # Load config
        config_path = path / 'config.yaml'
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        config = ProphetConfig.from_yaml(config_path)

        # Create instance
        forecaster = cls(config)

        # Load Prophet model
        model_path = path / 'model.json'
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path, 'r') as f:
            model_json = json.load(f)
            forecaster.model_ = model_from_json(model_json)

        return forecaster
