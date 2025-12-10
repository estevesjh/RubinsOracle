"""NeuralProphet-based forecaster implementation for Rubin's Oracle.

This module implements the NeuralProphetForecaster class using NeuralProphet
for neural network-based time series forecasting.
"""

from __future__ import annotations

from pathlib import Path
import warnings
import torch

# Suppress NeuralProphet and PyTorch warnings
# Monkeypatch torch.load to avoid weights_only warning
_original_load = torch.load
def safe_load(*a, **k):
    if "weights_only" not in k:
        k["weights_only"] = False
    return _original_load(*a, **k)
torch.load = safe_load

# Filter specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module=r"neuralprophet")
warnings.filterwarnings("ignore", category=UserWarning, module=r"pytorch_lightning")
warnings.filterwarnings("ignore", message="weights_only=False")
warnings.filterwarnings("ignore", message="concatenation with empty or all-NA")
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")

import pandas as pd

from rubin_oracle.config import NeuralProphetConfig
from rubin_oracle.preprocessing import preprocess_for_forecast
from rubin_oracle.utils import prepare_regular_frequency, validate_input

try:
    from neuralprophet import NeuralProphet, save, load
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

        # Prepare data with regular frequency from config
        df = prepare_regular_frequency(
            df,
            freq=self.config.freq,
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
            growth='linear',  # Hardcoded default
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

        # Add decomposed components as regressors if decomposition was applied
        if self.config.use_decomposition:
            decomposed_components = [
                'y_high', 'y_p0', 'y_p1', 'y_p2', 'y_p3', 'y_p4', 'y_p5', 'y_56day_trend'
            ]
            for component in decomposed_components:
                if component in df.columns:
                    self.model_.add_future_regressor(component)

        # Add AR layers if specified
        if self.config.ar_layers:
            # NeuralProphet uses ar_layers parameter during initialization
            # This creates a non-linear AR component
            pass

        # Fit the model with configured frequency
        self.model_.fit(df, freq=self.config.freq)

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

        # Validate and prepare input with configured frequency
        df = validate_input(df)

        # Apply preprocessing if decomposition was used during training
        if self.config.use_decomposition:
            df = preprocess_for_forecast(
                df,
                decompose=True,
                freq=self.config.freq_per_day,
                savgol_mode=self.config.savgol_mode,
                train_start_date=None,  # Don't filter for prediction
                train_end_date=None,
                lag_days=self.config.lag_days,
            )

        df = prepare_regular_frequency(
            df,
            freq=self.config.freq,
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

    def standardize_output(
        self,
        df: pd.DataFrame,
        issue_time: pd.Timestamp | None = None
    ) -> pd.DataFrame:
        """Convert NeuralProphet forecast output to standardized long format.

        Transforms from NeuralProphet's forecast output to standard format
        with one row per forecast step. Based on parse_raw_forecast logic.

        Args:
            df: NeuralProphet forecast output (raw=False)
            issue_time: Time when forecast was issued. If None, uses first 'ds' value.

        Returns:
            DataFrame with standardized columns:
                - ds (datetime): Target timestamp (what the forecast is FOR)
                - yhat (float): Point forecast
                - yhat_lower (float): Lower uncertainty bound (16th percentile)
                - yhat_upper (float): Upper uncertainty bound (84th percentile)
                - step (int): Forecast horizon (1, 2, 3, ...)
                - lead_time (float): Lead time in hours

        Example:
            >>> predictions = forecaster.predict(recent_df)
            >>> standardized = forecaster.standardize_output(predictions)
            >>> # Now in long format with explicit steps
        """
        # Determine issue time
        if issue_time is None:
            if 'ds' not in df.columns:
                raise ValueError("DataFrame must have 'ds' column")
            issue_time = pd.to_datetime(df['ds'].iloc[0])

        # Ensure issue_time has timezone
        if issue_time.tzinfo is None:
            issue_time = issue_time.tz_localize("UTC")

        # Ensure ds column is datetime with timezone
        if 'ds' in df.columns:
            df = df.copy()
            df['ds'] = pd.to_datetime(df['ds'])
            if df['ds'].dt.tz is None:
                df['ds'] = df['ds'].dt.tz_localize("UTC")
            else:
                df['ds'] = df['ds'].dt.tz_convert("UTC")

        # Determine frequency from config
        freq_td = pd.to_timedelta(self.config.freq)

        # Find yhat columns (yhat1, yhat2, ...)
        yhat_cols = [c for c in df.columns if c.startswith("yhat") and c[4:].isdigit()]

        if not yhat_cols:
            # Fallback for single forecast or other formats
            if "yhat" in df.columns:
                yhat_cols = ["yhat1"]
                df = df.rename(columns={"yhat": "yhat1"})
            else:
                return pd.DataFrame()

        indices = sorted([int(c.replace("yhat", "")) for c in yhat_cols])

        long_rows = []

        for i in indices:
            step_col = f"yhat{i}"
            # For raw=False, yhat{i} at time t is the prediction for t made i steps ago.
            # We want the prediction made AT issue_time for time t = issue_time + i*freq.
            # So we look at the row where ds = issue_time + i*freq, and take column yhat{i}.

            target_time = issue_time + i * freq_td
            row = df[df['ds'] == target_time]

            if row.empty:
                continue

            yhat = row[step_col].values[0]

            # Handle quantiles
            # NeuralProphet raw=False output includes columns like 'yhat1 16.0%', 'yhat1 84.0%'
            lower_col = f"{step_col} 16.0%"
            upper_col = f"{step_col} 84.0%"

            yhat_lower = row[lower_col].values[0] if lower_col in row.columns else yhat
            yhat_upper = row[upper_col].values[0] if upper_col in row.columns else yhat

            lead_time = i * freq_td.total_seconds() / 3600.0

            long_rows.append({
                "ds": target_time,
                "yhat": yhat,
                "yhat_lower": yhat_lower,
                "yhat_upper": yhat_upper,
                "step": i,
                "lead_time": lead_time,
            })

        return pd.DataFrame(long_rows)

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

        # Save NeuralProphet model using their function
        model_path = str(path / 'model.np')
        save(self.model_, model_path)

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

        # Load NeuralProphet model using their function
        model_path = str(path / 'model.np')
        forecaster.model_ = load(model_path)

        return forecaster
