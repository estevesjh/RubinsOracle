"""NeuralProphet-based forecaster implementation for Rubin's Oracle.

This module implements the NeuralProphetForecaster class using NeuralProphet
for neural network-based time series forecasting.
"""

from __future__ import annotations

from pathlib import Path
import warnings
import os
import sys
import torch

# Force tqdm to show in terminal
os.environ.setdefault("TQDM_DISABLE", "0")

# Set PyTorch Lightning to show progress
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.INFO)

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

import json

import numpy as np
import pandas as pd

from rubin_oracle.base import ValidationMixin
from rubin_oracle.config import NeuralProphetConfig
from rubin_oracle.preprocessing import BandpassDecomposer, RubinVMDDecomposer, preprocess_for_forecast
from rubin_oracle.utils import prepare_regular_frequency, validate_input

try:
    from neuralprophet import NeuralProphet, save, load
except ImportError:
    raise ImportError(
        "NeuralProphet is required but not installed. "
        "Install it with: pip install neuralprophet"
    )


class NeuralProphetForecaster(ValidationMixin):
    """Time series forecaster using NeuralProphet with optional signal decomposition.

    Implements the Forecaster protocol using NeuralProphet for neural network-based
    time series forecasting. Supports autoregressive modeling with lagged features
    and optional signal decomposition preprocessing.

    Attributes:
        name: Human-readable name of the forecaster
        config: NeuralProphetConfig with model hyperparameters
        model_: Fitted NeuralProphet model (available after fit())

    Example:
        >>> config = NeuralProphetConfig(
        ...     lag_days=48,
        ...     n_forecast=24,
        ...     use_decomposition=True
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
        self._decomposer: BandpassDecomposer | RubinVMDDecomposer | None = None
        self._regressor_cols: list[str] = []
        self._training_window_size: int | None = None

    def _decompose(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply signal decomposition based on config.

        Args:
            df: DataFrame with 'ds' and 'y' columns

        Returns:
            DataFrame with decomposed component columns added
        """
        if self.config.decomposer.method == 'none':
            return df

        if self._decomposer is None:
            cfg = self.config.decomposer

            if cfg.method == 'bandpass':
                self._decomposer = BandpassDecomposer(
                    freq=cfg.freq,
                    period_pairs=cfg.period_pairs,
                    filter_type=cfg.filter_type,
                    edge_method=cfg.edge_method,
                    edge_pad_periods=cfg.edge_pad_periods,
                    nan_fill=cfg.nan_fill,
                    nan_fill_period=cfg.nan_fill_period,
                    use_edge_weighting=cfg.use_edge_weighting,
                    butter_order=cfg.butter_order,
                    verbose=cfg.verbose,
                    include_residual=cfg.include_residual,
                )
            elif cfg.method == 'vmd':
                self._decomposer = RubinVMDDecomposer(
                    freq=cfg.freq,
                    alpha=cfg.alpha,
                    K_stage1=cfg.K_stage1,
                    K_stage2=cfg.K_stage2,
                    verbose=cfg.verbose,
                    include_residual=cfg.include_residual,
                )

        return self._decomposer.decompose(df)

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cyclic time features (hour of day, day of year).

        Uses sin/cos encoding to preserve circular nature of time.

        Args:
            df: DataFrame with 'ds' column

        Returns:
            DataFrame with added time feature columns
        """
        df = df.copy()

        # Hour of day (0-23) -> cyclic encoding
        hour = df['ds'].dt.hour + df['ds'].dt.minute / 60.0
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)

        # Day of year (1-366) -> cyclic encoding
        doy = df['ds'].dt.dayofyear
        df['doy_sin'] = np.sin(2 * np.pi * doy / 365.25)
        df['doy_cos'] = np.cos(2 * np.pi * doy / 365.25)

        return df

    def _get_regressor_columns(self, df: pd.DataFrame) -> list[str]:
        """Get decomposed component and time feature columns to use as regressors.

        Args:
            df: DataFrame with decomposed columns and time features

        Returns:
            List of column names to use as lagged regressors
        """
        valid_cols = []

        # Add decomposed component columns
        if self.config.use_decomposition:
            candidate_cols = [
                c for c in df.columns
                if c.startswith('y_') and c not in ['y_trend']
            ]

            # Filter by variance threshold to remove near-constant features
            variance_threshold = 1e-6
            for col in candidate_cols:
                if col in df.columns and df[col].var() > variance_threshold:
                    valid_cols.append(col)

        # Add time features if present
        time_features = ['hour_sin', 'hour_cos', 'doy_sin', 'doy_cos']
        for col in time_features:
            if col in df.columns:
                valid_cols.append(col)

        return valid_cols

    def fit(
        self,
        df: pd.DataFrame,
        window_size: int | None = None,
        verbose: bool = False,
    ) -> 'NeuralProphetForecaster':
        """Fit the NeuralProphet model to training data.

        Args:
            df: Training data with columns:
                - ds (datetime): Timestamps
                - y (float): Target values
            window_size: Optional limit on training data size (most recent N samples)
            verbose: Whether to print training progress

        Returns:
            Self for method chaining

        Raises:
            ValueError: If data is invalid or insufficient
        """
        # Validate input
        df = validate_input(df)

        # Apply date filtering from config
        if self.config.train_end_date is not None:
            end_date = pd.to_datetime(self.config.train_end_date)
            df = df[df['ds'] <= end_date]

        if self.config.train_start_date is not None:
            start_date = pd.to_datetime(self.config.train_start_date)
            df = df[df['ds'] >= start_date]

        # Apply window size limit
        if window_size is not None and len(df) > window_size:
            df = df.tail(window_size).copy()
        elif self._training_window_size is not None and len(df) > self._training_window_size:
            df = df.tail(self._training_window_size).copy()

        self._training_window_size = len(df)

        if verbose:
            print(f"Training data: {len(df)} samples")
            print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")

        # Apply decomposition if configured
        if self.config.use_decomposition:
            df = self._decompose(df)

        # Add time features (hour of day, day of year) if enabled
        if self.config.use_time_features:
            df = self._add_time_features(df)

        # Get regressor columns (decomposition + time features)
        self._regressor_cols = self._get_regressor_columns(df)
        if verbose:
            print(f"Regressor columns: {len(self._regressor_cols)}")

        # Prepare data for NeuralProphet
        cols_to_keep = ['ds', 'y'] + self._regressor_cols
        df_model = df[[c for c in cols_to_keep if c in df.columns]].copy()

        # Ensure regular frequency
        df_model = prepare_regular_frequency(df_model, freq=self.config.freq)

        # Handle missing values
        if self.config.impute_missing:
            df_model = df_model.interpolate(method='linear')

        # Initialize NeuralProphet
        self.model_ = NeuralProphet(
            n_lags=self.config.lag_days,
            n_forecasts=self.config.n_forecast,
            learning_rate=self.config.learning_rate,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            yearly_seasonality=self.config.yearly_seasonality,
            weekly_seasonality=self.config.weekly_seasonality,
            daily_seasonality=self.config.daily_seasonality,
            seasonality_mode=self.config.seasonality_mode,
            n_changepoints=self.config.n_changepoints,
            changepoints_range=self.config.changepoints_range,
            ar_reg=self.config.ar_reg,
            trend_reg=self.config.trend_reg,
            loss_func=self.config.loss_func,
            optimizer=self.config.optimizer,
            quantiles=self.config.quantiles,
            normalize='off',
            ar_layers=self.config.ar_layers if self.config.ar_layers else [],
        )

        # Add decomposed features as lagged regressors
        for col in self._regressor_cols:
            if col in df_model.columns:
                self.model_.add_lagged_regressor(col, n_lags=self.config.lag_days)

        # Prepare validation split if early stopping enabled
        if self.config.early_stopping and self.config.valid_pct > 0:
            df_train, df_val = self.model_.split_df(
                df_model,
                freq=self.config.freq,
                valid_p=self.config.valid_pct,
            )
            # Train with early stopping (NeuralProphet doesn't expose patience parameter)
            self.metrics_ = self.model_.fit(
                df_train,
                freq=self.config.freq,
                validation_df=df_val,
                early_stopping=True,
                progress="bar",  # Show progress bar with metrics
            )
        else:
            # Train without early stopping
            self.metrics_ = self.model_.fit(df_model, freq=self.config.freq, progress="bar")

        return self.metrics_

    def predict(
        self,
        df: pd.DataFrame | None = None,
        periods: int | None = None,
    ) -> pd.DataFrame:
        """Generate forecasts using the fitted NeuralProphet model.

        Args:
            df: Historical data for autoregressive prediction.
                Must contain at least lag_days of observations.
                Required for NeuralProphet due to AR component.
            periods: Number of periods to forecast (overrides config.n_forecast)

        Returns:
            DataFrame with NeuralProphet forecast columns:
                - ds: Forecast timestamps
                - yhat1, yhat2, ...: Point forecasts for each step
                - yhat_lower1, yhat_upper1, ...: Uncertainty bounds

        Raises:
            RuntimeError: If model hasn't been fitted yet
            ValueError: If df is not provided or has insufficient data
        """
        if self.model_ is None:
            raise RuntimeError(
                "Model has not been fitted yet. Call fit() before predict()."
            )

        if df is None:
            raise ValueError(
                "NeuralProphet requires recent historical data for prediction. "
                "Please provide a DataFrame with at least {self.config.lag_days} observations."
            )

        # Validate and prepare input
        df = validate_input(df)

        # Apply decomposition if configured
        if self.config.use_decomposition:
            df = self._decompose(df)

        # Add time features if enabled
        if self.config.use_time_features:
            df = self._add_time_features(df)

        # Prepare data
        cols_to_keep = ['ds', 'y'] + self._regressor_cols
        df_model = df[[c for c in cols_to_keep if c in df.columns]].copy()

        # Ensure regular frequency
        df_model = prepare_regular_frequency(df_model, freq=self.config.freq)

        # Handle missing values
        if self.config.impute_missing:
            df_model = df_model.interpolate(method='linear')

        # Use most recent data needed for AR
        n_needed = self.config.lag_days + self.config.n_forecast + 10
        if len(df_model) > n_needed:
            df_model = df_model.tail(n_needed).copy()

        # Create future dataframe
        # n_historic_predictions=True gives us predictions for historical timestamps
        # where yhat{step} = forecast for ds + step intervals (what we need for residuals)
        future = self.model_.make_future_dataframe(
            df=df_model,
            periods=periods or self.config.n_forecast,
            n_historic_predictions=True,
        )

        # Generate predictions
        forecast = self.model_.predict(future, decompose=False)

        # Ensure ds is in local time (America/Santiago)
        if forecast['ds'].dt.tz is None:
            forecast['ds'] = forecast['ds'].dt.tz_localize('UTC').dt.tz_convert('America/Santiago')
        else:
            forecast['ds'] = forecast['ds'].dt.tz_convert('America/Santiago')

        return forecast

    def _fit_and_predict(
        self,
        df_history: pd.DataFrame,
        forecast_time: pd.Timestamp,
    ) -> pd.DataFrame:
        """Fit on history and generate forecast for validation.

        This method is called by ValidationMixin.validate() for each forecast time.
        It recomputes decomposition to avoid data leakage.

        Args:
            df_history: Historical data up to forecast_time
            forecast_time: The time at which the forecast is issued

        Returns:
            DataFrame with columns: ds, yhat1, yhat2, ...
        """
        # For NeuralProphet, we use the already-fitted model
        # but recompute decomposition on the current history
        return self.predict(df_history)

    def standardize_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert NeuralProphet output to standardized format.

        Transforms multi-step forecast output to a consistent schema.

        Args:
            df: NeuralProphet forecast output with yhat1, yhat2, ... columns

        Returns:
            DataFrame with standardized columns:
                - ds: Target timestamp
                - yhat: Point forecast (from yhat1)
                - yhat_lower: Lower bound (from first quantile)
                - yhat_upper: Upper bound (from second quantile)
                - step: Forecast horizon (always 1 for first step)

        Note:
            For multi-step analysis, use the raw output with yhat1, yhat2, etc.
        """
        df = df.copy()

        # Use first step for standardized output
        result = pd.DataFrame({
            'ds': df['ds'],
            'yhat': df['yhat1'] if 'yhat1' in df.columns else df.get('yhat'),
            'step': 1,
        })

        # Add uncertainty bounds if available
        if 'yhat1 16.0%' in df.columns:
            result['yhat_lower'] = df['yhat1 16.0%']
        elif 'yhat_lower1' in df.columns:
            result['yhat_lower'] = df['yhat_lower1']

        if 'yhat1 84.0%' in df.columns:
            result['yhat_upper'] = df['yhat1 84.0%']
        elif 'yhat_upper1' in df.columns:
            result['yhat_upper'] = df['yhat_upper1']

        return result

    def save(self, path: str | Path) -> None:
        """Save the fitted NeuralProphet model to disk.

        Saves the model, configuration, and metadata.

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

        # Save NeuralProphet model
        save(self.model_, str(path / 'model.np'))

        # Save config
        self.config.to_yaml(path / 'config.yaml')

        # Save metadata
        metadata = {
            'regressor_cols': self._regressor_cols,
            'training_window_size': self._training_window_size,
        }
        with open(path / 'metadata.json', 'w') as f:
            json.dump(metadata, f)

    @classmethod
    def load(cls, path: str | Path) -> 'NeuralProphetForecaster':
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
        model_path = path / 'model.np'
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        forecaster.model_ = load(str(model_path))

        # Load metadata
        metadata_path = path / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            forecaster._regressor_cols = metadata.get('regressor_cols', [])
            forecaster._training_window_size = metadata.get('training_window_size')

        # Recreate decomposer if decomposition was used
        if config.use_decomposition:
            cfg = config.decomposer
            if cfg.method == 'bandpass':
                forecaster._decomposer = BandpassDecomposer(
                    freq=cfg.freq,
                    period_pairs=cfg.period_pairs,
                    filter_type=cfg.filter_type,
                    edge_method=cfg.edge_method,
                    edge_pad_periods=cfg.edge_pad_periods,
                    nan_fill=cfg.nan_fill,
                    nan_fill_period=cfg.nan_fill_period,
                    use_edge_weighting=cfg.use_edge_weighting,
                    butter_order=cfg.butter_order,
                    verbose=cfg.verbose,
                    include_residual=cfg.include_residual,
                )
            elif cfg.method == 'vmd':
                forecaster._decomposer = RubinVMDDecomposer(
                    freq=cfg.freq,
                    alpha=cfg.alpha,
                    K_stage1=cfg.K_stage1,
                    K_stage2=cfg.K_stage2,
                    verbose=cfg.verbose,
                    include_residual=cfg.include_residual,
                )

        return forecaster
