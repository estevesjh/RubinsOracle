"""NeuralProphet-based forecaster implementation for Rubin's Oracle.

This module implements the NeuralProphetForecaster class using NeuralProphet
for neural network-based time series forecasting.
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from rubin_oracle.base import ValidationMixin
from rubin_oracle.config import NeuralProphetConfig
from rubin_oracle.preprocessing import (
    BandpassDecomposer,
    RubinVMDDecomposer,
)
from rubin_oracle.utils import (
    MetricsCalculator,
    OutputFormatter,
    SampleConverter,
    add_time_features,
    prepare_regular_frequency,
    validate_input,
)

# Force tqdm to show in terminal
os.environ.setdefault("TQDM_DISABLE", "0")

# Set PyTorch Lightning to show progress
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

try:
    from neuralprophet import NeuralProphet, load, save
except ImportError as err:
    raise ImportError(
        "NeuralProphet is required but not installed. Install it with: pip install neuralprophet"
    ) from err


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
        self.name = "neural_prophet"
        self.model_: NeuralProphet | None = None
        self._decomposer: BandpassDecomposer | RubinVMDDecomposer | None = None
        self._regressor_cols: list[str] = []
        self._training_window_size: int | None = None

        # Cache sample conversions
        self._converter = SampleConverter(
            freq=config.freq,
            lag_days=config.lag_days,
            n_forecast=config.n_forecast,
        )

    def _decompose(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply signal decomposition based on config.

        Args:
            df: DataFrame with 'ds' and 'y' columns

        Returns:
            DataFrame with decomposed component columns added
        """
        if self.config.decomposer.method == "none":
            return df

        if self._decomposer is None:
            cfg = self.config.decomposer

            if cfg.method == "bandpass":
                self._decomposer = BandpassDecomposer(
                    freq=cfg.freq,
                    period_pairs=cfg.period_pairs,
                    filter_type=cfg.filter_type,
                    # Pre-filter edge padding
                    edge_method=cfg.edge_method,
                    edge_pad_periods=cfg.edge_pad_periods,
                    # Post-filter edge correction
                    pad_method=cfg.pad_method,
                    pad_num_periods=cfg.pad_num_periods,
                    pad_max_periods=cfg.pad_max_periods,
                    pad_target_periods=cfg.pad_target_periods,
                    pad_arima_order=cfg.pad_arima_order,
                    pad_bands=cfg.pad_bands,
                    # NaN handling
                    nan_fill=cfg.nan_fill,
                    nan_fill_period=cfg.nan_fill_period,
                    nan_fill_max_gap=cfg.nan_fill_max_gap,
                    # Filter parameters
                    savgol_polyorder=cfg.savgol_polyorder,
                    butter_order=cfg.butter_order,
                    verbose=cfg.verbose,
                )
            elif cfg.method == "vmd":
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
        return add_time_features(df)

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
            candidate_cols = [c for c in df.columns if c.startswith("y_") and c not in ["y_trend"]]

            # Filter by variance threshold to remove near-constant features
            variance_threshold = 1e-6
            for col in candidate_cols:
                if col in df.columns and df[col].var() > variance_threshold:
                    valid_cols.append(col)

        # Add time features if present
        time_features = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
        for col in time_features:
            if col in df.columns:
                valid_cols.append(col)

        return valid_cols

    def fit(
        self,
        df: pd.DataFrame,
        window_size: int | None = None,
        verbose: bool = False,
    ) -> NeuralProphetForecaster:
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
            df = df[df["ds"] <= end_date]

        if self.config.train_start_date is not None:
            start_date = pd.to_datetime(self.config.train_start_date)
            df = df[df["ds"] >= start_date]

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
        cols_to_keep = ["ds", "y"] + self._regressor_cols
        df_model = df[[c for c in cols_to_keep if c in df.columns]].copy()

        # Ensure regular frequency
        df_model = prepare_regular_frequency(df_model, freq=self.config.freq)

        # Handle missing values
        if self.config.impute_missing:
            df_model = df_model.interpolate(method="linear")

        # Initialize NeuralProphet
        self.model_ = NeuralProphet(
            n_lags=self._converter.lag_samples,
            n_forecasts=self._converter.n_forecast_samples,
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
            ar_layers=self.config.ar_layers if self.config.ar_layers else [],
        )

        # Add custom seasonalities if defined
        if self.config.custom_seasonalities:
            for season in self.config.custom_seasonalities:
                self.model_.add_seasonality(
                    name=season["name"],
                    period=season["period"],
                    fourier_order=season.get("fourier_order", 7),
                    condition_name=season.get("condition_name", None),
                )

        # Add decomposed features as lagged regressors
        for col in self._regressor_cols:
            if col in df_model.columns:
                self.model_.add_lagged_regressor(col, n_lags=self._converter.lag_samples)

        # Train without early stopping
        self._np_metrics_ = self.model_.fit(df_model, freq=self.config.freq, progress="bar")

        # keep the fitted dataset
        self._fit_df = df_model
        latest_timestamp = df_model["ds"].max()
        self.latest_timestamp = latest_timestamp.tz_localize("America/Santiago")

        # Compute standardized metrics
        self._compute_metrics()

        return self

    def _compute_metrics(self, df: pd.DataFrame = None) -> None:
        """Compute in-sample fit metrics.

        Args:
            df: Training data with 'ds' and 'y' columns
        """
        if self.model_ is None:
            self.metrics_ = None
            return

        if df is None:
            merged = self.fitted(window_days=14)
        else:
            # Get in-sample predictions (fitted values)
            fitted = self.forecast(df, include_historical_data=True)
            # Merge to get y values aligned with predictions
            df_eval = df[["ds", "y"]].copy()
            if df_eval["ds"].dt.tz is not None:
                df_eval["ds"] = df_eval["ds"].dt.tz_localize(None)
            if fitted["ds"].dt.tz is not None:
                fitted["ds"] = fitted["ds"].dt.tz_localize(None)
            merged = fitted.merge(df_eval, on="ds", how="inner")

        y_true = merged["y"].dropna().values
        y_pred = merged["yhat"].dropna().values

        # Ensure same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

        if len(y_true) == 0:
            self.metrics_ = None
            return

        # Compute metrics using helper
        self.metrics_ = MetricsCalculator.compute_metrics(y_true, y_pred)

    # @cached_property
    def fitted(self, window_days: int) -> pd.DataFrame:
        """Generate forecasts up to the training data maximum date.

        Returns forecasts for all laggged timestamps in the fitted training data
        using a back rolling window approach.
        """
        forecast = self.forecast(self._fit_df, window_days=window_days)
        if forecast["ds"].dt.tz is not None:
            forecast["ds"] = forecast["ds"].dt.tz_localize(None)

        # Drop 'y' from forecast if present to avoid suffix issues in merge
        # We want 'y' from self._fit_df (actual values), not from forecast
        if "y" in forecast.columns:
            forecast = forecast.drop(columns=["y"])

        merged = forecast.merge(self._fit_df[["ds", "y"]], on="ds", how="left")
        columns = ["ds", "issue_time", "y", "yhat"]
        for cols in forecast.columns.to_list():
            if cols not in columns:
                columns.append(cols)
        return merged[columns].dropna(subset=["yhat"])

    def forecast(
        self,
        df_test: pd.DataFrame,
        issue_time: pd.Timestamp | None = None,
        include_history: bool = True,
        window_days: float = 14,
    ) -> pd.DataFrame:
        """Generate forecasts using rolling window approach with optional issue time filtering.

        This method generates multi-step ahead forecasts for test data. When
        `include_historical_data=True`, it automatically prepends necessary historical
        data from the training set to provide NeuralProphet with sufficient autoregressive
        context (lag_days).

        **Filtering Behavior:**
        - If `issue_time` is provided: Returns only forecasts issued at that specific time
        - If `issue_time` is None: Returns all forecasts for all issue times in the data

        Args:
            df_test: Test data DataFrame with columns:
                - ds (datetime): Forecast timestamps (should extend beyond training period)
                - y (float): Actual values (can be NaN for pure forecasting)
            issue_time: The forecast issue timestamp (when the forecast is made).
                If provided, filters output to return only predictions issued at this time.
                If None (default), returns all forecasts across all issue times.
            include_history: Whether to prepend historical data from training set
                to provide autoregressive context (default: True).
                Set to False to forecast using only df_test data.

        Returns:
            DataFrame with forecast columns:
                - ds: Forecast timestamp
                - yhat: Point forecast
                - yhat_lower: Lower uncertainty bound (if quantiles enabled)
                - yhat_upper: Upper uncertainty bound (if quantiles enabled)
                - step: Forecast step (1, 2, ..., n_forecast)
                - issue_time: Forecast issue time (when prediction was made)

        Raises:
            RuntimeError: If model hasn't been fitted yet
            ValueError: If df_test is None or has invalid structure

        Example:
            >>> forecaster.fit(df_train)
            >>> # Forecast with historical data (prepended automatically)
            >>> forecast = forecaster.forecast(df_test)
            >>> # Forecast filtered to specific issue time
            >>> forecast = forecaster.forecast(df_test, pd.Timestamp('2024-01-15 12:00:00'))
            >>> # Forecast without prepending historical data
            >>> forecast = forecaster.forecast(df_test, include_historical_data=False)
        """
        if df_test["ds"].dt.tz is not None:
            df_test["ds"] = df_test["ds"].dt.tz_convert(None)

        if self._decompose is not None:
            start_date = (self.latest_timestamp).tz_convert(None)
            test_date_min = df_test["ds"].min()
            if test_date_min > self._fit_df["ds"].min():
                history = self._fit_df[self._fit_df["ds"] <= start_date]
                df_test_up = df_test[df_test["ds"] >= start_date]
                df_input = pd.concat([history, df_test_up])
            else:
                df_input = df_test

        elif include_history:
            test_date_min = df_test["ds"].min()
            lag_days = self.config.lag_days + window_days
            start_date = (self.latest_timestamp - pd.Timedelta(days=lag_days)).tz_convert(None)

            # add historical data
            if test_date_min >= start_date:
                df_test_up = df_test[df_test["ds"] > self.latest_timestamp.tz_convert(None)]
                history = self._fit_df[self._fit_df["ds"] >= start_date]
                df_input = pd.concat([history, df_test_up])
            else:
                df_input = df_test
        else:
            df_input = df_test

        _forecast = self.predict(df_input, include_history=True, window_days=window_days)
        forecast = self.standardize_output(_forecast)
        forecast = forecast.merge(_forecast[["ds", "y"]], on="ds", how="inner")
        if issue_time is not None:
            return forecast[forecast["issue_time"] == issue_time].copy()
        else:
            return forecast

    def predict(
        self, df: pd.DataFrame, include_history: bool | None = False, window_days: float | None = 14
    ) -> pd.DataFrame:
        """Generate forecasts using NeuralProphet with rolling window approach.

        Extends input df with NaN values up to n_forecast steps to ensure all historical
        timestamps have valid multi-step forecasts (yhat1, yhat2, ..., yhat{n_forecast}).

        **NeuralProphet Idiosyncrasies (Important):**

        NeuralProphet makes a back rolling window predictions where each yhat_i is a forecast
        i steps ahead. For a given timestamp t:
        - yhat1 = forecast for t+1 (next period)
        - yhat2 = forecast for t+2
        - ...
        - yhat{n_forecast} = forecast for t+n_forecast

        The critical quirk: NeuralProphet only produces valid predictions when the input
        dataframe has future values to reference. If your df ends at timestamp T, only
        timestamps up to T - n_forecast will have valid yhat1 values. Rows after that
        will be NaN because there's insufficient future data.

        **This method solves this by:**
        1. Taking the input df
        2. Extending it with n_forecast NaN rows to simulate future values
        3. Passing the extended df directly to model.predict()

        Now every timestamp in the original df has n_forecast rows after it, so all
        yhat1, yhat2, ..., yhat{n_forecast} columns will have valid values.

        The extended NaN rows are a side effect - the forecast you care about is for
        the original df rows. You can merge the forecast back to the original df by
        joining on 'ds'.

        Args:
            df: Historical data for autoregressive prediction.
                Must contain at least lag_days observations.
                Must have 'ds' (datetime) and 'y' (target) columns.

        Returns:
            DataFrame with columns:
                - ds: Forecast timestamps (includes both original and NaN-extended rows)
                - yhat1, yhat2, ...: Point forecasts for each step
                - yhat_lower1, yhat_upper1, ...: Uncertainty bounds (if quantiles configured)

            Note: Predictions for the NaN-extended rows may be less reliable than
            the original df rows.

        Raises:
            RuntimeError: If model hasn't been fitted yet
            ValueError: If df is None or has insufficient data
        """
        if self.model_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() before predict().")

        if df is None:
            raise ValueError(
                f"NeuralProphet requires recent historical data for prediction. "
                f"Please provide a DataFrame with at least {self.config.lag_days} observations."
            )

        # Prepare the input data (validate, decompose, add time features, regularize frequency)
        df_model = self._prepare_df(
            df=df,
            regressor_cols=self._regressor_cols,
            config=self.config,
            decomposer=self._decomposer,
        )

        # Use most recent data needed for AR (autoregressive)
        # Keep: lag_samples (for looking back) + n_forecast_samples (for rolling window) + buffer
        n_needed = self._converter.lag_samples + self._converter.n_forecast_samples

        if (not include_history) and (not self._decompose):
            if len(df_model) > n_needed:
                df_model = df_model.tail(n_needed).copy()

        # Extend with NaN rows for n_forecast_samples periods
        # This ensures every historical timestamp has enough "future" to reference
        df_extended = self._extend_with_nans(
            df=df_model,
            periods=self._converter.n_forecast_samples,
            freq=self.config.freq,
            regressor_cols=self._regressor_cols,
        )

        # Generate predictions directly from the extended dataframe
        # No need for make_future_dataframe() - pass extended df directly
        window_size = int(window_days * self._converter.steps_per_day)
        forecast = self.model_.predict(df_extended.tail(window_size), decompose=False)

        # Ensure ds is in local time (America/Santiago)
        if forecast["ds"].dt.tz is None:
            forecast["ds"] = forecast["ds"].dt.tz_localize("America/Santiago")
        else:
            forecast["ds"] = forecast["ds"].dt.tz_convert("America/Santiago")

        return forecast

    def standardize_output(self, forecast: pd.DataFrame) -> pd.DataFrame:
        """Convert NeuralProphet multi-step forecast to standardized long format.

        NeuralProphet produces multi-step forecasts where yhat{i} represents a prediction
        i steps ahead. This method transforms the wide format (one row per forecast timestamp
        with yhat1, yhat2, ..., yhat{n_forecast} columns) to a long format (one row per
        step per forecast timestamp).

        Args:
            forecast: NeuralProphet forecast output with columns:
                - ds: Forecast timestamp
                - yhat1, yhat2, ..., yhat{n_forecast}: Point forecasts for each step
                - (optional) yhat_lower1, yhat_upper1, ...: Uncertainty bounds

        Returns:
            DataFrame with standardized columns in long format:
                - ds, yhat, yhat_lower, yhat_upper, step
        """
        return OutputFormatter.standardize_neuralprophet_output(
            forecast, self.config.freq, self._converter.n_forecast_samples
        )

    def save(self, path: str | Path) -> None:
        """Save the fitted NeuralProphet model to disk.

        Saves the model, configuration, and metadata.

        Args:
            path: Directory path where model will be saved

        Raises:
            RuntimeError: If model hasn't been fitted yet
        """
        if self.model_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() before save().")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save NeuralProphet model
        save(self.model_, str(path / "model.np"))

        # Save config
        self.config.to_yaml(path / "config.yaml")

        # Save metadata
        metadata = {
            "regressor_cols": self._regressor_cols,
            "training_window_size": self._training_window_size,
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f)

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
        config_path = path / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        config = NeuralProphetConfig.from_yaml(config_path)

        # Create instance
        forecaster = cls(config)

        # Load NeuralProphet model
        model_path = path / "model.np"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        forecaster.model_ = load(str(model_path))

        # Load metadata
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            forecaster._regressor_cols = metadata.get("regressor_cols", [])
            forecaster._training_window_size = metadata.get("training_window_size")

        # Recreate decomposer if decomposition was used
        if config.use_decomposition:
            cfg = config.decomposer
            if cfg.method == "bandpass":
                forecaster._decomposer = BandpassDecomposer(
                    freq=cfg.freq,
                    period_pairs=cfg.period_pairs,
                    filter_type=cfg.filter_type,
                    # Pre-filter edge padding
                    edge_method=cfg.edge_method,
                    edge_pad_periods=cfg.edge_pad_periods,
                    # Post-filter edge correction
                    pad_method=cfg.pad_method,
                    pad_num_periods=cfg.pad_num_periods,
                    pad_max_periods=cfg.pad_max_periods,
                    pad_target_periods=cfg.pad_target_periods,
                    pad_arima_order=cfg.pad_arima_order,
                    pad_bands=cfg.pad_bands,
                    # NaN handling
                    nan_fill=cfg.nan_fill,
                    nan_fill_period=cfg.nan_fill_period,
                    nan_fill_max_gap=cfg.nan_fill_max_gap,
                    # Filter parameters
                    savgol_polyorder=cfg.savgol_polyorder,
                    butter_order=cfg.butter_order,
                    verbose=cfg.verbose,
                )
            elif cfg.method == "vmd":
                forecaster._decomposer = RubinVMDDecomposer(
                    freq=cfg.freq,
                    alpha=cfg.alpha,
                    K_stage1=cfg.K_stage1,
                    K_stage2=cfg.K_stage2,
                    verbose=cfg.verbose,
                    include_residual=cfg.include_residual,
                )

        return forecaster

    @staticmethod
    def _prepare_df(
        df: pd.DataFrame,
        regressor_cols: list[str],
        config,
        decomposer=None,
    ) -> pd.DataFrame:
        """Prepare input dataframe for NeuralProphet prediction.

        Common preprocessing for both fitted() and forecast() methods.
        Handles: validation, decomposition, time features, frequency regularization,
        missing value imputation.

        Skips decomposition and time feature addition if already applied to avoid
        reprocessing and creating NaN values in regressor columns.

        Args:
            df: Raw input dataframe with 'ds' and 'y' columns
            regressor_cols: List of regressor column names to include
            config: NeuralProphetConfig object with settings
            decomposer: Optional decomposer instance for signal decomposition

        Returns:
            Prepared dataframe with proper columns and frequency
        """
        # Validate and prepare input
        df = validate_input(df)

        # Check if decomposition has already been applied
        # (indicated by presence of decomposed component columns like y_trend, y_c0, etc.)
        has_decomposed_cols = any(col.startswith("y_") for col in df.columns if col != "y")

        # Apply decomposition only if decomposer is provided and not already applied
        if decomposer is not None and not has_decomposed_cols:
            df = decomposer.decompose(df)

        # Check if time features have already been added
        has_time_features = all(
            col in df.columns for col in ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
        )

        # Add time features only if enabled and not already present
        if config.use_time_features and not has_time_features:
            df = add_time_features(df)

        # Prepare data - keep only relevant columns
        cols_to_keep = ["ds", "y"] + regressor_cols
        df_model = df[[c for c in cols_to_keep if c in df.columns]].copy()

        # Ensure regular frequency
        df_model = prepare_regular_frequency(df_model, freq=config.freq)

        # Handle missing values
        if config.impute_missing:
            df_model = df_model.interpolate(method="linear")

        return df_model

    @staticmethod
    def _extend_with_nans(
        df: pd.DataFrame,
        periods: int,
        freq: str,
        regressor_cols: list[str],
    ) -> pd.DataFrame:
        """Extend dataframe with NaN rows for future periods.

        This allows NeuralProphet to generate valid predictions for all historical
        timestamps by providing future dates to reference.

        Args:
            df: Dataframe with 'ds' and 'y' columns
            periods: Number of periods to extend
            freq: Frequency string (e.g., '15min', 'H', 'D')
            regressor_cols: List of regressor column names to add as NaN

        Returns:
            Extended dataframe with NaN values for future rows
        """
        df = df.copy()
        last_date = df["ds"].max()

        # Generate future dates
        future_dates = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[
            1:
        ]  # Skip the last training date

        # Create rows with NaNs for future periods
        future_rows = pd.DataFrame(
            {
                "ds": future_dates,
                "y": np.nan,
            }
        )

        # Add NaN columns for regressors
        for col in regressor_cols:
            if col in df.columns:
                future_rows[col] = np.nan

        result = pd.concat([df, future_rows], ignore_index=True)
        return result

    def plot(
        self,
        df_test: pd.DataFrame | None = None,
        lead_time: float = -1.0,
        window_days: float | None = 7,
        ax: plt.Axes | None = None,
        figsize: tuple[float, float] = (14, 6),
        title: str | None = None,
        show_residuals: bool = False,
    ) -> tuple[plt.Figure, plt.Axes | tuple[plt.Axes, plt.Axes]]:
        """Plot fitted and actual values for a given lead time.

        Shows the model's predictions (yhat) for a specific forecast horizon (lead_time)
        overlaid with actual values from training data. Optionally includes test data and
        its forecasts.

        **Lead Time Explanation:**
        Lead time is the forecast horizon in hours - how far ahead you're predicting.
        For example:
        - If freq='15min' and lead_time=1.0, that's 4 steps ahead (1 hour / 15min = 4)
        - If freq='H' and lead_time=12.0, that's 12 steps ahead

        Args:
            lead_time: Forecast horizon in hours (float). Default -1.0 shows the
                default/primary forecast. Examples: 0.25 (15 min), 1.0 (1 hour),
                12.0 (12 hours), 24.0 (1 day). Must be >= config.freq and <=
                (config.n_forecast × config.freq).
            ax: Optional matplotlib axes to plot on. If None, creates new figure and axes.
            df_test: Optional test dataframe with 'ds' and 'y' columns. If provided,
                overlays actual test values on the plot.
            window_days: Number of days to display from the end backwards (default: 7).
                If None, shows all training data. Does not affect test data display.
            figsize: Figure size as (width, height) tuple in inches (default: (14, 6)).
            title: Custom plot title. If None, auto-generated from model name and lead_time.
            show_residuals: Whether to show residuals plot below main plot (default: False).

        Returns:
            tuple: (fig, axes) where axes is either a single Axes object or tuple of
                (ax_main, ax_residuals) if show_residuals is True.

        Raises:
            RuntimeError: If model hasn't been fitted (model_ is None).
            RuntimeError: If fitted data not available (_fit_df not set).

        Example:
            >>> forecaster = NeuralProphetForecaster(config)
            >>> forecaster.fit(df_train)
            >>> fig, ax = forecaster.plot(lead_time=12.0)
            >>> plt.show()

            >>> # With test data and residuals
            >>> fig, (ax_main, ax_resid) = forecaster.plot(
            ...     lead_time=1.0, df_test=df_test, show_residuals=True
            ... )
            >>> plt.show()
        """
        if self.model_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() before plot().")

        if not hasattr(self, "_fit_df"):
            raise RuntimeError("Fitted data not available. Ensure fit() was called properly.")

        # Get fitted values using clean API
        fitted_forecast = self.fitted(window_days + self.config.n_forecast)
        fitted_forecast["ds"] = pd.to_datetime(fitted_forecast["ds"])

        # Apply window filter to training data
        if window_days is not None:
            cutoff = (self.latest_timestamp - pd.Timedelta(days=window_days)).tz_convert(None)
            fitted_forecast = fitted_forecast[fitted_forecast["ds"] >= cutoff]

        # Split between train and test data
        forecast_start = self.latest_timestamp.tz_convert(None)
        forecast_df = fitted_forecast[fitted_forecast["ds"] >= forecast_start].copy()
        fit_merged = fitted_forecast[fitted_forecast["ds"] <= forecast_start].copy()

        # Make a selection for the lead_time
        if lead_time <= 0:
            fit_merged = fit_merged.drop_duplicates("ds").groupby("ds").first().reset_index()
        else:
            fit_merged = fit_merged[fit_merged["lead_time"] == lead_time].copy()

        fit_merged["residual"] = fit_merged["y"] - fit_merged["yhat"]
        # Drop duplicates and keep first occurrence per ds
        forecast_df = forecast_df.drop_duplicates("ds").groupby("ds").first().reset_index()

        # Create figure
        if ax is None:
            if show_residuals:
                fig, (ax_main, ax_resid) = plt.subplots(
                    2, 1, figsize=figsize, height_ratios=[3, 1], sharex=True
                )
            else:
                fig, ax_main = plt.subplots(figsize=figsize)
                ax_resid = None
        else:
            ax_main = ax
            ax_resid = None
            fig = ax.get_figure()

        # Plot training data and fitted model
        ax_main.plot(fit_merged["ds"], fit_merged["y"], "k-", lw=1, alpha=0.8, label="Data")
        ax_main.plot(
            fit_merged["ds"], fit_merged["yhat"], "r-", lw=1, alpha=0.8, label="Fitted model"
        )
        ax_main.fill_between(
            fit_merged["ds"],
            fit_merged["yhat_lower"],
            fit_merged["yhat_upper"],
            color="red",
            alpha=0.1,
            label="Uncertainty",
        )

        # Mark forecast start
        ax_main.axvline(
            forecast_start, color="gray", linestyle="--", alpha=0.5, label="Forecast Time"
        )

        # Plot forecast
        ax_main.plot(forecast_df["ds"], forecast_df["yhat"], "b-", lw=2, label="Forecast")
        if "yhat_lower" in forecast_df.columns and "yhat_upper" in forecast_df.columns:
            ax_main.fill_between(
                forecast_df["ds"],
                forecast_df["yhat_lower"],
                forecast_df["yhat_upper"],
                color="blue",
                alpha=0.1,
            )

        # Overlay test data if provided
        if df_test is not None:
            df_test_plot = df_test.copy()
            df_test_plot["ds"] = pd.to_datetime(df_test_plot["ds"])
            ax_main.plot(
                df_test_plot["ds"], df_test_plot["y"], "g-", lw=2, alpha=0.8, label="Actual"
            )

        ax_main.set_ylabel("Value")
        ax_main.set_title(title or f"{self.name} - Fit and Forecast")
        ax_main.legend(loc="best")
        ax_main.grid(True, alpha=0.3)

        # Plot residuals if requested
        if show_residuals and ax_resid is not None:
            ax_resid.plot(fit_merged["ds"], fit_merged["residual"], "g-", lw=0.5, alpha=0.8)
            ax_resid.axhline(0, color="k", linestyle="-", alpha=0.3)
            ax_resid.set_xlabel("Date")
            ax_resid.set_ylabel("Residual")
            ax_resid.grid(True, alpha=0.3)

        if ax is None:
            plt.tight_layout()

        if show_residuals:
            return fig, (ax_main, ax_resid)
        return fig, ax_main
