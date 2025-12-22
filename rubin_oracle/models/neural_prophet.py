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
        "NeuralProphet is required but not installed. " "Install it with: pip install neuralprophet"
    ) from err


def _lead_time_to_step(lead_time: float, freq: str, n_forecast: int) -> int:
    """Convert lead time in hours to the corresponding yhat step number.

    Args:
        lead_time: Forecast horizon in hours (e.g., 12.0, 24.0)
        freq: Frequency string (e.g., '15min', 'H')
        n_forecast: Maximum number of forecast steps

    Returns:
        Step number (e.g., 4 for yhat4)

    Raises:
        ValueError: If lead_time is invalid or exceeds n_forecast steps
    """
    try:
        # Convert lead_time (hours) to timedelta
        lead_td = pd.Timedelta(hours=lead_time)
        freq_td = pd.to_timedelta(freq)
    except Exception as e:
        raise ValueError(f"Invalid lead_time '{lead_time}' or freq '{freq}'. " f"Error: {e}") from e

    # Calculate step
    if freq_td <= pd.Timedelta(0):
        raise ValueError(f"Frequency must be positive, got: {freq}")

    step = int(lead_td / freq_td)

    if step < 1:
        raise ValueError(
            f"Lead time '{lead_time}' hours is less than frequency '{freq}'. "
            f"Lead time must be >= frequency."
        )

    if step > n_forecast:
        raise ValueError(
            f"Lead time '{lead_time}' hours corresponds to step {step}, "
            f"but n_forecast is {n_forecast}."
        )

    return step


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
        if self.config.decomposer.method == "none":
            return df

        if self._decomposer is None:
            cfg = self.config.decomposer

            if cfg.method == "bandpass":
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
        df = df.copy()

        # Hour of day (0-23) -> cyclic encoding
        hour = df["ds"].dt.hour + df["ds"].dt.minute / 60.0
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

        # Day of year (1-366) -> cyclic encoding
        doy = df["ds"].dt.dayofyear
        df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
        df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

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
            ar_layers=self.config.ar_layers if self.config.ar_layers else [],
        )

        # Add decomposed features as lagged regressors
        for col in self._regressor_cols:
            if col in df_model.columns:
                self.model_.add_lagged_regressor(col, n_lags=self.config.lag_days)

        # Train without early stopping
        self._np_metrics_ = self.model_.fit(df_model, freq=self.config.freq, progress="bar")

        # keep the fitted dataset
        self._fit_df = df_model
        self.latest_timestamp = df_model["ds"].max()

        # Compute standardized metrics
        self._compute_metrics(df_model)

        return self

    def _compute_metrics(self, df: pd.DataFrame) -> None:
        """Compute in-sample fit metrics.

        Args:
            df: Training data with 'ds' and 'y' columns
        """
        if self.model_ is None:
            self.metrics_ = None
            return

        try:
            # Get in-sample predictions (fitted values)
            fitted = self.predict(df)

            # Merge to get y values aligned with predictions
            df_eval = df[["ds", "y"]].copy()
            if df_eval["ds"].dt.tz is not None:
                df_eval["ds"] = df_eval["ds"].dt.tz_localize(None)
            if fitted["ds"].dt.tz is not None:
                fitted["ds"] = fitted["ds"].dt.tz_localize(None)

            merged = fitted.merge(df_eval, on="ds", how="inner")

            # Use yhat1 as the primary prediction
            yhat_col = "yhat1" if "yhat1" in merged.columns else "yhat"
            y_true = merged["y"].dropna().values
            y_pred = merged[yhat_col].dropna().values

            # Ensure same length
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]

            if len(y_true) == 0:
                self.metrics_ = None
                return

            # Compute metrics using helper
            self.metrics_ = MetricsCalculator.compute_metrics(y_true, y_pred)
        except Exception:
            # If metrics computation fails, set to None
            self.metrics_ = None

    def fitted(self) -> pd.DataFrame:
        """Generate forecasts up to the training data maximum date.

        Returns forecasts for all timestamps in the fitted training data using a rolling
        window approach.

        **NeuralProphet Idiosyncrasies:**

        NeuralProphet operates as an autoregressive model with a rolling window of size
        `lag_days`. When you call predict(df), it produces forecasts with multiple steps
        (yhat1, yhat2, ..., yhat{n_forecast}). Each yhat_i represents a forecast i steps
        ahead from its reference timestamp.

        However, NeuralProphet only returns predictions where there are enough future
        values to reference. If your dataframe ends at time T, predictions at T will have
        references up to T+n_forecast. Rows near the end (after T - n_forecast) won't have
        valid yhat1 values because there's no future data to predict from.

        **Solution:** Extend the input df with NaN values for n_forecast periods. This gives
        every historical point enough "future" to reference, ensuring all timestamps get
        valid yhat1, yhat2, ..., yhat{n_forecast} values.

        Returns:
            DataFrame with NeuralProphet forecast columns:
                - ds: Forecast timestamps
                - yhat1, yhat2, ...: Point forecasts for each step
                - yhat_lower1, yhat_upper1, ...: Uncertainty bounds (if quantiles enabled)

        Raises:
            RuntimeError: If model hasn't been fitted yet
        """
        if self.model_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() before fitted().")

        if not hasattr(self, "_fit_df"):
            raise RuntimeError("Fitted data not available. Ensure fit() was called properly.")

        return self.predict(self._fit_df, include_history=True)

    def forecast(self, df_test: pd.DataFrame, include_history: bool | None = True) -> pd.DataFrame:
        """Generate forecasts for test data using proper rolling window approach.

        This method addresses NeuralProphet's rolling window requirements by constructing
        the proper input dataframe for test forecasting. It ensures the model has sufficient
        historical context (lag_days + n_forecast) to generate valid predictions.

        **Why this method exists:**

        Unlike predict(), which can work with any historical dataframe, forecast() is
        specifically designed for generating predictions on new test data after the training
        period. It automatically:
        1. Filters test data to only include timestamps after the training period
        2. Prepends the necessary historical window from training data
        3. Passes the combined dataframe to predict() for forecasting

        This ensures NeuralProphet has the required autoregressive context (lag_days) plus
        the rolling window buffer (n_forecast) to generate valid multi-step forecasts.

        Args:
            df_test: Test data with columns:
                - ds (datetime): Timestamps (should extend beyond training period)
                - y (float): Actual values (can be NaN for pure forecasting)

        Returns:
            DataFrame with NeuralProphet forecast columns:
                - ds: Forecast timestamps
                - yhat1, yhat2, ..., yhat{n_forecast}: Point forecasts for each step
                - yhat_lower1, yhat_upper1, ...: Uncertainty bounds (if quantiles enabled)

            Only returns forecasts for timestamps after the training period.

        Raises:
            RuntimeError: If model hasn't been fitted yet
            ValueError: If df_test is None or invalid

        Example:
            >>> forecaster.fit(df_train)  # Train on historical data
            >>> test_forecast = forecaster.forecast(df_test)  # Forecast on new data
            >>> # Get 12-hour ahead predictions
            >>> predictions_12h = test_forecast[['ds', 'yhat48']]  # 48 steps @ 15min
        """
        if include_history:
            issue_time = self.latest_timestamp
            df_test_up = df_test[df_test["ds"] > issue_time]
            history = self._fit_df.tail(self.config.n_forecast + self.config.lag_days)
            df_input = pd.concat([history, df_test_up])
        else:
            df_input = df_test
        return self.predict(df_input)

    def predict(self, df: pd.DataFrame, include_history: bool | None = False) -> pd.DataFrame:
        """Generate forecasts using NeuralProphet with rolling window approach.

        Extends input df with NaN values up to n_forecast steps to ensure all historical
        timestamps have valid multi-step forecasts (yhat1, yhat2, ..., yhat{n_forecast}).

        **NeuralProphet Idiosyncrasies (Important):**

        NeuralProphet makes rolling window predictions where each yhat_i is a forecast
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
        df_model = prepare_df(
            df=df,
            regressor_cols=self._regressor_cols,
            config=self.config,
            decomposer=self._decomposer,
        )

        # Use most recent data needed for AR (autoregressive)
        # Keep: lag_days (for looking back) + n_forecast (for rolling window) + buffer
        n_needed = self.config.lag_days + self.config.n_forecast + 10

        if not include_history:
            if len(df_model) > n_needed:
                df_model = df_model.tail(n_needed).copy()

        # Extend with NaN rows for n_forecast periods
        # This ensures every historical timestamp has enough "future" to reference
        df_extended = extend_with_nans(
            df=df_model,
            periods=self.config.n_forecast,
            freq=self.config.freq,
            regressor_cols=self._regressor_cols,
        )

        # Generate predictions directly from the extended dataframe
        # No need for make_future_dataframe() - pass extended df directly
        forecast = self.model_.predict(df_extended, decompose=False)

        # Ensure ds is in local time (America/Santiago)
        if forecast["ds"].dt.tz is None:
            forecast["ds"] = forecast["ds"].dt.tz_localize("UTC").dt.tz_convert("America/Santiago")
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
            forecast, self.config.freq, self.config.n_forecast
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
                    edge_method=cfg.edge_method,
                    edge_pad_periods=cfg.edge_pad_periods,
                    nan_fill=cfg.nan_fill,
                    nan_fill_period=cfg.nan_fill_period,
                    use_edge_weighting=cfg.use_edge_weighting,
                    butter_order=cfg.butter_order,
                    verbose=cfg.verbose,
                    include_residual=cfg.include_residual,
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

    def plot(
        self,
        lead_time: float = 12.0,
        axs: plt.axes | None = None,
        df_test: pd.DataFrame | None = None,
        window_days: float | None = None,
        title: str | None = None,
        # show_grid: bool = True,
    ) -> plt.Figure:
        """Plot fitted and actual values for a given lead time.

        Shows the model's predictions (yhat{i}) for a specific forecast horizon (lead_time)
        overlaid with actual values from training data. Optionally includes test data and
        its forecasts.

        **Lead Time Explanation:**
        Lead time is the forecast horizon in hours - how far ahead you're predicting.
        For example:
        - If freq='15min' and lead_time=1.0, that's 4 steps ahead (1 hour / 15min = 4)
        - If freq='H' and lead_time=12.0, that's 12 steps ahead
        - The method finds the corresponding yhat column (e.g., yhat4 or yhat12)

        Args:
            lead_time: Forecast horizon in hours (float)
                Examples: 0.25 (15 min), 1.0 (1 hour), 12.0 (12 hours), 24.0 (1 day)
                Must be >= config.freq and <= (config.n_forecast × config.freq)
            axs: Optional matplotlib axes to plot on. If None, creates new axes.
            df_test: Optional test dataframe with 'ds' and 'y' columns
                If provided, overlays actual test values and their forecasts
            window_days: Optional number of days to display (from end backwards).
                If None, shows all training data. If provided, shows only the
                most recent window_days of training data. Does not affect test data.
            title: Custom plot title. If None, auto-generated from lead_time
            show_grid: Whether to show grid on plot

        Returns:
            matplotlib.figure.Figure object for further customization

        Raises:
            RuntimeError: If model hasn't been fitted
            ValueError: If lead_time is invalid for the configured frequency
            ValueError: If lead_time exceeds n_forecast steps

        Example:
            >>> forecaster = NeuralProphetForecaster(config)
            >>> forecaster.fit(df_train)
            >>> fig = forecaster.plot(lead_time=12.0)  # 12 hours ahead
            >>> plt.show()

            >>> # With test data
            >>> fig = forecaster.plot(lead_time=1.0, df_test=df_test)  # 1 hour ahead
            >>> plt.show()
        """
        if self.model_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() before plot().")

        if not hasattr(self, "_fit_df"):
            raise RuntimeError("Fitted data not available. Ensure fit() was called properly.")

        # Create or get figure
        if axs is None:
            fig = plt.figure(figsize=(14, 6))
            axs = plt.gca()
        else:
            fig = axs.figure

        # Get fitted values using clean API
        fitted_forecast = self.fitted()
        fitted_forecast["ds"] = pd.to_datetime(fitted_forecast["ds"])

        # Apply window filter to training data
        if window_days is not None:
            cutoff = (self.latest_timestamp - pd.Timedelta(days=window_days)).tz_localize(
                "America/Santiago"
            )
            fitted_forecast = fitted_forecast[fitted_forecast["ds"] >= cutoff]

        # Find the step number for the given lead_time
        step = _lead_time_to_step(lead_time, self.config.freq, self.config.n_forecast)

        # Get the yhat column name
        yhat_col = f"yhat{step}"
        if yhat_col not in fitted_forecast.columns:
            raise ValueError(
                f"Column '{yhat_col}' not found in forecast. "
                f"Maximum step available: {self.config.n_forecast}"
            )

        issue_time = fitted_forecast[["ds", "y"]].dropna()["ds"].max()

        # Plot fitted data
        axs.plot(
            fitted_forecast["ds"],
            fitted_forecast["y"],
            label="Training Actual",
            color="black",
            linewidth=2,
            marker="o",
            markersize=3,
            alpha=0.7,
        )

        mask_latest = fitted_forecast["ds"] <= issue_time

        # Plot fitted forecast for the lead_time
        axs.plot(
            fitted_forecast["ds"].loc[mask_latest],
            fitted_forecast[yhat_col].loc[mask_latest],
            label=f"Trained model for ({lead_time})",
            color="r",
            linewidth=2,
            linestyle="--",
            marker="s",
            markersize=3,
            alpha=0.7,
        )

        # Plot fitted forecast for the lead_time
        axs.plot(
            fitted_forecast["ds"].loc[~mask_latest],
            fitted_forecast[yhat_col].loc[~mask_latest],
            label=f"Fitted Forecast ({lead_time})",
            color="blue",
            linewidth=2,
            linestyle="--",
            marker="s",
            markersize=3,
            alpha=0.7,
        )

        # plot the issue time
        axs.axvline(issue_time, color="grey", ls="--")

        # If test data provided, overlay it
        if df_test is not None:
            # Plot test actual values
            axs.plot(
                df_test["ds"],
                df_test["y"],
                label="Test Actual",
                color="g",
                linewidth=2,
                marker="o",
                markersize=3,
                alpha=0.7,
            )

        # Format plot
        axs.set_xlabel("Local Time", fontsize=12, fontweight="bold")
        axs.set_ylabel(r"Temperature [$^\circ$ C]", fontsize=12, fontweight="bold")

        if title is None:
            if df_test is not None:
                title = f"NeuralProphet Forecast: Training vs Test ({lead_time} lead time)"
            else:
                title = f"NeuralProphet Fitted Forecast ({lead_time} lead time)"

        axs.set_title(title, fontsize=14, fontweight="bold", pad=20)

        axs.legend(loc="best", fontsize=10, framealpha=0.95)
        axs.grid(True, alpha=0.3)
        plt.tight_layout()

        return fig


def prepare_df(
    df: pd.DataFrame,
    regressor_cols: list[str],
    config,
    decomposer=None,
) -> pd.DataFrame:
    """Prepare input dataframe for NeuralProphet prediction.

    Common preprocessing for both fitted() and forecast() methods.
    Handles: validation, decomposition, time features, frequency regularization,
    missing value imputation.

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

    # Apply decomposition if decomposer is provided
    if decomposer is not None:
        df = decomposer.decompose(df)

    # Add time features if enabled
    if config.use_time_features:
        df = _add_time_features(df)

    # Prepare data - keep only relevant columns
    cols_to_keep = ["ds", "y"] + regressor_cols
    df_model = df[[c for c in cols_to_keep if c in df.columns]].copy()

    # Ensure regular frequency
    df_model = prepare_regular_frequency(df_model, freq=config.freq)

    # Handle missing values
    if config.impute_missing:
        df_model = df_model.interpolate(method="linear")

    return df_model


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclic time features (hour of day, day of year).

    Uses sin/cos encoding to preserve circular nature of time.

    Args:
        df: DataFrame with 'ds' column

    Returns:
        DataFrame with added time feature columns
    """
    df = df.copy()

    # Hour of day (0-23) -> cyclic encoding
    hour = df["ds"].dt.hour + df["ds"].dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    # Day of year (1-366) -> cyclic encoding
    doy = df["ds"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    return df


def extend_with_nans(
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
