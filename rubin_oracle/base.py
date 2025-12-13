"""Base protocol and interfaces for Rubin's Oracle forecasters.

This module defines the Forecaster protocol that all forecasting models must implement,
ensuring a consistent interface across different model types.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

from rubin_oracle.config import BaseForecasterConfig



# =============================================================================
# Retraining Strategies
# =============================================================================

class RetrainingStrategy(Protocol):
    """Protocol for different retraining strategies."""
    
    def should_retrain(self, forecast_time: pd.Timestamp) -> bool:
        """Determine if model should be retrained at this time."""
        ...


class NoRetraining:
    """Never retrain the model after initial training."""
    
    def should_retrain(self, forecast_time: pd.Timestamp) -> bool:
        return False


class MonthlyRetraining:
    """Retrain model once per month."""
    
    def __init__(self):
        self.last_retrain_month = None
    
    def should_retrain(self, forecast_time: pd.Timestamp) -> bool:
        current_month = (forecast_time.year, forecast_time.month)
        if self.last_retrain_month is None or current_month != self.last_retrain_month:
            self.last_retrain_month = current_month
            return True
        return False


class DailyRetraining:
    """Retrain model once per day."""

    def __init__(self):
        self.last_retrain_date = None

    def should_retrain(self, forecast_time: pd.Timestamp) -> bool:
        current_date = forecast_time.date()
        if self.last_retrain_date is None or current_date != self.last_retrain_date:
            self.last_retrain_date = current_date
            return True
        return False


# =============================================================================
# Validation Mixin
# =============================================================================

class ValidationMixin:
    """Mixin providing walk-forward validation for forecasters.

    This mixin adds a `validate()` method that performs walk-forward validation
    by iterating through forecast times, optionally retraining the model, and
    collecting forecasts with their corresponding actuals.

    Requirements for the implementing class:
        - config.n_forecast: int - number of forecast steps
        - config.freq: str - pandas frequency string (e.g., 'h', '30min')
        - fit(df: pd.DataFrame) -> Self - fit the model on training data
        - _fit_and_predict(df_history: pd.DataFrame, forecast_time: pd.Timestamp) -> pd.DataFrame
            Method that fits on history and returns forecast DataFrame

    Example:
        >>> class MyForecaster(ValidationMixin):
        ...     def fit(self, df): ...
        ...     def _fit_and_predict(self, df_history, forecast_time): ...
        ...
        >>> forecaster = MyForecaster(config)
        >>> results = forecaster.validate(
        ...     df=full_data,
        ...     forecast_times=monthly_dates,
        ...     retrain_strategy=MonthlyRetraining()
        ... )
    """

    def validate(
        self,
        df: pd.DataFrame,
        forecast_times: list[pd.Timestamp],
        retrain_strategy: RetrainingStrategy | None = None,
        save_forecasts: bool = True,
        output_path: Path | None = None,
        verbose: bool = True,
        force_retrain: bool = False,
    ) -> pd.DataFrame:
        """Perform walk-forward validation with configurable retraining.

        Args:
            df: Full dataset containing both training and validation data.
                Must have 'ds' (datetime) and 'y' (target) columns.
            forecast_times: List of timestamps at which to issue forecasts.
                Each forecast uses only data up to that time.
            retrain_strategy: Strategy determining when to retrain the model.
                If None, uses NoRetraining (train once at first forecast).
            save_forecasts: Whether to save individual forecasts to disk.
            output_path: Directory to save forecasts if save_forecasts is True.
            verbose: Whether to print progress messages.
            force_retrain: If True, ignore cached models and always retrain.

        Returns:
            DataFrame containing all forecasts with columns:
                - ds: Target timestamp (what the forecast is FOR)
                - forecast_time: When the forecast was issued
                - y: Actual value (merged from data)
                - yhat1, yhat2, ...: Forecast at each step
                - res1, res2, ...: Residual (y - yhat) at each step

        Raises:
            RuntimeError: If _fit_and_predict is not implemented
            ValueError: If df is missing required columns
        """
        if verbose:
            print("=" * 70)
            print(f"Validation: Walk-Forward with {self.__class__.__name__}")
            print("=" * 70)
            print(f"Forecast times: {len(forecast_times)}")
            print(f"Forecast horizon: {self.config.n_forecast} steps")
            strategy_name = retrain_strategy.__class__.__name__ if retrain_strategy else 'NoRetraining'
            print(f"Retrain strategy: {strategy_name}")
            print(f"Force retrain: {force_retrain}")
            print()

        if retrain_strategy is None:
            retrain_strategy = NoRetraining()

        # Ensure df has required columns
        if 'ds' not in df.columns or 'y' not in df.columns:
            raise ValueError("DataFrame must contain 'ds' and 'y' columns")

        # Ensure ds is datetime with America/Santiago timezone
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df['ds']):
            df['ds'] = pd.to_datetime(df['ds'])
        # Ensure timezone is America/Santiago
        if df['ds'].dt.tz is None:
            # Naive datetime - assume it's already in Santiago local time
            # Handle DST ambiguous times by inferring from surrounding timestamps
            df['ds'] = df['ds'].dt.tz_localize('America/Santiago', ambiguous='infer', nonexistent='shift_forward')
        else:
            df['ds'] = df['ds'].dt.tz_convert('America/Santiago')

        all_forecasts = []
        model_fitted = False

        # Get model directory from config if available
        model_dir = getattr(self.config, 'model_dir', None)
        if model_dir:
            model_dir = Path(model_dir)
            model_dir.mkdir(parents=True, exist_ok=True)

        for i, forecast_time in enumerate(forecast_times, 1):
            if verbose:
                print(f"[{i}/{len(forecast_times)}] Forecast at {forecast_time}")

            # Check if we need to retrain
            needs_retrain = retrain_strategy.should_retrain(forecast_time)

            # make sure forecast_time is timezone-aware in America/Santiago
            if forecast_time.tzinfo is None:
                forecast_time = forecast_time.tz_localize('America/Santiago')
            else:
                forecast_time = forecast_time.tz_convert('America/Santiago')
                
            if needs_retrain or not model_fitted:
                # Generate model filename based on training date
                train_date = forecast_time.strftime('%Y-%m-%d')
                model_path = model_dir / f"model_{train_date}" if model_dir else None

                # Check if saved model exists for this date (skip if force_retrain)
                if model_path and model_path.exists() and not force_retrain:
                    if verbose:
                        print(f"  Loading cached model from {model_path}")
                    try:
                        loaded = self.__class__.load(model_path)
                        self.model_ = loaded.model_
                        self._decomposer = getattr(loaded, '_decomposer', None)
                        self._regressor_cols = getattr(loaded, '_regressor_cols', [])
                        model_fitted = True
                        continue
                    except Exception as e:
                        if verbose:
                            print(f"  Failed to load model: {e}, retraining...")

                if verbose:
                    print("  Retraining model...")
                # Fit on all data up to forecast_time
                df_train = df[df['ds'] <= forecast_time].copy()
                self.fit(df_train)
                model_fitted = True

                # Save model for future use
                if model_path:
                    if verbose:
                        print(f"  Saving model to {model_path}")
                    try:
                        self.save(model_path)
                    except Exception as e:
                        if verbose:
                            print(f"  Failed to save model: {e}")

            # Issue forecast
            try:
                # Get history up to forecast_time
                df_history = df[df['ds'] <= forecast_time].copy()

                # Call the forecaster's fit_and_predict method
                forecast = self._fit_and_predict(df_history, forecast_time)

                if verbose:
                    print(f"  Forecast range: {forecast['ds'].min()} to {forecast['ds'].max()}")

                # Merge forecast with actuals to get y values
                if 'y' in forecast.columns:
                    forecast = forecast.drop(columns=['y'])
                actuals = df[['ds', 'y']].drop_duplicates(subset='ds', keep='first')
                forecast = forecast.merge(actuals, on='ds', how='left')

                # Compute residuals: non-NaN yhat{step} values are aligned with y
                for step in range(1, self.config.n_forecast + 1):
                    yhat_col = f'yhat{step}'
                    res_col = f'res{step}'
                    if yhat_col in forecast.columns:
                        forecast[res_col] = forecast['y'] - forecast[yhat_col]

                forecast['forecast_time'] = forecast_time
                all_forecasts.append(forecast)

                if verbose:
                    n_valid = sum(1 for c in forecast.columns if c.startswith('yhat') and forecast[c].notna().any())
                    print(f"  Forecast complete ({n_valid} valid steps)")

            except Exception as e:
                if verbose:
                    print(f"  Forecast failed: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Concatenate all forecasts
        if not all_forecasts:
            print("Warning: No successful forecasts generated")
            return pd.DataFrame()

        df_results = pd.concat(all_forecasts, ignore_index=True)

        # Save raw forecasts if requested
        if save_forecasts and output_path is not None:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

            forecast_path = output_path / 'validation_forecasts.csv'
            df_results.to_csv(forecast_path, index=False)
            if verbose:
                print(f"\nSaved forecasts to: {forecast_path}")

        # Print summary statistics
        if verbose and len(df_results) > 0:
            print("\n" + "=" * 70)
            print("Validation Summary")
            print("=" * 70)
            print(f"Total forecast attempts: {len(forecast_times)}")
            print(f"Successful forecasts: {len(all_forecasts)}")
            print(f"Total forecast rows: {len(df_results)}")

            # Compute metrics by step
            print(f"\nMetrics by Lead Time:")
            freq_hours = pd.to_timedelta(self.config.freq).total_seconds() / 3600
            for step in range(1, min(self.config.n_forecast + 1, 11)):
                res_col = f'res{step}'
                if res_col in df_results.columns:
                    subset = df_results[res_col].dropna()
                    if len(subset) > 0:
                        mae = subset.abs().mean()
                        rmse = (subset ** 2).mean() ** 0.5
                        lead_hours = step * freq_hours
                        print(f"  Step {step:2d} ({lead_hours:5.1f}h): MAE={mae:6.4f}, RMSE={rmse:6.4f}")

        return df_results

    def _fit_and_predict(
        self,
        df_history: pd.DataFrame,
        forecast_time: pd.Timestamp,
    ) -> pd.DataFrame:
        """Fit model on history and generate forecast.

        This method must be implemented by forecaster classes using this mixin.
        It should:
        1. Optionally refit the model on df_history (or use cached model)
        2. Generate a forecast starting from forecast_time
        3. Return a DataFrame with 'ds' and 'yhat1', 'yhat2', ... columns

        Args:
            df_history: Historical data up to and including forecast_time
            forecast_time: The time at which the forecast is issued

        Returns:
            DataFrame with forecast results

        Raises:
            NotImplementedError: If not overridden by implementing class
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _fit_and_predict() "
            "to use ValidationMixin"
        )


# =============================================================================
# Forecaster Protocol
# =============================================================================

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
    
    def validate(
        self,
        df: pd.DataFrame,
        forecast_times: list[pd.Timestamp],
        retrain_strategy: RetrainingStrategy | None = None,
        save_forecasts: bool = True,
        output_path: Path | None = None,
        verbose: bool = True,
        force_retrain: bool = False,
    ) -> pd.DataFrame:
        """Validate forecaster with Rubin-style operational schedule.

        Args:
            df: Full dataset for validation
            forecast_times: List of timestamps to generate forecasts for
            retrain_strategy: Strategy for retraining the model during validation
            save_forecasts: Whether to save individual forecasts to disk
            output_path: Directory to save forecasts if save_forecasts is True
            verbose: Whether to print progress messages
            force_retrain: If True, ignore cached models and always retrain

        Returns:
            DataFrame containing all forecasts generated during validation
        """