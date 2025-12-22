"""Prophet-based forecaster implementation for Rubin's Oracle.

This module implements the ProphetForecaster class using Facebook Prophet
for time series forecasting.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_from_json, model_to_json

from rubin_oracle.base import ValidationMixin
from rubin_oracle.config import ProphetConfig
from rubin_oracle.utils import FrequencyConverter, OutputFormatter, validate_input


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
        self.metrics_: dict | None = None

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

        # Limit to lag_days window unless train_on_all_history is set
        # (train_on_all_history is used by EnsembleForecaster for components)
        if not getattr(self.config, "train_on_all_history", False):
            backward = pd.Timedelta(f"{self.config.lag_days} days") + pd.Timedelta("1hour")
            df = df[df["ds"] >= (df["ds"].max() - backward)]

        # Apply date filtering if specified (Prophet doesn't use decomposition)
        # if self.config.train_start_date is not None:
        #     df = df[df['ds'] >= pd.to_datetime(self.config.train_start_date)]
        # if self.config.train_end_date is not None:
        #     df = df[df['ds'] <= pd.to_datetime(self.config.train_end_date)]

        # Prophet doesn't support timezone-aware timestamps - remove timezone
        df = df.copy()
        if df["ds"].dt.tz is not None:
            df["ds"] = df["ds"].dt.tz_localize(None)

        # Store training data for predict() when df is not provided
        self._fit_df = df.copy()

        # Initialize Prophet with config parameters
        self.model_ = Prophet(
            growth=self.config.growth,
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

        # Add custom seasonalities if defined
        if self.config.custom_seasonalities:
            for season in self.config.custom_seasonalities:
                self.model_.add_seasonality(
                    name=season["name"],
                    period=season["period"],
                    fourier_order=season.get("fourier_order", 3),
                    mode=self.config.seasonality_mode,
                )

        # Fit the model
        self.model_.fit(df)

        # Compute in-sample metrics
        self._compute_metrics(df)

        return self

    def _compute_metrics(self, df: pd.DataFrame) -> None:
        """Compute in-sample fit metrics.

        Args:
            df: Training data with 'ds' and 'y' columns
        """
        if self.model_ is None:
            return

        # Prophet doesn't support timezone-aware timestamps - remove timezone if present
        df = df.copy()
        if df["ds"].dt.tz is not None:
            df["ds"] = df["ds"].dt.tz_localize(None)

        # Get in-sample predictions (fitted values)
        fitted = self.model_.predict(df)
        y_true = df["y"].values
        y_pred = fitted["yhat"].values

        # Compute metrics
        residuals = y_true - y_pred
        n = len(y_true)

        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))

        # R² (coefficient of determination)
        ss_res: float = float(np.sum(residuals**2))
        ss_tot: float = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        self.metrics_ = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "n_samples": n,
        }

    def compute_metrics(self, df: pd.DataFrame) -> dict | None:
        """Compute and return forecast metrics for the given data.

        Args:
            df: Data with 'ds' and 'y' columns

        Returns:
            Dictionary with metrics (rmse, mae, r2, n_samples) or None if computation fails
        """
        self._compute_metrics(df)
        return self.metrics_

    def fitted(self) -> pd.DataFrame:
        """Get in-sample fitted values on training data.

        Returns:
            DataFrame with columns:
                - ds: Timestamps from training data
                - y: Actual values
                - yhat: Fitted values
                - yhat_lower: Lower uncertainty bound
                - yhat_upper: Upper uncertainty bound

        Raises:
            RuntimeError: If model hasn't been fitted yet
        """
        if self.model_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        if self._fit_df is None:
            raise RuntimeError("No training data available.")

        # Prophet requires timezone-naive timestamps
        df = self._fit_df.copy()
        if df["ds"].dt.tz is not None:
            df["ds"] = df["ds"].dt.tz_localize(None)

        # Get fitted values
        forecast = self.model_.predict(df)
        result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()

        # Add actual values
        result = result.merge(df[["ds", "y"]], on="ds", how="left")

        return result

    def forecast(self, periods: int | None = None) -> pd.DataFrame:
        """Generate future forecasts starting from end of training data.

        Args:
            periods: Number of periods to forecast. If None, uses config.n_forecast.

        Returns:
            DataFrame with columns:
                - ds: Future timestamps
                - yhat: Point forecast
                - yhat_lower: Lower uncertainty bound
                - yhat_upper: Upper uncertainty bound

        Raises:
            RuntimeError: If model hasn't been fitted yet
        """
        if self.model_ is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        if periods is None:
            periods = self.config.n_forecast

        # Convert periods from days to samples based on frequency
        periods_samples = FrequencyConverter.days_to_samples(periods, self.config.freq)

        # Create future dataframe and predict
        future = self.model_.make_future_dataframe(periods=periods_samples, freq=self.config.freq)
        forecast = self.model_.predict(future)
        forecast = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

        # Return only future periods
        return forecast.tail(periods_samples).reset_index(drop=True)

    def predict(
        self,
        df: pd.DataFrame | None = None,
        periods: int | None = None,
    ) -> pd.DataFrame:
        """Generate forecasts (deprecated - use fitted() or forecast() instead).

        This method is kept for backwards compatibility.
        Use fitted() for in-sample predictions, forecast() for future predictions.
        """
        if periods == 0:
            return self.fitted()
        return self.forecast(periods=periods)

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
        return OutputFormatter.standardize_prophet_output(df, self.config.freq)

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
            Each row i has only yhat{i+1} filled (the forecast for that step).
        """
        # Refit the model on historical data
        self.fit(df_history)

        # Generate forecast - Prophet returns 96 rows with ds, yhat, yhat_lower, yhat_upper
        forecast = self.predict(periods=self.config.n_forecast)

        # Convert Prophet's row format to ValidationMixin's column format
        # Each row i should have ds=target_timestamp and yhat{i+1}=forecast_value
        n_steps = len(forecast)

        # Initialize result with ds column
        df_result = pd.DataFrame({"ds": forecast["ds"].values})

        # Initialize all yhat columns with NaN
        for step in range(1, n_steps + 1):
            df_result[f"yhat{step}"] = pd.NA
            df_result[f"yhat_lower{step}"] = pd.NA
            df_result[f"yhat_upper{step}"] = pd.NA

        # Fill in the diagonal - row i gets yhat{i+1}
        for i in range(n_steps):
            step = i + 1
            df_result.loc[i, f"yhat{step}"] = forecast.iloc[i]["yhat"]
            df_result.loc[i, f"yhat_lower{step}"] = forecast.iloc[i]["yhat_lower"]
            df_result.loc[i, f"yhat_upper{step}"] = forecast.iloc[i]["yhat_upper"]

        # Add timezone back for compatibility with validation merge
        if df_result["ds"].dt.tz is None:
            df_result["ds"] = df_result["ds"].dt.tz_localize("America/Santiago")

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
            raise RuntimeError("Model has not been fitted yet. Call fit() before save().")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save Prophet model
        model_path = path / "model.json"
        with open(model_path, "w") as f:
            json.dump(model_to_json(self.model_), f)

        # Save config
        config_path = path / "config.yaml"
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
        config_path = path / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        config = ProphetConfig.from_yaml(config_path)

        # Create instance
        forecaster = cls(config)

        # Load Prophet model
        model_path = path / "model.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path) as f:
            model_json = json.load(f)
            forecaster.model_ = model_from_json(model_json)

        return forecaster

    def plot(
        self,
        forecast_df: pd.DataFrame | None = None,
        df_test: pd.DataFrame | None = None,
        window_days: float | None = None,
        figsize: tuple[float, float] = (14, 6),
        show_residuals: bool = False,
        title: str | None = None,
        ax=None,
    ):
        """Plot training data, fitted model, and forecast.

        Args:
            forecast_df: Optional forecast DataFrame with 'ds' and 'yhat' columns.
                If None, generates forecast automatically.
            df_test: Optional test data to overlay actual values on forecast.
            window_days: Number of days to show. If None, shows all training data.
            figsize: Figure size as (width, height).
            show_residuals: If True, adds a second panel with residuals.
            title: Custom plot title. If None, uses model name.
            ax: Optional matplotlib axes to plot on. If None, creates new figure.

        Returns:
            matplotlib figure and axes (fig, ax) or (fig, (ax1, ax2)) if show_residuals.

        Raises:
            RuntimeError: If model hasn't been fitted yet.
        """
        if self.model_ is None or self._fit_df is None:
            raise RuntimeError("Model must be fitted before plotting.")

        try:
            import matplotlib.pyplot as plt
        except ImportError as err:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib"
            ) from err

        # Get fitted values using clean API
        fit_merged = self.fitted()
        fit_merged["ds"] = pd.to_datetime(fit_merged["ds"])
        fit_merged["residual"] = fit_merged["y"] - fit_merged["yhat"]

        # Apply window filter
        if window_days is not None:
            cutoff = fit_merged["ds"].max() - pd.Timedelta(days=window_days)
            fit_merged = fit_merged[fit_merged["ds"] >= cutoff]

        # Create figure
        if ax is None:
            if show_residuals:
                fig, (ax1, ax2) = plt.subplots(
                    2, 1, figsize=figsize, height_ratios=[3, 1], sharex=True
                )
            else:
                fig, ax1 = plt.subplots(figsize=figsize)
                ax2 = None
        else:
            ax1 = ax
            ax2 = None
            fig = ax.get_figure()

        # Plot training data and fitted model
        ax1.plot(fit_merged["ds"], fit_merged["y"], "k-", lw=1, alpha=0.8, label="Data")
        ax1.plot(fit_merged["ds"], fit_merged["yhat"], "r-", lw=1, alpha=0.8, label="Fitted model")
        ax1.fill_between(
            fit_merged["ds"],
            fit_merged["yhat_lower"],
            fit_merged["yhat_upper"],
            color="red",
            alpha=0.1,
            label="Uncertainty",
        )

        # Generate forecast using clean API
        if forecast_df is None:
            forecast_df = self.forecast()

        forecast_df = forecast_df.copy()
        forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])

        # Mark forecast start
        forecast_start = forecast_df["ds"].iloc[0]
        ax1.axvline(forecast_start, color="gray", linestyle="--", alpha=0.5, label="Forecast Time")

        # Plot forecast
        ax1.plot(forecast_df["ds"], forecast_df["yhat"], "b-", lw=2, label="Forecast")
        if "yhat_lower" in forecast_df.columns and "yhat_upper" in forecast_df.columns:
            ax1.fill_between(
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
            ax1.plot(df_test_plot["ds"], df_test_plot["y"], "g-", lw=2, alpha=0.8, label="Actual")

        ax1.set_ylabel("Value")
        ax1.set_title(title or f"{self.name} - Fit and Forecast")
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)

        # Plot residuals if requested
        if show_residuals and ax2 is not None:
            ax2.plot(fit_merged["ds"], fit_merged["residual"], "g-", lw=0.5, alpha=0.8)
            ax2.axhline(0, color="k", linestyle="-", alpha=0.3)
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Residual")
            ax2.grid(True, alpha=0.3)

        if ax is None:
            plt.tight_layout()

        if show_residuals:
            return fig, (ax1, ax2)
        return fig, ax1
