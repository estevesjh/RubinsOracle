"""Ensemble forecaster combining multiple Prophet/NeuralProphet models.

This module implements the EnsembleForecaster class which operates on decomposed
frequency bands, training specialized models for each band and combining their
forecasts with optional bias correction.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from ..base import ValidationMixin
from ..config import (
    ComponentConfig,
    EnsembleConfig,
)
from ..utils import FrequencyConverter, MetricsCalculator

if TYPE_CHECKING:
    from ..base import Forecaster


class EnsembleForecaster(ValidationMixin):
    """Ensemble forecaster combining multiple Prophet/NeuralProphet models.

    Each component model operates on a subset of decomposed frequency bands,
    optionally at different resolutions. Forecasts are combined and optionally
    post-processed with bias correction or ETS blending.

    Example:
        >>> config = EnsembleConfig.from_yaml("configs/ensemble_dual_prophet.yaml")
        >>> model = EnsembleForecaster(config)
        >>> model.fit(df_train)
        >>> forecast = model.predict(df_history, periods=96)
    """

    def __init__(self, config: EnsembleConfig):
        """Initialize EnsembleForecaster.

        Args:
            config: Ensemble configuration
        """
        self.config = config
        self.name = config.name
        self._decomposer = None
        self._components: list[tuple[ComponentConfig, Forecaster]] = []
        self._df_decomposed_cache: pd.DataFrame | None = None
        self._is_fitted = False
        self.metrics_: dict | None = None

        # Compute frequency constants
        self._freq_per_hour = self._parse_freq_per_hour(config.output_freq)
        self._steps_per_day_base = self._freq_per_hour * 24

    @property
    def components(self) -> dict[str, Forecaster]:
        """Get component models as a dictionary by name.

        Example:
            >>> model.components['high_freq']  # Access high_freq model
            >>> model.components['low_freq']   # Access low_freq model
        """
        return {config.name: forecaster for config, forecaster in self._components}

    def get_component_names(self) -> list[str]:
        """Get list of component model names.

        Returns:
            List of component names (e.g., ['high_freq', 'low_freq'])
        """
        return [config.name for config, _ in self._components]

    def __getattr__(self, name: str):
        """Allow access to component models via attribute (e.g., model.high_freq_model)."""
        if name.endswith("_model"):
            component_name = name[:-6]  # Remove '_model' suffix
            for config, forecaster in self._components:
                if config.name == component_name:
                    return forecaster
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def _parse_freq_per_hour(self, freq: str) -> float:
        """Parse frequency string to steps per hour."""
        return FrequencyConverter.freq_to_samples_per_hour(freq)

    def _create_decomposer(self):
        """Create decomposer based on config method."""
        method = self.config.decomposer.method

        if method == "none":
            raise ValueError("EnsembleForecaster requires decomposition (method != 'none')")

        # Get decomposer parameters, excluding method-specific ones
        if method == "bandpass":
            from ..preprocessing import BandpassDecomposer

            # Exclude VMD-specific parameters
            exclude_keys = {"method", "alpha", "K_stage1", "K_stage2"}
            params = {
                k: v
                for k, v in self.config.decomposer.model_dump().items()
                if k not in exclude_keys
            }
            # Convert freq from string (e.g., "15min") to samples per day
            if isinstance(params["freq"], str):
                freq_per_hour = self._parse_freq_per_hour(params["freq"])
                params["freq"] = int(freq_per_hour * 24)
            return BandpassDecomposer(**params)
        elif method == "vmd":
            from ..preprocessing import RubinVMDDecomposer

            # Exclude bandpass-specific parameters
            exclude_keys = {
                "method",
                "period_pairs",
                "filter_type",
                "edge_method",
                "edge_pad_periods",
                "nan_fill",
                "nan_fill_period",
                "use_edge_weighting",
                "butter_order",
                "savgol_polyorder",
                "savgol_butter_cleanup",
            }
            params = {
                k: v
                for k, v in self.config.decomposer.model_dump().items()
                if k not in exclude_keys
            }
            return RubinVMDDecomposer(**params)
        else:
            raise ValueError(f"Unknown decomposer method: {method}")

    def _validate_decomposition(self, df: pd.DataFrame, df_decomposed: pd.DataFrame) -> dict:
        """Verify decomposition and report signal recovery metrics.

        Args:
            df: Original dataframe with 'y' column
            df_decomposed: Decomposed dataframe with band columns

        Returns:
            Dict with 'mean_bias' and 'variance_recovered'
        """
        # Detect band columns (works with bandpass, VMD, etc.)
        band_cols = [
            c for c in df_decomposed.columns if c.startswith(("y_band_", "y_imf_", "y_mode_"))
        ]

        if not band_cols:
            warnings.warn("No band columns found in decomposed data", stacklevel=2)
            return {"mean_bias": np.nan, "variance_recovered": np.nan}

        reconstructed = df_decomposed[band_cols].sum(axis=1)
        original = df["y"].values

        residual = original - reconstructed.values

        # Compute metrics (NaN-safe)
        original_var = np.nanvar(original)
        residual_var = np.nanvar(residual)
        recovered_var_ratio = 1 - (residual_var / original_var) if original_var > 0 else 1.0
        mean_bias = np.nanmean(residual)
        max_abs_residual: float = float(np.nanmax(np.abs(residual)))

        # Print signal recovery report
        print("=" * 60)
        print("DECOMPOSITION SIGNAL RECOVERY")
        print("=" * 60)
        print(f"Original signal variance:    {original_var:.4f}")
        print(f"Residual variance:           {residual_var:.6f}")
        print(f"Variance recovered:          {recovered_var_ratio*100:.2f}%")
        print(f"Mean bias:                   {mean_bias:.4f}°C")
        print(f"Max absolute residual:       {max_abs_residual:.4f}°C")
        print("-" * 60)
        for col in band_cols:
            band_var = df_decomposed[col].var(skipna=True)
            pct = (
                band_var / original_var * 100
                if original_var > 0 and not np.isnan(original_var)
                else 0
            )
            print(f"  {col}: variance={band_var:.4f} ({pct:.1f}%)")
        print("=" * 60)

        if np.abs(mean_bias) > 0.01:
            warnings.warn(f"Decomposition bias detected: {mean_bias:.4f}°C", stacklevel=2)
        if recovered_var_ratio < 0.99:
            warnings.warn(f"Low variance recovery: {recovered_var_ratio*100:.1f}%", stacklevel=2)

        return {"mean_bias": mean_bias, "variance_recovered": recovered_var_ratio}

    def _get_band_columns(self, df_decomposed: pd.DataFrame) -> list[str]:
        """Get list of band column names."""
        return [c for c in df_decomposed.columns if c.startswith(("y_band_", "y_imf_", "y_mode_"))]

    def _create_model(self, comp_config: ComponentConfig) -> Forecaster:
        """Create a model instance for a component.

        Args:
            comp_config: Component configuration

        Returns:
            Forecaster instance (ProphetForecaster or NeuralProphetForecaster)
        """
        if comp_config.model_type == "prophet":
            from ..config import ProphetConfig
            from .prophet import ProphetForecaster

            # Build custom seasonalities from component config
            # Prophet expects period in days (not samples)
            custom_seasonalities = None
            if comp_config.seasonalities:
                custom_seasonalities = [
                    {"name": s.name, "period": s.period, "fourier_order": s.fourier_order}
                    for s in comp_config.seasonalities
                ]

            # Determine frequency based on resolution/downsampling
            freq = comp_config.downsample_to or comp_config.resolution

            # Use component's lag_days if specified, otherwise use ensemble's lag_days_max
            lag_days = (
                comp_config.lag_days
                if comp_config.lag_days is not None
                else self.config.lag_days_max
            )

            config = ProphetConfig(
                lag_days=lag_days,
                n_forecast=self.config.n_forecast,
                freq=freq,
                yearly_seasonality=False,
                weekly_seasonality=False,
                daily_seasonality=False,
                n_changepoints=comp_config.n_changepoints,
                changepoint_prior_scale=comp_config.changepoint_prior_scale,
                growth=comp_config.growth,
                custom_seasonalities=custom_seasonalities,
                train_on_all_history=True,  # Train on full history for ensemble components
            )
            return ProphetForecaster(config)

        elif comp_config.model_type == "neural_prophet":
            from ..config import NeuralProphetConfig
            from .neural_prophet import NeuralProphetForecaster

            # Build custom seasonalities from component config
            custom_seasonalities = None
            if comp_config.seasonalities:
                # Determine frequency and convert period from days to samples
                freq = comp_config.downsample_to or comp_config.resolution
                freq_per_hour = self._parse_freq_per_hour(freq)
                freq_per_day = freq_per_hour * 24

                custom_seasonalities = [
                    {
                        "name": s.name,
                        "period": int(s.period * freq_per_day),  # Convert days to samples
                        "fourier_order": s.fourier_order,
                    }
                    for s in comp_config.seasonalities
                ]
            else:
                freq = comp_config.downsample_to or comp_config.resolution

            # Use component's lag_days if specified, otherwise use ensemble's lag_days_max
            lag_days = (
                comp_config.lag_days
                if comp_config.lag_days is not None
                else self.config.lag_days_max
            )

            # All components forecast the same time span (n_forecast days)
            # Each converts internally to samples based on its resolution
            config = NeuralProphetConfig(
                lag_days=lag_days,
                n_forecast=self.config.n_forecast,
                freq=freq,
                yearly_seasonality=False,
                weekly_seasonality=False,
                daily_seasonality=False,
                ar_layers=comp_config.ar_layers,
                ar_reg=comp_config.ar_reg,
                epochs=comp_config.epochs,
                learning_rate=comp_config.learning_rate,
                n_changepoints=comp_config.n_changepoints,
                custom_seasonalities=custom_seasonalities,
                # Disable early stopping and validation for ensemble components
                early_stopping=False,
                valid_pct=0.0,
                train_on_all_history=True,  # Train on full history for ensemble components
            )
            return NeuralProphetForecaster(config)

        else:
            raise ValueError(f"Unknown model type: {comp_config.model_type}")

    def _downsample(self, df: pd.DataFrame, target_freq: str) -> pd.DataFrame:
        """Downsample dataframe to target frequency.

        Args:
            df: DataFrame with 'ds' and 'y' columns
            target_freq: Target frequency (e.g., '4h')

        Returns:
            Downsampled DataFrame
        """
        df_copy = df.copy()
        df_copy = df_copy.set_index("ds")
        df_resampled = df_copy[["y"]].resample(target_freq).mean()
        return df_resampled.reset_index()

    def _upsample_forecast(
        self, yhat: np.ndarray, from_freq: str, to_freq: str, n_periods: int
    ) -> np.ndarray:
        """Upsample forecast from coarse to fine resolution.

        Args:
            yhat: Forecast values at coarse resolution
            from_freq: Source frequency (e.g., '4h')
            to_freq: Target frequency (e.g., '15min')
            n_periods: Number of output periods

        Returns:
            Upsampled forecast array
        """
        from_per_hour = self._parse_freq_per_hour(from_freq)
        to_per_hour = self._parse_freq_per_hour(to_freq)

        ratio = to_per_hour / from_per_hour

        # Create interpolation
        x_from = np.arange(len(yhat)) * ratio
        x_to = np.arange(n_periods)

        f = interp1d(x_from, yhat, kind="linear", fill_value="extrapolate")
        return f(x_to)

    def _steps_per_day(self, comp_config: ComponentConfig) -> int:
        """Get steps per day for a component."""
        freq = comp_config.downsample_to or comp_config.resolution
        return int(self._parse_freq_per_hour(freq) * 24)

    def _get_component_data(
        self, df: pd.DataFrame, df_decomposed: pd.DataFrame, comp_config: ComponentConfig
    ) -> pd.DataFrame:
        """Extract data for a component from decomposed signal.

        Args:
            df: Original dataframe
            df_decomposed: Decomposed dataframe
            comp_config: Component configuration

        Returns:
            DataFrame with 'ds' and 'y' for the component
        """
        band_cols = self._get_band_columns(df_decomposed)

        # Sum assigned bands
        selected_cols = []
        for idx in comp_config.band_indices:
            # Find column matching index
            for col in band_cols:
                if f"_{idx}" in col:
                    selected_cols.append(col)
                    break

        if not selected_cols:
            raise ValueError(f"No bands found for indices {comp_config.band_indices}")

        df_comp = df[["ds"]].copy()
        df_comp["y"] = df_decomposed[selected_cols].sum(axis=1)

        # Remove timezone (Prophet doesn't support timezone-aware timestamps)
        if df_comp["ds"].dt.tz is not None:
            df_comp["ds"] = df_comp["ds"].dt.tz_localize(None)

        # Downsample if needed
        if comp_config.downsample_to:
            df_comp = self._downsample(df_comp, comp_config.downsample_to)

        return df_comp

    def fit(self, df: pd.DataFrame) -> EnsembleForecaster:
        """Fit all component models.

        1. Decompose signal into frequency bands
        2. Validate decomposition (check variance recovery)
        3. For each component:
           a. Sum assigned bands
           b. Optionally downsample
           c. Fit component model

        Args:
            df: Training data with 'ds' and 'y' columns

        Returns:
            self
        """
        # Validate input
        if "ds" not in df.columns or "y" not in df.columns:
            raise ValueError("DataFrame must have 'ds' and 'y' columns")

        # Initialize decomposer
        self._decomposer = self._create_decomposer()

        # Decompose full signal
        print(f"\nFitting EnsembleForecaster with {len(self.config.components)} components...")
        df_decomposed = self._decomposer.decompose(df)
        self._df_decomposed_cache = df_decomposed

        # Validate decomposition
        self._validate_decomposition(df, df_decomposed)

        # Clear previous components
        self._components = []

        # Fit each component
        for i, comp_config in enumerate(self.config.components):
            print(f"\n[{i+1}/{len(self.config.components)}] Fitting component: {comp_config.name}")
            print(f"  Model type: {comp_config.model_type}")
            print(f"  Bands: {comp_config.band_indices}")
            print(f"  Lag days: {comp_config.lag_days}")
            if comp_config.downsample_to:
                print(f"  Downsampled to: {comp_config.downsample_to}")

            # Get component data
            df_comp = self._get_component_data(df, df_decomposed, comp_config)
            print(f"  Training samples: {len(df_comp)}")

            # Create and fit model
            model = self._create_model(comp_config)
            model.fit(df_comp)

            self._components.append((comp_config, model))

        # Compute ensemble metrics
        self._compute_metrics(df, df_decomposed)

        self._is_fitted = True
        print("\nEnsembleForecaster fitted successfully.")
        return self

    def _compute_metrics(self, df: pd.DataFrame, df_decomposed: pd.DataFrame) -> None:
        """Compute ensemble fit metrics.

        Aggregates metrics from component models and computes overall
        in-sample reconstruction error.

        Args:
            df: Original training data
            df_decomposed: Decomposed training data
        """
        # Collect component metrics
        component_metrics = {}
        for comp_config, model in self._components:
            if hasattr(model, "metrics_") and model.metrics_ is not None:
                component_metrics[comp_config.name] = model.metrics_

        # Compute overall ensemble in-sample metrics
        # Get combined in-sample predictions
        y_true = df["y"].values.astype(float)
        y_pred = np.full_like(y_true, np.nan, dtype=float)

        component_preds = []
        for comp_config, model in self._components:
            # Get component data
            df_comp = self._get_component_data(df, df_decomposed, comp_config)

            # Get in-sample predictions
            try:
                if hasattr(model, "model_") and model.model_ is not None:
                    # Prophet: get fitted values
                    if comp_config.model_type == "prophet":
                        fitted = model.model_.predict(model.model_.history)
                        comp_yhat = fitted["yhat"].values.astype(float)
                    else:
                        # NeuralProphet - use stored _fit_df for predictions
                        if hasattr(model, "_fit_df") and model._fit_df is not None:
                            fc = model.predict(model._fit_df, periods=0)
                        else:
                            fc = model.predict(df_comp, periods=0)
                        # Get yhat1 (1-step ahead fitted values)
                        if "yhat1" in fc.columns:
                            comp_yhat = fc["yhat1"].values.astype(float)
                        elif "yhat" in fc.columns:
                            comp_yhat = fc["yhat"].values.astype(float)
                        else:
                            yhat_cols = [
                                c for c in fc.columns if c.startswith("yhat") and c[4:].isdigit()
                            ]
                            comp_yhat = (
                                fc[yhat_cols[0]].values.astype(float)
                                if yhat_cols
                                else df_comp["y"].values.astype(float)
                            )
                else:
                    comp_yhat = df_comp["y"].values.astype(float)
            except Exception as e:
                warnings.warn(
                    f"Could not get predictions for {comp_config.name}: {e}", stacklevel=2
                )
                comp_yhat = df_comp["y"].values.astype(float)

            # Upsample if needed (keep NaN - don't replace with 0)
            if comp_config.downsample_to:
                comp_yhat = self._upsample_forecast(
                    comp_yhat, comp_config.downsample_to, self.config.output_freq, len(y_true)
                )

            # Align to y_true length
            aligned = np.full(len(y_true), np.nan)
            min_len = min(len(comp_yhat), len(y_true))
            aligned[-min_len:] = comp_yhat[-min_len:]
            component_preds.append(aligned)

        # Sum components only where ALL have valid values
        component_preds = np.array(component_preds)
        all_valid_mask = ~np.any(np.isnan(component_preds), axis=0)
        y_pred[all_valid_mask] = np.sum(component_preds[:, all_valid_mask], axis=0)

        # Compute overall metrics (NaN-safe)
        # Filter out NaN values from both y_true and y_pred
        valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)

        # If no valid samples (common with NeuralProphet due to AR lags alignment),
        # report component metrics only
        if valid_mask.sum() == 0:
            self.metrics_ = {
                "rmse": float("nan"),
                "mae": float("nan"),
                "r2": float("nan"),
                "n_samples": 0,
                "components": component_metrics,
                "note": "Overall metrics unavailable due to AR lag alignment; see component metrics",
            }
            # Print metrics summary (component metrics only)
            print("\n" + "=" * 60)
            print("ENSEMBLE FIT METRICS")
            print("=" * 60)
            print("Overall metrics: N/A (component AR lags don't align)")
            print("-" * 60)
            for name, m in component_metrics.items():
                print(f"  {name}: RMSE={m['rmse']:.4f}, MAE={m['mae']:.4f}, R²={m['r2']:.4f}")
            print("=" * 60)
            return
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]

        if len(y_true_valid) == 0:
            self.metrics_ = {
                "rmse": float("nan"),
                "mae": float("nan"),
                "r2": float("nan"),
                "n_samples": 0,
                "components": component_metrics,
            }
            return

        # Compute metrics using helper
        metrics_dict = MetricsCalculator.compute_metrics(y_true_valid, y_pred_valid)
        self.metrics_ = {
            **metrics_dict,
            "components": component_metrics,
        }

        # Print metrics summary
        print("\n" + "=" * 60)
        print("ENSEMBLE FIT METRICS")
        print("=" * 60)
        print(f"Overall RMSE: {metrics_dict['rmse']:.4f}")
        print(f"Overall MAE:  {metrics_dict['mae']:.4f}")
        print(f"Overall R²:   {metrics_dict['r2']:.4f}")
        print("-" * 60)
        for name, metrics in component_metrics.items():
            print(
                f"  {name}: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, R²={metrics['r2']:.4f}"
            )
        print("=" * 60)

    def _combine_forecasts(self, forecasts: list[np.ndarray]) -> np.ndarray:
        """Combine component forecasts.

        Args:
            forecasts: List of forecast arrays (one per component)

        Returns:
            Combined forecast array
        """
        if self.config.combine_method == "sum":
            return np.sum(forecasts, axis=0)

        elif self.config.combine_method == "weighted":
            if self.config.component_weights is None:
                # Equal weights
                weights = np.ones(len(forecasts)) / len(forecasts)
            else:
                weights = np.array(self.config.component_weights)
                if len(weights) != len(forecasts):
                    raise ValueError("Number of weights must match number of components")

            return np.sum([w * f for w, f in zip(weights, forecasts)], axis=0)

        else:
            raise ValueError(f"Unknown combine method: {self.config.combine_method}")

    def _compute_robust_bias(
        self, y_history: np.ndarray, yhat_history: np.ndarray, window_hours: float = 6.0
    ) -> float:
        """Compute robust bias from recent forecast residuals.

        Args:
            y_history: Recent actual values
            yhat_history: Recent forecasts (aligned with y_history)
            window_hours: Lookback window for computing bias

        Returns:
            Bias estimate
        """
        window_steps = int(window_hours * self._freq_per_hour)
        window_steps = min(window_steps, len(y_history), len(yhat_history))

        if window_steps < 1:
            return 0.0

        residuals = y_history[-window_steps:] - yhat_history[-window_steps:]

        method = self.config.post_processor.bias_method
        if method == "median":
            return float(np.median(residuals))
        elif method == "mean":
            return float(np.mean(residuals))
        else:  # "last"
            return float(y_history[-1] - yhat_history[-1])

    def _apply_post_processing(
        self,
        yhat: np.ndarray,
        y_history: np.ndarray,
        yhat_history: np.ndarray | None = None,
        has_neural_prophet: bool = False,
    ) -> np.ndarray:
        """Apply post-processing to combined forecast.

        Args:
            yhat: Combined forecast
            y_history: Recent actual values
            yhat_history: Recent in-sample forecasts (optional)
            has_neural_prophet: Whether any component is NeuralProphet

        Returns:
            Post-processed forecast
        """
        pp = self.config.post_processor

        # Skip bias correction for NeuralProphet unless explicitly enabled
        if has_neural_prophet and not pp.bias_for_neural_prophet:
            apply_bias = False
        else:
            apply_bias = pp.bias_correction

        # Apply bias correction
        if apply_bias:
            if yhat_history is not None and len(yhat_history) > 0:
                bias = self._compute_robust_bias(y_history, yhat_history, pp.bias_window_hours)
            else:
                # Fallback: use last value difference
                bias = y_history[-1] - yhat[0]
            yhat = yhat + bias

        # Apply ETS blend
        if pp.ets_blend:
            yhat = self._apply_ets_blend(yhat, y_history, pp.ets_tau)

        return yhat

    def _apply_ets_blend(self, yhat: np.ndarray, y_history: np.ndarray, tau: int) -> np.ndarray:
        """Blend forecast with exponential smoothing.

        Args:
            yhat: Forecast to blend
            y_history: Recent history
            tau: Decay constant in steps

        Returns:
            Blended forecast
        """
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            n_history = min(96, len(y_history))
            y_recent = y_history[-n_history:]

            model = ExponentialSmoothing(
                y_recent, trend="add", seasonal=None, initialization_method="estimated"
            )
            fit = model.fit(optimized=True)
            yhat_ets = fit.forecast(len(yhat))

        except Exception:
            # Fallback: linear extrapolation
            n_fit = min(8, len(y_history))
            x = np.arange(n_fit)
            slope, intercept = np.polyfit(x, y_history[-n_fit:], 1)
            x_future = np.arange(n_fit, n_fit + len(yhat))
            yhat_ets = slope * x_future + intercept

        # Exponentially decaying blend
        alpha = np.exp(-np.arange(len(yhat)) / tau)
        return alpha * yhat_ets + (1 - alpha) * yhat

    def _get_fitted_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get in-sample fitted values for the ensemble.

        Combines fitted values from all components by:
        1. Getting in-sample predictions from each component model
        2. Upsampling downsampled components
        3. Summing all components

        Args:
            df: Historical data with 'ds' and 'y' columns (must match training data)

        Returns:
            DataFrame with 'ds' and 'yhat' columns
        """
        df = df.copy()
        df["ds"] = pd.to_datetime(df["ds"])

        # Store original timezone for later
        original_tz = df["ds"].dt.tz

        # Remove timezone for merging (Prophet strips timezone)
        if original_tz is not None:
            df["ds"] = df["ds"].dt.tz_localize(None)

        # Initialize combined yhat
        yhat_combined = np.zeros(len(df), dtype=float)

        for comp_config, model in self._components:
            if not hasattr(model, "_fit_df") or model._fit_df is None:
                continue

            # Get in-sample predictions from component
            if comp_config.model_type == "prophet":
                # Prophet: predict on training data
                fit_pred = model.model_.predict(model._fit_df)
                comp_yhat = fit_pred["yhat"].values
                comp_ds = pd.to_datetime(fit_pred["ds"])
            else:
                # NeuralProphet: use yhat1 (1-step ahead prediction)
                fit_pred = model.predict(model._fit_df, periods=0)
                if "yhat1" in fit_pred.columns:
                    comp_yhat = fit_pred["yhat1"].values
                elif "yhat" in fit_pred.columns:
                    comp_yhat = fit_pred["yhat"].values
                else:
                    yhat_cols = [c for c in fit_pred.columns if c.startswith("yhat")]
                    comp_yhat = (
                        fit_pred[yhat_cols[0]].values if yhat_cols else np.zeros(len(fit_pred))
                    )
                comp_ds = pd.to_datetime(fit_pred["ds"])

            # Ensure comp_ds is timezone-naive for merging
            if comp_ds.dt.tz is not None:
                comp_ds = comp_ds.dt.tz_localize(None)

            # Upsample if needed
            if comp_config.downsample_to:
                comp_yhat = self._upsample_forecast(
                    comp_yhat, comp_config.downsample_to, self.config.output_freq, len(df)
                )
                # Upsampled values align with the end of df
                min_len = min(len(comp_yhat), len(df))
                yhat_combined[-min_len:] += comp_yhat[-min_len:]
            else:
                # No downsampling - align by timestamp
                # Create a mapping from component timestamps to yhat values
                comp_df = pd.DataFrame({"ds": comp_ds, "comp_yhat": comp_yhat})
                merged = df[["ds"]].merge(comp_df, on="ds", how="left")
                merged["comp_yhat"] = merged["comp_yhat"].fillna(0)
                yhat_combined += merged["comp_yhat"].values

        # Restore original timezone
        result_ds = df["ds"]
        if original_tz is not None:
            result_ds = result_ds.dt.tz_localize(original_tz)

        return pd.DataFrame({"ds": result_ds, "yhat": yhat_combined})

    def predict(self, df: pd.DataFrame | None = None, periods: int | None = None) -> pd.DataFrame:
        """Generate ensemble forecast.

        1. Decompose input history (if provided)
        2. Generate forecasts from each component
        3. Upsample if needed
        4. Combine forecasts
        5. Apply post-processing

        Args:
            df: History dataframe (required for NeuralProphet components)
            periods: Number of periods to forecast

        Returns:
            DataFrame with 'ds', 'yhat', and step columns
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        periods = periods or self.config.n_forecast
        forecasts = []

        # Decompose history if provided
        df_decomposed = None
        if df is not None:
            df_decomposed = self._decomposer.decompose(df)

        # Check if any component is NeuralProphet
        has_neural_prophet = any(
            comp_config.model_type == "neural_prophet" for comp_config, _ in self._components
        )

        # Generate forecasts from each component
        for comp_config, model in self._components:
            # Get component-specific data
            if df_decomposed is not None:
                df_comp = self._get_component_data(df, df_decomposed, comp_config)
            else:
                df_comp = None

            # Calculate periods at component resolution
            if comp_config.downsample_to:
                from_per_hour = self._parse_freq_per_hour(comp_config.downsample_to)
                to_per_hour = self._freq_per_hour
                comp_periods = int(np.ceil(periods * from_per_hour / to_per_hour)) + 1
            else:
                comp_periods = periods

            # Generate forecast
            fc = model.predict(df_comp, periods=comp_periods)

            # Extract yhat values
            if "yhat" in fc.columns:
                # Prophet format: simple yhat column
                yhat = fc["yhat"].values
            else:
                # NeuralProphet multi-step format: yhat1, yhat2, ..., yhatN
                # For forecasting, extract multi-step values from the LAST valid row
                yhat_cols = sorted(
                    [c for c in fc.columns if c.startswith("yhat") and c[4:].isdigit()],
                    key=lambda x: int(x[4:]),
                )
                if not yhat_cols:
                    raise ValueError(f"No yhat column found in forecast from {comp_config.name}")

                # Find the last row with valid yhat1 (the forecast origin)
                valid_mask = fc["yhat1"].notna()
                if valid_mask.any():
                    last_valid_idx = valid_mask[valid_mask].index[-1]
                    # Extract yhat1, yhat2, ..., yhatN from this row as the multi-step forecast
                    yhat = np.array(
                        [fc.loc[last_valid_idx, col] for col in yhat_cols[:comp_periods]]
                    )
                else:
                    # Fallback: use last non-NaN values from each column
                    yhat = np.array(
                        [
                            fc[col].dropna().iloc[-1] if fc[col].notna().any() else np.nan
                            for col in yhat_cols[:comp_periods]
                        ]
                    )

            # Upsample if needed
            if comp_config.downsample_to:
                yhat = self._upsample_forecast(
                    yhat, comp_config.downsample_to, self.config.output_freq, periods
                )

            # Ensure correct length
            yhat = yhat[:periods]
            if len(yhat) < periods:
                # Pad with last value if needed
                yhat = np.pad(yhat, (0, periods - len(yhat)), mode="edge")

            forecasts.append(yhat)

        # Combine forecasts
        yhat_combined = self._combine_forecasts(forecasts)

        # Apply post-processing
        if df is not None:
            y_history = df["y"].values
            # TODO: Get in-sample forecasts for robust bias
            yhat_combined = self._apply_post_processing(
                yhat_combined, y_history, None, has_neural_prophet
            )

        # Build output DataFrame
        return self._build_output(df, yhat_combined, periods)

    def _build_output(
        self, df: pd.DataFrame | None, yhat: np.ndarray, periods: int
    ) -> pd.DataFrame:
        """Build output DataFrame from forecast.

        Args:
            df: Input history (for timestamps)
            yhat: Forecast values
            periods: Number of periods

        Returns:
            DataFrame with 'ds', 'yhat', 'step' columns
        """
        # Generate timestamps
        if df is not None and "ds" in df.columns:
            last_ds = df["ds"].iloc[-1]
            freq = self.config.output_freq
            ds = pd.date_range(start=last_ds, periods=periods + 1, freq=freq)[1:]
        else:
            ds = pd.RangeIndex(periods)

        result = pd.DataFrame(
            {
                "ds": ds,
                "yhat": yhat[:periods],
                "step": np.arange(1, periods + 1),
            }
        )

        return result

    def standardize_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize output format.

        Args:
            df: Prediction output

        Returns:
            Standardized DataFrame
        """
        return df

    def _fit_and_predict(
        self, df_history: pd.DataFrame, forecast_time: pd.Timestamp
    ) -> pd.DataFrame:
        """Fit and predict for validation.

        Args:
            df_history: History up to forecast_time
            forecast_time: Time of forecast

        Returns:
            Multi-step forecast DataFrame
        """
        # Refit is handled by ValidationMixin
        return self.predict(df_history, self.config.n_forecast)

    def save(self, path: str | Path) -> None:
        """Save ensemble model to directory.

        Args:
            path: Directory path for saving
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        self.config.to_yaml(path / "config.yaml")

        # Save each component model
        for i, (comp_config, model) in enumerate(self._components):
            comp_dir = path / f"component_{i}_{comp_config.name}"
            model.save(comp_dir)

    @classmethod
    def load(cls, path: str | Path) -> EnsembleForecaster:
        """Load ensemble model from directory.

        Args:
            path: Directory path to load from

        Returns:
            Loaded EnsembleForecaster
        """
        path = Path(path)

        # Load config
        config = EnsembleConfig.from_yaml(path / "config.yaml")
        instance = cls(config)

        # Load component models
        for i, comp_config in enumerate(config.components):
            comp_dir = path / f"component_{i}_{comp_config.name}"

            if comp_config.model_type == "prophet":
                from .prophet import ProphetForecaster

                model = ProphetForecaster.load(comp_dir)
            else:
                from .neural_prophet import NeuralProphetForecaster

                model = NeuralProphetForecaster.load(comp_dir)

            instance._components.append((comp_config, model))

        instance._is_fitted = True
        instance._decomposer = instance._create_decomposer()

        return instance

    def plot(
        self,
        df: pd.DataFrame,
        df_test: pd.DataFrame | None = None,
        window_days: float | None = None,
        figsize: tuple[float, float] = (14, 6),
        title: str | None = None,
        ax=None,
    ):
        """Plot ensemble historical data, fitted values, and forecast.

        Shows the historical data with in-sample fitted values and generates
        a forecast. Optionally overlays actual test data.

        Args:
            df: Historical data with 'ds' and 'y' columns.
            df_test: Optional test data to overlay actual values on forecast.
            window_days: Number of days to show. If None, shows all training data.
            figsize: Figure size as (width, height).
            title: Custom plot title. If None, uses ensemble name.
            ax: Optional matplotlib axes to plot on. If None, creates new figure.

        Returns:
            matplotlib figure and axes (fig, ax).

        Raises:
            RuntimeError: If model hasn't been fitted yet.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before plotting.")

        try:
            import matplotlib.pyplot as plt
        except ImportError as err:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib"
            ) from err

        # Get in-sample fitted values
        fitted = self._get_fitted_values(df)

        # Generate forecast
        forecast = self.predict(df, periods=self.config.n_forecast)

        # Prepare data for plotting
        df_plot = df.copy()
        df_plot["ds"] = pd.to_datetime(df_plot["ds"])

        # Merge fitted values with df_plot
        df_plot = df_plot.merge(fitted[["ds", "yhat"]], on="ds", how="left")

        # Apply window filter
        if window_days is not None:
            cutoff = df_plot["ds"].max() - pd.Timedelta(days=window_days)
            df_plot = df_plot[df_plot["ds"] >= cutoff]

        # Create figure
        if ax is None:
            fig, ax1 = plt.subplots(figsize=figsize)
        else:
            ax1 = ax
            fig = ax.get_figure()

        # Plot historical data
        ax1.plot(df_plot["ds"], df_plot["y"], "k-", lw=1, alpha=0.8, label="Historical data")

        # Plot fitted values
        ax1.plot(df_plot["ds"], df_plot["yhat"], "r-", lw=1, alpha=0.8, label="Fitted")

        # Plot forecast
        forecast_start = df_plot["ds"].max()
        ax1.axvline(forecast_start, color="gray", linestyle="--", alpha=0.5, label="Forecast start")
        ax1.plot(forecast["ds"], forecast["yhat"], "b-", lw=2, label="Forecast")

        # Plot uncertainty bands if available
        if "yhat_lower" in forecast.columns and "yhat_upper" in forecast.columns:
            ax1.fill_between(
                forecast["ds"],
                forecast["yhat_lower"],
                forecast["yhat_upper"],
                color="blue",
                alpha=0.1,
                label="Uncertainty",
            )

        # Overlay test data if provided
        if df_test is not None:
            df_test_plot = df_test.copy()
            df_test_plot["ds"] = pd.to_datetime(df_test_plot["ds"])
            ax1.plot(df_test_plot["ds"], df_test_plot["y"], "g-", lw=2, alpha=0.8, label="Actual")

        ax1.set_ylabel("Temperature (°C)")
        ax1.set_title(title or f"{self.name} - Ensemble Forecast")
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)

        if ax is None:
            plt.tight_layout()

        return fig, ax1

    def plot_components(
        self,
        df: pd.DataFrame,
        df_test: pd.DataFrame | None = None,
        window_days: float | None = 7,
        figsize: tuple[float, float] = (14, 12),
    ):
        """Plot each component model in separate panels.

        Creates a panel for each component (high_freq, low_freq, etc.)
        using each component's own plot() method.
        The final panel shows the combined ensemble forecast.

        Args:
            df: Historical data with 'ds' and 'y' columns.
            df_test: Optional test data to overlay actual values on forecast.
            window_days: Days to show for each panel. Default 7.
            figsize: Figure size as (width, height).

        Returns:
            matplotlib figure and axes array.

        Raises:
            RuntimeError: If model hasn't been fitted yet.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before plotting.")

        try:
            import matplotlib.pyplot as plt
        except ImportError as err:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib"
            ) from err

        n_components = len(self._components)
        fig, axes = plt.subplots(n_components + 1, 1, figsize=figsize, sharex=False)

        # Plot each component using its own plot() method
        for i, (comp_config, model) in enumerate(self._components):
            ax = axes[i]

            # Use component model's plot() method
            if comp_config.model_type == "prophet":
                model.plot(
                    df_test=df_test,
                    window_days=window_days,
                    title=f"{comp_config.name} ({comp_config.model_type})",
                    ax=ax,
                )
            else:
                # NeuralProphet needs df parameter
                model.plot(
                    df=model._fit_df if hasattr(model, "_fit_df") else df,
                    df_test=df_test,
                    window_days=window_days,
                    title=f"{comp_config.name} ({comp_config.model_type})",
                    ax=ax,
                )

            ax.set_ylabel(f"{comp_config.name} (°C)")

        # Final panel: combined ensemble
        ax_sum = axes[-1]

        # Get fitted values and forecast
        fitted = self._get_fitted_values(df)
        forecast = self.predict(df, periods=self.config.n_forecast)

        # Prepare data for plotting
        df_plot = df.copy()
        df_plot["ds"] = pd.to_datetime(df_plot["ds"])
        df_plot = df_plot.merge(fitted[["ds", "yhat"]], on="ds", how="left")

        # Apply window filter
        if window_days is not None:
            cutoff = df_plot["ds"].max() - pd.Timedelta(days=window_days)
            df_plot = df_plot[df_plot["ds"] >= cutoff]

        # Plot historical data and fitted
        ax_sum.plot(df_plot["ds"], df_plot["y"], "k-", lw=1, alpha=0.8, label="Historical data")
        ax_sum.plot(df_plot["ds"], df_plot["yhat"], "r-", lw=1, alpha=0.8, label="Fitted")

        # Plot forecast
        forecast_start = df_plot["ds"].max()
        ax_sum.axvline(
            forecast_start, color="gray", linestyle="--", alpha=0.5, label="Forecast start"
        )
        ax_sum.plot(forecast["ds"], forecast["yhat"], "b-", lw=2, label="Ensemble forecast")

        # Overlay test data if provided
        if df_test is not None:
            df_test_plot = df_test.copy()
            df_test_plot["ds"] = pd.to_datetime(df_test_plot["ds"])
            ax_sum.plot(
                df_test_plot["ds"], df_test_plot["y"], "g-", lw=2, alpha=0.8, label="Actual"
            )

        ax_sum.set_xlabel("Date")
        ax_sum.set_ylabel("Combined (°C)")
        ax_sum.set_title(f"Combined Ensemble (last {window_days}d + forecast)")
        ax_sum.legend(loc="upper right", fontsize=8)
        ax_sum.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, axes
