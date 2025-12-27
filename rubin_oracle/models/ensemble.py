"""Ensemble Forecaster combining multiple NeuralProphet models on decomposed signals."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rubin_oracle.config import NeuralProphetConfig
from rubin_oracle.models.neural_prophet import NeuralProphetForecaster
from rubin_oracle.utils.postprocessing import PostProcessor


class EnsembleForecaster:
    """Ensemble forecaster using multiple NeuralProphet models on pre-decomposed signals.

    Expects input data to be pre-decomposed externally. Fits separate NeuralProphet
    models on high-frequency and low-frequency components and combines their forecasts.

    Example:
        >>> from rubin_oracle.config import NeuralProphetConfig
        >>> high_cfg = NeuralProphetConfig.from_yaml("configs/high_freq.yaml")
        >>> low_cfg = NeuralProphetConfig.from_yaml("configs/low_freq.yaml")
        >>> forecaster = EnsembleForecaster(
        ...     high_freq_config=high_cfg,
        ...     low_freq_config=low_cfg,
        ...     high_freq_cols=["y_high_12h", "y_high_24h"],
        ...     low_freq_cols=["y_low_7d"],
        ... )
        >>> forecaster.fit(df_decomposed)
        >>> forecast = forecaster.predict(df_recent_decomposed)
    """

    def __init__(
        self,
        high_freq_config: NeuralProphetConfig,
        low_freq_config: NeuralProphetConfig,
        high_freq_cols: list[str],
        low_freq_cols: list[str],
        combine_method: str = "sum",
        bias_correction: bool = True,
        bias_window_hours: float = 6.0,
        bias_method: str = "median",
        output_freq: str | None = None,
    ):
        """Initialize EnsembleForecaster.

        Args:
            high_freq_config: NeuralProphetConfig for high-frequency model
            low_freq_config: NeuralProphetConfig for low-frequency model
            high_freq_cols: Column names for high-frequency components (e.g., ["y_high_12h"])
            low_freq_cols: Column names for low-frequency components (e.g., ["y_low_7d"])
            combine_method: How to combine forecasts ("sum" or "weighted")
            bias_correction: Apply bias correction post-processing
            bias_window_hours: Hours of recent data for bias estimation
            bias_method: Bias calculation method ("median" or "mean")
            output_freq: Output frequency for combined forecast (defaults to high_freq_config.freq)
        """
        self.name = "ensemble"

        # Store configs
        self._high_freq_config = high_freq_config
        self._low_freq_config = low_freq_config

        # Frequency settings - extract from configs
        self._high_freq = high_freq_config.freq
        self._low_freq = low_freq_config.freq
        self._output_freq = output_freq or self._high_freq  # Default to finest resolution

        # Column mappings
        self._high_freq_cols = high_freq_cols
        self._low_freq_cols = low_freq_cols

        # Combination settings
        self._combine_method = combine_method

        # Bias correction settings
        self._bias_correction = bias_correction
        self._bias_window_hours = bias_window_hours
        self._bias_method = bias_method

        # Models
        self._models: dict[str, NeuralProphetForecaster] = {}

        # State
        self._is_fitted = False
        self._training_end: pd.Timestamp | None = None
        self.metrics_: dict | None = None

    def _init_models(self) -> None:
        """Initialize NeuralProphet models for each component."""
        self._models["high_freq"] = NeuralProphetForecaster(self._high_freq_config)
        self._models["low_freq"] = NeuralProphetForecaster(self._low_freq_config)

    def _resample_to_freq(self, df: pd.DataFrame, target_freq: str) -> pd.DataFrame:
        """Downsample data to target frequency (e.g., 1h -> 3h).

        Args:
            df: DataFrame with 'ds' and 'y' columns
            target_freq: Target frequency string (e.g., "3h")

        Returns:
            Resampled DataFrame
        """
        df_resampled = df.set_index("ds").resample(target_freq).mean().reset_index()
        return df_resampled.dropna()

    def _upsample_to_freq(self, df: pd.DataFrame, target_freq: str) -> pd.DataFrame:
        """Upsample data to finer frequency (e.g., 3h -> 1h) using forward fill.

        Args:
            df: DataFrame with 'ds' column and value columns
            target_freq: Target frequency string (e.g., "1h")

        Returns:
            Upsampled DataFrame
        """
        _df = df.drop_duplicates("ds").reset_index(drop=True)
        df_upsampled = _df.set_index("ds").resample(target_freq).interpolate().reset_index()
        return df_upsampled

    def fit(self, df: pd.DataFrame, verbose: bool = False) -> EnsembleForecaster:
        """Fit ensemble on pre-decomposed data.

        Args:
            df: Pre-decomposed data with 'ds', 'y', and component columns
            verbose: Print progress

        Returns:
            Self for method chaining
        """
        df = df.copy()

        # Initialize models
        self._init_models()

        self._training_end = df["ds"].max()

        # Fit high_freq model
        if verbose:
            print(f"Fitting high_freq on columns: {self._high_freq_cols} (freq={self._high_freq})")
        df_high = df[["ds"]].copy()
        df_high["y"] = df[self._high_freq_cols].sum(axis=1)
        # Resample if high_freq differs from input (output) frequency
        if self._high_freq != self._output_freq:
            df_high = self._resample_to_freq(df_high, self._high_freq)
        self._models["high_freq"].fit(df_high[["ds", "y"]], verbose=verbose)

        # Fit low_freq model
        if verbose:
            print(f"Fitting low_freq on columns: {self._low_freq_cols} (freq={self._low_freq})")
        df_low = df[["ds"]].copy()
        df_low["y"] = df[self._low_freq_cols].sum(axis=1)
        # Resample if low_freq differs from input (output) frequency
        if self._low_freq != self._output_freq:
            df_low = self._resample_to_freq(df_low, self._low_freq)
        self._models["low_freq"].fit(df_low[["ds", "y"]], verbose=verbose)

        self._is_fitted = True
        self._compute_metrics()

        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate ensemble forecast from pre-decomposed data.

        Args:
            df: Pre-decomposed recent data for AR context

        Returns:
            DataFrame with combined forecast at output_freq resolution
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() first")

        # High-freq forecast
        df_high = df[["ds"]].copy()
        df_high["y"] = df[self._high_freq_cols].sum(axis=1)

        df_low = df[["ds"]].copy()
        df_low["y"] = df[self._low_freq_cols].sum(axis=1)

        forecast_high_raw = self._models["high_freq"].predict(
            df, include_history=True, window_days=int(2 * self._models["low_freq"].config.lag_days)
        )
        forecast_low_raw = self._models["low_freq"].predict(
            df, include_history=True, window_days=int(2 * self._models["low_freq"].config.lag_days)
        )
        forecast_high = self._models["high_freq"].standardize_output(forecast_high_raw)
        forecast_low = self._models["low_freq"].standardize_output(forecast_low_raw)

        # Upsample output if needed
        if self._high_freq != self._output_freq:
            forecast_high = self._upsample_to_freq(forecast_high, self._output_freq)

        # Upsample output if needed
        if self._low_freq != self._output_freq:
            forecast_low = self._upsample_to_freq(forecast_low, self._output_freq)

        # Combine forecasts (merge on timestamp to handle alignment)
        combined = self._combine_forecasts(forecast_high, forecast_low)

        all_cols = self._high_freq_cols + self._low_freq_cols
        df["y"] = df[all_cols].sum(axis=1)
        self._fit_df = df

        # Post-process
        if self._bias_correction:
            combined = self._apply_bias_correction(combined, df)

        return combined

    def _combine_forecasts(
        self, forecast_high: pd.DataFrame, forecast_low: pd.DataFrame
    ) -> pd.DataFrame:
        """Combine component forecasts into ensemble prediction.

        Merges on timestamp to handle different forecast lengths from resampling.
        """
        # Normalize ds columns for merging
        forecast_high = forecast_high.copy()
        forecast_low = forecast_low.copy()

        # Get yhat columns from low_freq to merge
        yhat_cols = [c for c in forecast_low.columns if c.startswith("yhat")]
        merge_cols = ["ds"] + yhat_cols

        # Merge on ds
        result = forecast_high.merge(
            forecast_low[merge_cols],
            on="ds",
            how="left",
            suffixes=("_high", "_low"),
        )

        # Sum yhat columns
        if self._combine_method == "sum":
            result["yhat"] = result["yhat_high"] + result["yhat_low"]
            result["yhat_lower"] = result["yhat_lower_high"] + result["yhat_lower_low"]
            result["yhat_upper"] = result["yhat_upper_high"] + result["yhat_upper_low"]
        return result

    def _apply_bias_correction(
        self, forecast: pd.DataFrame, df_recent: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply bias correction using recent residuals."""
        # Calculate number of rows from hours and frequency
        freq_hours = pd.to_timedelta(self._output_freq).total_seconds() / 3600
        bias_window_rows = int(self._bias_window_hours / freq_hours)

        # Create actual_df with 'y' column
        actual_df = df_recent[["y"]].copy()

        return PostProcessor.apply_bias_correction(
            forecast_df=forecast,
            actual_df=actual_df,
            bias_window_rows=bias_window_rows,
            bias_method=self._bias_method,
        )

    def fitted(self, window_days: float = 14.0) -> pd.DataFrame:
        """Get combined fitted values from all components.

        Args:
            window_days: Days of history to return

        Returns:
            DataFrame with ds, y, yhat, yhat_lower, yhat_upper columns at output_freq
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet.")

        # Ensure window is at least as large as max lag_days + buffer for n_forecasts
        # Need at least (n_lags + n_forecasts) samples, so add 2 day buffer
        max_lag = max(self._high_freq_config.lag_days, self._low_freq_config.lag_days)
        window_days = max(window_days, max_lag + 2)

        # Get fitted values from each component
        high_fitted = self._models["high_freq"].fitted(window_days=window_days)
        low_fitted = self._models["low_freq"].fitted(
            window_days=int(3 * self._models["low_freq"].config.lag_days)
        )

        # Upsample if frequencies differ from output
        if self._high_freq != self._output_freq:
            high_fitted = self._upsample_to_freq(high_fitted, self._output_freq)
        if self._low_freq != self._output_freq:
            low_fitted = self._upsample_to_freq(low_fitted, self._output_freq)

        # # Normalize ds columns for merging
        # if high_fitted["ds"].dt.tz is not None:
        #     high_fitted["ds"] = high_fitted["ds"].dt.tz_localize(None)
        # if low_fitted["ds"].dt.tz is not None:
        #     low_fitted["ds"] = low_fitted["ds"].dt.tz_localize(None)

        # Check for uncertainty bands before renaming
        has_uncertainty = "yhat_lower" in high_fitted.columns and "yhat_lower" in low_fitted.columns

        # Rename columns to indicate component
        high_fitted = high_fitted.rename(
            columns={
                "yhat": "yhat_high",
                "yhat_lower": "yhat_lower_high",
                "yhat_upper": "yhat_upper_high",
            }
        )
        low_fitted = low_fitted.rename(
            columns={
                "yhat": "yhat_low",
                "yhat_lower": "yhat_lower_low",
                "yhat_upper": "yhat_upper_low",
            }
        )

        # Merge on ds to handle alignment
        high_cols = ["ds", "yhat_high"]
        low_cols = ["ds", "yhat_low"]
        if has_uncertainty:
            high_cols += ["yhat_lower_high", "yhat_upper_high"]
            low_cols += ["yhat_lower_low", "yhat_upper_low"]

        combined = high_fitted[high_cols].merge(
            low_fitted[low_cols],
            on="ds",
            how="left",
        )
        combined["yhat_low"] = combined["yhat_low"].fillna(0)
        combined["yhat"] = combined["yhat_high"] + combined["yhat_low"]

        # Combine uncertainty bands
        if has_uncertainty:
            combined["yhat_lower_low"] = combined["yhat_lower_low"].fillna(0)
            combined["yhat_upper_low"] = combined["yhat_upper_low"].fillna(0)
            combined["yhat_lower"] = combined["yhat_lower_high"] + combined["yhat_lower_low"]
            combined["yhat_upper"] = combined["yhat_upper_high"] + combined["yhat_upper_low"]

        window_days_samples = int(
            window_days * 24 * pd.to_timedelta(self._output_freq).total_seconds() / 3600
        )
        return combined.tail(np.abs(window_days_samples))

    def _compute_metrics(self) -> None:
        """Compute ensemble in-sample metrics."""
        self.metrics_ = {}
        for comp_name, model in self._models.items():
            if model.metrics_:
                for key, val in model.metrics_.items():
                    self.metrics_[f"{comp_name}_{key}"] = val

    def save(self, path: str | Path) -> None:
        """Save ensemble to directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save configs
        self._high_freq_config.to_yaml(path / "high_freq_config.yaml")
        self._low_freq_config.to_yaml(path / "low_freq_config.yaml")

        # Save column mappings and settings
        import json

        settings = {
            "high_freq_cols": self._high_freq_cols,
            "low_freq_cols": self._low_freq_cols,
            "combine_method": self._combine_method,
            "bias_correction": self._bias_correction,
            "bias_window_hours": self._bias_window_hours,
            "bias_method": self._bias_method,
            "output_freq": self._output_freq,
        }
        with open(path / "settings.json", "w") as f:
            json.dump(settings, f, indent=2)

        # Save each component model
        for comp_name, model in self._models.items():
            model.save(path / comp_name)

    @classmethod
    def load(cls, path: str | Path) -> EnsembleForecaster:
        """Load ensemble from directory."""
        import json

        path = Path(path)

        # Load configs
        high_cfg = NeuralProphetConfig.from_yaml(path / "high_freq_config.yaml")
        low_cfg = NeuralProphetConfig.from_yaml(path / "low_freq_config.yaml")

        # Load settings
        with open(path / "settings.json") as f:
            settings = json.load(f)

        forecaster = cls(
            high_freq_config=high_cfg,
            low_freq_config=low_cfg,
            high_freq_cols=settings["high_freq_cols"],
            low_freq_cols=settings["low_freq_cols"],
            combine_method=settings["combine_method"],
            bias_correction=settings["bias_correction"],
            bias_window_hours=settings["bias_window_hours"],
            bias_method=settings["bias_method"],
            output_freq=settings.get("output_freq"),  # Backwards compatible
        )

        # Load component models
        forecaster._models["high_freq"] = NeuralProphetForecaster.load(path / "high_freq")
        forecaster._models["low_freq"] = NeuralProphetForecaster.load(path / "low_freq")

        forecaster._is_fitted = True
        return forecaster

    def plot_components(self, window_days: float = 7.0) -> None:
        """Plot each component model separately."""
        for comp_name, model in self._models.items():
            print(f"\n{comp_name}:")
            model.plot(window_days=window_days, title=comp_name)
            plt.show()

    def plot(
        self,
        df_test: pd.DataFrame | None = None,
        window_days: float = 7.0,
        figsize: tuple[float, float] = (14, 10),
        show_components: bool = True,
        show_residuals: bool = False,
        title: str | None = None,
    ) -> tuple[plt.Figure, np.ndarray]:
        """Plot ensemble forecast with component breakdown.

        Creates a multi-panel plot showing:
        - Top panel: Original signal + combined ensemble forecast
        - Middle panels: Individual component forecasts (high_freq, low_freq)
        - Bottom panel (optional): Residuals

        Args:
            df_test: Optional test data to overlay actual values
            window_days: Number of days to display from training end
            figsize: Figure size as (width, height)
            show_components: Show individual component forecasts
            show_residuals: Show residuals panel
            title: Custom plot title

        Returns:
            (fig, axes) tuple

        Raises:
            RuntimeError: If model hasn't been fitted
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        # Determine number of panels
        n_panels = 1  # Always have main panel
        if show_components:
            n_panels += len(self._models)
        if show_residuals:
            n_panels += 1

        # Create figure
        height_ratios = [2] + [1] * (n_panels - 1)
        fig, axes = plt.subplots(
            n_panels,
            1,
            figsize=figsize,
            sharex=True,
            height_ratios=height_ratios,
        )

        if n_panels == 1:
            axes = [axes]

        # Get combined fitted values
        df_fitted = self.fitted(window_days=window_days).sort_values("ds")

        # Apply window filter
        initial_date = self._training_end - pd.Timedelta(days=window_days)
        end_date = self._training_end + pd.Timedelta(days=1.25)

        cutoff = self._training_end - pd.Timedelta(days=int(3 * window_days))
        df_plot = df_fitted[df_fitted["ds"] >= cutoff].copy()

        # Separate historical fit from forecast
        forecast_start = self._training_end
        df_history = df_plot[df_plot["ds"] <= forecast_start]
        df_forecast = df_plot[df_plot["ds"] > forecast_start]

        # Panel 0: Main plot
        ax_main = axes[0]

        if "y" in df_history.columns:
            ax_main.plot(df_history["ds"], df_history["y"], "k-", lw=1, alpha=0.8, label="Actual")

        ax_main.plot(
            df_history["ds"], df_history["yhat"], "r-", lw=1, alpha=0.7, label="Fitted (ensemble)"
        )

        if "yhat_lower" in df_history.columns:
            ax_main.fill_between(
                df_history["ds"],
                df_history["yhat_lower"],
                df_history["yhat_upper"],
                color="red",
                alpha=0.1,
            )

        ax_main.axvline(
            forecast_start, color="gray", linestyle="--", alpha=0.5, label="Forecast start"
        )

        if len(df_forecast) > 0:
            ax_main.plot(df_forecast["ds"], df_forecast["yhat"], "b-", lw=2, label="Forecast")
            if "yhat_lower" in df_forecast.columns:
                ax_main.fill_between(
                    df_forecast["ds"],
                    df_forecast["yhat_lower"],
                    df_forecast["yhat_upper"],
                    color="blue",
                    alpha=0.15,
                )

        if df_test is not None:
            df_test_plot = df_test.copy()
            df_test_plot["ds"] = pd.to_datetime(df_test_plot["ds"])
            if df_test_plot["ds"].dt.tz is not None:
                df_test_plot["ds"] = df_test_plot["ds"].dt.tz_localize(None)
            df_test_plot = df_test_plot[df_test_plot["ds"] >= cutoff]
            ax_main.plot(
                df_test_plot["ds"], df_test_plot["y"], "g-", lw=1.5, alpha=0.8, label="Test actual"
            )

        ax_main.set_ylabel("Value")
        ax_main.set_title(title or f"Ensemble Forecast ({self.name})")
        ax_main.legend(loc="upper left", fontsize=9)
        ax_main.grid(True, alpha=0.3)

        # Component panels
        panel_idx = 1
        colors = {"high_freq": "#e74c3c", "low_freq": "#3498db"}
        col_labels = {"high_freq": self._high_freq_cols, "low_freq": self._low_freq_cols}

        if show_components:
            for comp_name, model in self._models.items():
                ax = axes[panel_idx]
                color = colors.get(comp_name, f"C{panel_idx}")

                try:
                    comp_fitted = model.fitted(window_days=int(3 * window_days)).sort_values("ds")
                    comp_fitted = comp_fitted[comp_fitted["ds"] >= cutoff]

                    comp_history = comp_fitted[comp_fitted["ds"] <= forecast_start]
                    comp_forecast = comp_fitted[comp_fitted["ds"] > forecast_start]

                    if "y" in comp_history.columns:
                        ax.plot(
                            comp_history["ds"],
                            comp_history["y"],
                            "k-",
                            lw=0.8,
                            alpha=0.6,
                            label=f"Actual ({comp_name})",
                        )

                    ax.plot(
                        comp_history["ds"],
                        comp_history["yhat"],
                        color=color,
                        lw=1,
                        alpha=0.8,
                        label="Fitted",
                    )

                    if "yhat_lower" in comp_history.columns:
                        ax.fill_between(
                            comp_history["ds"],
                            comp_history["yhat_lower"],
                            comp_history["yhat_upper"],
                            color=color,
                            alpha=0.1,
                        )

                    ax.axvline(forecast_start, color="gray", linestyle="--", alpha=0.3)

                    if len(comp_forecast) > 0:
                        ax.plot(
                            comp_forecast["ds"],
                            comp_forecast["yhat"],
                            color=color,
                            lw=2,
                            linestyle="-",
                            label="Forecast",
                        )
                        sample_cut = min(len(comp_forecast), int(window_days * 24))
                        ymean, ystd = (
                            np.nanmedian(comp_forecast["yhat"].tail(sample_cut)),
                            np.nanstd(comp_forecast["yhat"]),
                        )
                        ymax, ymin = ymean + 3 * ystd, ymean - 3 * ystd
                        ax.set_ylim(ymin, ymax)

                except Exception as e:
                    ax.text(
                        0.5,
                        0.5,
                        f"Error plotting {comp_name}: {e}",
                        transform=ax.transAxes,
                        ha="center",
                    )

                cols = col_labels.get(comp_name, [])
                ax.set_ylabel(f"{comp_name}\n({', '.join(cols)})")
                ax.legend(loc="upper left", fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.axhline(0, color="gray", linestyle="-", alpha=0.2)

                panel_idx += 1

        # Residuals panel
        if show_residuals and "y" in df_history.columns:
            ax = axes[panel_idx]
            residuals = df_history["y"] - df_history["yhat"]

            ax.plot(df_history["ds"], residuals, "g-", lw=0.5, alpha=0.8)
            ax.axhline(0, color="k", linestyle="-", alpha=0.3)
            ax.fill_between(
                df_history["ds"], residuals, 0, where=residuals > 0, color="green", alpha=0.2
            )
            ax.fill_between(
                df_history["ds"], residuals, 0, where=residuals < 0, color="red", alpha=0.2
            )

            ax.set_ylabel("Residuals")
            ax.grid(True, alpha=0.3)

            rmse = np.sqrt(np.mean(residuals**2))
            ax.text(
                0.02,
                0.95,
                f"RMSE: {rmse:.3f}",
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )

        # Final formatting
        axes[-1].set_xlabel("Date")
        axes[-1].set_xlim(initial_date, end_date)
        plt.tight_layout()

        return fig, axes
