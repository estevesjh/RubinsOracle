"""Post-processing utilities for Rubin's Oracle forecasters.

This module provides utilities for post-processing forecast outputs including
Savitzky-Golay smoothing, bias correction, and ETS blending.
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


class PostProcessor:
    """Static utility class for forecast post-processing operations.

    Handles smoothing, bias correction, and other forecast adjustments.
    """

    @staticmethod
    def apply_savgol_smoothing(
        df: pd.DataFrame,
        window_length: int = 5,
        polyorder: int = 3,
        mode: str = "nearest",
    ) -> pd.DataFrame:
        """Apply Savitzky-Goyal smoothing to yhat column.

        Args:
            df: DataFrame with 'yhat' column (standardized forecast format)
            window_length: Length of smoothing window (must be odd)
            polyorder: Polynomial order
            mode: Edge handling mode ('nearest', 'constant', 'mirror', etc.)

        Returns:
            DataFrame with smoothed 'yhat' values

        Raises:
            ValueError: If window_length is even or too large
        """
        result = df.copy()

        # Ensure window_length is odd
        if window_length % 2 == 0:
            window_length += 1

        # Check window is not larger than data
        if window_length > len(result):
            window_length = len(result) if len(result) % 2 == 1 else len(result) - 1

        if window_length < 3:
            window_length = 3

        # Apply Savitzky-Golay filter
        result["yhat"] = savgol_filter(
            result["yhat"].values,
            window_length=window_length,
            polyorder=min(polyorder, window_length - 1),
            mode=mode,
        )

        return result

    @staticmethod
    def apply_bias_correction(
        forecast_df: pd.DataFrame,
        actual_df: pd.DataFrame,
        bias_window_hours: float = 6.0,
        bias_method: str = "median",
    ) -> pd.DataFrame:
        """Apply bias correction based on recent errors.

        Args:
            forecast_df: DataFrame with forecast (columns: ds, yhat, step)
            actual_df: DataFrame with actuals (columns: ds, y)
            bias_window_hours: Hours of recent history to use for bias
            bias_method: 'mean' or 'median' for bias calculation

        Returns:
            DataFrame with bias-corrected 'yhat' values
        """
        result = forecast_df.copy()

        # Merge forecast with actuals
        merged = result.merge(actual_df[["ds", "y"]], on="ds", how="left")

        # Filter to recent window
        if len(merged) > 0 and "ds" in merged.columns:
            latest_time = merged["ds"].max()
            window_start = latest_time - pd.Timedelta(hours=bias_window_hours)
            recent = merged[merged["ds"] >= window_start]

            if len(recent) > 0:
                # Calculate bias (actual - forecast)
                bias_data = recent.dropna(subset=["y", "yhat"])
                if len(bias_data) > 0:
                    residuals = bias_data["y"] - bias_data["yhat"]

                    if bias_method == "median":
                        bias = residuals.median()
                    else:  # mean
                        bias = residuals.mean()

                    # Apply bias correction
                    result["yhat"] = result["yhat"] + bias

        return result

    @staticmethod
    def apply_ets_blending(
        forecast_df: pd.DataFrame,
        alpha: float = 0.3,
    ) -> pd.DataFrame:
        """Apply exponential triple smoothing (ETS) blending.

        Applies exponential smoothing to forecast values across steps.

        Args:
            forecast_df: DataFrame with forecast (columns: ds, yhat, step)
            alpha: Smoothing parameter (0 < alpha <= 1)

        Returns:
            DataFrame with ETS-blended 'yhat' values
        """
        result = forecast_df.copy()

        if len(result) == 0:
            return result

        # Group by ds and apply ETS within each group
        def apply_ets_to_group(group):
            if len(group) <= 1:
                return group

            values = group["yhat"].values.copy()
            smoothed = np.zeros_like(values, dtype=float)

            # Initialize
            smoothed[0] = values[0]

            # Apply exponential smoothing
            for i in range(1, len(values)):
                smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]

            group["yhat"] = smoothed
            return group

        result = result.groupby("ds", group_keys=False).apply(apply_ets_to_group)

        return result
