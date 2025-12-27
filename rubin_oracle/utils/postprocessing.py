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
        bias_window_rows: int = 6,
        bias_method: str = "median",
    ) -> pd.DataFrame:
        """Apply bias correction based on recent errors.

        Args:
            forecast_df: DataFrame with forecast (must have 'yhat' column)
            actual_df: DataFrame with actuals (must have 'y' column)
            bias_window_rows: Number of recent rows to use for bias calculation
            bias_method: 'mean' or 'median' for bias calculation

        Returns:
            DataFrame with bias-corrected 'yhat' values
        """
        result = forecast_df.copy()

        if len(actual_df) == 0 or "y" not in actual_df.columns:
            return result
        if "yhat" not in result.columns:
            return result

        # Get recent actuals and forecasts by row position (tail)
        n_rows = min(bias_window_rows, len(actual_df), len(result))
        if n_rows == 0:
            return result

        recent_actual = actual_df["y"].tail(n_rows).values
        recent_forecast = result["yhat"].head(n_rows).values

        # Calculate bias from valid pairs
        valid_mask = ~(np.isnan(recent_actual) | np.isnan(recent_forecast))
        if valid_mask.sum() == 0:
            return result

        residuals = recent_actual[valid_mask] - recent_forecast[valid_mask]

        if bias_method == "median":
            bias = np.median(residuals)
        else:  # mean
            bias = np.mean(residuals)

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
