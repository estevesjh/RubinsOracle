"""Signal preprocessing and decomposition for Rubin's Oracle.

This module implements signal decomposition using Savitzky-Golay filters to extract
multi-frequency components from temperature time series data. These decomposed
components can be used as regressors in forecasting models.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


class SignalDecomposer:
    """Decomposes time series into multiple frequency components.

    Uses Savitzky-Golay filters to extract trend components at different time scales,
    from high-frequency (sub-daily) to low-frequency (multi-week) variations.

    Attributes:
        freq: Data frequency in observations per day (e.g., 4 for 15min data, 24 for hourly)
        polyorder: Polynomial order for Savitzky-Golay filter
        mode: How to handle NaN values ('drop' or 'keep')

    Example:
        >>> decomposer = SignalDecomposer(freq=4)  # 15-minute data
        >>> df_with_components = decomposer.decompose(df)
        >>> # Now df has y_high, y_p0, y_p1, ..., y_p5 columns
    """

    def __init__(
        self,
        freq: int = 4,
        polyorder: int = 3,
        mode: Literal['drop', 'keep'] = 'drop',
        savgol_mode: Literal['mirror', 'nearest', 'constant', 'wrap', 'interp'] = 'nearest'
    ):
        """Initialize the signal decomposer.

        Args:
            freq: Number of observations per day (4 = 15min, 24 = hourly)
            polyorder: Polynomial order for Savitzky-Golay filter (must be < window_length)
            mode: How to handle NaN values created by filtering:
                - 'drop': Remove rows with NaN in decomposed components
                - 'keep': Keep all rows (NaN will be at start/end of each component)
            savgol_mode: Boundary mode for Savitzky-Golay filter:
                - 'nearest': Extend with edge values (recommended for temp data)
                - 'mirror': Mirror values at edges
                - 'constant': Use constant value at edges
                - 'wrap': Wrap around to other end
                - 'interp': Polynomial interpolation
        """
        self.freq = freq
        self.polyorder = polyorder
        self.mode = mode
        self.savgol_mode = savgol_mode

        # Define window lengths for different frequency components
        # Based on multiples of observations per day
        self._windows = self._calculate_windows()

    def _calculate_windows(self) -> dict[str, int]:
        """Calculate window lengths for each frequency component.

        Returns:
            Dictionary mapping component names to window lengths
        """
        windows = {
            'p0': 1 * self.freq + 1,           # Sub-daily
            'day': int(1.4 * 24 * self.freq) + 1,  # Daily
            '3day': int(2.4 * 24 * self.freq) + 1,  # 3-day
            '6day': 6 * 24 * self.freq + 1,         # 6-day
            '11day': 11 * 24 * self.freq + 1,       # 11-day
            '14day': 17 * 24 * self.freq + 1,       # 14-day (using 17)
            '28day': 28 * 24 * self.freq + 1,       # 28-day
            '56day': 57 * 24 * self.freq + 1,       # 56-day (using 57)
        }

        # Ensure all windows are odd (required by savgol_filter)
        for key, window in windows.items():
            if window % 2 == 0:
                windows[key] = window + 1

        return windows

    def _apply_savgol(
        self,
        y: np.ndarray,
        window_length: int,
        name: str
    ) -> np.ndarray:
        """Apply Savitzky-Golay filter with configurable boundary handling.

        Args:
            y: Input signal
            window_length: Filter window length
            name: Component name (for logging)

        Returns:
            Filtered signal (with NaN at boundaries if mode='interp')
        """
        if len(y) < window_length:
            warnings.warn(
                f"Data length ({len(y)}) is shorter than window length ({window_length}) "
                f"for {name} component. Returning NaN array."
            )
            return np.full_like(y, np.nan)

        try:
            filtered = savgol_filter(
                y,
                window_length=window_length,
                polyorder=self.polyorder,
                mode=self.savgol_mode  # Use configurable mode
            )

            # Only set explicit NaN boundaries if mode is 'interp'
            # Other modes handle boundaries appropriately
            if self.savgol_mode == 'interp':
                half_window = window_length // 2
                filtered[:half_window] = np.nan
                filtered[-half_window:] = np.nan

            return filtered

        except Exception as e:
            warnings.warn(f"Error applying Savitzky-Golay filter for {name}: {e}")
            return np.full_like(y, np.nan)

    def decompose(self, df: pd.DataFrame) -> pd.DataFrame:
        """Decompose time series into multi-frequency components.

        Args:
            df: Input dataframe with columns:
                - ds (datetime): Timestamps
                - y (float): Target values (or tempMean as fallback)

        Returns:
            DataFrame with original columns plus decomposed components:
                - y_p0_trend, y_day_trend, ..., y_56day_trend: Trend components
                - y_high: High-frequency residual (y - y_p0_trend)
                - y_p0: Sub-daily variation (y_p0_trend - y_day_trend)
                - y_p1: 1-3 day variation (y_3day_trend - y_6day_trend)
                - y_p2: 3-6 day variation (y_6day_trend - y_11day_trend)
                - y_p3: 6-11 day variation (y_11day_trend - y_14day_trend)
                - y_p4: 11-28 day variation (y_14day_trend - y_28day_trend)
                - y_p5: 28-56 day variation (y_28day_trend - y_56day_trend)

        Raises:
            ValueError: If required columns are missing
        """
        df = df.copy()

        # Ensure 'y' exists
        if 'y' not in df.columns:
            if 'tempMean' in df.columns:
                df['y'] = df['tempMean']
            else:
                raise ValueError("Column 'y' or 'tempMean' not found in dataframe.")

        # Fill NaN values for filtering (required for continuous signal)
        # Use forward fill then backward fill to handle edges
        y_filled = df['y'].fillna(method='ffill').fillna(method='bfill').values

        # Apply Savitzky-Golay filters for each trend component
        print("Applying signal decomposition...")

        print(f"Computing p0 trend (w={self._windows['p0']})...")
        df['y_p0_trend'] = self._apply_savgol(y_filled, self._windows['p0'], 'p0')

        print(f"Computing day trend (w={self._windows['day']})...")
        df['y_day_trend'] = self._apply_savgol(y_filled, self._windows['day'], 'day')

        print(f"Computing 3day trend (w={self._windows['3day']})...")
        df['y_3day_trend'] = self._apply_savgol(y_filled, self._windows['3day'], '3day')

        print(f"Computing 6day trend (w={self._windows['6day']})...")
        df['y_6day_trend'] = self._apply_savgol(y_filled, self._windows['6day'], '6day')

        print(f"Computing 11day trend (w={self._windows['11day']})...")
        df['y_11day_trend'] = self._apply_savgol(y_filled, self._windows['11day'], '11day')

        print(f"Computing 14day trend (w={self._windows['14day']})...")
        df['y_14day_trend'] = self._apply_savgol(y_filled, self._windows['14day'], '14day')

        print(f"Computing 28day trend (w={self._windows['28day']})...")
        df['y_28day_trend'] = self._apply_savgol(y_filled, self._windows['28day'], '28day')

        print(f"Computing 56day trend (w={self._windows['56day']})...")
        df['y_56day_trend'] = self._apply_savgol(y_filled, self._windows['56day'], '56day')

        # Calculate frequency components as differences between trends
        print("Calculating frequency components...")
        df['y_high'] = df['y'] - df['y_p0_trend']
        df['y_p0'] = df['y_p0_trend'] - df['y_day_trend']
        df['y_p1'] = df['y_3day_trend'] - df['y_6day_trend']
        df['y_p2'] = df['y_6day_trend'] - df['y_11day_trend']
        df['y_p3'] = df['y_11day_trend'] - df['y_14day_trend']
        df['y_p4'] = df['y_14day_trend'] - df['y_28day_trend']
        df['y_p5'] = df['y_28day_trend'] - df['y_56day_trend']

        # Handle NaN values based on mode
        if self.mode == 'drop':
            # Drop rows where any decomposed component is NaN
            component_cols = ['y_high', 'y_p0', 'y_p1', 'y_p2', 'y_p3', 'y_p4', 'y_p5']
            initial_len = len(df)
            df = df.dropna(subset=component_cols)
            final_len = len(df)

            if final_len < initial_len:
                print(f"Dropped {initial_len - final_len} rows with NaN in decomposed components")

        print(f"Decomposition complete. Dataset shape: {df.shape}")

        return df


def preprocess_for_forecast(
    df: pd.DataFrame,
    decompose: bool = True,
    freq: int = 4,
    savgol_mode: str = 'nearest',
    train_start_date: str | pd.Timestamp | None = None,
    train_end_date: str | pd.Timestamp | None = None,
    lag_days: int | None = None,
) -> pd.DataFrame:
    """Preprocess data for forecasting with optional decomposition and date filtering.

    This function prepares time series data by:
    1. Optionally applying signal decomposition to extract frequency components
    2. Filtering data to a specified date range
    3. Ensuring sufficient history for lagged features

    Args:
        df: Input dataframe with 'ds' and 'y' columns
        decompose: Whether to apply signal decomposition
        freq: Data frequency in observations per day (for decomposition)
        savgol_mode: Boundary mode for Savitzky-Golay filter ('nearest' recommended)
        train_start_date: Start date for training data (inclusive)
            - For NeuralProphet: '2023-09-10' or later recommended
            - For Prophet: Uses lag_days before train_end_date if not specified
        train_end_date: End date for training data (inclusive)
        lag_days: Number of historical observations needed for AR models
            - If provided and train_start_date is None, automatically calculates start date

    Returns:
        Preprocessed dataframe with:
            - Filtered date range
            - Decomposed components (if decompose=True)
            - Sufficient history for lag_days (if specified)

    Example:
        >>> # For NeuralProphet: decompose with date filtering
        >>> df_train = preprocess_for_forecast(
        ...     df,
        ...     decompose=True,
        ...     freq=4,
        ...     train_start_date='2023-09-10',
        ...     lag_days=48
        ... )

        >>> # For Prophet: just filter dates, use lag for history
        >>> df_train = preprocess_for_forecast(
        ...     df,
        ...     decompose=False,
        ...     train_end_date='2024-01-01',
        ...     lag_days=48
        ... )
    """
    df = df.copy()

    # Ensure ds is datetime
    if 'ds' in df.columns:
        df['ds'] = pd.to_datetime(df['ds'])

    # Sort by timestamp
    df = df.sort_values('ds').reset_index(drop=True)

    # Apply decomposition first on full history (if requested)
    if decompose:
        decomposer = SignalDecomposer(freq=freq, mode='drop', savgol_mode=savgol_mode)
        df = decomposer.decompose(df)

    # Filter by date range
    if train_end_date is not None:
        train_end_date = pd.to_datetime(train_end_date)
        df = df[df['ds'] <= train_end_date]

    if train_start_date is not None:
        train_start_date = pd.to_datetime(train_start_date)
        df = df[df['ds'] >= train_start_date]
    elif lag_days is not None and train_end_date is not None:
        # If start date not specified but lag_days is, calculate it
        # Ensure we have enough history before the cutoff
        # lag_days is in observations, need to convert based on freq
        if 'ds' in df.columns and len(df) > 0:
            # Calculate time delta for lag_days observations
            # Assuming uniform spacing, take median
            if len(df) > 1:
                time_diffs = df['ds'].diff().dropna()
                median_diff = time_diffs.median()
                lookback_period = median_diff * lag_days
                calculated_start = train_end_date - lookback_period

                # Add a buffer to ensure we have enough data
                calculated_start = calculated_start - pd.Timedelta(days=1)

                df = df[df['ds'] >= calculated_start]
                print(f"Automatically set training start date to {calculated_start} "
                      f"to ensure {lag_days} lag observations")

    # Validate we have enough data
    if lag_days is not None and len(df) < lag_days:
        warnings.warn(
            f"After preprocessing, only {len(df)} observations remain, "
            f"but {lag_days} lag observations were requested. "
            f"This may cause issues with AR models."
        )

    return df.reset_index(drop=True)
