"""Data preprocessing utilities for Rubin's Oracle.

This module provides helper functions for validating and preprocessing
time series data for forecasting models.
"""

from __future__ import annotations

import pandas as pd


def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and prepare input DataFrame for forecasting.

    Ensures the DataFrame has required columns with proper types and
    performs basic data quality checks.

    Args:
        df: Input DataFrame to validate

    Returns:
        Validated DataFrame with proper types

    Raises:
        ValueError: If required columns are missing or have invalid types
        ValueError: If DataFrame is empty or contains invalid data

    Example:
        >>> df = pd.DataFrame({
        ...     'ds': pd.date_range('2024-01-01', periods=100, freq='h'),
        ...     'y': np.random.randn(100)
        ... })
        >>> validated_df = validate_input(df)
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty")

    # Check required columns
    required_cols = {'ds', 'y'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Make a copy to avoid modifying the original
    df = df.copy()

    # Ensure 'ds' is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['ds']):
        try:
            df['ds'] = pd.to_datetime(df['ds'])
        except Exception as e:
            raise ValueError(f"Could not convert 'ds' column to datetime: {e}")

    # Ensure 'y' is numeric
    if not pd.api.types.is_numeric_dtype(df['y']):
        try:
            df['y'] = pd.to_numeric(df['y'])
        except Exception as e:
            raise ValueError(f"Could not convert 'y' column to numeric: {e}")

    # Check for all NaN values in target
    if df['y'].isna().all():
        raise ValueError("Target column 'y' contains only NaN values")

    # Sort by datetime
    df = df.sort_values('ds').reset_index(drop=True)

    # Check for duplicate timestamps
    n_duplicates = df['ds'].duplicated().sum()
    if n_duplicates > 0:
        raise ValueError(
            f"Found {n_duplicates} duplicate timestamps. "
            "Please aggregate or remove duplicates before forecasting."
        )

    return df


def compute_temp_mean(
    df: pd.DataFrame,
    temp_max_col: str = 'tempMax',
    temp_min_col: str = 'tempMin',
) -> pd.DataFrame:
    """Compute temperature mean from tempMax and tempMin.

    This is the proper way to calculate mean temperature from daily
    max/min values, rather than simple interpolation.

    Args:
        df: DataFrame with temperature columns
        temp_max_col: Name of maximum temperature column
        temp_min_col: Name of minimum temperature column

    Returns:
        DataFrame with added 'y' column containing mean temperature

    Example:
        >>> df = pd.DataFrame({
        ...     'ds': dates,
        ...     'tempMax': [25, 26, 27],
        ...     'tempMin': [15, 16, 14]
        ... })
        >>> df = compute_temp_mean(df)
        >>> df['y']  # (tempMax + tempMin) / 2
        [20.0, 21.0, 20.5]
    """
    if temp_max_col not in df.columns or temp_min_col not in df.columns:
        raise ValueError(
            f"Columns '{temp_max_col}' and '{temp_min_col}' are required "
            "for temperature mean calculation"
        )

    df = df.copy()
    df['y'] = (df[temp_max_col] + df[temp_min_col]) / 2.0

    return df


def prepare_regular_frequency(
    df: pd.DataFrame,
    freq: str = 'h',
    interpolate: bool = True,
    method: str = 'time',
) -> pd.DataFrame:
    """Prepare time series with regular frequency.

    Ensures data has consistent temporal spacing by reindexing to a regular
    grid and optionally interpolating missing values.

    IMPORTANT: For temperature data, use compute_temp_mean() first to calculate
    the mean from tempMax/tempMin before calling this function. Simple interpolation
    is not appropriate for temperature aggregation.

    Args:
        df: DataFrame with 'ds' and 'y' columns
        freq: Target frequency ('15min', 'h', 'D', etc.)
        interpolate: Whether to interpolate missing values
        method: Interpolation method ('time', 'linear', 'nearest', 'ffill', 'bfill')

    Returns:
        DataFrame with regular frequency and no gaps

    Example:
        >>> # For temperature data, first compute mean from max/min
        >>> df = compute_temp_mean(df, 'tempMax', 'tempMin')
        >>> # Then prepare regular frequency
        >>> df_hourly = prepare_regular_frequency(df, freq='h', interpolate=False)
    """
    # Validate input first
    df = validate_input(df)

    # Set datetime as index temporarily
    df = df.set_index('ds')

    # Sort index and drop duplicates
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]

    # Build regular frequency grid
    full_idx = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=freq,
        name='ds',
    )

    # Reindex to regular grid
    df_regular = df.reindex(full_idx)

    # Interpolate if requested
    if interpolate:
        if method in ['time', 'linear']:
            df_regular['y'] = df_regular['y'].interpolate(method=method)
        elif method == 'ffill':
            df_regular['y'] = df_regular['y'].ffill()
        elif method == 'bfill':
            df_regular['y'] = df_regular['y'].bfill()
        elif method == 'nearest':
            df_regular['y'] = df_regular['y'].interpolate(method='nearest')

    # Reset index to get 'ds' column back
    df_regular = df_regular.reset_index()

    return df_regular


def check_missing_values(df: pd.DataFrame) -> dict[str, int | float]:
    """Analyze missing values in the time series.

    Args:
        df: DataFrame with 'ds' and 'y' columns

    Returns:
        Dictionary with missing value statistics:
            - n_missing: Number of missing values
            - pct_missing: Percentage of missing values
            - has_missing: Boolean indicating if any values are missing

    Example:
        >>> stats = check_missing_values(df)
        >>> if stats['has_missing']:
        ...     print(f"Warning: {stats['pct_missing']:.1f}% missing values")
    """
    df = validate_input(df)

    n_missing = df['y'].isna().sum()
    n_total = len(df)
    pct_missing = (n_missing / n_total * 100) if n_total > 0 else 0.0

    return {
        'n_missing': int(n_missing),
        'pct_missing': float(pct_missing),
        'has_missing': bool(n_missing > 0),
    }


def get_frequency(df: pd.DataFrame) -> str | None:
    """Infer the frequency of the time series.

    Args:
        df: DataFrame with 'ds' column

    Returns:
        Frequency string (e.g., 'h', 'D', 'W') or None if irregular

    Example:
        >>> freq = get_frequency(df)
        >>> print(f"Detected frequency: {freq}")
        Detected frequency: h
    """
    df = validate_input(df)

    if len(df) < 3:
        return None

    # Try to infer frequency
    freq = pd.infer_freq(df['ds'])

    return freq
