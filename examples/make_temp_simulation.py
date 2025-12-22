from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_data(
    num_days: int = 14,
    start_date: str | pd.Timestamp = "2025-01-01",
    noise_std: float = 0.3,
    freq_resolution: str = "15min",
    random_seed: int | None = None,
    return_format: str = "dataframe",
) -> pd.DataFrame | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate temperature data using a fitted Fourier model with polynomial trend.

    The model is based on temperature patterns from the last 3 weeks (Dec 11 - Jan 1, 2025)
    in Santiago, Chile. It captures 7-day, 2-day, daily, and semi-daily cycles with a slight cooling trend.

    Parameters
    ----------
    num_days : int, default=14
        Number of days to simulate

    start_date : str or pd.Timestamp, default='2025-01-01'
        Starting date for simulation (format: 'YYYY-MM-DD' or pd.Timestamp)

    noise_std : float, default=0.3
        Standard deviation of random Gaussian noise added to simulate real data variation (°C).
        Realistic values: 0.1-0.5°C

    freq_resolution : str, default='15min'
        Time resolution for simulation. Common values: '15min', '30min', '1H', '1D'

    random_seed : int or None, default=None
        Random seed for reproducibility. If None, results will vary each call.

    return_format : str, default='dataframe'
        Format of return value: 'dataframe' or 'arrays'
        - 'dataframe': Returns pd.DataFrame with columns [timestamp, temperature]
        - 'arrays': Returns tuple (timestamps, temperatures, t_hours)

    Returns:
    -------
    pd.DataFrame or tuple
        If return_format='dataframe':
            DataFrame with columns:
            - timestamp: pd.DatetimeIndex
            - temperature: float (°C)

        If return_format='arrays':
            Tuple of (timestamps, temperatures, t_hours)
            - timestamps: np.array of pd.Timestamp objects
            - temperatures: np.array of temperature values (°C)
            - t_hours: np.array of time in hours from start

    Examples:
    --------
    >>> # Generate 14 days of data with default settings
    >>> df = simulate_data()
    >>> print(df.head())

    >>> # Generate 30 days starting from a specific date with low noise
    >>> df = simulate_data(num_days=30, start_date='2025-02-01', noise_std=0.1, random_seed=42)

    >>> # Get data as arrays for custom processing
    >>> timestamps, temps, hours = simulate_data(num_days=7, return_format='arrays', random_seed=123)

    Notes:
    -----
    Fitted coefficients (from 3-week calibration period, Dec 11 - Jan 1, 2025):
    - Mean temperature: 15.0881°C
    - 168h (7-day) cycle amplitude: 0.5208°C
    - 48h (2-day) cycle amplitude: 0.3691°C
    - 24h (1-day) cycle amplitude: 2.0066°C (dominant)
    - 12h (semi-daily) cycle amplitude: 0.5483°C
    - Linear trend: -0.02868°C/hour (slight cooling)
    - Fit quality: R² = 0.716, RMSE = 1.077°C

    The model equation is:
        T(t) = mean + c1*t + c2*t² + c3*t³
             + A1*cos(2π*t/168) + B1*sin(2π*t/168)  [7-day cycle]
             + A2*cos(2π*t/48) + B2*sin(2π*t/48)    [2-day cycle]
             + A3*cos(2π*t/24) + B3*sin(2π*t/24)    [1-day cycle]
             + A4*cos(2π*t/12) + B4*sin(2π*t/12)    [12-hour cycle]
             + noise ~ N(0, σ²)
    """
    # Fitted coefficients from the 3-week calibration (Dec 11 - Jan 1, 2025)
    mean_temp = 15.088131809841654
    c1 = -0.028682864132385803
    c2 = 0.00016003200839700564
    c3 = -2.0825281058118086e-07

    A1, B1 = -0.5199650882744151, 0.029175527599569095  # 168h (7-day) component
    A2, B2 = 0.05612667319884401, 0.3648083639887463  # 48h (2-day) component
    A3, B3 = -1.0619913279688276, -1.70253617350471  # 24h (1-day) component
    A4, B4 = 0.10430733439646522, 0.5382504761607834  # 12h (semi-daily) component

    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    # Parse start date
    if isinstance(start_date, str):
        start_date = pd.Timestamp(start_date, tz="UTC")
    elif start_date.tzinfo is None:
        start_date = start_date.tz_localize("UTC")

    # Convert freq_resolution to numeric for calculations
    freq_map = {
        "15min": 4,  # 4 records per hour
        "30min": 2,  # 2 records per hour
        "1H": 1,  # 1 record per hour
        "1D": 1 / 24,  # fractional records per hour
        "1h": 1,
        "h": 1,
    }

    records_per_hour = freq_map.get(freq_resolution, 4)
    total_hours = num_days * 24
    total_records = int(total_hours * records_per_hour)

    # Create time array
    time_index = pd.date_range(start=start_date, periods=total_records, freq=freq_resolution)
    t_hours = np.arange(total_records) / records_per_hour

    # Calculate temperature components
    trend = c1 * t_hours + c2 * t_hours**2 + c3 * t_hours**3

    # Fourier terms
    f_168h = A1 * np.cos(2 * np.pi * t_hours / 168) + B1 * np.sin(2 * np.pi * t_hours / 168)
    f_48h = A2 * np.cos(2 * np.pi * t_hours / 48) + B2 * np.sin(2 * np.pi * t_hours / 48)
    f_24h = A3 * np.cos(2 * np.pi * t_hours / 24) + B3 * np.sin(2 * np.pi * t_hours / 24)
    f_12h = A4 * np.cos(2 * np.pi * t_hours / 12) + B4 * np.sin(2 * np.pi * t_hours / 12)

    # Combine all components
    temperature = mean_temp + trend + f_168h + f_48h + f_24h + f_12h

    # Add realistic noise
    noise = np.random.normal(0, noise_std, len(temperature))
    temperature = temperature + noise

    # Return in requested format
    if return_format == "arrays":
        return time_index.to_numpy(), temperature, t_hours
    else:  # dataframe (default)
        return pd.DataFrame({"timestamp": time_index, "temperature": temperature})


if __name__ == "__main__":
    # Example usage
    print("=" * 70)
    print("FOURIER TEMPERATURE SIMULATOR")
    print("=" * 70)

    # Example 1: Default usage
    print("\n[Example 1] Default 14 days with noise")
    df1 = simulate_data()
    print(df1.head(10))
    print(f"Shape: {df1.shape}")
    print(
        f"Temperature range: {df1['temperature'].min():.2f}°C to {df1['temperature'].max():.2f}°C"
    )

    # Example 2: Custom parameters
    print("\n[Example 2] 30 days starting from 2025-02-01 with low noise")
    df2 = simulate_data(num_days=30, start_date="2025-02-01", noise_std=0.1, random_seed=42)
    print(f"Generated {len(df2)} records")
    print(f"Date range: {df2['timestamp'].min()} to {df2['timestamp'].max()}")

    # Example 3: Hourly resolution
    print("\n[Example 3] 7 days at hourly resolution")
    df3 = simulate_data(num_days=7, freq_resolution="1H", noise_std=0.2, random_seed=123)
    print(f"Generated {len(df3)} records (hourly)")
    print(f"Sample:\n{df3.head()}")

    # Example 4: Return as arrays
    print("\n[Example 4] Return as arrays for custom processing")
    timestamps, temps, hours = simulate_data(num_days=3, return_format="arrays", random_seed=999)
    print(f"Timestamps shape: {timestamps.shape}")
    print(f"Temperatures shape: {temps.shape}")
    print(f"Hours shape: {hours.shape}")
    print(f"First 5 values: {temps[:5]}")

    # Example 5: Reproducibility with seed
    print("\n[Example 5] Reproducibility with random_seed")
    df_a = simulate_data(num_days=2, random_seed=777, noise_std=0.5)
    df_b = simulate_data(num_days=2, random_seed=777, noise_std=0.5)
    print(
        f"Are both simulations identical? {(df_a['temperature'].values == df_b['temperature'].values).all()}"
    )

    print("\n" + "=" * 70)
    print("✓ All examples completed successfully!")
    print("=" * 70)
