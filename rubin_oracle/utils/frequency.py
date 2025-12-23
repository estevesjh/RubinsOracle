"""Frequency conversion utilities for Rubin's Oracle forecasters.

This module provides static methods for converting between frequency strings
and numeric values (samples per day/hour), and for converting time-based
parameters like lag_days and n_forecast to sample counts.
"""

import pandas as pd


class FrequencyConverter:
    """Static utility class for frequency conversion operations.

    This class handles conversion between:
    - Frequency strings (e.g., '15min', '1h', '4h', '1D')
    - Samples per day/hour
    - Days to samples (for lag_days, n_forecast parameters)
    - Data resampling/downsampling between different frequencies
    """

    # Frequency string to timedelta mapping
    _FREQ_TO_TIMEDELTA = {
        "1min": pd.Timedelta(minutes=1),
        "5min": pd.Timedelta(minutes=5),
        "10min": pd.Timedelta(minutes=10),
        "15min": pd.Timedelta(minutes=15),
        "30min": pd.Timedelta(minutes=30),
        "1h": pd.Timedelta(hours=1),
        "1H": pd.Timedelta(hours=1),
        "h": pd.Timedelta(hours=1),
        "4h": pd.Timedelta(hours=4),
        "4H": pd.Timedelta(hours=4),
        "6h": pd.Timedelta(hours=6),
        "6H": pd.Timedelta(hours=6),
        "12h": pd.Timedelta(hours=12),
        "12H": pd.Timedelta(hours=12),
        "1D": pd.Timedelta(days=1),
        "1d": pd.Timedelta(days=1),
    }

    @staticmethod
    def freq_to_timedelta(freq: str) -> pd.Timedelta:
        """Convert frequency string to pandas Timedelta.

        Args:
            freq: Frequency string (e.g., '15min', '1h', '1D')

        Returns:
            pandas Timedelta object

        Raises:
            ValueError: If frequency string is not recognized
        """
        if freq in FrequencyConverter._FREQ_TO_TIMEDELTA:
            return FrequencyConverter._FREQ_TO_TIMEDELTA[freq]

        try:
            return pd.to_timedelta(freq)
        except Exception as e:
            raise ValueError(f"Invalid frequency string: {freq}") from e

    @staticmethod
    def freq_to_samples_per_day(freq: str) -> float:
        """Convert frequency string to samples per day.

        Args:
            freq: Frequency string (e.g., '15min' → 96, '1h' → 24, '1D' → 1)

        Returns:
            Number of samples per day

        Raises:
            ValueError: If frequency is invalid or exceeds 1 day
        """
        td = FrequencyConverter.freq_to_timedelta(freq)
        samples_per_day = pd.Timedelta(days=1) / td
        return float(samples_per_day)

    @staticmethod
    def freq_to_samples_per_hour(freq: str) -> float:
        """Convert frequency string to samples per hour.

        Args:
            freq: Frequency string (e.g., '15min' → 4, '1h' → 1)

        Returns:
            Number of samples per hour

        Raises:
            ValueError: If frequency is invalid or exceeds 1 hour
        """
        td = FrequencyConverter.freq_to_timedelta(freq)
        samples_per_hour = pd.Timedelta(hours=1) / td
        return float(samples_per_hour)

    @staticmethod
    def days_to_samples(days: float, freq: str) -> int:
        """Convert days to number of samples for given frequency.

        Args:
            days: Number of days (e.g., 7 for 1 week)
            freq: Frequency string (e.g., '15min', '1h')

        Returns:
            Number of samples

        Example:
            >>> FrequencyConverter.days_to_samples(1, '15min')
            96
            >>> FrequencyConverter.days_to_samples(7, '1h')
            168
        """
        samples_per_day = FrequencyConverter.freq_to_samples_per_day(freq)
        return int(days * samples_per_day)

    @staticmethod
    def samples_to_days(samples: int, freq: str) -> float:
        """Convert number of samples to days for given frequency.

        Args:
            samples: Number of samples
            freq: Frequency string

        Returns:
            Number of days
        """
        samples_per_day = FrequencyConverter.freq_to_samples_per_day(freq)
        return samples / samples_per_day

    @staticmethod
    def hours_to_samples(hours: float, freq: str) -> int:
        """Convert hours to number of samples for given frequency.

        Args:
            hours: Number of hours
            freq: Frequency string

        Returns:
            Number of samples
        """
        samples_per_hour = FrequencyConverter.freq_to_samples_per_hour(freq)
        return int(hours * samples_per_hour)

    @staticmethod
    def samples_to_hours(samples: int, freq: str) -> float:
        """Convert number of samples to hours for given frequency.

        Args:
            samples: Number of samples
            freq: Frequency string

        Returns:
            Number of hours
        """
        samples_per_hour = FrequencyConverter.freq_to_samples_per_hour(freq)
        return samples / samples_per_hour

    @staticmethod
    def lead_time_to_step(lead_time: float, freq: str, n_forecast: int) -> int:
        """Convert lead time in hours to the corresponding step number.

        Args:
            lead_time: Forecast horizon in hours (e.g., 12.0, 24.0)
            freq: Frequency string (e.g., '15min', 'H')
            n_forecast: Maximum number of forecast steps

        Returns:
            Step number (e.g., 4 for yhat4)

        Raises:
            ValueError: If lead_time is invalid or exceeds n_forecast steps
        """
        step = FrequencyConverter.hours_to_samples(lead_time, freq)

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


class SampleConverter:
    """Caches frequency-based sample conversions for a given configuration.

    Avoids repeated computation of lag_samples and n_forecast_samples
    by caching values on initialization.

    Attributes:
        freq: Frequency string
        lag_days: Lag window in days
        n_forecast: Forecast horizon in days
        lag_samples: Cached lag in samples
        n_forecast_samples: Cached forecast horizon in samples
        freq_td: Cached frequency as timedelta
    """

    def __init__(self, freq: str, lag_days: float, n_forecast: float):
        """Initialize with frequency and time parameters.

        Args:
            freq: Frequency string (e.g., '15min', '1h')
            lag_days: Lag window in days
            n_forecast: Forecast horizon in days
        """
        self.freq = freq
        self.lag_days = lag_days
        self.n_forecast = n_forecast

        # Cache computed values
        self.freq_td = FrequencyConverter.freq_to_timedelta(freq)
        self.lag_samples = FrequencyConverter.days_to_samples(lag_days, freq)
        self.n_forecast_samples = FrequencyConverter.days_to_samples(n_forecast, freq)
        self.steps_per_day = FrequencyConverter.freq_to_samples_per_day(freq)

    def lead_time_to_step(self, lead_time: float) -> int:
        """Convert lead time in hours to step number using cached n_forecast_samples.

        Args:
            lead_time: Forecast horizon in hours

        Returns:
            Step number
        """
        return FrequencyConverter.lead_time_to_step(lead_time, self.freq, self.n_forecast_samples)
