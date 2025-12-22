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
