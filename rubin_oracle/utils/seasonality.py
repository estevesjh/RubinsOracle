"""Seasonality conversion utilities for Rubin's Oracle forecasters.

This module provides utilities for building model-specific seasonality configurations
from standardized inputs, including period conversion from days to samples for NeuralProphet.
"""

from rubin_oracle.utils.frequency import FrequencyConverter


class SeasonalityConverter:
    """Static utility class for seasonality configuration conversion.

    Handles building seasonality configs for both Prophet and NeuralProphet,
    including automatic period conversion from days to samples.
    """

    @staticmethod
    def build_prophet_seasonality(name: str, period_days: float, fourier_order: int) -> dict:
        """Build Prophet seasonality config.

        Prophet expects period in days (e.g., 1.0 for daily, 7.0 for weekly).

        Args:
            name: Seasonality name (e.g., 'daily', 'weekly')
            period_days: Period in days
            fourier_order: Fourier order for seasonality

        Returns:
            Dictionary with keys: name, period, fourier_order
        """
        return {
            "name": name,
            "period": period_days,
            "fourier_order": fourier_order,
        }

    @staticmethod
    def build_neuralprophet_seasonality(
        name: str, period_days: float, fourier_order: int, freq: str
    ) -> dict:
        """Build NeuralProphet seasonality config.

        NeuralProphet expects period in samples (based on frequency).
        This automatically converts period_days to the appropriate number of samples.

        Args:
            name: Seasonality name (e.g., 'daily', 'weekly')
            period_days: Period in days (e.g., 1.0 for daily, 7.0 for weekly)
            fourier_order: Fourier order for seasonality
            freq: Frequency string (e.g., '15min', '1h')

        Returns:
            Dictionary with keys: name, period, fourier_order
            where period is in samples for the given frequency
        """
        # Convert period from days to samples
        period_samples = FrequencyConverter.days_to_samples(period_days, freq)

        return {
            "name": name,
            "period": int(period_samples),
            "fourier_order": fourier_order,
        }

    @staticmethod
    def convert_seasonality_for_neuralprophet(
        seasonality_config: list[dict], freq: str
    ) -> list[dict]:
        """Convert a list of seasonality configs to NeuralProphet format.

        Takes seasonality configs with periods in days and converts them
        to NeuralProphet format with periods in samples.

        Args:
            seasonality_config: List of dicts with keys: name, period (in days), fourier_order
            freq: Frequency string (e.g., '15min', '1h')

        Returns:
            List of dicts with periods converted to samples for NeuralProphet
        """
        converted = []
        for season in seasonality_config:
            converted.append(
                SeasonalityConverter.build_neuralprophet_seasonality(
                    name=season["name"],
                    period_days=season["period"],
                    fourier_order=season.get("fourier_order", 3),
                    freq=freq,
                )
            )
        return converted
