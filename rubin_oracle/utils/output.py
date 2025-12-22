"""Output formatting utilities for Rubin's Oracle forecasters.

This module provides utilities to standardize outputs from different forecasters
to a common schema with columns: ds, yhat, yhat_lower, yhat_upper, step.
"""

import numpy as np
import pandas as pd


class OutputFormatter:
    """Static utility class for standardizing forecast outputs.

    Converts model-specific output formats to a standardized schema.
    All outputs are guaranteed to have:
    - ds: Target prediction timestamp
    - yhat: Point forecast
    - yhat_lower: Lower uncertainty bound
    - yhat_upper: Upper uncertainty bound
    - step: Forecast step (1, 2, 3, ...)
    """

    @staticmethod
    def standardize_prophet_output(df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Standardize Prophet output by adding step column.

        Prophet already outputs long format (one row per timestamp) with
        ds, yhat, yhat_lower, yhat_upper columns. This just adds the step column.

        Args:
            df: Prophet forecast output with columns ds, yhat, yhat_lower, yhat_upper
            freq: Frequency string (e.g., '15min', '1h')

        Returns:
            DataFrame with added 'step' column
        """
        if "ds" not in df.columns or "yhat" not in df.columns:
            raise ValueError("Prophet output must have 'ds' and 'yhat' columns")

        result = df.copy()

        # Add step column: enumerate forecast timestamps
        result["step"] = range(1, len(result) + 1)

        # Reorder columns
        cols = ["ds", "yhat", "yhat_lower", "yhat_upper", "step"]
        available_cols = [col for col in cols if col in result.columns]
        result = result[available_cols]

        return result

    @staticmethod
    def standardize_neuralprophet_output(
        df: pd.DataFrame, freq: str, n_forecast: int
    ) -> pd.DataFrame:
        """Convert NeuralProphet wide format to standardized long format.

        NeuralProphet outputs wide format with yhat1, yhat2, ..., yhat{n_forecast}
        columns. This converts to long format (one row per step per timestamp).

        Args:
            df: NeuralProphet forecast with columns ds, yhat1, yhat2, ..., yhatN
            freq: Frequency string (e.g., '15min', '1h')
            n_forecast: Number of forecast steps

        Returns:
            DataFrame in standardized long format with columns:
            ds, yhat, yhat_lower, yhat_upper, step
        """
        if "ds" not in df.columns:
            raise ValueError("NeuralProphet forecast must have 'ds' column")

        # Find all yhat columns
        yhat_cols = [col for col in df.columns if col.startswith("yhat") and col[4:].isdigit()]
        if not yhat_cols:
            raise ValueError("No yhat columns found in NeuralProphet forecast")

        # Build results list - one row per step per forecast timestamp
        results = []

        for step in range(1, n_forecast + 1):
            yhat_col = f"yhat{step}"
            yhat_lower_col = f"yhat_lower{step}"
            yhat_upper_col = f"yhat_upper{step}"

            # Skip if this yhat column doesn't exist
            if yhat_col not in df.columns:
                continue

            # Extract this step's forecasts for all timestamps
            for _, row in df.iterrows():
                yhat_val = row[yhat_col]

                # Skip NaN predictions
                if pd.isna(yhat_val):
                    continue

                # Calculate target timestamp
                forecast_ts = row["ds"]
                freq_td = pd.to_timedelta(freq)
                target_ts = forecast_ts + (step * freq_td)

                # Extract uncertainty bounds if available
                yhat_lower = row[yhat_lower_col] if yhat_lower_col in df.columns else np.nan
                yhat_upper = row[yhat_upper_col] if yhat_upper_col in df.columns else np.nan

                results.append(
                    {
                        "ds": target_ts,
                        "yhat": yhat_val,
                        "yhat_lower": yhat_lower,
                        "yhat_upper": yhat_upper,
                        "step": step,
                    }
                )

        if not results:
            raise ValueError("No valid predictions found in NeuralProphet forecast")

        result_df = pd.DataFrame(results)

        # Reorder columns
        cols = ["ds", "yhat", "yhat_lower", "yhat_upper", "step"]
        result_df = result_df[cols]

        return result_df

    @staticmethod
    def standardize_ensemble_output(df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Standardize Ensemble output by adding step column.

        Ensemble outputs similar to Prophet (already in long format).
        This just adds the step column.

        Args:
            df: Ensemble forecast output with columns ds, yhat, yhat_lower, yhat_upper
            freq: Frequency string (e.g., '15min', '1h')

        Returns:
            DataFrame with added 'step' column
        """
        # Same as Prophet since Ensemble also outputs long format
        return OutputFormatter.standardize_prophet_output(df, freq)
