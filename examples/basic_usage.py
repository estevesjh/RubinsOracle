"""Example usage of Rubin's Oracle forecasting package.

This script demonstrates how to use both Prophet and NeuralProphet
forecasters with the unified API.
"""

from pathlib import Path

import pandas as pd
import numpy as np

from rubin_oracle import ProphetForecaster, NeuralProphetForecaster
from rubin_oracle.config import ProphetConfig, NeuralProphetConfig


def generate_sample_data(n_points: int = 1000, freq: str = 'h') -> pd.DataFrame:
    """Generate sample time series data for demonstration.

    Args:
        n_points: Number of data points
        freq: Frequency of the time series

    Returns:
        DataFrame with 'ds' and 'y' columns
    """
    # Create datetime index
    dates = pd.date_range(start='2024-01-01', periods=n_points, freq=freq)

    # Generate synthetic temperature data with daily seasonality
    hours = np.arange(n_points) % 24
    daily_pattern = 15 + 5 * np.sin(2 * np.pi * hours / 24)  # Daily cycle
    noise = np.random.randn(n_points) * 2  # Random noise

    df = pd.DataFrame({
        'ds': dates,
        'y': daily_pattern + noise
    })

    return df


def example_prophet():
    """Example using ProphetForecaster."""
    print("=" * 60)
    print("Prophet Forecaster Example")
    print("=" * 60)

    # Generate sample data
    df = generate_sample_data(n_points=500)
    print(f"\nGenerated {len(df)} hourly observations")

    # Option 1: Create config from code
    config = ProphetConfig(
        lag_days=48,
        n_forecast=24,
        daily_seasonality=True,
        weekly_seasonality=False,
    )

    # Option 2: Load config from YAML (uncomment to use)
    # config = ProphetConfig.from_yaml("rubin_oracle/configs/prophet_default.yaml")

    # Create and fit forecaster
    forecaster = ProphetForecaster(config)
    print(f"\nFitting {forecaster.name} model...")
    forecaster.fit(df)

    # Generate predictions
    print("\nGenerating 24-hour forecast...")
    predictions = forecaster.predict(periods=24)

    # Standardize output
    standardized = forecaster.standardize_output(predictions)

    print("\nForecast (first 5 steps):")
    print(standardized.head())

    # Save model
    save_path = Path("models/prophet_example")
    forecaster.save(save_path)
    print(f"\nModel saved to {save_path}")

    # Load model
    loaded_forecaster = ProphetForecaster.load(save_path)
    print(f"Model loaded successfully: {loaded_forecaster.name}")

    return standardized


def example_neural_prophet():
    """Example using NeuralProphetForecaster."""
    print("\n" + "=" * 60)
    print("NeuralProphet Forecaster Example")
    print("=" * 60)

    # Generate sample data
    df = generate_sample_data(n_points=500)
    print(f"\nGenerated {len(df)} hourly observations")

    # Split into train and recent (for AR prediction)
    train_df = df.iloc[:-48]  # All but last 48 hours
    recent_df = df.iloc[-48:]  # Last 48 hours for AR prediction

    # Create config
    config = NeuralProphetConfig(
        lag_days=48,
        n_forecast=24,
        daily_seasonality=True,
        epochs=10,  # Reduced for example
        batch_size=64,
    )

    # Create and fit forecaster
    forecaster = NeuralProphetForecaster(config)
    print(f"\nFitting {forecaster.name} model...")
    forecaster.fit(train_df)

    # Generate predictions (requires recent data for AR)
    print("\nGenerating 24-hour forecast...")
    predictions = forecaster.predict(recent_df)

    # Standardize output
    standardized = forecaster.standardize_output(predictions)

    print("\nForecast (first 5 steps):")
    print(standardized.head())

    # Save model
    save_path = Path("models/neural_prophet_example")
    forecaster.save(save_path)
    print(f"\nModel saved to {save_path}")

    # Load model
    loaded_forecaster = NeuralProphetForecaster.load(save_path)
    print(f"Model loaded successfully: {loaded_forecaster.name}")

    return standardized


def compare_forecasters():
    """Compare Prophet and NeuralProphet forecasters."""
    print("\n" + "=" * 60)
    print("Comparing Forecasters")
    print("=" * 60)

    # Generate same data for both
    df = generate_sample_data(n_points=500)

    # Prophet
    prophet_config = ProphetConfig(n_forecast=24, daily_seasonality=True)
    prophet_model = ProphetForecaster(prophet_config)
    prophet_model.fit(df)
    prophet_pred = prophet_model.standardize_output(prophet_model.predict())

    # NeuralProphet
    neural_config = NeuralProphetConfig(
        lag_days=48,
        n_forecast=24,
        daily_seasonality=True,
        epochs=10,
    )
    neural_model = NeuralProphetForecaster(neural_config)
    neural_model.fit(df.iloc[:-48])
    neural_pred = neural_model.standardize_output(neural_model.predict(df.iloc[-48:]))

    print("\nProphet mean forecast:", prophet_pred['yhat'].mean())
    print("NeuralProphet mean forecast:", neural_pred['yhat'].mean())

    # Both outputs have the same schema!
    print("\nProphet output columns:", prophet_pred.columns.tolist())
    print("NeuralProphet output columns:", neural_pred.columns.tolist())


if __name__ == "__main__":
    # Run examples
    example_prophet()

    try:
        example_neural_prophet()
        compare_forecasters()
    except ImportError:
        print("\n⚠️  NeuralProphet not installed. Install with: pip install neuralprophet")

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
