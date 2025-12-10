"""Example workflow for temperature forecasting with proper data handling.

This example demonstrates the correct approach for temperature data:
1. Compute mean temperature from tempMax and tempMin
2. Handle frequency conversion properly
3. Train forecasting models
"""

import pandas as pd
import numpy as np
from pathlib import Path

from rubin_oracle import ProphetForecaster, NeuralProphetForecaster
from rubin_oracle.config import ProphetConfig, NeuralProphetConfig
from rubin_oracle.utils import compute_temp_mean, prepare_regular_frequency


def generate_sample_temperature_data(n_days: int = 60, freq: str = '15min') -> pd.DataFrame:
    """Generate synthetic temperature data with tempMax and tempMin.

    Args:
        n_days: Number of days of data
        freq: Data frequency

    Returns:
        DataFrame with ds, tempMax, tempMin columns
    """
    # Create datetime index
    dates = pd.date_range(start='2024-01-01', periods=n_days * 96, freq=freq)  # 96 15-min periods per day

    # Generate synthetic temperature with daily cycle
    hours = (dates.hour + dates.minute / 60.0)

    # Daily temperature pattern (peaks in afternoon)
    base_temp = 15 + 8 * np.sin(2 * np.pi * (hours - 6) / 24)
    noise = np.random.randn(len(dates)) * 1.5

    # Create tempMax and tempMin
    temp_mean = base_temp + noise
    temp_range = 3 + np.random.randn(len(dates)) * 0.5  # Daily temperature range

    df = pd.DataFrame({
        'ds': dates,
        'tempMax': temp_mean + temp_range / 2,
        'tempMin': temp_mean - temp_range / 2,
    })

    return df


def example_temperature_workflow():
    """Complete workflow for temperature forecasting."""
    print("=" * 70)
    print("Temperature Forecasting Workflow")
    print("=" * 70)

    # Step 1: Generate or load temperature data with tempMax/tempMin
    print("\n1. Loading temperature data...")
    df_raw = generate_sample_temperature_data(n_days=60, freq='15min')
    print(f"   Raw data: {len(df_raw)} rows at 15-minute frequency")
    print(f"   Columns: {df_raw.columns.tolist()}")
    print(f"\n   Sample data:")
    print(df_raw.head())

    # Step 2: Compute mean temperature from tempMax and tempMin
    # IMPORTANT: Do NOT use simple interpolation for temperature aggregation!
    print("\n2. Computing temperature mean from tempMax and tempMin...")
    df_with_mean = compute_temp_mean(df_raw, temp_max_col='tempMax', temp_min_col='tempMin')
    print(f"   Added 'y' column: (tempMax + tempMin) / 2")
    print(f"\n   Sample with mean:")
    print(df_with_mean[['ds', 'tempMax', 'tempMin', 'y']].head())

    # Step 3: Prepare data for modeling (keep only ds and y)
    df_model = df_with_mean[['ds', 'y']].copy()

    # For this example, we'll use the data at its native frequency (15min)
    # If you need to change frequency, use prepare_regular_frequency() but
    # ONLY AFTER computing the mean temperature

    # Step 4: Split into train and recent data
    split_idx = int(len(df_model) * 0.9)
    train_df = df_model.iloc[:split_idx]
    recent_df = df_model.iloc[split_idx - 192:]  # Keep last 192 steps (48h at 15min)

    print(f"\n3. Data split:")
    print(f"   Training: {len(train_df)} observations")
    print(f"   Recent (for AR): {len(recent_df)} observations (48 hours)")

    # Step 5: Configure and train Prophet
    print("\n4. Training Prophet model...")
    prophet_config = ProphetConfig(
        freq='15min',  # Match the data frequency
        lag_days=192,  # 48 hours at 15min frequency
        n_forecast=96,  # 24 hours forecast
        daily_seasonality=True,
        weekly_seasonality=False,
    )

    prophet_model = ProphetForecaster(prophet_config)
    prophet_model.fit(train_df)
    print(f"   Model trained on {len(train_df)} observations")

    # Step 6: Generate Prophet predictions
    print("\n5. Generating Prophet forecasts (24 hours ahead)...")
    prophet_pred = prophet_model.predict(periods=96)  # 96 steps = 24 hours at 15min
    prophet_std = prophet_model.standardize_output(prophet_pred)

    print(f"   Forecast: {len(prophet_std)} steps ahead")
    print(f"\n   First 5 forecast steps:")
    print(prophet_std.head())

    # Step 7: Configure and train NeuralProphet (if available)
    try:
        print("\n6. Training NeuralProphet model...")
        neural_config = NeuralProphetConfig(
            freq='15min',
            lag_days=192,  # 48 hours at 15min
            n_forecast=96,  # 24 hours forecast
            daily_seasonality=True,
            epochs=10,  # Reduced for example
            ar_layers=[],  # Linear AR
        )

        neural_model = NeuralProphetForecaster(neural_config)
        neural_model.fit(train_df)
        print(f"   Model trained on {len(train_df)} observations")

        # Step 8: Generate NeuralProphet predictions
        print("\n7. Generating NeuralProphet forecasts (24 hours ahead)...")
        neural_pred = neural_model.predict(recent_df)
        neural_std = neural_model.standardize_output(neural_pred)

        print(f"   Forecast: {len(neural_std)} steps ahead")
        print(f"\n   First 5 forecast steps:")
        print(neural_std.head())

        # Step 9: Compare forecasts
        print("\n8. Comparing forecasts:")
        print(f"   Prophet mean forecast: {prophet_std['yhat'].mean():.2f}°C")
        print(f"   NeuralProphet mean forecast: {neural_std['yhat'].mean():.2f}°C")
        print(f"   Actual recent mean: {recent_df['y'].tail(96).mean():.2f}°C")

    except ImportError:
        print("\n⚠️  NeuralProphet not installed. Skipping NeuralProphet example.")
        print("   Install with: pip install neuralprophet")

    print("\n" + "=" * 70)
    print("Workflow complete!")
    print("=" * 70)

    print("\n💡 Key takeaways:")
    print("   1. Always compute temperature mean from tempMax/tempMin")
    print("   2. Do NOT use simple interpolation for temperature aggregation")
    print("   3. Set freq parameter to match your data frequency")
    print("   4. For 15-min data: use lag_days=192 for 48h, n_forecast=96 for 24h")
    print("   5. For hourly data: use lag_days=48 for 48h, n_forecast=24 for 24h")


if __name__ == "__main__":
    example_temperature_workflow()
