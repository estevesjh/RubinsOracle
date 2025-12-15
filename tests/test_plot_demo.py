"""Demo script to test Prophet and NeuralProphet plot functions."""

import warnings
warnings.filterwarnings("ignore")
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Patch torch.load for compatibility
import torch
_original_load = torch.load
def safe_load(*a, **k):
    if "weights_only" not in k:
        k["weights_only"] = False
    return _original_load(*a, **k)
torch.load = safe_load

from rubin_oracle.config import ProphetConfig, NeuralProphetConfig
from rubin_oracle.models.prophet import ProphetForecaster
from rubin_oracle.models.neural_prophet import NeuralProphetForecaster


def generate_test_data(n_days: int = 7, freq: str = '15min') -> pd.DataFrame:
    """Generate synthetic temperature data with multiple frequency components."""
    np.random.seed(42)

    # Samples per day
    spd = 96 if freq == '15min' else 24
    n_samples = n_days * spd

    # Time index
    ds = pd.date_range(start='2024-01-01', periods=n_samples, freq=freq)
    t = np.arange(n_samples) / spd  # Time in days

    # Generate signal with known frequency components
    # Sub-daily oscillation (period = 0.5 days = 12h)
    sub_daily = 2.0 * np.sin(2 * np.pi * t / 0.5)

    # Daily cycle (period = 1 day)
    daily = 5.0 * np.sin(2 * np.pi * t / 1.0 - np.pi / 4)

    # Weekly cycle (period = 7 days)
    weekly = 3.0 * np.sin(2 * np.pi * t / 7.0)

    # Base temperature + slight trend
    base = 15.0 + 0.05 * t

    # Combine with noise
    y = base + sub_daily + daily + weekly
    y += np.random.normal(0, 0.5, n_samples)

    df = pd.DataFrame({'ds': ds, 'y': y})
    return df


if __name__ == '__main__':

    print("=" * 70)
    print("PLOT DEMO: Prophet and NeuralProphet Forecasters")
    print("=" * 70)

    # Generate test data (7 days)
    df = generate_test_data(n_days=7, freq='15min')
    print(f"\nTest data: {len(df)} samples over {len(df)/96:.1f} days")
    print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")

    # Split into train/test
    df_train = df.iloc[:-96]  # Last 24h for testing
    df_test = df.iloc[-96:]
    print(f"Train: {len(df_train)} samples, Test: {len(df_test)} samples")

    # =========================================
    # 1. Prophet Forecaster
    # =========================================
    print("\n" + "-" * 50)
    print("1. Fitting ProphetForecaster...")
    print("-" * 50)

    prophet_config = ProphetConfig(
        lag_days=7,
        n_forecast=96,  # 24 hours at 15min
        freq='15min',
        daily_seasonality=True,
        weekly_seasonality=True,
    )
    prophet_model = ProphetForecaster(prophet_config)
    prophet_model.fit(df_train)

    # print(f"   Metrics: RMSE={prophet_model.metrics_['rmse']:.4f}, R²={prophet_model.metrics_['r2']:.4f}")

    # Create figure with 2 rows
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Plot Prophet (uses stored data, no df needed)
    prophet_model.plot(
        df_test=df_test,
        window_days=3,
        title="Prophet Forecaster",
        ax=axes[0],
    )

    # =========================================
    # 2. NeuralProphet Forecaster
    # =========================================
    print("\n" + "-" * 50)
    print("2. Fitting NeuralProphetForecaster...")
    print("-" * 50)

    np_config = NeuralProphetConfig(
        lag_days=96,  # n_lags = 96 samples = 1 day at 15min freq
        n_forecast=96,  # 24 hours at 15min
        freq='15min',  # Must match data frequency!
        epochs=12,
        learning_rate=0.01,
        batch_size=32,
        daily_seasonality=10,  # Fourier order for daily
        weekly_seasonality=5,  # Fourier order for weekly
        ar_reg=0.1,  # Light regularization
        trend_reg=0.1,
        n_changepoints=10,
    )
    np_model = NeuralProphetForecaster(np_config)
    np_model.fit(df_train)

    # if np_model.metrics_:
    #     print(f"   Metrics: RMSE={np_model.metrics_['rmse']:.4f}, R²={np_model.metrics_['r2']:.4f}")

    # Plot NeuralProphet (uses stored data, no df needed)
    np_model.plot(
        df_test=df_test,
        lead_time=24,
        window_days=3,
        title="NeuralProphet Forecaster",
        axs=axes[1],
    )

    plt.tight_layout()

    # Save figure
    output_path = 'outputs/test_plot_demo.png'
    os.makedirs('outputs', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n" + "=" * 70)
    print(f"Plot saved to: {output_path}")
    print("=" * 70)

    plt.show()

