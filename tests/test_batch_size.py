"""Test NeuralProphet performance with different batch sizes."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from rubin_oracle.config import NeuralProphetConfig
from rubin_oracle.models.neural_prophet import NeuralProphetForecaster

warnings.filterwarnings("ignore")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "temp2024.csv"

# 1h data: 1 sample/hour, 24 samples/day
SAMPLES_PER_HOUR = 1
SAMPLES_PER_DAY = 24

# Batch sizes to test
BATCH_CONFIGS = {
    "16": 16,
    "32": 32,
    "64": 64,
    "128": 128,
    "256": 256,
    "512": 512,
    "1024": 1024,
}


def load_data(days: int = 30) -> pd.DataFrame:
    """Load temperature data (last N days), resampled to 1h."""
    df = pd.read_csv(DATA_PATH)
    df["ds"] = pd.to_datetime(df["ds"], utc=True).dt.tz_convert(None)
    df = df[["ds", "y"]].copy()

    # Resample to 1h frequency
    df = df.set_index("ds").resample("1h").mean().reset_index()

    # Filter to last N days
    end_date = df["ds"].max()
    start_date = end_date - pd.Timedelta(days=days)
    df = df[df["ds"] >= start_date].reset_index(drop=True)

    return df


def test_batch_size(
    batch_name: str, batch_size: int, df_train: pd.DataFrame, df_test: pd.DataFrame
) -> dict:
    """Train model with specified batch size and return metrics."""
    print("=" * 60)
    print(f"Testing: batch_size={batch_name}")
    print("=" * 60)

    config = NeuralProphetConfig(
        lag_days=SAMPLES_PER_DAY,  # 1 day lookback
        n_forecast=SAMPLES_PER_DAY,  # 1 day ahead
        freq="1h",
        epochs=100,
        batch_size=batch_size,
        learning_rate=None,  # Trigger LR finder
        lr_finder_candidates=[0.01, 0.03, 0.05, 0.1, 0.5],
        daily_seasonality=12,
        weekly_seasonality=False,
        yearly_seasonality=False,
        ar_reg=0.1,
        trend_reg=0.1,
    )

    print(f"batch_size: {config.batch_size}")

    # Train model
    model = NeuralProphetForecaster(config)
    model.fit(df_train)

    # Get test metrics
    metrics = model.compute_metrics(df_test)

    test_rmse = metrics["rmse"] if metrics else np.nan
    test_rmse_last = metrics.get("rmse_last_step", np.nan) if metrics else np.nan
    test_r2 = metrics["r2"] if metrics else np.nan

    print(f"Test RMSE: {test_rmse:.4f}, RMSE_last: {test_rmse_last:.4f}, R²: {test_r2:.4f}")
    print()

    return {
        "batch_size": batch_name,
        "test_rmse": test_rmse,
        "test_rmse_last": test_rmse_last,
        "test_r2": test_r2,
    }


def main():
    print("NeuralProphet - Batch Size Comparison")
    print("=" * 60)
    print()

    # Load real data (90 days)
    df = load_data(days=90)

    # Split into train/test (last 7 days for testing)
    test_samples = SAMPLES_PER_DAY * 7
    df_train = df.iloc[:-test_samples]
    df_test = df.iloc[-test_samples:]

    print(f"Data: {len(df)} samples ({len(df) / SAMPLES_PER_DAY:.1f} days)")
    print(f"Train: {len(df_train)} samples, Test: {len(df_test)} samples")
    print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
    print()

    # Test each batch size
    results = []
    for batch_name, batch_size in BATCH_CONFIGS.items():
        try:
            result = test_batch_size(batch_name, batch_size, df_train, df_test)
            results.append(result)
        except Exception as e:
            print(f"ERROR: batch_size={batch_name} failed: {e}")
            import traceback

            traceback.print_exc()
            results.append(
                {
                    "batch_size": batch_name,
                    "test_rmse": np.nan,
                    "test_rmse_last": np.nan,
                    "test_r2": np.nan,
                }
            )

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # Find best (by RMSE at last step)
    valid_results = results_df.dropna(subset=["test_rmse_last"])
    if len(valid_results) > 0:
        best_idx = valid_results["test_rmse_last"].idxmin()
        print()
        print(f"Best batch_size: {valid_results.loc[best_idx, 'batch_size']}")
        print(f"Best RMSE_last: {valid_results.loc[best_idx, 'test_rmse_last']:.4f}")


if __name__ == "__main__":
    main()
