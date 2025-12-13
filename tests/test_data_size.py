"""Test script to compare bandpass model performance with different data sizes."""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from rubin_oracle import NeuralProphetConfig, NeuralProphetForecaster

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "temp2024.csv"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Date ranges to test (days back from end of dataset)
DATE_RANGES = {
    "3 days": 3,
    "10 days": 10,
    "30 days": 30,
    "3 months": 90,
    "6 months": 180,
    "1 year": None,  # Full dataset
}


def load_full_data() -> pd.DataFrame:
    """Load full temperature dataset."""
    df = pd.read_csv(DATA_PATH)
    df['ds'] = pd.to_datetime(df['ds'], utc=True).dt.tz_convert(None)
    df = df[['ds', 'y']].copy()
    return df


def filter_by_days(df: pd.DataFrame, days: int | None) -> pd.DataFrame:
    """Filter data to last N days."""
    if days is None:
        return df.copy()

    end_date = df['ds'].max()
    start_date = end_date - pd.Timedelta(days=days)
    filtered = df[df['ds'] >= start_date].copy().reset_index(drop=True)
    return filtered


def test_data_size(size_name: str, days: int | None, df_full: pd.DataFrame, config_path: Path) -> dict:
    """Train model with specified date range and return metrics."""
    print(f"=" * 60)
    print(f"Testing: {size_name}")
    print(f"=" * 60)

    # Filter data by date
    df = filter_by_days(df_full, days)
    print(f"Samples: {len(df)}")
    print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")

    # Load config
    config = NeuralProphetConfig.from_yaml(config_path)

    # Train model
    model = NeuralProphetForecaster(config)
    metrics_df = model.fit(df, verbose=False)

    # Get final metrics
    train_rmse = np.nan
    train_mae = np.nan

    if metrics_df is not None and isinstance(metrics_df, pd.DataFrame) and len(metrics_df) > 0:
        last_metrics = metrics_df.iloc[-1]
        train_rmse = float(last_metrics.get('RMSE', np.nan))
        train_mae = float(last_metrics.get('MAE', np.nan))

    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Training MAE: {train_mae:.4f}")
    print()

    return {
        'size': size_name,
        'n_samples': len(df),
        'train_rmse': train_rmse,
        'train_mae': train_mae,
    }


def main():
    print("Bandpass Model - Data Size Comparison")
    print("=" * 60)
    print()

    config_path = CONFIGS_DIR / "neuralprophet_bandpass.yaml"

    # Load full dataset once
    df_full = load_full_data()
    print(f"Full dataset: {len(df_full)} samples")
    print(f"Full date range: {df_full['ds'].min()} to {df_full['ds'].max()}")
    print()

    # Test each date range
    results = []
    for size_name, days in DATE_RANGES.items():
        try:
            result = test_data_size(size_name, days, df_full, config_path)
            results.append(result)
        except Exception as e:
            print(f"ERROR: {size_name} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'size': size_name,
                'n_samples': 0,
                'train_rmse': np.nan,
                'train_mae': np.nan,
            })

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # Calculate improvement from baseline (first entry)
    print()
    print(f"Improvement vs {results_df.iloc[0]['size']} baseline:")
    baseline_rmse = results_df.iloc[0]['train_rmse']
    for _, row in results_df.iterrows():
        improvement = (baseline_rmse - row['train_rmse']) / baseline_rmse * 100
        print(f"  {row['size']}: {improvement:+.1f}%")


if __name__ == "__main__":
    main()
