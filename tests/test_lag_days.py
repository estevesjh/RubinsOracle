"""Test script to compare bandpass model performance with different lag_days."""

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

# 15-min data: 4 samples/hour, 96 samples/day
SAMPLES_PER_DAY = 96

# Lag configurations to test
LAG_CONFIGS = {
    "12 hours": 48,
    "1 day": 96,
    "2 days": 192,
    "3 days": 288,
    "1 week": 672,
    "2 weeks": 1344,
}


def load_data(days: int = 365) -> pd.DataFrame:
    """Load temperature data (last N days)."""
    df = pd.read_csv(DATA_PATH)
    df['ds'] = pd.to_datetime(df['ds'], utc=True).dt.tz_convert(None)
    df = df[['ds', 'y']].copy()

    # Filter to last N days
    end_date = df['ds'].max()
    start_date = end_date - pd.Timedelta(days=days)
    df = df[df['ds'] >= start_date].reset_index(drop=True)

    return df


def test_lag_days(lag_name: str, lag_days: int, df: pd.DataFrame, base_config_path: Path) -> dict:
    """Train model with specified lag_days and return metrics."""
    print(f"=" * 60)
    print(f"Testing: {lag_name} ({lag_days} samples)")
    print(f"=" * 60)

    # Load base config and override lag_days
    config = NeuralProphetConfig.from_yaml(base_config_path)
    config = config.model_copy(update={'lag_days': lag_days})

    print(f"lag_days: {config.lag_days}")
    print(f"n_forecast: {config.n_forecast}")

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
        'lag': lag_name,
        'lag_days': lag_days,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
    }


def main():
    print("Bandpass Model - Lag Days Comparison")
    print("=" * 60)
    print()

    config_path = CONFIGS_DIR / "neuralprophet_bandpass.yaml"

    # Load data (use 30 days for reasonable training time)
    df = load_data(days=120)
    print(f"Data: {len(df)} samples")
    print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
    print()

    # Test each lag configuration
    results = []
    for lag_name, lag_days in LAG_CONFIGS.items():
        try:
            result = test_lag_days(lag_name, lag_days, df, config_path)
            results.append(result)
        except Exception as e:
            print(f"ERROR: {lag_name} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'lag': lag_name,
                'lag_days': lag_days,
                'train_rmse': np.nan,
                'train_mae': np.nan,
            })

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # Find best
    valid_results = results_df.dropna(subset=['train_rmse'])
    if len(valid_results) > 0:
        best_idx = valid_results['train_rmse'].idxmin()
        print()
        print(f"Best lag: {valid_results.loc[best_idx, 'lag']}")
        print(f"Best RMSE: {valid_results.loc[best_idx, 'train_rmse']:.4f}")


if __name__ == "__main__":
    main()
