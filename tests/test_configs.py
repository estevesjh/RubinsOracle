"""Test script to compare NeuralProphet configs with different decomposers."""

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

CONFIGS = [
    "neuralprophet_default.yaml",
    "neuralprophet_bandpass.yaml",
    "neuralprophet_vmd.yaml",
]


def load_data(n_samples: int = 100000) -> pd.DataFrame:
    """Load temperature data."""
    df = pd.read_csv(DATA_PATH)
    # Parse datetime, keep local time, remove timezone info for NeuralProphet
    df['ds'] = pd.to_datetime(df['ds'])
    df = df[['ds', 'y']].copy()

    # Use subset for faster testing
    if n_samples and len(df) > n_samples:
        df = df.tail(n_samples).reset_index(drop=True)

    print(f"Data: {len(df)} samples")
    print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
    print()
    return df


def test_config(config_name: str, df: pd.DataFrame) -> dict:
    """Train model with config and return metrics from training."""
    print(f"=" * 60)
    print(f"Testing: {config_name}")
    print(f"=" * 60)

    # Load config
    config_path = CONFIGS_DIR / config_name
    config = NeuralProphetConfig.from_yaml(config_path)
    print(f"Decomposer: {config.decomposer.method}")

    # Use all data for training
    train_df = df.copy()
    print(f"Train samples: {len(train_df)}")

    # Create and train model
    model = NeuralProphetForecaster(config)
    metrics_df = model.fit(train_df, verbose=False)

    # Get final training metrics from returned DataFrame
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
        'config': config_name,
        'decomposer': config.decomposer.method,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
    }


def main():
    print("NeuralProphet Config Comparison")
    print("=" * 60)
    print()

    # Load data
    df = load_data(n_samples=10000)

    # Test each config
    results = []
    for config_name in CONFIGS:
        try:
            result = test_config(config_name, df)
            results.append(result)
        except Exception as e:
            print(f"ERROR: {config_name} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'config': config_name,
                'decomposer': 'error',
                'train_rmse': np.nan,
                'train_mae': np.nan,
            })

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # Best model
    valid_results = results_df.dropna(subset=['train_rmse'])
    if len(valid_results) > 0:
        best_idx = valid_results['train_rmse'].idxmin()
        print()
        print(f"Best model: {valid_results.loc[best_idx, 'config']}")
        print(f"Best RMSE: {valid_results.loc[best_idx, 'train_rmse']:.4f}")


if __name__ == "__main__":
    main()
