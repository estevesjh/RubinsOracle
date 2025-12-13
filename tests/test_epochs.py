"""Test script to compare bandpass model performance with different epochs."""

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

# Epochs to test
EPOCH_CONFIGS = {
    "20 epochs": 20,
    "30 epochs": 30,
    "50 epochs": 50,
    "100 epochs": 100,
}


def load_data(days: int = 120) -> pd.DataFrame:
    """Load temperature data (last N days)."""
    df = pd.read_csv(DATA_PATH)
    df['ds'] = pd.to_datetime(df['ds'], utc=True).dt.tz_convert(None)
    df = df[['ds', 'y']].copy()

    # Filter to last N days
    end_date = df['ds'].max()
    start_date = end_date - pd.Timedelta(days=days)
    df = df[df['ds'] >= start_date].reset_index(drop=True)

    return df


def test_epochs(epoch_name: str, epochs: int, df: pd.DataFrame, base_config_path: Path) -> dict:
    """Train model with specified epochs and return metrics."""
    print(f"=" * 60)
    print(f"Testing: {epoch_name}")
    print(f"=" * 60)

    # Load base config and override epochs
    config = NeuralProphetConfig.from_yaml(base_config_path)
    config = config.model_copy(update={'epochs': epochs})

    print(f"epochs: {config.epochs}")
    print(f"lag_days: {config.lag_days}")

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
        'config': epoch_name,
        'epochs': epochs,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
    }


def main():
    print("Bandpass Model - Epochs Comparison")
    print("=" * 60)
    print()

    config_path = CONFIGS_DIR / "neuralprophet_bandpass.yaml"

    # Load data
    df = load_data(days=120)
    print(f"Data: {len(df)} samples")
    print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
    print()

    # Test each epoch configuration
    results = []
    for epoch_name, epochs in EPOCH_CONFIGS.items():
        try:
            result = test_epochs(epoch_name, epochs, df, config_path)
            results.append(result)
        except Exception as e:
            print(f"ERROR: {epoch_name} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'config': epoch_name,
                'epochs': epochs,
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
        print(f"Best config: {valid_results.loc[best_idx, 'config']}")
        print(f"Best RMSE: {valid_results.loc[best_idx, 'train_rmse']:.4f}")


if __name__ == "__main__":
    main()
