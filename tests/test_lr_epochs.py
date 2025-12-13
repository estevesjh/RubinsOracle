"""Test script to compare bandpass model with different learning rate + epochs combinations."""

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

# Learning rates to test
LEARNING_RATES = [0.01]

# Epochs to test (100 was best in isolation, testing higher values)
EPOCHS = [50, 100, 150]


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


def test_lr_epochs(lr: float, epochs: int, df: pd.DataFrame, base_config_path: Path) -> dict:
    """Train model with specified lr and epochs, return metrics."""
    config_name = f"lr={lr}, epochs={epochs}"
    print(f"=" * 60)
    print(f"Testing: {config_name}")
    print(f"=" * 60)

    # Load base config and override
    config = NeuralProphetConfig.from_yaml(base_config_path)
    config = config.model_copy(update={
        'learning_rate': lr,
        'epochs': epochs,
    })

    print(f"learning_rate: {config.learning_rate}")
    print(f"epochs: {config.epochs}")

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
        'learning_rate': lr,
        'epochs': epochs,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
    }


def main():
    print("Bandpass Model - Learning Rate + Epochs Grid Search")
    print("=" * 60)
    print()

    config_path = CONFIGS_DIR / "neuralprophet_bandpass.yaml"

    # Load data
    df = load_data(days=45)
    print(f"Data: {len(df)} samples")
    print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
    print()

    # Grid search
    results = []
    total = len(LEARNING_RATES) * len(EPOCHS)
    current = 0

    for lr in LEARNING_RATES:
        for epochs in EPOCHS:
            current += 1
            print(f"[{current}/{total}]")
            try:
                result = test_lr_epochs(lr, epochs, df, config_path)
                results.append(result)
            except Exception as e:
                print(f"ERROR: lr={lr}, epochs={epochs} failed: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'learning_rate': lr,
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

    # Pivot table for easier reading
    print()
    print("RMSE by Learning Rate x Epochs:")
    pivot = results_df.pivot(index='learning_rate', columns='epochs', values='train_rmse')
    print(pivot.to_string())

    # Find best
    valid_results = results_df.dropna(subset=['train_rmse'])
    if len(valid_results) > 0:
        best_idx = valid_results['train_rmse'].idxmin()
        print()
        print(f"Best config: lr={valid_results.loc[best_idx, 'learning_rate']}, epochs={valid_results.loc[best_idx, 'epochs']}")
        print(f"Best RMSE: {valid_results.loc[best_idx, 'train_rmse']:.4f}")


if __name__ == "__main__":
    main()
