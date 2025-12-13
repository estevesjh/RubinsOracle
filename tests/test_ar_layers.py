"""Test different AR layer configurations for NeuralProphet.

Compares:
- Baseline: ar_layers=[] (linear AR)
- Single layer: ar_layers=[24]
- Deep: ar_layers=[48, 24, 48]
"""

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
from pathlib import Path

from rubin_oracle import NeuralProphetConfig, NeuralProphetForecaster
from rubin_oracle.base import NoRetraining

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "temp2024.csv"
CONFIGS_DIR = PROJECT_ROOT / "configs"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ar_layer_test"

# AR layer configurations to test
AR_CONFIGS = {
    "linear": [],
    "shallow_24": [24],
    "deep_48_24_48": [48, 24, 48],
    "deep_24_12_24": [24, 12, 24],
}

# Number of forecast times for quick test
N_FORECASTS = 20


def load_data() -> pd.DataFrame:
    """Load temperature dataset."""
    df = pd.read_csv(DATA_PATH)
    df['ds'] = pd.to_datetime(df['ds'], utc=True).dt.tz_convert('America/Santiago')
    df = df[['ds', 'y']].drop_duplicates(subset='ds', keep='first').reset_index(drop=True)
    return df


def compute_metrics(results: pd.DataFrame) -> dict:
    """Compute metrics from validation results."""
    metrics = {}

    for step in [1, 4, 24, 48, 96]:
        yhat_col = f'yhat{step}'
        if yhat_col in results.columns:
            subset = results[['y', yhat_col]].dropna()
            if len(subset) > 0:
                residuals = subset['y'] - subset[yhat_col]
                metrics[f'rmse_{step}'] = np.sqrt((residuals ** 2).mean())
                metrics[f'mae_{step}'] = residuals.abs().mean()
                metrics[f'bias_{step}'] = residuals.mean()

    return metrics


def run_test(ar_layers: list, df: pd.DataFrame, base_config: NeuralProphetConfig) -> dict:
    """Run validation test with specific AR layers configuration."""

    # Create modified config
    config_dict = base_config.model_dump()
    config_dict['ar_layers'] = ar_layers
    config_dict['model_dir'] = None  # Don't use cached models
    config = NeuralProphetConfig(**config_dict)

    print(f"\n{'='*60}")
    print(f"Testing ar_layers={ar_layers}")
    print(f"{'='*60}")

    # Create forecaster
    model = NeuralProphetForecaster(config)

    # Generate forecast times
    forecast_times = pd.date_range(
        start='2024-12-01',
        end='2024-12-03',
        periods=N_FORECASTS
    ).tolist()

    # Run validation
    start_time = time.time()

    results = model.validate(
        df=df,
        forecast_times=forecast_times,
        retrain_strategy=NoRetraining(),
        save_forecasts=False,
        verbose=True,
        force_retrain=True,  # Always retrain, don't use cache
    )

    elapsed = time.time() - start_time

    # Compute metrics
    metrics = compute_metrics(results)
    metrics['train_time'] = elapsed
    metrics['n_forecasts'] = len(forecast_times)

    return metrics


def main():
    print("AR Layer Configuration Test")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data()
    print(f"Loaded {len(df)} samples")
    print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")

    # Load base config
    base_config = NeuralProphetConfig.from_yaml(CONFIGS_DIR / "neuralprophet_bandpass.yaml")
    print(f"\nBase config: {base_config.name}")
    print(f"  epochs: {base_config.epochs}")
    print(f"  lag_days: {base_config.lag_days}")
    print(f"  n_forecast: {base_config.n_forecast}")

    # Run tests
    all_results = {}

    for name, ar_layers in AR_CONFIGS.items():
        try:
            metrics = run_test(ar_layers, df, base_config)
            all_results[name] = metrics
        except Exception as e:
            print(f"Error testing {name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print comparison table
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)

    # Header
    steps = [1, 4, 24, 48, 96]
    header = f"{'Config':<20} {'Time':>8}"
    for step in steps:
        header += f" {'RMSE'+str(step):>8}"
    print(header)
    print("-" * 80)

    # Results
    for name, metrics in all_results.items():
        row = f"{name:<20} {metrics.get('train_time', 0):>7.1f}s"
        for step in steps:
            rmse = metrics.get(f'rmse_{step}', np.nan)
            row += f" {rmse:>8.4f}"
        print(row)

    print("-" * 80)

    # Save results
    results_df = pd.DataFrame(all_results).T
    results_df.to_csv(OUTPUT_DIR / "ar_layer_comparison.csv")
    print(f"\nResults saved to: {OUTPUT_DIR / 'ar_layer_comparison.csv'}")

    # Find best config for each step
    print("\nBest config by step:")
    for step in steps:
        col = f'rmse_{step}'
        if col in results_df.columns:
            best = results_df[col].idxmin()
            best_val = results_df.loc[best, col]
            print(f"  Step {step:2d} ({step/4:.1f}h): {best} (RMSE={best_val:.4f})")


if __name__ == "__main__":
    main()
