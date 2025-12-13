"""Test script for ValidationMixin walk-forward validation."""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from rubin_oracle import NeuralProphetConfig, NeuralProphetForecaster
from rubin_oracle.base import NoRetraining, MonthlyRetraining, DailyRetraining

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "temp2024.csv"
CONFIGS_DIR = PROJECT_ROOT / "configs"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "validation"

# 15-min data: 4 samples/hour, 96 samples/day
SAMPLES_PER_DAY = 96


def load_data() -> pd.DataFrame:
    """Load full temperature dataset in local time (America/Santiago)."""
    df = pd.read_csv(DATA_PATH)
    df['ds'] = pd.to_datetime(df['ds'],utc=True).dt.tz_convert('America/Santiago')
    df = df[['ds', 'y']].copy()
    # Drop duplicates from DST transitions (keep first occurrence)
    df = df.drop_duplicates(subset='ds', keep='first').reset_index(drop=True)
    return df


def generate_forecast_times(
    start_date: str = '2024-12-01',
    end_date: str = '2024-12-08',
    n_forecasts: int = 40,
) -> list[pd.Timestamp]:
    """Generate list of forecast times.

    Args:
        start_date: Start date for forecasts (YYYY-MM-DD)
        end_date: End date for forecasts (YYYY-MM-DD)
        n_forecasts: Number of forecast times to generate

    Returns:
        List of forecast timestamps evenly spaced between start and end
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    return list(pd.date_range(start=start, end=end, periods=n_forecasts))


def main():
    print("ValidationMixin Test - Walk-Forward Validation")
    print("=" * 70)
    print()

    # Load data
    df = load_data()
    print(f"Full dataset: {len(df)} samples")
    print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
    print()

    # Load config (use bandpass as best performer)
    config_path = CONFIGS_DIR / "neuralprophet_bandpass.yaml"
    config = NeuralProphetConfig.from_yaml(config_path)

    # Override for faster testing (optional)
    # config = config.model_copy(update={'epochs': 10})

    print(f"Config: {config.name}")
    print(f"  lag_days: {config.lag_days}")
    print(f"  n_forecast: {config.n_forecast}")
    print(f"  decomposer: {config.decomposer.method}")
    print()

    # Generate 40 forecast times over 1 week in December
    forecast_times = generate_forecast_times(
        start_date='2024-12-01',
        end_date='2024-12-08',
        n_forecasts=1000,
    )
    print(f"Forecast times: {len(forecast_times)}")
    print(f"  First: {forecast_times[0]}")
    print(f"  Last: {forecast_times[-1]}")
    print()

    # Create forecaster
    model = NeuralProphetForecaster(config)

    # Run validation with NoRetraining (train once)
    print("Running validation with NoRetraining strategy...")
    print()

    results = model.validate(
        df=df,
        forecast_times=forecast_times,
        retrain_strategy=NoRetraining(),
        save_forecasts=True,
        output_path=OUTPUT_DIR,
        verbose=True,
    )

    # Additional analysis
    if len(results) > 0:
        print()
        print("=" * 70)
        print("Additional Analysis")
        print("=" * 70)

        # Filter for nighttime only (8pm - 6am)
        results['hour'] = results['ds'].dt.hour
        night_mask = (results['hour'] >= 20) | (results['hour'] < 6)
        night_results = results[night_mask]

        print(f"\nNighttime only (8pm-6am): {len(night_results)} samples")
        print("-" * 50)

        for step in [1, 4, 24, 48, 96]:  # 15min, 1hr, 6hr, 12hr, 1day
            res_col = f'res{step}'
            if res_col in night_results.columns:
                subset = night_results[res_col].dropna()
                if len(subset) > 0:
                    mae = subset.abs().mean()
                    rmse = (subset ** 2).mean() ** 0.5
                    hours = step * 0.25  # 15-min intervals
                    print(f"Step {step:3d} ({hours:5.1f}h): MAE={mae:.4f}, RMSE={rmse:.4f}, n={len(subset)}")

        # Also show daytime for comparison
        day_results = results[~night_mask]
        print(f"\nDaytime (6am-8pm): {len(day_results)} samples")
        print("-" * 50)

        for step in [1, 4, 24, 48, 96]:
            res_col = f'res{step}'
            if res_col in day_results.columns:
                subset = day_results[res_col].dropna()
                if len(subset) > 0:
                    mae = subset.abs().mean()
                    rmse = (subset ** 2).mean() ** 0.5
                    hours = step * 0.25
                    print(f"Step {step:3d} ({hours:5.1f}h): MAE={mae:.4f}, RMSE={rmse:.4f}, n={len(subset)}")

        print()
        print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
