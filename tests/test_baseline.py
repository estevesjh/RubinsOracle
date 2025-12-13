"""Baseline validation test with bi-weekly retraining and 500 epochs."""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from rubin_oracle import NeuralProphetConfig, NeuralProphetForecaster
from rubin_oracle.config import DecomposerConfig

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "temp2024.csv"
CONFIGS_DIR = PROJECT_ROOT / "configs"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baseline"


class BiWeeklyRetraining:
    """Retrain model every two weeks."""

    def __init__(self):
        self.last_retrain_week = None

    def should_retrain(self, forecast_time: pd.Timestamp) -> bool:
        # Get the bi-weekly period (week number // 2)
        current_biweek = (forecast_time.year, forecast_time.isocalendar()[1] // 2)
        if self.last_retrain_week is None or current_biweek != self.last_retrain_week:
            self.last_retrain_week = current_biweek
            return True
        return False


# Best period pair configuration (4-band winner)
PERIOD_PAIRS = [
    (0.25, 0.75),     # Sub-daily (6-18 hours)
    (0.75, 3.0),      # Daily (18h-3 days)
    (3.0, 14.0),      # Weekly (3-14 days)
    (14.0, 120.0),    # Monthly to seasonal
]


def load_data() -> pd.DataFrame:
    """Load full temperature dataset in local time (America/Santiago)."""
    df = pd.read_csv(DATA_PATH)
    df['ds'] = pd.to_datetime(df['ds'], utc=True).dt.tz_convert('America/Santiago')
    df = df[['ds', 'y']].copy()
    # Drop duplicates from DST transitions (keep first occurrence)
    df = df.drop_duplicates(subset='ds', keep='first').reset_index(drop=True)
    return df


def generate_forecast_times(
    start_date: str,
    end_date: str,
    n_forecasts: int,
) -> list[pd.Timestamp]:
    """Generate list of forecast times."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    return list(pd.date_range(start=start, end=end, periods=n_forecasts))


def main():
    print("=" * 70)
    print("BASELINE VALIDATION")
    print("=" * 70)
    print()
    print("Configuration:")
    print("  - Epochs: 500")
    print("  - Retraining: Bi-weekly")
    print("  - Period: October to end of dataset")
    print("  - Band config: 4-band (best performer)")
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load base config
    config_path = CONFIGS_DIR / "neuralprophet_bandpass.yaml"
    config = NeuralProphetConfig.from_yaml(config_path)

    # Override epochs to 500
    config = config.model_copy(update={'epochs': 500})

    # Update decomposer config with winning period_pairs
    decomposer_dict = config.decomposer.model_dump()
    decomposer_dict['period_pairs'] = PERIOD_PAIRS
    new_decomposer = DecomposerConfig(**decomposer_dict)
    config = config.model_copy(update={'decomposer': new_decomposer})

    print(f"Config: {config.name}")
    print(f"  epochs: {config.epochs}")
    print(f"  lag_days: {config.lag_days}")
    print(f"  n_forecast: {config.n_forecast}")
    print(f"  decomposer: {config.decomposer.method}")
    print(f"  period_pairs: {PERIOD_PAIRS}")
    print()

    # Load data
    df = load_data()
    print(f"Full dataset: {len(df)} samples")
    print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
    print()

    # Generate forecast times from October 1st to end of dataset
    # Using 100 forecasts spread across ~3 months
    forecast_times = generate_forecast_times(
        start_date='2024-10-01',
        end_date='2024-12-31',
        n_forecasts=100,
    )
    print(f"Forecast times: {len(forecast_times)}")
    print(f"  First: {forecast_times[0]}")
    print(f"  Last: {forecast_times[-1]}")
    print()

    # Create forecaster
    model = NeuralProphetForecaster(config)

    # Run validation with BiWeeklyRetraining
    print("Running validation with BiWeeklyRetraining strategy...")
    print()

    results = model.validate(
        df=df,
        forecast_times=forecast_times,
        retrain_strategy=BiWeeklyRetraining(),
        force_retrain=True,
        save_forecasts=True,
        output_path=OUTPUT_DIR,
        verbose=True,
    )

    # Extract validation metrics at specific horizons
    # 15-min intervals: 8h=32, 12h=48, 15h=60, 24h=96
    horizons = {
        '8h': 32,
        '12h': 48,
        '15h': 60,
        '24h': 96,
    }

    if len(results) > 0:
        print()
        print("=" * 70)
        print("BASELINE RESULTS")
        print("=" * 70)
        print()
        print("Validation Metrics by Horizon:")
        for name, step in horizons.items():
            res_col = f'res{step}'
            if res_col in results.columns:
                res = results[res_col].dropna()
                if len(res) > 0:
                    mae = res.abs().mean()
                    rmse = (res ** 2).mean() ** 0.5
                    print(f"  {name} (step {step}): MAE={mae:.4f}, RMSE={rmse:.4f}")

        # Additional analysis: nighttime vs daytime
        print()
        print("-" * 50)
        print("Nighttime (8pm-6am) vs Daytime (6am-8pm):")
        print("-" * 50)

        results['hour'] = results['ds'].dt.hour
        night_mask = (results['hour'] >= 20) | (results['hour'] < 6)
        night_results = results[night_mask]
        day_results = results[~night_mask]

        print(f"\nNighttime samples: {len(night_results)}")
        for name, step in horizons.items():
            res_col = f'res{step}'
            if res_col in night_results.columns:
                res = night_results[res_col].dropna()
                if len(res) > 0:
                    mae = res.abs().mean()
                    rmse = (res ** 2).mean() ** 0.5
                    print(f"  {name}: MAE={mae:.4f}, RMSE={rmse:.4f}")

        print(f"\nDaytime samples: {len(day_results)}")
        for name, step in horizons.items():
            res_col = f'res{step}'
            if res_col in day_results.columns:
                res = day_results[res_col].dropna()
                if len(res) > 0:
                    mae = res.abs().mean()
                    rmse = (res ** 2).mean() ** 0.5
                    print(f"  {name}: MAE={mae:.4f}, RMSE={rmse:.4f}")

    print()
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
