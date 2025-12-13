"""Test script to compare different bandpass period_pairs configurations using validation."""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from rubin_oracle import NeuralProphetConfig, NeuralProphetForecaster
from rubin_oracle.config import DecomposerConfig
from rubin_oracle.base import NoRetraining

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "temp2024.csv"
CONFIGS_DIR = PROJECT_ROOT / "configs"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "validation_period_pairs2"

# Period pair configurations to test
PERIOD_CONFIGS = {
    # "4-band": [
    #     (0.25, 0.75),     # Sub-daily (6-18 hours)
    #     (0.75, 3.0),      # Daily (18h-3 days)
    #     (3.0, 14.0),      # Weekly (3-14 days)
    #     (14.0, 120.0),    # Monthly to seasonal
    # ],
    # "4-band-daily": [
    #     (0.10, 0.65),     # Sub-daily (6-18 hours)
    #     (0.65, 1.1),      # Daily (18h-3 days)
    #     (1.1,  7.0),      # Weekly (3-14 days)
    #     (7.0, 120.0),     # Monthly to seasonal
    # ],
    "5-band-base": [
        (0.10, 0.65),     # Sub-daily (6-18 hours)
        (0.65, 1.1),      # Daily (18h-3 days)
        (1.1,  6.0),      # Weekly (3-14 days)
        (6.0, 12.0),      # Weekly to monthly
        (12.0, 120.0),    # Monthly to seasonal
    ],

    "5-band-daily": [
        (0.15, 0.55),     # Sub-daily (6-18 hours)
        (0.9, 1.1),      # Daily (18h-3 days)
        (1.1,  3.0),      # Weekly (3-14 days)
        (3.0, 12.0),      # Weekly to monthly
        (12.0, 120.0),    # Monthly to seasonal
    ],
    "8-band-daily": [
        (0.14, 0.18),     # Sub-daily (6-18 hours)
        (0.18, 0.23),      # Daily (18h-3 days)
        (0.23, 0.28),      # Weekly (3-14 days)
        (0.28, 0.55),      # Weekly to monthly
        (0.55, 1.05),
        (2.00, 3.0),    # Monthly to seasonal
        (3.0,  12.0),    # Monthly to seasonal
        (12.0, 120.0),    # Monthly to seasonal
    ],
}

def load_data(days: int = 120) -> pd.DataFrame:
    """Load temperature data (last N days)."""
    df = pd.read_csv(DATA_PATH)
    df['ds'] = pd.to_datetime(df['ds'], utc=True).dt.tz_convert("America/Santiago")
    df = df[['ds', 'y']].copy()

    # Filter to last N days
    end_date = df['ds'].max()
    start_date = end_date - pd.Timedelta(days=days)
    df = df[df['ds'] >= start_date].reset_index(drop=True)

    return df


def generate_forecast_times(
    start_date: str = '2024-12-01',
    end_date: str = '2024-12-08',
    n_forecasts: int = 100,
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


def test_period_config(
    config_name: str,
    period_pairs: list,
    df: pd.DataFrame,
    forecast_times: list[pd.Timestamp],
    base_config_path: Path,
) -> dict:
    """Train and validate model with specified period_pairs and return metrics."""
    print(f"=" * 70)
    print(f"Testing: {config_name}")
    print(f"  Bands: {len(period_pairs)}")
    for i, (low, high) in enumerate(period_pairs):
        print(f"    {i+1}. [{low:.2f}, {high:.2f}] days")
    print(f"=" * 70)

    # Load base config
    config = NeuralProphetConfig.from_yaml(base_config_path)

    # Override epochs to 50
    config = config.model_copy(update={'epochs': 70})

    # Update decomposer config with new period_pairs
    decomposer_dict = config.decomposer.model_dump()
    decomposer_dict['period_pairs'] = period_pairs
    new_decomposer = DecomposerConfig(**decomposer_dict)
    config = config.model_copy(update={'decomposer': new_decomposer})

    # Create forecaster
    model = NeuralProphetForecaster(config)

    # Run validation with NoRetraining (train once)
    print("Running validation with NoRetraining strategy...")
    print()

    output_path = OUTPUT_DIR / config_name.replace(" ", "_")
    output_path.mkdir(parents=True, exist_ok=True)

    results = model.validate(
        df=df,
        forecast_times=forecast_times,
        retrain_strategy=NoRetraining(),
        force_retrain=False,  # Force retrain instead of loading cached models
        save_forecasts=True,
        output_path=output_path,
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

    metrics = {
        'config': config_name,
        'n_bands': len(period_pairs),
    }

    if len(results) > 0:
        print(f"\nValidation Metrics by Horizon:")
        for name, step in horizons.items():
            res_col = f'res{step}'
            if res_col in results.columns:
                res = results[res_col].dropna()
                if len(res) > 0:
                    mae = res.abs().mean()
                    rmse = (res ** 2).mean() ** 0.5
                    metrics[f'mae_{name}'] = mae
                    metrics[f'rmse_{name}'] = rmse
                    print(f"  {name} (step {step}): MAE={mae:.4f}, RMSE={rmse:.4f}")
                else:
                    metrics[f'mae_{name}'] = np.nan
                    metrics[f'rmse_{name}'] = np.nan
            else:
                metrics[f'mae_{name}'] = np.nan
                metrics[f'rmse_{name}'] = np.nan
    print()

    metrics['results'] = results
    return metrics


def main():
    print("Bandpass Period Pairs Validation Comparison")
    print("=" * 70)
    print()

    config_path = CONFIGS_DIR / "neuralprophet_bandpass.yaml"

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data(days=365)
    print(f"Full dataset: {len(df)} samples")
    print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
    print()

    # Generate 100 forecast times over 1 week in December
    forecast_times = generate_forecast_times(
        start_date='2024-12-10',
        end_date='2024-12-24',
        n_forecasts=300,
    )
    print(f"Forecast times: {len(forecast_times)}")
    print(f"  First: {forecast_times[0]}")
    print(f"  Last: {forecast_times[-1]}")
    print()

    # Test each period configuration
    results = []
    for config_name, period_pairs in PERIOD_CONFIGS.items():
        try:
            result = test_period_config(
                config_name, period_pairs, df, forecast_times, config_path
            )
            results.append(result)
        except Exception as e:
            print(f"ERROR: {config_name} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'config': config_name,
                'n_bands': len(period_pairs),
                'mae_8h': np.nan, 'rmse_8h': np.nan,
                'mae_12h': np.nan, 'rmse_12h': np.nan,
                'mae_15h': np.nan, 'rmse_15h': np.nan,
                'mae_24h': np.nan, 'rmse_24h': np.nan,
                'results': None,
            })

    # Summary
    print("=" * 70)
    print("SUMMARY - RMSE by Horizon")
    print("=" * 70)
    summary_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != 'results'}
        for r in results
    ])

    # Display RMSE columns
    rmse_cols = ['config', 'n_bands', 'rmse_8h', 'rmse_12h', 'rmse_15h', 'rmse_24h']
    print(summary_df[rmse_cols].to_string(index=False))

    print()
    print("=" * 70)
    print("SUMMARY - MAE by Horizon")
    print("=" * 70)
    mae_cols = ['config', 'n_bands', 'mae_8h', 'mae_12h', 'mae_15h', 'mae_24h']
    print(summary_df[mae_cols].to_string(index=False))

    # Find best for each horizon
    print()
    print("=" * 70)
    print("BEST CONFIG BY HORIZON (RMSE)")
    print("=" * 70)
    for horizon in ['8h', '12h', '15h', '24h']:
        col = f'rmse_{horizon}'
        if col in summary_df.columns:
            valid = summary_df.dropna(subset=[col])
            if len(valid) > 0:
                best_idx = valid[col].idxmin()
                print(f"  {horizon}: {valid.loc[best_idx, 'config']} (RMSE={valid.loc[best_idx, col]:.4f})")

    print()
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
