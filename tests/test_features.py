"""Test script to compare different feature options (time features, residual)."""

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

# Feature combinations to test
FEATURE_CONFIGS = {
    "baseline": {"use_time_features": False, "include_residual": False},
    "time_features": {"use_time_features": True, "include_residual": False},
    "residual": {"use_time_features": False, "include_residual": True},
    "both": {"use_time_features": True, "include_residual": True},
}


def load_data(days: int = 90) -> pd.DataFrame:
    """Load temperature data (last N days)."""
    df = pd.read_csv(DATA_PATH)
    df['ds'] = pd.to_datetime(df['ds'], utc=True).dt.tz_convert(None)
    df = df[['ds', 'y']].copy()

    # Filter to last N days
    end_date = df['ds'].max()
    start_date = end_date - pd.Timedelta(days=days)
    df = df[df['ds'] >= start_date].reset_index(drop=True)

    return df


def test_feature_config(
    config_name: str,
    use_time_features: bool,
    include_residual: bool,
    df: pd.DataFrame,
    base_config_path: Path
) -> dict:
    """Train model with specified feature options and return metrics."""
    print(f"=" * 60)
    print(f"Testing: {config_name}")
    print(f"  use_time_features: {use_time_features}")
    print(f"  include_residual: {include_residual}")
    print(f"=" * 60)

    # Load base config
    config = NeuralProphetConfig.from_yaml(base_config_path)

    # Override feature options
    config = config.model_copy(update={'use_time_features': use_time_features})

    # Update decomposer config for include_residual
    decomposer_dict = config.decomposer.model_dump()
    decomposer_dict['include_residual'] = include_residual
    from rubin_oracle.config import DecomposerConfig
    new_decomposer = DecomposerConfig(**decomposer_dict)
    config = config.model_copy(update={'decomposer': new_decomposer})

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
        'config': config_name,
        'use_time_features': use_time_features,
        'include_residual': include_residual,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
    }


def main():
    print("Feature Options Comparison")
    print("=" * 60)
    print()

    config_path = CONFIGS_DIR / "neuralprophet_bandpass.yaml"

    # Load data
    df = load_data(days=90)
    print(f"Data: {len(df)} samples")
    print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
    print()

    # Test each feature configuration
    results = []
    for config_name, options in FEATURE_CONFIGS.items():
        try:
            result = test_feature_config(
                config_name,
                options['use_time_features'],
                options['include_residual'],
                df,
                config_path
            )
            results.append(result)
        except Exception as e:
            print(f"ERROR: {config_name} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'config': config_name,
                'use_time_features': options['use_time_features'],
                'include_residual': options['include_residual'],
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
        baseline_rmse = valid_results[valid_results['config'] == 'baseline']['train_rmse'].values[0]

        print()
        print(f"Best config: {valid_results.loc[best_idx, 'config']}")
        print(f"Best RMSE: {valid_results.loc[best_idx, 'train_rmse']:.4f}")

        print()
        print("Improvement vs baseline:")
        for _, row in valid_results.iterrows():
            improvement = (baseline_rmse - row['train_rmse']) / baseline_rmse * 100
            print(f"  {row['config']}: {improvement:+.1f}%")


if __name__ == "__main__":
    main()
