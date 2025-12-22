"""Compare Savgol vs Bandpass decomposition configs."""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rubin_oracle import (
    BiWeeklyRetraining,
    NeuralProphetConfig,
    NeuralProphetForecaster,
)

warnings.filterwarnings("ignore")

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "temp2024.csv"
CONFIGS_DIR = PROJECT_ROOT / "configs"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "savgol_vs_bandpass"

# Test parameters
N_DAYS = 150
N_FORECASTS = 100
EPOCHS = 30

# 15-min data
STEPS_PER_HOUR = 4


def load_data(n_days: int = N_DAYS) -> pd.DataFrame:
    """Load temperature dataset (last n_days)."""
    df = pd.read_csv(DATA_PATH)
    df["ds"] = pd.to_datetime(df["ds"], utc=True).dt.tz_convert("America/Santiago")
    df = df[["ds", "y"]].copy()
    df = df.drop_duplicates(subset="ds", keep="first").reset_index(drop=True)

    # Get last n_days
    end_date = df["ds"].max()
    start_date = end_date - pd.Timedelta(days=n_days)
    df = df[df["ds"] >= start_date].reset_index(drop=True)

    return df


def generate_forecast_times(df: pd.DataFrame, n_forecasts: int) -> list[pd.Timestamp]:
    """Generate forecast times spread across the validation period."""
    # Use last 10% of data for validation
    start_idx = int(len(df) * 0.95)
    start_date = df["ds"].iloc[start_idx]
    end_date = df["ds"].iloc[-97]  # Leave room for 24h forecast

    return list(pd.date_range(start=start_date, end=end_date, periods=n_forecasts))


def compute_metrics_by_step(results: pd.DataFrame, max_step: int = 96) -> pd.DataFrame:
    """Compute RMSE and MAE by forecast step."""
    metrics = []

    for step in range(1, max_step + 1):
        yhat_col = f"yhat{step}"
        if yhat_col not in results.columns:
            continue

        subset = results[["y", yhat_col]].dropna()
        if len(subset) < 5:
            continue

        residuals = subset["y"] - subset[yhat_col]

        rmse = np.sqrt((residuals**2).mean())
        mae = residuals.abs().mean()
        bias = residuals.mean()

        metrics.append(
            {
                "step": step,
                "hours": step / STEPS_PER_HOUR,
                "rmse": rmse,
                "mae": mae,
                "bias": bias,
                "n": len(subset),
            }
        )

    return pd.DataFrame(metrics)


def run_validation(config_name: str, df: pd.DataFrame, forecast_times: list) -> pd.DataFrame:
    """Run validation for a config and return metrics by step."""
    print(f"\n{'=' * 60}")
    print(f"Running: {config_name}")
    print(f"{'=' * 60}")

    # Load config
    config_path = CONFIGS_DIR / f"{config_name}.yaml"
    config = NeuralProphetConfig.from_yaml(config_path)

    # Override epochs
    config = config.model_copy(update={"epochs": EPOCHS})

    print(f"  Epochs: {config.epochs}")
    print(f"  Decomposer: {config.decomposer.method}")
    if hasattr(config.decomposer, "filter_type"):
        print(f"  Filter type: {config.decomposer.filter_type}")
    if hasattr(config.decomposer, "period_pairs"):
        print(f"  Period pairs: {config.decomposer.period_pairs}")

    # Create forecaster and run validation
    model = NeuralProphetForecaster(config)

    results = model.validate(
        df=df,
        forecast_times=forecast_times,
        retrain_strategy=BiWeeklyRetraining(),
        force_retrain=True,
        save_forecasts=False,
        verbose=True,
    )

    # Compute metrics by step
    metrics = compute_metrics_by_step(results)
    metrics["config"] = config_name

    return metrics


def plot_comparison(metrics_dict: dict, output_dir: Path):
    """Plot RMSE comparison between configs."""
    _fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {
        "neuralprophet_savgol_base": "blue",
        "neuralprophet_bandpass": "orange",
        "neuralprophet_butter_savgol_periods": "green",
        "neuralprophet_optimized_bands": "red",
    }
    labels = {
        "neuralprophet_savgol_base": "Savgol (6 bands)",
        "neuralprophet_bandpass": "Butterworth (original)",
        "neuralprophet_butter_savgol_periods": "Butterworth (savgol periods)",
        "neuralprophet_optimized_bands": "Optimized (5 bands)",
    }

    # RMSE vs lead time
    ax = axes[0]
    for name, metrics in metrics_dict.items():
        ax.plot(
            metrics["hours"],
            metrics["rmse"],
            "o-",
            color=colors.get(name, "gray"),
            label=labels.get(name, name),
            lw=2,
            markersize=3,
            alpha=0.8,
        )

    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="1°C target")
    ax.set_xlabel("Lead Time (hours)", fontsize=11)
    ax.set_ylabel("RMSE (°C)", fontsize=11)
    ax.set_title(f"RMSE vs Lead Time ({EPOCHS} epochs, {N_FORECASTS} forecasts)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 24)

    # MAE vs lead time
    ax = axes[1]
    for name, metrics in metrics_dict.items():
        ax.plot(
            metrics["hours"],
            metrics["mae"],
            "o-",
            color=colors.get(name, "gray"),
            label=labels.get(name, name),
            lw=2,
            markersize=3,
            alpha=0.8,
        )

    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="1°C target")
    ax.set_xlabel("Lead Time (hours)", fontsize=11)
    ax.set_ylabel("MAE (°C)", fontsize=11)
    ax.set_title(f"MAE vs Lead Time ({EPOCHS} epochs, {N_FORECASTS} forecasts)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 24)

    plt.tight_layout()
    plt.savefig(output_dir / "rmse_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to: {output_dir / 'rmse_comparison.png'}")


def print_summary(metrics_dict: dict):
    """Print summary metrics at key horizons."""
    horizons = {"4h": 16, "8h": 32, "12h": 48, "16h": 64, "20h": 80, "24h": 96}

    print("\n" + "=" * 70)
    print("SUMMARY: RMSE at Key Horizons")
    print("=" * 70)

    header = f"{'Horizon':<10}"
    for name in metrics_dict:
        short_name = name.replace("neuralprophet_", "")
        header += f"{short_name:<20}"
    print(header)
    print("-" * 70)

    for horizon_name, step in horizons.items():
        row = f"{horizon_name:<10}"
        for _name, metrics in metrics_dict.items():
            step_metrics = metrics[metrics["step"] == step]
            if len(step_metrics) > 0:
                rmse = step_metrics["rmse"].values[0]
                row += f"{rmse:<20.4f}"
            else:
                row += f"{'N/A':<20}"
        print(row)

    # Compute average RMSE
    print("-" * 70)
    row = f"{'Average':<10}"
    for _name, metrics in metrics_dict.items():
        avg_rmse = metrics["rmse"].mean()
        row += f"{avg_rmse:<20.4f}"
    print(row)


def main():
    print("=" * 70)
    print("SAVGOL vs BANDPASS COMPARISON")
    print("=" * 70)
    print("\nParameters:")
    print(f"  Data: last {N_DAYS} days")
    print(f"  Forecast times: {N_FORECASTS}")
    print(f"  Epochs: {EPOCHS}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data(N_DAYS)
    print(f"\nDataset: {len(df)} samples ({len(df) / 96:.1f} days)")
    print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")

    # Generate forecast times
    forecast_times = generate_forecast_times(df, N_FORECASTS)
    print(f"\nForecast times: {len(forecast_times)}")
    print(f"  First: {forecast_times[0]}")
    print(f"  Last: {forecast_times[-1]}")

    # Run configs: savgol_base (6 bands) vs optimized (5 bands)
    configs = ["neuralprophet_savgol_base", "neuralprophet_optimized_bands"]
    metrics_dict = {}

    for config_name in configs:
        metrics = run_validation(config_name, df, forecast_times)
        metrics_dict[config_name] = metrics

        # Save metrics
        metrics.to_csv(OUTPUT_DIR / f"{config_name}_metrics.csv", index=False)

    # Plot comparison
    plot_comparison(metrics_dict, OUTPUT_DIR)

    # Print summary
    print_summary(metrics_dict)

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
