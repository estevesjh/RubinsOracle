"""Test Prophet forecaster validation."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rubin_oracle import DailyRetraining, ProphetConfig, ProphetForecaster

logging.disable(logging.INFO)  # Suppress cmdstanpy INFO messages
warnings.filterwarnings("ignore")

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "temp2024.csv"
CONFIGS_DIR = PROJECT_ROOT / "configs"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "prophet_test"

# Test parameters
N_DAYS = 130
N_FORECASTS = 100

# Configs to compare
PROPHET_CONFIGS = [
    "prophet_default",  # baseline: sub_daily, daily, multi_day, weekly
    "prophet_no_subdaily",  # daily, multi_day, weekly
    "prophet_default_biweekly",  # sub_daily, daily, weekly, bi-weekly
    # 'prophet_daily_weekly',  # just daily + weekly
]

STEPS_PER_HOUR = 4


def load_data(n_days: int = N_DAYS) -> pd.DataFrame:
    """Load temperature dataset (last n_days)."""
    df = pd.read_csv(DATA_PATH)
    df["ds"] = pd.to_datetime(df["ds"], utc=True).dt.tz_convert("America/Santiago")
    df = df[["ds", "y"]].copy()
    df = df.drop_duplicates(subset="ds", keep="first").reset_index(drop=True)

    end_date = df["ds"].max()
    start_date = end_date - pd.Timedelta(days=n_days)
    df = df[df["ds"] >= start_date].reset_index(drop=True)

    return df


def generate_forecast_times(df: pd.DataFrame, n_forecasts: int) -> list[pd.Timestamp]:
    """Generate forecast times spread across validation period (last 5%)."""
    start_idx = int(len(df) * 0.95)
    start_date = df["ds"].iloc[start_idx]
    end_date = df["ds"].iloc[-97]

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

        metrics.append(
            {
                "step": step,
                "hours": step / STEPS_PER_HOUR,
                "rmse": np.sqrt((residuals**2).mean()),
                "mae": residuals.abs().mean(),
                "bias": residuals.mean(),
                "n": len(subset),
            }
        )

    return pd.DataFrame(metrics)


def run_validation(config_name: str, df: pd.DataFrame, forecast_times: list) -> pd.DataFrame:
    """Run validation for Prophet and return metrics by step."""
    print(f"\n{'=' * 60}")
    print(f"Running: {config_name}")
    print(f"{'=' * 60}")

    config_path = CONFIGS_DIR / f"{config_name}.yaml"
    config = ProphetConfig.from_yaml(config_path)

    print(f"  lag_days: {config.lag_days}")
    print(f"  n_forecast: {config.n_forecast}")
    print(f"  Custom seasonalities: {[s['name'] for s in (config.custom_seasonalities or [])]}")

    model = ProphetForecaster(config)

    results = model.validate(
        df=df,
        forecast_times=forecast_times,
        retrain_strategy=DailyRetraining(),
        force_retrain=True,
        save_forecasts=False,
        verbose=True,
    )

    metrics = compute_metrics_by_step(results)
    metrics["config"] = config_name

    return metrics


def plot_comparison(all_metrics: pd.DataFrame, output_dir: Path):
    """Plot RMSE and MAE vs lead time for all configs."""
    _fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["blue", "green", "orange", "red", "purple"]
    configs = all_metrics["config"].unique()

    # RMSE vs lead time
    ax = axes[0]
    for i, config in enumerate(configs):
        df = all_metrics[all_metrics["config"] == config]
        label = config.replace("prophet_", "")
        ax.plot(df["hours"], df["rmse"], "-", color=colors[i % len(colors)], lw=2, label=label)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="1°C target")
    ax.set_xlabel("Lead Time (hours)", fontsize=11)
    ax.set_ylabel("RMSE (°C)", fontsize=11)
    ax.set_title("RMSE vs Lead Time (Prophet configs)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 24)

    # MAE vs lead time
    ax = axes[1]
    for i, config in enumerate(configs):
        df = all_metrics[all_metrics["config"] == config]
        label = config.replace("prophet_", "")
        ax.plot(df["hours"], df["mae"], "-", color=colors[i % len(colors)], lw=2, label=label)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="1°C target")
    ax.set_xlabel("Lead Time (hours)", fontsize=11)
    ax.set_ylabel("MAE (°C)", fontsize=11)
    ax.set_title("MAE vs Lead Time (Prophet configs)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 24)

    plt.tight_layout()
    plt.savefig(output_dir / "prophet_seasonality_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to: {output_dir / 'prophet_seasonality_comparison.png'}")


def print_summary(all_metrics: pd.DataFrame):
    """Print summary metrics at key horizons for all configs."""
    horizons = {"4h": 16, "8h": 32, "12h": 48, "16h": 64, "20h": 80, "24h": 96}
    configs = all_metrics["config"].unique()

    print("\n" + "=" * 80)
    print("SUMMARY: Prophet RMSE at Key Horizons by Config")
    print("=" * 80)

    # Header
    header = f"{'Horizon':<10}"
    for config in configs:
        short_name = config.replace("prophet_", "")[:12]
        header += f"{short_name:<14}"
    print(header)
    print("-" * 80)

    # Each horizon
    for horizon_name, step in horizons.items():
        row = f"{horizon_name:<10}"
        for config in configs:
            cfg_metrics = all_metrics[
                (all_metrics["config"] == config) & (all_metrics["step"] == step)
            ]
            if len(cfg_metrics) > 0:
                rmse = cfg_metrics["rmse"].values[0]
                row += f"{rmse:<14.4f}"
            else:
                row += f"{'N/A':<14}"
        print(row)

    # Average
    print("-" * 80)
    row = f"{'Average':<10}"
    for config in configs:
        cfg_metrics = all_metrics[all_metrics["config"] == config]
        row += f"{cfg_metrics['rmse'].mean():<14.4f}"
    print(row)


def main():
    print("=" * 60)
    print("PROPHET SEASONALITY COMPARISON TEST")
    print("=" * 60)
    print("\nParameters:")
    print(f"  Data: last {N_DAYS} days")
    print(f"  Forecast times: {N_FORECASTS}")
    print("  Retraining: Daily")
    print(f"  Configs: {len(PROPHET_CONFIGS)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(N_DAYS)
    print(f"\nDataset: {len(df)} samples ({len(df) / 96:.1f} days)")
    print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")

    forecast_times = generate_forecast_times(df, N_FORECASTS)
    print(f"\nForecast times: {len(forecast_times)}")
    print(f"  First: {forecast_times[0]}")
    print(f"  Last: {forecast_times[-1]}")

    # Run validation for all configs
    all_metrics = []
    for config_name in PROPHET_CONFIGS:
        metrics = run_validation(config_name, df, forecast_times)
        metrics.to_csv(OUTPUT_DIR / f"{config_name}_metrics.csv", index=False)
        all_metrics.append(metrics)

    all_metrics = pd.concat(all_metrics, ignore_index=True)
    all_metrics.to_csv(OUTPUT_DIR / "prophet_all_metrics.csv", index=False)

    plot_comparison(all_metrics, OUTPUT_DIR)
    print_summary(all_metrics)

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
