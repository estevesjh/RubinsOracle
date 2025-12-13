"""Plot validation results from forecast CSV."""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_PATH = PROJECT_ROOT / "outputs" / "validation" / "validation_forecasts.csv"
OUTPUT_DIR = Path(__file__).parent / "figures"

# 15-min data
STEPS_PER_HOUR = 4
STEPS_PER_DAY = 96

# Outlier threshold (°C)
OUTLIER_THRESHOLD = 5.0


def load_results(usecols=None):
    """Load validation results (memory efficient)."""
    print(f"Loading results from {RESULTS_PATH}...")
    df = pd.read_csv(RESULTS_PATH, usecols=usecols, low_memory=True)
    df['ds'] = pd.to_datetime(df['ds'])
    if 'forecast_time' in df.columns:
        df['forecast_time'] = pd.to_datetime(df['forecast_time'], format='mixed')
    print(f"Loaded {len(df):,} rows")
    return df


def compute_metrics_by_step(df, steps=None, outlier_threshold=OUTLIER_THRESHOLD):
    """
    Compute comprehensive metrics by forecast step.

    Returns DataFrame with columns:
        step, hours, N, bias, rmse, mae, std,
        p68, p87, p91, p95, outlier_frac, laplace_b
    """
    if steps is None:
        steps = list(range(1, 97))

    metrics = []
    for step in steps:
        yhat_col = f'yhat{step}'
        if yhat_col not in df.columns:
            continue

        subset = df[['y', yhat_col]].dropna()
        if len(subset) < 10:
            continue

        residuals = subset['y'] - subset[yhat_col]
        abs_residuals = residuals.abs()

        # Filter outliers for main metrics
        mask = abs_residuals <= outlier_threshold
        residuals_filtered = residuals[mask]

        if len(residuals_filtered) < 10:
            continue

        # Basic statistics (on filtered data)
        n = len(residuals_filtered)
        n_total = len(residuals)
        bias = residuals_filtered.mean()
        rmse = np.sqrt((residuals_filtered ** 2).mean())
        mae = residuals_filtered.abs().mean()
        std = residuals_filtered.std()

        # Percentiles (on filtered data)
        abs_res_filtered = residuals_filtered.abs()
        p68 = abs_res_filtered.quantile(0.68)
        p87 = abs_res_filtered.quantile(0.87)
        p91 = abs_res_filtered.quantile(0.91)
        p95 = abs_res_filtered.quantile(0.95)

        # Outlier fraction (on full data)
        outlier_frac = (abs_residuals > outlier_threshold).sum() / len(abs_residuals)

        #

        # Laplace fit (on filtered data)
        try:
            loc, scale = stats.laplace.fit(residuals_filtered)
            laplace_b = scale
        except:
            laplace_b = np.nan

        hours = step / STEPS_PER_HOUR
        metrics.append({
            'step': step,
            'hours': hours,
            'N': n,
            'N_total': n_total,
            'bias': bias,
            'rmse': rmse,
            'mae': mae,
            'std': std,
            'laplace_b': laplace_b,
            'p68': p68,
            'p87': p87,
            'p91': p91,
            'p95': p95,
            'outlier_frac': outlier_frac,
        })

    return pd.DataFrame(metrics)


def plot_validation_metrics(metrics, output_dir):
    """
    Generate validation_metrics.png with 4 subplots:
    - RMSE vs lead time
    - Bias vs lead time
    - Percentiles vs lead time
    - Outlier fraction vs lead time
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    hours = metrics['hours']

    # RMSE vs lead time
    ax = axes[0, 0]
    ax.plot(hours, metrics['rmse'], 'o-', color='blue', lw=2, markersize=3)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='1°C target')
    ax.axhline(y=2.0, color='orange', linestyle='--', alpha=0.7, label='2°C threshold')
    ax.set_xlabel('Lead Time (hours)', fontsize=11)
    ax.set_ylabel('RMSE (°C)', fontsize=11)
    ax.set_title('Forecast RMSE vs Lead Time', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 24)

    # Bias vs lead time
    ax = axes[0, 1]
    ax.plot(hours, metrics['bias'], 'o-', color='green', lw=2, markersize=3)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.fill_between(hours, -0.5, 0.5, alpha=0.15, color='green', label='±0.5°C')
    ax.set_xlabel('Lead Time (hours)', fontsize=11)
    ax.set_ylabel('Bias (°C)', fontsize=11)
    ax.set_title('Forecast Bias vs Lead Time', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 24)

    # Percentiles vs lead time
    ax = axes[1, 0]
    ax.plot(hours, metrics['p68'], 'o-', label='68%', lw=2, markersize=3)
    ax.plot(hours, metrics['p87'], 's-', label='87%', lw=2, markersize=3)
    ax.plot(hours, metrics['p95'], '^-', label='95%', lw=2, markersize=3)
    ax.set_xlabel('Lead Time (hours)', fontsize=11)
    ax.set_ylabel('Absolute Error Percentile (°C)', fontsize=11)
    ax.set_title('Error Distribution vs Lead Time', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 24)

    # Outlier fraction vs lead time
    ax = axes[1, 1]
    ax.plot(hours, metrics['outlier_frac'] * 100, 'o-', color='red', lw=2, markersize=3)
    ax.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='5% threshold')
    ax.set_xlabel('Lead Time (hours)', fontsize=11)
    ax.set_ylabel(f'Outlier Fraction (%) [|res|>{OUTLIER_THRESHOLD}°C]', fontsize=11)
    ax.set_title('Outliers vs Lead Time', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 24)

    plt.tight_layout()
    plt.savefig(output_dir / 'validation_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: validation_metrics.png")


def plot_metrics_by_horizon(df, output_dir):
    """Plot RMSE and MAE vs forecast horizon."""
    metrics = compute_metrics_by_step(df)

    if len(metrics) == 0:
        print("No metrics to plot")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # RMSE
    ax = axes[0]
    ax.plot(metrics['hours'], metrics['rmse'], 'b-', linewidth=2)
    ax.set_xlabel('Forecast Horizon (hours)', fontsize=12)
    ax.set_ylabel('RMSE (°C)', fontsize=12)
    ax.set_title('RMSE vs Forecast Horizon', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axvline(x=6, color='r', linestyle='--', alpha=0.5, label='6h')
    ax.axvline(x=12, color='g', linestyle='--', alpha=0.5, label='12h')
    ax.axvline(x=24, color='orange', linestyle='--', alpha=0.5, label='24h')
    ax.legend()

    # MAE
    ax = axes[1]
    ax.plot(metrics['hours'], metrics['mae'], 'g-', linewidth=2)
    ax.set_xlabel('Forecast Horizon (hours)', fontsize=12)
    ax.set_ylabel('MAE (°C)', fontsize=12)
    ax.set_title('MAE vs Forecast Horizon', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.axvline(x=6, color='r', linestyle='--', alpha=0.5, label='6h')
    ax.axvline(x=12, color='g', linestyle='--', alpha=0.5, label='12h')
    ax.axvline(x=24, color='orange', linestyle='--', alpha=0.5, label='24h')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_by_horizon.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: metrics_by_horizon.png")

    return metrics


def plot_metrics_by_hour_of_day(df, output_dir, step=24):
    """Plot error by hour of day for a specific forecast step."""
    yhat_col = f'yhat{step}'
    if yhat_col not in df.columns:
        print(f"Column {yhat_col} not found")
        return

    df_plot = df[['ds', 'y', yhat_col]].dropna().copy()
    df_plot['residual'] = df_plot['y'] - df_plot[yhat_col]

    # Filter outliers
    df_plot = df_plot[df_plot['residual'].abs() <= OUTLIER_THRESHOLD]
    df_plot['hour'] = df_plot['ds'].dt.hour

    # Group by hour
    hourly_metrics = df_plot.groupby('hour')['residual'].agg([
        ('mae', lambda x: x.abs().mean()),
        ('rmse', lambda x: np.sqrt((x ** 2).mean())),
        ('bias', 'mean'),
        ('count', 'count'),
    ]).reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))

    x = hourly_metrics['hour']
    ax.bar(x - 0.2, hourly_metrics['rmse'], width=0.4, label='RMSE', alpha=0.8)
    ax.bar(x + 0.2, hourly_metrics['mae'], width=0.4, label='MAE', alpha=0.8)

    ax.set_xlabel('Hour of Day (Local Time)', fontsize=12)
    ax.set_ylabel('Error (°C)', fontsize=12)
    ax.set_title(f'Forecast Error by Hour of Day (Step {step} = {step/4:.1f}h ahead)', fontsize=14)
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Highlight night hours
    ax.axvspan(20, 24, alpha=0.1, color='blue', label='Night')
    ax.axvspan(0, 6, alpha=0.1, color='blue')

    plt.tight_layout()
    plt.savefig(output_dir / f'metrics_by_hour_step{step}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: metrics_by_hour_step{step}.png")


def plot_residual_distribution(df, output_dir, steps=[1, 24, 48, 96]):
    """Plot residual distribution with Laplace fit for selected steps."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, step in enumerate(steps):
        yhat_col = f'yhat{step}'
        if yhat_col not in df.columns:
            continue

        subset = df[['y', yhat_col]].dropna()
        residuals = subset['y'] - subset[yhat_col]

        # Filter outliers for plotting
        residuals_filtered = residuals[residuals.abs() <= OUTLIER_THRESHOLD]

        ax = axes[i]
        ax.hist(residuals_filtered, bins=50, density=True, alpha=0.7,
                edgecolor='black', color='steelblue')

        # Fit and plot Laplace distribution
        if len(residuals_filtered) > 10:
            try:
                loc, scale = stats.laplace.fit(residuals_filtered)
                x = np.linspace(residuals_filtered.min(), residuals_filtered.max(), 100)
                ax.plot(x, stats.laplace.pdf(x, loc, scale), 'r-', lw=2,
                       label=f'Laplace\nμ={loc:.2f}, b={scale:.2f}')
            except:
                pass

        # Add stats
        mean = residuals_filtered.mean()
        std = residuals_filtered.std()
        ax.axvline(mean, color='orange', linestyle='--', lw=2, label=f'Bias: {mean:.2f}')

        hours = step / STEPS_PER_HOUR
        n_outliers = (residuals.abs() > OUTLIER_THRESHOLD).sum()
        ax.set_xlabel('Residual (°C)', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'Step {step} ({hours:.1f}h) - N={len(residuals_filtered):,} (outliers={n_outliers})', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Residual Distribution (|res| ≤ {OUTLIER_THRESHOLD}°C)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'residual_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: residual_distribution.png")


def plot_sample_forecasts(df, output_dir, n_samples=3):
    """Plot sample forecast trajectories vs actuals."""
    # Get unique forecast times
    forecast_times = df['forecast_time'].dropna().unique()

    if len(forecast_times) < n_samples:
        n_samples = len(forecast_times)

    # Sample random forecast times
    np.random.seed(42)
    sample_times = np.random.choice(forecast_times, n_samples, replace=False)
    sample_times = sorted(sample_times)

    fig, axes = plt.subplots(n_samples, 1, figsize=(14, 4 * n_samples))
    if n_samples == 1:
        axes = [axes]

    yhat_cols = [f'yhat{i}' for i in range(1, 97)]
    lower_cols = [f'yhat{i} 16.0%' for i in range(1, 97)]
    upper_cols = [f'yhat{i} 84.0%' for i in range(1, 97)]

    for ax, ft in zip(axes, sample_times):
        # Get data for this forecast time
        mask = df['forecast_time'] == ft
        df_fc = df[mask].sort_values('ds')

        if len(df_fc) == 0:
            continue

        # Get actual values
        actuals = df_fc['y'].values
        times = df_fc['ds'].values

        # Get first row's forecast (yhat1 to yhat96)
        first_row = df_fc.iloc[0]
        yhats = [first_row.get(c, np.nan) for c in yhat_cols]
        lowers = [first_row.get(c, np.nan) for c in lower_cols]
        uppers = [first_row.get(c, np.nan) for c in upper_cols]

        # Create forecast times
        fc_times = times[:len(yhats)]

        # Plot
        ax.plot(times, actuals, 'b-', linewidth=2, label='Actual', alpha=0.8)
        ax.plot(fc_times, yhats, 'r--', linewidth=2, label='Forecast', alpha=0.8)
        ax.fill_between(fc_times, lowers, uppers, color='red', alpha=0.2, label='68% CI')

        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Temperature (°C)', fontsize=11)
        ax.set_title(f'Forecast issued at {pd.Timestamp(ft)}', fontsize=12)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'sample_forecasts.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: sample_forecasts.png")


def plot_error_over_time(df, output_dir, step=24):
    """Plot rolling error over time."""
    yhat_col = f'yhat{step}'
    if yhat_col not in df.columns:
        print(f"Column {yhat_col} not found")
        return

    df_plot = df[['ds', 'y', yhat_col]].dropna().copy()
    df_plot['residual'] = df_plot['y'] - df_plot[yhat_col]

    # Filter outliers
    df_plot = df_plot[df_plot['residual'].abs() <= OUTLIER_THRESHOLD]
    df_plot['abs_error'] = df_plot['residual'].abs()
    df_plot = df_plot.sort_values('ds')

    # Daily rolling average
    df_plot = df_plot.set_index('ds')
    rolling = df_plot['abs_error'].rolling('1D').mean()

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(rolling.index, rolling.values, 'b-', linewidth=1.5)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Rolling MAE (°C)', fontsize=12)
    ax.set_title(f'Daily Rolling MAE - Step {step} ({step/4:.1f}h ahead)', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'error_over_time_step{step}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: error_over_time_step{step}.png")


def print_summary_table(metrics):
    """Print a formatted summary table (similar to paper Table 1)."""
    print("\n" + "=" * 90)
    print("FORECAST VALIDATION SUMMARY")
    print("=" * 90)
    print(f"{'Step':>6} {'Hours':>6} {'N':>8} {'Bias':>8} {'RMSE':>8} {'MAE':>8} "
          f"{'68%':>8} {'95%':>8} {'Outlier%':>10}")
    print("-" * 90)

    key_steps = [1, 4, 8, 16, 24, 48, 96]  # 15min, 1h, 2h, 4h, 6h, 12h, 24h

    for step in key_steps:
        row = metrics[metrics['step'] == step]
        if len(row) > 0:
            r = row.iloc[0]
            print(f"{r['step']:>6.0f} {r['hours']:>6.1f}h {r['N']:>8.0f} "
                  f"{r['bias']:>8.3f} {r['rmse']:>8.3f} {r['mae']:>8.3f} "
                  f"{r['p68']:>8.3f} {r['p95']:>8.3f} "
                  f"{r['outlier_frac']*100:>9.1f}%")

    print("=" * 90)
    print(f"Outlier threshold: |residual| > {OUTLIER_THRESHOLD}°C")


def main():
    print("Validation Results Plotter")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_results()
    # mask the first day
    mask = (df['ds'].dt.day > 1) & (df['ds'].dt.day < 21)
    df = df[mask].copy()
    
    # Generate plots
    print("\nGenerating plots...")

    # 1. Compute comprehensive metrics
    metrics = compute_metrics_by_step(df)

    if len(metrics) == 0:
        print("No metrics computed!")
        return

    # Save metrics to CSV
    metrics.to_csv(OUTPUT_DIR / 'metrics_by_step.csv', index=False)
    print(f"Saved: metrics_by_step.csv")

    # 2. Main validation metrics plot (4 subplots)
    plot_validation_metrics(metrics, OUTPUT_DIR)

    # 3. Simple RMSE/MAE by horizon
    plot_metrics_by_horizon(df, OUTPUT_DIR)

    # 4. Metrics by hour of day (for 6h and 24h forecasts)
    plot_metrics_by_hour_of_day(df, OUTPUT_DIR, step=24)  # 6h
    plot_metrics_by_hour_of_day(df, OUTPUT_DIR, step=96)  # 24h

    # 5. Residual distributions with Laplace fit
    plot_residual_distribution(df, OUTPUT_DIR)

    # 6. Sample forecasts
    plot_sample_forecasts(df, OUTPUT_DIR, n_samples=3)

    # 7. Error over time
    plot_error_over_time(df, OUTPUT_DIR, step=24)
    plot_error_over_time(df, OUTPUT_DIR, step=96)

    # Print summary
    print_summary_table(metrics)

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
