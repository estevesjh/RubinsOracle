# Signal Decomposition Methods Guide

A comprehensive guide to multi-scale signal decomposition for time series forecasting.

## Overview

Signal decomposition extracts frequency bands from time series data, separating:
- **Sub-daily**: ~12h cycles (e.g., tidal, temperature swings)
- **Daily**: ~24h cycles (diurnal patterns)
- **Weekly**: ~7d cycles (human activity patterns)
- **Monthly**: ~28d cycles (lunar, billing cycles)
- **Trend**: Long-term drift

This guide covers four decomposition methods, their trade-offs, and edge effect mitigation strategies critical for forecasting.

---

## Methods Comparison

| Method | Approach | Frequency Control | Edge Effects | Speed | Best For |
|--------|----------|-------------------|--------------|-------|----------|
| **Butterworth** | IIR bandpass filter | Precise | Moderate | Fast | Known periodic signals |
| **SavGol+Butter** | Lowpass difference + cleanup | Good | Moderate | Fast | Smooth extraction |
| **STL** | Iterative LOESS | Period-based | Low | Medium | Robust decomposition |
| **VMD** | Variational optimization | Adaptive | Low | Slow | Non-stationary signals |

---

## 1. Butterworth Bandpass Filter

### How It Works

Butterworth filters have maximally flat frequency response in the passband. We use `sosfiltfilt` (second-order sections, forward-backward) to achieve zero phase shift.

```
Signal → Pad → Butterworth Bandpass → Unpad → Component
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `butter_order` | Filter steepness (higher = sharper cutoff) | 4 |
| `period_pairs` | List of (low, high) period bounds in days | See below |
| `edge_method` | Padding method for edges | `'reflect'` |
| `edge_pad_periods` | Padding length as multiple of period | 2.0 |

### Usage

```python
from signal_decomposer_v3 import SignalDecomposer

decomposer = SignalDecomposer(
    freq=96,  # 15-min data
    period_pairs=[
        (0.25, 0.75),   # Sub-daily
        (0.75, 1.25),   # Daily
        (1.5, 7.0),     # Weekly
        (7.0, 30.0),    # Monthly
        (30.0, 180.0),  # Trend
    ],
    filter_type='butterworth',
    butter_order=4,
    edge_method='reflect',
    edge_pad_periods=2.0,
)

df_decomposed = decomposer.decompose(df)
```

### Output Columns

- `y_band_0`: Sub-daily component
- `y_band_1`: Daily component
- `y_band_2`: Weekly component
- `y_band_3`: Monthly component
- `y_band_4`: Trend component

### Pros & Cons

✅ Precise frequency control  
✅ Fast computation  
✅ Well-understood theory  
❌ Edge effects at boundaries  
❌ Ringing with sharp cutoffs  

---

## 2. Savitzky-Golay + Butterworth Cleanup

### How It Works

1. Apply two SavGol lowpass filters with different window sizes
2. Subtract to get bandpass: `bandpass = lowpass_short - lowpass_long`
3. Clean up with Butterworth to remove spectral leakage

```
Signal → SavGol(short) - SavGol(long) → Butterworth Cleanup → Component
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `savgol_polyorder` | Polynomial order | 3 |
| `savgol_butter_cleanup` | Apply Butterworth after SavGol | `True` |
| `savgol_butter_margin` | Frequency margin for cleanup (0.1 = 10%) | 0.1 |

### Usage

```python
decomposer = SignalDecomposer(
    freq=96,
    period_pairs=[(0.5, 1), (1, 7), (7, 28), (28, 180)],
    filter_type='savgol',
    savgol_polyorder=3,
    savgol_butter_cleanup=True,
    savgol_butter_margin=0.1,
    edge_method='reflect',
)

df_decomposed = decomposer.decompose(df)
```

### Pros & Cons

✅ Smooth output  
✅ Preserves local trends  
✅ Good for noisy data  
❌ Less precise frequency isolation than pure Butterworth  
❌ Window size selection can be tricky  

---

## 3. STL (Seasonal-Trend Decomposition using LOESS)

### How It Works

STL iteratively extracts seasonal components at specified periods using locally weighted regression (LOESS).

```
Signal → STL(period_1) → seasonal_1 + remainder
         remainder → STL(period_2) → seasonal_2 + remainder
         ...
         final remainder → trend
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `periods` | List of periods in days to extract | `[0.5, 1, 7, 28]` |
| `robust` | Downweight outliers | `False` |
| `edge_method` | Padding method | `'reflect'` |
| `edge_pad_periods` | Padding length | 2.0 |

### Usage

```python
from stl_decomposer import STLDecomposer

decomposer = STLDecomposer(
    freq=96,
    periods=[0.5, 1.0, 7.0, 28.0],
    robust=False,
    edge_method='reflect',
    edge_pad_periods=2.0,
)

df_decomposed = decomposer.decompose(df)
```

### Output Columns

- `y_period_0.5d`: Sub-daily (12h)
- `y_period_1.0d`: Daily (24h)
- `y_period_7.0d`: Weekly
- `y_period_28.0d`: Monthly
- `y_trend`: Long-term trend

### Pros & Cons

✅ Robust to outliers (with `robust=True`)  
✅ Interpretable components  
✅ Good for exploratory analysis  
✅ Lower edge effects than Butterworth  
❌ Less precise frequency isolation  
❌ Designed for single-period, not multi-band  

---

## 4. VMD (Variational Mode Decomposition)

### How It Works

VMD decomposes a signal into K intrinsic mode functions (IMFs) by solving a variational optimization problem. Each IMF is compact around a center frequency.

Our two-stage approach:
1. **Stage 1**: VMD on original signal → Extract IMF1 (lowest frequency mode)
2. **Stage 2**: Apply Butterworth bandpass to IMF1 for daily/weekly/monthly/trend
3. **Stage 3**: VMD on residual → Extract sub-daily from IMF2

```
Signal → VMD(K=5) → IMF1 → Butterworth → daily, weekly, monthly, trend
                  → residual → VMD(K=3) → sub-daily, noise
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `alpha` | Bandwidth constraint (higher = narrower modes) | 2000 |
| `K_stage1` | Number of modes in first VMD | 5 |
| `K_stage2` | Number of modes in second VMD | 3 |
| `butter_order` | Order for Butterworth extraction | 4 |
| `butter_margin` | Frequency margin | 0.1 |

### Usage

```python
from rubin_vmd_decomposer import RubinVMDDecomposer

decomposer = RubinVMDDecomposer(
    freq=96,
    alpha=2000,
    K_stage1=5,
    K_stage2=3,
    butter_order=4,
)

df_decomposed = decomposer.decompose(df)
```

### Output Columns

- `y_imf3`: High-frequency noise
- `y_subdaily`: Sub-daily (~12h)
- `y_vmd_daily`: Daily [0.75, 1.25]d
- `y_vmd_weekly`: Weekly [1.5, 7.0]d
- `y_vmd_monthly`: Monthly [7.0, 30.0]d
- `y_vmd_trend`: Trend [30.0, 200.0]d

### Pros & Cons

✅ Adaptive to data (no predefined frequencies)  
✅ Good for non-stationary signals  
✅ Handles mode mixing well  
❌ Slower computation  
❌ Requires parameter tuning (alpha, K)  
❌ Can be unstable with short signals  

---

## Edge Effect Mitigation

Edge effects are critical for forecasting since we care most about recent data.

### Padding Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `reflect` | `[d c b \| a b c d \| c b a]` | General purpose (recommended) |
| `symmetric` | `[c b a \| a b c d \| d c b]` | Smoother transitions |
| `constant` | `[a a a \| a b c d \| d d d]` | Signals that plateau |
| `extrapolate` | Linear extension | Trending data |
| `none` | No padding | Raw filter behavior |

### Recommended Settings

```python
# For most forecasting applications
edge_method='reflect'
edge_pad_periods=2.0  # Pad 2x the max period at each edge
```

### Edge Weighting (Optional)

For extra caution at the right edge:

```python
decomposer = SignalDecomposer(
    ...,
    use_edge_weighting=True,
    edge_weight_decay=0.1,  # Slow decay
)
```

This blends the filtered output with a local linear trend at edges, reducing reliance on uncertain edge values.

---

## Choosing a Method

### Decision Tree

```
Is your signal stationary with known periods?
├── YES → Use Butterworth (fastest, most precise)
└── NO → Is robustness to outliers important?
    ├── YES → Use STL with robust=True
    └── NO → Is the signal highly non-stationary?
        ├── YES → Use VMD (adaptive)
        └── NO → Use STL or SavGol+Butter
```

### By Use Case

| Use Case | Recommended Method |
|----------|-------------------|
| Temperature forecasting | Butterworth or STL |
| Financial time series | VMD (non-stationary) |
| Sensor data with outliers | STL (robust=True) |
| Exploratory analysis | STL |
| Real-time processing | Butterworth (fastest) |
| Research/prototyping | Any (compare all) |

---

## API Reference

### Common Interface

All decomposers share a similar API:

```python
# Initialize
decomposer = Decomposer(freq=96, periods=..., edge_method='reflect')

# Decompose
df_decomposed = decomposer.decompose(df)

# Get component info
info = decomposer.get_component_info()

# Reconstruct original signal
reconstructed = decomposer.reconstruct(df_decomposed)
```

### Input DataFrame

```python
df = pd.DataFrame({
    'ds': pd.date_range('2023-01-01', periods=1000, freq='15min'),
    'y': signal_values,
})
```

### Output DataFrame

Original columns plus decomposed components:

```python
# Butterworth/SavGol
['ds', 'y', 'y_band_0', 'y_band_1', 'y_band_2', 'y_band_3', 'y_band_4']

# STL
['ds', 'y', 'y_period_0.5d', 'y_period_1.0d', 'y_period_7.0d', 'y_period_28.0d', 'y_trend']

# VMD
['ds', 'y', 'y_imf3', 'y_subdaily', 'y_vmd_daily', 'y_vmd_weekly', 'y_vmd_monthly', 'y_vmd_trend']
```

---

## Synthetic Test Results

Tested on 6-month synthetic signal with known components:

| Component | True Amplitude | Butterworth RMSE | STL RMSE |
|-----------|---------------|------------------|----------|
| Sub-daily (12h) | 3 | 0.19 | 0.11 |
| Daily (24h) | 5 | 0.26 | 0.06 |
| Weekly (7d) | 2 | 0.77 | 0.13 |
| Monthly (28d) | 1 | 1.04 | 0.24 |

STL shows better extraction accuracy when periods are known exactly.

---

## Files

| File | Description |
|------|-------------|
| `signal_decomposer_v3.py` | Butterworth & SavGol decomposer |
| `stl_decomposer.py` | STL LOESS decomposer |
| `rubin_vmd_decomposer.py` | Two-stage VMD decomposer |
| `comparison_notebook_final.ipynb` | Side-by-side comparison |

---

## References

- Butterworth, S. (1930). "On the Theory of Filter Amplifiers"
- Cleveland, R. B. et al. (1990). "STL: A Seasonal-Trend Decomposition Procedure Based on Loess"
- Dragomiretskiy, K. & Zosso, D. (2014). "Variational Mode Decomposition"
- Savitzky, A. & Golay, M. (1964). "Smoothing and Differentiation of Data by Simplified Least Squares Procedures"
