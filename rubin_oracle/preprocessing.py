"""
Signal Decomposition Module

Provides flexible time series decomposition using Savitzky-Golay or Butterworth filters
to extract multi-frequency components from signals.

Classes:
    BandpassDecomposer: Bandpass decomposer using period_pairs (direct bandpass filtering)
        with edge effect mitigation and periodic NaN filling for forecasting applications
    RubinVMDDecomposer: Two-stage VMD + Butterworth decomposition specifically designed
        for Rubin Observatory temperature data (requires vmdpy)

Functions:
    fill_nan_periodic: Fill NaN values using same hour-of-day from adjacent periods
    preprocess_for_forecast: Convenience function to preprocess data with decomposition
"""

import warnings
from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy.signal import butter, savgol_filter, sosfiltfilt
from statsmodels.tsa.seasonal import STL

# =============================================================================
# Periodic NaN Fill Functions
# =============================================================================


def fill_nan_periodic(
    y: np.ndarray | pd.Series,
    freq: int = 96,
    period_days: float = 1.0,
    max_gap_periods: int = 7,
    verbose: bool = True,
) -> np.ndarray:
    """Fill NaN values using same hour-of-day from adjacent periods.

    For daily cycles, this fills missing values with the weighted average of the
    same time-of-day from the day before and day after. This preserves
    the periodic structure better than linear interpolation.

    Args:
        y: Signal with NaN values
        freq: Observations per day (default: 96 for 15-min data)
        period_days: Period to use for filling (default: 1.0 for daily)
        max_gap_periods: Maximum gap size to fill (in periods)
            Gaps longer than this will remain NaN
        verbose: Whether to print fill statistics

    Returns:
        Array with NaN filled using periodic values

    Example:
        >>> y_filled = fill_nan_periodic(df['y'], freq=96, period_days=1.0)
    """
    if isinstance(y, pd.Series):
        y = y.values.copy()
    else:
        y = y.copy()

    period = int(period_days * freq)
    max_gap = max_gap_periods * period

    # Find NaN indices
    nan_mask = np.isnan(y)
    nan_indices = np.where(nan_mask)[0]

    if len(nan_indices) == 0:
        return y

    # Group consecutive NaN into gaps
    gaps = []
    if len(nan_indices) > 0:
        gap_start = nan_indices[0]
        gap_end = nan_indices[0]

        for i in range(1, len(nan_indices)):
            if nan_indices[i] == gap_end + 1:
                gap_end = nan_indices[i]
            else:
                gaps.append((gap_start, gap_end))
                gap_start = nan_indices[i]
                gap_end = nan_indices[i]
        gaps.append((gap_start, gap_end))

    filled_count = 0

    for gap_start, gap_end in gaps:
        gap_length = gap_end - gap_start + 1

        # Skip gaps that are too long
        if gap_length > max_gap:
            warnings.warn(
                f"Gap at index {gap_start}-{gap_end} ({gap_length} points) "
                f"exceeds max_gap_periods={max_gap_periods}. Skipping.",
                stacklevel=2,
            )
            continue

        # Fill each point in the gap
        for i in range(gap_start, gap_end + 1):
            candidates = []

            # Look back multiple periods
            for k in range(1, max_gap_periods + 1):
                idx_back = i - k * period
                if idx_back >= 0 and not np.isnan(y[idx_back]):
                    candidates.append((y[idx_back], 1.0 / k))
                    break

            # Look forward multiple periods
            for k in range(1, max_gap_periods + 1):
                idx_forward = i + k * period
                if idx_forward < len(y) and not np.isnan(y[idx_forward]):
                    candidates.append((y[idx_forward], 1.0 / k))
                    break

            if candidates:
                values = [c[0] for c in candidates]
                weights = [c[1] for c in candidates]
                y[i] = np.average(values, weights=weights)
                filled_count += 1

    # Final pass: linear interpolation for any remaining NaN
    remaining_nan = np.isnan(y)
    if remaining_nan.any():
        y_series = pd.Series(y)
        y = y_series.interpolate(method="linear").ffill().bfill().values

    if verbose:
        print(f"Filled {filled_count} NaN values using periodic fill (period={period_days}d)")

    return y


# =============================================================================
# Bandpass Decomposer
# =============================================================================

"""
BandpassDecomposer with unified padding methods for edge correction.

Supports multiple padding strategies:
- constant: Pad with edge values (default)
- reflect: Mirror reflection at edges
- symmetric: Symmetric reflection at edges
- periodic: Copy from ±1 period away (preserves local amplitude)
- stl: STL decomposition extrapolation (good for stationary signals)
- arima: ARIMA forecasting (captures local dynamics)

Usage:
    decomposer = BandpassDecomposer(
        freq=96,
        period_pairs=[(0.4, 0.6), (0.75, 1.5), (1.5, 2.5), (2.5, 14.)],
        filter_type='butterworth',
        pad_method='arima',         # 'constant', 'periodic', 'stl', 'arima'
        pad_num_periods=1,          # Number of periods to replace at each edge
        pad_max_periods=2.0,        # Max pad size in periods (caps low-freq bands)
    )
    df_decomposed = decomposer.decompose(df)
"""


class BandpassDecomposer:
    """Decomposes time series into bandpass-filtered frequency components.

    Supports multiple padding methods for edge effect mitigation:
    - Pre-filter padding: constant, reflect, symmetric, extrapolate
    - Post-filter edge correction: periodic, stl, arima
    """

    def __init__(
        self,
        freq: int = 96,
        period_pairs: Optional[list[tuple[float, float]]] = None,
        filter_type: Literal["savgol", "butterworth"] = "savgol",
        mode: Literal["drop", "keep"] = "keep",
        # NaN handling
        nan_fill: Literal["periodic", "linear", "ffill"] = "periodic",
        nan_fill_period: float = 1.0,
        nan_fill_max_gap: int = 7,
        # Pre-filter edge padding (scipy filtfilt)
        edge_method: Literal[
            "none", "reflect", "symmetric", "constant", "extrapolate"
        ] = "constant",
        edge_pad_periods: float = 4.0,
        # Post-filter edge correction (pad_*)
        pad_method: Literal["none", "periodic", "stl", "arima"] = "none",
        pad_num_periods: int = 1,
        pad_max_periods: float = 2.0,
        pad_target_periods: Optional[dict] = None,  # {band_idx: period_obs} for exact periods
        pad_arima_order: tuple[int, int, int] = (2, 0, 2),
        pad_bands: Optional[list[int]] = None,  # Which bands to apply padding to (None = all)
        # Savitzky-Golay parameters
        savgol_polyorder: int = 3,
        # Butterworth parameters
        butter_order: int = 4,
        # Output control
        verbose: bool = False,
    ):
        """Initialize the bandpass decomposer.

        Args:
            freq: Number of observations per day (96 for 15-min data)
            period_pairs: List of (period_low, period_high) tuples in days
            filter_type: 'savgol' or 'butterworth'
            mode: 'drop' or 'keep' NaN rows

            nan_fill: Method to fill NaN before filtering
            nan_fill_period: Period in days for periodic fill
            nan_fill_max_gap: Max gap size for periodic fill

            edge_method: Pre-filter padding method for scipy filtfilt
            edge_pad_periods: How many periods to pad at each edge (pre-filter)

            pad_method: Post-filter edge correction method
                - 'none': No post-filter correction
                - 'periodic': Copy from ±1 period away (preserves local amplitude)
                - 'stl': STL decomposition extrapolation (mean-adjusted)
                - 'arima': ARIMA forecasting (captures local dynamics)
            pad_num_periods: Number of periods to replace at each edge
            pad_max_periods: Max pad size in periods (caps low-freq bands)
            pad_target_periods: Dict mapping band_idx to exact period_obs for STL/ARIMA
            pad_arima_order: ARIMA order (p, d, q) for arima method
            pad_bands: List of band indices to apply padding to (None = all bands)

            savgol_polyorder: Polynomial order for Savitzky-Golay
            butter_order: Order of Butterworth filter
            verbose: Print progress messages
        """
        self.freq = freq
        self.filter_type = filter_type
        self.mode = mode
        self.verbose = verbose

        # NaN handling
        self.nan_fill = nan_fill
        self.nan_fill_period = nan_fill_period
        self.nan_fill_max_gap = nan_fill_max_gap

        # Pre-filter edge padding
        self.edge_method = edge_method
        self.edge_pad_periods = edge_pad_periods

        # Post-filter edge correction (pad_*)
        self.pad_method = pad_method
        self.pad_num_periods = pad_num_periods
        self.pad_max_periods = pad_max_periods
        self.pad_target_periods = pad_target_periods or {}
        self.pad_arima_order = pad_arima_order
        self.pad_bands = pad_bands  # None means all bands

        # Period pairs
        if period_pairs is None:
            self.period_pairs = [
                (0.25, 0.75),
                (0.75, 1.25),
                (1.5, 7.0),
                (7.0, 30.0),
            ]
        else:
            self.period_pairs = period_pairs

        # Filter parameters
        self.savgol_polyorder = savgol_polyorder
        self.butter_order = butter_order

        self._filter_specs = self._calculate_filter_specs()

    def _calculate_filter_specs(self) -> list[dict]:
        """Calculate filter specifications for each frequency band."""
        specs = []

        for i, (period_low, period_high) in enumerate(self.period_pairs):
            period_low_obs = period_low * self.freq
            period_high_obs = period_high * self.freq

            spec = {
                "name": f"band_{i}",
                "period_low_days": period_low,
                "period_high_days": period_high,
                "period_low_obs": period_low_obs,
                "period_high_obs": period_high_obs,
            }

            if self.filter_type == "savgol":
                low_window = int(period_low_obs * 1.5)
                high_window = int(period_high_obs * 1.5)
                low_window = max(self.savgol_polyorder + 2, low_window)
                high_window = max(self.savgol_polyorder + 2, high_window)
                if low_window % 2 == 0:
                    low_window += 1
                if high_window % 2 == 0:
                    high_window += 1
                spec["low_window"] = low_window
                spec["high_window"] = high_window

            elif self.filter_type == "butterworth":
                nyquist = 0.5
                f_low = 1 / period_high_obs
                f_high = 1 / period_low_obs
                spec["lowcut"] = max(0.001, min(0.999, f_low / nyquist))
                spec["highcut"] = max(0.001, min(0.999, f_high / nyquist))
                if spec["lowcut"] >= spec["highcut"]:
                    mid = (spec["lowcut"] + spec["highcut"]) / 2
                    spec["lowcut"] = mid * 0.8
                    spec["highcut"] = mid * 1.2

            spec["pad_length"] = int(self.edge_pad_periods * max(period_low_obs, period_high_obs))
            specs.append(spec)

        return specs

    # =========================================================================
    # Pre-filter padding methods (for scipy filtfilt)
    # =========================================================================

    def _pad_signal(self, y: np.ndarray, pad_length: int) -> tuple[np.ndarray, int]:
        """Pad signal at edges to reduce edge effects (pre-filter)."""
        if self.edge_method == "none" or pad_length <= 0:
            return y, 0

        n = len(y)
        pad_length = min(pad_length, n - 1)
        if pad_length < 1:
            return y, 0

        if self.edge_method == "reflect":
            left_pad = y[1 : pad_length + 1][::-1]
            right_pad = y[-(pad_length + 1) : -1][::-1]
        elif self.edge_method == "symmetric":
            left_pad = y[:pad_length][::-1]
            right_pad = y[-pad_length:][::-1]
        elif self.edge_method == "constant":
            left_pad = np.full(pad_length, y[0])
            right_pad = np.full(pad_length, y[-1])
        elif self.edge_method == "extrapolate":
            n_fit = min(pad_length, n // 4, 100)
            n_fit = max(2, n_fit)
            left_slope, left_intercept = np.polyfit(np.arange(n_fit), y[:n_fit], 1)
            left_pad = left_intercept + left_slope * np.arange(-pad_length, 0)
            right_x = np.arange(n - n_fit, n)
            right_slope, right_intercept = np.polyfit(right_x, y[-n_fit:], 1)
            right_pad = right_intercept + right_slope * np.arange(n, n + pad_length)
        else:
            return y, 0

        return np.concatenate([left_pad, y, right_pad]), pad_length

    def _unpad_signal(self, y_padded: np.ndarray, pad_length: int) -> np.ndarray:
        """Remove padding from signal."""
        if pad_length <= 0:
            return y_padded
        return y_padded[pad_length:-pad_length]

    # =========================================================================
    # Post-filter edge correction methods (pad_*)
    # =========================================================================

    def _pad_periodic(self, band: np.ndarray, period_obs: int, num_periods: int) -> np.ndarray:
        """
        Fix edges by copying from ±1 period away.

        Preserves local amplitude - best for real data with amplitude variation.

        Args:
            band: Signal array (already filtered)
            period_obs: Period in observations
            num_periods: Number of periods to replace at each edge
        """
        band_fixed = band.copy()
        n = len(band)
        pad_len = num_periods * period_obs

        if pad_len > n // 3:
            return band_fixed

        # Right edge: copy from 1 period earlier
        for i in range(pad_len):
            idx = n - pad_len + i
            src_idx = idx - period_obs
            if 0 <= src_idx < n - pad_len:
                band_fixed[idx] = band[src_idx]

        # Left edge: copy from 1 period later
        for i in range(pad_len):
            src_idx = i + period_obs
            if pad_len <= src_idx < n:
                band_fixed[i] = band[src_idx]

        return band_fixed

    def _pad_stl(self, band: np.ndarray, period_obs: int, num_periods: int) -> np.ndarray:
        """
        Fix edges using STL decomposition with phase-aligned extrapolation.

        Mean-adjusted to avoid bias. Good for stationary periodic signals.

        Args:
            band: Signal array (already filtered)
            period_obs: Period in observations (must match dominant frequency)
            num_periods: Number of periods to replace at each edge
        """

        n = len(band)
        band_fixed = band.copy()
        pad_len = num_periods * period_obs

        if n < 3 * pad_len:
            return band_fixed

        middle = band[pad_len:-pad_len]
        middle_mean = np.mean(middle)

        if len(middle) < 2 * period_obs:
            return band_fixed

        try:
            stl = STL(middle, period=period_obs, seasonal=7, robust=True).fit()
            seasonal = stl.seasonal
            middle_len = len(middle)
            one_period = seasonal[-period_obs:]

            # Right edge - phase aligned, mean adjusted
            right_phases = (middle_len + np.arange(pad_len)) % period_obs
            right_seasonal = one_period[right_phases]
            band_fixed[-pad_len:] = right_seasonal - np.mean(right_seasonal) + middle_mean

            # Left edge - phase aligned, mean adjusted
            left_phases = np.arange(-pad_len, 0) % period_obs
            left_seasonal = one_period[left_phases]
            band_fixed[:pad_len] = left_seasonal - np.mean(left_seasonal) + middle_mean

        except Exception as e:
            if self.verbose:
                warnings.warn(f"STL padding failed: {e}", stacklevel=2)

        return band_fixed

    def _pad_arima(self, band: np.ndarray, period_obs: int, num_periods: int) -> np.ndarray:
        """
        Fix edges using ARIMA forecasting.

        Captures local dynamics - best for real data with amplitude variation.
        Note: Can be unstable for num_periods > 1.

        Args:
            band: Signal array (already filtered)
            period_obs: Period in observations
            num_periods: Number of periods to replace at each edge
        """
        from statsmodels.tsa.arima.model import ARIMA

        n = len(band)
        band_fixed = band.copy()
        pad_len = num_periods * period_obs

        if n < 3 * pad_len:
            return band_fixed

        middle = band[pad_len:-pad_len]

        try:
            # Fit ARIMA on middle section and forecast right edge
            model = ARIMA(middle, order=self.pad_arima_order)
            fitted = model.fit()
            forecast_right = fitted.forecast(steps=pad_len)
            band_fixed[-pad_len:] = forecast_right

            # For left edge, fit on reversed middle and forecast
            model_left = ARIMA(middle[::-1], order=self.pad_arima_order)
            fitted_left = model_left.fit()
            forecast_left = fitted_left.forecast(steps=pad_len)
            band_fixed[:pad_len] = forecast_left[::-1]

        except Exception as e:
            if self.verbose:
                warnings.warn(f"ARIMA padding failed: {e}", stacklevel=2)

        return band_fixed

    def _apply_pad_correction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply post-filter edge correction to selected bands."""
        if self.pad_method == "none":
            return df

        for band_idx, (p_lo, p_hi) in enumerate(self.period_pairs):
            # Skip bands not in pad_bands (if specified)
            if self.pad_bands is not None and band_idx not in self.pad_bands:
                if self.verbose:
                    print(f"  Skip band_{band_idx} (not in pad_bands)")
                continue

            band_col = f"y_band_{band_idx}"

            if band_col not in df.columns:
                continue

            band = df[band_col].values

            # Get period_obs: use target if specified, otherwise use band center
            if band_idx in self.pad_target_periods:
                period_obs = self.pad_target_periods[band_idx]
            else:
                center_period_days = (p_lo + p_hi) / 2
                period_obs = max(2, int(center_period_days * self.freq))

            # Cap num_periods by pad_max_periods
            max_periods = int(self.pad_max_periods * self.freq / period_obs)
            num_periods = min(self.pad_num_periods, max(1, max_periods))

            if self.verbose:
                print(
                    f"  Pad {band_col} ({p_lo}-{p_hi}d): method={self.pad_method}, "
                    f"period={period_obs}, num_periods={num_periods}"
                )

            # Apply padding method
            if self.pad_method == "periodic":
                df[band_col] = self._pad_periodic(band, period_obs, num_periods)
            elif self.pad_method == "stl":
                df[band_col] = self._pad_stl(band, period_obs, num_periods)
            elif self.pad_method == "arima":
                df[band_col] = self._pad_arima(band, period_obs, num_periods)

        return df

    # =========================================================================
    # Filter methods
    # =========================================================================

    def _apply_savgol_bandpass(
        self,
        y,
        low_window,
        high_window,
        name,
        pad_length,
        is_trend=False,
        period_low_days=None,
        period_high_days=None,
    ):
        """Apply Savitzky-Golay bandpass filter."""
        y_padded, actual_pad = self._pad_signal(y, pad_length)

        if len(y_padded) < max(low_window, high_window):
            warnings.warn(f"{name}: Signal too short for window sizes.", stacklevel=2)
            return np.full_like(y, np.nan)

        nyquist_length = len(y_padded) // 2

        try:
            if high_window > nyquist_length or is_trend:
                if low_window > len(y_padded):
                    low_window = len(y_padded) if len(y_padded) % 2 == 1 else len(y_padded) - 1
                lowpass = savgol_filter(y_padded, low_window, self.savgol_polyorder, mode="nearest")
                bandpass = self._unpad_signal(lowpass, actual_pad)
            else:
                lowpass_low = savgol_filter(
                    y_padded, low_window, self.savgol_polyorder, mode="nearest"
                )
                lowpass_high = savgol_filter(
                    y_padded, high_window, self.savgol_polyorder, mode="nearest"
                )
                bandpass_padded = lowpass_low - lowpass_high
                bandpass = self._unpad_signal(bandpass_padded, actual_pad)

            return bandpass

        except Exception as e:
            warnings.warn(f"Error in Savitzky-Golay for {name}: {e}", stacklevel=2)
            return np.full_like(y, np.nan)

    def _apply_butterworth_bandpass(self, y, lowcut, highcut, name, pad_length, is_trend=False):
        """Apply Butterworth bandpass filter."""
        y_padded, actual_pad = self._pad_signal(y, pad_length)

        try:
            if is_trend or (highcut - lowcut) < 0.001:
                sos = butter(self.butter_order, highcut, btype="low", output="sos")
            else:
                sos = butter(self.butter_order, [lowcut, highcut], btype="band", output="sos")

            filtered_padded = sosfiltfilt(
                sos, y_padded, padtype="odd", padlen=min(100, len(y_padded) // 4)
            )
            return self._unpad_signal(filtered_padded, actual_pad)

        except Exception as e:
            warnings.warn(f"Error in Butterworth for {name}: {e}", stacklevel=2)
            return np.full_like(y, np.nan)

    # =========================================================================
    # Main decompose method
    # =========================================================================

    def decompose(self, df: pd.DataFrame) -> pd.DataFrame:
        """Decompose time series into bandpass-filtered frequency components."""
        df = df.copy()

        if "y" not in df.columns:
            if "temperature" in df.columns:
                df["y"] = df["temperature"]
            elif "tempMean" in df.columns:
                df["y"] = df["tempMean"]
            else:
                raise ValueError("Column 'y', 'temperature', or 'tempMean' not found.")

        nan_count = df["y"].isna().sum()

        if nan_count > 0:
            if self.nan_fill == "periodic":
                y_filled = fill_nan_periodic(
                    df["y"].values,
                    self.freq,
                    self.nan_fill_period,
                    self.nan_fill_max_gap,
                    self.verbose,
                )
            elif self.nan_fill == "linear":
                y_filled = df["y"].interpolate(method="linear").ffill().bfill().values
            else:
                y_filled = df["y"].ffill().bfill().values
        else:
            y_filled = df["y"].values

        if self.verbose:
            print(f"\nBandpass Decomposition - {self.filter_type.upper()}")
            print(f"Data: {len(y_filled)} obs ({len(y_filled) / self.freq:.1f} days)")
            print(f"Edge method: {self.edge_method}, Pad method: {self.pad_method}")

        for idx, spec in enumerate(self._filter_specs):
            name = spec["name"]
            period_low = spec["period_low_days"]
            period_high = spec["period_high_days"]
            pad_length = spec["pad_length"]
            is_trend = idx == len(self._filter_specs) - 1

            if self.filter_type == "savgol":
                df[f"y_{name}"] = self._apply_savgol_bandpass(
                    y_filled,
                    spec["low_window"],
                    spec["high_window"],
                    name,
                    pad_length,
                    is_trend=is_trend,
                    period_low_days=period_low,
                    period_high_days=period_high,
                )
            elif self.filter_type == "butterworth":
                df[f"y_{name}"] = self._apply_butterworth_bandpass(
                    y_filled, spec["lowcut"], spec["highcut"], name, pad_length, is_trend=is_trend
                )

        if self.mode == "drop":
            component_cols = [f"y_{spec['name']}" for spec in self._filter_specs]
            df = df.dropna(subset=component_cols)

        # Apply post-filter edge correction
        df = self._apply_pad_correction(df)

        if self.verbose:
            print(f"Decomposition complete. Shape: {df.shape}\n")

        return df

    def get_component_names(self) -> list[str]:
        """Get list of component column names."""
        return [f"y_{spec['name']}" for spec in self._filter_specs]

    def reconstruct(self, df: pd.DataFrame) -> np.ndarray:
        """Reconstruct signal from bandpass components."""
        reconstructed = np.zeros(len(df))
        for spec in self._filter_specs:
            col = f"y_{spec['name']}"
            if col in df.columns:
                reconstructed += df[col].values
        return reconstructed


# =============================================================================
# Rubin VMD Decomposer
# =============================================================================

# Check for vmdpy availability
try:
    from vmdpy import VMD

    VMDPY_AVAILABLE = True
except ImportError:
    VMDPY_AVAILABLE = False


class RubinVMDDecomposer:
    """Two-stage VMD + Butterworth decomposition for Rubin temperature data.

    This decomposer uses a hybrid approach specifically tuned for Rubin Observatory
    temperature data characteristics:

    Stage 1: VMD on original signal
        - Extract IMF1 (lowest frequency mode)
        - Apply Butterworth bandpass filters to separate IMF1 into:
            - Daily: [0.5, 2.0] days
            - Weekly: [1.5, 9.0] days
            - Monthly: [21.0, 38.0] days
            - Trend: [30.0, 200.0] days

    Stage 2: VMD on residual (signal - monthly - trend)
        - Extract high-frequency components:
            - Sub-daily: IMF2 from residual VMD
            - IMF3: Highest frequency (noise-like)

    Final output (high freq to low freq):
        - y_imf3: Highest frequency component (noise)
        - y_subdaily: Sub-daily component (~12h)
        - y_vmd_daily: Daily band (~24h)
        - y_vmd_weekly: Weekly band (~7d)
        - y_vmd_monthly: Monthly band (~28d)
        - y_vmd_trend: Long-term trend

    Requires: vmdpy (pip install vmdpy)

    Example:
        >>> decomposer = RubinVMDDecomposer(freq=96, alpha=2000)
        >>> df_decomposed = decomposer.decompose(df)
        >>> reconstructed = decomposer.reconstruct(df_decomposed)
    """

    def __init__(
        self,
        freq: int = 96,
        alpha: float = 2000,
        K_stage1: int = 5,
        K_stage2: int = 3,
        tau: float = 0,
        DC: int = 0,
        init: int = 1,
        tol: float = 1e-7,
        butter_order: int = 4,
        butter_margin: float = 0.1,
        verbose: bool = True,
        include_residual: bool = False,
    ):
        """Initialize the Rubin VMD Decomposer.

        Args:
            freq: Number of observations per day (96 for 15-min, 48 for 30-min, 24 for hourly)
            alpha: VMD bandwidth constraint (higher = narrower bands, default: 2000)
            K_stage1: Number of modes for first VMD (default: 5, we only use IMF1)
            K_stage2: Number of modes for second VMD (default: 3)
            tau: Noise tolerance (0 = no noise)
            DC: Whether to include DC component (0 = no, 1 = yes)
            init: Initialization method (1 = uniform)
            tol: Convergence tolerance
            butter_order: Order of Butterworth filters
            butter_margin: Frequency margin for Butterworth (default: 0.1 = 10%)
            verbose: Whether to print progress messages
            include_residual: Whether to include residual (original - reconstructed) as feature
        """
        if not VMDPY_AVAILABLE:
            raise ImportError(
                "vmdpy is required for RubinVMDDecomposer. Install with: pip install vmdpy"
            )

        self.freq = freq
        self.alpha = alpha
        self.K_stage1 = K_stage1
        self.K_stage2 = K_stage2
        self.tau = tau
        self.DC = DC
        self.init = init
        self.tol = tol
        self.butter_order = butter_order
        self.butter_margin = butter_margin
        self.verbose = verbose
        self.include_residual = include_residual

        self.butter_periods = [
            (0.5, 2.0),  # Daily band
            (1.5, 9.0),  # Weekly band
            (21.0, 38.0),  # Monthly band
            (30.0, 200.0),  # Trend band
        ]

        self.band_names = ["vmd_daily", "vmd_weekly", "vmd_monthly", "vmd_trend"]

    def _run_vmd(self, signal: np.ndarray, K: int) -> np.ndarray:
        """Run VMD decomposition."""
        original_length = len(signal)

        u, u_hat, omega = VMD(signal, self.alpha, self.tau, K, self.DC, self.init, self.tol)

        final_omega = omega[-1, :]
        sorted_indices = np.argsort(final_omega)
        u_sorted = u[sorted_indices]

        if u_sorted.shape[1] != original_length:
            new_u = np.zeros((K, original_length))
            min_len = min(u_sorted.shape[1], original_length)
            new_u[:, :min_len] = u_sorted[:, :min_len]

            if u_sorted.shape[1] < original_length:
                for i in range(K):
                    new_u[i, min_len:] = u_sorted[i, -1]

            u_sorted = new_u

        return u_sorted

    def _apply_butterworth_bandpass(
        self,
        signal: np.ndarray,
        period_low: float,
        period_high: float,
        name: str,
        is_trend: bool = False,
    ) -> np.ndarray:
        """Apply Butterworth bandpass filter."""
        try:
            period_low_obs = period_low * self.freq
            period_high_obs = period_high * self.freq

            nyquist = 0.5
            f_low = 1 / period_high_obs
            f_high = 1 / period_low_obs

            f_low_margin = f_low * (1 - self.butter_margin)
            f_high_margin = f_high * (1 + self.butter_margin)

            lowcut = max(0.001, min(0.999, f_low_margin / nyquist))
            highcut = max(0.001, min(0.999, f_high_margin / nyquist))

            if is_trend or lowcut >= highcut:
                sos = butter(self.butter_order, highcut, btype="low", output="sos")
            else:
                sos = butter(self.butter_order, [lowcut, highcut], btype="band", output="sos")

            filtered = sosfiltfilt(sos, signal)

            if len(filtered) != len(signal):
                if len(filtered) > len(signal):
                    filtered = filtered[: len(signal)]
                else:
                    filtered = np.pad(filtered, (0, len(signal) - len(filtered)), mode="edge")

            return filtered

        except Exception as e:
            warnings.warn(f"Error applying Butterworth for {name}: {e}", stacklevel=2)
            return np.full(len(signal), np.nan)

    def decompose(self, df: pd.DataFrame) -> pd.DataFrame:
        """Decompose time series using two-stage VMD + Butterworth."""
        df = df.copy()

        if "y" not in df.columns:
            if "tempMean" in df.columns:
                df["y"] = df["tempMean"]
            else:
                raise ValueError("Column 'y' or 'tempMean' not found in dataframe.")

        y = df["y"].ffill().bfill().values

        if self.verbose:
            print(f"\n{'=' * 70}")
            print("Rubin VMD Decomposition")
            print(f"{'=' * 70}")
            print(f"Data frequency: {self.freq} obs/day")
            print(f"Data length: {len(y)} observations ({len(y) / self.freq:.1f} days)")
            print(f"VMD alpha: {self.alpha}")
            print(f"{'=' * 70}\n")

        # Stage 1
        if self.verbose:
            print(f"STAGE 1: VMD on original signal (K={self.K_stage1})...")

        imfs_stage1 = self._run_vmd(y, self.K_stage1)
        imf1 = imfs_stage1[0]

        if self.verbose:
            print("  Extracted IMF1 (lowest frequency mode)")
            print(f"  IMF1 std: {np.std(imf1):.4f}")
            print("\nApplying Butterworth bandpass to IMF1...")

        for i, ((period_low, period_high), name) in enumerate(
            zip(self.butter_periods[1:], self.band_names[1:])
        ):
            is_trend = i == len(self.butter_periods) - 2

            if self.verbose:
                print(f"  Computing {name} ({period_low:.2f}-{period_high:.1f}d)...")

            df[f"y_{name}"] = self._apply_butterworth_bandpass(
                imf1, period_low, period_high, name, is_trend=is_trend
            )

        # Stage 2
        if self.verbose:
            print(f"\nSTAGE 2: VMD on residual signal (K={self.K_stage2})...")

        residual = y - df["y_vmd_monthly"].values - df["y_vmd_trend"].values

        if self.verbose:
            print(f"  Residual std: {np.std(residual):.4f}")

        imfs_stage2 = self._run_vmd(residual, self.K_stage2)

        daily = imfs_stage2[0]
        subdaily = imfs_stage2[1]
        imf3 = imfs_stage2[2]

        df["y_vmd_daily"] = self._apply_butterworth_bandpass(
            daily, self.butter_periods[0][0], self.butter_periods[0][1], "vmd_daily"
        )
        df["y_subdaily"] = subdaily
        df["y_imf3"] = imf3

        # Compute residual (original - reconstructed) if enabled
        if self.include_residual:
            reconstructed = self.reconstruct(df)
            df["y_residual"] = y - reconstructed

        if self.verbose:
            print(f"  Extracted sub-daily (IMF2, std: {np.std(subdaily):.4f})")
            print(f"  Extracted IMF3 / noise (std: {np.std(imf3):.4f})")
            if self.include_residual:
                print(f"  Residual std: {np.std(df['y_residual'].values):.4f}")

            print(f"\n{'=' * 70}")
            print("Decomposition complete!")
            print(f"{'=' * 70}")
            print("\nOutput components (high freq -> low freq):")
            print("  1. y_imf3        : Highest freq / noise")
            print("  2. y_subdaily    : Sub-daily component")
            print(f"  3. y_vmd_daily   : {self.butter_periods[0]} days")
            print(f"  4. y_vmd_weekly  : {self.butter_periods[1]} days")
            print(f"  5. y_vmd_monthly : {self.butter_periods[2]} days")
            print(f"  6. y_vmd_trend   : {self.butter_periods[3]} days")
            if self.include_residual:
                print("  7. y_residual    : Reconstruction residual")
            print(f"\nDataset shape: {df.shape}")
            print(f"{'=' * 70}\n")

        return df

    def get_component_names(self) -> list[str]:
        """Get list of component column names."""
        names = [
            "y_imf3",
            "y_subdaily",
            "y_vmd_daily",
            "y_vmd_weekly",
            "y_vmd_monthly",
            "y_vmd_trend",
        ]
        if self.include_residual:
            names.append("y_residual")
        return names

    def get_component_info(self) -> pd.DataFrame:
        """Get information about the decomposed components."""
        info = [
            {"component": "y_imf3", "source": "Residual VMD (Stage 2)", "period": "Very high freq"},
            {
                "component": "y_subdaily",
                "source": "Residual VMD (Stage 2)",
                "period": "Sub-daily (~12h)",
            },
            {
                "component": "y_vmd_daily",
                "source": "Butterworth on IMF1",
                "period": f"{self.butter_periods[0]} days",
            },
            {
                "component": "y_vmd_weekly",
                "source": "Butterworth on IMF1",
                "period": f"{self.butter_periods[1]} days",
            },
            {
                "component": "y_vmd_monthly",
                "source": "Butterworth on IMF1",
                "period": f"{self.butter_periods[2]} days",
            },
            {
                "component": "y_vmd_trend",
                "source": "Butterworth on IMF1",
                "period": f"{self.butter_periods[3]} days",
            },
        ]
        if self.include_residual:
            info.append(
                {
                    "component": "y_residual",
                    "source": "Original - Reconstructed",
                    "period": "Reconstruction error",
                }
            )
        return pd.DataFrame(info)

    def reconstruct(self, df: pd.DataFrame) -> np.ndarray:
        """Reconstruct the original signal from components (excludes residual)."""
        # Use base components only, not residual
        components = [
            "y_imf3",
            "y_subdaily",
            "y_vmd_daily",
            "y_vmd_weekly",
            "y_vmd_monthly",
            "y_vmd_trend",
        ]
        reconstructed = np.zeros(len(df))
        for comp in components:
            if comp in df.columns:
                reconstructed += df[comp].values
        return reconstructed


# =============================================================================
# Convenience Functions
# =============================================================================


def preprocess_for_forecast(
    df: pd.DataFrame,
    decompose: bool = True,
    freq: int = 96,
    period_pairs: Optional[list[tuple[float, float]]] = None,
    filter_type: Literal["savgol", "butterworth"] = "butterworth",
    train_start_date: str | pd.Timestamp | None = None,
    train_end_date: str | pd.Timestamp | None = None,
    **decomposer_kwargs,
) -> pd.DataFrame:
    """Preprocess data for forecasting with optional bandpass decomposition.

    Args:
        df: Input dataframe with 'ds' and 'y' columns
        decompose: Whether to apply signal decomposition
        freq: Data frequency in observations per day
        period_pairs: List of (period_low, period_high) tuples for decomposition
        filter_type: Type of filter ('savgol' or 'butterworth')
        train_start_date: Start date for training data (inclusive)
        train_end_date: End date for training data (inclusive)
        **decomposer_kwargs: Additional arguments for BandpassDecomposer

    Returns:
        Preprocessed dataframe with filtered date range and optional decomposition

    Example:
        >>> df_processed = preprocess_for_forecast(
        ...     df,
        ...     decompose=True,
        ...     freq=96,
        ...     period_pairs=[(0.5, 1.5), (1.5, 7), (7, 30)],
        ...     filter_type='butterworth',
        ...     train_end_date='2023-12-01',
        ... )
    """
    df = df.copy()

    if "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"])

    df = df.sort_values("ds").reset_index(drop=True)

    if decompose:
        decomposer = BandpassDecomposer(
            freq=freq,
            period_pairs=period_pairs,
            filter_type=filter_type,
            mode="keep",
            **decomposer_kwargs,
        )
        df = decomposer.decompose(df)

    if train_end_date is not None:
        train_end_date = pd.to_datetime(train_end_date)
        df = df[df["ds"] <= train_end_date]

    if train_start_date is not None:
        train_start_date = pd.to_datetime(train_start_date)
        df = df[df["ds"] >= train_start_date]

    return df.reset_index(drop=True)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    """Example usage and testing"""

    # Create synthetic data (15-min frequency = 96 obs/day)
    np.random.seed(42)
    freq = 96
    dates = pd.date_range("2023-01-01", "2023-07-01", freq="15min")
    n = len(dates)
    t = np.arange(n)

    # Synthetic signal with multiple frequencies
    y = (
        6 * np.sin(2 * np.pi * t / (freq * 0.5))  # Sub-daily (12h)
        + 10 * np.sin(2 * np.pi * t / freq)  # Daily (24h)
        + 5 * np.sin(2 * np.pi * t / (freq * 7))  # Weekly
        + 3 * np.sin(2 * np.pi * t / (freq * 28))  # Monthly
        + np.random.randn(n) * 0.5  # Noise
        + 20
        + 0.002 * t  # Trend
    )

    df = pd.DataFrame({"ds": dates, "y": y})

    # =========================================================================
    # Test BandpassDecomposer
    # =========================================================================
    print("=" * 80)
    print("Testing BandpassDecomposer (Butterworth)")
    print("=" * 80)

    decomposer = BandpassDecomposer(
        freq=freq,
        period_pairs=[
            (0.25, 0.75),  # Sub-daily
            (0.75, 1.25),  # Daily
            (1.5, 7.0),  # Weekly
            (7.0, 30.0),  # Monthly
            (30.0, 180.0),  # Trend
        ],
        filter_type="butterworth",
        edge_method="reflect",
        edge_pad_periods=2.0,
    )

    df_decomposed = decomposer.decompose(df.copy())

    print("\nComponent info:")
    print(decomposer.get_component_info().to_string(index=False))

    # Test reconstruction
    reconstructed = decomposer.reconstruct(df_decomposed)
    reconstruction_rmse = np.sqrt(np.mean((df_decomposed["y"].values - reconstructed) ** 2))
    print(f"\nReconstruction RMSE: {reconstruction_rmse:.4f}")

    # =========================================================================
    # Test RubinVMDDecomposer (if vmdpy available)
    # =========================================================================
    if VMDPY_AVAILABLE:
        print("\n" + "=" * 80)
        print("Testing RubinVMDDecomposer")
        print("=" * 80)

        vmd_decomposer = RubinVMDDecomposer(
            freq=freq,
            alpha=2000,
            K_stage1=5,
            K_stage2=3,
        )

        df_vmd = vmd_decomposer.decompose(df.copy())

        print("\nComponent info:")
        print(vmd_decomposer.get_component_info().to_string(index=False))

        # Test reconstruction
        reconstructed_vmd = vmd_decomposer.reconstruct(df_vmd)
        reconstruction_rmse_vmd = np.sqrt(np.mean((df_vmd["y"].values - reconstructed_vmd) ** 2))
        print(f"\nReconstruction RMSE: {reconstruction_rmse_vmd:.4f}")
    else:
        print("\n[Skipping RubinVMDDecomposer test - vmdpy not installed]")

    print("\n" + "=" * 80)
    print("All tests complete!")
    print("=" * 80)
