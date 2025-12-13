"""
Signal Decomposition Module v3

Provides bandpass filtering using Savitzky-Golay or Butterworth filters
to extract specific frequency bands from time series signals.

Edge Effect Mitigation:
- Reflect padding before filtering (configurable)
- Exponential weighting to trust recent observations more
- Configurable padding length based on period

NaN Handling:
- Periodic fill: fills NaN using same hour-of-day from adjacent days
- Preserves daily cycle structure better than linear interpolation

For forecasting applications, the right edge (most recent data) is critical.
This module implements strategies to minimize edge artifacts.
"""

import warnings
from typing import Literal, List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, butter, sosfiltfilt


def fill_nan_periodic(
    y: np.ndarray | pd.Series,
    freq: int = 96,
    period_days: float = 1.0,
    max_gap_periods: int = 7
) -> np.ndarray:
    """Fill NaN values using same hour-of-day from adjacent periods.
    
    For daily cycles, this fills missing values with the average of the
    same time-of-day from the day before and day after. This preserves
    the periodic structure better than linear interpolation.
    
    Args:
        y: Signal with NaN values
        freq: Observations per day (default: 96 for 15-min data)
        period_days: Period to use for filling (default: 1.0 for daily)
        max_gap_periods: Maximum gap size to fill (in periods)
            Gaps longer than this will remain NaN
    
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
                f"exceeds max_gap_periods={max_gap_periods}. Skipping."
            )
            continue
        
        # Fill each point in the gap
        for i in range(gap_start, gap_end + 1):
            candidates = []
            
            # Look back multiple periods
            for k in range(1, max_gap_periods + 1):
                idx_back = i - k * period
                if idx_back >= 0 and not np.isnan(y[idx_back]):
                    # Weight closer periods more
                    candidates.append((y[idx_back], 1.0 / k))
                    break  # Use closest available
            
            # Look forward multiple periods
            for k in range(1, max_gap_periods + 1):
                idx_forward = i + k * period
                if idx_forward < len(y) and not np.isnan(y[idx_forward]):
                    candidates.append((y[idx_forward], 1.0 / k))
                    break  # Use closest available
            
            if candidates:
                # Weighted average
                values = [c[0] for c in candidates]
                weights = [c[1] for c in candidates]
                y[i] = np.average(values, weights=weights)
                filled_count += 1
    
    # Final pass: linear interpolation for any remaining NaN
    remaining_nan = np.isnan(y)
    if remaining_nan.any():
        # Use pandas for easy interpolation
        y_series = pd.Series(y)
        y = y_series.interpolate(method='linear').ffill().bfill().values
    
    print(f"Filled {filled_count} NaN values using periodic fill (period={period_days}d)")
    
    return y


def fill_nan_periodic_df(
    df: pd.DataFrame,
    freq: int = 96,
    period_days: float = 1.0,
    max_gap_periods: int = 7,
    y_col: str = 'y'
) -> pd.DataFrame:
    """Fill NaN in DataFrame using periodic fill.
    
    Args:
        df: DataFrame with 'y' column
        freq: Observations per day
        period_days: Period for filling
        max_gap_periods: Maximum gap size
        y_col: Column name for target variable
    
    Returns:
        DataFrame with NaN filled
    """
    df = df.copy()
    df[y_col] = fill_nan_periodic(
        df[y_col].values,
        freq=freq,
        period_days=period_days,
        max_gap_periods=max_gap_periods
    )
    return df


class SignalDecomposer:
    """Decomposes time series into bandpass-filtered frequency components.
    
    Uses pairs of periods to create bandpass filters that isolate specific
    frequency bands. Supports Savitzky-Golay and Butterworth filtering with
    edge effect mitigation for forecasting applications.
    
    Attributes:
        freq: Data frequency in observations per day
        period_pairs: List of (low, high) period pairs in days
        filter_type: Type of filter ('savgol' or 'butterworth')
        edge_method: Padding method for edge effect mitigation
        
    Example:
        >>> decomposer = SignalDecomposer(
        ...     freq=96,
        ...     period_pairs=[(0.5, 1), (1, 7), (7, 28)],
        ...     filter_type='butterworth',
        ...     edge_method='reflect',
        ...     edge_pad_periods=2.0
        ... )
        >>> df_decomposed = decomposer.decompose(df)
    """
    
    def __init__(
        self,
        freq: int = 96,
        period_pairs: Optional[List[Tuple[float, float]]] = None,
        filter_type: Literal['savgol', 'butterworth'] = 'butterworth',
        mode: Literal['drop', 'keep'] = 'drop',
        # NaN handling
        nan_fill: Literal['periodic', 'linear', 'ffill'] = 'periodic',
        nan_fill_period: float = 1.0,
        nan_fill_max_gap: int = 7,
        # Edge effect mitigation
        edge_method: Literal['none', 'reflect', 'symmetric', 'constant', 'extrapolate'] = 'reflect',
        edge_pad_periods: float = 2.0,
        # Edge weighting (for forecasting - trust recent data more)
        use_edge_weighting: bool = False,
        edge_weight_decay: float = 0.1,
        # Savitzky-Golay specific parameters
        savgol_polyorder: int = 3,
        savgol_butter_cleanup: bool = True,
        savgol_butter_margin: float = 0.1,
        # Butterworth specific parameters
        butter_order: int = 4,
    ):
        """Initialize the signal decomposer with bandpass filtering.
        
        Args:
            freq: Number of observations per day
                - 15-min data: 96
                - 30-min data: 48
                - Hourly data: 24
            period_pairs: List of (period_low, period_high) tuples in days.
                Each pair defines a frequency band to extract.
                Example: [(0.5, 1), (1, 7), (7, 28)]
            filter_type: Type of filter
                - 'savgol': Savitzky-Golay (difference of two lowpass filters)
                - 'butterworth': Butterworth IIR bandpass filter
            mode: How to handle NaN values in output
                - 'drop': Remove rows with NaN
                - 'keep': Keep all rows
            nan_fill: Method to fill NaN before filtering
                - 'periodic': Use same hour from adjacent days (best for daily cycles)
                - 'linear': Linear interpolation
                - 'ffill': Forward fill then backward fill
            nan_fill_period: Period in days for periodic fill (default: 1.0 for daily)
            nan_fill_max_gap: Maximum gap size in periods for periodic fill
            edge_method: Method to mitigate edge effects
                - 'none': No padding (worst edge effects)
                - 'reflect': Reflect signal at edges (recommended)
                - 'symmetric': Symmetric reflection including endpoint
                - 'constant': Pad with edge values
                - 'extrapolate': Linear extrapolation at edges
            edge_pad_periods: How many periods to pad at each edge (default: 2.0)
                Larger = better edge handling but slower
                For period pair (a, b), uses max(a, b) * edge_pad_periods observations
            use_edge_weighting: If True, apply exponential weighting to trust recent data more
            edge_weight_decay: Decay rate for edge weighting (0.1 = slow decay, 1.0 = fast decay)
            savgol_polyorder: Polynomial order for Savitzky-Golay filter
            savgol_butter_cleanup: If True, apply Butterworth bandpass after Savitzky-Golay
            savgol_butter_margin: Frequency margin for Butterworth cleanup (default: 0.1 = 10%)
            butter_order: Order of Butterworth filter
        """
        self.freq = freq
        self.filter_type = filter_type
        self.mode = mode
        
        # NaN handling parameters
        self.nan_fill = nan_fill
        self.nan_fill_period = nan_fill_period
        self.nan_fill_max_gap = nan_fill_max_gap
        
        # Edge effect parameters
        self.edge_method = edge_method
        self.edge_pad_periods = edge_pad_periods
        self.use_edge_weighting = use_edge_weighting
        self.edge_weight_decay = edge_weight_decay
        
        # Set default period pairs if not provided
        if period_pairs is None:
            self.period_pairs = [
                (0.25, 0.75),  # Sub-daily
                (0.75, 1.25),  # Daily
                (1.5, 7.0),    # Weekly
                (7.0, 30.0),   # Monthly
                (30.0, 180.0), # Trend
            ]
        else:
            self.period_pairs = period_pairs
        
        # Savitzky-Golay parameters
        self.savgol_polyorder = savgol_polyorder
        self.savgol_butter_cleanup = savgol_butter_cleanup
        self.savgol_butter_margin = savgol_butter_margin
        
        # Butterworth parameters
        self.butter_order = butter_order
        
        # Calculate filter specifications
        self._filter_specs = self._calculate_filter_specs()
    
    def _calculate_filter_specs(self) -> List[dict]:
        """Calculate filter specifications for each frequency band."""
        specs = []
        
        for i, (period_low, period_high) in enumerate(self.period_pairs):
            period_low_obs = period_low * self.freq
            period_high_obs = period_high * self.freq
            
            spec = {
                'name': f'band_{i}',
                'period_low_days': period_low,
                'period_high_days': period_high,
                'period_low_obs': period_low_obs,
                'period_high_obs': period_high_obs,
            }
            
            if self.filter_type == 'savgol':
                # Window lengths for Savitzky-Golay
                low_window = int(period_low_obs * 1.5)
                high_window = int(period_high_obs * 1.5)
                
                # Ensure odd and larger than polyorder
                low_window = max(self.savgol_polyorder + 2, low_window)
                high_window = max(self.savgol_polyorder + 2, high_window)
                
                if low_window % 2 == 0:
                    low_window += 1
                if high_window % 2 == 0:
                    high_window += 1
                    
                spec['low_window'] = low_window
                spec['high_window'] = high_window
                
            elif self.filter_type == 'butterworth':
                nyquist = 0.5
                f_low = 1 / period_high_obs
                f_high = 1 / period_low_obs
                
                spec['lowcut'] = max(0.001, min(0.999, f_low / nyquist))
                spec['highcut'] = max(0.001, min(0.999, f_high / nyquist))
                
                if spec['lowcut'] >= spec['highcut']:
                    mid = (spec['lowcut'] + spec['highcut']) / 2
                    spec['lowcut'] = mid * 0.8
                    spec['highcut'] = mid * 1.2
            
            # Calculate padding length for this band
            spec['pad_length'] = int(self.edge_pad_periods * max(period_low_obs, period_high_obs))
            
            specs.append(spec)
        
        return specs
    
    def _pad_signal(self, y: np.ndarray, pad_length: int) -> Tuple[np.ndarray, int]:
        """Pad signal at edges to reduce edge effects.
        
        Args:
            y: Input signal
            pad_length: Number of points to pad at each edge
            
        Returns:
            Tuple of (padded_signal, actual_pad_length)
        """
        if self.edge_method == 'none' or pad_length <= 0:
            return y, 0
        
        n = len(y)
        pad_length = min(pad_length, n - 1)  # Can't pad more than signal length
        
        if pad_length < 1:
            return y, 0
        
        if self.edge_method == 'reflect':
            # Reflect without including endpoint: [d c b | a b c d | c b a]
            left_pad = y[1:pad_length+1][::-1]
            right_pad = y[-(pad_length+1):-1][::-1]
            
        elif self.edge_method == 'symmetric':
            # Reflect including endpoint: [c b a | a b c d | d c b]
            left_pad = y[:pad_length][::-1]
            right_pad = y[-pad_length:][::-1]
            
        elif self.edge_method == 'constant':
            # Pad with edge values
            left_pad = np.full(pad_length, y[0])
            right_pad = np.full(pad_length, y[-1])
            
        elif self.edge_method == 'extrapolate':
            # Linear extrapolation
            n_fit = min(pad_length, n // 4, 100)  # Use up to 100 points for fit
            n_fit = max(2, n_fit)
            
            # Left edge
            left_slope, left_intercept = np.polyfit(np.arange(n_fit), y[:n_fit], 1)
            left_pad = left_intercept + left_slope * np.arange(-pad_length, 0)
            
            # Right edge  
            right_x = np.arange(n - n_fit, n)
            right_slope, right_intercept = np.polyfit(right_x, y[-n_fit:], 1)
            right_pad = right_intercept + right_slope * np.arange(n, n + pad_length)
            
        else:
            return y, 0
        
        padded = np.concatenate([left_pad, y, right_pad])
        return padded, pad_length
    
    def _unpad_signal(self, y_padded: np.ndarray, pad_length: int) -> np.ndarray:
        """Remove padding from signal."""
        if pad_length <= 0:
            return y_padded
        return y_padded[pad_length:-pad_length]
    
    def _apply_edge_weighting(self, original: np.ndarray, filtered: np.ndarray) -> np.ndarray:
        """Apply exponential weighting to blend filtered result with original at edges.
        
        At the right edge (most recent data), we want to trust the original signal
        more because filtering has more uncertainty there.
        
        Args:
            original: Original signal
            filtered: Filtered signal
            
        Returns:
            Weighted blend of filtered and original
        """
        if not self.use_edge_weighting:
            return filtered
        
        n = len(filtered)
        result = filtered.copy()
        
        # Only apply weighting at the right edge (for forecasting)
        # The weight decays from 1 (trust filtered) to 0 (trust original trend)
        edge_region = min(n // 10, 100)  # Last 10% or 100 points
        
        if edge_region < 2:
            return filtered
        
        # Create weights: 1 at interior, decaying to 0.5 at edge
        weights = np.ones(edge_region)
        decay_indices = np.arange(edge_region)
        weights = 1.0 - 0.5 * (1 - np.exp(-self.edge_weight_decay * (edge_region - decay_indices)))
        
        # Blend at right edge
        # filtered * weight + local_trend * (1 - weight)
        # Use local linear trend as fallback
        fit_region = min(edge_region * 2, n // 2)
        if fit_region >= 2:
            x_fit = np.arange(n - fit_region, n)
            slope, intercept = np.polyfit(x_fit, original[n - fit_region:], 1)
            local_trend = intercept + slope * np.arange(n - edge_region, n)
            
            result[-edge_region:] = (
                weights * filtered[-edge_region:] + 
                (1 - weights) * local_trend
            )
        
        return result
    
    def _apply_savgol_bandpass(
        self,
        y: np.ndarray,
        low_window: int,
        high_window: int,
        name: str,
        pad_length: int,
        is_trend: bool = False,
        period_low_days: float = None,
        period_high_days: float = None
    ) -> np.ndarray:
        """Apply Savitzky-Golay bandpass filter with edge padding."""
        
        # Pad signal
        y_padded, actual_pad = self._pad_signal(y, pad_length)
        
        if len(y_padded) < max(low_window, high_window):
            warnings.warn(f"{name}: Signal too short for window sizes. Returning NaN.")
            return np.full_like(y, np.nan)
        
        nyquist_length = len(y_padded) // 2
        
        try:
            if high_window > nyquist_length or is_trend:
                # Trend extraction: lowpass only
                if low_window > len(y_padded):
                    low_window = len(y_padded) if len(y_padded) % 2 == 1 else len(y_padded) - 1
                
                lowpass = savgol_filter(y_padded, low_window, self.savgol_polyorder, mode='nearest')
                
                # Unpad
                bandpass = self._unpad_signal(lowpass, actual_pad)
                
                # Apply Butterworth cleanup
                if self.savgol_butter_cleanup and period_low_days is not None:
                    bandpass = self._apply_butter_lowpass(bandpass, period_low_days, name)
                
            else:
                # Normal bandpass
                lowpass_low = savgol_filter(y_padded, low_window, self.savgol_polyorder, mode='nearest')
                lowpass_high = savgol_filter(y_padded, high_window, self.savgol_polyorder, mode='nearest')
                
                bandpass_padded = lowpass_low - lowpass_high
                
                # Unpad
                bandpass = self._unpad_signal(bandpass_padded, actual_pad)
                
                # Apply Butterworth cleanup
                if self.savgol_butter_cleanup and period_low_days is not None:
                    bandpass = self._apply_butter_bandpass_cleanup(
                        bandpass, period_low_days, period_high_days, name
                    )
            
            # Apply edge weighting
            bandpass = self._apply_edge_weighting(y, bandpass)
            
            return bandpass
            
        except Exception as e:
            warnings.warn(f"Error in Savitzky-Golay for {name}: {e}")
            return np.full_like(y, np.nan)
    
    def _apply_butterworth_bandpass(
        self,
        y: np.ndarray,
        lowcut: float,
        highcut: float,
        name: str,
        pad_length: int,
        is_trend: bool = False
    ) -> np.ndarray:
        """Apply Butterworth bandpass filter with edge padding."""
        
        # Pad signal
        y_padded, actual_pad = self._pad_signal(y, pad_length)
        
        try:
            if is_trend or (highcut - lowcut) < 0.001:
                # Trend: use lowpass
                sos = butter(self.butter_order, highcut, btype='low', output='sos')
            else:
                # Normal bandpass
                sos = butter(self.butter_order, [lowcut, highcut], btype='band', output='sos')
            
            # Apply filter with padding handled by sosfiltfilt
            # sosfiltfilt already does some padding, but we've added extra for long-period filters
            filtered_padded = sosfiltfilt(sos, y_padded, padtype='odd', padlen=min(100, len(y_padded)//4))
            
            # Unpad
            filtered = self._unpad_signal(filtered_padded, actual_pad)
            
            # Apply edge weighting
            filtered = self._apply_edge_weighting(y, filtered)
            
            return filtered
            
        except Exception as e:
            warnings.warn(f"Error in Butterworth for {name}: {e}")
            return np.full_like(y, np.nan)
    
    def _apply_butter_bandpass_cleanup(
        self,
        signal: np.ndarray,
        period_low_days: float,
        period_high_days: float,
        name: str
    ) -> np.ndarray:
        """Apply Butterworth bandpass to clean up SavGol output."""
        try:
            period_low_obs = period_low_days * self.freq
            period_high_obs = period_high_days * self.freq
            
            nyquist = 0.5
            f_low = 1 / period_high_obs
            f_high = 1 / period_low_obs
            
            margin = self.savgol_butter_margin
            lowcut = max(0.001, min(0.999, f_low * (1 - margin) / nyquist))
            highcut = max(0.001, min(0.999, f_high * (1 + margin) / nyquist))
            
            if lowcut >= highcut:
                return signal
            
            sos = butter(self.butter_order, [lowcut, highcut], btype='band', output='sos')
            return sosfiltfilt(sos, signal, padtype='odd', padlen=min(50, len(signal)//4))
            
        except Exception as e:
            warnings.warn(f"Butterworth cleanup failed for {name}: {e}")
            return signal
    
    def _apply_butter_lowpass(
        self,
        signal: np.ndarray,
        period_low_days: float,
        name: str
    ) -> np.ndarray:
        """Apply Butterworth lowpass for trend extraction."""
        try:
            period_low_obs = period_low_days * self.freq
            nyquist = 0.5
            f_high = 1 / period_low_obs
            
            margin = self.savgol_butter_margin
            highcut = max(0.001, min(0.999, f_high * (1 + margin) / nyquist))
            
            sos = butter(self.butter_order, highcut, btype='low', output='sos')
            return sosfiltfilt(sos, signal, padtype='odd', padlen=min(50, len(signal)//4))
            
        except Exception as e:
            warnings.warn(f"Butterworth lowpass failed for {name}: {e}")
            return signal
    
    def decompose(self, df: pd.DataFrame) -> pd.DataFrame:
        """Decompose time series into bandpass-filtered frequency components.
        
        Args:
            df: Input dataframe with columns:
                - ds (datetime): Timestamps
                - y (float): Target values
                
        Returns:
            DataFrame with original columns plus bandpass components
        """
        df = df.copy()
        
        if 'y' not in df.columns:
            if 'tempMean' in df.columns:
                df['y'] = df['tempMean']
            else:
                raise ValueError("Column 'y' or 'tempMean' not found.")
        
        # Count NaN before filling
        nan_count = df['y'].isna().sum()
        
        # Fill NaN values based on method
        if nan_count > 0:
            if self.nan_fill == 'periodic':
                y_filled = fill_nan_periodic(
                    df['y'].values,
                    freq=self.freq,
                    period_days=self.nan_fill_period,
                    max_gap_periods=self.nan_fill_max_gap
                )
            elif self.nan_fill == 'linear':
                y_filled = df['y'].interpolate(method='linear').ffill().bfill().values
                print(f"Filled {nan_count} NaN values using linear interpolation")
            else:  # ffill
                y_filled = df['y'].ffill().bfill().values
                print(f"Filled {nan_count} NaN values using forward/backward fill")
        else:
            y_filled = df['y'].values
        
        print(f"\n{'='*70}")
        print(f"Signal Decomposition - {self.filter_type.upper()}")
        print(f"{'='*70}")
        print(f"Data frequency: {self.freq} obs/day")
        print(f"Data length: {len(y_filled)} observations ({len(y_filled)/self.freq:.1f} days)")
        print(f"NaN handling: {self.nan_fill} (period={self.nan_fill_period}d)")
        print(f"Edge method: {self.edge_method} (pad={self.edge_pad_periods} periods)")
        print(f"Edge weighting: {self.use_edge_weighting}")
        print(f"Number of bands: {len(self.period_pairs)}")
        print(f"{'='*70}\n")
        
        for idx, spec in enumerate(self._filter_specs):
            name = spec['name']
            period_low = spec['period_low_days']
            period_high = spec['period_high_days']
            pad_length = spec['pad_length']
            is_trend = (idx == len(self._filter_specs) - 1)
            
            if self.filter_type == 'savgol':
                low_window = spec['low_window']
                high_window = spec['high_window']
                print(f"Computing {name} ({period_low:.2f}-{period_high:.2f}d, "
                      f"windows=[{low_window}, {high_window}], pad={pad_length})...")
                
                df[f'y_{name}'] = self._apply_savgol_bandpass(
                    y_filled, low_window, high_window, name, pad_length,
                    is_trend=is_trend,
                    period_low_days=period_low,
                    period_high_days=period_high
                )
                
            elif self.filter_type == 'butterworth':
                lowcut = spec['lowcut']
                highcut = spec['highcut']
                print(f"Computing {name} ({period_low:.2f}-{period_high:.2f}d, "
                      f"freqs=[{lowcut:.4f}, {highcut:.4f}], pad={pad_length})...")
                
                df[f'y_{name}'] = self._apply_butterworth_bandpass(
                    y_filled, lowcut, highcut, name, pad_length, is_trend=is_trend
                )
        
        if self.mode == 'drop':
            component_cols = [f'y_{spec["name"]}' for spec in self._filter_specs]
            initial_len = len(df)
            df = df.dropna(subset=component_cols)
            if len(df) < initial_len:
                print(f"\nDropped {initial_len - len(df)} rows with NaN")
        
        print(f"\n{'='*70}")
        print(f"Decomposition complete. Shape: {df.shape}")
        print(f"{'='*70}\n")
        
        return df
    
    def get_component_info(self) -> pd.DataFrame:
        """Get information about the bandpass components."""
        info = []
        for spec in self._filter_specs:
            component_info = {
                'component': f'y_{spec["name"]}',
                'period_low_days': spec['period_low_days'],
                'period_high_days': spec['period_high_days'],
                'pad_length': spec['pad_length'],
            }
            
            if self.filter_type == 'savgol':
                component_info['low_window'] = spec['low_window']
                component_info['high_window'] = spec['high_window']
            elif self.filter_type == 'butterworth':
                component_info['lowcut'] = spec['lowcut']
                component_info['highcut'] = spec['highcut']
            
            info.append(component_info)
        
        return pd.DataFrame(info)
    
    def reconstruct(self, df: pd.DataFrame) -> np.ndarray:
        """Reconstruct signal from components."""
        reconstructed = np.zeros(len(df))
        for spec in self._filter_specs:
            col = f"y_{spec['name']}"
            if col in df.columns:
                reconstructed += df[col].values
        return reconstructed


def preprocess_for_forecast(
    df: pd.DataFrame,
    decompose: bool = True,
    freq: int = 96,
    period_pairs: Optional[List[Tuple[float, float]]] = None,
    filter_type: Literal['savgol', 'butterworth'] = 'butterworth',
    **decomposer_kwargs
) -> pd.DataFrame:
    """Preprocess data for forecasting with optional bandpass decomposition."""
    df = df.copy()
    
    if 'ds' in df.columns:
        df['ds'] = pd.to_datetime(df['ds'])
    
    df = df.sort_values('ds').reset_index(drop=True)
    
    if decompose:
        if 'mode' not in decomposer_kwargs:
            decomposer_kwargs['mode'] = 'keep'
            
        decomposer = SignalDecomposer(
            freq=freq,
            period_pairs=period_pairs,
            filter_type=filter_type,
            **decomposer_kwargs
        )
        df = decomposer.decompose(df)
    
    return df.reset_index(drop=True)


if __name__ == "__main__":
    """Test the decomposer"""
    
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-07-01', freq='15min')
    n = len(dates)
    t = np.arange(n)
    freq = 96
    
    # Synthetic signal
    subdaily = 3 * np.sin(2 * np.pi * t / (freq * 0.5))
    daily = 5 * np.sin(2 * np.pi * t / freq)
    weekly = 2 * np.sin(2 * np.pi * t / (freq * 7))
    monthly = 1 * np.sin(2 * np.pi * t / (freq * 28))
    noise = 0.5 * np.random.randn(n)
    trend = 5 + 4 * t / (96 * 24 * 7)
    
    y = subdaily + daily + weekly + monthly + noise + trend
    
    df = pd.DataFrame({'ds': dates, 'y': y})
    
    print("Testing Butterworth with reflect padding:")
    print("=" * 80)
    
    period_pairs = [
        (0.25, 0.75),
        (0.75, 1.25),
        (1.5, 7.0),
        (7.0, 30.0),
        (30.0, 180.0),
    ]
    
    decomposer = SignalDecomposer(
        freq=freq,
        period_pairs=period_pairs,
        filter_type='butterworth',
        edge_method='reflect',
        edge_pad_periods=2.0,
        use_edge_weighting=False,
    )
    
    df_decomposed = decomposer.decompose(df.copy())
    
    print("\nComponent info:")
    print(decomposer.get_component_info())
    
    # Check RMSE at edges
    edge_n = 7 * freq  # Last 7 days
    print(f"\nRMSE (last 7 days vs full signal):")
    
    true_components = [
        (subdaily, 'y_band_0', 'Sub-daily'),
        (daily, 'y_band_1', 'Daily'),
        (weekly, 'y_band_2', 'Weekly'),
        (monthly, 'y_band_3', 'Monthly'),
    ]
    
    for true_val, col, name in true_components:
        rmse_full = np.sqrt(np.mean((df_decomposed[col] - true_val) ** 2))
        rmse_edge = np.sqrt(np.mean((df_decomposed[col].iloc[-edge_n:] - true_val[-edge_n:]) ** 2))
        print(f"  {name:<12}: Full={rmse_full:.4f}, Edge={rmse_edge:.4f}")
