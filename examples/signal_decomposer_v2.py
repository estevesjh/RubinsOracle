"""
Signal Decomposition Module

Provides flexible time series decomposition using either Savitzky-Golay or Butterworth filters
to extract multi-frequency components from signals.
"""

import warnings
from typing import Literal, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, butter, filtfilt


class SignalDecomposer:
    """Decomposes time series into multiple frequency components.
    
    Supports two filtering methods:
    - Savitzky-Golay: Non-parametric smoothing filter (good for trends)
    - Butterworth: IIR bandpass filter (good for frequency isolation)
    
    The decomposer creates bandpass filters based on specified day periods,
    automatically converting them to the appropriate frequency bands.
    
    Attributes:
        freq: Data frequency in observations per day (e.g., 4 for 15min, 24 for hourly)
        periods_days: List of day periods defining frequency bands
        filter_type: Type of filter ('savgol' or 'butterworth')
        mode: How to handle NaN values ('drop' or 'keep')
        
    Example:
        >>> # Define periods: sub-daily, daily, 3-day, weekly, bi-weekly, monthly
        >>> periods = [1, 3, 7, 14, 28]
        >>> 
        >>> # Using Savitzky-Golay filter
        >>> decomposer = SignalDecomposer(
        ...     freq=4,  # 15-minute data
        ...     periods_days=periods,
        ...     filter_type='savgol'
        ... )
        >>> df_decomposed = decomposer.decompose(df)
        >>> 
        >>> # Using Butterworth filter
        >>> decomposer = SignalDecomposer(
        ...     freq=4,
        ...     periods_days=periods,
        ...     filter_type='butterworth',
        ...     butter_order=4
        ... )
        >>> df_decomposed = decomposer.decompose(df)
    """
    
    def __init__(
        self,
        freq: int = 4,
        periods_days: Optional[List[float]] = None,
        filter_type: Literal['savgol', 'butterworth'] = 'savgol',
        mode: Literal['drop', 'keep'] = 'drop',
        # Savitzky-Golay specific parameters
        savgol_polyorder: int = 3,
        savgol_mode: Literal['mirror', 'nearest', 'constant', 'wrap', 'interp'] = 'nearest',
        # Butterworth specific parameters
        butter_order: int = 4,
        butter_padlen: Optional[int] = None,
    ):
        """Initialize the signal decomposer.
        
        Args:
            freq: Number of observations per day (4 = 15min, 24 = hourly)
            periods_days: List of period lengths in days defining frequency bands.
                Example: [1, 3, 7, 14, 28] creates bands:
                    - < 1 day (high frequency)
                    - 1-3 days
                    - 3-7 days
                    - 7-14 days
                    - 14-28 days
                    - > 28 days (low frequency/trend)
                If None, uses default: [0.5, 1, 3, 6, 11, 14, 28, 56]
            filter_type: Type of filter to use:
                - 'savgol': Savitzky-Golay filter (non-causal, good for smooth trends)
                - 'butterworth': Butterworth bandpass filter (causal option, good for oscillations)
            mode: How to handle NaN values created by filtering:
                - 'drop': Remove rows with NaN in decomposed components
                - 'keep': Keep all rows (NaN will be at boundaries)
            savgol_polyorder: Polynomial order for Savitzky-Golay filter
            savgol_mode: Boundary mode for Savitzky-Golay filter
            butter_order: Order of Butterworth filter (higher = sharper cutoff)
            butter_padlen: Padding length for filtfilt (None = automatic)
        """
        self.freq = freq
        self.filter_type = filter_type
        self.mode = mode
        
        # Set default periods if not provided
        if periods_days is None:
            self.periods_days = [0.5, 1, 3, 6, 11, 14, 28, 56]
        else:
            self.periods_days = sorted(periods_days)
        
        # Savitzky-Golay parameters
        self.savgol_polyorder = savgol_polyorder
        self.savgol_mode = savgol_mode
        
        # Butterworth parameters
        self.butter_order = butter_order
        self.butter_padlen = butter_padlen
        
        # Calculate filter specifications
        self._filter_specs = self._calculate_filter_specs()
        
    def _calculate_filter_specs(self) -> List[dict]:
        """Calculate filter specifications for each frequency band.
        
        Returns:
            List of dictionaries with filter specifications:
                - name: Component name (e.g., 'p0', 'p1', ...)
                - period_days: Period in days
                - period_obs: Period in observations
                - For savgol: window_length
                - For butterworth: lowcut, highcut (in Hz)
        """
        specs = []
        
        for i, period_days in enumerate(self.periods_days):
            period_obs = period_days * self.freq
            
            spec = {
                'name': f'p{i}',
                'period_days': period_days,
                'period_obs': period_obs,
            }
            
            if self.filter_type == 'savgol':
                # Window length: approximately the period length
                # Use a multiplier to smooth over the period
                window_length = int(period_obs * 1.5)
                # Ensure odd
                if window_length % 2 == 0:
                    window_length += 1
                # Ensure larger than polyorder
                window_length = max(window_length, self.savgol_polyorder + 2)
                spec['window_length'] = window_length
                
            elif self.filter_type == 'butterworth':
                # Convert period to frequency (Hz)
                # Nyquist frequency is freq/2 (half the sampling rate)
                nyquist = self.freq / 2
                
                # Critical frequencies for bandpass
                # For a period P, we want to isolate frequencies around 1/P
                # Create a band: [1/(1.5*P), 1/(0.5*P)]
                f_low = 1 / (period_days * 1.5)  # Lower bound
                f_high = 1 / (period_days * 0.5)  # Upper bound
                
                # Normalize by Nyquist frequency
                spec['lowcut'] = f_low / nyquist
                spec['highcut'] = f_high / nyquist
                
                # Ensure frequencies are valid (0 < f < 1)
                spec['lowcut'] = max(0.001, min(0.999, spec['lowcut']))
                spec['highcut'] = max(0.001, min(0.999, spec['highcut']))
                
                # Ensure lowcut < highcut
                if spec['lowcut'] >= spec['highcut']:
                    spec['lowcut'] = spec['highcut'] * 0.5
            
            specs.append(spec)
        
        return specs
    
    def _apply_savgol_lowpass(
        self,
        y: np.ndarray,
        window_length: int,
        name: str
    ) -> np.ndarray:
        """Apply Savitzky-Golay filter as a lowpass filter.
        
        Args:
            y: Input signal
            window_length: Filter window length
            name: Component name (for logging)
            
        Returns:
            Smoothed signal (trend component)
        """
        if len(y) < window_length:
            warnings.warn(
                f"Data length ({len(y)}) is shorter than window length ({window_length}) "
                f"for {name} component. Returning NaN array."
            )
            return np.full_like(y, np.nan)
        
        try:
            filtered = savgol_filter(
                y,
                window_length=window_length,
                polyorder=self.savgol_polyorder,
                mode=self.savgol_mode
            )
            
            # Set NaN at boundaries if using interp mode
            if self.savgol_mode == 'interp':
                half_window = window_length // 2
                filtered[:half_window] = np.nan
                filtered[-half_window:] = np.nan
            
            return filtered
            
        except Exception as e:
            warnings.warn(f"Error applying Savitzky-Golay filter for {name}: {e}")
            return np.full_like(y, np.nan)
    
    def _apply_butterworth_bandpass(
        self,
        y: np.ndarray,
        lowcut: float,
        highcut: float,
        name: str
    ) -> np.ndarray:
        """Apply Butterworth bandpass filter.
        
        Args:
            y: Input signal
            lowcut: Low cutoff frequency (normalized by Nyquist)
            highcut: High cutoff frequency (normalized by Nyquist)
            name: Component name (for logging)
            
        Returns:
            Bandpass filtered signal
        """
        try:
            # Design Butterworth bandpass filter
            sos = butter(
                self.butter_order,
                [lowcut, highcut],
                btype='band',
                output='sos'
            )
            
            # Apply filter (forward-backward to avoid phase shift)
            filtered = filtfilt(
                sos,
                y,
                padlen=self.butter_padlen,
                axis=0
            )
            
            return filtered
            
        except Exception as e:
            warnings.warn(f"Error applying Butterworth filter for {name}: {e}")
            return np.full_like(y, np.nan)
    
    def _apply_butterworth_lowpass(
        self,
        y: np.ndarray,
        cutoff: float,
        name: str
    ) -> np.ndarray:
        """Apply Butterworth lowpass filter.
        
        Args:
            y: Input signal
            cutoff: Cutoff frequency (normalized by Nyquist)
            name: Component name (for logging)
            
        Returns:
            Lowpass filtered signal (trend component)
        """
        try:
            # Design Butterworth lowpass filter
            sos = butter(
                self.butter_order,
                cutoff,
                btype='low',
                output='sos'
            )
            
            # Apply filter
            filtered = filtfilt(
                sos,
                y,
                padlen=self.butter_padlen,
                axis=0
            )
            
            return filtered
            
        except Exception as e:
            warnings.warn(f"Error applying Butterworth lowpass filter for {name}: {e}")
            return np.full_like(y, np.nan)
    
    def decompose(self, df: pd.DataFrame) -> pd.DataFrame:
        """Decompose time series into multi-frequency components.
        
        The decomposition creates:
        1. Trend components at different time scales (y_p0_trend, y_p1_trend, ...)
        2. Bandpass components between consecutive trends (y_p0, y_p1, ...)
        3. High-frequency residual (y_high)
        4. Low-frequency trend (y_trend)
        
        Args:
            df: Input dataframe with columns:
                - ds (datetime): Timestamps
                - y (float): Target values (or tempMean as fallback)
                
        Returns:
            DataFrame with original columns plus decomposed components:
                - y_p{i}_trend: Lowpass filtered at period i
                - y_p{i}: Bandpass component between periods i and i+1
                - y_high: High-frequency residual (< shortest period)
                - y_trend: Low-frequency trend (> longest period)
                
        Raises:
            ValueError: If required columns are missing
        """
        df = df.copy()
        
        # Ensure 'y' exists
        if 'y' not in df.columns:
            if 'tempMean' in df.columns:
                df['y'] = df['tempMean']
            else:
                raise ValueError("Column 'y' or 'tempMean' not found in dataframe.")
        
        # Fill NaN values for filtering
        y_filled = df['y'].fillna(method='ffill').fillna(method='bfill').values
        
        print(f"\n{'='*70}")
        print(f"Signal Decomposition - {self.filter_type.upper()} Filter")
        print(f"{'='*70}")
        print(f"Data frequency: {self.freq} obs/day")
        print(f"Data length: {len(y_filled)} observations")
        print(f"Period bands: {self.periods_days} days")
        print(f"{'='*70}\n")
        
        # Store trend components
        trends = {}
        
        if self.filter_type == 'savgol':
            # Apply Savitzky-Golay filters (lowpass at each period)
            for spec in self._filter_specs:
                name = spec['name']
                window = spec['window_length']
                period = spec['period_days']
                
                print(f"Computing {name} trend (period={period:.1f}d, window={window})...")
                trends[name] = self._apply_savgol_lowpass(y_filled, window, name)
                df[f'y_{name}_trend'] = trends[name]
        
        elif self.filter_type == 'butterworth':
            # Apply Butterworth lowpass filters at each period
            for spec in self._filter_specs:
                name = spec['name']
                # For lowpass, use the highcut frequency (filters out faster components)
                cutoff = spec['highcut']
                period = spec['period_days']
                
                print(f"Computing {name} trend (period={period:.1f}d, cutoff={cutoff:.4f})...")
                trends[name] = self._apply_butterworth_lowpass(y_filled, cutoff, name)
                df[f'y_{name}_trend'] = trends[name]
        
        # Calculate bandpass components as differences between consecutive trends
        print("\nCalculating frequency band components...")
        
        # High-frequency: original - first trend
        df['y_high'] = df['y'] - df[f'y_{self._filter_specs[0]["name"]}_trend']
        print(f"  y_high: residual (< {self.periods_days[0]:.1f}d)")
        
        # Bandpass components: difference between consecutive trends
        for i in range(len(self._filter_specs) - 1):
            name_current = self._filter_specs[i]['name']
            name_next = self._filter_specs[i + 1]['name']
            period_low = self.periods_days[i]
            period_high = self.periods_days[i + 1]
            
            df[f'y_{name_current}'] = (
                df[f'y_{name_current}_trend'] - df[f'y_{name_next}_trend']
            )
            print(f"  y_{name_current}: {period_low:.1f}-{period_high:.1f} day band")
        
        # Low-frequency trend: last trend component
        last_name = self._filter_specs[-1]['name']
        df['y_trend'] = df[f'y_{last_name}_trend']
        print(f"  y_trend: > {self.periods_days[-1]:.1f}d (from {last_name})")
        
        # Handle NaN values based on mode
        if self.mode == 'drop':
            component_cols = ['y_high'] + [f'y_{spec["name"]}' for spec in self._filter_specs[:-1]] + ['y_trend']
            initial_len = len(df)
            df = df.dropna(subset=component_cols)
            final_len = len(df)
            
            if final_len < initial_len:
                print(f"\nDropped {initial_len - final_len} rows with NaN in decomposed components")
        
        print(f"\n{'='*70}")
        print(f"Decomposition complete. Dataset shape: {df.shape}")
        print(f"{'='*70}\n")
        
        return df
    
    def get_component_info(self) -> pd.DataFrame:
        """Get information about the decomposed components.
        
        Returns:
            DataFrame with component specifications
        """
        info = []
        
        # High frequency
        info.append({
            'component': 'y_high',
            'period_low_days': 0,
            'period_high_days': self.periods_days[0],
            'description': f'High-frequency residual (< {self.periods_days[0]:.1f}d)'
        })
        
        # Bandpass components
        for i in range(len(self.periods_days) - 1):
            name = f'y_p{i}'
            info.append({
                'component': name,
                'period_low_days': self.periods_days[i],
                'period_high_days': self.periods_days[i + 1],
                'description': f'{self.periods_days[i]:.1f}-{self.periods_days[i + 1]:.1f} day band'
            })
        
        # Low frequency trend
        info.append({
            'component': 'y_trend',
            'period_low_days': self.periods_days[-1],
            'period_high_days': np.inf,
            'description': f'Low-frequency trend (> {self.periods_days[-1]:.1f}d)'
        })
        
        return pd.DataFrame(info)


def preprocess_for_forecast(
    df: pd.DataFrame,
    decompose: bool = True,
    freq: int = 4,
    periods_days: Optional[List[float]] = None,
    filter_type: Literal['savgol', 'butterworth'] = 'savgol',
    train_start_date: str | pd.Timestamp | None = None,
    train_end_date: str | pd.Timestamp | None = None,
    lag_days: int | None = None,
    **decomposer_kwargs
) -> pd.DataFrame:
    """Preprocess data for forecasting with optional decomposition and date filtering.
    
    This function prepares time series data by:
    1. Optionally applying signal decomposition to extract frequency components
    2. Filtering data to a specified date range
    3. Ensuring sufficient history for lagged features
    
    Args:
        df: Input dataframe with 'ds' and 'y' columns
        decompose: Whether to apply signal decomposition
        freq: Data frequency in observations per day (for decomposition)
        periods_days: List of period lengths in days for decomposition
        filter_type: Type of filter ('savgol' or 'butterworth')
        train_start_date: Start date for training data (inclusive)
        train_end_date: End date for training data (inclusive)
        lag_days: Number of historical observations needed for AR models
        **decomposer_kwargs: Additional arguments for SignalDecomposer
        
    Returns:
        Preprocessed dataframe with filtered date range and optional decomposition
        
    Example:
        >>> # Using Butterworth filter with custom periods
        >>> df_train = preprocess_for_forecast(
        ...     df,
        ...     decompose=True,
        ...     freq=4,
        ...     periods_days=[1, 3, 7, 14, 28],
        ...     filter_type='butterworth',
        ...     train_start_date='2023-09-10',
        ...     lag_days=48
        ... )
    """
    df = df.copy()
    
    # Ensure ds is datetime
    if 'ds' in df.columns:
        df['ds'] = pd.to_datetime(df['ds'])
    
    # Sort by timestamp
    df = df.sort_values('ds').reset_index(drop=True)
    
    # Apply decomposition first on full history (if requested)
    if decompose:
        decomposer = SignalDecomposer(
            freq=freq,
            periods_days=periods_days,
            filter_type=filter_type,
            mode='drop',
            **decomposer_kwargs
        )
        df = decomposer.decompose(df)
    
    # Filter by date range
    if train_end_date is not None:
        train_end_date = pd.to_datetime(train_end_date)
        df = df[df['ds'] <= train_end_date]
    
    if train_start_date is not None:
        train_start_date = pd.to_datetime(train_start_date)
        df = df[df['ds'] >= train_start_date]
    elif lag_days is not None and train_end_date is not None:
        # Calculate start date based on lag_days
        if 'ds' in df.columns and len(df) > 1:
            time_diffs = df['ds'].diff().dropna()
            median_diff = time_diffs.median()
            lookback_period = median_diff * lag_days
            calculated_start = train_end_date - lookback_period - pd.Timedelta(days=1)
            
            df = df[df['ds'] >= calculated_start]
            print(f"Automatically set training start date to {calculated_start} "
                  f"to ensure {lag_days} lag observations")
    
    # Validate we have enough data
    if lag_days is not None and len(df) < lag_days:
        warnings.warn(
            f"After preprocessing, only {len(df)} observations remain, "
            f"but {lag_days} lag observations were requested. "
            f"This may cause issues with AR models."
        )
    
    return df.reset_index(drop=True)


if __name__ == "__main__":
    """Example usage and testing"""
    
    # Create synthetic data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='15min')
    
    # Synthetic signal with multiple frequencies
    t = np.arange(len(dates))
    y = (
        10 * np.sin(2 * np.pi * t / (4 * 24)) +  # Daily cycle
        5 * np.sin(2 * np.pi * t / (4 * 24 * 7)) +  # Weekly cycle
        3 * np.sin(2 * np.pi * t / (4 * 24 * 28)) +  # Monthly cycle
        np.random.randn(len(dates)) * 0.5 +  # Noise
        20  # Offset
    )
    
    df = pd.DataFrame({'ds': dates, 'y': y})
    
    print("Testing Savitzky-Golay decomposition:")
    print("=" * 80)
    decomposer_sg = SignalDecomposer(
        freq=4,
        periods_days=[1, 7, 14, 28],
        filter_type='savgol'
    )
    df_sg = decomposer_sg.decompose(df.copy())
    print("\nComponent info:")
    print(decomposer_sg.get_component_info())
    
    print("\n\nTesting Butterworth decomposition:")
    print("=" * 80)
    decomposer_bw = SignalDecomposer(
        freq=4,
        periods_days=[1, 7, 14, 28],
        filter_type='butterworth',
        butter_order=4
    )
    df_bw = decomposer_bw.decompose(df.copy())
    print("\nComponent info:")
    print(decomposer_bw.get_component_info())
    
    print("\n\nTesting preprocessing function:")
    print("=" * 80)
    df_processed = preprocess_for_forecast(
        df,
        decompose=True,
        freq=4,
        periods_days=[1, 3, 7, 14],
        filter_type='savgol',
        train_end_date='2023-12-01',
        lag_days=96
    )
    print(f"Processed shape: {df_processed.shape}")
    print(f"Columns: {df_processed.columns.tolist()}")

def savgol_filter_bandpass(y, low_window, high_window, polyorder=3, mode='nearest'):
    """Apply Savitzky-Golay bandpass filter by subtracting two lowpass filters."""
    lowpass_high = savgol_filter(
        y,
        window_length=high_window,
        polyorder=polyorder,
        mode=mode
    )
    lowpass_low = savgol_filter(
        y,
        window_length=low_window,
        polyorder=polyorder,
        mode=mode
    )
    bandpass = lowpass_low - lowpass_high
    return bandpass

savgol_filter_bandpass(y, periods[0], periods[1], polyorder=3, mode='nearest')