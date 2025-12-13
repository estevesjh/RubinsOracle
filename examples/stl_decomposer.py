"""
STL LOESS Decomposer Module

Seasonal-Trend decomposition using LOESS (Locally Estimated Scatterplot Smoothing).

This module provides multi-scale signal decomposition using iterative STL decomposition.
Each frequency band is extracted by running STL with the appropriate period, then
subtracting from the residual to get the next band.

Output components (high freq → low freq):
    - y_period_Xd: Seasonal component at X days period
    - y_trend: Long-term trend (remainder after all seasonal extractions)
"""

import warnings
from typing import Literal, List, Tuple, Optional
import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.seasonal import STL
    STL_AVAILABLE = True
except ImportError:
    STL_AVAILABLE = False
    warnings.warn(
        "statsmodels not installed. STLDecomposer will not be available. "
        "Install with: pip install statsmodels"
    )


class STLDecomposer:
    """Multi-scale signal decomposition using iterative STL (LOESS).
    
    This decomposer extracts multiple frequency bands by iteratively applying
    STL decomposition. For each period, it extracts the seasonal component
    at that frequency scale.
    
    The approach:
    1. Start with the original signal
    2. For each period (from shortest to longest):
       - Apply STL with that period
       - Extract the seasonal component as that frequency band
       - Continue with the residual + trend for next iteration
    3. The final trend becomes the lowest frequency band
    
    Attributes:
        freq: Data frequency in observations per day
        periods: List of periods in days to extract
        
    Example:
        >>> decomposer = STLDecomposer(freq=96, periods=[0.5, 1, 7, 28])
        >>> df_decomposed = decomposer.decompose(df)
    """
    
    def __init__(
        self,
        freq: int = 96,
        periods: Optional[List[float]] = None,
        mode: Literal['drop', 'keep'] = 'drop',
        # STL parameters
        seasonal: int = 7,
        trend: Optional[int] = None,
        low_pass: Optional[int] = None,
        seasonal_deg: int = 1,
        trend_deg: int = 1,
        low_pass_deg: int = 1,
        robust: bool = False,
        seasonal_jump: int = 1,
        trend_jump: int = 1,
        low_pass_jump: int = 1,
    ):
        """Initialize the STL Decomposer.
        
        Args:
            freq: Number of observations per day
                - 15-min data: 96 (default)
                - 30-min data: 48
                - Hourly data: 24
                - Daily data: 1
            periods: List of periods in days to extract.
                Example: [0.5, 1, 7, 28] for sub-daily, daily, weekly, monthly
                If None, uses default periods [0.5, 1, 7, 28]
            mode: How to handle NaN values
                - 'drop': Remove rows with NaN
                - 'keep': Keep all rows (interpolate NaN for STL)
            seasonal: Length of the seasonal smoother (must be odd, >= 3)
            trend: Length of the trend smoother (must be odd, default: computed)
            low_pass: Length of the low-pass filter (must be odd, default: computed)
            seasonal_deg: Degree of seasonal LOESS (0 or 1)
            trend_deg: Degree of trend LOESS (0 or 1)
            low_pass_deg: Degree of low-pass LOESS (0 or 1)
            robust: If True, use robust fitting (downweights outliers)
            seasonal_jump: Seasonal smoother jump (larger = faster, less accurate)
            trend_jump: Trend smoother jump
            low_pass_jump: Low-pass smoother jump
        """
        if not STL_AVAILABLE:
            raise ImportError(
                "statsmodels is required for STLDecomposer. "
                "Install with: pip install statsmodels"
            )
        
        self.freq = freq
        self.mode = mode
        
        # Set default periods if not provided
        if periods is None:
            self.periods = [0.5, 1.0, 7.0, 28.0]
        else:
            # Sort periods from shortest to longest
            self.periods = sorted(periods)
        
        # STL parameters
        self.seasonal = seasonal
        self.trend = trend
        self.low_pass = low_pass
        self.seasonal_deg = seasonal_deg
        self.trend_deg = trend_deg
        self.low_pass_deg = low_pass_deg
        self.robust = robust
        self.seasonal_jump = seasonal_jump
        self.trend_jump = trend_jump
        self.low_pass_jump = low_pass_jump
        
        # Store filter specs for reporting
        self._filter_specs = None
    
    def _calculate_stl_periods(self) -> List[dict]:
        """Calculate STL periods from periods list.
        
        Returns:
            List of dicts with period info for each band
        """
        specs = []
        
        for period_days in self.periods:
            # Convert to observations
            period_obs = int(round(period_days * self.freq))
            
            # Ensure period is at least 2
            period_obs = max(2, period_obs)
            
            specs.append({
                'name': f'period_{period_days}d',
                'period_days': period_days,
                'period_obs': period_obs,
            })
        
        return specs
    
    def _ensure_odd(self, n: int) -> int:
        """Ensure a number is odd (required for STL smoothers)."""
        if n % 2 == 0:
            return n + 1
        return n
    
    def _apply_stl(
        self,
        y: np.ndarray,
        period: int,
        name: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply STL decomposition.
        
        Args:
            y: Input signal
            period: Seasonal period in observations
            name: Component name (for logging)
            
        Returns:
            Tuple of (seasonal, trend, residual) components
        """
        try:
            # Ensure period is valid
            if period < 2:
                warnings.warn(f"{name}: Period {period} too small, using 2")
                period = 2
            
            if period > len(y) // 2:
                warnings.warn(
                    f"{name}: Period {period} too large for data length {len(y)}. "
                    f"Using {len(y) // 2}"
                )
                period = len(y) // 2
            
            # Calculate seasonal smoother length (must be odd, >= 7 recommended)
            seasonal = self._ensure_odd(max(7, period // 2 * 2 + 1))
            
            # Apply STL
            stl = STL(
                y,
                period=period,
                seasonal=seasonal,
                trend=self.trend,
                low_pass=self.low_pass,
                seasonal_deg=self.seasonal_deg,
                trend_deg=self.trend_deg,
                low_pass_deg=self.low_pass_deg,
                robust=self.robust,
                seasonal_jump=self.seasonal_jump,
                trend_jump=self.trend_jump,
                low_pass_jump=self.low_pass_jump,
            )
            
            result = stl.fit()
            
            return result.seasonal, result.trend, result.resid
            
        except Exception as e:
            warnings.warn(f"Error applying STL for {name}: {e}")
            return np.zeros_like(y), y.copy(), np.zeros_like(y)
    
    def decompose(self, df: pd.DataFrame) -> pd.DataFrame:
        """Decompose time series using iterative STL.
        
        Args:
            df: Input dataframe with columns:
                - ds (datetime): Timestamps
                - y (float): Target values
                
        Returns:
            DataFrame with original columns plus decomposed components:
                - y_period_Xd for each period
                - y_trend for the final trend
        """
        df = df.copy()
        
        # Ensure 'y' column exists
        if 'y' not in df.columns:
            raise ValueError("DataFrame must have a 'y' column")
        
        # Handle NaN values
        if self.mode == 'drop':
            df = df.dropna(subset=['y']).reset_index(drop=True)
        
        # Fill NaN for processing (STL requires complete data)
        y = df['y'].ffill().bfill().values.astype(float)
        
        # Calculate STL periods
        self._filter_specs = self._calculate_stl_periods()
        
        print(f"\n{'='*70}")
        print(f"Signal Decomposition - STL (LOESS) Method")
        print(f"{'='*70}")
        print(f"Data frequency: {self.freq} obs/day")
        print(f"Data length: {len(y)} observations ({len(y)/self.freq:.1f} days)")
        print(f"Periods to extract: {self.periods} days")
        print(f"Robust: {self.robust}")
        print(f"{'='*70}\n")
        
        # Iterative decomposition
        # Strategy: Extract bands from highest frequency to lowest
        # Each iteration: STL extracts seasonal at that scale, rest goes to next
        
        current_signal = y.copy()
        
        for idx, spec in enumerate(self._filter_specs):
            name = spec['name']
            period_obs = spec['period_obs']
            period_days = spec['period_days']
            
            print(f"Computing {name} (period={period_days}d, {period_obs} obs)...")
            
            # Check if period is valid for remaining signal
            if period_obs > len(current_signal) // 2:
                print(f"  → Skipping: period too large for remaining data")
                continue
            
            # Apply STL to extract seasonal component at this scale
            seasonal, trend, resid = self._apply_stl(
                current_signal, period_obs, name
            )
            
            # This band is the seasonal component
            df[f'y_{name}'] = seasonal
            
            # Continue with trend + residual for next iteration
            current_signal = trend + resid
            
            print(f"  → Extracted seasonal (std={np.std(seasonal):.4f})")
        
        # Final remaining signal is the trend
        df['y_trend'] = current_signal
        print(f"Computing trend (remaining signal)...")
        print(f"  → Trend (std={np.std(current_signal):.4f})")
        
        print(f"\n{'='*70}")
        print(f"Decomposition complete. Dataset shape: {df.shape}")
        print(f"{'='*70}\n")
        
        return df
    
    def get_component_info(self) -> pd.DataFrame:
        """Get information about the decomposed components.
        
        Returns:
            DataFrame with component specifications
        """
        if self._filter_specs is None:
            self._filter_specs = self._calculate_stl_periods()
        
        info = []
        for spec in self._filter_specs:
            info.append({
                'component': f"y_{spec['name']}",
                'period_days': spec['period_days'],
                'period_obs': spec['period_obs'],
            })
        
        # Add trend
        info.append({
            'component': 'y_trend',
            'period_days': None,
            'period_obs': None,
        })
        
        return pd.DataFrame(info)
    
    def reconstruct(self, df: pd.DataFrame) -> np.ndarray:
        """Reconstruct the original signal from components.
        
        Args:
            df: DataFrame with decomposed components
            
        Returns:
            Reconstructed signal
        """
        if self._filter_specs is None:
            self._filter_specs = self._calculate_stl_periods()
        
        reconstructed = np.zeros(len(df))
        
        # Add all period components
        for spec in self._filter_specs:
            col = f"y_{spec['name']}"
            if col in df.columns:
                reconstructed += df[col].values
        
        # Add trend
        if 'y_trend' in df.columns:
            reconstructed += df['y_trend'].values
        
        return reconstructed


# ============================================================================
# Test code
# ============================================================================
if __name__ == "__main__":
    """Example usage and testing"""
    
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-07-01', freq='15min')
    n = len(dates)
    t = np.arange(n)
    freq = 96
    
    # Synthetic signal
    y = (
        6 * np.sin(2 * np.pi * t / (freq * 0.5)) +      # Sub-daily (12h)
        10 * np.sin(2 * np.pi * t / freq) +              # Daily (24h)
        5 * np.sin(2 * np.pi * t / (freq * 7)) +         # Weekly
        3 * np.sin(2 * np.pi * t / (freq * 28)) +        # Monthly
        np.random.randn(n) * 0.5 +                       # Noise
        20 + 0.01 * t                                     # Trend
    )
    
    df = pd.DataFrame({'ds': dates, 'y': y})
    
    print("Testing STL Decomposer:")
    print("=" * 80)
    
    # Test with periods list
    decomposer = STLDecomposer(
        freq=freq,
        periods=[0.5, 1.0, 7.0, 28.0],
        mode='keep',
        robust=False
    )
    
    df_decomposed = decomposer.decompose(df)
    
    print("\nComponent info:")
    print(decomposer.get_component_info().to_string(index=False))
    
    print("\nOutput columns:")
    print([c for c in df_decomposed.columns if c.startswith('y_')])
    
    # Test reconstruction
    reconstructed = decomposer.reconstruct(df_decomposed)
    reconstruction_error = np.mean((df_decomposed['y'].values - reconstructed) ** 2)
    print(f"\nReconstruction MSE: {reconstruction_error:.6f}")
    
    # Compare with true components
    print("\n" + "="*70)
    print("COMPONENT EXTRACTION ACCURACY (vs True)")
    print("="*70)
    
    true_components = [
        ('y_period_0.5d', 6 * np.sin(2 * np.pi * t / (freq * 0.5)), 'Sub-daily (0.5d)'),
        ('y_period_1.0d', 10 * np.sin(2 * np.pi * t / freq), 'Daily (1d)'),
        ('y_period_7.0d', 5 * np.sin(2 * np.pi * t / (freq * 7)), 'Weekly (7d)'),
        ('y_period_28.0d', 3 * np.sin(2 * np.pi * t / (freq * 28)), 'Monthly (28d)'),
    ]
    
    for col, true_val, name in true_components:
        if col in df_decomposed.columns:
            rmse = np.sqrt(np.mean((df_decomposed[col].values - true_val) ** 2))
            print(f"{name:<20}: RMSE = {rmse:.4f}")
