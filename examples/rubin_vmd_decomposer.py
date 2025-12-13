"""
Rubin VMD Decomposer Module

Two-stage VMD + Butterworth decomposition specifically designed for Rubin temperature data.

Stage 1: VMD on original signal → Extract IMF1 (lowest freq) → Apply Butterworth bandpass
         - Daily: [0.75, 1.25] days
         - Weekly: [1.5, 7] days  
         - Monthly: [7, 30] days
         - Trend: [30, 200] days

Stage 2: VMD on residual (signal - IMF1) → Extract high freq components
         - Sub-daily: IMF2 (2nd mode from residual VMD)
         - IMF3: Highest frequency (noise-like)

Final output (high freq → low freq):
    - y_imf3: Highest frequency component (from residual VMD)
    - y_subdaily: Sub-daily component (IMF2 from residual VMD)
    - y_vmd_daily: [0.75, 1.25] days band (Butterworth on IMF1)
    - y_vmd_weekly: [1.5, 7] days band (Butterworth on IMF1)
    - y_vmd_monthly: [7, 30] days band (Butterworth on IMF1)
    - y_vmd_trend: [30, 200] days band (Butterworth on IMF1)
"""

import warnings
from typing import Literal, List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

try:
    from vmdpy import VMD
    VMDPY_AVAILABLE = True
except ImportError:
    VMDPY_AVAILABLE = False
    warnings.warn(
        "vmdpy not installed. RubinVMDDecomposer will not be available. "
        "Install with: pip install vmdpy"
    )


class RubinVMDDecomposer:
    """Two-stage VMD + Butterworth decomposition for Rubin temperature data.
    
    This decomposer uses a hybrid approach:
    1. VMD extracts the slow component (IMF1) from the original signal
    2. Butterworth bandpass filters separate IMF1 into Daily, Weekly, Monthly, Trend bands
    3. VMD on the residual extracts high-frequency components (sub-daily, noise)
    
    This approach is specifically tuned for the Rubin temperature dataset characteristics.
    
    Attributes:
        freq: Data frequency in observations per day (default: 96 for 15-min data)
        alpha: VMD bandwidth constraint (default: 2000, higher = narrower bands)
        
    Example:
        >>> decomposer = RubinVMDDecomposer(freq=96, alpha=2000)
        >>> df_decomposed = decomposer.decompose(df)
        >>> # Result has columns: y_imf3, y_subdaily, y_vmd_daily, y_vmd_weekly, y_vmd_monthly, y_vmd_trend
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
    ):
        """Initialize the Rubin VMD Decomposer.
        
        Args:
            freq: Number of observations per day
                - 15-min data: 96 (default)
                - 30-min data: 48
                - Hourly data: 24
            alpha: VMD bandwidth constraint parameter
                - Higher alpha = narrower bandwidth modes
                - Default: 2000 (good for temperature data)
            K_stage1: Number of modes for first VMD (on original signal)
                - Default: 5 (we only use IMF1)
            K_stage2: Number of modes for second VMD (on residual)
                - Default: 3 (we use IMF2 for sub-daily, IMF3 for noise)
            tau: Noise tolerance (0 = no noise)
            DC: Whether to include DC component (0 = no, 1 = yes)
            init: Initialization method (1 = uniform)
            tol: Convergence tolerance
            butter_order: Order of Butterworth filters for bandpass
            butter_margin: Frequency margin for Butterworth (default: 0.1 = 10%)
        """
        if not VMDPY_AVAILABLE:
            raise ImportError(
                "vmdpy is required for RubinVMDDecomposer. "
                "Install with: pip install vmdpy"
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
        
        # Butterworth period pairs for IMF1 decomposition
        # Now includes daily band
        self.butter_periods = [
            (0.5, 2.0),   # Daily band (18-30 hours)
            (1.5, 9.0),     # Weekly band
            (21.0, 38.0),    # Monthly band
            (30.0, 200.0),  # Trend band
        ]
        
        self.band_names = ['vmd_daily', 'vmd_weekly', 'vmd_monthly', 'vmd_trend']
    
    def _run_vmd(self, signal: np.ndarray, K: int) -> np.ndarray:
        """Run VMD decomposition.
        
        Args:
            signal: Input signal
            K: Number of modes to extract
            
        Returns:
            Array of shape (K, len(signal)) containing the modes
            Modes are ordered from lowest frequency (IMF1) to highest (IMF_K)
        """
        original_length = len(signal)
        
        # VMD returns: u (modes), u_hat (spectra), omega (center frequencies over iterations)
        u, u_hat, omega = VMD(
            signal,
            self.alpha,
            self.tau,
            K,
            self.DC,
            self.init,
            self.tol
        )
        
        # u has shape (K, N) - but N might be different from input length
        # omega has shape (iterations, K) - take final row for converged center frequencies
        final_omega = omega[-1, :]  # Final center frequencies for each mode
        
        # Sort modes by center frequency (ascending: lowest freq first)
        sorted_indices = np.argsort(final_omega)
        u_sorted = u[sorted_indices]
        
        # Ensure output length matches input (VMD sometimes produces slightly shorter output)
        if u_sorted.shape[1] != original_length:
            # Pad or trim to match original length
            new_u = np.zeros((K, original_length))
            min_len = min(u_sorted.shape[1], original_length)
            new_u[:, :min_len] = u_sorted[:, :min_len]
            
            # Pad with edge values if needed
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
        is_trend: bool = False
    ) -> np.ndarray:
        """Apply Butterworth bandpass filter.
        
        Args:
            signal: Input signal
            period_low: Lower period bound in days
            period_high: Upper period bound in days
            name: Band name (for logging)
            is_trend: If True, use lowpass from 0 to cutoff
            
        Returns:
            Bandpass filtered signal (same length as input)
        """
        try:
            # Convert periods to frequencies
            period_low_obs = period_low * self.freq
            period_high_obs = period_high * self.freq
            
            # Nyquist frequency
            nyquist = 0.5
            
            # Calculate frequency bounds
            f_low = 1 / period_high_obs  # Lower cutoff (longer period)
            f_high = 1 / period_low_obs  # Higher cutoff (shorter period)
            
            # Apply margin
            f_low_margin = f_low * (1 - self.butter_margin)
            f_high_margin = f_high * (1 + self.butter_margin)
            
            # Normalize by Nyquist
            lowcut = f_low_margin / nyquist
            highcut = f_high_margin / nyquist
            
            # Ensure valid frequencies
            lowcut = max(0.001, min(0.999, lowcut))
            highcut = max(0.001, min(0.999, highcut))
            
            if is_trend or lowcut >= highcut:
                # Use lowpass for trend
                sos = butter(self.butter_order, highcut, btype='low', output='sos')
            else:
                # Normal bandpass
                sos = butter(self.butter_order, [lowcut, highcut], btype='band', output='sos')
            
            filtered = sosfiltfilt(sos, signal)
            
            # Ensure output length matches input
            if len(filtered) != len(signal):
                if len(filtered) > len(signal):
                    filtered = filtered[:len(signal)]
                else:
                    filtered = np.pad(filtered, (0, len(signal) - len(filtered)), mode='edge')
            
            return filtered
            
        except Exception as e:
            warnings.warn(f"Error applying Butterworth for {name}: {e}")
            return np.full(len(signal), np.nan)
    
    def decompose(self, df: pd.DataFrame) -> pd.DataFrame:
        """Decompose time series using two-stage VMD + Butterworth.
        
        Args:
            df: Input dataframe with columns:
                - ds (datetime): Timestamps
                - y (float): Target values
                
        Returns:
            DataFrame with original columns plus decomposed components:
                - y_imf3: Highest frequency / noise (from residual VMD)
                - y_subdaily: Sub-daily component (IMF2 from residual VMD)
                - y_vmd_daily: [0.75, 1.25] days band (Butterworth on IMF1)
                - y_vmd_weekly: [1.5, 7] days band (Butterworth on IMF1)
                - y_vmd_monthly: [7, 30] days band (Butterworth on IMF1)
                - y_vmd_trend: [30, 200] days band (Butterworth on IMF1)
        """
        df = df.copy()
        
        # Ensure 'y' exists
        if 'y' not in df.columns:
            if 'tempMean' in df.columns:
                df['y'] = df['tempMean']
            else:
                raise ValueError("Column 'y' or 'tempMean' not found in dataframe.")
        
        # Fill NaN values
        y = df['y'].ffill().bfill().values
        
        print(f"\n{'='*70}")
        print(f"Rubin VMD Decomposition")
        print(f"{'='*70}")
        print(f"Data frequency: {self.freq} obs/day")
        print(f"Data length: {len(y)} observations ({len(y)/self.freq:.1f} days)")
        print(f"VMD alpha: {self.alpha}")
        print(f"{'='*70}\n")
        
        # ================================================================
        # STAGE 1: VMD on original signal → Extract IMF1
        # ================================================================
        print(f"STAGE 1: VMD on original signal (K={self.K_stage1})...")
        
        imfs_stage1 = self._run_vmd(y, self.K_stage1)
        
        # IMF1 is the lowest frequency mode (index 0 after sorting)
        imf1 = imfs_stage1[0]
        
        print(f"  ✓ Extracted IMF1 (lowest frequency mode)")
        print(f"  IMF1 std: {np.std(imf1):.4f}")
        
        # ================================================================
        # Apply Butterworth bandpass to IMF1 → Daily, Weekly, Monthly, Trend
        # ================================================================
        print(f"\nApplying Butterworth bandpass to IMF1...")
        
        for i, ((period_low, period_high), name) in enumerate(zip(self.butter_periods[1:], self.band_names[1:])):
            is_trend = (i == len(self.butter_periods) - 1)
            
            print(f"  Computing {name} ({period_low:.2f}-{period_high:.1f}d)...")
            
            df[f'y_{name}'] = self._apply_butterworth_bandpass(
                imf1, period_low, period_high, name, is_trend=is_trend
            )
        
        # ================================================================
        # STAGE 2: VMD on residual → Extract sub-daily (IMF2) and noise (IMF3)
        # ================================================================
        print(f"\nSTAGE 2: VMD on residual signal (K={self.K_stage2})...")
        
        # Compute residual
        residual = y - df['y_vmd_monthly'].values - df['y_vmd_trend'].values
        
        print(f"  Residual std: {np.std(residual):.4f}")
        
        imfs_stage2 = self._run_vmd(residual, self.K_stage2)
        
        # After sorting by frequency (ascending):
        # imfs_stage2[0] = IMF1 (lowest freq in residual) - discard
        # imfs_stage2[1] = IMF2 (middle freq) - this is sub-daily
        # imfs_stage2[2] = IMF3 (highest freq) - this is noise
        
        daily = imfs_stage2[0]      # Lowest freq from residual (discarded)
        subdaily = imfs_stage2[1]  # Sub-daily component
        imf3 = imfs_stage2[2]       # Highest freq (noise-like)
        
        df['y_vmd_daily'] = self._apply_butterworth_bandpass(
            daily, self.butter_periods[0][0], self.butter_periods[0][1], 'vmd_daily'
        )
        df['y_subdaily'] = subdaily
        df['y_imf3'] = imf3
        
        print(f"  ✓ Extracted sub-daily (IMF2, std: {np.std(subdaily):.4f})")
        print(f"  ✓ Extracted IMF3 / noise (std: {np.std(imf3):.4f})")
        print(f"  (IMF1 from residual discarded)")
        
        # ================================================================
        # Summary
        # ================================================================
        print(f"\n{'='*70}")
        print(f"Decomposition complete!")
        print(f"{'='*70}")
        print(f"\nOutput components (high freq → low freq):")
        print(f"  1. y_imf3        : Highest freq / noise (from residual VMD)")
        print(f"  2. y_subdaily    : Sub-daily component (IMF2 from residual VMD)")
        print(f"  3. y_vmd_daily   : {self.butter_periods[0]} days (Butterworth on IMF1)")
        print(f"  4. y_vmd_weekly  : {self.butter_periods[1]} days (Butterworth on IMF1)")
        print(f"  5. y_vmd_monthly : {self.butter_periods[2]} days (Butterworth on IMF1)")
        print(f"  6. y_vmd_trend   : {self.butter_periods[3]} days (Butterworth on IMF1)")
        print(f"\nDataset shape: {df.shape}")
        print(f"{'='*70}\n")
        
        return df
    
    def get_component_info(self) -> pd.DataFrame:
        """Get information about the decomposed components.
        
        Returns:
            DataFrame with component specifications
        """
        info = [
            {
                'component': 'y_imf3',
                'source': 'Residual VMD (Stage 2)',
                'description': 'Highest frequency / noise',
                'period': 'Very high freq',
            },
            {
                'component': 'y_subdaily',
                'source': 'Residual VMD (Stage 2)',
                'description': 'Sub-daily component (IMF2)',
                'period': 'Sub-daily (~12h)',
            },
            {
                'component': 'y_vmd_daily',
                'source': 'Butterworth on IMF1 (Stage 1)',
                'description': f'Period {self.butter_periods[0]} days',
                'period': 'Daily (~24h)',
            },
            {
                'component': 'y_vmd_weekly',
                'source': 'Butterworth on IMF1 (Stage 1)',
                'description': f'Period {self.butter_periods[1]} days',
                'period': 'Weekly (~7d)',
            },
            {
                'component': 'y_vmd_monthly',
                'source': 'Butterworth on IMF1 (Stage 1)',
                'description': f'Period {self.butter_periods[2]} days',
                'period': 'Monthly (~28d)',
            },
            {
                'component': 'y_vmd_trend',
                'source': 'Butterworth on IMF1 (Stage 1)',
                'description': f'Period {self.butter_periods[3]} days',
                'period': 'Long-term trend',
            },
        ]
        
        return pd.DataFrame(info)
    
    def reconstruct(self, df: pd.DataFrame) -> np.ndarray:
        """Reconstruct the original signal from components.
        
        Args:
            df: DataFrame with decomposed components
            
        Returns:
            Reconstructed signal
        """
        components = ['y_imf3', 'y_subdaily', 'y_vmd_daily', 'y_vmd_weekly', 'y_vmd_monthly', 'y_vmd_trend']
        
        reconstructed = np.zeros(len(df))
        for comp in components:
            if comp in df.columns:
                reconstructed += df[comp].values
        
        return reconstructed


def preprocess_rubin_data(
    df: pd.DataFrame,
    freq: int = 96,
    alpha: float = 2000,
    **kwargs
) -> pd.DataFrame:
    """Convenience function to preprocess Rubin temperature data.
    
    Args:
        df: Input dataframe with 'ds' and 'y' columns
        freq: Data frequency in observations per day
        alpha: VMD bandwidth constraint
        **kwargs: Additional arguments for RubinVMDDecomposer
        
    Returns:
        Preprocessed dataframe with VMD components
    """
    decomposer = RubinVMDDecomposer(freq=freq, alpha=alpha, **kwargs)
    return decomposer.decompose(df)


# ============================================================================
# Test code
# ============================================================================
if __name__ == "__main__":
    """Example usage and testing"""
    
    # Create synthetic data similar to Rubin temperature
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-07-01', freq='15min')
    n = len(dates)
    t = np.arange(n)
    freq = 96
    
    # Synthetic signal with multiple frequencies
    y = (
        6 * np.sin(2 * np.pi * t / (freq * 0.5)) +      # Sub-daily (12h)
        10 * np.sin(2 * np.pi * t / freq) +              # Daily (24h)
        5 * np.sin(2 * np.pi * t / (freq * 7)) +         # Weekly
        3 * np.sin(2 * np.pi * t / (freq * 28)) +        # Monthly
        np.random.randn(n) * 0.5 +                       # Noise
        20 + 0.01 * t                                     # Trend
    )
    
    df = pd.DataFrame({'ds': dates, 'y': y})
    
    print("Testing Rubin VMD Decomposer:")
    print("=" * 80)
    
    decomposer = RubinVMDDecomposer(
        freq=freq,
        alpha=2000,
        K_stage1=5,
        K_stage2=3
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
