"""Metrics calculation utilities for Rubin's Oracle forecasters.

This module provides utilities for computing forecast accuracy metrics including
RMSE, MAE, R², and step-specific metrics across all forecaster types.
"""

import numpy as np
import pandas as pd


class MetricsCalculator:
    """Static utility class for metrics computation.

    Computes standard forecast accuracy metrics (RMSE, MAE, R²) from predictions
    and actuals, with support for step-specific analysis.
    """

    @staticmethod
    def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Root Mean Squared Error.

        Args:
            y_true: Array of actual values
            y_pred: Array of predicted values

        Returns:
            RMSE value
        """
        if len(y_true) == 0:
            return np.nan
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Mean Absolute Error.

        Args:
            y_true: Array of actual values
            y_pred: Array of predicted values

        Returns:
            MAE value
        """
        if len(y_true) == 0:
            return np.nan
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute R² (coefficient of determination).

        Args:
            y_true: Array of actual values
            y_pred: Array of predicted values

        Returns:
            R² value (1.0 is perfect, can be negative)
        """
        if len(y_true) == 0:
            return np.nan

        ss_res: float = float(np.sum((y_true - y_pred) ** 2))
        ss_tot: float = float(np.sum((y_true - np.mean(y_true)) ** 2))

        if ss_tot == 0:
            return 0.0 if ss_res == 0 else np.nan

        return float(1 - (ss_res / ss_tot))

    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Compute all standard metrics.

        Args:
            y_true: Array of actual values
            y_pred: Array of predicted values

        Returns:
            Dictionary with keys: 'rmse', 'mae', 'r2', 'n_samples'
        """
        return {
            "rmse": MetricsCalculator.compute_rmse(y_true, y_pred),
            "mae": MetricsCalculator.compute_mae(y_true, y_pred),
            "r2": MetricsCalculator.compute_r2(y_true, y_pred),
            "n_samples": len(y_true),
        }

    @staticmethod
    def compute_step_metrics(df: pd.DataFrame, n_forecast: int) -> dict:
        """Compute step-specific metrics from standardized forecast DataFrame.

        Args:
            df: DataFrame with columns 'step', 'y', 'yhat' (standardized format)
            n_forecast: Maximum number of forecast steps

        Returns:
            Dictionary with step-specific metrics
        """
        metrics = {}

        # Last step metrics
        last_step_data = df[df["step"] == n_forecast].dropna(subset=["y", "yhat"])
        if len(last_step_data) > 0:
            y_true_last = last_step_data["y"].values
            y_pred_last = last_step_data["yhat"].values
            metrics["rmse_last_step"] = MetricsCalculator.compute_rmse(y_true_last, y_pred_last)

        # Middle step metrics
        middle_step = (n_forecast + 1) // 2
        middle_step_data = df[df["step"] == middle_step].dropna(subset=["y", "yhat"])
        if len(middle_step_data) > 0:
            y_true_mid = middle_step_data["y"].values
            y_pred_mid = middle_step_data["yhat"].values
            metrics["rmse_middle_step"] = MetricsCalculator.compute_rmse(y_true_mid, y_pred_mid)

        return metrics
