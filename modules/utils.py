"""
Utility Functions
SINTA 1 Bitcoin Forecasting System
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import yaml
import os


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'config', 'config.yaml'
        )
    
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}


def normalize_series(series: pd.Series) -> pd.Series:
    """Normalize series to 0-1 range."""
    min_val = series.min()
    max_val = series.max()
    
    if max_val == min_val:
        return pd.Series([0.5] * len(series), index=series.index)
    
    return (series - min_val) / (max_val - min_val)


def standardize_series(series: pd.Series) -> pd.Series:
    """Standardize series (z-score normalization)."""
    mean = series.mean()
    std = series.std()
    
    if std == 0:
        return pd.Series([0] * len(series), index=series.index)
    
    return (series - mean) / std


def resample_ohlcv(df: pd.DataFrame, target_interval: str) -> pd.DataFrame:
    """
    Resample OHLCV data to different interval.
    
    Args:
        df: DataFrame with OHLCV data
        target_interval: Target interval (e.g., '1h', '4h', '1d')
    
    Returns:
        Resampled DataFrame
    """
    if df.empty:
        return df
    
    df = df.copy()
    df.set_index('timestamp', inplace=True)
    
    resampled = df.resample(target_interval).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    resampled = resampled.reset_index()
    
    return resampled


def format_price(value: float) -> str:
    """Format price for display."""
    return f"${value:,.2f}"


def format_percentage(value: float, include_sign: bool = True) -> str:
    """Format percentage for display."""
    if include_sign:
        return f"{value:+.2f}%"
    return f"{value:.2f}%"


def format_number(value: float, decimals: int = 2) -> str:
    """Format number with thousands separator."""
    return f"{value:,.{decimals}f}"


def get_date_range(days: int = 30) -> tuple:
    """Get date range for data fetching."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    return start_date, end_date


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Safe division to avoid zero division errors."""
    if b == 0:
        return default
    return a / b


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(max_val, value))


def moving_average(series: pd.Series, window: int) -> pd.Series:
    """Calculate simple moving average."""
    return series.rolling(window=window, min_periods=1).mean()


def exponential_moving_average(series: pd.Series, span: int) -> pd.Series:
    """Calculate exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def detect_outliers(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Detect outliers using z-score method."""
    z_scores = np.abs(standardize_series(series))
    return z_scores > threshold


def remove_outliers(df: pd.DataFrame, columns: list, threshold: float = 3.0) -> pd.DataFrame:
    """Remove outlier rows from DataFrame."""
    df = df.copy()
    
    for col in columns:
        if col in df.columns:
            outliers = detect_outliers(df[col], threshold)
            df = df[~outliers]
    
    return df.reset_index(drop=True)


if __name__ == "__main__":
    print("Testing Utility Functions...")
    print("=" * 50)
    
    # Test formatting
    print(f"\nPrice: {format_price(97500.123)}")
    print(f"Percentage: {format_percentage(2.5)}")
    print(f"Number: {format_number(1234567.89)}")
    
    # Test normalization
    data = pd.Series([10, 20, 30, 40, 50])
    normalized = normalize_series(data)
    print(f"\nNormalized: {normalized.tolist()}")
