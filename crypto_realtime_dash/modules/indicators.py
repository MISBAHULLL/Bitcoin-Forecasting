"""
Technical Indicators Module - RSI, SMA, MACD
SINTA 1 Bitcoin Forecasting System
"""

import pandas as pd
import numpy as np
from typing import Tuple


def calculate_sma(df: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.Series:
    """
    Calculate Simple Moving Average.
    
    Args:
        df: DataFrame with price data
        period: SMA period
        column: Column to use
    
    Returns:
        Series with SMA values
    """
    return df[column].rolling(window=period, min_periods=1).mean()


def calculate_ema(df: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.Series:
    """
    Calculate Exponential Moving Average.
    
    Args:
        df: DataFrame with price data
        period: EMA period
        column: Column to use
    
    Returns:
        Series with EMA values
    """
    return df[column].ewm(span=period, adjust=False).mean()


def calculate_rsi(df: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.Series:
    """
    Calculate Relative Strength Index.
    
    Formula:
        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
    
    Args:
        df: DataFrame with price data
        period: RSI period
        column: Column to use
    
    Returns:
        Series with RSI values (0-100)
    """
    delta = df[column].diff()
    
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    
    # Avoid division by zero
    avg_loss = avg_loss.replace(0, 0.001)
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    column: str = 'close'
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        df: DataFrame with price data
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
        column: Column to use
    
    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    ema_fast = df[column].ewm(span=fast, adjust=False).mean()
    ema_slow = df[column].ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(
    df: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
    column: str = 'close'
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Args:
        df: DataFrame with price data
        period: SMA period
        std_dev: Standard deviation multiplier
        column: Column to use
    
    Returns:
        Tuple of (Middle band, Upper band, Lower band)
    """
    sma = df[column].rolling(window=period, min_periods=1).mean()
    std = df[column].rolling(window=period, min_periods=1).std()
    
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    
    return sma, upper, lower


def calculate_volatility(df: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.Series:
    """
    Calculate volatility (rolling standard deviation of returns).
    
    Args:
        df: DataFrame with price data
        period: Rolling period
        column: Column to use
    
    Returns:
        Series with volatility values
    """
    returns = df[column].pct_change()
    return returns.rolling(window=period, min_periods=1).std()


def calculate_returns(df: pd.DataFrame, column: str = 'close') -> pd.Series:
    """
    Calculate daily returns.
    
    Args:
        df: DataFrame with price data
        column: Column to use
    
    Returns:
        Series with returns
    """
    return df[column].pct_change()


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators.
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with all indicators added
    """
    if df.empty or 'close' not in df.columns:
        return df
    
    df = df.copy()
    
    # SMA
    df['sma_14'] = calculate_sma(df, period=14)
    df['sma_50'] = calculate_sma(df, period=50)
    
    print("[+] Calculated SMA-14, SMA-50")
    
    # EMA (for dashboard overlay)
    df['ema_12'] = calculate_ema(df, period=12)
    df['ema_26'] = calculate_ema(df, period=26)
    
    # RSI
    df['rsi_14'] = calculate_rsi(df, period=14)
    
    print("[+] Calculated RSI-14")
    
    # MACD
    df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df)
    
    print("[+] Calculated MACD (12, 26, 9)")
    
    # Bollinger Bands
    df['bb_middle'], df['bb_upper'], df['bb_lower'] = calculate_bollinger_bands(df)
    
    print("[+] Calculated Bollinger Bands")
    
    # Volatility and Returns
    df['volatility'] = calculate_volatility(df)
    df['returns'] = calculate_returns(df)
    
    print("[+] Calculated Volatility and Returns")
    
    # Trading signals
    df['rsi_signal'] = df['rsi_14'].apply(
        lambda x: 'Overbought' if x > 70 else ('Oversold' if x < 30 else 'Neutral')
    )
    
    df['macd_signal_type'] = df.apply(
        lambda row: 'Bullish' if row['macd'] > row['macd_signal'] else 'Bearish',
        axis=1
    )
    
    # Fill NaN values
    df = df.ffill().bfill()
    
    return df


def get_indicator_summary(df: pd.DataFrame) -> dict:
    """
    Get summary of current indicator values.
    
    Args:
        df: DataFrame with indicators
    
    Returns:
        Dictionary with summary
    """
    if df.empty:
        return {}
    
    latest = df.iloc[-1]
    
    summary = {
        'price': latest.get('close', 0),
        'sma_14': latest.get('sma_14', 0),
        'sma_50': latest.get('sma_50', 0),
        'rsi_14': latest.get('rsi_14', 50),
        'macd': latest.get('macd', 0),
        'macd_signal': latest.get('macd_signal', 0),
        'macd_hist': latest.get('macd_hist', 0),
        'volatility': latest.get('volatility', 0)
    }
    
    # Add interpretations
    summary['rsi_state'] = (
        'Overbought' if summary['rsi_14'] > 70 
        else 'Oversold' if summary['rsi_14'] < 30 
        else 'Neutral'
    )
    
    summary['macd_trend'] = 'Bullish' if summary['macd'] > summary['macd_signal'] else 'Bearish'
    
    summary['price_vs_sma'] = 'Above' if summary['price'] > summary['sma_14'] else 'Below'
    
    return summary


if __name__ == "__main__":
    print("Testing Indicators Module...")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    n = 100
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n, freq='h')
    prices = 97000 + np.cumsum(np.random.randn(n) * 100)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices - np.random.uniform(0, 50, n),
        'high': prices + np.random.uniform(0, 100, n),
        'low': prices - np.random.uniform(0, 100, n),
        'close': prices,
        'volume': np.random.uniform(100, 1000, n)
    })
    
    # Compute indicators
    df = compute_all_indicators(df)
    
    # Get summary
    print("\nCurrent Indicator Summary:")
    summary = get_indicator_summary(df)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
