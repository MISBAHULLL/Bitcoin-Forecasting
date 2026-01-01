"""
Price Fetcher Module - Binance OHLCV Real-time API
SINTA 1 Bitcoin Forecasting System

With improved fallback for regions where Binance is blocked.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Binance API endpoints
BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
BINANCE_TICKER = "https://api.binance.com/api/v3/ticker/price"
BINANCE_24HR = "https://api.binance.com/api/v3/ticker/24hr"

# Alternative APIs
COINGECKO_API = "https://api.coingecko.com/api/v3"

# Shorter timeout for faster fallback
API_TIMEOUT = 5  # seconds

# Timeframe mapping
TIMEFRAMES = {
    '1m': '1 Minute',
    '5m': '5 Minutes',
    '15m': '15 Minutes',
    '30m': '30 Minutes',
    '1h': '1 Hour',
    '4h': '4 Hours',
    '1d': '1 Day'
}

# Global flag to skip Binance if connection failed
_binance_available = True  # Start with True since VPN is now active
_binance_last_check = 0
BINANCE_CHECK_INTERVAL = 60  # Re-check Binance availability every 1 minute (reduced for VPN users)


def is_binance_available() -> bool:
    """Check if Binance API is accessible (cache result for 5 minutes)."""
    global _binance_available, _binance_last_check
    
    current_time = time.time()
    if current_time - _binance_last_check < BINANCE_CHECK_INTERVAL:
        return _binance_available
    
    try:
        response = requests.get(
            "https://api.binance.com/api/v3/ping",
            timeout=3,
            verify=False
        )
        _binance_available = response.status_code == 200
    except Exception:
        _binance_available = False
    
    _binance_last_check = current_time
    print(f"[*] Binance API available: {_binance_available}")
    return _binance_available


def fetch_binance_klines(symbol: str = 'BTCUSDT', interval: str = '1h', limit: int = 500) -> pd.DataFrame:
    """Fetch OHLCV from Binance API."""
    # Skip if Binance was recently unavailable
    if not is_binance_available():
        print("[*] Skipping Binance (unavailable), using fallback...")
        return pd.DataFrame()
    
    try:
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': min(limit, 1000)
        }
        
        response = requests.get(
            BINANCE_KLINES, 
            params=params, 
            timeout=API_TIMEOUT, 
            verify=False
        )
        
        if response.status_code == 200:
            data = response.json()
            
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
            df['open'] = pd.to_numeric(df['open'])
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['close'] = pd.to_numeric(df['close'])
            df['volume'] = pd.to_numeric(df['volume'])
            
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            print(f"[+] Binance: Fetched {len(df)} candles ({interval})")
            return df
            
        else:
            print(f"[!] Binance API error: {response.status_code}")
            
    except requests.exceptions.Timeout:
        global _binance_available
        _binance_available = False
        print("[!] Binance timeout - marked as unavailable, using fallback...")
    except requests.exceptions.ConnectionError:
        _binance_available = False
        print("[!] Binance connection error - marked as unavailable, using fallback...")
    except Exception as e:
        print(f"[!] Binance error: {type(e).__name__}: {e}")
    
    return pd.DataFrame()


def fetch_coingecko_ohlc(days: int = 30) -> pd.DataFrame:
    """Fetch OHLC from CoinGecko (fallback)."""
    try:
        url = f"{COINGECKO_API}/coins/bitcoin/ohlc"
        params = {'vs_currency': 'usd', 'days': min(days, 365)}
        
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            df = pd.DataFrame(data, columns=['timestamp_ms', 'open', 'high', 'low', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
            df['volume'] = 0  # CoinGecko OHLC doesn't include volume
            
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            print(f"[+] CoinGecko: Fetched {len(df)} candles")
            return df
            
    except Exception as e:
        print(f"[!] CoinGecko error: {e}")
    
    return pd.DataFrame()


def fetch_current_price(symbol: str = 'BTCUSDT') -> dict:
    """Get current BTC price with fallback."""
    result = None
    
    # Try Binance first only if available
    if is_binance_available():
        try:
            response = requests.get(
                BINANCE_24HR, 
                params={'symbol': symbol}, 
                timeout=API_TIMEOUT, 
                verify=False
            )
            
            if response.status_code == 200:
                data = response.json()
                result = {
                    'price': float(data.get('lastPrice', 0)),
                    'change_24h': float(data.get('priceChangePercent', 0)),
                    'high_24h': float(data.get('highPrice', 0)),
                    'low_24h': float(data.get('lowPrice', 0)),
                    'volume_24h': float(data.get('volume', 0)),
                    'source': 'Binance',
                    'timestamp': datetime.now().isoformat()
                }
                return result
                
        except requests.exceptions.Timeout:
            global _binance_available
            _binance_available = False
            print("[!] Binance ticker timeout - switching to CoinGecko")
        except Exception as e:
            print(f"[!] Binance ticker error: {type(e).__name__}")
    
    # Fallback to CoinGecko
    try:
        url = f"{COINGECKO_API}/simple/price"
        params = {
            'ids': 'bitcoin',
            'vs_currencies': 'usd',
            'include_24hr_change': 'true',
            'include_24hr_vol': 'true'
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json().get('bitcoin', {})
            result = {
                'price': data.get('usd', 0),
                'change_24h': data.get('usd_24h_change', 0),
                'volume_24h': data.get('usd_24h_vol', 0),
                'source': 'CoinGecko',
                'timestamp': datetime.now().isoformat()
            }
            print(f"[+] CoinGecko: Price ${result['price']:,.0f}")
            return result
            
    except Exception as e:
        print(f"[!] CoinGecko ticker error: {type(e).__name__}")
    
    # Return sample data if all APIs fail
    print("[*] Using sample price data")
    return {
        'price': 97500 + np.random.uniform(-500, 500),
        'change_24h': np.random.uniform(-3, 3),
        'source': 'Sample',
        'timestamp': datetime.now().isoformat()
    }


def generate_sample_ohlcv(limit: int = 500, interval: str = '1h') -> pd.DataFrame:
    """Generate sample OHLCV when APIs fail."""
    print("[*] Generating sample OHLCV data...")
    
    np.random.seed(int(time.time()) % 1000)
    
    # Parse interval to timedelta
    interval_map = {
        '1m': timedelta(minutes=1),
        '5m': timedelta(minutes=5),
        '15m': timedelta(minutes=15),
        '30m': timedelta(minutes=30),
        '1h': timedelta(hours=1),
        '4h': timedelta(hours=4),
        '1d': timedelta(days=1)
    }
    
    delta = interval_map.get(interval, timedelta(hours=1))
    
    end_time = datetime.now()
    timestamps = [end_time - delta * i for i in range(limit)][::-1]
    
    # Generate realistic price movement
    base_price = 97000
    returns = np.random.randn(limit) * 0.005  # 0.5% volatility
    prices = base_price * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'close': prices
    })
    
    # Generate OHLV
    df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
    df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.abs(np.random.randn(limit)) * 0.003)
    df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.abs(np.random.randn(limit)) * 0.003)
    df['volume'] = np.random.uniform(100, 1000, limit)
    
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    print(f"[+] Generated {len(df)} sample candles")
    return df


def get_ohlcv(interval: str = '1h', limit: int = 500) -> pd.DataFrame:
    """Get OHLCV with API fallbacks."""
    # Try Binance first
    df = fetch_binance_klines(interval=interval, limit=limit)
    
    if df.empty:
        # Try CoinGecko
        days = limit // 24 if interval in ['1h'] else (limit if interval == '1d' else 30)
        df = fetch_coingecko_ohlc(days=days)
    
    if df.empty:
        # Generate sample data
        df = generate_sample_ohlcv(limit=limit, interval=interval)
    
    return df


if __name__ == "__main__":
    print("Testing Price Fetcher...")
    print("=" * 50)
    
    # Test current price
    print("\nCurrent Price:")
    price = fetch_current_price()
    print(f"  ${price.get('price', 0):,.2f} ({price.get('source')})")
    print(f"  24h Change: {price.get('change_24h', 0):+.2f}%")
    
    # Test OHLCV
    print("\nOHLCV Data:")
    df = get_ohlcv(interval='1h', limit=100)
    print(f"  Records: {len(df)}")
    if not df.empty:
        print(f"  Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Price: ${df['close'].iloc[-1]:,.2f}")
