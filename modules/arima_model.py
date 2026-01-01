"""
ARIMA Model Module - ARIMA & ARIMAX Forecasting
SINTA 1 Bitcoin Forecasting System
Only ARIMA and ARIMAX - No Prophet, XGBoost, LSTM
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from typing import Tuple, Dict, Optional, List
import warnings

warnings.filterwarnings('ignore')


class ARIMAModel:
    """ARIMA model for univariate forecasting."""
    
    def __init__(self, order: Tuple[int, int, int] = (5, 1, 0)):
        """Initialize with (p, d, q) order."""
        self.order = order
        self.model = None
        self.fitted = None
        self.train_data = None
    
    def fit(self, data: pd.Series) -> 'ARIMAModel':
        """Fit model to data (close prices)."""
        self.train_data = data.values
        
        try:
            self.model = ARIMA(self.train_data, order=self.order)
            self.fitted = self.model.fit()
            
            print(f"[+] ARIMA{self.order} fitted")
            print(f"    AIC: {self.fitted.aic:.2f} | BIC: {self.fitted.bic:.2f}")
            
        except Exception as e:
            print(f"[!] ARIMA fitting error: {e}")
            raise
        
        return self
    
    def predict(self, steps: int) -> np.ndarray:
        """Forecast n steps ahead."""
        if self.fitted is None:
            raise ValueError("Model not fitted")
        
        return self.fitted.forecast(steps=steps)
    
    def get_residuals(self) -> np.ndarray:
        """Get model residuals."""
        if self.fitted is None:
            return np.array([])
        return self.fitted.resid
    
    def get_summary(self) -> str:
        """Get model summary."""
        if self.fitted is None:
            return "Model not fitted"
        return str(self.fitted.summary())


class ARIMAXModel:
    """ARIMAX with exogenous variables."""
    
    def __init__(self, order: Tuple[int, int, int] = (5, 1, 0)):
        """Initialize with (p, d, q) order."""
        self.order = order
        self.model = None
        self.fitted = None
        self.train_data = None
        self.exog_train = None
        self.exog_columns = []
    
    def fit(self, data: pd.Series, exog: pd.DataFrame) -> 'ARIMAXModel':
        """Fit model with exogenous variables."""
        self.train_data = data.values
        self.exog_train = exog.values
        self.exog_columns = exog.columns.tolist()
        
        try:
            self.model = SARIMAX(
                self.train_data,
                exog=self.exog_train,
                order=self.order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.fitted = self.model.fit(disp=False)
            
            print(f"[+] ARIMAX{self.order} fitted")
            print(f"    AIC: {self.fitted.aic:.2f} | BIC: {self.fitted.bic:.2f}")
            print(f"    Exog: {self.exog_columns}")
            
        except Exception as e:
            print(f"[!] ARIMAX fitting error: {e}")
            raise
        
        return self
    
    def predict(self, steps: int, exog_future: pd.DataFrame) -> np.ndarray:
        """Forecast with future exogenous values."""
        if self.fitted is None:
            raise ValueError("Model not fitted")
        
        return self.fitted.forecast(steps=steps, exog=exog_future.values)
    
    def get_residuals(self) -> np.ndarray:
        """Get model residuals."""
        if self.fitted is None:
            return np.array([])
        return self.fitted.resid
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get exogenous variable coefficients."""
        if self.fitted is None:
            return pd.DataFrame()
        
        params = self.fitted.params
        pvalues = self.fitted.pvalues
        
        # Extract exog coefficients
        importance = []
        for i, col in enumerate(self.exog_columns):
            param_name = f'x{i+1}'
            if param_name in params.index:
                importance.append({
                    'feature': col,
                    'coefficient': params[param_name],
                    'pvalue': pvalues[param_name],
                    'significant': pvalues[param_name] < 0.05
                })
        
        return pd.DataFrame(importance)


def auto_select_order(data: pd.Series, max_p: int = 5, max_q: int = 5) -> Tuple[int, int, int]:
    """Auto-select best (p, d, q) using AIC."""
    # Determine d using ADF test
    adf_result = adfuller(data.dropna())
    d = 0 if adf_result[1] < 0.05 else 1
    
    # Try second differencing if still non-stationary
    if d == 1:
        diff_data = data.diff().dropna()
        adf_result = adfuller(diff_data)
        if adf_result[1] > 0.05:
            d = 2
    
    best_aic = np.inf
    best_order = (1, d, 1)
    
    print(f"[*] Auto-selecting ARIMA order (d={d})...")
    
    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            if p == 0 and q == 0:
                continue
            
            try:
                model = ARIMA(data.values, order=(p, d, q))
                fitted = model.fit()
                
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = (p, d, q)
                    
            except Exception:
                continue
    
    print(f"[+] Best order: {best_order} (AIC: {best_aic:.2f})")
    
    return best_order


def train_arima(
    df: pd.DataFrame,
    order: Tuple[int, int, int] = None,
    test_size: float = 0.2,
    auto_order: bool = False
) -> Dict:
    """Train ARIMA and return predictions."""
    data = df['close'].copy()
    
    # Train/test split
    split_idx = int(len(data) * (1 - test_size))
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    # Auto-select order
    if order is None or auto_order:
        order = auto_select_order(train_data)
    
    # Fit model
    model = ARIMAModel(order=order)
    model.fit(train_data)
    
    # Predict
    predictions = model.predict(len(test_data))
    
    return {
        'model': model,
        'predictions': predictions,
        'actual': test_data.values,
        'residuals': model.get_residuals(),
        'order': order,
        'train_size': len(train_data),
        'test_size': len(test_data),
        'aic': model.fitted.aic,
        'bic': model.fitted.bic
    }


def train_arimax(
    df: pd.DataFrame,
    exog_columns: List[str] = None,
    order: Tuple[int, int, int] = None,
    test_size: float = 0.2,
    auto_order: bool = False
) -> Dict:
    """Train ARIMAX with exog variables."""
    # Default exogenous columns
    if exog_columns is None:
        exog_columns = ['fused_sentiment', 'rsi_14', 'sma_14', 'macd', 'volume']
    
    # Filter available columns
    available = [c for c in exog_columns if c in df.columns]
    if not available:
        raise ValueError(f"No exog columns found. Available: {df.columns.tolist()}")
    
    data = df['close'].copy()
    exog = df[available].copy()
    
    # Fill NaN values
    exog = exog.ffill().bfill().fillna(0)
    
    # Train/test split
    split_idx = int(len(data) * (1 - test_size))
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    exog_train = exog[:split_idx]
    exog_test = exog[split_idx:]
    
    # Auto-select order
    if order is None or auto_order:
        order = auto_select_order(train_data)
    
    # Fit model
    model = ARIMAXModel(order=order)
    model.fit(train_data, exog_train)
    
    # Predict
    predictions = model.predict(len(test_data), exog_test)
    
    # Feature importance
    importance = model.get_feature_importance()
    
    return {
        'model': model,
        'predictions': predictions,
        'actual': test_data.values,
        'residuals': model.get_residuals(),
        'exog_columns': available,
        'feature_importance': importance,
        'order': order,
        'train_size': len(train_data),
        'test_size': len(test_data),
        'aic': model.fitted.aic,
        'bic': model.fitted.bic
    }


def forecast_multi_horizon(
    model,
    horizons: List[int] = [1, 3, 7],
    exog_future: pd.DataFrame = None
) -> Dict[int, np.ndarray]:
    """Forecast for multiple horizons."""
    forecasts = {}
    
    for h in horizons:
        try:
            if exog_future is not None:
                # ARIMAX
                forecasts[h] = model.predict(h, exog_future.head(h))
            else:
                # ARIMA
                forecasts[h] = model.predict(h)
        except Exception as e:
            print(f"[!] Forecast error for horizon {h}: {e}")
            forecasts[h] = np.array([])
    
    return forecasts


if __name__ == "__main__":
    print("Testing ARIMA Model Module...")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    n = 200
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n, freq='D')
    prices = 97000 + np.cumsum(np.random.randn(n) * 500)
    
    df = pd.DataFrame({
        'date': dates,
        'close': prices,
        'fused_sentiment': np.random.uniform(-0.5, 0.5, n),
        'rsi_14': np.random.uniform(30, 70, n),
        'sma_14': prices + np.random.randn(n) * 50,
        'macd': np.random.uniform(-100, 100, n),
        'volume': np.random.uniform(100, 1000, n)
    })
    
    print("\nTesting ARIMA Baseline:")
    arima_result = train_arima(df, order=(5, 1, 0))
    print(f"  Predictions: {len(arima_result['predictions'])}")
    
    print("\nTesting ARIMAX:")
    arimax_result = train_arimax(df, order=(5, 1, 0))
    print(f"  Predictions: {len(arimax_result['predictions'])}")
    print(f"  Exog: {arimax_result['exog_columns']}")
