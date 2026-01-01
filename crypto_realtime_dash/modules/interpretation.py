"""
Interpretation Module - Advanced Signal Engine
SINTA 1 Bitcoin Forecasting System
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime


# Signal Matrix Rules

SIGNAL_MATRIX = {
    # Format: (sentiment, rsi_zone, macd_trend, change_24h) -> signal, confidence, reason
    
    # === STRONG BUY SIGNALS ===
    ("bearish", "oversold", "bullish", "negative"): ("BUY", 0.85, "Capitulation reversal - strong contrarian buy"),
    ("bearish", "oversold", "bullish", "positive"): ("BUY", 0.80, "Recovery starting from oversold"),
    ("neutral", "oversold", "bullish", "positive"): ("BUY", 0.75, "Technical reversal confirmed"),
    ("neutral", "oversold", "bullish", "negative"): ("BUY", 0.70, "Potential bottom forming"),
    ("bullish", "oversold", "bullish", "positive"): ("BUY", 0.90, "All signals aligned bullish"),
    ("bullish", "neutral", "bullish", "positive"): ("BUY", 0.75, "Bullish momentum building"),
    
    # === STRONG SELL SIGNALS ===
    ("bullish", "overbought", "bearish", "positive"): ("SELL", 0.85, "Euphoria reversal - strong contrarian sell"),
    ("bullish", "overbought", "bearish", "negative"): ("SELL", 0.80, "Distribution phase starting"),
    ("neutral", "overbought", "bearish", "negative"): ("SELL", 0.75, "Technical breakdown confirmed"),
    ("neutral", "overbought", "bearish", "positive"): ("SELL", 0.70, "Potential top forming"),
    ("bearish", "overbought", "bearish", "negative"): ("SELL", 0.90, "All signals aligned bearish"),
    ("bearish", "neutral", "bearish", "negative"): ("SELL", 0.75, "Bearish momentum building"),
    
    # === WAIT/ACCUMULATE SIGNALS ===
    ("bearish", "oversold", "bearish", "negative"): ("WAIT", 0.65, "Falling knife - wait for reversal confirmation"),
    ("bearish", "neutral", "bullish", "positive"): ("ACCUMULATE", 0.60, "Potential trend change - accumulate cautiously"),
    ("bullish", "overbought", "bullish", "positive"): ("WAIT", 0.65, "Overextended rally - wait for pullback"),
    ("bullish", "neutral", "bearish", "negative"): ("REDUCE", 0.60, "Potential trend change - reduce position"),
    
    # === HOLD SIGNALS ===
    ("neutral", "neutral", "bullish", "positive"): ("HOLD", 0.50, "Neutral with slight bullish bias"),
    ("neutral", "neutral", "bullish", "negative"): ("HOLD", 0.50, "Mixed signals - maintain position"),
    ("neutral", "neutral", "bearish", "positive"): ("HOLD", 0.50, "Mixed signals - maintain position"),
    ("neutral", "neutral", "bearish", "negative"): ("HOLD", 0.50, "Neutral with slight bearish bias"),
}


def classify_sentiment_zone(sentiment: float) -> str:
    """Classify sentiment into zones."""
    if sentiment > 0.15:
        return "bullish"
    elif sentiment < -0.15:
        return "bearish"
    return "neutral"


def classify_rsi_zone(rsi: float) -> str:
    """Classify RSI into zones."""
    if rsi > 70:
        return "overbought"
    elif rsi < 30:
        return "oversold"
    return "neutral"


def classify_macd_trend(macd: float, signal: float) -> str:
    """Classify MACD trend."""
    return "bullish" if macd > signal else "bearish"


def classify_24h_change(change: float) -> str:
    """Classify 24h price change."""
    return "positive" if change >= 0 else "negative"


class InterpretationEngine:
    """Rule-based market signal engine."""
    
    def __init__(self):
        self.insights = []
        self.scores = {
            "bullish": 0,
            "bearish": 0,
            "confidence": 0.5
        }
    
    def clear(self):
        """Clear all insights."""
        self.insights = []
        self.scores = {"bullish": 0, "bearish": 0, "confidence": 0.5}
    
    def _add_insight(self, category: str, level: str, message: str, signal: str = None, weight: float = 1.0):
        """Add an insight to the list with weight."""
        self.insights.append({
            'category': category,
            'level': level,
            'message': message,
            'signal': signal,
            'weight': weight,
            'timestamp': datetime.now().isoformat()
        })
        
        # Update scores
        if signal == 'buy':
            self.scores['bullish'] += weight
        elif signal == 'sell':
            self.scores['bearish'] += weight
    
    def interpret_rsi(self, rsi: float):
        """Interpret RSI indicator."""
        if rsi > 80:
            self._add_insight('RSI', 'critical', f'RSI {rsi:.1f} - Extreme overbought', 'sell', 1.5)
        elif rsi > 70:
            self._add_insight('RSI', 'warning', f'RSI {rsi:.1f} - Overbought zone', 'sell', 1.0)
        elif rsi < 20:
            self._add_insight('RSI', 'critical', f'RSI {rsi:.1f} - Extreme oversold', 'buy', 1.5)
        elif rsi < 30:
            self._add_insight('RSI', 'warning', f'RSI {rsi:.1f} - Oversold zone', 'buy', 1.0)
        else:
            self._add_insight('RSI', 'info', f'RSI {rsi:.1f} - Neutral', 'hold', 0.5)
    
    def interpret_macd(self, macd: float, signal: float, histogram: float):
        """Interpret MACD indicator."""
        if macd > signal and histogram > 0:
            strength = "Strong" if histogram > abs(macd) * 0.1 else "Moderate"
            self._add_insight('MACD', 'info', f'{strength} bullish ({macd:.0f}/{signal:.0f})', 'buy', 1.0 if strength == "Strong" else 0.7)
        elif macd < signal and histogram < 0:
            strength = "Strong" if abs(histogram) > abs(macd) * 0.1 else "Moderate"
            self._add_insight('MACD', 'warning', f'{strength} bearish ({macd:.0f}/{signal:.0f})', 'sell', 1.0 if strength == "Strong" else 0.7)
        else:
            self._add_insight('MACD', 'info', 'MACD transitioning', 'hold', 0.3)
    
    def interpret_sma(self, price: float, sma_14: float, sma_50: float = None):
        """Interpret SMA indicators."""
        if price > sma_14:
            self._add_insight('SMA', 'info', f'Above SMA-14 (${sma_14:,.0f})', 'buy', 0.5)
        else:
            self._add_insight('SMA', 'info', f'Below SMA-14 (${sma_14:,.0f})', 'sell', 0.5)
        
        if sma_50:
            if sma_14 > sma_50:
                self._add_insight('SMA', 'info', 'Golden cross (SMA-14 > SMA-50)', 'buy', 0.8)
            else:
                self._add_insight('SMA', 'warning', 'Death cross (SMA-14 < SMA-50)', 'sell', 0.8)
    
    def interpret_sentiment(self, fused_sentiment: float, news_sentiment: float = None, twitter_sentiment: float = None):
        """Interpret sentiment data."""
        if fused_sentiment > 0.3:
            self._add_insight('Sentiment', 'info', f'Strong bullish ({fused_sentiment:.3f})', 'buy', 1.2)
        elif fused_sentiment > 0.1:
            self._add_insight('Sentiment', 'info', f'Moderately bullish ({fused_sentiment:.3f})', 'buy', 0.7)
        elif fused_sentiment < -0.3:
            self._add_insight('Sentiment', 'warning', f'Strong bearish ({fused_sentiment:.3f})', 'sell', 1.2)
        elif fused_sentiment < -0.1:
            self._add_insight('Sentiment', 'info', f'Moderately bearish ({fused_sentiment:.3f})', 'sell', 0.7)
        else:
            self._add_insight('Sentiment', 'info', f'Neutral ({fused_sentiment:.3f})', 'hold', 0.3)
    
    def interpret_24h_change(self, change: float):
        """Interpret 24h price change."""
        if change > 5:
            self._add_insight('24h Change', 'warning', f'Strong pump +{change:.1f}%', 'sell', 0.8)
        elif change > 2:
            self._add_insight('24h Change', 'info', f'Positive momentum +{change:.1f}%', 'buy', 0.5)
        elif change < -5:
            self._add_insight('24h Change', 'warning', f'Strong dump {change:.1f}%', 'buy', 0.8)
        elif change < -2:
            self._add_insight('24h Change', 'info', f'Negative momentum {change:.1f}%', 'sell', 0.5)
        else:
            self._add_insight('24h Change', 'info', f'Sideways {change:+.1f}%', 'hold', 0.2)
    
    def interpret_volatility(self, volatility: float):
        """Interpret volatility."""
        vol_pct = volatility * 100
        if vol_pct > 5:
            self._add_insight('Volatility', 'critical', f'High volatility ({vol_pct:.1f}%)', 'hold', 0.5)
        elif vol_pct > 3:
            self._add_insight('Volatility', 'warning', f'Elevated volatility ({vol_pct:.1f}%)', 'hold', 0.3)
        else:
            self._add_insight('Volatility', 'info', f'Normal volatility ({vol_pct:.1f}%)', 'hold', 0.1)
    
    def interpret_model_performance(self, arima_rmse: float, arimax_rmse: float):
        """Interpret model comparison."""
        improvement = (arima_rmse - arimax_rmse) / arima_rmse * 100
        if improvement > 10:
            self._add_insight('Model', 'info', f'ARIMAX +{improvement:.1f}% better', None, 0)
        elif improvement > 0:
            self._add_insight('Model', 'info', f'ARIMAX +{improvement:.1f}% better', None, 0)
        else:
            self._add_insight('Model', 'warning', f'ARIMA +{-improvement:.1f}% better', None, 0)
    
    def get_combined_signal(self, sentiment: float, rsi: float, macd: float, macd_signal: float, change_24h: float) -> Dict:
        """Get combined signal from matrix lookup."""
        sent_zone = classify_sentiment_zone(sentiment)
        rsi_zone = classify_rsi_zone(rsi)
        macd_trend = classify_macd_trend(macd, macd_signal)
        change_zone = classify_24h_change(change_24h)
        
        key = (sent_zone, rsi_zone, macd_trend, change_zone)
        
        if key in SIGNAL_MATRIX:
            signal, confidence, reason = SIGNAL_MATRIX[key]
            return {
                "signal": signal,
                "confidence": confidence,
                "reason": reason,
                "components": {
                    "sentiment": sent_zone,
                    "rsi": rsi_zone,
                    "macd": macd_trend,
                    "24h_change": change_zone
                }
            }
        
        # Default fallback
        return {
            "signal": "HOLD",
            "confidence": 0.40,
            "reason": "No strong signal - mixed conditions",
            "components": {
                "sentiment": sent_zone,
                "rsi": rsi_zone,
                "macd": macd_trend,
                "24h_change": change_zone
            }
        }
    
    def get_all_insights(self) -> List[Dict]:
        """Get all generated insights."""
        return self.insights
    
    def get_signal_summary(self) -> Dict[str, int]:
        """Get summary of signals."""
        signals = {'buy': 0, 'sell': 0, 'hold': 0}
        for insight in self.insights:
            signal = insight.get('signal')
            if signal in signals:
                signals[signal] += insight.get('weight', 1)
        return signals
    
    def get_overall_signal(self) -> str:
        """Get overall trading signal based on weighted scores."""
        bullish = self.scores['bullish']
        bearish = self.scores['bearish']
        
        diff = bullish - bearish
        
        if diff > 2.0:
            return 'STRONG BUY'
        elif diff > 1.0:
            return 'BUY'
        elif diff < -2.0:
            return 'STRONG SELL'
        elif diff < -1.0:
            return 'SELL'
        else:
            return 'HOLD'
    
    def get_confidence(self) -> float:
        """Get confidence in the signal."""
        bullish = self.scores['bullish']
        bearish = self.scores['bearish']
        total = bullish + bearish
        
        if total == 0:
            return 0.5
        
        return min(abs(bullish - bearish) / total, 1.0)


def generate_interpretation(df: pd.DataFrame, model_metrics: Dict = None, change_24h: float = 0) -> Dict:
    """Generate trading insights from indicators."""
    if df.empty:
        return {'summary': 'Insufficient data for interpretation'}
    
    engine = InterpretationEngine()
    latest = df.iloc[-1]
    
    # RSI
    rsi = latest.get('rsi_14', 50)
    if 'rsi_14' in latest:
        engine.interpret_rsi(rsi)
    
    # MACD
    macd = latest.get('macd', 0)
    macd_sig = latest.get('macd_signal', 0)
    macd_hist = latest.get('macd_hist', 0)
    if all(k in latest for k in ['macd', 'macd_signal', 'macd_hist']):
        engine.interpret_macd(macd, macd_sig, macd_hist)
    
    # SMA
    if 'close' in latest and 'sma_14' in latest:
        sma_50 = latest.get('sma_50', None)
        engine.interpret_sma(latest['close'], latest['sma_14'], sma_50)
    
    # Sentiment
    sentiment = latest.get('fused_sentiment', 0)
    if 'fused_sentiment' in latest:
        news = latest.get('news_sentiment', None)
        twitter = latest.get('twitter_sentiment', None)
        engine.interpret_sentiment(sentiment, news, twitter)
    
    # 24h Change
    engine.interpret_24h_change(change_24h)
    
    # Volatility
    if 'volatility' in latest:
        engine.interpret_volatility(latest['volatility'])
    
    # Model comparison
    if model_metrics and 'ARIMA' in model_metrics and 'ARIMAX' in model_metrics:
        arima_rmse = model_metrics['ARIMA'].get('RMSE', 0)
        arimax_rmse = model_metrics['ARIMAX'].get('RMSE', 0)
        if arima_rmse > 0 and arimax_rmse > 0:
            engine.interpret_model_performance(arima_rmse, arimax_rmse)
    
    # Get combined signal
    combined = engine.get_combined_signal(sentiment, rsi, macd, macd_sig, change_24h)
    
    return {
        'insights': engine.get_all_insights(),
        'signal_summary': engine.get_signal_summary(),
        'overall_signal': engine.get_overall_signal(),
        'combined_signal': combined,
        'confidence': engine.get_confidence()
    }


if __name__ == "__main__":
    print("Testing Advanced Interpretation Module...")
    print("=" * 60)
    
    engine = InterpretationEngine()
    
    # Test with bearish sentiment + oversold RSI + bullish MACD
    print("\nScenario: Bearish Sentiment + Oversold RSI + Bullish MACD")
    result = engine.get_combined_signal(
        sentiment=-0.25,  # Bearish
        rsi=25,           # Oversold
        macd=100,         # Above signal
        macd_signal=80,
        change_24h=-3.5   # Negative
    )
    print(f"  Signal: {result['signal']}")
    print(f"  Confidence: {result['confidence']:.0%}")
    print(f"  Reason: {result['reason']}")
    
    print("\nScenario: Bullish Sentiment + Overbought RSI + Bearish MACD")
    result = engine.get_combined_signal(
        sentiment=0.3,    # Bullish
        rsi=75,           # Overbought
        macd=-50,         # Below signal
        macd_signal=20,
        change_24h=4.0    # Positive
    )
    print(f"  Signal: {result['signal']}")
    print(f"  Confidence: {result['confidence']:.0%}")
    print(f"  Reason: {result['reason']}")
