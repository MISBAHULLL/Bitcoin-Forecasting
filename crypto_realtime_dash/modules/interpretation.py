"""
Interpretation Module - Rule-based Insight Generator
SINTA 1 Bitcoin Forecasting System
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime


class InterpretationEngine:
    """Rule-based interpretation engine for market insights."""
    
    def __init__(self):
        self.insights = []
    
    def clear(self):
        """Clear all insights."""
        self.insights = []
    
    def _add_insight(self, category: str, level: str, message: str, signal: str = None):
        """Add an insight to the list."""
        self.insights.append({
            'category': category,
            'level': level,  # info, warning, critical
            'message': message,
            'signal': signal,  # buy, sell, hold
            'timestamp': datetime.now().isoformat()
        })
    
    def interpret_rsi(self, rsi: float):
        """Interpret RSI indicator."""
        if rsi > 80:
            self._add_insight(
                'RSI', 'critical',
                f'RSI at {rsi:.1f} - Extreme overbought. High probability of correction.',
                'sell'
            )
        elif rsi > 70:
            self._add_insight(
                'RSI', 'warning',
                f'RSI at {rsi:.1f} - Overbought territory. Consider taking profits.',
                'sell'
            )
        elif rsi < 20:
            self._add_insight(
                'RSI', 'critical',
                f'RSI at {rsi:.1f} - Extreme oversold. Strong bounce potential.',
                'buy'
            )
        elif rsi < 30:
            self._add_insight(
                'RSI', 'warning',
                f'RSI at {rsi:.1f} - Oversold territory. Potential buying opportunity.',
                'buy'
            )
        else:
            self._add_insight(
                'RSI', 'info',
                f'RSI at {rsi:.1f} - Neutral zone. No extreme signals.',
                'hold'
            )
    
    def interpret_macd(self, macd: float, signal: float, histogram: float):
        """Interpret MACD indicator."""
        if macd > signal and histogram > 0:
            if histogram > abs(macd) * 0.1:
                self._add_insight(
                    'MACD', 'info',
                    f'Strong bullish momentum. MACD ({macd:.2f}) above signal ({signal:.2f}).',
                    'buy'
                )
            else:
                self._add_insight(
                    'MACD', 'info',
                    f'Bullish MACD crossover. Momentum favors upside.',
                    'hold'
                )
        elif macd < signal and histogram < 0:
            if abs(histogram) > abs(macd) * 0.1:
                self._add_insight(
                    'MACD', 'warning',
                    f'Strong bearish momentum. MACD ({macd:.2f}) below signal ({signal:.2f}).',
                    'sell'
                )
            else:
                self._add_insight(
                    'MACD', 'info',
                    f'Bearish MACD crossover. Momentum favors downside.',
                    'hold'
                )
        else:
            self._add_insight(
                'MACD', 'info',
                'MACD transitioning. Watch for confirmed crossover.',
                'hold'
            )
    
    def interpret_sma(self, price: float, sma_14: float, sma_50: float = None):
        """Interpret SMA indicators."""
        if price > sma_14:
            self._add_insight(
                'SMA', 'info',
                f'Price (${price:,.0f}) above SMA-14 (${sma_14:,.0f}). Short-term uptrend.',
                'hold'
            )
        else:
            self._add_insight(
                'SMA', 'info',
                f'Price (${price:,.0f}) below SMA-14 (${sma_14:,.0f}). Short-term downtrend.',
                'hold'
            )
        
        if sma_50:
            if sma_14 > sma_50:
                self._add_insight(
                    'SMA', 'info',
                    'Golden cross pattern: SMA-14 above SMA-50. Bullish structure.',
                    'buy'
                )
            else:
                self._add_insight(
                    'SMA', 'warning',
                    'Death cross pattern: SMA-14 below SMA-50. Bearish structure.',
                    'sell'
                )
    
    def interpret_sentiment(self, fused_sentiment: float, news_sentiment: float = None, twitter_sentiment: float = None):
        """Interpret sentiment data."""
        # Fused sentiment
        if fused_sentiment > 0.3:
            self._add_insight(
                'Sentiment', 'info',
                f'Strong bullish sentiment ({fused_sentiment:.3f}). Market optimism high.',
                'buy'
            )
        elif fused_sentiment > 0.1:
            self._add_insight(
                'Sentiment', 'info',
                f'Moderately bullish sentiment ({fused_sentiment:.3f}). Positive market tone.',
                'hold'
            )
        elif fused_sentiment < -0.3:
            self._add_insight(
                'Sentiment', 'warning',
                f'Strong bearish sentiment ({fused_sentiment:.3f}). Market fear elevated.',
                'sell'
            )
        elif fused_sentiment < -0.1:
            self._add_insight(
                'Sentiment', 'info',
                f'Moderately bearish sentiment ({fused_sentiment:.3f}). Negative market tone.',
                'hold'
            )
        else:
            self._add_insight(
                'Sentiment', 'info',
                f'Neutral sentiment ({fused_sentiment:.3f}). Market undecided.',
                'hold'
            )
        
        # Source divergence
        if news_sentiment is not None and twitter_sentiment is not None:
            if abs(news_sentiment - twitter_sentiment) > 0.3:
                self._add_insight(
                    'Sentiment', 'warning',
                    f'Sentiment divergence: News ({news_sentiment:.2f}) vs Twitter ({twitter_sentiment:.2f}). Mixed signals.',
                    'hold'
                )
    
    def interpret_model_performance(self, arima_rmse: float, arimax_rmse: float):
        """Interpret model comparison."""
        improvement = (arima_rmse - arimax_rmse) / arima_rmse * 100
        
        if improvement > 10:
            self._add_insight(
                'Model', 'info',
                f'ARIMAX outperforms ARIMA by {improvement:.1f}%. Sentiment adds predictive value.',
                None
            )
        elif improvement > 0:
            self._add_insight(
                'Model', 'info',
                f'ARIMAX slightly better than ARIMA ({improvement:.1f}%). Sentiment contributes marginally.',
                None
            )
        else:
            self._add_insight(
                'Model', 'warning',
                f'ARIMA outperforms ARIMAX by {-improvement:.1f}%. Sentiment may be adding noise.',
                None
            )
    
    def interpret_volatility(self, volatility: float):
        """Interpret volatility."""
        vol_pct = volatility * 100
        
        if vol_pct > 5:
            self._add_insight(
                'Volatility', 'critical',
                f'High volatility ({vol_pct:.1f}%). Exercise caution with position sizing.',
                'hold'
            )
        elif vol_pct > 3:
            self._add_insight(
                'Volatility', 'warning',
                f'Elevated volatility ({vol_pct:.1f}%). Market uncertainty present.',
                'hold'
            )
        else:
            self._add_insight(
                'Volatility', 'info',
                f'Normal volatility ({vol_pct:.1f}%). Market conditions stable.',
                'hold'
            )
    
    def get_all_insights(self) -> List[Dict]:
        """Get all generated insights."""
        return self.insights
    
    def get_signal_summary(self) -> Dict[str, int]:
        """Get summary of signals."""
        signals = {'buy': 0, 'sell': 0, 'hold': 0}
        
        for insight in self.insights:
            signal = insight.get('signal')
            if signal in signals:
                signals[signal] += 1
        
        return signals
    
    def get_overall_signal(self) -> str:
        """Get overall trading signal based on all insights."""
        summary = self.get_signal_summary()
        
        buy = summary['buy']
        sell = summary['sell']
        
        if buy > sell + 1:
            return 'BUY'
        elif sell > buy + 1:
            return 'SELL'
        else:
            return 'HOLD'


def generate_interpretation(df: pd.DataFrame, model_metrics: Dict = None) -> Dict:
    """
    Generate interpretation from data.
    
    Args:
        df: DataFrame with indicators
        model_metrics: Optional model comparison metrics
    
    Returns:
        Dictionary with interpretations
    """
    if df.empty:
        return {'summary': 'Insufficient data for interpretation'}
    
    engine = InterpretationEngine()
    latest = df.iloc[-1]
    
    # RSI
    if 'rsi_14' in latest:
        engine.interpret_rsi(latest['rsi_14'])
    
    # MACD
    if all(k in latest for k in ['macd', 'macd_signal', 'macd_hist']):
        engine.interpret_macd(latest['macd'], latest['macd_signal'], latest['macd_hist'])
    
    # SMA
    if 'close' in latest and 'sma_14' in latest:
        sma_50 = latest.get('sma_50', None)
        engine.interpret_sma(latest['close'], latest['sma_14'], sma_50)
    
    # Sentiment
    if 'fused_sentiment' in latest:
        news = latest.get('news_sentiment', None)
        twitter = latest.get('twitter_sentiment', None)
        engine.interpret_sentiment(latest['fused_sentiment'], news, twitter)
    
    # Volatility
    if 'volatility' in latest:
        engine.interpret_volatility(latest['volatility'])
    
    # Model comparison
    if model_metrics and 'ARIMA' in model_metrics and 'ARIMAX' in model_metrics:
        arima_rmse = model_metrics['ARIMA'].get('RMSE', 0)
        arimax_rmse = model_metrics['ARIMAX'].get('RMSE', 0)
        if arima_rmse > 0 and arimax_rmse > 0:
            engine.interpret_model_performance(arima_rmse, arimax_rmse)
    
    return {
        'insights': engine.get_all_insights(),
        'signal_summary': engine.get_signal_summary(),
        'overall_signal': engine.get_overall_signal()
    }


if __name__ == "__main__":
    print("Testing Interpretation Module...")
    print("=" * 50)
    
    engine = InterpretationEngine()
    
    # Test interpretations
    engine.interpret_rsi(75)
    engine.interpret_macd(100, 80, 20)
    engine.interpret_sentiment(0.25)
    engine.interpret_volatility(0.04)
    
    print("\nGenerated Insights:")
    for insight in engine.get_all_insights():
        print(f"  [{insight['level'].upper()}] {insight['category']}: {insight['message']}")
        if insight['signal']:
            print(f"           Signal: {insight['signal'].upper()}")
    
    print(f"\nOverall Signal: {engine.get_overall_signal()}")
