"""
Modules Package
SINTA 1 Bitcoin Forecasting System
"""

# Database
from .db import db, Database

# Data Fetching
from .price_fetcher import get_ohlcv, fetch_current_price, TIMEFRAMES
from .news_fetcher import get_all_news
from .twitter_fetcher import get_tweets

# Analysis
from .sentiment import (
    compute_sentiment,
    classify_sentiment,
    analyze_news_sentiment,
    analyze_tweets_sentiment,
    aggregate_daily_sentiment,
    fuse_sentiment
)
from .indicators import compute_all_indicators, get_indicator_summary

# Models
from .arima_model import train_arima, train_arimax, ARIMAModel, ARIMAXModel

# Evaluation
from .evaluation import calculate_all_metrics, compare_models

# Interpretation
from .interpretation import generate_interpretation, InterpretationEngine

# Utilities
from .utils import load_config, format_price, format_percentage

__all__ = [
    'db', 'Database',
    'get_ohlcv', 'fetch_current_price', 'TIMEFRAMES',
    'get_all_news', 'get_tweets',
    'compute_sentiment', 'classify_sentiment',
    'analyze_news_sentiment', 'analyze_tweets_sentiment',
    'aggregate_daily_sentiment', 'fuse_sentiment',
    'compute_all_indicators', 'get_indicator_summary',
    'train_arima', 'train_arimax', 'ARIMAModel', 'ARIMAXModel',
    'calculate_all_metrics', 'compare_models',
    'generate_interpretation', 'InterpretationEngine',
    'load_config', 'format_price', 'format_percentage'
]
