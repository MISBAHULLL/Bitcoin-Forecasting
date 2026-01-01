"""
Main Pipeline - ETL -> Features -> Forecasting
SINTA 1 Bitcoin Forecasting System
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Import modules
from modules.db import db
from modules.price_fetcher import get_ohlcv, fetch_current_price
from modules.news_fetcher import get_all_news
from modules.twitter_fetcher import get_tweets
from modules.sentiment import (
    analyze_news_sentiment,
    analyze_tweets_sentiment,
    aggregate_daily_sentiment,
    fuse_sentiment
)
from modules.indicators import compute_all_indicators
from modules.arima_model import train_arima, train_arimax
from modules.evaluation import compare_models, generate_evaluation_report
from modules.interpretation import generate_interpretation


def run_pipeline(interval: str = '1h', limit: int = 500, test_size: float = 0.2):
    """Run ETL -> Features -> Forecasting pipeline."""
    print("=" * 70)
    print("  BITCOIN SENTIMENT FORECASTING PIPELINE")
    print("  SINTA 1 Research System")
    print("=" * 70)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration: interval={interval}, limit={limit}, test_size={test_size}")
    
    results = {}
    
    # --- Phase 1: Data Acquisition ---
    print("\n" + "=" * 70)
    print("  PHASE 1: DATA ACQUISITION")
    print("=" * 70)
    
    # 1.1 Fetch OHLCV data
    print("\n[1/4] Fetching Bitcoin OHLCV data...")
    df_price = get_ohlcv(interval=interval, limit=limit)
    
    if df_price.empty:
        print("[!] Failed to fetch price data")
        return None
    
    print(f"  Records: {len(df_price)}")
    print(f"  Range: {df_price['timestamp'].min()} to {df_price['timestamp'].max()}")
    print(f"  Current Price: ${df_price['close'].iloc[-1]:,.2f}")
    
    # Save to database
    saved = db.save_prices(df_price)
    print(f"  Saved to DB: {saved} records")
    
    results['price_data'] = df_price
    
    # 1.2 Fetch news data
    print("\n[2/4] Fetching news data...")
    df_news = get_all_news()
    
    if not df_news.empty:
        saved = db.save_news(df_news)
        print(f"  Saved to DB: {saved} records")
    
    results['news_data'] = df_news
    
    # 1.3 Fetch tweet data
    print("\n[3/4] Fetching Twitter data...")
    df_tweets = get_tweets(limit=100)
    
    if not df_tweets.empty:
        saved = db.save_tweets(df_tweets)
        print(f"  Saved to DB: {saved} records")
    
    results['tweet_data'] = df_tweets
    
    # 1.4 Current price
    print("\n[4/4] Fetching current price...")
    current_price = fetch_current_price()
    print(f"  Price: ${current_price.get('price', 0):,.2f}")
    print(f"  24h Change: {current_price.get('change_24h', 0):+.2f}%")
    print(f"  Source: {current_price.get('source', 'Unknown')}")
    
    results['current_price'] = current_price
    
    # --- Phase 2: Feature Engineering ---
    print("\n" + "=" * 70)
    print("  PHASE 2: FEATURE ENGINEERING")
    print("=" * 70)
    
    # 2.1 Technical indicators
    print("\n[1/3] Computing technical indicators...")
    df_price = compute_all_indicators(df_price)
    
    # 2.2 Sentiment analysis
    print("\n[2/3] Analyzing sentiment (VADER)...")
    
    # News sentiment
    if not df_news.empty:
        df_news = analyze_news_sentiment(df_news)
        df_news_daily = aggregate_daily_sentiment(df_news, source='news')
    else:
        df_news_daily = pd.DataFrame()
    
    # Tweet sentiment
    if not df_tweets.empty:
        df_tweets = analyze_tweets_sentiment(df_tweets)
        df_tweets_daily = aggregate_daily_sentiment(df_tweets, source='twitter')
    else:
        df_tweets_daily = pd.DataFrame()
    
    # 2.3 Fuse sentiment
    print("\n[3/3] Fusing sentiment sources...")
    df_sentiment = fuse_sentiment(df_news_daily, df_tweets_daily)
    
    results['sentiment_data'] = df_sentiment
    
    # --- Phase 3: Data Merging ---
    print("\n" + "=" * 70)
    print("  PHASE 3: DATA MERGING")
    print("=" * 70)
    
    # Merge price and sentiment
    df_merged = df_price.copy()
    df_merged['date'] = pd.to_datetime(df_merged['timestamp']).dt.date
    
    if not df_sentiment.empty:
        df_sentiment['date'] = pd.to_datetime(df_sentiment['date']).dt.date
        
        merge_cols = ['date', 'fused_sentiment', 'news_sentiment', 'twitter_sentiment']
        available_cols = [c for c in merge_cols if c in df_sentiment.columns]
        
        df_merged = df_merged.merge(
            df_sentiment[available_cols],
            on='date',
            how='left'
        )
    
    # Fill missing sentiment
    for col in ['fused_sentiment', 'news_sentiment', 'twitter_sentiment']:
        if col not in df_merged.columns:
            df_merged[col] = 0
        df_merged[col] = df_merged[col].fillna(0)
    
    # Clean data
    df_merged = df_merged.dropna(subset=['close', 'rsi_14', 'macd']).reset_index(drop=True)
    
    print(f"[+] Merged dataset: {len(df_merged)} records")
    print(f"  Columns: {list(df_merged.columns)}")
    
    # Save daily features to database
    daily_features = df_merged.groupby('date').agg({
        'close': 'last',
        'volume': 'sum',
        'rsi_14': 'last',
        'sma_14': 'last',
        'macd': 'last',
        'macd_signal': 'last',
        'macd_hist': 'last',
        'fused_sentiment': 'last',
        'news_sentiment': 'last',
        'twitter_sentiment': 'last'
    }).reset_index()
    
    saved = db.save_daily_features(daily_features)
    print(f"[+] Saved daily features to DB: {saved} records")
    
    results['merged_data'] = df_merged
    
    # --- Phase 4: ARIMA/ARIMAX Forecasting ---
    print("\n" + "=" * 70)
    print("  PHASE 4: ARIMA/ARIMAX FORECASTING")
    print("=" * 70)
    
    model_results = {}
    
    # 4.1 ARIMA Baseline
    print("\n[1/2] Training ARIMA Baseline...")
    try:
        arima_result = train_arima(df_merged, order=(5, 1, 0), test_size=test_size)
        model_results['ARIMA'] = arima_result
    except Exception as e:
        print(f"  [!] ARIMA failed: {e}")
    
    # 4.2 ARIMAX with exogenous variables
    print("\n[2/2] Training ARIMAX...")
    try:
        exog_columns = ['fused_sentiment', 'rsi_14', 'sma_14', 'macd', 'volume']
        arimax_result = train_arimax(df_merged, exog_columns=exog_columns, order=(5, 1, 0), test_size=test_size)
        model_results['ARIMAX'] = arimax_result
    except Exception as e:
        print(f"  [!] ARIMAX failed: {e}")
    
    results['model_results'] = model_results
    
    # --- Phase 5: Evaluation ---
    print("\n" + "=" * 70)
    print("  PHASE 5: EVALUATION")
    print("=" * 70)
    
    if model_results:
        comparison = compare_models(model_results)
        print("\nModel Comparison:")
        print(comparison.to_string(index=False))
        
        report = generate_evaluation_report(model_results)
        print(f"\n{report}")
        
        results['comparison'] = comparison
    
    # --- Phase 6: Interpretation ---
    print("\n" + "=" * 70)
    print("  PHASE 6: INTERPRETATION")
    print("=" * 70)
    
    interpretation = generate_interpretation(df_merged, model_results)
    
    print("\nInsights:")
    for insight in interpretation.get('insights', [])[:5]:
        signal = f"[{insight['signal'].upper()}]" if insight.get('signal') else ""
        print(f"  [{insight['level'].upper()}] {insight['category']}: {insight['message']} {signal}")
    
    print(f"\n[OVERALL SIGNAL] {interpretation.get('overall_signal', 'HOLD')}")
    
    results['interpretation'] = interpretation
    
    # --- Complete ---
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nTo launch dashboard:")
    print("  python app_dash.py")
    print("  Then open: http://localhost:8050")
    
    return results


if __name__ == "__main__":
    results = run_pipeline(interval='1h', limit=200, test_size=0.2)
