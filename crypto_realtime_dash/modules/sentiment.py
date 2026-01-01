"""
Sentiment Analysis Module - VADER Sentiment Scoring
SINTA 1 Bitcoin Forecasting System
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize VADER analyzer
analyzer = SentimentIntensityAnalyzer()

# Thresholds
BULLISH_THRESHOLD = 0.05
BEARISH_THRESHOLD = -0.05

# Fusion weights
FUSION_WEIGHTS = {
    'news': 0.4,
    'twitter': 0.4,
    'lunarcrush': 0.2
}


def compute_sentiment(text: str) -> Dict[str, float]:
    """Get VADER sentiment scores for text."""
    if not text or not isinstance(text, str):
        return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    
    return analyzer.polarity_scores(text)


def classify_sentiment(score: float) -> str:
    """Classify score as Bullish/Bearish/Neutral."""
    if score >= BULLISH_THRESHOLD:
        return 'Bullish'
    elif score <= BEARISH_THRESHOLD:
        return 'Bearish'
    else:
        return 'Neutral'


def get_sentiment_color(label: str) -> str:
    """Get color for sentiment label."""
    colors = {
        'Bullish': '#3fb950',
        'Bearish': '#f85149',
        'Neutral': '#d29922'
    }
    return colors.get(label, '#8b949e')


def analyze_news_sentiment(df_news: pd.DataFrame) -> pd.DataFrame:
    """Add VADER sentiment to news DataFrame."""
    if df_news.empty:
        return df_news
    
    df = df_news.copy()
    
    # Combine title and summary for analysis
    df['text'] = df['title'].fillna('') + ' ' + df['summary'].fillna('')
    
    # Compute sentiment
    sentiment_scores = df['text'].apply(compute_sentiment)
    
    df['sentiment_score'] = sentiment_scores.apply(lambda x: x['compound'])
    df['sentiment_label'] = df['sentiment_score'].apply(classify_sentiment)
    df['sentiment_pos'] = sentiment_scores.apply(lambda x: x['pos'])
    df['sentiment_neg'] = sentiment_scores.apply(lambda x: x['neg'])
    df['sentiment_neu'] = sentiment_scores.apply(lambda x: x['neu'])
    
    # Count by category
    bullish = (df['sentiment_label'] == 'Bullish').sum()
    bearish = (df['sentiment_label'] == 'Bearish').sum()
    neutral = (df['sentiment_label'] == 'Neutral').sum()
    
    print(f"[+] News sentiment: Bullish={bullish}, Bearish={bearish}, Neutral={neutral}")
    
    return df


def analyze_tweets_sentiment(df_tweets: pd.DataFrame) -> pd.DataFrame:
    """Add VADER sentiment to tweets DataFrame."""
    if df_tweets.empty:
        return df_tweets
    
    df = df_tweets.copy()
    
    # Compute sentiment
    sentiment_scores = df['text'].apply(compute_sentiment)
    
    df['sentiment_score'] = sentiment_scores.apply(lambda x: x['compound'])
    df['sentiment_label'] = df['sentiment_score'].apply(classify_sentiment)
    df['sentiment_pos'] = sentiment_scores.apply(lambda x: x['pos'])
    df['sentiment_neg'] = sentiment_scores.apply(lambda x: x['neg'])
    df['sentiment_neu'] = sentiment_scores.apply(lambda x: x['neu'])
    
    # Weight by engagement
    if 'likes' in df.columns and 'retweets' in df.columns:
        engagement = df['likes'] + df['retweets'] * 2
        df['weighted_sentiment'] = df['sentiment_score'] * (1 + np.log1p(engagement) / 10)
    else:
        df['weighted_sentiment'] = df['sentiment_score']
    
    # Count by category
    bullish = (df['sentiment_label'] == 'Bullish').sum()
    bearish = (df['sentiment_label'] == 'Bearish').sum()
    neutral = (df['sentiment_label'] == 'Neutral').sum()
    
    print(f"[+] Tweet sentiment: Bullish={bullish}, Bearish={bearish}, Neutral={neutral}")
    
    return df


def aggregate_daily_sentiment(df: pd.DataFrame, source: str = 'news') -> pd.DataFrame:
    """Aggregate sentiment scores by day."""
    if df.empty or 'sentiment_score' not in df.columns:
        return pd.DataFrame()
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    
    # Aggregate by date
    daily = df.groupby('date').agg({
        'sentiment_score': 'mean',
        'sentiment_pos': 'mean',
        'sentiment_neg': 'mean',
        'sentiment_neu': 'mean'
    }).reset_index()
    
    # Add count
    counts = df.groupby('date').size().reset_index(name=f'{source}_count')
    daily = daily.merge(counts, on='date', how='left')
    
    # Rename columns
    daily = daily.rename(columns={'sentiment_score': f'{source}_sentiment'})
    
    # Add label counts
    if 'sentiment_label' in df.columns:
        label_counts = df.groupby(['date', 'sentiment_label']).size().unstack(fill_value=0).reset_index()
        for label in ['Bullish', 'Bearish', 'Neutral']:
            if label not in label_counts.columns:
                label_counts[label] = 0
        label_counts.columns = ['date'] + [f'{source}_{c.lower()}' for c in label_counts.columns[1:]]
        daily = daily.merge(label_counts, on='date', how='left')
    
    daily = daily.sort_values('date').reset_index(drop=True)
    
    return daily


def fuse_sentiment(
    news_sentiment: pd.DataFrame,
    twitter_sentiment: pd.DataFrame,
    lunarcrush_sentiment: Optional[pd.DataFrame] = None,
    weights: Dict[str, float] = None
) -> pd.DataFrame:
    """Fuse sentiment: 40% news + 40% twitter + 20% lunarcrush."""
    weights = weights or FUSION_WEIGHTS
    
    # Start with news
    if not news_sentiment.empty and 'news_sentiment' in news_sentiment.columns:
        fused = news_sentiment[['date', 'news_sentiment']].copy()
    else:
        fused = pd.DataFrame(columns=['date', 'news_sentiment'])
    
    # Merge twitter
    if not twitter_sentiment.empty and 'twitter_sentiment' in twitter_sentiment.columns:
        twitter_cols = twitter_sentiment[['date', 'twitter_sentiment']].copy()
        
        if fused.empty:
            fused = twitter_cols
        else:
            fused = fused.merge(twitter_cols, on='date', how='outer')
    
    # Merge LunarCrush
    if lunarcrush_sentiment is not None and not lunarcrush_sentiment.empty:
        if 'lunarcrush_sentiment' in lunarcrush_sentiment.columns:
            lc_cols = lunarcrush_sentiment[['date', 'lunarcrush_sentiment']].copy()
            fused = fused.merge(lc_cols, on='date', how='outer')
    
    if fused.empty:
        return pd.DataFrame()
    
    # Fill missing values
    for col in ['news_sentiment', 'twitter_sentiment', 'lunarcrush_sentiment']:
        if col not in fused.columns:
            fused[col] = 0
        fused[col] = fused[col].fillna(0)
    
    # Calculate fused sentiment
    if fused['lunarcrush_sentiment'].any():
        fused['fused_sentiment'] = (
            fused['news_sentiment'] * weights['news'] +
            fused['twitter_sentiment'] * weights['twitter'] +
            fused['lunarcrush_sentiment'] * weights['lunarcrush']
        )
    else:
        # Redistribute LunarCrush weight
        total = weights['news'] + weights['twitter']
        news_w = weights['news'] / total
        twitter_w = weights['twitter'] / total
        
        fused['fused_sentiment'] = (
            fused['news_sentiment'] * news_w +
            fused['twitter_sentiment'] * twitter_w
        )
    
    # Add label
    fused['fused_label'] = fused['fused_sentiment'].apply(classify_sentiment)
    
    fused = fused.sort_values('date').reset_index(drop=True)
    
    print(f"[+] Fused sentiment for {len(fused)} days")
    
    return fused


def get_sentiment_summary(df: pd.DataFrame) -> Dict:
    """Get sentiment summary statistics."""
    if df.empty:
        return {}
    
    sent_col = None
    for col in ['fused_sentiment', 'sentiment_score', 'news_sentiment']:
        if col in df.columns:
            sent_col = col
            break
    
    if not sent_col:
        return {}
    
    return {
        'mean': df[sent_col].mean(),
        'std': df[sent_col].std(),
        'min': df[sent_col].min(),
        'max': df[sent_col].max(),
        'latest': df[sent_col].iloc[-1] if len(df) > 0 else 0,
        'trend': 'Improving' if len(df) > 1 and df[sent_col].iloc[-1] > df[sent_col].iloc[-7:].mean() else 'Declining'
    }


if __name__ == "__main__":
    print("Testing Sentiment Module...")
    print("=" * 50)
    
    # Test VADER
    test_texts = [
        "Bitcoin surges to new all-time high! Amazing news!",
        "BTC crashes dramatically, market in panic",
        "Bitcoin price stable at current levels"
    ]
    
    print("\nVADER Analysis:")
    for text in test_texts:
        score = compute_sentiment(text)['compound']
        label = classify_sentiment(score)
        print(f"  [{label}] ({score:+.3f}): {text[:50]}...")
