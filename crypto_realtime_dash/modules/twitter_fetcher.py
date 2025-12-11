"""
Twitter Fetcher Module - snscrape-based Real-time Tweet Mining
SINTA 1 Bitcoin Forecasting System
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from typing import List, Dict

# Try to import snscrape
try:
    import snscrape.modules.twitter as sntwitter
    SNSCRAPE_AVAILABLE = True
except ImportError:
    SNSCRAPE_AVAILABLE = False
    print("[!] snscrape not installed. Using sample tweets.")


def clean_tweet(text: str) -> str:
    """Clean tweet text for sentiment analysis."""
    if not text:
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtag symbol but keep the text
    text = re.sub(r'#', '', text)
    
    # Remove RT prefix
    text = re.sub(r'^RT[\s]+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove special characters
    text = re.sub(r'[^\w\s.,!?\'\"-]', '', text)
    
    return text[:500]


def fetch_tweets_snscrape(query: str = "Bitcoin OR BTC", limit: int = 100, days_back: int = 7) -> List[Dict]:
    """
    Fetch tweets using snscrape.
    
    Args:
        query: Search query
        limit: Maximum number of tweets
        days_back: How many days back to search
    
    Returns:
        List of tweet dictionaries
    """
    if not SNSCRAPE_AVAILABLE:
        return []
    
    tweets = []
    since_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    full_query = f"{query} lang:en since:{since_date}"
    
    try:
        scraper = sntwitter.TwitterSearchScraper(full_query)
        
        for i, tweet in enumerate(scraper.get_items()):
            if i >= limit:
                break
            
            tweets.append({
                'tweet_id': str(tweet.id),
                'timestamp': tweet.date.replace(tzinfo=None),
                'username': tweet.user.username if tweet.user else 'unknown',
                'text': clean_tweet(tweet.rawContent),
                'likes': tweet.likeCount or 0,
                'retweets': tweet.retweetCount or 0
            })
        
        print(f"[+] snscrape: Fetched {len(tweets)} tweets")
        
    except Exception as e:
        print(f"[!] snscrape error: {e}")
    
    return tweets


def generate_sample_tweets(count: int = 100) -> List[Dict]:
    """Generate sample tweet data when snscrape is unavailable."""
    print("[*] Generating sample tweets...")
    
    sample_templates = [
        "Bitcoin is looking bullish today! $BTC to the moon!",
        "Just bought more BTC. Long term holder here.",
        "Bitcoin breaking resistance levels. Very bullish!",
        "BTC price action looking strong. Next stop $100k",
        "Sold my Bitcoin. Market looking weak.",
        "Bitcoin dump incoming? Be careful out there.",
        "BTC forming a bearish pattern. Not looking good.",
        "Lost money on Bitcoin again. This is frustrating.",
        "Bitcoin trading sideways. Waiting for breakout.",
        "BTC price stable. Nothing exciting happening.",
        "Watching Bitcoin closely. Could go either way.",
        "Bitcoin consolidating. Normal market behavior.",
        "Accumulating more BTC on this dip.",
        "Bitcoin technical analysis suggests bullish trend.",
        "BTC whales are buying. Follow the smart money.",
        "Bitcoin hash rate at ATH. Network stronger than ever.",
        "Institutional money flowing into Bitcoin.",
        "Another day, another Bitcoin all-time high soon!",
        "BTC RSI overbought. Might see a correction.",
        "Bitcoin MACD showing bullish crossover!"
    ]
    
    tweets = []
    base_time = datetime.now()
    
    np.random.seed(int(datetime.now().timestamp()) % 1000)
    
    for i in range(count):
        template = np.random.choice(sample_templates)
        
        tweets.append({
            'tweet_id': f'sample_{i}_{int(base_time.timestamp())}',
            'timestamp': base_time - timedelta(hours=np.random.randint(0, 168)),
            'username': f'crypto_user_{np.random.randint(1000, 9999)}',
            'text': template,
            'likes': np.random.randint(0, 1000),
            'retweets': np.random.randint(0, 500)
        })
    
    return tweets


def get_tweets(limit: int = 100, days_back: int = 7) -> pd.DataFrame:
    """
    Get tweets with fallback to sample data.
    
    Args:
        limit: Maximum number of tweets
        days_back: Days to look back
    
    Returns:
        DataFrame with tweets
    """
    # Try snscrape first
    tweets = fetch_tweets_snscrape(limit=limit, days_back=days_back)
    
    # Fallback to sample data
    if not tweets:
        tweets = generate_sample_tweets(count=limit)
    
    df = pd.DataFrame(tweets)
    
    if not df.empty:
        df = df.sort_values('timestamp', ascending=False).reset_index(drop=True)
    
    print(f"[+] Total tweets collected: {len(df)}")
    
    return df


def aggregate_daily_tweets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate tweet sentiment by day.
    
    Args:
        df: DataFrame with tweet data including sentiment
    
    Returns:
        DataFrame with daily aggregates
    """
    if df.empty or 'sentiment_score' not in df.columns:
        return pd.DataFrame()
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    
    # Aggregate by date
    daily = df.groupby('date').agg({
        'sentiment_score': 'mean',
        'tweet_id': 'count',
        'likes': 'sum',
        'retweets': 'sum'
    }).reset_index()
    
    daily.columns = ['date', 'avg_sentiment', 'tweet_count', 'total_likes', 'total_retweets']
    
    # Count sentiment categories
    if 'sentiment_label' in df.columns:
        label_counts = df.groupby(['date', 'sentiment_label']).size().unstack(fill_value=0).reset_index()
        for label in ['Bullish', 'Bearish', 'Neutral']:
            if label not in label_counts.columns:
                label_counts[label] = 0
        
        daily = daily.merge(label_counts[['date', 'Bullish', 'Bearish', 'Neutral']], on='date', how='left')
    
    daily = daily.sort_values('date').reset_index(drop=True)
    
    return daily


if __name__ == "__main__":
    print("Testing Twitter Fetcher...")
    print("=" * 50)
    
    df = get_tweets(limit=50)
    
    if not df.empty:
        print(f"\nSample tweets:")
        for i, row in df.head(3).iterrows():
            print(f"\n  @{row['username']} ({row['timestamp']})")
            print(f"  {row['text'][:100]}...")
            print(f"  Likes: {row['likes']} | Retweets: {row['retweets']}")
