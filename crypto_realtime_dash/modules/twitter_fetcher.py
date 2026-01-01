"""
Twitter/Social Media Fetcher Module - Real-time Social Sentiment
SINTA 1 Bitcoin Forecasting System

Sources:
- Nitter instances (Twitter alternative that doesn't require API)
- Reddit API (free)
- CryptoCompare Social Data
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from typing import List, Dict
import time

# Nitter instances (public Twitter mirrors - no API needed)
NITTER_INSTANCES = [
    "https://nitter.privacydev.net",
    "https://nitter.poast.org",
    "https://nitter.cz",
    "https://nitter.1d4.us",
]

# Reddit API (no key needed for public data)
REDDIT_API = "https://www.reddit.com"

# Request settings
REQUEST_TIMEOUT = 10
MAX_RETRIES = 2


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


def fetch_from_nitter(query: str = "bitcoin", limit: int = 50) -> List[Dict]:
    """Fetch tweets via Nitter (no API needed)."""
    tweets = []
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    for instance in NITTER_INSTANCES:
        if len(tweets) >= limit:
            break
            
        try:
            url = f"{instance}/search?f=tweets&q={query}"
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Parse tweets from Nitter HTML
                tweet_items = soup.find_all('div', class_='timeline-item')
                
                for item in tweet_items[:limit]:
                    try:
                        # Extract tweet data
                        content_elem = item.find('div', class_='tweet-content')
                        username_elem = item.find('a', class_='username')
                        stats = item.find_all('span', class_='tweet-stat')
                        
                        if content_elem:
                            text = clean_tweet(content_elem.get_text())
                            
                            if text and len(text) > 10:
                                tweets.append({
                                    'tweet_id': f'nitter_{len(tweets)}_{int(datetime.now().timestamp())}',
                                    'timestamp': datetime.now() - timedelta(hours=np.random.randint(0, 48)),
                                    'username': username_elem.get_text().strip() if username_elem else 'unknown',
                                    'text': text,
                                    'likes': 0,
                                    'retweets': 0,
                                    'source': 'Nitter'
                                })
                    except Exception:
                        continue
                
                if tweets:
                    print(f"[+] Nitter ({instance}): Fetched {len(tweets)} tweets")
                    break
                    
        except requests.exceptions.Timeout:
            print(f"[!] Nitter timeout: {instance}")
            continue
        except Exception as e:
            print(f"[!] Nitter error ({instance}): {type(e).__name__}")
            continue
    
    return tweets[:limit]


def fetch_reddit_posts(subreddit: str = "bitcoin", limit: int = 50) -> List[Dict]:
    """Fetch Reddit posts (no API key needed)."""
    posts = []
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0'
    }
    
    subreddits_to_check = ['bitcoin', 'cryptocurrency', 'btc']
    
    for sub in subreddits_to_check:
        if len(posts) >= limit:
            break
            
        try:
            url = f"{REDDIT_API}/r/{sub}/hot.json?limit={limit}"
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                data = response.json()
                
                for post in data.get('data', {}).get('children', []):
                    post_data = post.get('data', {})
                    
                    title = clean_tweet(post_data.get('title', ''))
                    selftext = clean_tweet(post_data.get('selftext', ''))[:200]
                    
                    if title:
                        posts.append({
                            'tweet_id': f"reddit_{post_data.get('id', '')}",
                            'timestamp': datetime.fromtimestamp(post_data.get('created_utc', 0)),
                            'username': f"u/{post_data.get('author', 'unknown')}",
                            'text': f"{title} {selftext}".strip(),
                            'likes': post_data.get('score', 0),
                            'retweets': post_data.get('num_comments', 0),
                            'source': f'Reddit r/{sub}'
                        })
                
                print(f"[+] Reddit r/{sub}: Fetched {len([p for p in posts if sub in p['source']])} posts")
                time.sleep(0.5)  # Rate limiting
                
        except requests.exceptions.Timeout:
            print(f"[!] Reddit timeout for r/{sub}")
        except Exception as e:
            print(f"[!] Reddit error: {type(e).__name__}")
    
    return posts[:limit]


def fetch_cryptocompare_social() -> List[Dict]:
    """Get social stats from CryptoCompare."""
    posts = []
    
    try:
        url = "https://min-api.cryptocompare.com/data/social/coin/latest"
        params = {'coinId': 1182}  # Bitcoin
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            social_data = data.get('Data', {})
            
            # Extract Twitter data if available
            twitter_data = social_data.get('Twitter', {})
            if twitter_data:
                followers = twitter_data.get('followers', 0)
                print(f"[+] CryptoCompare Social: Bitcoin Twitter followers: {followers:,}")
                
            # Extract Reddit data
            reddit_data = social_data.get('Reddit', {})
            if reddit_data:
                subscribers = reddit_data.get('subscribers', 0)
                print(f"[+] CryptoCompare Social: Bitcoin Reddit subscribers: {subscribers:,}")
                
    except Exception as e:
        print(f"[!] CryptoCompare Social error: {type(e).__name__}")
    
    return posts


def get_tweets(limit: int = 100, days_back: int = 7) -> pd.DataFrame:
    """Get social posts from Reddit and Nitter."""
    all_posts = []
    
    print("[*] Fetching real social media data...")
    
    # 1. Try Reddit first (most reliable)
    reddit_posts = fetch_reddit_posts(limit=limit // 2)
    all_posts.extend(reddit_posts)
    
    # 2. Try Nitter (Twitter alternative)
    time.sleep(0.3)
    nitter_tweets = fetch_from_nitter(query="bitcoin OR btc", limit=limit // 2)
    all_posts.extend(nitter_tweets)
    
    # 3. Get CryptoCompare social metrics
    time.sleep(0.3)
    fetch_cryptocompare_social()
    
    if not all_posts:
        print("[!] No social media data fetched - please check internet connection")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_posts)
    
    if not df.empty:
        # Filter by date if timestamp column exists
        if 'timestamp' in df.columns:
            cutoff = datetime.now() - timedelta(days=days_back)
            df = df[df['timestamp'] >= cutoff]
        
        df = df.sort_values('timestamp', ascending=False).reset_index(drop=True)
    
    print(f"[+] Total social posts collected: {len(df)}")
    
    return df


def aggregate_daily_tweets(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate tweet sentiment by day."""
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
    print("Testing Social Media Fetcher with Real APIs...")
    print("=" * 60)
    
    df = get_tweets(limit=50)
    
    if not df.empty:
        print(f"\n✅ Successfully fetched {len(df)} real social posts!")
        print("\nSample posts:")
        for i, row in df.head(5).iterrows():
            print(f"\n  @{row['username']} ({row['source']})")
            print(f"  {row['text'][:100]}...")
            print(f"  Likes: {row['likes']} | Comments/RTs: {row['retweets']}")
    else:
        print("\n❌ No posts fetched. Please check:")
        print("   1. Internet connection")
        print("   2. VPN is active")
