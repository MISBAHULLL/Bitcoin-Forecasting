"""
News Fetcher Module - Coindesk & Cointelegraph RSS
SINTA 1 Bitcoin Forecasting System
"""

import feedparser
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict
import re
import time

# RSS Feed URLs
RSS_FEEDS = {
    'coindesk': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
    'cointelegraph': 'https://cointelegraph.com/rss'
}

# Alternative feeds
ALTERNATIVE_FEEDS = {
    'bitcoin_magazine': 'https://bitcoinmagazine.com/feed',
    'decrypt': 'https://decrypt.co/feed'
}


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    
    # Remove HTML tags
    text = BeautifulSoup(text, 'html.parser').get_text()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?\'"-]', '', text)
    
    return text[:1000]  # Limit length


def parse_date(date_str: str) -> datetime:
    """Parse various date formats."""
    if not date_str:
        return datetime.now()
    
    # Common date formats
    formats = [
        '%a, %d %b %Y %H:%M:%S %z',
        '%a, %d %b %Y %H:%M:%S %Z',
        '%Y-%m-%dT%H:%M:%S%z',
        '%Y-%m-%d %H:%M:%S',
        '%d %b %Y %H:%M:%S'
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt).replace(tzinfo=None)
        except ValueError:
            continue
    
    return datetime.now()


def fetch_rss_feed(url: str, source: str) -> List[Dict]:
    """
    Fetch and parse RSS feed.
    
    Args:
        url: RSS feed URL
        source: Source name
    
    Returns:
        List of article dictionaries
    """
    articles = []
    
    try:
        feed = feedparser.parse(url)
        
        if feed.bozo and not feed.entries:
            print(f"[!] RSS error for {source}: {feed.bozo_exception}")
            return []
        
        for entry in feed.entries[:30]:  # Limit to 30 per source
            article = {
                'timestamp': parse_date(entry.get('published', '')),
                'title': clean_text(entry.get('title', '')),
                'summary': clean_text(entry.get('summary', entry.get('description', ''))),
                'source': source,
                'url': entry.get('link', '')
            }
            
            # Filter Bitcoin-related articles
            content = (article['title'] + ' ' + article['summary']).lower()
            keywords = ['bitcoin', 'btc', 'crypto', 'cryptocurrency']
            
            if any(kw in content for kw in keywords):
                articles.append(article)
        
        print(f"[+] {source}: Fetched {len(articles)} articles")
        
    except Exception as e:
        print(f"[!] Error fetching {source}: {e}")
    
    return articles


def fetch_coindesk() -> List[Dict]:
    """Fetch news from Coindesk RSS."""
    return fetch_rss_feed(RSS_FEEDS['coindesk'], 'Coindesk')


def fetch_cointelegraph() -> List[Dict]:
    """Fetch news from Cointelegraph RSS."""
    return fetch_rss_feed(RSS_FEEDS['cointelegraph'], 'Cointelegraph')


def fetch_alternative_sources() -> List[Dict]:
    """Fetch from alternative sources if main ones fail."""
    articles = []
    
    for source, url in ALTERNATIVE_FEEDS.items():
        try:
            articles.extend(fetch_rss_feed(url, source.replace('_', ' ').title()))
        except Exception:
            continue
    
    return articles


def generate_sample_news(count: int = 20) -> List[Dict]:
    """Generate sample news data when APIs fail."""
    print("[*] Generating sample news...")
    
    sample_titles = [
        "Bitcoin surges past $98,000 amid institutional buying",
        "Major hedge fund announces Bitcoin allocation",
        "Bitcoin ETF sees record inflows this week",
        "Cryptocurrency adoption grows in emerging markets",
        "Bitcoin mining difficulty reaches new all-time high",
        "Central banks exploring Bitcoin reserves",
        "Bitcoin volatility decreases as market matures",
        "Institutional investors increase crypto exposure",
        "Bitcoin network hash rate hits new record",
        "Major payment company adds Bitcoin support",
        "Bitcoin dominance increases amid altcoin selloff",
        "Regulatory clarity boosts Bitcoin sentiment",
        "Bitcoin futures open interest reaches peak",
        "Long-term holders accumulate more Bitcoin",
        "Bitcoin correlation with stocks decreases",
        "Lightning Network capacity reaches new high",
        "Bitcoin mining becomes more energy efficient",
        "Major bank launches Bitcoin custody service",
        "Bitcoin options market shows bullish sentiment",
        "Crypto winter fears fade as Bitcoin rallies"
    ]
    
    articles = []
    base_time = datetime.now()
    
    for i, title in enumerate(sample_titles[:count]):
        articles.append({
            'timestamp': base_time - timedelta(hours=i*2),
            'title': title,
            'summary': f"Analysis of {title.lower()}. Market experts weigh in on the implications.",
            'source': 'Sample',
            'url': f'https://example.com/news/{i}'
        })
    
    return articles


def get_all_news() -> pd.DataFrame:
    """
    Fetch news from all sources.
    
    Returns:
        DataFrame with all news articles
    """
    all_articles = []
    
    # Fetch from main sources
    all_articles.extend(fetch_coindesk())
    time.sleep(0.5)  # Rate limiting
    
    all_articles.extend(fetch_cointelegraph())
    time.sleep(0.5)
    
    # If not enough articles, try alternatives
    if len(all_articles) < 10:
        all_articles.extend(fetch_alternative_sources())
    
    # If still not enough, generate samples
    if len(all_articles) < 5:
        all_articles.extend(generate_sample_news(20))
    
    # Convert to DataFrame
    df = pd.DataFrame(all_articles)
    
    if not df.empty:
        # Remove duplicates by title
        df = df.drop_duplicates(subset=['title']).reset_index(drop=True)
        
        # Sort by timestamp
        df = df.sort_values('timestamp', ascending=False).reset_index(drop=True)
    
    print(f"[+] Total unique articles: {len(df)}")
    
    return df


if __name__ == "__main__":
    print("Testing News Fetcher...")
    print("=" * 50)
    
    df = get_all_news()
    
    if not df.empty:
        print(f"\nSample articles:")
        for i, row in df.head(5).iterrows():
            print(f"\n  [{row['source']}] {row['timestamp']}")
            print(f"  {row['title'][:80]}...")
