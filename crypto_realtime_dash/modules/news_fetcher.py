"""
News Fetcher Module - Real-time Crypto News from Multiple Sources
SINTA 1 Bitcoin Forecasting System

Sources:
- CoinDesk RSS
- CoinTelegraph RSS
- Bitcoin Magazine RSS
- Decrypt RSS
- CryptoCompare News API (free)
- NewsData.io (free tier)
"""

import feedparser
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict
import re
import time

# RSS Feed URLs - These are reliable and don't require API keys
RSS_FEEDS = {
    'coindesk': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
    'cointelegraph': 'https://cointelegraph.com/rss',
    'bitcoinmagazine': 'https://bitcoinmagazine.com/feed',
    'decrypt': 'https://decrypt.co/feed',
    'theblock': 'https://www.theblock.co/rss.xml',
    'cryptonews': 'https://cryptonews.com/news/feed/',
}

# Free API endpoints
CRYPTOCOMPARE_NEWS = "https://min-api.cryptocompare.com/data/v2/news/"
NEWSDATA_API = "https://newsdata.io/api/1/news"

# Request timeout (seconds)
REQUEST_TIMEOUT = 10


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    
    # Remove HTML tags
    text = BeautifulSoup(text, 'html.parser').get_text()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?\'\"-]', '', text)
    
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
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%d %H:%M:%S',
        '%d %b %Y %H:%M:%S',
        '%Y-%m-%d'
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
        # Use a custom user agent to avoid blocks
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        # Fetch with requests first for better error handling
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        
        feed = feedparser.parse(response.content)
        
        if feed.bozo and not feed.entries:
            print(f"[!] RSS parse error for {source}: {getattr(feed, 'bozo_exception', 'Unknown error')}")
            return []
        
        for entry in feed.entries[:30]:  # Limit to 30 per source
            article = {
                'timestamp': parse_date(entry.get('published', entry.get('pubDate', ''))),
                'title': clean_text(entry.get('title', '')),
                'summary': clean_text(entry.get('summary', entry.get('description', ''))),
                'source': source,
                'url': entry.get('link', '')
            }
            
            # Filter Bitcoin-related articles
            content = (article['title'] + ' ' + article['summary']).lower()
            keywords = ['bitcoin', 'btc', 'crypto', 'cryptocurrency', 'blockchain', 'halving', 'mining']
            
            if any(kw in content for kw in keywords):
                articles.append(article)
        
        print(f"[+] {source}: Fetched {len(articles)} Bitcoin-related articles")
        
    except requests.exceptions.Timeout:
        print(f"[!] Timeout fetching {source}")
    except requests.exceptions.ConnectionError:
        print(f"[!] Connection error for {source}")
    except Exception as e:
        print(f"[!] Error fetching {source}: {type(e).__name__}: {e}")
    
    return articles


def fetch_cryptocompare_news(limit: int = 50) -> List[Dict]:
    """
    Fetch news from CryptoCompare API (free, no API key required).
    
    Args:
        limit: Number of articles to fetch
    
    Returns:
        List of article dictionaries
    """
    articles = []
    
    try:
        params = {
            'lang': 'EN',
            'categories': 'BTC,Bitcoin,Mining,Regulation,Blockchain',
            'excludeCategories': 'Sponsored'
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(
            CRYPTOCOMPARE_NEWS, 
            params=params, 
            headers=headers,
            timeout=REQUEST_TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            news_items = data.get('Data', [])
            
            for item in news_items[:limit]:
                articles.append({
                    'timestamp': datetime.fromtimestamp(item.get('published_on', 0)),
                    'title': clean_text(item.get('title', '')),
                    'summary': clean_text(item.get('body', ''))[:500],
                    'source': item.get('source_info', {}).get('name', 'CryptoCompare'),
                    'url': item.get('url', ''),
                    'categories': item.get('categories', '')
                })
            
            print(f"[+] CryptoCompare: Fetched {len(articles)} articles")
        else:
            print(f"[!] CryptoCompare API error: {response.status_code}")
            
    except requests.exceptions.Timeout:
        print("[!] CryptoCompare timeout")
    except Exception as e:
        print(f"[!] CryptoCompare error: {type(e).__name__}: {e}")
    
    return articles


def fetch_lunarcrush_news() -> List[Dict]:
    """
    Fetch social data from LunarCrush (requires free API key).
    Note: This is a placeholder - you need to register at lunarcrush.com for an API key.
    """
    # LunarCrush requires API key, this is a placeholder
    # Register at https://lunarcrush.com/developers for free API access
    print("[*] LunarCrush requires API key - skipping")
    return []


def fetch_coindesk() -> List[Dict]:
    """Fetch news from Coindesk RSS."""
    return fetch_rss_feed(RSS_FEEDS['coindesk'], 'CoinDesk')


def fetch_cointelegraph() -> List[Dict]:
    """Fetch news from Cointelegraph RSS."""
    return fetch_rss_feed(RSS_FEEDS['cointelegraph'], 'CoinTelegraph')


def fetch_bitcoinmagazine() -> List[Dict]:
    """Fetch news from Bitcoin Magazine RSS."""
    return fetch_rss_feed(RSS_FEEDS['bitcoinmagazine'], 'Bitcoin Magazine')


def fetch_decrypt() -> List[Dict]:
    """Fetch news from Decrypt RSS."""
    return fetch_rss_feed(RSS_FEEDS['decrypt'], 'Decrypt')


def fetch_all_rss_sources() -> List[Dict]:
    """Fetch from all RSS sources with rate limiting."""
    all_articles = []
    
    for source_name, url in RSS_FEEDS.items():
        try:
            articles = fetch_rss_feed(url, source_name.replace('_', ' ').title())
            all_articles.extend(articles)
            time.sleep(0.3)  # Rate limiting between requests
        except Exception as e:
            print(f"[!] Failed to fetch {source_name}: {e}")
            continue
    
    return all_articles


def get_all_news(use_sample_fallback: bool = False) -> pd.DataFrame:
    """
    Fetch news from all sources.
    
    Args:
        use_sample_fallback: If True, use sample data when APIs fail.
                            Set to False to ONLY use real data.
    
    Returns:
        DataFrame with all news articles
    """
    all_articles = []
    
    print("[*] Fetching real news data from multiple sources...")
    
    # 1. Fetch from CryptoCompare API (most reliable)
    cryptocompare_articles = fetch_cryptocompare_news(limit=30)
    all_articles.extend(cryptocompare_articles)
    
    # 2. Fetch from main RSS sources
    time.sleep(0.3)
    coindesk_articles = fetch_coindesk()
    all_articles.extend(coindesk_articles)
    
    time.sleep(0.3)
    cointelegraph_articles = fetch_cointelegraph()
    all_articles.extend(cointelegraph_articles)
    
    # 3. Fetch from alternative RSS sources if needed
    if len(all_articles) < 20:
        time.sleep(0.3)
        all_articles.extend(fetch_bitcoinmagazine())
        
        time.sleep(0.3)
        all_articles.extend(fetch_decrypt())
    
    # Only use sample fallback if explicitly enabled AND no articles found
    if len(all_articles) == 0 and use_sample_fallback:
        print("[!] No real articles found - this should not happen with VPN")
        print("[!] Please check your internet connection")
        return pd.DataFrame()
    
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
    print("Testing News Fetcher with Real APIs...")
    print("=" * 60)
    
    df = get_all_news(use_sample_fallback=False)
    
    if not df.empty:
        print(f"\n✅ Successfully fetched {len(df)} real articles!")
        print("\nSample articles:")
        for i, row in df.head(5).iterrows():
            print(f"\n  [{row['source']}] {row['timestamp']}")
            print(f"  {row['title'][:80]}...")
    else:
        print("\n❌ No articles fetched. Please check:")
        print("   1. Internet connection")
        print("   2. VPN is active")
        print("   3. No firewall blocking")
