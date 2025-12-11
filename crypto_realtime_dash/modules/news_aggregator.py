"""
News Aggregator Module - Real-time Crypto News & Events
SINTA 1 Bitcoin Forecasting System

Integrates news from:
- news_fetcher.py (RSS feeds + CryptoCompare)
- Real cryptocurrency events calendar
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict
import requests

# Import from news_fetcher
try:
    from modules.news_fetcher import get_all_news, clean_text
except ImportError:
    from news_fetcher import get_all_news, clean_text

# Import sentiment analyzer
try:
    from modules.sentiment import analyze_text_sentiment
except ImportError:
    try:
        from sentiment import analyze_text_sentiment
    except ImportError:
        # Simple fallback sentiment
        def analyze_text_sentiment(text):
            positive_words = ['bullish', 'surge', 'rally', 'gain', 'rise', 'up', 'high', 'record', 'buy', 'growth']
            negative_words = ['bearish', 'crash', 'dump', 'fall', 'drop', 'down', 'low', 'sell', 'fear', 'loss']
            
            text_lower = text.lower()
            pos_count = sum(1 for w in positive_words if w in text_lower)
            neg_count = sum(1 for w in negative_words if w in text_lower)
            
            if pos_count > neg_count:
                return {'score': 0.3 + (pos_count * 0.1), 'label': 'Bullish'}
            elif neg_count > pos_count:
                return {'score': -0.3 - (neg_count * 0.1), 'label': 'Bearish'}
            return {'score': 0, 'label': 'Neutral'}


# ============================================
# MAJOR CRYPTO EVENTS CALENDAR 2024-2025
# ============================================

CRYPTO_EVENTS = [
    # FOMC Meetings 2024-2025
    {"date": "2024-12-18", "event": "FOMC Meeting", "impact": "High", "description": "Fed interest rate decision"},
    {"date": "2025-01-29", "event": "FOMC Meeting", "impact": "High", "description": "Fed interest rate decision"},
    {"date": "2025-03-19", "event": "FOMC Meeting", "impact": "High", "description": "Fed interest rate decision + SEP"},
    {"date": "2025-05-07", "event": "FOMC Meeting", "impact": "High", "description": "Fed interest rate decision"},
    {"date": "2025-06-18", "event": "FOMC Meeting", "impact": "High", "description": "Fed interest rate decision + SEP"},
    {"date": "2025-07-30", "event": "FOMC Meeting", "impact": "High", "description": "Fed interest rate decision"},
    
    # Bitcoin Events
    {"date": "2024-04-20", "event": "Bitcoin Halving", "impact": "Critical", "description": "Block reward halved to 3.125 BTC"},
    {"date": "2025-01-20", "event": "SEC ETF Review", "impact": "High", "description": "Potential spot ETF approval deadline"},
    {"date": "2025-04-20", "event": "Bitcoin Halving Anniversary", "impact": "Medium", "description": "1 year since halving"},
    
    # Economic Reports (recurring pattern)
    {"date": "2024-12-06", "event": "US Jobs Report", "impact": "Medium", "description": "Non-farm payrolls data"},
    {"date": "2024-12-11", "event": "CPI Data", "impact": "High", "description": "Consumer Price Index release"},
    {"date": "2024-12-12", "event": "PPI Data", "impact": "Medium", "description": "Producer Price Index release"},
    {"date": "2025-01-10", "event": "US Jobs Report", "impact": "Medium", "description": "Non-farm payrolls data"},
    {"date": "2025-01-15", "event": "CPI Data", "impact": "High", "description": "Consumer Price Index release"},
]


def get_upcoming_events(days_ahead: int = 30) -> List[Dict]:
    """Get upcoming major events within the next N days."""
    today = datetime.now().date()
    upcoming = []
    
    for event in CRYPTO_EVENTS:
        try:
            event_date = datetime.strptime(event["date"], "%Y-%m-%d").date()
            days_until = (event_date - today).days
            
            if 0 <= days_until <= days_ahead:
                upcoming.append({
                    **event,
                    "days_until": days_until,
                    "date_formatted": event_date.strftime("%b %d, %Y")
                })
        except ValueError:
            continue
    
    return sorted(upcoming, key=lambda x: x["days_until"])


def get_past_events(days_back: int = 7) -> List[Dict]:
    """Get recent major events from the past N days."""
    today = datetime.now().date()
    past = []
    
    for event in CRYPTO_EVENTS:
        try:
            event_date = datetime.strptime(event["date"], "%Y-%m-%d").date()
            days_ago = (today - event_date).days
            
            if 0 <= days_ago <= days_back:
                past.append({
                    **event,
                    "days_ago": days_ago,
                    "date_formatted": event_date.strftime("%b %d, %Y")
                })
        except ValueError:
            continue
    
    return sorted(past, key=lambda x: x["days_ago"])


# ============================================
# REAL NEWS INTEGRATION
# ============================================

def fetch_aggregated_news(limit: int = 10) -> List[Dict]:
    """
    Fetch aggregated news from real sources.
    Uses the news_fetcher module to get real RSS and API data.
    """
    try:
        # Get real news from news_fetcher
        df = get_all_news(use_sample_fallback=False)
        
        if df.empty:
            print("[!] No real news available")
            return []
        
        news_items = []
        now = datetime.now()
        
        for i, row in df.head(limit).iterrows():
            # Calculate hours ago
            if 'timestamp' in row and pd.notna(row['timestamp']):
                hours_ago = max(0, int((now - pd.to_datetime(row['timestamp'])).total_seconds() / 3600))
            else:
                hours_ago = 1
            
            # Get sentiment from title and summary
            text = f"{row.get('title', '')} {row.get('summary', '')}"
            sentiment_result = analyze_text_sentiment(text)
            
            sentiment_score = sentiment_result.get('score', 0)
            sentiment_label = sentiment_result.get('label', 'Neutral')
            
            # Determine impact based on source
            source = row.get('source', 'Unknown')
            if source.lower() in ['coindesk', 'cointelegraph', 'bloomberg', 'reuters']:
                impact = 'High'
            elif source.lower() in ['bitcoin magazine', 'decrypt']:
                impact = 'Medium'
            else:
                impact = 'Low'
            
            news_items.append({
                "title": row.get('title', 'No title'),
                "source": source,
                "sentiment": sentiment_score,
                "sentiment_label": sentiment_label,
                "category": "Market",  # Could be enhanced with NLP
                "impact": impact,
                "published": row.get('timestamp', now).strftime("%Y-%m-%d %H:%M") if hasattr(row.get('timestamp', now), 'strftime') else str(row.get('timestamp', '')),
                "hours_ago": hours_ago,
                "url": row.get('url', '')
            })
        
        return sorted(news_items, key=lambda x: x["hours_ago"])
        
    except Exception as e:
        print(f"[!] Error fetching aggregated news: {type(e).__name__}: {e}")
        return []


def get_market_summary() -> Dict:
    """Get overall market sentiment summary from real news."""
    news = fetch_aggregated_news(limit=20)
    
    if not news:
        return {"sentiment": 0, "label": "Neutral", "confidence": 0}
    
    avg_sentiment = sum(n["sentiment"] for n in news) / len(news)
    
    bullish_count = sum(1 for n in news if n["sentiment"] > 0.1)
    bearish_count = sum(1 for n in news if n["sentiment"] < -0.1)
    
    return {
        "sentiment": avg_sentiment,
        "label": "Bullish" if avg_sentiment > 0.1 else ("Bearish" if avg_sentiment < -0.1 else "Neutral"),
        "bullish_news": bullish_count,
        "bearish_news": bearish_count,
        "neutral_news": len(news) - bullish_count - bearish_count,
        "confidence": min(abs(avg_sentiment) * 2, 1.0),
        "source_count": len(set(n["source"] for n in news))
    }


def get_news_df() -> pd.DataFrame:
    """Get news as DataFrame for display."""
    news = fetch_aggregated_news(limit=20)
    if news:
        return pd.DataFrame(news)
    return pd.DataFrame()


if __name__ == "__main__":
    print("Testing News Aggregator Module with Real APIs...")
    print("=" * 60)
    
    print("\n📅 Upcoming Events:")
    events = get_upcoming_events(days_ahead=30)
    if events:
        for event in events[:5]:
            print(f"  [{event['impact']}] {event['date_formatted']}: {event['event']}")
            print(f"       {event['description']} (in {event['days_until']} days)")
    else:
        print("  No upcoming events in the next 30 days")
    
    print("\n📰 Latest Real News:")
    news = fetch_aggregated_news(limit=5)
    if news:
        for n in news:
            sentiment_icon = "🟢" if n["sentiment"] > 0.1 else ("🔴" if n["sentiment"] < -0.1 else "🟡")
            print(f"  {sentiment_icon} [{n['source']}] {n['title'][:60]}...")
            print(f"       Sentiment: {n['sentiment']:.2f} | {n['hours_ago']}h ago")
    else:
        print("  ❌ No news fetched - check internet/VPN")
    
    print("\n📊 Market Summary:")
    summary = get_market_summary()
    if summary.get('source_count', 0) > 0:
        print(f"  ✅ Overall: {summary['label']} ({summary['sentiment']:.3f})")
        print(f"  Bullish: {summary['bullish_news']} | Bearish: {summary['bearish_news']} | Neutral: {summary['neutral_news']}")
        print(f"  Sources: {summary['source_count']}")
    else:
        print("  ❌ No market summary available")
