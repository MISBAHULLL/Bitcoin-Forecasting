"""
News Aggregator Module - Crypto News & Events
SINTA 1 Bitcoin Forecasting System
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict
import random


# ============================================
# MAJOR CRYPTO EVENTS CALENDAR
# ============================================

CRYPTO_EVENTS = [
    # FOMC Meetings 2024-2025
    {"date": "2024-12-18", "event": "FOMC Meeting", "impact": "High", "description": "Fed interest rate decision"},
    {"date": "2025-01-29", "event": "FOMC Meeting", "impact": "High", "description": "Fed interest rate decision"},
    {"date": "2025-03-19", "event": "FOMC Meeting", "impact": "High", "description": "Fed interest rate decision + SEP"},
    {"date": "2025-05-07", "event": "FOMC Meeting", "impact": "High", "description": "Fed interest rate decision"},
    
    # Bitcoin Events
    {"date": "2024-04-20", "event": "Bitcoin Halving", "impact": "Critical", "description": "Block reward halved to 3.125 BTC"},
    {"date": "2025-01-20", "event": "SEC ETF Review", "impact": "High", "description": "Potential spot ETF approval deadline"},
    
    # Economic Reports
    {"date": "2024-12-06", "event": "US Jobs Report", "impact": "Medium", "description": "Non-farm payrolls data"},
    {"date": "2024-12-11", "event": "CPI Data", "impact": "High", "description": "Consumer Price Index release"},
    {"date": "2024-12-12", "event": "PPI Data", "impact": "Medium", "description": "Producer Price Index release"},
]


def get_upcoming_events(days_ahead: int = 7) -> List[Dict]:
    """Get upcoming major events within the next N days."""
    today = datetime.now().date()
    upcoming = []
    
    for event in CRYPTO_EVENTS:
        event_date = datetime.strptime(event["date"], "%Y-%m-%d").date()
        days_until = (event_date - today).days
        
        if 0 <= days_until <= days_ahead:
            upcoming.append({
                **event,
                "days_until": days_until,
                "date_formatted": event_date.strftime("%b %d, %Y")
            })
    
    return sorted(upcoming, key=lambda x: x["days_until"])


def get_past_events(days_back: int = 7) -> List[Dict]:
    """Get recent major events from the past N days."""
    today = datetime.now().date()
    past = []
    
    for event in CRYPTO_EVENTS:
        event_date = datetime.strptime(event["date"], "%Y-%m-%d").date()
        days_ago = (today - event_date).days
        
        if 0 <= days_ago <= days_back:
            past.append({
                **event,
                "days_ago": days_ago,
                "date_formatted": event_date.strftime("%b %d, %Y")
            })
    
    return sorted(past, key=lambda x: x["days_ago"])


# ============================================
# SIMULATED NEWS FEED 
# ============================================

SAMPLE_NEWS = [
    {
        "title": "Bitcoin Surges Past $97K as Institutional Demand Grows",
        "source": "CoinDesk",
        "sentiment": 0.7,
        "category": "Market",
        "impact": "Medium"
    },
    {
        "title": "Federal Reserve Signals Cautious Approach to Rate Cuts",
        "source": "Reuters",
        "sentiment": -0.2,
        "category": "Macro",
        "impact": "High"
    },
    {
        "title": "Ethereum Layer 2 Solutions See Record Transaction Volume",
        "source": "CoinTelegraph",
        "sentiment": 0.5,
        "category": "Technology",
        "impact": "Low"
    },
    {
        "title": "Whale Alert: Large BTC Transfer Detected from Exchange",
        "source": "WhaleAlert",
        "sentiment": -0.3,
        "category": "On-chain",
        "impact": "Medium"
    },
    {
        "title": "MicroStrategy Adds More Bitcoin to Treasury Holdings",
        "source": "Bloomberg",
        "sentiment": 0.6,
        "category": "Institutional",
        "impact": "Medium"
    },
    {
        "title": "Regulatory Clarity Expected in Q1 2025 for Crypto Sector",
        "source": "CoinDesk",
        "sentiment": 0.4,
        "category": "Regulation",
        "impact": "High"
    },
    {
        "title": "Bitcoin Mining Difficulty Reaches All-Time High",
        "source": "CoinTelegraph",
        "sentiment": 0.2,
        "category": "Mining",
        "impact": "Low"
    },
    {
        "title": "Market Analysts Predict Consolidation Phase Before Next Rally",
        "source": "TradingView",
        "sentiment": 0.0,
        "category": "Analysis",
        "impact": "Medium"
    },
]


def fetch_aggregated_news(limit: int = 10) -> List[Dict]:
    """
    Fetch aggregated news from multiple sources.
    Returns simulated data - ready for real API integration.
    """
    now = datetime.now()
    news_items = []
    
    for i, item in enumerate(SAMPLE_NEWS[:limit]):
        hours_ago = random.randint(1, 24)
        news_items.append({
            "title": item["title"],
            "source": item["source"],
            "sentiment": item["sentiment"],
            "sentiment_label": "Bullish" if item["sentiment"] > 0.1 else ("Bearish" if item["sentiment"] < -0.1 else "Neutral"),
            "category": item["category"],
            "impact": item["impact"],
            "published": (now - timedelta(hours=hours_ago)).strftime("%Y-%m-%d %H:%M"),
            "hours_ago": hours_ago
        })
    
    return sorted(news_items, key=lambda x: x["hours_ago"])


def get_market_summary() -> Dict:
    """Get overall market sentiment summary from news."""
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
        "confidence": min(abs(avg_sentiment) * 2, 1.0)
    }


def get_news_df() -> pd.DataFrame:
    """Get news as DataFrame for display."""
    news = fetch_aggregated_news(limit=10)
    return pd.DataFrame(news)


if __name__ == "__main__":
    print("Testing News Aggregator Module...")
    print("=" * 60)
    
    print("\n📅 Upcoming Events:")
    for event in get_upcoming_events(days_ahead=30):
        print(f"  [{event['impact']}] {event['date_formatted']}: {event['event']}")
        print(f"       {event['description']} (in {event['days_until']} days)")
    
    print("\n📰 Latest News:")
    for news in fetch_aggregated_news(limit=5):
        sentiment_icon = "🟢" if news["sentiment"] > 0.1 else ("🔴" if news["sentiment"] < -0.1 else "🟡")
        print(f"  {sentiment_icon} [{news['source']}] {news['title']}")
        print(f"       Sentiment: {news['sentiment']:.2f} | {news['hours_ago']}h ago")
    
    print("\n📊 Market Summary:")
    summary = get_market_summary()
    print(f"  Overall: {summary['label']} ({summary['sentiment']:.3f})")
    print(f"  Bullish: {summary['bullish_news']} | Bearish: {summary['bearish_news']} | Neutral: {summary['neutral_news']}")
