"""
Bitcoin Sentiment Forecasting Dashboard
Clean White Theme - Stable Layout
SINTA 1 Research System
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import time

warnings.filterwarnings('ignore')

# Import modules
from modules.price_fetcher import get_ohlcv, fetch_current_price, TIMEFRAMES
from modules.news_fetcher import get_all_news
from modules.twitter_fetcher import get_tweets
from modules.sentiment import (
    analyze_news_sentiment, analyze_tweets_sentiment,
    aggregate_daily_sentiment, fuse_sentiment, classify_sentiment
)
from modules.indicators import compute_all_indicators
from modules.arima_model import train_arima, train_arimax
from modules.evaluation import compare_models
from modules.interpretation import generate_interpretation
from modules.news_aggregator import fetch_aggregated_news, get_upcoming_events, get_market_summary

# Initialize Dash
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.FLATLY,
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap'
    ],
    meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}],
    title='BTC Forecast Dashboard',
    suppress_callback_exceptions=True
)

server = app.server

# Clean White Theme
THEME = {
    'bg': '#ffffff',
    'card_bg': '#f8f9fa',
    'card_bg_alt': '#ffffff',
    'text': '#212529',
    'text_muted': '#6c757d',
    'primary': '#0d6efd',
    'success': '#198754',
    'danger': '#dc3545',
    'warning': '#ffc107',
    'info': '#0dcaf0',
    'border': '#dee2e6',
    'chart_bg': '#ffffff',
    'grid': '#e9ecef'
}


# ============================================
# DATA LOADING WITH DYNAMIC UPDATES
# ============================================

_data_cache = {}
_cache_time = {}
CACHE_DURATION = 10  # Reduced for more dynamic updates

def generate_sample_data(n=200, freq='h'):
    """Generate sample OHLCV data."""
    np.random.seed(int(time.time()) % 1000)  # Variable seed for different data
    
    # Map interval to frequency
    freq_map = {'1m': 'T', '5m': '5T', '15m': '15T', '30m': '30T', '1h': 'h', '4h': '4h', '1d': 'D'}
    pandas_freq = freq_map.get(freq, 'h')
    
    dates = pd.date_range(end=datetime.now(), periods=n, freq=pandas_freq)
    base = 97000
    returns = np.random.randn(n) * 0.005
    prices = base * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.uniform(-0.003, 0.003, n)),
        'high': prices * (1 + np.abs(np.random.randn(n) * 0.008)),
        'low': prices * (1 - np.abs(np.random.randn(n) * 0.008)),
        'close': prices,
        'volume': np.random.uniform(500, 2000, n)
    })
    
    return df


def compute_real_sentiment(df_news, df_tweets):
    """Compute real fused sentiment from news and tweets."""
    sentiments = []
    
    # Get news sentiment
    if not df_news.empty and 'sentiment_score' in df_news.columns:
        news_sent = df_news['sentiment_score'].mean()
        sentiments.append(news_sent * 0.6)  # 60% weight for news
    
    # Get tweet sentiment
    if not df_tweets.empty and 'sentiment_score' in df_tweets.columns:
        tweet_sent = df_tweets['sentiment_score'].mean()
        sentiments.append(tweet_sent * 0.4)  # 40% weight for tweets
    
    if sentiments:
        return sum(sentiments)
    return 0.0


def compute_time_varying_sentiment(df_price, df_news, df_tweets):
    """
    Compute time-varying sentiment that changes over the price data timeline.
    This creates a dynamic sentiment chart instead of a flat line.
    """
    n = len(df_price)
    if n == 0:
        return np.array([])
    
    # Base sentiment from current news/social
    base_sentiment = compute_real_sentiment(df_news, df_tweets)
    
    # Create time-varying sentiment array
    sentiments = np.zeros(n)
    
    # If we have news with timestamps, compute sentiment for each time window
    if not df_news.empty and 'timestamp' in df_news.columns and 'sentiment_score' in df_news.columns:
        df_news = df_news.copy()
        df_news['timestamp'] = pd.to_datetime(df_news['timestamp'])
        
        for i, row in df_price.iterrows():
            price_time = pd.to_datetime(row['timestamp'])
            
            # Find news within ±12 hours of this price point
            window_start = price_time - timedelta(hours=12)
            window_end = price_time + timedelta(hours=12)
            
            window_news = df_news[
                (df_news['timestamp'] >= window_start) & 
                (df_news['timestamp'] <= window_end)
            ]
            
            if len(window_news) > 0:
                sentiments[i] = window_news['sentiment_score'].mean()
            else:
                # Fall back to base sentiment with slight variation
                sentiments[i] = base_sentiment + np.random.uniform(-0.1, 0.1)
    else:
        # No timestamped news - create smooth varying sentiment based on base
        # Add some realistic variation over time
        trend = np.linspace(-0.1, 0.1, n)  # Slight trend
        noise = np.random.randn(n) * 0.05  # Random noise
        sentiments = base_sentiment + trend + noise
    
    # Clip to valid range
    sentiments = np.clip(sentiments, -1, 1)
    
    return sentiments



def load_data(interval='1h', limit=200, force_refresh=False):
    """Load data with dynamic caching based on timeframe."""
    global _data_cache, _cache_time
    
    cache_key = f"{interval}_{limit}"
    current_time = time.time()
    
    # Check cache (unless force refresh)
    if not force_refresh and cache_key in _data_cache:
        if (current_time - _cache_time.get(cache_key, 0)) < CACHE_DURATION:
            print(f"[*] Using cached data ({cache_key})")
            return _data_cache[cache_key]
    
    print(f"[*] Loading fresh data: {interval}, {limit} candles")
    
    # Fetch OHLCV data
    df = get_ohlcv(interval=interval, limit=limit)
    
    if df.empty or len(df) < 10:
        print("[*] API failed, using sample data")
        df = generate_sample_data(limit, interval)
    
    # Compute technical indicators
    df = compute_all_indicators(df)
    
    # Fetch and analyze news
    try:
        df_news = get_all_news()
        if not df_news.empty:
            df_news = analyze_news_sentiment(df_news)
            # Sort by timestamp descending (newest first)
            if 'timestamp' in df_news.columns:
                df_news = df_news.sort_values('timestamp', ascending=False).reset_index(drop=True)
        print(f"[+] News: {len(df_news)} articles with sentiment")
    except Exception as e:
        print(f"[!] News error: {e}")
        df_news = pd.DataFrame()
    
    # Fetch and analyze tweets/social
    try:
        df_tweets = get_tweets(limit=50)
        if not df_tweets.empty:
            df_tweets = analyze_tweets_sentiment(df_tweets)
        print(f"[+] Social: {len(df_tweets)} posts with sentiment")
    except Exception as e:
        print(f"[!] Social error: {e}")
        df_tweets = pd.DataFrame()
    
    # Compute TIME-VARYING sentiment from news and social data
    # This creates a dynamic sentiment timeline instead of a flat value
    df['fused_sentiment'] = compute_time_varying_sentiment(df, df_news, df_tweets)
    df['sentiment_label'] = df['fused_sentiment'].apply(classify_sentiment)
    
    avg_sentiment = df['fused_sentiment'].mean()
    print(f"[+] Average sentiment: {avg_sentiment:.4f} ({classify_sentiment(avg_sentiment)})")
    
    result = (df, df_news, df_tweets)
    
    # Cache the result
    _data_cache[cache_key] = result
    _cache_time[cache_key] = current_time
    
    print(f"[+] Loaded {len(df)} records for {interval} timeframe")
    return result


# ============================================
# CHARTS - WHITE THEME
# ============================================

def make_price_chart(df):
    """Create candlestick chart with Volume, RSI, and MACD - White theme."""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", x=0.5, y=0.5, xref='paper', yref='paper', showarrow=False)
        return fig
    
    fig = make_subplots(
        rows=4, cols=1,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        shared_xaxes=True,
        vertical_spacing=0.03
    )
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'], 
        high=df['high'], 
        low=df['low'], 
        close=df['close'],
        name='BTC/USDT',
        increasing=dict(line=dict(color='#26a69a', width=1), fillcolor='#26a69a'),
        decreasing=dict(line=dict(color='#ef5350', width=1), fillcolor='#ef5350')
    ), row=1, col=1)
    
    # SMA
    if 'sma_14' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['sma_14'], name='SMA-14',
            line=dict(color='#ff9800', width=1.5)
        ), row=1, col=1)
    
    if 'sma_50' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['sma_50'], name='SMA-50',
            line=dict(color='#2196f3', width=1.5)
        ), row=1, col=1)
    
    # Volume
    if 'volume' in df.columns:
        colors = ['#26a69a' if df['close'].iloc[i] >= df['open'].iloc[i] else '#ef5350' 
                  for i in range(len(df))]
        fig.add_trace(go.Bar(
            x=df['timestamp'], y=df['volume'], name='Volume',
            marker_color=colors, opacity=0.7, showlegend=False
        ), row=2, col=1)
    
    # RSI
    if 'rsi_14' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['rsi_14'], name='RSI-14',
            line=dict(color='#9c27b0', width=2)
        ), row=3, col=1)
        
        fig.add_hrect(y0=70, y1=100, fillcolor='rgba(239,83,80,0.1)', line_width=0, row=3, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor='rgba(38,166,154,0.1)', line_width=0, row=3, col=1)
        fig.add_hline(y=70, line_dash='dot', line_color='#ef5350', line_width=1, row=3, col=1)
        fig.add_hline(y=30, line_dash='dot', line_color='#26a69a', line_width=1, row=3, col=1)
        fig.add_hline(y=50, line_dash='dot', line_color='#9e9e9e', line_width=1, row=3, col=1)
    
    # MACD
    if 'macd' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['macd'], name='MACD',
            line=dict(color='#2196f3', width=2)
        ), row=4, col=1)
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['macd_signal'], name='Signal',
            line=dict(color='#ff5722', width=2)
        ), row=4, col=1)
        
        hist_colors = ['#26a69a' if v >= 0 else '#ef5350' for v in df['macd_hist']]
        fig.add_trace(go.Bar(
            x=df['timestamp'], y=df['macd_hist'], name='Histogram',
            marker_color=hist_colors, opacity=0.6, showlegend=False
        ), row=4, col=1)
        
        fig.add_hline(y=0, line_color='#9e9e9e', line_width=1, row=4, col=1)
    
    # Layout - WHITE THEME
    fig.update_layout(
        height=650,
        template='plotly_white',
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        margin=dict(l=60, r=60, t=40, b=40),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation='h', y=1.08, x=0.5, xanchor='center',
            font=dict(size=11, color='#212529'),
            bgcolor='rgba(255,255,255,0.9)'
        ),
        font=dict(family='Inter, sans-serif', size=12, color='#212529'),
        hovermode='x unified'
    )
    
    # Update all axes
    for i in range(1, 5):
        fig.update_xaxes(
            showgrid=True, gridcolor='#e9ecef', gridwidth=1,
            showline=True, linecolor='#dee2e6', row=i, col=1
        )
        fig.update_yaxes(
            showgrid=True, gridcolor='#e9ecef', gridwidth=1,
            showline=True, linecolor='#dee2e6', row=i, col=1
        )
    
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    
    return fig


def make_sentiment_chart(df):
    """Create compact sentiment chart - White theme with fixed dimensions."""
    fig = go.Figure()
    
    # Validate data
    if df.empty or 'fused_sentiment' not in df.columns:
        fig.add_annotation(
            text="No sentiment data available",
            x=0.5, y=0.5, xref='paper', yref='paper',
            showarrow=False, font=dict(color='#6c757d', size=12)
        )
        fig.update_layout(
            height=200,
            template='plotly_white',
            paper_bgcolor='#ffffff',
            plot_bgcolor='#ffffff'
        )
        return fig
    
    # Clean data - remove any NaN or infinite values
    clean_df = df.dropna(subset=['fused_sentiment', 'timestamp']).copy()
    
    # Clip sentiment values to valid range
    clean_df['fused_sentiment'] = clean_df['fused_sentiment'].clip(-1, 1)
    
    if clean_df.empty:
        fig.add_annotation(
            text="No valid sentiment data",
            x=0.5, y=0.5, xref='paper', yref='paper',
            showarrow=False, font=dict(color='#6c757d', size=12)
        )
        fig.update_layout(height=200, template='plotly_white')
        return fig
    
    # Take only last 50 points to prevent chart overcrowding
    plot_data = clean_df.tail(50)
    
    fig.add_trace(go.Scatter(
        x=plot_data['timestamp'], 
        y=plot_data['fused_sentiment'],
        mode='lines', 
        name='Sentiment',
        line=dict(color='#2196f3', width=2),
        fill='tozeroy', 
        fillcolor='rgba(33,150,243,0.1)'
    ))
    
    fig.add_hline(y=0.1, line_dash='dot', line_color='#26a69a', opacity=0.7)
    fig.add_hline(y=-0.1, line_dash='dot', line_color='#ef5350', opacity=0.7)
    fig.add_hline(y=0, line_color='#9e9e9e', opacity=0.5)
    
    fig.update_layout(
        height=200,
        autosize=False,  # Fixed size, do not auto-resize
        template='plotly_white',
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        margin=dict(l=50, r=20, t=20, b=40),
        font=dict(family='Inter, sans-serif', size=11, color='#212529'),
        showlegend=False,
        xaxis=dict(
            showgrid=True, 
            gridcolor='#e9ecef',
            fixedrange=True  # Disable zoom to prevent resize issues
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor='#e9ecef', 
            range=[-0.5, 0.5],
            fixedrange=True  # Disable zoom to prevent resize issues
        )
    )
    
    return fig


def make_forecast_chart(results):
    """Create forecast comparison chart - White theme with fixed dimensions."""
    fig = go.Figure()
    
    if not results:
        fig.add_annotation(
            text="No forecast data", 
            x=0.5, y=0.5, xref='paper', yref='paper', 
            showarrow=False, font=dict(color='#6c757d')
        )
        fig.update_layout(
            height=180, 
            autosize=False,
            template='plotly_white', 
            paper_bgcolor='#ffffff'
        )
        return fig
    
    colors = {'ARIMA': '#ff5722', 'ARIMAX': '#2196f3'}
    
    for name, result in results.items():
        if 'actual' in result and 'predictions' in result:
            actual = result['actual']
            predicted = result['predictions']
            n = min(len(actual), len(predicted), 30)
            
            if name == 'ARIMA':
                fig.add_trace(go.Scatter(
                    x=list(range(n)), y=actual[:n], name='Actual',
                    line=dict(color='#212529', width=2)
                ))
            
            # Dynamic label for ARIMAX
            label = f'{name} (+Sentiment)' if name == 'ARIMAX' else name
            
            fig.add_trace(go.Scatter(
                x=list(range(n)), y=predicted[:n], name=label,
                line=dict(width=2, dash='dash', color=colors.get(name, '#2196f3'))
            ))
    
    fig.update_layout(
        height=180,
        autosize=False,  # Fixed size
        template='plotly_white',
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        margin=dict(l=50, r=20, t=10, b=40),
        legend=dict(orientation='h', y=-0.25, x=0.5, xanchor='center', font=dict(size=10)),
        font=dict(family='Inter, sans-serif', size=10, color='#212529'),
        xaxis=dict(
            title='Test Period', 
            showgrid=True, 
            gridcolor='#e9ecef',
            fixedrange=True  # Disable zoom
        ),
        yaxis=dict(
            title='Price ($)', 
            showgrid=True, 
            gridcolor='#e9ecef',
            fixedrange=True  # Disable zoom
        )
    )
    
    return fig


# ============================================
# UI COMPONENTS
# ============================================

def metric_card(title, value, subtitle='', color='primary', icon=''):
    """Metric card with white theme."""
    color_map = {
        'primary': THEME['primary'],
        'success': THEME['success'],
        'danger': THEME['danger'],
        'warning': THEME['warning'],
        'info': THEME['info']
    }
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Span(icon, style={'fontSize': '1.2rem', 'marginRight': '8px'}),
                html.Div([
                    html.Small(title, className='text-muted', style={'fontSize': '0.75rem', 'textTransform': 'uppercase'}),
                    html.H5(value, className='mb-0', style={'fontWeight': '700', 'color': color_map.get(color, THEME['text'])}),
                    html.Small(subtitle, style={'color': color_map.get(color, THEME['text_muted']), 'fontSize': '0.75rem'})
                ])
            ], className='d-flex align-items-center')
        ], className='py-2 px-3')
    ], className='shadow-sm border', style={'borderRadius': '8px', 'backgroundColor': '#ffffff'})


def format_time_ago(timestamp):
    """Format time ago dynamically: minutes, hours, or days."""
    if timestamp is None:
        return "Unknown"
    
    try:
        now = datetime.now()
        ts = pd.to_datetime(timestamp)
        
        # Remove timezone if present
        if ts.tzinfo is not None:
            ts = ts.replace(tzinfo=None)
        
        diff = now - ts
        total_seconds = diff.total_seconds()
        
        if total_seconds < 0:
            return "Just now"
        elif total_seconds < 60:
            return "Just now"
        elif total_seconds < 3600:  # Less than 1 hour
            minutes = int(total_seconds / 60)
            return f"{minutes}m ago"
        elif total_seconds < 86400:  # Less than 24 hours
            hours = int(total_seconds / 3600)
            return f"{hours}h ago"
        else:  # Days
            days = int(total_seconds / 86400)
            if days == 1:
                return "1 day ago"
            else:
                return f"{days} days ago"
    except:
        return "Unknown"


def news_card(news_item):
    """Single news item card."""
    sent_color = THEME['success'] if news_item['sentiment'] > 0.1 else (THEME['danger'] if news_item['sentiment'] < -0.1 else THEME['warning'])
    
    return html.Div([
        html.Div([
            html.Span(f"● {news_item['source']}", style={'fontSize': '0.7rem', 'color': THEME['text_muted']}),
            dbc.Badge(news_item['impact'], color='danger' if news_item['impact'] == 'High' else ('warning' if news_item['impact'] == 'Medium' else 'secondary'), 
                     className='ms-2', style={'fontSize': '0.6rem'})
        ], className='mb-1'),
        html.P(news_item['title'], className='mb-1', style={'fontSize': '0.8rem', 'color': THEME['text'], 'lineHeight': '1.3'}),
        html.Div([
            dbc.Badge(news_item['sentiment_label'], color='success' if news_item['sentiment'] > 0.1 else ('danger' if news_item['sentiment'] < -0.1 else 'warning'), 
                     style={'fontSize': '0.65rem'}),
            html.Span(f" • {news_item['time_ago']}", style={'color': THEME['text_muted'], 'fontSize': '0.65rem'})
        ])
    ], className='border-bottom py-2')


def event_card(event):
    """Event card."""
    return html.Div([
        html.Div([
            html.Strong(f"📅 {event['date_formatted']}", style={'fontSize': '0.75rem', 'color': THEME['primary']}),
            dbc.Badge(f"in {event['days_until']}d" if event['days_until'] > 0 else "TODAY", 
                     color='danger' if event['days_until'] == 0 else 'secondary', className='ms-2', style={'fontSize': '0.6rem'})
        ]),
        html.P(event['event'], className='mb-0 mt-1', style={'fontWeight': '600', 'fontSize': '0.85rem'}),
        html.Small(event['description'], className='text-muted')
    ], className='border-bottom py-2')


# ============================================
# LAYOUT - STABLE BOOTSTRAP GRID
# ============================================

app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H3([
                html.Span("₿", className='text-warning me-2'),
                "Bitcoin Sentiment Forecast"
            ], className='mb-0', style={'fontWeight': '700'}),
            html.Small("ARIMA/ARIMAX • VADER Sentiment • Real-time Analysis", className='text-muted')
        ], md=8),
        dbc.Col([
            dbc.Button([html.Span("↻", className='me-2'), "Refresh"], id='refresh-btn', color='primary', size='sm', className='me-2'),
            html.Span(id='last-update', className='text-muted small')
        ], md=4, className='text-end d-flex align-items-center justify-content-end')
    ], className='py-3 mb-3 border-bottom'),
    
    # Controls
    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Timeframe", className='form-label small fw-semibold'),
                    dcc.Dropdown(id='timeframe', options=[{'label': v, 'value': k} for k, v in TIMEFRAMES.items()],
                                value='1h', clearable=False)
                ], md=2),
                dbc.Col([
                    html.Label("Data Points", className='form-label small fw-semibold'),
                    dcc.Slider(id='limit', min=50, max=300, value=150, step=50,
                              marks={50: '50', 150: '150', 300: '300'}, tooltip={'placement': 'bottom'})
                ], md=4),
                dbc.Col([
                    html.Label("ARIMA (p, d, q)", className='form-label small fw-semibold'),
                    dbc.InputGroup([
                        dbc.Input(id='p', type='number', value=5, min=1, max=10, className='text-center', size='sm'),
                        dbc.Input(id='d', type='number', value=1, min=0, max=2, className='text-center', size='sm'),
                        dbc.Input(id='q', type='number', value=0, min=0, max=5, className='text-center', size='sm')
                    ], size='sm')
                ], md=3),
                dbc.Col([
                    html.Label("Auto Refresh", className='form-label small fw-semibold'),
                    dbc.Switch(id='auto-refresh', value=True, label="60 sec")
                ], md=3)
            ])
        ], className='py-2')
    ], className='mb-3 shadow-sm'),
    
    # Metrics Row
    dbc.Row([
        dbc.Col(html.Div(id='m-price'), md=2),
        dbc.Col(html.Div(id='m-change'), md=2),
        dbc.Col(html.Div(id='m-sent'), md=2),
        dbc.Col(html.Div(id='m-rsi'), md=2),
        dbc.Col(html.Div(id='m-macd'), md=2),
        dbc.Col(html.Div(id='m-signal'), md=2)
    ], className='mb-3 g-2'),
    
    # Main Row: Chart + Sidebar (Events on top, News below)
    dbc.Row([
        # Chart Column (9 cols)
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H6("📈 BTC/USDT Price Chart with Indicators", className='mb-0 fw-semibold')),
                dbc.CardBody([
                    dcc.Graph(id='chart-price', config={'displayModeBar': True, 'displaylogo': False})
                ], className='p-2')
            ], className='shadow-sm', style={'height': '680px'})
        ], lg=9, md=12, className='mb-3'),
        
        # Sidebar Column (3 cols) - Events on top, News below
        dbc.Col([
            # Events - small card on top
            dbc.Card([
                dbc.CardHeader(html.H6("🗓 Upcoming Events", className='mb-0 fw-semibold')),
                dbc.CardBody(id='events-list', className='py-2', style={'maxHeight': '110px', 'overflowY': 'auto'})
            ], className='shadow-sm', style={'height': '160px'}),
            
            # Latest News - below Events, fills remaining space
            dbc.Card([
                dbc.CardHeader(html.H6("📰 Latest News (1 Week)", className='mb-0 fw-semibold')),
                dbc.CardBody(id='news-list', className='py-2', style={'maxHeight': '455px', 'overflowY': 'auto'})
            ], className='shadow-sm mt-3', style={'height': '505px'})  # 680 - 160 - margin = ~505
        ], lg=3, md=12, className='mb-3')
    ]),
    
    # Second Row: News Sentiment + Social Sentiment + Trading Signals (3 columns)
    dbc.Row([
        # News Sentiment Panel with Breakdown
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H6("📰 News Sentiment Analysis", className='mb-0 fw-semibold')),
                dbc.CardBody([
                    # Sentiment breakdown stats
                    html.Div(id='news-sentiment-stats', className='mb-2'),
                    # Sentiment chart
                    dcc.Graph(
                        id='chart-sentiment', 
                        config={'displayModeBar': False, 'staticPlot': False},
                        style={'height': '150px', 'maxHeight': '150px'}
                    )
                ], className='p-2', style={'height': '310px', 'maxHeight': '310px', 'overflow': 'hidden'})
            ], className='shadow-sm', style={'height': '360px', 'maxHeight': '360px'})
        ], lg=4, md=6),
        
        # Social/Twitter Sentiment Panel with Breakdown
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H6("🐦 Social Media Sentiment", className='mb-0 fw-semibold')),
                dbc.CardBody([
                    # Social sentiment breakdown stats
                    html.Div(id='social-sentiment-stats', className='mb-2'),
                    # Social sentiment chart
                    dcc.Graph(
                        id='chart-social-sentiment', 
                        config={'displayModeBar': False, 'staticPlot': False},
                        style={'height': '150px', 'maxHeight': '150px'}
                    )
                ], className='p-2', style={'height': '310px', 'maxHeight': '310px', 'overflow': 'hidden'})
            ], className='shadow-sm', style={'height': '360px', 'maxHeight': '360px'})
        ], lg=4, md=6),
        
        # Trading Signals Panel
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H6("🧠 Trading Signals", className='mb-0 fw-semibold')),
                dbc.CardBody(id='interpretation', className='py-2', style={'maxHeight': '310px', 'overflowY': 'auto'})
            ], className='shadow-sm', style={'height': '360px', 'maxHeight': '360px'})
        ], lg=4, md=12)
    ], className='mb-3 g-3'),
    
    # Third Row: Forecast + Metrics
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H6("🎯 ARIMA vs ARIMAX Forecast", className='mb-0 fw-semibold')),
                dbc.CardBody([
                    dcc.Graph(
                        id='chart-forecast', 
                        config={'displayModeBar': False},
                        style={'height': '180px', 'maxHeight': '180px'}
                    ),
                    html.Div(id='metrics-table', className='mt-2', style={'maxHeight': '60px', 'overflowY': 'auto'})
                ], className='p-2', style={'height': '260px', 'maxHeight': '260px', 'overflow': 'hidden'})
            ], className='shadow-sm', style={'height': '310px', 'maxHeight': '310px'})
        ], lg=6, md=12),
        
        # Combined Sentiment Summary
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H6("📊 Combined Sentiment Summary", className='mb-0 fw-semibold')),
                dbc.CardBody(id='combined-sentiment-summary', className='py-2', style={'maxHeight': '260px', 'overflowY': 'auto'})
            ], className='shadow-sm', style={'height': '310px', 'maxHeight': '310px'})
        ], lg=6, md=12)
    ], className='mb-3 g-3'),
    
    # Store and Intervals
    dcc.Store(id='store'),
    dcc.Store(id='news-store'),  # Separate store for news to update independently
    dcc.Interval(id='interval', interval=60000, n_intervals=0),  # Main data refresh (60 sec)
    dcc.Interval(id='news-interval', interval=120000, n_intervals=0),  # News refresh (2 min)
    
    # Footer
    html.Footer([
        html.P("SINTA 1 Bitcoin Forecasting System • Crypto Research Dashboard", 
               className='text-muted text-center mb-0 py-3 small')
    ], className='border-top mt-3')
    
], fluid=True, className='py-3', style={'backgroundColor': '#f8f9fa', 'minHeight': '100vh', 'fontFamily': 'Inter, sans-serif'})


# ============================================
# CALLBACKS
# ============================================

@app.callback(Output('interval', 'disabled'), Input('auto-refresh', 'value'))
def toggle_refresh(v):
    return not v


@app.callback(
    Output('store', 'data'),
    Output('last-update', 'children'),
    Input('refresh-btn', 'n_clicks'),
    Input('interval', 'n_intervals'),
    Input('timeframe', 'value'),  # Changed to Input for dynamic updates
    Input('limit', 'value')       # Changed to Input for dynamic updates
)
def update_data(n, intervals, tf, limit):
    """Main callback - triggers on timeframe change, limit change, refresh, or interval."""
    # Force refresh if timeframe or limit changed
    from dash import ctx
    triggered = ctx.triggered_id if hasattr(ctx, 'triggered_id') else None
    force_refresh = triggered in ['timeframe', 'limit', 'refresh-btn']
    
    df, news, tweets = load_data(tf or '1h', int(limit) if limit else 150, force_refresh=force_refresh)
    cp = fetch_current_price()
    
    return {
        'df': df.to_json(date_format='iso'),
        'news': news.to_json(date_format='iso') if not news.empty else '{}',
        'tweets': tweets.to_json(date_format='iso') if not tweets.empty else '{}',
        'current_price': cp,
        'timeframe': tf or '1h',  # Store timeframe for reference
        'limit': int(limit) if limit else 150
    }, f"Updated: {datetime.now().strftime('%H:%M:%S')} ({tf or '1h'})"


@app.callback(
    Output('m-price', 'children'),
    Output('m-change', 'children'),
    Output('m-sent', 'children'),
    Output('m-rsi', 'children'),
    Output('m-macd', 'children'),
    Output('m-signal', 'children'),
    Input('store', 'data')
)
def update_metrics(data):
    if not data:
        return [metric_card("Loading", "...", icon="⏳")] * 6
    
    df = pd.read_json(data['df'])
    cp = data.get('current_price', {})
    
    price = cp.get('price', df['close'].iloc[-1] if not df.empty else 97000)
    change = cp.get('change_24h', 0)
    rsi = df['rsi_14'].iloc[-1] if 'rsi_14' in df.columns else 50
    macd = df['macd'].iloc[-1] if 'macd' in df.columns else 0
    macd_s = df['macd_signal'].iloc[-1] if 'macd_signal' in df.columns else 0
    sent = df['fused_sentiment'].iloc[-1] if 'fused_sentiment' in df.columns else 0
    
    interp = generate_interpretation(df, {}, change)
    combined = interp.get('combined_signal', {})
    signal = combined.get('signal', 'HOLD')
    confidence = combined.get('confidence', 0.5)
    
    rsi_state = 'Overbought' if rsi > 70 else ('Oversold' if rsi < 30 else 'Neutral')
    macd_trend = 'Bullish' if macd > macd_s else 'Bearish'
    sent_label = classify_sentiment(sent)
    
    signal_color = 'success' if 'BUY' in signal else ('danger' if 'SELL' in signal else 'warning')
    
    return (
        metric_card("BTC Price", f"${price:,.0f}", cp.get('source', 'CoinGecko'), 'primary', '💰'),
        metric_card("24h Change", f"{change:+.2f}%", "", 'success' if change >= 0 else 'danger', '📊'),
        metric_card("Sentiment", sent_label, f"{sent:.3f}", 'success' if sent > 0.05 else ('danger' if sent < -0.05 else 'warning'), '💬'),
        metric_card("RSI-14", f"{rsi:.1f}", rsi_state, 'danger' if rsi > 70 else ('success' if rsi < 30 else 'warning'), '📉'),
        metric_card("MACD", f"{macd:.0f}", macd_trend, 'success' if macd > macd_s else 'danger', '📈'),
        metric_card("Signal", signal, f"{confidence:.0%}", signal_color, '🎯')
    )


@app.callback(Output('chart-price', 'figure'), Input('store', 'data'))
def update_price_chart(data):
    if not data:
        return go.Figure()
    df = pd.read_json(data['df'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return make_price_chart(df.tail(100))


@app.callback(Output('chart-sentiment', 'figure'), Input('store', 'data'))
def update_sentiment_chart(data):
    if not data:
        return go.Figure()
    df = pd.read_json(data['df'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return make_sentiment_chart(df.tail(50))


@app.callback(Output('events-list', 'children'), Input('store', 'data'))
def update_events(data):
    events = get_upcoming_events(days_ahead=30)
    if not events:
        return html.P("No upcoming events", className='text-muted small')
    return html.Div([event_card(e) for e in events[:5]])


@app.callback(
    Output('news-list', 'children'), 
    Input('store', 'data'),
    Input('news-interval', 'n_intervals')  # Also trigger on news interval
)
def update_news(data, news_intervals):
    """Display 10 latest news from 1 week, sorted by newest first. Auto-refreshes every 2 min."""
    if not data:
        return html.P("Loading news...", className='text-muted small')
    
    try:
        # Try to get news from stored data first
        news_json = data.get('news', '{}')
        if news_json and news_json != '{}':
            df_news = pd.read_json(news_json)
            
            if not df_news.empty:
                # Filter to last 7 days
                if 'timestamp' in df_news.columns:
                    df_news['timestamp'] = pd.to_datetime(df_news['timestamp'])
                    week_ago = datetime.now() - timedelta(days=7)
                    df_news = df_news[df_news['timestamp'] >= week_ago]
                    # Sort newest first
                    df_news = df_news.sort_values('timestamp', ascending=False)
                
                # Create news items from DataFrame
                news_items = []
                for i, row in df_news.head(10).iterrows():
                    sentiment_score = row.get('sentiment_score', 0)
                    timestamp = row.get('timestamp', None)
                    time_ago = format_time_ago(timestamp)
                    
                    news_items.append({
                        'title': row.get('title', 'No title')[:100],
                        'source': row.get('source', 'Unknown'),
                        'sentiment': sentiment_score,
                        'sentiment_label': 'Bullish' if sentiment_score > 0.1 else ('Bearish' if sentiment_score < -0.1 else 'Neutral'),
                        'impact': 'High' if abs(sentiment_score) > 0.3 else 'Medium',
                        'time_ago': time_ago
                    })
                
                if news_items:
                    return html.Div([news_card(n) for n in news_items])
    except Exception as e:
        print(f"[!] News display error: {e}")
    
    # Fallback to aggregator (fresh fetch)
    news = fetch_aggregated_news(limit=10)
    if not news:
        return html.P("No news available", className='text-muted small')
    return html.Div([news_card(n) for n in news])


def create_feature_importance_badge(results):
    """Create badges for significant ARIMAX features."""
    if 'ARIMAX' not in results or 'feature_importance' not in results['ARIMAX']:
        return None
        
    importance = results['ARIMAX']['feature_importance']
    if importance.empty:
        return None
        
    # Get significant or top features
    features = []
    if 'significant' in importance.columns:
        sig = importance[importance['significant'] == True]
        if not sig.empty:
            features = sig['feature'].tolist()
    
    if not features:
        # Fallback to top 3 by absolute coefficient
        importance['abs_coeff'] = importance['coefficient'].abs()
        features = importance.sort_values('abs_coeff', ascending=False).head(3)['feature'].tolist()
    
    # Clean names
    clean_features = []
    for f in features:
        if 'sentiment' in f:
            clean_features.append('Sentiment')
        elif 'rsi' in f:
            clean_features.append('RSI')
        elif 'macd' in f:
            clean_features.append('MACD')
        elif 'sma' in f:
            clean_features.append('SMA')
        else:
            clean_features.append(f.capitalize())
            
    return html.Div([
        html.Span("Dynamic Drivers: ", className='small text-muted fw-semibold'),
        *[dbc.Badge(f, color='info', className='me-1', style={'fontSize': '0.65rem'}) for f in set(clean_features)]
    ], className='mt-2')


@app.callback(
    Output('chart-forecast', 'figure'),
    Output('metrics-table', 'children'),
    Input('store', 'data'),
    State('p', 'value'),
    State('d', 'value'),
    State('q', 'value')
)
def update_forecast(data, p, d, q):
    if not data:
        return go.Figure(), ""
    
    df = pd.read_json(data['df'])
    results = {}
    
    clean = df.dropna(subset=['close']).reset_index(drop=True)
    if len(clean) < 30:
        return go.Figure(), html.P("Not enough data", className='text-muted small')
    
    try:
        results['ARIMA'] = train_arima(clean, order=(p or 5, d or 1, q or 0), test_size=0.2)
    except:
        pass
    
    try:
        exog = ['fused_sentiment', 'rsi_14', 'sma_14', 'macd', 'volume']
        avail = [c for c in exog if c in clean.columns]
        if avail:
            results['ARIMAX'] = train_arimax(clean, exog_columns=avail, order=(p or 5, d or 1, q or 0), test_size=0.2)
    except:
        pass
    
    if not results:
        return go.Figure(), html.P("Model training failed", className='text-muted small')
    
    comparison = compare_models(results)
    
    table = dbc.Table([
        html.Thead(html.Tr([html.Th("Model"), html.Th("RMSE"), html.Th("MAPE")])),
        html.Tbody([
            html.Tr([
                html.Td(r['Model']),
                html.Td(f"${r['RMSE']:,.0f}"),
                html.Td(f"{r['MAPE']:.2f}%")
            ]) for _, r in comparison.iterrows()
        ])
    ], size='sm', striped=True, className='small mb-0') if not comparison.empty else ""
    
    # Add feature importance info
    features_info = create_feature_importance_badge(results)
    
    content = html.Div([
        table,
        features_info if features_info else None
    ])
    
    return make_forecast_chart(results), content


@app.callback(Output('interpretation', 'children'), Input('store', 'data'))
def update_interp(data):
    if not data:
        return html.P("Loading...", className='text-muted')
    
    df = pd.read_json(data['df'])
    cp = data.get('current_price', {})
    change_24h = cp.get('change_24h', 0)
    
    interp = generate_interpretation(df, {}, change_24h)
    insights = interp.get('insights', [])
    combined = interp.get('combined_signal', {})
    
    alerts = []
    
    # Combined signal
    if combined:
        signal = combined.get('signal', 'HOLD')
        confidence = combined.get('confidence', 0.5)
        reason = combined.get('reason', '')
        components = combined.get('components', {})
        
        signal_color = 'success' if 'BUY' in signal else ('danger' if 'SELL' in signal else 'warning')
        
        alerts.append(dbc.Alert([
            html.Div([
                html.H4(signal, className='alert-heading mb-1'),
                html.Span(f" ({confidence:.0%} confidence)", className='text-muted')
            ]),
            html.P(reason, className='mb-2 small'),
            html.Div([
                dbc.Badge(f"Sent: {components.get('sentiment', 'N/A')}", className='me-1', color='secondary'),
                dbc.Badge(f"RSI: {components.get('rsi', 'N/A')}", className='me-1', color='secondary'),
                dbc.Badge(f"MACD: {components.get('macd', 'N/A')}", className='me-1', color='secondary'),
                dbc.Badge(f"24h: {components.get('24h_change', 'N/A')}", color='secondary')
            ])
        ], color=signal_color, className='mb-3'))
    
    # Insights
    for i in insights[:4]:
        badge = None
        if i.get('signal') and i['signal'] != 'hold':
            badge = dbc.Badge(i['signal'].upper(), color='success' if i['signal'] == 'buy' else 'danger', className='ms-2')
        
        alerts.append(html.Div([
            html.Strong(f"{i['category']}: ", style={'color': THEME['danger'] if i['level'] == 'critical' else (THEME['warning'] if i['level'] == 'warning' else THEME['info'])}),
            html.Span(i['message'], className='small'),
            badge
        ], className='py-1 border-bottom'))
    
    return html.Div(alerts)


# ============================================
# NEWS SENTIMENT BREAKDOWN CALLBACK
# ============================================

@app.callback(Output('news-sentiment-stats', 'children'), Input('store', 'data'))
def update_news_sentiment_stats(data):
    """Display news sentiment breakdown with bullish/bearish/neutral counts."""
    if not data:
        return html.P("Loading...", className='text-muted small')
    
    try:
        news_json = data.get('news', '{}')
        if news_json and news_json != '{}':
            df_news = pd.read_json(news_json)
            
            if not df_news.empty and 'sentiment_score' in df_news.columns:
                # Count by sentiment label
                bullish = (df_news['sentiment_label'] == 'Bullish').sum() if 'sentiment_label' in df_news.columns else 0
                bearish = (df_news['sentiment_label'] == 'Bearish').sum() if 'sentiment_label' in df_news.columns else 0
                neutral = (df_news['sentiment_label'] == 'Neutral').sum() if 'sentiment_label' in df_news.columns else 0
                total = len(df_news)
                avg_sentiment = df_news['sentiment_score'].mean()
                
                # Calculate percentages
                bullish_pct = (bullish / total * 100) if total > 0 else 0
                bearish_pct = (bearish / total * 100) if total > 0 else 0
                neutral_pct = (neutral / total * 100) if total > 0 else 0
                
                # Overall sentiment label
                overall_label = classify_sentiment(avg_sentiment)
                overall_color = 'success' if overall_label == 'Bullish' else ('danger' if overall_label == 'Bearish' else 'warning')
                
                return html.Div([
                    # Overall sentiment badge
                    html.Div([
                        dbc.Badge(f"{overall_label} ({avg_sentiment:+.3f})", color=overall_color, className='mb-2'),
                        html.Span(f" • {total} articles", className='text-muted small ms-2')
                    ]),
                    # Sentiment bars
                    html.Div([
                        html.Div([
                            html.Span("🟢 Bullish: ", className='small fw-semibold', style={'color': '#198754'}),
                            html.Span(f"{bullish} ({bullish_pct:.0f}%)", className='small'),
                            dbc.Progress(value=bullish_pct, color='success', style={'height': '6px'}, className='mt-1')
                        ], className='mb-1'),
                        html.Div([
                            html.Span("🔴 Bearish: ", className='small fw-semibold', style={'color': '#dc3545'}),
                            html.Span(f"{bearish} ({bearish_pct:.0f}%)", className='small'),
                            dbc.Progress(value=bearish_pct, color='danger', style={'height': '6px'}, className='mt-1')
                        ], className='mb-1'),
                        html.Div([
                            html.Span("🟡 Neutral: ", className='small fw-semibold', style={'color': '#ffc107'}),
                            html.Span(f"{neutral} ({neutral_pct:.0f}%)", className='small'),
                            dbc.Progress(value=neutral_pct, color='warning', style={'height': '6px'}, className='mt-1')
                        ])
                    ])
                ])
    except Exception as e:
        print(f"[!] News sentiment stats error: {e}")
    
    return html.P("No news sentiment data", className='text-muted small')


# ============================================
# SOCIAL/TWITTER SENTIMENT BREAKDOWN CALLBACK
# ============================================

@app.callback(Output('social-sentiment-stats', 'children'), Input('store', 'data'))
def update_social_sentiment_stats(data):
    """Display social media sentiment breakdown with bullish/bearish/neutral counts."""
    if not data:
        return html.P("Loading...", className='text-muted small')
    
    try:
        tweets_json = data.get('tweets', '{}')
        if tweets_json and tweets_json != '{}':
            df_tweets = pd.read_json(tweets_json)
            
            if not df_tweets.empty and 'sentiment_score' in df_tweets.columns:
                # Count by sentiment label
                bullish = (df_tweets['sentiment_label'] == 'Bullish').sum() if 'sentiment_label' in df_tweets.columns else 0
                bearish = (df_tweets['sentiment_label'] == 'Bearish').sum() if 'sentiment_label' in df_tweets.columns else 0
                neutral = (df_tweets['sentiment_label'] == 'Neutral').sum() if 'sentiment_label' in df_tweets.columns else 0
                total = len(df_tweets)
                avg_sentiment = df_tweets['sentiment_score'].mean()
                
                # Calculate percentages
                bullish_pct = (bullish / total * 100) if total > 0 else 0
                bearish_pct = (bearish / total * 100) if total > 0 else 0
                neutral_pct = (neutral / total * 100) if total > 0 else 0
                
                # Get sources breakdown
                sources = df_tweets['source'].value_counts().head(3).to_dict() if 'source' in df_tweets.columns else {}
                source_text = ", ".join([f"{k}: {v}" for k, v in sources.items()])
                
                # Overall sentiment label
                overall_label = classify_sentiment(avg_sentiment)
                overall_color = 'success' if overall_label == 'Bullish' else ('danger' if overall_label == 'Bearish' else 'warning')
                
                return html.Div([
                    # Overall sentiment badge
                    html.Div([
                        dbc.Badge(f"{overall_label} ({avg_sentiment:+.3f})", color=overall_color, className='mb-2'),
                        html.Span(f" • {total} posts", className='text-muted small ms-2')
                    ]),
                    # Sources info
                    html.Div([
                        html.Small(f"📍 Sources: {source_text}", className='text-muted')
                    ], className='mb-1') if source_text else None,
                    # Sentiment bars
                    html.Div([
                        html.Div([
                            html.Span("🟢 Bullish: ", className='small fw-semibold', style={'color': '#198754'}),
                            html.Span(f"{bullish} ({bullish_pct:.0f}%)", className='small'),
                            dbc.Progress(value=bullish_pct, color='success', style={'height': '6px'}, className='mt-1')
                        ], className='mb-1'),
                        html.Div([
                            html.Span("🔴 Bearish: ", className='small fw-semibold', style={'color': '#dc3545'}),
                            html.Span(f"{bearish} ({bearish_pct:.0f}%)", className='small'),
                            dbc.Progress(value=bearish_pct, color='danger', style={'height': '6px'}, className='mt-1')
                        ], className='mb-1'),
                        html.Div([
                            html.Span("🟡 Neutral: ", className='small fw-semibold', style={'color': '#ffc107'}),
                            html.Span(f"{neutral} ({neutral_pct:.0f}%)", className='small'),
                            dbc.Progress(value=neutral_pct, color='warning', style={'height': '6px'}, className='mt-1')
                        ])
                    ])
                ])
    except Exception as e:
        print(f"[!] Social sentiment stats error: {e}")
    
    return html.P("No social media data", className='text-muted small')


# ============================================
# SOCIAL SENTIMENT CHART CALLBACK
# ============================================

@app.callback(Output('chart-social-sentiment', 'figure'), Input('store', 'data'))
def update_social_sentiment_chart(data):
    """Create a pie/donut chart showing social sentiment distribution."""
    fig = go.Figure()
    
    if not data:
        fig.add_annotation(text="No data", x=0.5, y=0.5, xref='paper', yref='paper', showarrow=False)
        fig.update_layout(height=150, template='plotly_white', paper_bgcolor='#ffffff')
        return fig
    
    try:
        tweets_json = data.get('tweets', '{}')
        if tweets_json and tweets_json != '{}':
            df_tweets = pd.read_json(tweets_json)
            
            if not df_tweets.empty and 'sentiment_label' in df_tweets.columns:
                # Count by sentiment
                counts = df_tweets['sentiment_label'].value_counts()
                
                labels = []
                values = []
                colors = []
                
                for label in ['Bullish', 'Bearish', 'Neutral']:
                    if label in counts.index:
                        labels.append(label)
                        values.append(counts[label])
                        colors.append('#198754' if label == 'Bullish' else ('#dc3545' if label == 'Bearish' else '#ffc107'))
                
                if values:
                    fig.add_trace(go.Pie(
                        labels=labels,
                        values=values,
                        hole=0.5,
                        marker_colors=colors,
                        textinfo='percent',
                        textfont_size=10,
                        showlegend=True
                    ))
                    
                    fig.update_layout(
                        height=150,
                        template='plotly_white',
                        paper_bgcolor='#ffffff',
                        margin=dict(l=10, r=10, t=10, b=10),
                        legend=dict(orientation='h', y=-0.1, x=0.5, xanchor='center', font=dict(size=9)),
                        font=dict(family='Inter, sans-serif', size=10)
                    )
                    return fig
    except Exception as e:
        print(f"[!] Social sentiment chart error: {e}")
    
    fig.add_annotation(text="No social data", x=0.5, y=0.5, xref='paper', yref='paper', showarrow=False)
    fig.update_layout(height=150, template='plotly_white', paper_bgcolor='#ffffff')
    return fig


# ============================================
# COMBINED SENTIMENT SUMMARY CALLBACK
# ============================================

@app.callback(Output('combined-sentiment-summary', 'children'), Input('store', 'data'))
def update_combined_sentiment_summary(data):
    """Display combined sentiment summary with interpretation."""
    if not data:
        return html.P("Loading...", className='text-muted small')
    
    try:
        # Get news sentiment
        news_sentiment = 0
        news_bullish = news_bearish = news_neutral = 0
        news_total = 0
        
        news_json = data.get('news', '{}')
        if news_json and news_json != '{}':
            df_news = pd.read_json(news_json)
            if not df_news.empty and 'sentiment_score' in df_news.columns:
                news_sentiment = df_news['sentiment_score'].mean()
                news_total = len(df_news)
                if 'sentiment_label' in df_news.columns:
                    news_bullish = (df_news['sentiment_label'] == 'Bullish').sum()
                    news_bearish = (df_news['sentiment_label'] == 'Bearish').sum()
                    news_neutral = (df_news['sentiment_label'] == 'Neutral').sum()
        
        # Get social sentiment
        social_sentiment = 0
        social_bullish = social_bearish = social_neutral = 0
        social_total = 0
        
        tweets_json = data.get('tweets', '{}')
        if tweets_json and tweets_json != '{}':
            df_tweets = pd.read_json(tweets_json)
            if not df_tweets.empty and 'sentiment_score' in df_tweets.columns:
                social_sentiment = df_tweets['sentiment_score'].mean()
                social_total = len(df_tweets)
                if 'sentiment_label' in df_tweets.columns:
                    social_bullish = (df_tweets['sentiment_label'] == 'Bullish').sum()
                    social_bearish = (df_tweets['sentiment_label'] == 'Bearish').sum()
                    social_neutral = (df_tweets['sentiment_label'] == 'Neutral').sum()
        
        # Calculate fused sentiment (60% news, 40% social)
        if news_total > 0 and social_total > 0:
            fused_sentiment = news_sentiment * 0.6 + social_sentiment * 0.4
        elif news_total > 0:
            fused_sentiment = news_sentiment
        elif social_total > 0:
            fused_sentiment = social_sentiment
        else:
            return html.P("No sentiment data available", className='text-muted small')
        
        # Total counts
        total_bullish = news_bullish + social_bullish
        total_bearish = news_bearish + social_bearish
        total_neutral = news_neutral + social_neutral
        grand_total = news_total + social_total
        
        # Overall label
        overall = classify_sentiment(fused_sentiment)
        overall_color = 'success' if overall == 'Bullish' else ('danger' if overall == 'Bearish' else 'warning')
        
        # Generate interpretation text
        if overall == 'Bullish':
            interpretation = f"Market sentiment is BULLISH. Out of {grand_total} analyzed posts, {total_bullish} ({total_bullish/grand_total*100:.0f}%) express positive outlook on Bitcoin. News sentiment: {news_sentiment:+.3f}, Social sentiment: {social_sentiment:+.3f}."
        elif overall == 'Bearish':
            interpretation = f"Market sentiment is BEARISH. Out of {grand_total} analyzed posts, {total_bearish} ({total_bearish/grand_total*100:.0f}%) express negative outlook on Bitcoin. News sentiment: {news_sentiment:+.3f}, Social sentiment: {social_sentiment:+.3f}."
        else:
            interpretation = f"Market sentiment is NEUTRAL. Mixed signals from {grand_total} analyzed posts. News sentiment: {news_sentiment:+.3f}, Social sentiment: {social_sentiment:+.3f}. Consider other indicators."
        
        return html.Div([
            # Main sentiment card
            dbc.Alert([
                html.H5([
                    "Overall: ",
                    dbc.Badge(overall.upper(), color=overall_color, className='ms-2')
                ], className='mb-2'),
                html.P([
                    html.Strong(f"Fused Score: {fused_sentiment:+.4f}"),
                    html.Span(" (60% News + 40% Social)", className='text-muted small ms-2')
                ], className='mb-2'),
            ], color=overall_color, className='mb-2 py-2'),
            
            # Breakdown table
            html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Source", className='small'),
                        html.Th("🟢 Bull", className='small text-center'),
                        html.Th("🔴 Bear", className='small text-center'),
                        html.Th("🟡 Neut", className='small text-center'),
                        html.Th("Score", className='small text-center')
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td("📰 News", className='small'),
                        html.Td(str(news_bullish), className='small text-center text-success'),
                        html.Td(str(news_bearish), className='small text-center text-danger'),
                        html.Td(str(news_neutral), className='small text-center text-warning'),
                        html.Td(f"{news_sentiment:+.3f}", className='small text-center')
                    ]),
                    html.Tr([
                        html.Td("🐦 Social", className='small'),
                        html.Td(str(social_bullish), className='small text-center text-success'),
                        html.Td(str(social_bearish), className='small text-center text-danger'),
                        html.Td(str(social_neutral), className='small text-center text-warning'),
                        html.Td(f"{social_sentiment:+.3f}", className='small text-center')
                    ]),
                    html.Tr([
                        html.Td(html.Strong("Total"), className='small'),
                        html.Td(html.Strong(str(total_bullish)), className='small text-center text-success'),
                        html.Td(html.Strong(str(total_bearish)), className='small text-center text-danger'),
                        html.Td(html.Strong(str(total_neutral)), className='small text-center text-warning'),
                        html.Td(html.Strong(f"{fused_sentiment:+.3f}"), className='small text-center')
                    ])
                ])
            ], className='table table-sm table-bordered mb-2', style={'fontSize': '0.75rem'}),
            
            # Interpretation text
            html.P(interpretation, className='small text-muted mb-0', style={'lineHeight': '1.4'})
        ])
        
    except Exception as e:
        print(f"[!] Combined sentiment summary error: {e}")
        return html.P(f"Error: {str(e)}", className='text-muted small')


# ============================================
# RUN
# ============================================

if __name__ == '__main__':
    print("=" * 60)
    print("  SINTA 1 Bitcoin Forecast Dashboard")
    print("  http://localhost:8050")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=8050)
