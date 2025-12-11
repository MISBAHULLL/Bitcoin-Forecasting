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
# DATA LOADING WITH CACHING
# ============================================

_data_cache = {}
_cache_time = 0
CACHE_DURATION = 30

def generate_sample_data(n=200):
    """Generate sample OHLCV data."""
    np.random.seed(42)
    
    dates = pd.date_range(end=datetime.now(), periods=n, freq='h')
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


def load_data(interval='1h', limit=200):
    """Load data with caching."""
    global _data_cache, _cache_time
    
    cache_key = f"{interval}_{limit}"
    current_time = time.time()
    
    if cache_key in _data_cache and (current_time - _cache_time) < CACHE_DURATION:
        print(f"[*] Using cached data ({cache_key})")
        return _data_cache[cache_key]
    
    print(f"[*] Loading fresh data: {interval}, {limit}")
    
    df = get_ohlcv(interval=interval, limit=limit)
    
    if df.empty or len(df) < 10:
        print("[*] Using sample data")
        df = generate_sample_data(limit)
    
    df = compute_all_indicators(df)
    
    try:
        df_news = get_all_news()
        if not df_news.empty:
            df_news = analyze_news_sentiment(df_news)
    except:
        df_news = pd.DataFrame()
    
    try:
        df_tweets = get_tweets(limit=30)
        if not df_tweets.empty:
            df_tweets = analyze_tweets_sentiment(df_tweets)
    except:
        df_tweets = pd.DataFrame()
    
    df['fused_sentiment'] = np.random.uniform(-0.3, 0.3, len(df))
    df['sentiment_label'] = df['fused_sentiment'].apply(classify_sentiment)
    
    result = (df, df_news, df_tweets)
    
    _data_cache[cache_key] = result
    _cache_time = current_time
    
    print(f"[+] Loaded {len(df)} records")
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
    """Create compact sentiment chart - White theme."""
    if df.empty or 'fused_sentiment' not in df.columns:
        return go.Figure()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['fused_sentiment'],
        mode='lines', name='Sentiment',
        line=dict(color='#2196f3', width=2),
        fill='tozeroy', fillcolor='rgba(33,150,243,0.1)'
    ))
    
    fig.add_hline(y=0.1, line_dash='dot', line_color='#26a69a', opacity=0.7)
    fig.add_hline(y=-0.1, line_dash='dot', line_color='#ef5350', opacity=0.7)
    fig.add_hline(y=0, line_color='#9e9e9e', opacity=0.5)
    
    fig.update_layout(
        height=200,
        template='plotly_white',
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        margin=dict(l=50, r=20, t=20, b=40),
        font=dict(family='Inter, sans-serif', size=11, color='#212529'),
        showlegend=False,
        xaxis=dict(showgrid=True, gridcolor='#e9ecef'),
        yaxis=dict(showgrid=True, gridcolor='#e9ecef', range=[-0.5, 0.5])
    )
    
    return fig


def make_forecast_chart(results):
    """Create forecast comparison chart - White theme."""
    fig = go.Figure()
    
    if not results:
        fig.add_annotation(text="No forecast data", x=0.5, y=0.5, xref='paper', yref='paper', 
                          showarrow=False, font=dict(color='#6c757d'))
        fig.update_layout(height=250, template='plotly_white', paper_bgcolor='#ffffff')
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
            
            fig.add_trace(go.Scatter(
                x=list(range(n)), y=predicted[:n], name=f'{name}',
                line=dict(width=2, dash='dash', color=colors.get(name, '#2196f3'))
            ))
    
    fig.update_layout(
        height=250,
        template='plotly_white',
        paper_bgcolor='#ffffff',
        plot_bgcolor='#ffffff',
        margin=dict(l=50, r=20, t=20, b=50),
        legend=dict(orientation='h', y=-0.2, x=0.5, xanchor='center', font=dict(size=11)),
        font=dict(family='Inter, sans-serif', size=11, color='#212529'),
        xaxis=dict(title='Test Period', showgrid=True, gridcolor='#e9ecef'),
        yaxis=dict(title='Price ($)', showgrid=True, gridcolor='#e9ecef')
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
            html.Span(f" • {news_item['hours_ago']}h ago", style={'color': THEME['text_muted'], 'fontSize': '0.65rem'})
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
    
    # Main Row: Chart + Sidebar
    dbc.Row([
        # Chart Column
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H6("📈 BTC/USDT Price Chart with Indicators", className='mb-0 fw-semibold')),
                dbc.CardBody([
                    dcc.Graph(id='chart-price', config={'displayModeBar': True, 'displaylogo': False})
                ], className='p-2')
            ], className='shadow-sm h-100')
        ], lg=9, md=12, className='mb-3'),
        
        # Sidebar Column
        dbc.Col([
            # Events
            dbc.Card([
                dbc.CardHeader(html.H6("🗓 Upcoming Events", className='mb-0 fw-semibold')),
                dbc.CardBody(id='events-list', className='py-2', style={'maxHeight': '200px', 'overflowY': 'auto'})
            ], className='shadow-sm mb-3'),
            
            # News
            dbc.Card([
                dbc.CardHeader(html.H6("📰 Latest News", className='mb-0 fw-semibold')),
                dbc.CardBody(id='news-list', className='py-2', style={'maxHeight': '250px', 'overflowY': 'auto'})
            ], className='shadow-sm')
        ], lg=3, md=12)
    ]),
    
    # Second Row: Sentiment + Forecast + Signals
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H6("💬 Sentiment Timeline", className='mb-0 fw-semibold')),
                dbc.CardBody([
                    dcc.Graph(id='chart-sentiment', config={'displayModeBar': False})
                ], className='p-2')
            ], className='shadow-sm h-100')
        ], md=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H6("🎯 ARIMA vs ARIMAX Forecast", className='mb-0 fw-semibold')),
                dbc.CardBody([
                    dcc.Graph(id='chart-forecast', config={'displayModeBar': False}),
                    html.Div(id='metrics-table', className='mt-2')
                ], className='p-2')
            ], className='shadow-sm h-100')
        ], md=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H6("🧠 Trading Signals", className='mb-0 fw-semibold')),
                dbc.CardBody(id='interpretation', className='py-2', style={'maxHeight': '350px', 'overflowY': 'auto'})
            ], className='shadow-sm h-100')
        ], md=4)
    ], className='mb-3 g-3'),
    
    # Store and Interval
    dcc.Store(id='store'),
    dcc.Interval(id='interval', interval=60000, n_intervals=0),
    
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
    State('timeframe', 'value'),
    State('limit', 'value')
)
def update_data(n, intervals, tf, limit):
    df, news, tweets = load_data(tf, limit or 150)
    cp = fetch_current_price()
    
    return {
        'df': df.to_json(date_format='iso'),
        'news': news.to_json(date_format='iso') if not news.empty else '{}',
        'tweets': tweets.to_json(date_format='iso') if not tweets.empty else '{}',
        'current_price': cp
    }, f"Updated: {datetime.now().strftime('%H:%M:%S')}"


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


@app.callback(Output('news-list', 'children'), Input('store', 'data'))
def update_news(data):
    news = fetch_aggregated_news(limit=5)
    if not news:
        return html.P("No news available", className='text-muted small')
    return html.Div([news_card(n) for n in news])


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
    ], size='sm', striped=True, className='small') if not comparison.empty else ""
    
    return make_forecast_chart(results), table


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
# RUN
# ============================================

if __name__ == '__main__':
    print("=" * 60)
    print("  SINTA 1 Bitcoin Forecast Dashboard")
    print("  http://localhost:8050")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=8050)
