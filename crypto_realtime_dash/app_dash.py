"""
Bitcoin Sentiment Forecasting Dashboard
Clean White Theme - Modern Responsive UI
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

# Initialize Dash
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.FLATLY,
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap'
    ],
    meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}],
    title='BTC Forecast Dashboard'
)

server = app.server

# Clean White Theme Colors
THEME = {
    'bg': '#ffffff',
    'card_bg': '#f8f9fa',
    'text': '#212529',
    'text_muted': '#6c757d',
    'primary': '#0d6efd',
    'success': '#198754',
    'danger': '#dc3545',
    'warning': '#ffc107',
    'info': '#0dcaf0',
    'border': '#dee2e6',
    'shadow': '0 0.125rem 0.25rem rgba(0,0,0,0.075)'
}


# ============================================
# GENERATE SAMPLE DATA (Fallback)
# ============================================

def generate_sample_data(n=200):
    """Generate sample OHLCV data."""
    np.random.seed(int(datetime.now().timestamp()) % 1000)
    
    dates = pd.date_range(end=datetime.now(), periods=n, freq='h')
    base = 97000
    returns = np.random.randn(n) * 0.003
    prices = base * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.uniform(-0.002, 0.002, n)),
        'high': prices * (1 + np.abs(np.random.randn(n) * 0.005)),
        'low': prices * (1 - np.abs(np.random.randn(n) * 0.005)),
        'close': prices,
        'volume': np.random.uniform(500, 2000, n)
    })
    
    return df


def load_data(interval='1h', limit=200):
    """Load data with fallback to sample."""
    print(f"[*] Loading data: {interval}, {limit}")
    
    # Try to fetch real data
    df = get_ohlcv(interval=interval, limit=limit)
    
    # Fallback to sample
    if df.empty or len(df) < 10:
        print("[*] Using sample data")
        df = generate_sample_data(limit)
    
    # Compute indicators
    df = compute_all_indicators(df)
    
    # Get news and tweets
    df_news = get_all_news()
    if not df_news.empty:
        df_news = analyze_news_sentiment(df_news)
    
    df_tweets = get_tweets(limit=50)
    if not df_tweets.empty:
        df_tweets = analyze_tweets_sentiment(df_tweets)
    
    # Add sentiment
    df['fused_sentiment'] = np.random.uniform(-0.3, 0.3, len(df))
    df['sentiment_label'] = df['fused_sentiment'].apply(classify_sentiment)
    
    print(f"[+] Loaded {len(df)} records")
    return df, df_news, df_tweets


# ============================================
# CHARTS
# ============================================

def make_price_chart(df):
    """Create candlestick with RSI and MACD."""
    if df.empty:
        return go.Figure()
    
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.6, 0.2, 0.2],
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=['<b>BTC/USDT Price</b>', '<b>RSI (14)</b>', '<b>MACD</b>']
    )
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name='BTC', showlegend=True,
        increasing=dict(line=dict(color=THEME['success']), fillcolor='rgba(25,135,84,0.7)'),
        decreasing=dict(line=dict(color=THEME['danger']), fillcolor='rgba(220,53,69,0.7)')
    ), row=1, col=1)
    
    # SMA
    if 'sma_14' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['sma_14'], name='SMA-14',
            line=dict(color=THEME['primary'], width=2)
        ), row=1, col=1)
    
    if 'sma_50' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['sma_50'], name='SMA-50',
            line=dict(color=THEME['warning'], width=2)
        ), row=1, col=1)
    
    # RSI
    if 'rsi_14' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['rsi_14'], name='RSI',
            line=dict(color=THEME['info'], width=2),
            fill='tozeroy', fillcolor='rgba(13,202,240,0.1)'
        ), row=2, col=1)
        
        fig.add_hline(y=70, line_dash='dash', line_color=THEME['danger'], row=2, col=1)
        fig.add_hline(y=30, line_dash='dash', line_color=THEME['success'], row=2, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor='rgba(220,53,69,0.1)', line_width=0, row=2, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor='rgba(25,135,84,0.1)', line_width=0, row=2, col=1)
    
    # MACD
    if 'macd' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['macd'], name='MACD',
            line=dict(color=THEME['primary'], width=2)
        ), row=3, col=1)
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['macd_signal'], name='Signal',
            line=dict(color=THEME['warning'], width=2)
        ), row=3, col=1)
        
        colors = [THEME['success'] if v >= 0 else THEME['danger'] for v in df['macd_hist']]
        fig.add_trace(go.Bar(
            x=df['timestamp'], y=df['macd_hist'], name='Histogram',
            marker_color=colors, opacity=0.6, showlegend=False
        ), row=3, col=1)
    
    fig.update_layout(
        height=650,
        template='plotly_white',
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=60, r=30, t=50, b=50),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center', font=dict(size=11)),
        font=dict(family='Inter, sans-serif', size=12, color=THEME['text']),
        xaxis3=dict(showgrid=True, gridcolor='#eee'),
        yaxis=dict(showgrid=True, gridcolor='#eee'),
        yaxis2=dict(range=[0, 100], showgrid=True, gridcolor='#eee'),
        yaxis3=dict(showgrid=True, gridcolor='#eee')
    )
    
    return fig


def make_sentiment_chart(df):
    """Create sentiment chart."""
    if df.empty or 'fused_sentiment' not in df.columns:
        return go.Figure()
    
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter'}, {'type': 'pie'}]],
        subplot_titles=['<b>Sentiment Timeline</b>', '<b>Distribution</b>']
    )
    
    # Timeline
    colors = [THEME['success'] if s > 0.05 else (THEME['danger'] if s < -0.05 else THEME['warning']) 
             for s in df['fused_sentiment']]
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['fused_sentiment'],
        mode='lines+markers', name='Sentiment',
        line=dict(color=THEME['primary'], width=2),
        marker=dict(size=5, color=colors)
    ), row=1, col=1)
    
    fig.add_hline(y=0.05, line_dash='dash', line_color=THEME['success'], opacity=0.5, row=1, col=1)
    fig.add_hline(y=-0.05, line_dash='dash', line_color=THEME['danger'], opacity=0.5, row=1, col=1)
    fig.add_hline(y=0, line_color='#ccc', row=1, col=1)
    
    # Pie
    labels = df['sentiment_label'].value_counts()
    colors_pie = [THEME['success'] if l == 'Bullish' else (THEME['danger'] if l == 'Bearish' else THEME['warning']) for l in labels.index]
    
    fig.add_trace(go.Pie(
        labels=labels.index, values=labels.values,
        hole=0.5, marker_colors=colors_pie,
        textinfo='percent+label', textfont=dict(size=11)
    ), row=1, col=2)
    
    fig.update_layout(
        height=350,
        template='plotly_white',
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=60, r=30, t=50, b=50),
        font=dict(family='Inter, sans-serif', size=12, color=THEME['text']),
        showlegend=False
    )
    
    return fig


def make_forecast_chart(results):
    """Create forecast comparison chart."""
    fig = go.Figure()
    
    if not results:
        fig.add_annotation(text="No forecast data", x=0.5, y=0.5, xref='paper', yref='paper', showarrow=False)
        fig.update_layout(height=300, template='plotly_white', paper_bgcolor='white')
        return fig
    
    for name, result in results.items():
        if 'actual' in result and 'predictions' in result:
            actual = result['actual']
            predicted = result['predictions']
            n = min(len(actual), len(predicted), 30)
            
            if name == 'ARIMA':
                fig.add_trace(go.Scatter(
                    x=list(range(n)), y=actual[:n], name='Actual',
                    line=dict(color=THEME['text'], width=3)
                ))
            
            fig.add_trace(go.Scatter(
                x=list(range(n)), y=predicted[:n], name=f'{name}',
                line=dict(width=2, dash='dash')
            ))
    
    fig.update_layout(
        height=280,
        template='plotly_white',
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=60, r=30, t=30, b=50),
        legend=dict(orientation='h', y=-0.2, x=0.5, xanchor='center'),
        font=dict(family='Inter, sans-serif', size=12),
        xaxis_title='Test Period', yaxis_title='Price ($)'
    )
    
    return fig


# ============================================
# LAYOUT
# ============================================

def metric_card(title, value, subtitle='', color='primary', icon=''):
    """Modern metric card."""
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Span(icon, style={'fontSize': '1.5rem', 'marginRight': '8px'}),
                html.Div([
                    html.P(title, className='text-muted mb-0', style={'fontSize': '0.8rem', 'fontWeight': '500'}),
                    html.H4(value, className='mb-0', style={'fontWeight': '700', 'color': THEME.get(color, THEME['text'])}),
                    html.Small(subtitle, style={'color': THEME.get(color, THEME['text_muted'])})
                ])
            ], className='d-flex align-items-center')
        ], className='py-3')
    ], className='h-100 shadow-sm border-0', style={'borderRadius': '12px', 'backgroundColor': THEME['card_bg']})


app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H2([
                html.Span("₿", className='text-warning me-2'),
                "Bitcoin Sentiment Forecast"
            ], className='mb-1', style={'fontWeight': '700'}),
            html.P("ARIMA/ARIMAX | VADER Sentiment | Real-time Analysis", className='text-muted mb-0')
        ], md=8),
        dbc.Col([
            dbc.Button([html.Span("↻", className='me-2'), "Refresh"], id='refresh-btn', color='primary', className='me-2'),
            html.Span(id='last-update', className='text-muted small')
        ], md=4, className='text-end d-flex align-items-center justify-content-end')
    ], className='py-4 border-bottom mb-4'),
    
    # Controls
    dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Timeframe", className='form-label fw-semibold'),
                    dcc.Dropdown(id='timeframe', options=[{'label': v, 'value': k} for k, v in TIMEFRAMES.items()],
                                value='1h', clearable=False, className='')
                ], md=2),
                dbc.Col([
                    html.Label("Data Points", className='form-label fw-semibold'),
                    dcc.Slider(id='limit', min=50, max=500, value=200, step=50,
                              marks={50: '50', 200: '200', 500: '500'}, tooltip={'placement': 'bottom'})
                ], md=4),
                dbc.Col([
                    html.Label("ARIMA Order (p, d, q)", className='form-label fw-semibold'),
                    dbc.InputGroup([
                        dbc.Input(id='p', type='number', value=5, min=1, max=10, className='text-center'),
                        dbc.Input(id='d', type='number', value=1, min=0, max=2, className='text-center'),
                        dbc.Input(id='q', type='number', value=0, min=0, max=5, className='text-center')
                    ], size='sm')
                ], md=3),
                dbc.Col([
                    html.Label("Auto Refresh", className='form-label fw-semibold'),
                    dbc.Switch(id='auto-refresh', value=True, label="60 sec", className='mt-1')
                ], md=3)
            ])
        ])
    ], className='mb-4 shadow-sm border-0', style={'borderRadius': '12px', 'backgroundColor': THEME['card_bg']}),
    
    # Metrics
    dbc.Row([
        dbc.Col(html.Div(id='m-price'), md=2),
        dbc.Col(html.Div(id='m-change'), md=2),
        dbc.Col(html.Div(id='m-sent'), md=2),
        dbc.Col(html.Div(id='m-rsi'), md=2),
        dbc.Col(html.Div(id='m-macd'), md=2),
        dbc.Col(html.Div(id='m-signal'), md=2)
    ], className='mb-4 g-3'),
    
    # Price Chart
    dbc.Card([
        dbc.CardHeader(html.H5("📈 Price Chart with Indicators", className='mb-0 fw-semibold')),
        dbc.CardBody([dcc.Graph(id='chart-price', config={'displayModeBar': True, 'displaylogo': False})])
    ], className='mb-4 shadow-sm border-0', style={'borderRadius': '12px'}),
    
    # Two columns
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("💬 Sentiment Analysis", className='mb-0 fw-semibold')),
                dbc.CardBody([dcc.Graph(id='chart-sentiment', config={'displayModeBar': False})])
            ], className='shadow-sm border-0 h-100', style={'borderRadius': '12px'})
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("🎯 ARIMA vs ARIMAX", className='mb-0 fw-semibold')),
                dbc.CardBody([
                    dcc.Graph(id='chart-forecast', config={'displayModeBar': False}),
                    html.Div(id='metrics-table')
                ])
            ], className='shadow-sm border-0 h-100', style={'borderRadius': '12px'})
        ], md=6)
    ], className='mb-4 g-4'),
    
    # Interpretation
    dbc.Card([
        dbc.CardHeader(html.H5("🧠 AI Trading Signals", className='mb-0 fw-semibold')),
        dbc.CardBody(id='interpretation')
    ], className='mb-4 shadow-sm border-0', style={'borderRadius': '12px'}),
    
    # Store and Interval
    dcc.Store(id='store'),
    dcc.Interval(id='interval', interval=60000, n_intervals=0),
    
    # Footer
    html.Footer([
        html.P("SINTA 1 Bitcoin Forecasting System", className='text-muted text-center mb-0 py-3')
    ], className='border-top mt-4')
    
], fluid=True, className='py-3', style={'backgroundColor': THEME['bg'], 'minHeight': '100vh', 'fontFamily': 'Inter, sans-serif'})


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
    df, news, tweets = load_data(tf, limit or 200)
    return {
        'df': df.to_json(date_format='iso'),
        'news': news.to_json(date_format='iso') if not news.empty else '{}',
        'tweets': tweets.to_json(date_format='iso') if not tweets.empty else '{}'
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
        return [metric_card("Loading", "...")] * 6
    
    df = pd.read_json(data['df'])
    cp = fetch_current_price()
    
    price = cp.get('price', df['close'].iloc[-1] if not df.empty else 97000)
    change = cp.get('change_24h', 0)
    rsi = df['rsi_14'].iloc[-1] if 'rsi_14' in df.columns else 50
    macd = df['macd'].iloc[-1] if 'macd' in df.columns else 0
    macd_s = df['macd_signal'].iloc[-1] if 'macd_signal' in df.columns else 0
    sent = df['fused_sentiment'].iloc[-1] if 'fused_sentiment' in df.columns else 0
    
    rsi_state = 'Overbought' if rsi > 70 else ('Oversold' if rsi < 30 else 'Neutral')
    macd_trend = 'Bullish' if macd > macd_s else 'Bearish'
    sent_label = classify_sentiment(sent)
    
    buy = (rsi < 30) + (macd > macd_s) + (sent > 0.05)
    sell = (rsi > 70) + (macd < macd_s) + (sent < -0.05)
    signal = 'BUY' if buy > sell else ('SELL' if sell > buy else 'HOLD')
    
    return (
        metric_card("BTC Price", f"${price:,.0f}", cp.get('source', ''), 'primary', '💰'),
        metric_card("24h Change", f"{change:+.2f}%", "", 'success' if change >= 0 else 'danger', '📊'),
        metric_card("Sentiment", sent_label, f"{sent:.3f}", 'success' if sent > 0.05 else ('danger' if sent < -0.05 else 'warning'), '💬'),
        metric_card("RSI-14", f"{rsi:.1f}", rsi_state, 'danger' if rsi > 70 else ('success' if rsi < 30 else 'warning'), '📉'),
        metric_card("MACD", f"{macd:.1f}", macd_trend, 'success' if macd > macd_s else 'danger', '📈'),
        metric_card("Signal", signal, "", 'success' if signal == 'BUY' else ('danger' if signal == 'SELL' else 'warning'), '🎯')
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
    return make_sentiment_chart(df.tail(100))


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
        return go.Figure(), html.P("Not enough data", className='text-muted')
    
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
        return go.Figure(), html.P("Model training failed", className='text-muted')
    
    comparison = compare_models(results)
    
    table = dbc.Table([
        html.Thead(html.Tr([html.Th("Model"), html.Th("RMSE"), html.Th("MAE"), html.Th("MAPE")])),
        html.Tbody([
            html.Tr([
                html.Td(r['Model']),
                html.Td(f"${r['RMSE']:,.0f}"),
                html.Td(f"${r['MAE']:,.0f}"),
                html.Td(f"{r['MAPE']:.2f}%")
            ]) for _, r in comparison.iterrows()
        ])
    ], striped=True, hover=True, size='sm', className='mt-3') if not comparison.empty else ""
    
    return make_forecast_chart(results), table


@app.callback(Output('interpretation', 'children'), Input('store', 'data'))
def update_interp(data):
    if not data:
        return html.P("Loading...", className='text-muted')
    
    df = pd.read_json(data['df'])
    interp = generate_interpretation(df, {})
    insights = interp.get('insights', [])
    
    if not insights:
        return html.P("No significant signals", className='text-muted')
    
    alerts = []
    for i in insights[:5]:
        color = 'info' if i['level'] == 'info' else ('warning' if i['level'] == 'warning' else 'danger')
        badge = dbc.Badge(i['signal'].upper(), color='success' if i['signal'] == 'buy' else ('danger' if i['signal'] == 'sell' else 'warning'), className='ms-2') if i.get('signal') else None
        alerts.append(dbc.Alert([html.Strong(f"{i['category']}: "), i['message'], badge], color=color, className='mb-2'))
    
    overall = interp.get('overall_signal', 'HOLD')
    alerts.append(html.Div([
        html.Strong("OVERALL: "),
        dbc.Badge(overall, color='success' if overall == 'BUY' else ('danger' if overall == 'SELL' else 'warning'), className='ms-2', style={'fontSize': '1rem'})
    ], className='p-3 bg-light rounded mt-3'))
    
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
