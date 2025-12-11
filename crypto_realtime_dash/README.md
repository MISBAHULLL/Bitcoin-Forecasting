# SINTA 1 Bitcoin Forecasting System

**Production-grade Bitcoin forecasting with ARIMA/ARIMAX, VADER sentiment, and real-time data.**

## Features

- **Real-time OHLCV** from Binance API
- **Multi-source news** from Coindesk & Cointelegraph RSS
- **Twitter sentiment** via snscrape
- **VADER sentiment analysis** with fusion
- **ARIMA & ARIMAX** forecasting (no Prophet, XGBoost, LSTM)
- **Technical indicators**: RSI-14, SMA-14, MACD (12,26,9)
- **MySQL database** with phpMyAdmin compatibility
- **Dash dashboard** with 60-second auto-refresh

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Database
Edit `config/config.yaml`:
```yaml
database:
  host: localhost
  port: 3306
  user: root
  password: 
  database: crypto_forecast
```

### 3. Run Pipeline
```bash
python main_pipeline.py
```

### 4. Launch Dashboard
```bash
python app_dash.py
```
Open http://localhost:8050

## Project Structure

```
crypto_realtime_dash/
├── config/
│   └── config.yaml         # MySQL + API settings
├── modules/
│   ├── db.py               # MySQL CRUD
│   ├── price_fetcher.py    # Binance OHLCV
│   ├── news_fetcher.py     # RSS feeds
│   ├── twitter_fetcher.py  # snscrape
│   ├── sentiment.py        # VADER
│   ├── indicators.py       # RSI, SMA, MACD
│   ├── arima_model.py      # ARIMA/ARIMAX
│   ├── evaluation.py       # Metrics
│   ├── interpretation.py   # Insights
│   └── utils.py            # Helpers
├── main_pipeline.py        # ETL + Forecasting
├── app_dash.py             # Dashboard
└── requirements.txt
```

## Dashboard Features

1. **Header Metrics**: Price, Change, Sentiment, RSI, MACD, Signal
2. **Candlestick Chart**: OHLCV + SMA overlay + RSI panel + MACD panel
3. **Sentiment Analysis**: Timeline + Distribution charts
4. **Forecast Comparison**: ARIMA vs ARIMAX with metrics
5. **Interpretation Panel**: Auto-generated trading signals
6. **Data Explorer**: View MySQL tables (Prices, News, Tweets)

## Models

| Model | Description |
|-------|-------------|
| ARIMA | Univariate baseline (close price only) |
| ARIMAX | With exogenous: sentiment, RSI, SMA, MACD, volume |

## License

Academic Research Use - SINTA 1 Ready
