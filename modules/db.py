"""
Database Module - MySQL CRUD Operations
SINTA 1 Bitcoin Forecasting System
"""

import pymysql
import pandas as pd
import yaml
import os
from datetime import datetime
from typing import Optional, List, Dict, Any

# Load config
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.yaml')

def load_config() -> dict:
    """Load configuration from YAML file."""
    try:
        with open(CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {
            'database': {
                'host': 'localhost',
                'port': 3306,
                'user': 'root',
                'password': '',
                'database': 'crypto_forecast'
            }
        }

CONFIG = load_config()


class Database:
    """MySQL Database Handler."""
    
    def __init__(self):
        self.config = CONFIG.get('database', {})
        self.connection = None
        self._connect()
        self._create_tables()
    
    def _connect(self):
        """Establish database connection."""
        try:
            # First try to connect without database to create it
            conn = pymysql.connect(
                host=self.config.get('host', 'localhost'),
                port=self.config.get('port', 3306),
                user=self.config.get('user', 'root'),
                password=self.config.get('password', ''),
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
            
            # Create database if not exists
            db_name = self.config.get('database', 'crypto_forecast')
            with conn.cursor() as cursor:
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            conn.close()
            
            # Connect to the database
            self.connection = pymysql.connect(
                host=self.config.get('host', 'localhost'),
                port=self.config.get('port', 3306),
                user=self.config.get('user', 'root'),
                password=self.config.get('password', ''),
                database=db_name,
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor,
                autocommit=True
            )
            
            print(f"[+] Connected to MySQL: {db_name}")
            
        except pymysql.Error as e:
            print(f"[!] MySQL connection failed: {e}")
            print("[!] Running in file-based fallback mode")
            self.connection = None
    
    def _create_tables(self):
        """Create required tables if they don't exist."""
        if not self.connection:
            return
        
        tables = {
            'prices': """
                CREATE TABLE IF NOT EXISTS prices (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    timestamp DATETIME NOT NULL,
                    open FLOAT NOT NULL,
                    high FLOAT NOT NULL,
                    low FLOAT NOT NULL,
                    close FLOAT NOT NULL,
                    volume FLOAT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY unique_timestamp (timestamp)
                ) ENGINE=InnoDB
            """,
            
            'news': """
                CREATE TABLE IF NOT EXISTS news (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    timestamp DATETIME NOT NULL,
                    title VARCHAR(500) NOT NULL,
                    summary TEXT,
                    source VARCHAR(100),
                    url VARCHAR(500),
                    sentiment_score FLOAT,
                    sentiment_label VARCHAR(20),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY unique_title (title(255))
                ) ENGINE=InnoDB
            """,
            
            'tweets': """
                CREATE TABLE IF NOT EXISTS tweets (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    tweet_id VARCHAR(50) NOT NULL,
                    timestamp DATETIME NOT NULL,
                    username VARCHAR(100),
                    text TEXT,
                    likes INT DEFAULT 0,
                    retweets INT DEFAULT 0,
                    sentiment_score FLOAT,
                    sentiment_label VARCHAR(20),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY unique_tweet (tweet_id)
                ) ENGINE=InnoDB
            """,
            
            'daily_features': """
                CREATE TABLE IF NOT EXISTS daily_features (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    date DATE NOT NULL,
                    close FLOAT,
                    volume FLOAT,
                    rsi_14 FLOAT,
                    sma_14 FLOAT,
                    macd FLOAT,
                    macd_signal FLOAT,
                    macd_hist FLOAT,
                    news_sentiment FLOAT,
                    twitter_sentiment FLOAT,
                    lunarcrush_sentiment FLOAT,
                    fused_sentiment FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY unique_date (date)
                ) ENGINE=InnoDB
            """,
            
            'forecasts': """
                CREATE TABLE IF NOT EXISTS forecasts (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    date DATE NOT NULL,
                    model VARCHAR(50) NOT NULL,
                    horizon INT NOT NULL,
                    predicted_price FLOAT,
                    actual_price FLOAT,
                    rmse FLOAT,
                    mae FLOAT,
                    mape FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY unique_forecast (date, model, horizon)
                ) ENGINE=InnoDB
            """
        }
        
        try:
            with self.connection.cursor() as cursor:
                for table_name, sql in tables.items():
                    cursor.execute(sql)
                    print(f"[+] Table '{table_name}' ready")
        except pymysql.Error as e:
            print(f"[!] Error creating tables: {e}")
    
    def execute(self, sql: str, params: tuple = None) -> Optional[List[Dict]]:
        """Execute SQL query and return results."""
        if not self.connection:
            return None
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(sql, params)
                return cursor.fetchall()
        except pymysql.Error as e:
            print(f"[!] SQL Error: {e}")
            return None
    
    def insert_many(self, table: str, data: List[Dict], ignore_duplicates: bool = True):
        """Insert multiple rows into table."""
        if not self.connection or not data:
            return 0
        
        columns = list(data[0].keys())
        placeholders = ', '.join(['%s'] * len(columns))
        col_str = ', '.join([f'`{c}`' for c in columns])
        
        ignore = 'IGNORE' if ignore_duplicates else ''
        sql = f"INSERT {ignore} INTO `{table}` ({col_str}) VALUES ({placeholders})"
        
        try:
            with self.connection.cursor() as cursor:
                rows = [tuple(row.get(c) for c in columns) for row in data]
                cursor.executemany(sql, rows)
                return cursor.rowcount
        except pymysql.Error as e:
            print(f"[!] Insert error: {e}")
            return 0
    
    def upsert(self, table: str, data: Dict, unique_key: str):
        """Insert or update a row."""
        if not self.connection:
            return False
        
        columns = list(data.keys())
        placeholders = ', '.join(['%s'] * len(columns))
        col_str = ', '.join([f'`{c}`' for c in columns])
        update_str = ', '.join([f'`{c}` = VALUES(`{c}`)' for c in columns if c != unique_key])
        
        sql = f"""
            INSERT INTO `{table}` ({col_str}) VALUES ({placeholders})
            ON DUPLICATE KEY UPDATE {update_str}
        """
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(sql, tuple(data.values()))
                return True
        except pymysql.Error as e:
            print(f"[!] Upsert error: {e}")
            return False
    
    def get_prices(self, limit: int = 500) -> pd.DataFrame:
        """Get price data from database."""
        result = self.execute(
            "SELECT * FROM prices ORDER BY timestamp DESC LIMIT %s",
            (limit,)
        )
        if result:
            df = pd.DataFrame(result)
            return df.sort_values('timestamp').reset_index(drop=True)
        return pd.DataFrame()
    
    def get_news(self, limit: int = 100) -> pd.DataFrame:
        """Get news data from database."""
        result = self.execute(
            "SELECT * FROM news ORDER BY timestamp DESC LIMIT %s",
            (limit,)
        )
        return pd.DataFrame(result) if result else pd.DataFrame()
    
    def get_tweets(self, limit: int = 100) -> pd.DataFrame:
        """Get tweet data from database."""
        result = self.execute(
            "SELECT * FROM tweets ORDER BY timestamp DESC LIMIT %s",
            (limit,)
        )
        return pd.DataFrame(result) if result else pd.DataFrame()
    
    def get_daily_features(self, limit: int = 365) -> pd.DataFrame:
        """Get daily features from database."""
        result = self.execute(
            "SELECT * FROM daily_features ORDER BY date DESC LIMIT %s",
            (limit,)
        )
        if result:
            df = pd.DataFrame(result)
            return df.sort_values('date').reset_index(drop=True)
        return pd.DataFrame()
    
    def get_forecasts(self, model: str = None, limit: int = 100) -> pd.DataFrame:
        """Get forecasts from database."""
        if model:
            result = self.execute(
                "SELECT * FROM forecasts WHERE model = %s ORDER BY date DESC LIMIT %s",
                (model, limit)
            )
        else:
            result = self.execute(
                "SELECT * FROM forecasts ORDER BY date DESC LIMIT %s",
                (limit,)
            )
        return pd.DataFrame(result) if result else pd.DataFrame()
    
    def save_prices(self, df: pd.DataFrame) -> int:
        """Save price data to database."""
        if df.empty:
            return 0
        
        records = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].to_dict('records')
        return self.insert_many('prices', records)
    
    def save_news(self, df: pd.DataFrame) -> int:
        """Save news data to database."""
        if df.empty:
            return 0
        
        columns = ['timestamp', 'title', 'summary', 'source', 'url', 'sentiment_score', 'sentiment_label']
        available = [c for c in columns if c in df.columns]
        records = df[available].to_dict('records')
        return self.insert_many('news', records)
    
    def save_tweets(self, df: pd.DataFrame) -> int:
        """Save tweet data to database."""
        if df.empty:
            return 0
        
        columns = ['tweet_id', 'timestamp', 'username', 'text', 'likes', 'retweets', 'sentiment_score', 'sentiment_label']
        available = [c for c in columns if c in df.columns]
        records = df[available].to_dict('records')
        return self.insert_many('tweets', records)
    
    def save_daily_features(self, df: pd.DataFrame) -> int:
        """Save daily features to database."""
        if df.empty:
            return 0
        
        columns = ['date', 'close', 'volume', 'rsi_14', 'sma_14', 'macd', 'macd_signal', 'macd_hist',
                   'news_sentiment', 'twitter_sentiment', 'lunarcrush_sentiment', 'fused_sentiment']
        available = [c for c in columns if c in df.columns]
        records = df[available].to_dict('records')
        return self.insert_many('daily_features', records)
    
    def save_forecast(self, date, model: str, horizon: int, predicted: float, actual: float = None,
                      rmse: float = None, mae: float = None, mape: float = None) -> bool:
        """Save a forecast to database."""
        return self.upsert('forecasts', {
            'date': date,
            'model': model,
            'horizon': horizon,
            'predicted_price': predicted,
            'actual_price': actual,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }, 'date')
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            print("[+] Database connection closed")


# Global database instance
db = Database()


if __name__ == "__main__":
    print("Testing Database Module...")
    print("=" * 50)
    
    # Test connection
    if db.connection:
        print("[+] Database connected successfully")
        
        # Test insert
        test_data = [{
            'timestamp': datetime.now(),
            'open': 97500.0,
            'high': 98000.0,
            'low': 97000.0,
            'close': 97800.0,
            'volume': 1000.0
        }]
        
        rows = db.insert_many('prices', test_data)
        print(f"[+] Inserted {rows} test rows")
        
        # Test select
        prices = db.get_prices(5)
        print(f"[+] Retrieved {len(prices)} price records")
    else:
        print("[!] Database not connected")
