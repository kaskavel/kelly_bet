"""
Market data fetching and caching system
Handles stock data from yfinance with local database caching.
"""

import asyncio
import logging
import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path


class MarketDataManager:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.db_path = Path(config['database']['sqlite']['path'])
        
        # Cache settings
        self.cache_duration = config['data_sources']['stocks']['cache_duration']
        
    async def initialize(self):
        """Initialize database and create tables"""
        self.logger.info("Initializing market data manager...")
        
        # Create data directory
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create database tables
        await self._create_tables()
        self.logger.info("Market data manager initialized")
    
    async def _create_tables(self):
        """Create database tables for market data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Assets table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS assets (
            asset_id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT UNIQUE NOT NULL,
            asset_type TEXT NOT NULL,
            name TEXT,
            sector TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Price data table (OHLCV)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS price_data (
            price_id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id INTEGER,
            timestamp TIMESTAMP NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume INTEGER,
            interval TEXT DEFAULT '30min',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (asset_id) REFERENCES assets (asset_id),
            UNIQUE(asset_id, timestamp, interval)
        )
        ''')
        
        # Create indexes for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_price_data_timestamp ON price_data(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_price_data_symbol ON price_data(asset_id, timestamp)')
        
        conn.commit()
        conn.close()
    
    async def get_stock_data(self, symbol: str, days: int = 90, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get stock data for a symbol, using cache when possible
        
        Args:
            symbol: Stock symbol (e.g., 'GOOGL')
            days: Number of days of historical data
            force_refresh: Force API call even if cached data exists
        """
        self.logger.debug(f"Fetching data for {symbol} (days={days}, force_refresh={force_refresh})")
        
        # Check cache first unless force refresh
        if not force_refresh:
            cached_data = await self._get_cached_data(symbol, days)
            if cached_data is not None and not cached_data.empty:
                self.logger.debug(f"Using cached data for {symbol}")
                return cached_data
        
        # Fetch from API
        self.logger.info(f"Fetching fresh data from API for {symbol}")
        fresh_data = await self._fetch_from_api(symbol, days)
        
        if fresh_data is not None and not fresh_data.empty:
            # Store in database
            await self._store_data(symbol, fresh_data)
            return fresh_data
        else:
            # Fallback to cached data if API fails
            self.logger.warning(f"API fetch failed for {symbol}, trying cached data")
            cached_data = await self._get_cached_data(symbol, days, ignore_cache_time=True)
            return cached_data if cached_data is not None else pd.DataFrame()
    
    async def _get_cached_data(self, symbol: str, days: int, ignore_cache_time: bool = False) -> Optional[pd.DataFrame]:
        """Get cached data from database"""
        conn = sqlite3.connect(self.db_path)
        
        # Get asset_id
        cursor = conn.cursor()
        cursor.execute('SELECT asset_id FROM assets WHERE symbol = ?', (symbol,))
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            return None
        
        asset_id = result[0]
        
        # Calculate date threshold
        if not ignore_cache_time:
            cache_threshold = datetime.now() - timedelta(seconds=self.cache_duration)
        else:
            cache_threshold = datetime.now() - timedelta(days=365)  # Very old threshold for fallback
        
        start_date = datetime.now() - timedelta(days=days)
        
        # Query price data
        query = '''
        SELECT timestamp, open, high, low, close, volume
        FROM price_data 
        WHERE asset_id = ? 
          AND timestamp >= ? 
          AND created_at >= ?
        ORDER BY timestamp ASC
        '''
        
        df = pd.read_sql_query(
            query, 
            conn, 
            params=(asset_id, start_date.isoformat(), cache_threshold.isoformat())
        )
        conn.close()
        
        if df.empty:
            return None
        
        # Convert timestamp to datetime and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Convert to standard yfinance format
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        return df
    
    async def _fetch_from_api(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """Fetch data from yfinance API"""
        try:
            # Use asyncio to prevent blocking
            loop = asyncio.get_event_loop()
            
            # Calculate period
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Fetch data in thread pool to avoid blocking
            ticker = yf.Ticker(symbol)
            data = await loop.run_in_executor(
                None,
                lambda: ticker.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval='30m'  # 30-minute intervals
                )
            )
            
            if data.empty:
                self.logger.warning(f"No data returned from yfinance for {symbol}")
                return None
            
            # Clean data
            data = data.dropna()
            
            self.logger.info(f"Fetched {len(data)} records for {symbol} from API")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    async def _store_data(self, symbol: str, data: pd.DataFrame):
        """Store data in database"""
        if data.empty:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get or create asset
            cursor.execute('SELECT asset_id FROM assets WHERE symbol = ?', (symbol,))
            result = cursor.fetchone()
            
            if result:
                asset_id = result[0]
                # Update last_updated
                cursor.execute(
                    'UPDATE assets SET last_updated = CURRENT_TIMESTAMP WHERE asset_id = ?',
                    (asset_id,)
                )
            else:
                # Create new asset
                cursor.execute(
                    'INSERT INTO assets (symbol, asset_type) VALUES (?, ?)',
                    (symbol, 'stock')
                )
                asset_id = cursor.lastrowid
            
            # Store price data
            for timestamp, row in data.iterrows():
                cursor.execute('''
                INSERT OR REPLACE INTO price_data 
                (asset_id, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    asset_id,
                    timestamp.isoformat(),
                    float(row['Open']),
                    float(row['High']),
                    float(row['Low']),
                    float(row['Close']),
                    int(row['Volume']) if pd.notna(row['Volume']) else 0
                ))
            
            conn.commit()
            self.logger.debug(f"Stored {len(data)} records for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error storing data for {symbol}: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    async def get_latest_data(self, assets: List[Dict]) -> Dict[str, pd.DataFrame]:
        """Get latest data for multiple assets"""
        self.logger.info(f"Fetching data for {len(assets)} assets")
        
        # Filter for stocks only for now
        stock_assets = [asset for asset in assets if asset['type'] == 'stock']
        
        results = {}
        
        # Process stocks in batches to respect API limits
        batch_size = 10
        for i in range(0, len(stock_assets), batch_size):
            batch = stock_assets[i:i + batch_size]
            
            # Process batch concurrently
            tasks = [
                self.get_stock_data(asset['symbol'], days=90)
                for asset in batch
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Store results
            for asset, data in zip(batch, batch_results):
                if isinstance(data, Exception):
                    self.logger.error(f"Error fetching {asset['symbol']}: {data}")
                    results[asset['symbol']] = pd.DataFrame()
                else:
                    results[asset['symbol']] = data
            
            # Small delay between batches to be nice to the API
            if i + batch_size < len(stock_assets):
                await asyncio.sleep(1)
        
        self.logger.info(f"Retrieved data for {len([k for k, v in results.items() if not v.empty])} assets")
        return results
    
    async def cleanup(self):
        """Clean up resources"""
        # Could implement data retention cleanup here
        self.logger.info("Market data manager cleanup complete")