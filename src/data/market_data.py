"""
Market data fetching and caching system
Handles stock data from yfinance and crypto data from CCXT with local database caching.
"""

import asyncio
import logging
import sqlite3
import pandas as pd
import yfinance as yf
import ccxt
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

        # Initialize crypto exchange (Binance for comprehensive coverage)
        self.crypto_exchange = ccxt.binance({
            'sandbox': False,  # Use live data
            'enableRateLimit': True,  # Respect rate limits
            'timeout': 30000,  # 30 second timeout
        })
        
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
        conn = sqlite3.connect(self.db_path, timeout=60.0)
        conn.execute('PRAGMA busy_timeout = 60000')
        conn.execute('PRAGMA journal_mode = WAL')
        conn.execute('PRAGMA synchronous = NORMAL')
        cursor = conn.cursor()
        
        try:
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
        finally:
            conn.close()
    
    async def get_stock_data(self, symbol: str, days: int = 90, force_refresh: bool = False, extend_history: bool = True) -> pd.DataFrame:
        """
        Get stock data for a symbol, using cache when possible
        
        Args:
            symbol: Stock symbol (e.g., 'GOOGL')
            days: Number of days of historical data
            force_refresh: Force API call even if cached data exists
        """
        self.logger.debug(f"Fetching data for {symbol} (days={days}, force_refresh={force_refresh})")
        
        # Check if we need to extend history or just get recent data
        if extend_history:
            # Get existing data to see what we already have
            existing_data = await self._get_all_cached_data(symbol)
            
            if not existing_data.empty:
                # Check if recent data is fresh enough
                latest_timestamp = existing_data.index.max()
                
                # Handle timezone-aware vs timezone-naive datetime comparison
                current_time = datetime.now()
                latest_dt = latest_timestamp.to_pydatetime()
                
                # Make both timezone-naive for comparison
                if latest_dt.tzinfo is not None:
                    latest_dt = latest_dt.replace(tzinfo=None)
                
                hours_since_update = (current_time - latest_dt).total_seconds() / 3600

                if hours_since_update < (self.cache_duration / 3600):
                    self.logger.debug(f"Using cached data for {symbol} (last update: {hours_since_update:.1f}h ago)")
                    return existing_data.tail(days) if len(existing_data) > days else existing_data

                # Fetch only new data since last update - smart incremental
                if hours_since_update < 24:
                    # Less than 1 day old - fetch just 2 days to get latest
                    days_to_fetch = 2
                elif hours_since_update < 168:  # Less than 1 week
                    # Fetch a few extra days to cover gaps
                    days_to_fetch = min(7, int(hours_since_update / 24) + 2)
                else:
                    # Older than a week - fetch enough to backfill
                    days_to_fetch = min(30, int(hours_since_update / 24) + 5)

                self.logger.info(f"Extending history for {symbol} - fetching {days_to_fetch} days (last update: {hours_since_update:.1f}h ago)")
                fresh_data = await self._fetch_from_api(symbol, days_to_fetch)
            else:
                # First time - get full history
                self.logger.info(f"First time fetching {symbol} - getting {days} days")
                fresh_data = await self._fetch_from_api(symbol, days)
        else:
            # Old behavior - check cache first
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
            
            # Return all accumulated data if extending history, or just fresh data
            if extend_history:
                all_data = await self._get_all_cached_data(symbol)
                return all_data.tail(days) if len(all_data) > days else all_data
            else:
                return fresh_data
        else:
            # Fallback to cached data if API fails
            self.logger.warning(f"API fetch failed for {symbol}, trying cached data")
            if extend_history:
                cached_data = await self._get_all_cached_data(symbol)
                return cached_data.tail(days) if len(cached_data) > days else cached_data
            else:
                cached_data = await self._get_cached_data(symbol, days, ignore_cache_time=True)
                return cached_data if cached_data is not None else pd.DataFrame()
    
    async def _get_cached_data(self, symbol: str, days: int, ignore_cache_time: bool = False) -> Optional[pd.DataFrame]:
        """Get cached data from database"""
        conn = sqlite3.connect(self.db_path, timeout=60.0)
        conn.execute('PRAGMA busy_timeout = 60000')
        conn.execute('PRAGMA journal_mode = WAL')
        conn.execute('PRAGMA synchronous = NORMAL')
        
        try:
            # Get asset_id
            cursor = conn.cursor()
            cursor.execute('SELECT asset_id FROM assets WHERE symbol = ?', (symbol,))
            result = cursor.fetchone()
            
            if not result:
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
            
            if df.empty:
                return None
            
            # Convert timestamp to datetime and set as index
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Convert to standard yfinance format
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            return df
            
        finally:
            conn.close()
    
    async def _get_all_cached_data(self, symbol: str) -> pd.DataFrame:
        """Get all cached data for a symbol (no time limits)"""
        conn = sqlite3.connect(self.db_path, timeout=60.0)
        conn.execute('PRAGMA busy_timeout = 60000')
        conn.execute('PRAGMA journal_mode = WAL')
        conn.execute('PRAGMA synchronous = NORMAL')
        
        try:
            cursor = conn.cursor()
            
            # Get asset_id
            cursor.execute('SELECT asset_id FROM assets WHERE symbol = ?', (symbol,))
            result = cursor.fetchone()
            
            if not result:
                return pd.DataFrame()
            
            asset_id = result[0]
            
            # Query all price data
            query = '''
            SELECT timestamp, open, high, low, close, volume
            FROM price_data 
            WHERE asset_id = ?
            ORDER BY timestamp ASC
            '''
            
            df = pd.read_sql_query(query, conn, params=(asset_id,))
            
            if df.empty:
                return pd.DataFrame()
            
            # Convert timestamp to datetime and set as index
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Convert to standard yfinance format
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting all cached data for {symbol}: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
    
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
            # Always use daily data for consistency
            interval = '1d'  # Daily intervals - consistent granularity
            actual_days = min(days, 365)  # Limit to 1 year of daily data
            start_date = end_date - timedelta(days=actual_days)
            
            data = await loop.run_in_executor(
                None,
                lambda: ticker.history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval=interval
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
        """Store data in database with retry logic"""
        if data.empty:
            return
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                # More aggressive connection settings
                conn = sqlite3.connect(self.db_path, timeout=60.0)
                conn.execute('PRAGMA busy_timeout = 60000')
                conn.execute('PRAGMA journal_mode = WAL')
                conn.execute('PRAGMA synchronous = NORMAL')
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
                        INSERT OR IGNORE INTO price_data
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
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    conn.rollback()
                    raise e
                finally:
                    conn.close()
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Database error storing {symbol} (attempt {attempt + 1}/{max_retries}): {e}")
                    await asyncio.sleep(0.5 * (2 ** attempt))  # Exponential backoff: 0.5s, 1s, 2s, 4s
                else:
                    self.logger.error(f"Failed to store data for {symbol} after {max_retries} attempts: {e}")
    
    async def get_latest_data(self, assets: List[Dict]) -> Dict[str, pd.DataFrame]:
        """Get latest data for multiple assets (stocks and crypto)"""
        self.logger.info(f"Fetching data for {len(assets)} assets")

        # Separate stocks and crypto
        stock_assets = [asset for asset in assets if asset['type'] == 'stock']
        crypto_assets = [asset for asset in assets if asset['type'] == 'crypto']

        results = {}

        # Process stocks in batches to respect API limits
        if stock_assets:
            self.logger.info(f"Processing {len(stock_assets)} stock assets")
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
                        self.logger.error(f"Error fetching stock {asset['symbol']}: {data}")
                        results[asset['symbol']] = pd.DataFrame()
                    else:
                        results[asset['symbol']] = data

                # Small delay between batches to be nice to the API
                if i + batch_size < len(stock_assets):
                    await asyncio.sleep(1)

        # Process crypto assets
        if crypto_assets:
            self.logger.info(f"Processing {len(crypto_assets)} crypto assets")
            batch_size = 5  # Smaller batches for crypto to respect rate limits
            for i in range(0, len(crypto_assets), batch_size):
                batch = crypto_assets[i:i + batch_size]

                # Process batch concurrently
                tasks = [
                    self.get_crypto_data(asset['symbol'], days=90)
                    for asset in batch
                ]

                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Store results
                for asset, data in zip(batch, batch_results):
                    if isinstance(data, Exception):
                        self.logger.error(f"Error fetching crypto {asset['symbol']}: {data}")
                        results[asset['symbol']] = pd.DataFrame()
                    else:
                        results[asset['symbol']] = data

                # Longer delay between crypto batches (stricter rate limits)
                if i + batch_size < len(crypto_assets):
                    await asyncio.sleep(2)

        successful_fetches = len([k for k, v in results.items() if not v.empty])
        self.logger.info(f"Retrieved data for {successful_fetches}/{len(assets)} assets")
        return results

    async def get_crypto_data(self, symbol: str, days: int = 90, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get crypto data for a symbol using CCXT, with database caching

        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'ETH')
            days: Number of days of historical data
            force_refresh: Force API call even if cached data exists
        """
        self.logger.debug(f"Fetching crypto data for {symbol} (days={days}, force_refresh={force_refresh})")

        try:
            # Check cache first unless force refresh
            if not force_refresh:
                cached_data = await self._get_cached_crypto_data(symbol, days)
                if not cached_data.empty:
                    self.logger.debug(f"Using cached crypto data for {symbol}")
                    return cached_data

            # Fetch from Binance API
            symbol_pair = f"{symbol}/USDT"  # Convert to Binance pair format

            # Calculate timeframe - CCXT uses different timeframe notation
            # Use 30m for consistency with stock data
            timeframe = '30m'

            # Calculate since timestamp
            since_timestamp = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

            self.logger.debug(f"Fetching {symbol_pair} from Binance API...")

            # Fetch OHLCV data
            ohlcv_data = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.crypto_exchange.fetch_ohlcv(
                    symbol_pair,
                    timeframe,
                    since=since_timestamp,
                    limit=1000  # Max limit for most exchanges
                )
            )

            if not ohlcv_data:
                self.logger.warning(f"No crypto data returned for {symbol}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Ensure numeric types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Remove any rows with NaN values
            df = df.dropna()

            if df.empty:
                self.logger.warning(f"No valid crypto data for {symbol} after processing")
                return pd.DataFrame()

            # Store in database
            await self._store_crypto_data(symbol, df)

            self.logger.info(f"Retrieved {len(df)} crypto data points for {symbol}")
            return df

        except Exception as e:
            self.logger.error(f"Error fetching crypto data for {symbol}: {e}")
            return pd.DataFrame()

    async def _get_cached_crypto_data(self, symbol: str, days: int) -> pd.DataFrame:
        """Get cached crypto data if recent enough"""
        try:
            # Check if asset exists in database
            asset_id = await self._get_or_create_asset_id(symbol, 'crypto')
            if not asset_id:
                return pd.DataFrame()

            # Get recent data
            cutoff_time = datetime.now() - timedelta(days=days)

            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query('''
                SELECT timestamp, open, high, low, close, volume
                FROM price_data
                WHERE asset_id = ? AND timestamp >= ?
                ORDER BY timestamp
            ''', conn, params=(asset_id, cutoff_time.isoformat()))
            conn.close()

            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])

                # Check if data is recent enough (within last 2 hours)
                latest_time = df['timestamp'].max()
                if latest_time > datetime.now() - timedelta(hours=2):
                    self.logger.debug(f"Using cached crypto data for {symbol} (latest: {latest_time})")
                    return df

            return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Error getting cached crypto data for {symbol}: {e}")
            return pd.DataFrame()

    async def _store_crypto_data(self, symbol: str, df: pd.DataFrame):
        """Store crypto data in database"""
        if df.empty:
            return

        try:
            # Get or create asset ID
            asset_id = await self._get_or_create_asset_id(symbol, 'crypto')
            if not asset_id:
                self.logger.error(f"Could not get asset ID for crypto {symbol}")
                return

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Insert data
            for _, row in df.iterrows():
                cursor.execute('''
                    INSERT OR REPLACE INTO price_data
                    (asset_id, timestamp, open, high, low, close, volume, interval)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    asset_id,
                    row['timestamp'].isoformat(),
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    int(row['volume']),
                    '30min'
                ))

            conn.commit()
            conn.close()

            self.logger.debug(f"Stored {len(df)} crypto data points for {symbol}")

        except Exception as e:
            self.logger.error(f"Error storing crypto data for {symbol}: {e}")

    async def _get_or_create_asset_id(self, symbol: str, asset_type: str) -> Optional[int]:
        """Get existing asset ID or create new asset and return its ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Try to get existing asset
            cursor.execute('SELECT asset_id FROM assets WHERE symbol = ? AND asset_type = ?', (symbol, asset_type))
            result = cursor.fetchone()

            if result:
                conn.close()
                return result[0]

            # Create new asset
            cursor.execute('''
                INSERT INTO assets (symbol, asset_type, name, last_updated)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ''', (symbol, asset_type, None))

            asset_id = cursor.lastrowid
            conn.commit()
            conn.close()

            self.logger.debug(f"Created new asset: {symbol} ({asset_type}) with ID {asset_id}")
            return asset_id

        except Exception as e:
            self.logger.error(f"Error getting/creating asset ID for {symbol}: {e}")
            return None

    async def update_asset_names(self):
        """Update asset names in database using the asset_names utility"""
        from ..utils.asset_names import get_asset_name

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get all assets that don't have names or have empty names
            cursor.execute('SELECT asset_id, symbol, asset_type FROM assets WHERE name IS NULL OR name = ""')
            assets_to_update = cursor.fetchall()

            self.logger.info(f"Updating names for {len(assets_to_update)} assets...")

            for asset_id, symbol, asset_type in assets_to_update:
                # Get the full name
                full_name = get_asset_name(symbol, asset_type)

                # Update the database
                cursor.execute(
                    'UPDATE assets SET name = ?, last_updated = CURRENT_TIMESTAMP WHERE asset_id = ?',
                    (full_name, asset_id)
                )

            conn.commit()
            conn.close()

            self.logger.info(f"Successfully updated names for {len(assets_to_update)} assets")

        except Exception as e:
            self.logger.error(f"Error updating asset names: {e}")

    async def cleanup(self):
        """Clean up resources"""
        # Could implement data retention cleanup here
        self.logger.info("Market data manager cleanup complete")